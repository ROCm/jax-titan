"""The Flux2 KL autoencoder, which Ideogram 4 uses, in plain JAX.

The architecture follows diffusers' `AutoencoderKLFlux2`: a convolutional
encoder and decoder of residual blocks with one attention block in the
middle, 8x smaller latents of 32 channels. Images are [batch, height, width,
3] in [-1, 1] and latents [batch, height / 8, width / 8, 32]. A transformer's
image tokens are 2x2 patches of the latents, normalized with the statistics
of the model that reads them (`to_tokens`, `from_tokens`).

The model is frozen and small, so it runs in fp32 on one device, outside any
mesh.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jax_titan.models import pretrained

_LAYOUT = ("NHWC", "HWIO", "NHWC")


@dataclasses.dataclass(frozen=True)
class Flux2VaeConfig:
  """The fields of the model's `vae/config.json` that set its shape."""

  in_channels: int = 3
  latent_channels: int = 32
  block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
  layers_per_block: int = 2
  norm_num_groups: int = 32
  patch_size: tuple[int, int] = (2, 2)

  @property
  def token_channels(self) -> int:
    return self.latent_channels * self.patch_size[0] * self.patch_size[1]


def params_from_tensors(tensors: dict, cfg: Flux2VaeConfig) -> dict:
  """fp32 parameters from tensors named as diffusers' `AutoencoderKLFlux2` names them."""

  def array(name):
    return np.asarray(tensors[name]).astype(np.float32)

  def conv(name):
    return {"w": array(name + ".weight").transpose(2, 3, 1, 0), "b": array(name + ".bias")}

  def norm(name):
    return {"scale": array(name + ".weight"), "bias": array(name + ".bias")}

  def linear(name):
    return {"w": array(name + ".weight").T, "b": array(name + ".bias")}

  def resnet(name):
    p = {"norm1": norm(name + ".norm1"), "conv1": conv(name + ".conv1"),
         "norm2": norm(name + ".norm2"), "conv2": conv(name + ".conv2")}
    if name + ".conv_shortcut.weight" in tensors:
      p["shortcut"] = conv(name + ".conv_shortcut")
    return p

  def mid(name):
    attention = name + ".attentions.0"
    return {
        "resnets": [resnet(f"{name}.resnets.{i}") for i in range(2)],
        "attention": {"norm": norm(attention + ".group_norm"), "q": linear(attention + ".to_q"),
                      "k": linear(attention + ".to_k"), "v": linear(attention + ".to_v"),
                      "out": linear(attention + ".to_out.0")},
    }

  levels = len(cfg.block_out_channels)
  down, up = [], []
  for i in range(levels):
    block = {"resnets": [resnet(f"encoder.down_blocks.{i}.resnets.{j}")
                         for j in range(cfg.layers_per_block)]}
    if i < levels - 1:
      block["downsample"] = conv(f"encoder.down_blocks.{i}.downsamplers.0.conv")
    down.append(block)
    block = {"resnets": [resnet(f"decoder.up_blocks.{i}.resnets.{j}")
                         for j in range(cfg.layers_per_block + 1)]}
    if i < levels - 1:
      block["upsample"] = conv(f"decoder.up_blocks.{i}.upsamplers.0.conv")
    up.append(block)
  return {
      "encoder": {"conv_in": conv("encoder.conv_in"), "down": down, "mid": mid("encoder.mid_block"),
                  "norm_out": norm("encoder.conv_norm_out"), "conv_out": conv("encoder.conv_out")},
      "quant_conv": conv("quant_conv"),
      "post_quant_conv": conv("post_quant_conv"),
      "decoder": {"conv_in": conv("decoder.conv_in"), "mid": mid("decoder.mid_block"), "up": up,
                  "norm_out": norm("decoder.conv_norm_out"), "conv_out": conv("decoder.conv_out")},
  }


def load(path) -> tuple[Flux2VaeConfig, dict]:
  """The config and the fp32 parameters, on the default device, of the published
  weights at `path`, a safetensors file or a directory of them."""
  cfg = pretrained.config(Flux2VaeConfig, path)
  return cfg, pretrained.to_device(params_from_tensors(pretrained.read(path), cfg))


def _conv(x, p, stride=1, padding="SAME"):
  y = jax.lax.conv_general_dilated(x, p["w"], (stride, stride), padding, dimension_numbers=_LAYOUT)
  return y + p["b"]


def _group_norm(x, p, groups):
  b, h, w, c = x.shape
  g = x.reshape(b, h * w, groups, c // groups)
  mean = jnp.mean(g, axis=(1, 3), keepdims=True)
  var = jnp.mean(jnp.square(g - mean), axis=(1, 3), keepdims=True)
  g = (g - mean) * jax.lax.rsqrt(var + 1e-6)
  return g.reshape(x.shape) * p["scale"] + p["bias"]


def _resnet(x, p, groups):
  h = _conv(jax.nn.silu(_group_norm(x, p["norm1"], groups)), p["conv1"])
  h = _conv(jax.nn.silu(_group_norm(h, p["norm2"], groups)), p["conv2"])
  if "shortcut" in p:
    x = _conv(x, p["shortcut"])
  return x + h


def _attention(x, p, groups):
  """One head of attention over the positions of the feature map."""
  b, h, w, c = x.shape
  y = _group_norm(x, p["norm"], groups).reshape(b, h * w, 1, c)
  q, k, v = (y @ p[name]["w"] + p[name]["b"] for name in ("q", "k", "v"))
  y = jax.nn.dot_product_attention(q, k, v)
  return x + (y @ p["out"]["w"] + p["out"]["b"]).reshape(x.shape)


def _mid(x, p, groups):
  x = _resnet(x, p["resnets"][0], groups)
  x = _attention(x, p["attention"], groups)
  return _resnet(x, p["resnets"][1], groups)


def encode(cfg: Flux2VaeConfig, params: dict, images: jax.Array) -> jax.Array:
  """The mean latents [batch, height / 8, width / 8, latent_channels] of images in [-1, 1]."""
  p, groups = params["encoder"], cfg.norm_num_groups
  x = _conv(images.astype(jnp.float32), p["conv_in"])
  for block in p["down"]:
    for resnet in block["resnets"]:
      x = _resnet(x, resnet, groups)
    if "downsample" in block:
      x = _conv(jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0))), block["downsample"], 2, "VALID")
  x = _mid(x, p["mid"], groups)
  x = _conv(jax.nn.silu(_group_norm(x, p["norm_out"], groups)), p["conv_out"])
  return _conv(x, params["quant_conv"])[..., :cfg.latent_channels]


def decode(cfg: Flux2VaeConfig, params: dict, latents: jax.Array) -> jax.Array:
  """Images [batch, height, width, 3], nominally in [-1, 1], from latents."""
  p, groups = params["decoder"], cfg.norm_num_groups
  x = _conv(latents.astype(jnp.float32), params["post_quant_conv"])
  x = _mid(_conv(x, p["conv_in"]), p["mid"], groups)
  for block in p["up"]:
    for resnet in block["resnets"]:
      x = _resnet(x, resnet, groups)
    if "upsample" in block:
      x = _conv(jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2), block["upsample"])
  return _conv(jax.nn.silu(_group_norm(x, p["norm_out"], groups)), p["conv_out"])


def to_tokens(cfg: Flux2VaeConfig, latents: jax.Array, mean, std) -> jax.Array:
  """A transformer's image tokens [batch, tokens, token_channels] of latents:
  2x2 patches in row order, normalized by `mean` and `std` per token channel."""
  b, h, w, c = latents.shape
  ph, pw = cfg.patch_size
  x = latents.reshape(b, h // ph, ph, w // pw, pw, c).transpose(0, 1, 3, 2, 4, 5)
  x = x.reshape(b, (h // ph) * (w // pw), ph * pw * c)
  return (x - mean) / std


def from_tokens(cfg: Flux2VaeConfig, tokens: jax.Array, rows: int, mean, std) -> jax.Array:
  """The latents of image tokens laid out in `rows` rows, the inverse of `to_tokens`."""
  b, n, _ = tokens.shape
  ph, pw = cfg.patch_size
  columns, c = n // rows, cfg.latent_channels
  x = tokens.astype(jnp.float32) * std + mean
  x = x.reshape(b, rows, columns, ph, pw, c).transpose(0, 1, 3, 2, 4, 5)
  return x.reshape(b, rows * ph, columns * pw, c)
