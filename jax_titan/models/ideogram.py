"""Ideogram 4's flow-matching transformer with parameters grouped by pipeline stage.

The architecture follows diffusers' `Ideogram4Transformer2DModel`, and the
training step (noising, timesteps, loss) its example
`train_dreambooth_lora_ideogram4.py`. One sequence, the text tokens followed
by the image tokens, runs through identical blocks: RMSNorm before and after
attention and the SwiGLU MLP, normalized queries and keys, rotary embeddings
over three position axes, and a scale and a gate on each half from the
timestep. Text features come from the text encoder's hidden states
(`jax_titan.models.qwen3`) and image tokens from 2x2 patches of the
autoencoder's latents (`jax_titan.models.flux2_vae`).

A stage holds a contiguous run of blocks; stage 0 also holds the input
projections and the timestep embedding, and the last stage the output layer.
"""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_titan import ops
from jax_titan.models import pretrained

# Values of `indicator` and `segment_ids`, and the offset of image positions.
PADDING_SEGMENT = -1
IMAGE_INDICATOR = 2
TEXT_INDICATOR = 3
IMAGE_POSITION_OFFSET = 65536

# The mean and deviation of each channel of an image token that Ideogram 4 is
# trained with: `LATENT_SHIFT` and `LATENT_SCALE` of ideogram-oss/ideogram4
# (`latent_norm.py`), which its pipeline applies. The batch-norm statistics
# stored with the autoencoder are other numbers, and that pipeline does not
# read them.
TOKEN_MEAN = np.array([
    0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549, 0.05021099,
    -0.15083604, -0.15360136, -0.20131799, 0.01922352, 0.0622626, 0.10140969, -0.06739428,
    0.3758261, -0.233712, 0.35164491, -0.02590912, -0.0271935, -0.10833897, -0.1476848,
    -0.01130957, -0.2298372, 0.23526423, -0.10893522, 0.11957631, 0.04047799, 0.3134589,
    -0.17225064, -0.18646109, -0.34691978, -0.03571246, 0.02583857, 0.10190072, 0.28402294,
    0.26952152, -0.21634675, -0.17938656, 0.04358909, -0.15007621, -0.1548502, -0.18971131,
    0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698, -0.24269937, 0.35705471,
    -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137, -0.21863696, 0.23228983,
    -0.11739769, 0.11693044, 0.02563311, 0.31356594, -0.17420591, -0.19006285, -0.34905377,
    -0.04025005, 0.01924137, 0.07652984, 0.2995608, 0.2628057, -0.22011674, -0.12715361,
    0.04879879, -0.14075719, -0.15935895, -0.2123584, 0.01974813, 0.05523547, 0.10011992,
    -0.06428964, 0.37781868, -0.21491644, 0.34254215, -0.03153528, -0.0310082, -0.10761415,
    -0.14730405, -0.02475182, -0.2285588, 0.2515081, -0.10445128, 0.12446, 0.07062869,
    0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.0129177, 0.02578168, 0.07993659,
    0.28642181, 0.26038408, -0.22459419, -0.14820155, 0.04059549, -0.14043529, -0.16111187,
    -0.2020305, 0.02602069, 0.04852717, 0.10432153, -0.06309942, 0.38402443, -0.22397003,
    0.34814481, -0.03774432, -0.03381438, -0.11245691, -0.14128767, -0.02853208, -0.21752016,
    0.24872463, -0.11399775, 0.1222687, 0.05620835, 0.309178, -0.18065738, -0.19401479,
    -0.34495114, -0.01760592,
], np.float32)
TOKEN_STD = np.array([
    1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584, 1.56853198,
    1.62314944, 1.89106626, 1.58086668, 1.60822129, 1.60962993, 1.63322129, 1.56074359,
    1.73419528, 1.7919265, 1.64040632, 1.66802808, 1.60390303, 1.75480492, 1.63187587,
    1.64334594, 1.61722884, 1.60146046, 1.63459219, 1.55291476, 1.68771497, 1.68415657,
    1.78966054, 1.66631641, 1.65626686, 1.65976433, 1.63487607, 1.69513249, 1.72933756,
    1.91310663, 1.67035057, 1.72286863, 1.56719251, 1.61934825, 1.88628859, 1.56911539,
    1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606, 1.63732541,
    1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667, 1.61659342, 1.59722044,
    1.64103121, 1.5408531, 1.68610394, 1.67772755, 1.78998563, 1.66621713, 1.65458955,
    1.66041308, 1.64710857, 1.68163503, 1.74000294, 1.92784786, 1.67411194, 1.67395548,
    1.57406532, 1.62199356, 1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305,
    1.55644029, 1.73124302, 1.80666627, 1.6463621, 1.65932006, 1.60816188, 1.75682671,
    1.64695873, 1.63121722, 1.61380832, 1.60478651, 1.63396035, 1.53505068, 1.65534289,
    1.67132281, 1.80317197, 1.6767314, 1.65700938, 1.68426259, 1.65339716, 1.67540638,
    1.73298504, 1.94067348, 1.67893609, 1.70635117, 1.5730906, 1.61928553, 1.87148809,
    1.56244866, 1.56697152, 1.61584394, 1.62759496, 1.55480378, 1.73484107, 1.79055143,
    1.64688773, 1.66121492, 1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592,
    1.60792883, 1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476,
    1.65655173, 1.68533454,
], np.float32)


@dataclasses.dataclass(frozen=True)
class IdeogramConfig:
  """The fields and defaults of the model's `transformer/config.json`."""

  in_channels: int = 128
  num_layers: int = 34
  attention_head_dim: int = 256
  num_attention_heads: int = 18
  intermediate_size: int = 12288
  adaln_dim: int = 512
  llm_features_dim: int = 53248
  rope_theta: int = 5_000_000
  mrope_section: tuple[int, int, int] = (24, 20, 20)
  norm_eps: float = 1e-5

  @property
  def emb_dim(self) -> int:
    return self.attention_head_dim * self.num_attention_heads

  def flops_per_sample(self, image_tokens: int, text_tokens: int) -> float:
    """Training FLOPs per sample: 6 per weight and token, plus attention."""
    d, tokens = self.emb_dim, image_tokens + text_tokens
    per_token = 4 * d * d + 3 * d * self.intermediate_size
    weights = per_token * tokens * self.num_layers + d * (
        self.llm_features_dim * text_tokens + 2 * self.in_channels * image_tokens)
    attention = 2 * tokens * tokens * d * self.num_layers
    return 6.0 * (weights + attention)


_LAYER_AXES = {
    "attn_norm": ("embed",),
    "wq": ("embed", "heads"),
    "wk": ("embed", "heads"),
    "wv": ("embed", "heads"),
    "q_norm": ("head_dim",),
    "k_norm": ("head_dim",),
    "wo": ("heads", "embed"),
    "attn_out_norm": ("embed",),
    "mlp_norm": ("embed",),
    "w_gate": ("embed", "mlp"),
    "w_up": ("embed", "mlp"),
    "w_down": ("mlp", "embed"),
    "mlp_out_norm": ("embed",),
    "w_mod": ("cond", "modulation"),
    "b_mod": ("modulation",),
}
_FIRST_AXES = {
    "w_in": ("channels", "embed"),
    "b_in": ("embed",),
    "text_norm": ("text",),
    "w_text": ("text", "embed"),
    "b_text": ("embed",),
    "w_time_in": ("embed", "embed"),
    "b_time_in": ("embed",),
    "w_time_out": ("embed", "embed"),
    "b_time_out": ("embed",),
    "w_cond": ("embed", "cond"),
    "b_cond": ("cond",),
    "image_embed": ("indicator", "embed"),
}
_LAST_AXES = {
    "w_final_mod": ("cond", "embed"),
    "b_final_mod": ("embed",),
    "w_out": ("embed", "channels"),
    "b_out": ("channels",),
}


def logical_axes(path: tuple, leaf) -> tuple | None:
  """Logical axes of a parameter, or of an optimizer-state leaf mirroring one.

  Returns None for leaves that are not parameter-shaped, such as step counts.
  """
  keys = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
  if not keys:
    return None
  if "layers" in keys[:-1]:
    axes = ("layers",) + _LAYER_AXES.get(keys[-1], ())
  else:
    axes = (_FIRST_AXES | _LAST_AXES).get(keys[-1], ())
  return axes if len(axes) == getattr(leaf, "ndim", -1) else None


def decays(path: tuple, leaf) -> bool:
  """Whether a parameter takes weight decay: every one but the norms' weights."""
  del leaf
  return path[-1].key not in ("attn_norm", "q_norm", "k_norm", "attn_out_norm", "mlp_norm",
                              "mlp_out_norm", "text_norm")


def init_params(cfg: IdeogramConfig, key: jax.Array) -> dict:
  """Returns fp32 parameters of the whole model, the blocks stacked along a layer axis.

  Weights are normal with variance 1 / fan-in, biases zero and norm weights one.
  """
  n, e, f = cfg.num_layers, cfg.emb_dim, cfg.intermediate_size
  d, a, c, t = cfg.attention_head_dim, cfg.adaln_dim, cfg.in_channels, cfg.llm_features_dim
  keys = iter(jax.random.split(key, 16))

  def weight(*shape):
    return shape[-2] ** -0.5 * jax.random.normal(next(keys), shape, jnp.float32)

  def ones(*shape):
    return jnp.ones(shape, jnp.float32)

  def zeros(*shape):
    return jnp.zeros(shape, jnp.float32)

  return {
      "layers": {
          "attn_norm": ones(n, e),
          "wq": weight(n, e, e),
          "wk": weight(n, e, e),
          "wv": weight(n, e, e),
          "q_norm": ones(n, d),
          "k_norm": ones(n, d),
          "wo": weight(n, e, e),
          "attn_out_norm": ones(n, e),
          "mlp_norm": ones(n, e),
          "w_gate": weight(n, e, f),
          "w_up": weight(n, e, f),
          "w_down": weight(n, f, e),
          "mlp_out_norm": ones(n, e),
          "w_mod": weight(n, a, 4 * e),
          "b_mod": zeros(n, 4 * e),
      },
      "w_in": weight(c, e),
      "b_in": zeros(e),
      "text_norm": ones(t),
      "w_text": weight(t, e),
      "b_text": zeros(e),
      "w_time_in": weight(e, e),
      "b_time_in": zeros(e),
      "w_time_out": weight(e, e),
      "b_time_out": zeros(e),
      "w_cond": weight(e, a),
      "b_cond": zeros(a),
      "image_embed": e ** -0.5 * jax.random.normal(next(keys), (2, e), jnp.float32),
      "w_final_mod": weight(a, e),
      "b_final_mod": zeros(e),
      "w_out": weight(e, c),
      "b_out": zeros(c),
  }


def pretrained_config(path) -> IdeogramConfig:
  """The config of published weights, from the `config.json` beside them."""
  return pretrained.config(IdeogramConfig, path)


def caption(text: str) -> str:
  """`text` as Ideogram 4 reads a caption. The model is trained on JSON objects
  written without spaces, so one that is written otherwise is written so; any
  other text is returned as it is."""
  text = text.lstrip("\ufeff").strip()
  try:
    value = json.loads(text)
  except ValueError:
    return text
  if not isinstance(value, dict):
    return text
  return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def pretrained_params(path, cfg: IdeogramConfig) -> dict:
  """`init_params`-shaped fp32 parameters, on the host, from published weights.

  `path` is a safetensors file or a directory of them in the layout of
  Ideogram's own model, which the published checkpoints keep: one `qkv`
  weight per block, and the MLP's gate, up and down weights as `w1`, `w3`
  and `w2`.
  """
  tensors = pretrained.read(path)
  n, e = cfg.num_layers, cfg.emb_dim

  def kernel(name):
    return pretrained.weight(tensors, name + ".weight").T

  def vector(name):
    return np.asarray(tensors[name]).astype(np.float32)

  def stacked(fn, name):
    return np.stack([fn(f"layers.{i}.{name}") for i in range(n)])

  qkv = stacked(kernel, "attention.qkv")
  return {
      "layers": {
          "attn_norm": stacked(vector, "attention_norm1.weight"),
          "wq": qkv[..., :e],
          "wk": qkv[..., e:2 * e],
          "wv": qkv[..., 2 * e:],
          "q_norm": stacked(vector, "attention.norm_q.weight"),
          "k_norm": stacked(vector, "attention.norm_k.weight"),
          "wo": stacked(kernel, "attention.o"),
          "attn_out_norm": stacked(vector, "attention_norm2.weight"),
          "mlp_norm": stacked(vector, "ffn_norm1.weight"),
          "w_gate": stacked(kernel, "feed_forward.w1"),
          "w_up": stacked(kernel, "feed_forward.w3"),
          "w_down": stacked(kernel, "feed_forward.w2"),
          "mlp_out_norm": stacked(vector, "ffn_norm2.weight"),
          "w_mod": stacked(kernel, "adaln_modulation"),
          "b_mod": stacked(vector, "adaln_modulation.bias"),
      },
      "w_in": kernel("input_proj"),
      "b_in": vector("input_proj.bias"),
      "text_norm": vector("llm_cond_norm.weight"),
      "w_text": kernel("llm_cond_proj"),
      "b_text": vector("llm_cond_proj.bias"),
      "w_time_in": kernel("t_embedding.mlp_in"),
      "b_time_in": vector("t_embedding.mlp_in.bias"),
      "w_time_out": kernel("t_embedding.mlp_out"),
      "b_time_out": vector("t_embedding.mlp_out.bias"),
      "w_cond": kernel("adaln_proj"),
      "b_cond": vector("adaln_proj.bias"),
      "image_embed": vector("embed_image_indicator.weight"),
      "w_final_mod": kernel("final_layer.adaln_modulation"),
      "b_final_mod": vector("final_layer.adaln_modulation.bias"),
      "w_out": kernel("final_layer.linear"),
      "b_out": vector("final_layer.linear.bias"),
  }


def noised(latents, noise, t):
  """The model's input and velocity target for latents [..., tokens, channels]
  noised at timesteps `t` [...], where 0 is noise and 1 is data."""
  t = t[..., None, None]
  return (1 - t) * noise + t * latents, latents - noise


def flow_matching_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  """Mean squared error to the velocity, latents minus noise, over the image tokens."""
  pred = pred[:, -target.shape[1]:]
  return jnp.mean((target.astype(jnp.float32) - pred.astype(jnp.float32)) ** 2)


def timesteps(z: np.ndarray, resolution: int, *, mean: float = 0.0, std: float = 1.5) -> np.ndarray:
  """Training timesteps for standard normal `z`, for square images of `resolution` pixels.

  0 is noise and 1 is data. The noise level, one minus the timestep, is
  logit-normal with a mean that grows with the image size.
  """
  mu = mean + 0.5 * math.log(resolution * resolution / 512**2)
  return (1.0 / (1.0 + np.exp(mu + std * z))).astype(np.float32)


def sequence_ids(image_side: int, text_tokens: int) -> dict[str, np.ndarray]:
  """`position_ids` [tokens, 3], `segment_ids` and `indicator` [tokens] of one
  sample whose text fills all `text_tokens` positions."""
  rows, columns = np.divmod(np.arange(image_side * image_side), image_side)
  image = np.stack([np.zeros_like(rows), rows, columns], axis=1) + IMAGE_POSITION_OFFSET
  text = np.repeat(np.arange(text_tokens)[:, None], 3, axis=1)
  tokens = text_tokens + image_side * image_side
  return {
      "position_ids": np.concatenate([text, image]).astype(np.int32),
      "segment_ids": np.ones(tokens, np.int32),
      "indicator": np.concatenate([
          np.full(text_tokens, TEXT_INDICATOR), np.full(tokens - text_tokens, IMAGE_INDICATOR),
      ]).astype(np.int32),
  }


def padded_sequence_ids(image_side: int, valid: np.ndarray) -> dict[str, np.ndarray]:
  """`sequence_ids` of each sample of `valid` [..., text_tokens], the mask of
  its real text tokens, which follow the padding."""
  lead, text = valid.shape[:-1], valid.shape[-1]
  ids = {k: np.broadcast_to(v, lead + v.shape).copy()
         for k, v in sequence_ids(image_side, text).items()}
  ids["position_ids"][..., :text, :] = np.maximum(np.cumsum(valid, axis=-1) - 1, 0)[..., None]
  ids["segment_ids"][..., :text] = np.where(valid, 1, PADDING_SEGMENT)
  ids["indicator"][..., :text] = np.where(valid, TEXT_INDICATOR, 0)
  return ids


def _rope_frequencies(cfg: IdeogramConfig) -> np.ndarray:
  """The rotary frequencies, [head_dim // 2] in fp32.

  Computed on the host, every step rounded to fp32 as diffusers rounds it.
  A compiled program folds the same expression to other last bits, and at the
  image positions one unit in the last place of a frequency moves an angle by
  up to 0.004.
  """
  d = cfg.attention_head_dim
  exponent = np.arange(0, d, 2, dtype=np.float32) / np.float32(d)
  power = np.power(np.float64(cfg.rope_theta), exponent.astype(np.float64)).astype(np.float32)
  return np.float32(1.0) / power


def _rope_tables(cfg: IdeogramConfig, position_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
  """cos and sin of the rotary angles, [batch, tokens, head_dim // 2] in fp32.

  The frequencies take turns over the three position axes, as far as each
  axis's share of `mrope_section` reaches, and use the first axis after that.
  """
  d = cfg.attention_head_dim
  # fp32, not the compute dtype: bf16 cannot tell neighbouring image positions apart.
  angles = position_ids[..., None].astype(jnp.float32) * _rope_frequencies(cfg)
  i = np.arange(d // 2)
  second = (i % 3 == 1) & (i < 3 * cfg.mrope_section[1])
  third = (i % 3 == 2) & (i < 3 * cfg.mrope_section[2])
  angles = jnp.where(
      second, angles[..., 1, :], jnp.where(third, angles[..., 2, :], angles[..., 0, :]))
  return jnp.cos(angles), jnp.sin(angles)


def _time_embedding(p: dict, timestep: jax.Array, dtype) -> jax.Array:
  half = p["w_time_in"].shape[0] // 2
  freqs = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -(math.log(1e4) / (half - 1)))
  angles = (1e4 * timestep.astype(jnp.float32))[..., None] * freqs
  x = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1).astype(dtype)
  x = jax.nn.silu(ops.linear(x, p["w_time_in"]) + p["b_time_in"])
  return ops.linear(x, p["w_time_out"]) + p["b_time_out"]


def _embed(cfg: IdeogramConfig, params: dict, x: dict, dtype):
  """The hidden states before the first block, and what every block takes with them."""
  p = {k: params[k].astype(dtype) for k in _FIRST_AXES}
  indicator = x["indicator"]
  text_tokens = x["txt"].shape[1]
  image = indicator == IMAGE_INDICATOR
  image_mask = image[..., None].astype(dtype)
  text_mask = (indicator[:, :text_tokens] == TEXT_INDICATOR)[..., None].astype(dtype)

  img = jnp.pad(x["img"].astype(dtype), ((0, 0), (text_tokens, 0), (0, 0))) * image_mask
  h = (ops.linear(img, p["w_in"]) + p["b_in"]) * image_mask
  txt = ops.rms_norm(x["txt"].astype(dtype) * text_mask, p["text_norm"], 1e-6)
  txt = (ops.linear(txt, p["w_text"]) + p["b_text"]) * text_mask
  h = h + jnp.pad(txt, ((0, 0), (0, h.shape[1] - text_tokens), (0, 0)))
  h = h + p["image_embed"][image.astype(jnp.int32)]

  cond = _time_embedding(p, x["timestep"], dtype)[:, None]
  cond = jax.nn.silu(ops.linear(cond, p["w_cond"]) + p["b_cond"])
  cos, sin = _rope_tables(cfg, x["position_ids"])
  return h, x["segment_ids"], cos, sin, cond


def _block(cfg: IdeogramConfig, h: jax.Array, p: dict, mask, cos, sin, cond) -> jax.Array:
  b, s, e = h.shape
  heads = (b, s, cfg.num_attention_heads, cfg.attention_head_dim)
  mod = ops.linear(cond, p["w_mod"]) + p["b_mod"]
  attn_scale, attn_gate, mlp_scale, mlp_gate = jnp.split(mod, 4, axis=-1)

  x = ops.rms_norm(h, p["attn_norm"], cfg.norm_eps) * (1 + attn_scale)
  q = ops.rms_norm(ops.linear(x, p["wq"]).reshape(heads), p["q_norm"], 1e-5)
  k = ops.rms_norm(ops.linear(x, p["wk"]).reshape(heads), p["k_norm"], 1e-5)
  v = ops.linear(x, p["wv"]).reshape(heads)
  q, k = ops.rope(q, cos, sin), ops.rope(k, cos, sin)
  attn = ops.attention(q, k, v, causal=False, mask=mask).reshape(b, s, e)
  attn = ops.rms_norm(ops.linear(attn, p["wo"]), p["attn_out_norm"], cfg.norm_eps)
  h = h + jnp.tanh(attn_gate) * attn

  x = ops.rms_norm(h, p["mlp_norm"], cfg.norm_eps) * (1 + mlp_scale)
  mlp = ops.swiglu(x, p["w_gate"], p["w_up"], p["w_down"])
  return h + jnp.tanh(mlp_gate) * ops.rms_norm(mlp, p["mlp_out_norm"], cfg.norm_eps)


def _final(params: dict, h: jax.Array, cond: jax.Array, dtype) -> jax.Array:
  """The velocity of every token from the last block's hidden states."""
  p = {k: params[k].astype(dtype) for k in _LAST_AXES}
  x = h.astype(jnp.float32)
  x = x - jnp.mean(x, axis=-1, keepdims=True)
  x = (x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)).astype(dtype)
  scale = 1 + ops.linear(jax.nn.silu(cond), p["w_final_mod"]) + p["b_final_mod"]
  return ops.linear(x * scale, p["w_out"]) + p["b_out"]


class Model:
  """Ideogram 4 split into pipeline stages, for the trainer's layouts.

  `blocks` gives the blocks per stage; the default splits them as evenly as
  possible, earlier stages taking the extra blocks. Every split starts from
  the same weights. `constrain`, when given, is applied to the conditioning,
  and to the hidden states after the embedding and after every block.
  """

  logical_axes = staticmethod(logical_axes)
  decays = staticmethod(decays)

  def __init__(self, config: IdeogramConfig, *, num_stages: int,
               blocks: Sequence[int] | None = None, dtype=jnp.bfloat16, remat: bool = True,
               constrain: Callable | None = None):
    total = config.num_layers
    if blocks is None:
      blocks = tuple(total // num_stages + (i < total % num_stages) for i in range(num_stages))
    if len(blocks) != num_stages or sum(blocks) != total or min(blocks) < 1:
      raise ValueError(f"stage blocks {tuple(blocks)} do not split {total} blocks")
    self.config = config
    self.num_stages = num_stages
    self.blocks = tuple(blocks)
    self.dtype = dtype
    self._remat = remat
    self._constrain = constrain or (lambda h: h)

  def split(self, params: dict) -> tuple[dict, ...]:
    """Stage parameter dicts from `init_params`-shaped parameters."""
    stages, start = [], 0
    for i, n in enumerate(self.blocks):
      p = {"layers": {k: v[start:start + n] for k, v in params["layers"].items()}}
      if i == 0:
        p.update({k: params[k] for k in _FIRST_AXES})
      if i == self.num_stages - 1:
        p.update({k: params[k] for k in _LAST_AXES})
      stages.append(p)
      start += n
    return tuple(stages)

  def merge(self, stage_params: Sequence[dict]) -> dict:
    """`init_params`-shaped parameters, on the host, from stage parameters on
    whatever meshes: the inverse of `split`."""
    params = {"layers": {
        k: np.concatenate([np.asarray(p["layers"][k]) for p in stage_params])
        for k in _LAYER_AXES}}
    params.update({k: np.asarray(stage_params[0][k]) for k in _FIRST_AXES})
    params.update({k: np.asarray(stage_params[-1][k]) for k in _LAST_AXES})
    return params

  def init(self, seed: int = 0) -> tuple[dict, ...]:
    """Stage parameters, randomly initialized."""
    return self.split(jax.jit(init_params, static_argnums=0)(self.config, jax.random.key(seed)))

  def abstract(self, seed: int = 0) -> tuple[dict, ...]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return jax.eval_shape(lambda: self.split(init_params(self.config, jax.random.key(seed))))

  def load(self, path: str) -> tuple[dict, ...]:
    """Stage parameters: on the host from published weights, when `path` is a
    safetensors file or a directory of them (`pretrained_params`); otherwise
    on one device from `init_params`-shaped parameters that
    `jax_titan.trainer.checkpoint.save_params` wrote."""
    from jax_titan.trainer import checkpoint  # pylint: disable=g-import-not-at-top

    if pretrained.is_safetensors(path):
      return self.split(pretrained_params(path, self.config))
    device = jax.sharding.make_single_device_sharding(jax.local_devices()[0])
    target = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=device),
        jax.eval_shape(lambda: init_params(self.config, jax.random.key(0))))
    return self.split(checkpoint.restore_params(path, target))

  def forward(self, stage: int, params: dict, x):
    """Runs one stage.

    Stage 0 takes a dict of `img` noised latent tokens [batch, image_tokens,
    in_channels], `txt` text features of the text positions [batch,
    text_tokens, llm_features_dim], `timestep` [batch], and `position_ids`
    [batch, tokens, 3], `segment_ids` and `indicator` [batch, tokens] over the
    text tokens followed by the image tokens. Tokens attend within their
    segment, so padding gets a segment of its own. The last stage returns the
    predicted velocity [batch, tokens, in_channels].
    """
    cfg, dtype = self.config, self.dtype
    h, segment_ids, cos, sin, cond = _embed(cfg, params, x, dtype) if stage == 0 else x
    h, cond = self._constrain(h), self._constrain(cond)
    mask = (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None]

    def body(h, p):
      return self._constrain(_block(cfg, h, p, mask, cos, sin, cond)), None

    if self._remat:
      body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
    h, _ = jax.lax.scan(body, h, jax.tree.map(lambda w: w.astype(dtype), params["layers"]))
    if stage < self.num_stages - 1:
      return h, segment_ids, cos, sin, cond
    return _final(params, h, cond, dtype)

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """Stage functions for `jax_titan.mpmd.make_train_step`.

    The last stage also takes the velocity target of the image tokens and
    returns `flow_matching_loss`.
    """
    fns = [functools.partial(self.forward, i) for i in range(self.num_stages - 1)]
    last = self.num_stages - 1
    fns.append(lambda p, x, t: flow_matching_loss(self.forward(last, p, x), t))
    return fns

  def flops_per_sample(self, image_tokens: int, text_tokens: int) -> float:
    return self.config.flops_per_sample(image_tokens, text_tokens)
