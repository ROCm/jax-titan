"""Command A+, Cohere's sparse mixture-of-experts language model with a vision tower.

A decoder-only model in which one norm feeds a layer's attention and its MLP,
and both outputs are added to the layer's input. Three layers out of four
attend within a sliding window and rotate queries and keys by position; the
fourth attends to everything before it and gets no position at all. The MLP is
a set of experts, of which a router picks a few per token, averaged with one
shared MLP that every token goes through. The output head is the embedding
matrix. Images enter as tiles: a SigLIP vision tower and a projector turn each
tile into vectors that take the places of the image tokens among the token
embeddings.

Parameters are those of the published checkpoint
(`CohereLabs/command-a-plus-05-2026-bf16`), with kernels as [in, out], layers
and experts stacked, and the query and key dimensions of each head reordered
so that `jax_titan.ops.rope` rotates the pairs the checkpoint means.
`tensors_from_params` gives them back under the checkpoint's names. Every
token goes through every expert here, weighted by zero outside its router's
choice: the checkpoint's result at the cost of all experts.

Written after transformers' implementation (`cohere2_moe`, `cohere2_vision`,
`siglip`; Apache-2.0).
"""

from collections.abc import Callable
import dataclasses
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from jax_titan import ops
from jax_titan.models import pretrained


@dataclasses.dataclass(frozen=True)
class CommandAConfig:
  """The fields of the checkpoint's `config.json`: those of its `text_config`,
  those of its `vision_config` under `vision_`, and three of its top level."""

  vocab_size: int = 262144
  hidden_size: int = 4096
  num_hidden_layers: int = 32
  num_attention_heads: int = 128
  num_key_value_heads: int = 8
  head_dim: int = 128
  intermediate_size: int = 4096
  num_experts: int = 128
  num_experts_per_tok: int = 8
  # The shared MLP's intermediate size in units of an expert's.
  num_shared_experts: int = 4
  sliding_window: int = 4096
  # Every layer whose number is a multiple of this attends to all earlier tokens.
  sliding_window_pattern: int = 4
  layer_norm_eps: float = 1e-5
  rope_theta: float = 50000.0
  bos_token_id: int = 2
  eos_token_id: int = 255001
  pad_token_id: int = 0
  image_token_id: int = 255031
  downsample_factor: int = 2
  alignment_intermediate_size: int = 12288
  vision_hidden_size: int = 1152
  vision_intermediate_size: int = 4304
  vision_num_hidden_layers: int = 27
  vision_num_attention_heads: int = 16
  vision_image_size: int = 512
  vision_patch_size: int = 16
  vision_num_channels: int = 3
  vision_layer_norm_eps: float = 1e-6

  @property
  def sliding(self) -> tuple[bool, ...]:
    """Whether each layer attends within the sliding window."""
    return tuple((i + 1) % self.sliding_window_pattern != 0 for i in range(self.num_hidden_layers))

  @property
  def tile_tokens(self) -> int:
    """The vectors, and image tokens, of one tile."""
    return (self.vision_image_size // self.vision_patch_size // self.downsample_factor) ** 2

  def flops_per_token(self, seq_len: int) -> float:
    """Training FLOPs per token as the model is defined: 6 per weight a token
    meets, its router's experts only, plus attention."""
    e, f, d = self.hidden_size, self.intermediate_size, self.head_dim
    wide, narrow = self.num_attention_heads * d, self.num_key_value_heads * d
    mlp = 3 * e * f * (self.num_experts_per_tok + self.num_shared_experts)
    weights = self.num_hidden_layers * (2 * e * wide + 2 * e * narrow + e * self.num_experts + mlp)
    attention = self.num_hidden_layers * wide * seq_len
    return 6.0 * (weights + e * self.vocab_size + attention)


_AXES = {
    "embed": ("vocab", "embed"),
    "final_norm": ("embed",),
    "norm": ("embed",),
    "wq": ("embed", "heads"),
    "wk": ("embed", "kv_heads"),
    "wv": ("embed", "kv_heads"),
    "wo": ("heads", "embed"),
    "w_router": ("embed", "experts"),
    "w_gate": ("experts", "embed", "mlp"),
    "w_up": ("experts", "embed", "mlp"),
    "w_down": ("experts", "mlp", "embed"),
    "shared_gate": ("embed", "mlp"),
    "shared_up": ("embed", "mlp"),
    "shared_down": ("mlp", "embed"),
    # The vision tower and the projector.
    "w_patch": ("pixels", "embed"),
    "b_patch": ("embed",),
    "positions": ("patches", "embed"),
    "norm1": ("embed",),
    "norm1_bias": ("embed",),
    "norm2": ("embed",),
    "norm2_bias": ("embed",),
    "final_norm_bias": ("embed",),
    "bq": ("heads",),
    "bk": ("heads",),
    "bv": ("heads",),
    "bo": ("embed",),
    "w1": ("embed", "mlp"),
    "b1": ("mlp",),
    "w2": ("mlp", "embed"),
    "b2": ("embed",),
}


def logical_axes(path: tuple, leaf) -> tuple | None:
  """Logical axes of a parameter, or of an optimizer-state leaf mirroring one.

  Returns None for leaves that are not parameter-shaped, such as step counts.
  """
  keys = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
  if not keys:
    return None
  axes = (("layers",) if "layers" in keys[:-1] else ()) + _AXES.get(keys[-1], ())
  return axes if len(axes) == getattr(leaf, "ndim", -1) else None


def init_params(cfg: CommandAConfig, key: jax.Array, images: bool = True) -> dict:
  """Random fp32 parameters, layers and experts stacked; with `images`, also
  those of the vision tower and the projector.

  Weights are normal with variance 1 / fan-in, biases zero and norm weights one.
  """
  n, e, f, d = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
  q, kv, x = cfg.num_attention_heads * d, cfg.num_key_value_heads * d, cfg.num_experts
  shared = f * cfg.num_shared_experts
  keys = iter(jax.random.split(key, 24))

  def weight(*shape):
    return shape[-2] ** -0.5 * jax.random.normal(next(keys), shape, jnp.float32)

  def ones(*shape):
    return jnp.ones(shape, jnp.float32)

  def zeros(*shape):
    return jnp.zeros(shape, jnp.float32)

  params = {
      "embed": e ** -0.5 * jax.random.normal(next(keys), (cfg.vocab_size, e), jnp.float32),
      "layers": {
          "norm": ones(n, e),
          "wq": weight(n, e, q),
          "wk": weight(n, e, kv),
          "wv": weight(n, e, kv),
          "wo": weight(n, q, e),
          "w_router": weight(n, e, x),
          "w_gate": weight(n, x, e, f),
          "w_up": weight(n, x, e, f),
          "w_down": weight(n, x, f, e),
          "shared_gate": weight(n, e, shared),
          "shared_up": weight(n, e, shared),
          "shared_down": weight(n, shared, e),
      },
      "final_norm": ones(e),
  }
  if not images:
    return params
  m, v, g = cfg.vision_num_hidden_layers, cfg.vision_hidden_size, cfg.vision_intermediate_size
  a = cfg.alignment_intermediate_size
  patches = (cfg.vision_image_size // cfg.vision_patch_size) ** 2
  params["vision"] = {
      "w_patch": weight(cfg.vision_num_channels * cfg.vision_patch_size**2, v),
      "b_patch": zeros(v),
      "positions": v ** -0.5 * jax.random.normal(next(keys), (patches, v), jnp.float32),
      "layers": {
          "norm1": ones(m, v), "norm1_bias": zeros(m, v),
          "wq": weight(m, v, v), "bq": zeros(m, v),
          "wk": weight(m, v, v), "bk": zeros(m, v),
          "wv": weight(m, v, v), "bv": zeros(m, v),
          "wo": weight(m, v, v), "bo": zeros(m, v),
          "norm2": ones(m, v), "norm2_bias": zeros(m, v),
          "w1": weight(m, v, g), "b1": zeros(m, g),
          "w2": weight(m, g, v), "b2": zeros(m, v),
      },
      "final_norm": ones(v),
      "final_norm_bias": zeros(v),
  }
  params["projector"] = {
      "w1": weight(v * cfg.downsample_factor**2, a), "b1": zeros(a),
      "w2": weight(a // 2, e), "b2": zeros(e),
  }
  return params


def pretrained_config(path) -> CommandAConfig:
  """The config of a published checkpoint, from the `config.json` beside its weights."""

  types = []

  def flat(values):
    text, vision = values["text_config"], values["vision_config"]
    types.extend(text.get("layer_types") or ())
    fields = text | {"vision_" + k: v for k, v in vision.items()}
    top = ("image_token_id", "downsample_factor", "alignment_intermediate_size")
    fields |= {k: values[k] for k in top if k in values}
    theta = text.get("rope_parameters", {}).get("rope_theta", text.get("rope_theta"))
    if theta is not None:
      fields["rope_theta"] = float(theta)
    # Other routers, norms and combinations exist in transformers' model; this module has these.
    expected = {"expert_selection_fn": "sigmoid", "norm_topk_prob": True, "rms_norm_eps": None,
                "shared_expert_combination_strategy": "average"}
    other = {k: text[k] for k, v in expected.items() if k in text and text[k] != v}
    if other:
      raise NotImplementedError(f"{path}: a variant of the model that is not implemented: {other}")
    return fields

  cfg = pretrained.config(CommandAConfig, path, flat)
  if types and tuple(t == "sliding_attention" for t in types) != cfg.sliding:
    raise NotImplementedError(
        f"{path}: its full-attention layers are not every {cfg.sliding_window_pattern}th")
  return cfg


def _rotary_order(cfg: CommandAConfig) -> np.ndarray:
  """Where each dimension of a head goes: the checkpoint rotates the pairs
  (0, 1), (2, 3), ..., `ops.rope` the pairs (i, i + head_dim / 2)."""
  return np.concatenate([np.arange(0, cfg.head_dim, 2), np.arange(1, cfg.head_dim, 2)])


def params_from_tensors(tensors: dict, cfg: CommandAConfig, images: bool = True) -> dict:
  """`init_params`-shaped fp32 parameters from tensors named as the published
  checkpoint names them: every expert on its own, and the vision tower under
  `vision_tower.vision_model.` or, as transformers writes it, `vision_tower.`."""
  lm = "model.language_model."
  layers, order = range(cfg.num_hidden_layers), _rotary_order(cfg)

  def kernel(name):
    return pretrained.weight(tensors, name + ".weight").T

  def vector(name, kind="weight"):
    return np.asarray(tensors[f"{name}.{kind}"]).astype(np.float32)

  def stacked(fn, name, prefix=lm, count=cfg.num_hidden_layers, **kwargs):
    return np.stack([fn(f"{prefix}layers.{i}.{name}", **kwargs) for i in range(count)])

  def rotary(w):
    return w.reshape(*w.shape[:2], -1, cfg.head_dim)[..., order].reshape(w.shape)

  def experts(name):
    return np.stack([
        np.stack([kernel(f"{lm}layers.{i}.mlp.experts.{x}.{name}") for x in range(cfg.num_experts)])
        for i in layers])

  params = {
      "embed": vector(lm + "embed_tokens"),
      "layers": {
          "norm": stacked(vector, "input_layernorm"),
          "wq": rotary(stacked(kernel, "self_attn.q_proj")),
          "wk": rotary(stacked(kernel, "self_attn.k_proj")),
          "wv": stacked(kernel, "self_attn.v_proj"),
          "wo": stacked(kernel, "self_attn.o_proj"),
          "w_router": stacked(kernel, "mlp.gate"),
          "w_gate": experts("gate_proj"),
          "w_up": experts("up_proj"),
          "w_down": experts("down_proj"),
          "shared_gate": stacked(kernel, "mlp.shared_experts.gate_proj"),
          "shared_up": stacked(kernel, "mlp.shared_experts.up_proj"),
          "shared_down": stacked(kernel, "mlp.shared_experts.down_proj"),
      },
      "final_norm": vector(lm + "norm"),
  }
  if not images:
    return params
  patch = "embeddings.patch_embedding.weight"
  vm = next(k for k in tensors if k.endswith(patch))[:-len(patch)]
  enc, m = vm + "encoder.", cfg.vision_num_hidden_layers

  def vision(fn, name, **kwargs):
    return stacked(fn, name, prefix=enc, count=m, **kwargs)

  def bias(name):
    return vision(vector, name, kind="bias")

  # A patch is flattened as channel, row, column, the order of the checkpoint's convolution kernel.
  params["vision"] = {
      "w_patch": vector(vm + "embeddings.patch_embedding").reshape(cfg.vision_hidden_size, -1).T,
      "b_patch": vector(vm + "embeddings.patch_embedding", "bias"),
      "positions": vector(vm + "embeddings.position_embedding"),
      "layers": {
          "norm1": vision(vector, "layer_norm1"), "norm1_bias": bias("layer_norm1"),
          "wq": vision(kernel, "self_attn.q_proj"), "bq": bias("self_attn.q_proj"),
          "wk": vision(kernel, "self_attn.k_proj"), "bk": bias("self_attn.k_proj"),
          "wv": vision(kernel, "self_attn.v_proj"), "bv": bias("self_attn.v_proj"),
          "wo": vision(kernel, "self_attn.out_proj"), "bo": bias("self_attn.out_proj"),
          "norm2": vision(vector, "layer_norm2"), "norm2_bias": bias("layer_norm2"),
          "w1": vision(kernel, "mlp.fc1"), "b1": bias("mlp.fc1"),
          "w2": vision(kernel, "mlp.fc2"), "b2": bias("mlp.fc2"),
      },
      "final_norm": vector(vm + "post_layernorm"),
      "final_norm_bias": vector(vm + "post_layernorm", "bias"),
  }
  mm = "model.multi_modal_projector."
  params["projector"] = {
      "w1": kernel(mm + "linear_1"), "b1": vector(mm + "linear_1", "bias"),
      "w2": kernel(mm + "linear_2"), "b2": vector(mm + "linear_2", "bias"),
  }
  return params


def tensors_from_params(params: dict, cfg: CommandAConfig) -> dict[str, np.ndarray]:
  """The published checkpoint's tensors, by its names, from `init_params`-shaped
  parameters: the inverse of `params_from_tensors`."""
  lm, tensors = "model.language_model.", {}
  inverse = np.argsort(_rotary_order(cfg))

  def put(name, value, kind="weight"):
    tensors[f"{name}.{kind}"] = np.ascontiguousarray(value)

  def rotary(w):
    return w.reshape(w.shape[0], -1, cfg.head_dim)[..., inverse].reshape(w.shape)

  p = {k: np.asarray(v) for k, v in params["layers"].items()}
  put(lm + "embed_tokens", np.asarray(params["embed"]))
  for i in range(cfg.num_hidden_layers):
    layer = f"{lm}layers.{i}."
    put(layer + "input_layernorm", p["norm"][i])
    put(layer + "self_attn.q_proj", rotary(p["wq"][i]).T)
    put(layer + "self_attn.k_proj", rotary(p["wk"][i]).T)
    put(layer + "self_attn.v_proj", p["wv"][i].T)
    put(layer + "self_attn.o_proj", p["wo"][i].T)
    put(layer + "mlp.gate", p["w_router"][i].T)
    for part in ("gate", "up", "down"):
      for x in range(cfg.num_experts):
        put(f"{layer}mlp.experts.{x}.{part}_proj", p["w_" + part][i, x].T)
      put(f"{layer}mlp.shared_experts.{part}_proj", p["shared_" + part][i].T)
  put(lm + "norm", np.asarray(params["final_norm"]))
  if "vision" not in params:
    return tensors
  v = jax.tree.map(np.asarray, params["vision"])
  vm, size = "model.vision_tower.vision_model.", cfg.vision_patch_size
  put(vm + "embeddings.patch_embedding",
      v["w_patch"].T.reshape(cfg.vision_hidden_size, cfg.vision_num_channels, size, size))
  put(vm + "embeddings.patch_embedding", v["b_patch"], "bias")
  put(vm + "embeddings.position_embedding", v["positions"])
  names = (("layer_norm1", "norm1", "norm1_bias"), ("layer_norm2", "norm2", "norm2_bias"),
           ("self_attn.q_proj", "wq", "bq"), ("self_attn.k_proj", "wk", "bk"),
           ("self_attn.v_proj", "wv", "bv"), ("self_attn.out_proj", "wo", "bo"),
           ("mlp.fc1", "w1", "b1"), ("mlp.fc2", "w2", "b2"))
  for i in range(cfg.vision_num_hidden_layers):
    for theirs, weight, bias in names:
      put(f"{vm}encoder.layers.{i}.{theirs}", v["layers"][weight][i].T)
      put(f"{vm}encoder.layers.{i}.{theirs}", v["layers"][bias][i], "bias")
  put(vm + "post_layernorm", v["final_norm"])
  put(vm + "post_layernorm", v["final_norm_bias"], "bias")
  mm = "model.multi_modal_projector."
  for theirs, weight, bias in (("linear_1", "w1", "b1"), ("linear_2", "w2", "b2")):
    put(mm + theirs, np.asarray(params["projector"][weight]).T)
    put(mm + theirs, np.asarray(params["projector"][bias]), "bias")
  return tensors


def load(path, dtype=jnp.bfloat16, images: bool = True) -> tuple[CommandAConfig, dict]:
  """The config and the parameters in `dtype`, on the default device, of the
  published weights at `path`, a directory of safetensors files with their
  `config.json`; without `images`, the language model alone."""
  cfg = pretrained_config(path)
  return cfg, pretrained.to_device(params_from_tensors(pretrained.read(path), cfg, images), dtype)


def save(path, cfg: CommandAConfig, params: dict) -> None:
  """Writes `init_params`-shaped parameters as a checkpoint in the published
  layout, `model.safetensors` and `config.json`, which `load` and transformers read."""
  path = pathlib.Path(path)
  path.mkdir(parents=True, exist_ok=True)
  fields = dataclasses.asdict(cfg)
  vision = {k[len("vision_"):]: v for k, v in fields.items() if k.startswith("vision_")}
  top = ("image_token_id", "downsample_factor", "alignment_intermediate_size")
  text = {k: v for k, v in fields.items() if not k.startswith("vision_") and k not in top}
  text |= {"model_type": "cohere2_moe", "expert_selection_fn": "sigmoid", "logit_scale": 1.0,
           "layer_types": ["sliding_attention" if s else "full_attention" for s in cfg.sliding]}
  vision |= {"model_type": "siglip_vision_model", "hidden_act": "gelu_pytorch_tanh",
             "vision_use_head": False}
  config = {"architectures": ["Cohere2VisionForConditionalGeneration"],
            "model_type": "cohere2_vision", "text_config": text, "vision_config": vision,
            **{k: fields[k] for k in top}}
  (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
  # transformers reads a safetensors file only when its header names the framework it is for.
  pretrained.write(path / "model.safetensors", tensors_from_params(params, cfg), {"format": "pt"})


def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
  """The mean loss over the targets that are not negative."""
  logits = logits.astype(jnp.float32)
  keep = targets >= 0
  gold = jnp.take_along_axis(logits, jnp.maximum(targets, 0)[..., None], axis=-1)[..., 0]
  loss = (jax.nn.logsumexp(logits, axis=-1) - gold) * keep
  return jnp.sum(loss) / jnp.maximum(jnp.sum(keep), 1)


def _layer_norm(x: jax.Array, weight: jax.Array, eps: float,
                bias: jax.Array | None = None) -> jax.Array:
  y = x.astype(jnp.float32)
  y = y - jnp.mean(y, axis=-1, keepdims=True)
  y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps) * weight.astype(jnp.float32)
  if bias is not None:
    y = y + bias.astype(jnp.float32)
  return y.astype(x.dtype)


def _rope_tables(cfg: CommandAConfig, tokens: int) -> tuple[np.ndarray, np.ndarray]:
  """cos and sin of the rotary angles, [tokens, head_dim // 2] in fp32.

  On the host, as transformers rounds them: inside a compiled program XLA
  folds the frequencies to other last bits.
  """
  d = cfg.head_dim
  exponent = np.arange(0, d, 2, dtype=np.float32) / np.float32(d)
  power = np.power(np.float64(cfg.rope_theta), exponent.astype(np.float64)).astype(np.float32)
  angles = np.arange(tokens, dtype=np.float32)[:, None] * (np.float32(1.0) / power)
  return np.cos(angles), np.sin(angles)


def _mlp(cfg: CommandAConfig, x: jax.Array, p: dict) -> jax.Array:
  """The routed experts and the shared MLP on the normed input, averaged."""
  top, chosen = jax.lax.top_k(ops.linear(x, p["w_router"]), cfg.num_experts_per_tok)
  share = jax.nn.sigmoid(top)
  share = share / jnp.sum(share, axis=-1, keepdims=True)
  # [batch, tokens, experts]: a token's share of each expert, zero for those not chosen.
  chosen = jax.nn.one_hot(chosen, cfg.num_experts, dtype=x.dtype)
  share = jnp.sum(chosen * share[..., None], axis=-2)
  gate = jnp.einsum("bte,xef->btxf", x, p["w_gate"])
  up = jnp.einsum("bte,xef->btxf", x, p["w_up"])
  routed = jnp.einsum("btxf,xfe->bte", jax.nn.silu(gate) * up * share[..., None], p["w_down"])
  shared = ops.swiglu(x, p["shared_gate"], p["shared_up"], p["shared_down"])
  return (routed + shared) / 2


def _layer(cfg: CommandAConfig, h: jax.Array, p: dict, mask, cos, sin) -> jax.Array:
  b, s, _ = h.shape
  d = cfg.head_dim
  x = _layer_norm(h, p["norm"], cfg.layer_norm_eps)
  q = ops.linear(x, p["wq"]).reshape(b, s, cfg.num_attention_heads, d)
  k = ops.linear(x, p["wk"]).reshape(b, s, cfg.num_key_value_heads, d)
  v = ops.linear(x, p["wv"]).reshape(b, s, cfg.num_key_value_heads, d)
  q, k = ops.rope(q, cos, sin), ops.rope(k, cos, sin)
  attn = ops.attention(q, k, v, causal=True, mask=mask).reshape(b, s, -1)
  return h + ops.linear(attn, p["wo"]) + _mlp(cfg, x, p)


def image_features(cfg: CommandAConfig, params: dict, pixels: jax.Array,
                   dtype=jnp.bfloat16) -> jax.Array:
  """The vectors of image tiles, [batch, tiles * tile_tokens, hidden_size], tile after tile.

  `pixels` is [batch, tiles, channels, size, size], scaled to [-1, 1].
  """
  p = jax.tree.map(lambda w: w.astype(dtype), params["vision"])
  b, tiles, c, size, _ = pixels.shape
  patch, heads = cfg.vision_patch_size, cfg.vision_num_attention_heads
  side, eps = size // patch, cfg.vision_layer_norm_eps
  x = pixels.astype(dtype).reshape(b * tiles, c, side, patch, side, patch)
  x = x.transpose(0, 2, 4, 1, 3, 5).reshape(b * tiles, side * side, c * patch * patch)
  x = ops.linear(x, p["w_patch"]) + p["b_patch"] + p["positions"]

  def body(x, w):
    y = _layer_norm(x, w["norm1"], eps, w["norm1_bias"])
    q, k, v = ((ops.linear(y, w["w" + n]) + w["b" + n]).reshape(*y.shape[:2], heads, -1)
               for n in "qkv")
    attn = ops.attention(q, k, v, causal=False).reshape(y.shape)
    x = x + ops.linear(attn, w["wo"]) + w["bo"]
    y = _layer_norm(x, w["norm2"], eps, w["norm2_bias"])
    y = jax.nn.gelu(ops.linear(y, w["w1"]) + w["b1"], approximate=True)
    return x + ops.linear(y, w["w2"]) + w["b2"], None

  x, _ = jax.lax.scan(body, x, p["layers"])
  x = _layer_norm(x, p["final_norm"], eps, p["final_norm_bias"])
  # The projector joins each block of downsample_factor x downsample_factor patches.
  f, part = cfg.downsample_factor, side // cfg.downsample_factor
  x = x.reshape(b * tiles, part, f, part, f, -1).transpose(0, 1, 3, 2, 4, 5)
  x = x.reshape(b * tiles, part * part, -1)
  w = {k: v.astype(dtype) for k, v in params["projector"].items()}
  first, second = jnp.split(ops.linear(x, w["w1"]) + w["b1"], 2, axis=-1)
  x = ops.linear(first * jax.nn.silu(second), w["w2"]) + w["b2"]
  return x.reshape(b, tiles * part * part, -1)


def embed(cfg: CommandAConfig, params: dict, tokens: jax.Array, pixels: jax.Array | None = None,
          dtype=jnp.bfloat16) -> jax.Array:
  """The embeddings of `tokens` [batch, tokens], with the vectors of `pixels`
  at a sequence's image tokens, in order."""
  h = params["embed"][tokens].astype(dtype)
  if pixels is None:
    return h
  features = image_features(cfg, params, pixels, dtype)
  image = tokens == cfg.image_token_id
  index = jnp.clip(jnp.cumsum(image, axis=1) - 1, 0, features.shape[1] - 1)
  return jnp.where(image[..., None], jnp.take_along_axis(features, index[..., None], axis=1), h)


def hidden(cfg: CommandAConfig, layers: dict, h: jax.Array, valid: jax.Array | None = None, *,
           first: int = 0, remat: bool = False, constrain: Callable = lambda h: h) -> jax.Array:
  """`h` [batch, tokens, hidden_size] after `layers`, which start at layer `first` of the model.

  `valid` [batch, tokens] marks the real tokens; padding is not attended to.
  """
  s, count = h.shape[1], next(iter(layers.values())).shape[0]
  cos, sin = _rope_tables(cfg, s)
  position = np.arange(s)
  window = (position[:, None] - position[None, :] < cfg.sliding_window)[None, None]
  # A real token attends to the real tokens up to itself; padding, to itself.
  mask = np.eye(s, dtype=bool)[None, None]
  mask = mask | (True if valid is None else valid[:, None, None, :])
  sliding = np.asarray(cfg.sliding[first:first + count])

  def body(h, layer):
    p, slides = layer
    cos_, sin_ = jnp.where(slides, cos, 1.0), jnp.where(slides, sin, 0.0)
    return constrain(_layer(cfg, h, p, mask & (window | ~slides), cos_, sin_)), None

  if remat:
    body = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
  h, _ = jax.lax.scan(body, h, (jax.tree.map(lambda w: w.astype(h.dtype), layers), sliding))
  return h


def logits(cfg: CommandAConfig, params: dict, h: jax.Array) -> jax.Array:
  """The scores of every token of the vocabulary from the last layer's hidden states."""
  h = _layer_norm(h, params["final_norm"], cfg.layer_norm_eps)
  return ops.linear(h, params["embed"].astype(h.dtype).T)


def forward(cfg: CommandAConfig, params: dict, tokens: jax.Array, pixels: jax.Array | None = None,
            valid: jax.Array | None = None, dtype=jnp.bfloat16) -> jax.Array:
  """Logits [batch, tokens, vocab_size] of the next token at every position.

  `pixels` [batch, tiles, channels, size, size] holds the tiles of a sequence's
  images in the order of its image tokens; a sequence with fewer tiles than
  the batch's most leaves the last ones unused. `valid` marks the real tokens
  of padded sequences, whose outputs at padding mean nothing.
  """
  h = embed(cfg, params, tokens, pixels, dtype)
  return logits(cfg, params, hidden(cfg, params["layers"], h, valid))


class Model:
  """Command A+ for the trainer's layouts, as one stage.

  The output head is the embedding matrix, which a pipeline would need on its
  first stage and on its last, so the model does not split into stages. With
  `images` it takes tiles too and its parameters include the vision tower and
  the projector; without, they are left out and not trained. `constrain`, when
  given, is applied to the hidden states after the embedding and every layer.
  """

  logical_axes = staticmethod(logical_axes)

  def __init__(self, config: CommandAConfig, *, num_stages: int = 1, dtype=jnp.bfloat16,
               remat: bool = True, constrain: Callable | None = None, images: bool = False):
    if num_stages != 1:
      raise NotImplementedError(
          "Command A+ trains as one stage: its output head is its embedding matrix")
    self.config = config
    self.num_stages = 1
    self.dtype = dtype
    self.images = images
    self._remat = remat
    self._constrain = constrain or (lambda h: h)

  def _init(self, key):
    return init_params(self.config, key, self.images)

  def init(self, seed: int = 0) -> tuple[dict]:
    """Stage parameters, randomly initialized."""
    return (jax.jit(self._init)(jax.random.key(seed)),)

  def abstract(self, seed: int = 0) -> tuple[dict]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return (jax.eval_shape(self._init, jax.random.key(seed)),)

  def merge(self, stage_params) -> dict:
    """`init_params`-shaped parameters on the host."""
    return jax.tree.map(np.asarray, stage_params[0])

  def load(self, path: str) -> tuple[dict]:
    """Stage parameters: on the host from a published checkpoint, when `path`
    is a directory of safetensors files; otherwise on one device from
    parameters that `jax_titan.trainer.checkpoint.save_params` wrote."""
    from jax_titan.trainer import checkpoint  # pylint: disable=g-import-not-at-top

    if pretrained.is_safetensors(path):
      return (params_from_tensors(pretrained.read(path), self.config, self.images),)
    device = jax.sharding.make_single_device_sharding(jax.local_devices()[0])
    target = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=device), self.abstract()[0])
    return (checkpoint.restore_params(path, target),)

  def forward(self, stage: int, params: dict, x) -> jax.Array:
    """Logits from token ids [batch, tokens], or from a dict of `tokens` and
    `pixels` [batch, tiles, channels, size, size]."""
    del stage
    cfg = self.config
    tokens, pixels = (x["tokens"], x["pixels"]) if isinstance(x, dict) else (x, None)
    h = self._constrain(embed(cfg, params, tokens, pixels, self.dtype))
    h = hidden(cfg, params["layers"], h, remat=self._remat, constrain=self._constrain)
    return logits(cfg, params, h)

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """The one stage function for `jax_titan.mpmd.make_train_step`: the mean
    loss of the next tokens `t`, those that are not negative."""
    return [lambda p, x, t: cross_entropy(self.forward(0, p, x), t)]

  def flops_per_token(self, seq_len: int) -> float:
    return self.config.flops_per_token(seq_len)
