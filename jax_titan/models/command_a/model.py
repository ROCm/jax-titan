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
`weights.tensors_from_params` gives them back under the checkpoint's names. Every
token goes through every expert here, weighted by zero outside its router's
choice: the checkpoint's result at the cost of all experts.

Written after transformers' implementation (`cohere2_moe`, `cohere2_vision`,
`siglip`; Apache-2.0).
"""

from collections.abc import Callable, Collection, Sequence
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from jax_titan import ops
from jax_titan.models import losses
from jax_titan.models import weights as weights_lib


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

  def __post_init__(self):
    if self.num_shared_experts < 1:
      raise NotImplementedError("Command A+ without a shared MLP (num_shared_experts < 1)")

  @property
  def sliding(self) -> tuple[bool, ...]:
    """Whether each layer attends within the sliding window."""
    return tuple((i + 1) % self.sliding_window_pattern != 0 for i in range(self.num_hidden_layers))

  @property
  def tile_tokens(self) -> int:
    """The vectors, and image tokens, of one tile."""
    return (self.vision_image_size // self.vision_patch_size // self.downsample_factor) ** 2

  def flops_per_token(self, seq_len: int, frozen: Collection[str] = ()) -> float:
    """Training FLOPs per token as the model is defined: per weight a token
    meets 6, or 4 for a parameter named in `frozen`, which gets no gradient,
    its router's experts only, plus attention."""
    e, f, d = self.hidden_size, self.intermediate_size, self.head_dim
    wide, narrow = self.num_attention_heads * d, self.num_key_value_heads * d
    experts = 3 * e * f * self.num_experts_per_tok
    rest = 2 * e * (wide + narrow) + e * self.num_experts + 3 * e * f * self.num_shared_experts
    layers, attention = self.num_hidden_layers, self.num_hidden_layers * wide * seq_len
    return (layers * (6.0 * rest + (4.0 if "w_gate" in frozen else 6.0) * experts)
            + (4.0 if "embed" in frozen else 6.0) * e * self.vocab_size + 6.0 * attention)


CONFIGS = {"command-a-plus": CommandAConfig()}

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


def decays(path: tuple, leaf) -> bool:
  """Whether a parameter takes weight decay: every one but the weights of the
  norms and, in the vision tower, their biases."""
  del leaf
  return path[-1].key not in ("norm", "final_norm", "norm1", "norm1_bias", "norm2", "norm2_bias",
                              "final_norm_bias")


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

  On the host, with transformers' fp32 arithmetic: inside a compiled program
  XLA folds the frequencies to other last bits.
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
  at a sequence's image tokens, in order. A sequence has at most as many image
  tokens as `pixels` gives it vectors: further ones would repeat the last."""
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
  # Every position attends to itself too, so that a padded one attends to something.
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


_EXPERTS = ("w_gate", "w_up", "w_down")
_FREEZE = {"embed": ("embed",), "experts": _EXPERTS}


def _split_layers(cfg: CommandAConfig, stages: int) -> tuple[int, ...]:
  """The layers of each stage: as even as possible, earlier stages taking the extra ones."""
  total = cfg.num_hidden_layers
  return tuple(total // stages + (i < total % stages) for i in range(stages))


class Model:
  """Command A+ for the trainer's layouts.

  `layers` gives the layers per stage; the default splits them as evenly as
  possible, earlier stages taking the extra layers. `freeze` names groups of
  parameters that are not trained: `embed`, the embedding matrix, and
  `experts`, the routed experts of every layer. The output head is the
  embedding matrix, which the first stage and the last both need: the model
  splits into more than one stage only with the matrix frozen, and the two
  stages then hold a copy each.

  With `images` the model takes tiles too and its parameters include the
  vision tower and the projector; without, they are left out and not trained.
  `constrain`, when given, is applied to the hidden states after the embedding
  and every layer. `microbatches` is for a layout that runs a step's
  microbatches as one batch: the loss is then `losses.masked_cross_entropy`'s
  for that many.
  """

  logical_axes = staticmethod(logical_axes)
  decays = staticmethod(decays)

  def __init__(self, config: CommandAConfig, *, num_stages: int = 1,
               layers: Sequence[int] | None = None, dtype=jnp.bfloat16, remat: bool = True,
               constrain: Callable | None = None, images: bool = False,
               freeze: Sequence[str] = (), microbatches: int = 1):
    total = config.num_hidden_layers
    if layers is None:
      layers = _split_layers(config, num_stages)
    if len(layers) != num_stages or sum(layers) != total or min(layers) < 1:
      raise ValueError(f"stage layers {tuple(layers)} do not split {total} layers")
    if isinstance(freeze, str) or not set(freeze) <= _FREEZE.keys():
      raise ValueError(f"freeze takes a list of {', '.join(_FREEZE)}, not {freeze!r}")
    if num_stages > 1 and "embed" not in freeze:
      raise NotImplementedError(
          "Command A+ trains as one stage, or as more with its embedding frozen: its output "
          "head is its embedding matrix")
    self.config = config
    self.num_stages = num_stages
    self.layers = tuple(layers)
    self.dtype = dtype
    self.images = images
    self._remat = remat
    self._constrain = constrain or (lambda h: h)
    self._frozen = {name for group in freeze for name in _FREEZE[group]}
    self._microbatches = microbatches

  def is_frozen(self, path: tuple) -> bool:
    """Whether the parameter that the keys of `path` lead to in a stage's
    parameters is not trained."""
    return path[0] in ("embed", "layers") and path[-1] in self._frozen

  def _init(self, key):
    return init_params(self.config, key, self.images)

  def split(self, params: dict) -> tuple[dict, ...]:
    """Stage parameter dicts from `init_params`-shaped parameters."""
    stages, start = [], 0
    for i, n in enumerate(self.layers):
      p = {"layers": {k: v[start:start + n] for k, v in params["layers"].items()}}
      if i == 0:
        p |= {k: v for k, v in params.items() if k not in ("layers", "final_norm")}
      if i == self.num_stages - 1:
        p["final_norm"] = params["final_norm"]
        p.setdefault("embed", params["embed"])
      stages.append(p)
      start += n
    return tuple(stages)

  def merge(self, stage_params: Sequence[dict]) -> dict:
    """`init_params`-shaped parameters, on the host, from stage parameters on
    whatever meshes: what `split` took, or the part of it that the stages'
    trained parameters alone are."""
    first, last = stage_params[0], stage_params[-1]
    params = {"layers": {
        k: np.concatenate([np.asarray(p["layers"][k]) for p in stage_params])
        for k in first["layers"]}}
    params |= {k: jax.tree.map(np.asarray, v) for k, v in first.items() if k != "layers"}
    if "final_norm" in last:
      params["final_norm"] = np.asarray(last["final_norm"])
    return params

  def init(self, seed: int = 0) -> tuple[dict, ...]:
    """Stage parameters, randomly initialized."""
    return self.split(jax.jit(self._init)(jax.random.key(seed)))

  def abstract(self, seed: int = 0) -> tuple[dict, ...]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return jax.eval_shape(lambda: self.split(self._init(jax.random.key(seed))))

  def load(self, path: str) -> tuple[dict, ...]:
    """Stage parameters, on one device, from `init_params`-shaped parameters
    that `jax_titan.models.weights.save_params` wrote: all of them, which a
    run with frozen parameters does not save."""
    device = jax.sharding.make_single_device_sharding(jax.local_devices()[0])
    target = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=device),
        jax.eval_shape(self._init, jax.random.key(0)))
    saved = weights_lib.saved_tree(path)
    missing = sorted(target.keys() - saved.keys()) + sorted(
        f"layers.{k}" for k in target["layers"].keys() - saved.get("layers", {}).keys())
    if missing:
      raise ValueError(
          f"{path} lacks {', '.join(missing)}: a run saves the parameters it trained, not the "
          "ones it kept frozen. --init_from takes all parameters; the trained ones alone are "
          "for command_a.generate --params")
    return self.split(weights_lib.restore_params(path, target))

  def place(self, path: str, targets: Sequence[dict]) -> tuple[dict, ...]:
    """Stage parameters of the published weights at `path`, each read straight
    onto the sharding and in the dtype of its entry in `targets`, one tree of
    ShapeDtypeStructs per stage shaped like `abstract()`'s or holding a part
    of its leaves, which are then the only ones read."""
    from jax_titan.models.command_a import weights  # pylint: disable=g-import-not-at-top

    return weights.place(path, self.config, self.layers, self.images, targets)

  def forward(self, stage: int, params: dict, x) -> jax.Array:
    """Runs one stage.

    Stage 0 takes token ids [batch, tokens], or a dict of `tokens` and
    `pixels` [batch, tiles, channels, size, size]; later stages take the
    hidden states of the stage before. The last stage returns the logits.
    """
    cfg = self.config
    if stage == 0:
      tokens, pixels = (x["tokens"], x["pixels"]) if isinstance(x, dict) else (x, None)
      x = embed(cfg, params, tokens, pixels, self.dtype)
    h = hidden(cfg, params["layers"], self._constrain(x), first=sum(self.layers[:stage]),
               remat=self._remat, constrain=self._constrain)
    return logits(cfg, params, h) if stage == self.num_stages - 1 else h

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """Stage functions for `jax_titan.mpmd.make_train_step`.

    The last stage also takes the next tokens `t` and returns their mean loss,
    over those that are not negative.
    """
    last = self.num_stages - 1
    fns = [functools.partial(self.forward, i) for i in range(last)]
    fns.append(lambda p, x, t: losses.masked_cross_entropy(
        self.forward(last, p, x), t, self._microbatches))
    return fns

  def flops_per_token(self, seq_len: int) -> float:
    return self.config.flops_per_token(seq_len, self._frozen)
