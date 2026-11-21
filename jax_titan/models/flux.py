"""FLUX.1's denoiser, the rectified-flow transformer of Black Forest Labs.

The architecture follows diffusers' `FluxTransformer2DModel` (0.43.0). Image
tokens, the latents in 2x2 patches, and text tokens of T5 first run through
double-stream blocks, with weights of their own per stream and attention over
both, then through single-stream blocks over the joined sequence, text first.
Every block, and the output layer, is modulated by a vector of the timestep,
FLUX.1-dev's guidance and CLIP's pooled text features. Attention takes
normalized queries and keys and rotary embeddings over three position axes.
The training objective is `jax_titan.data.diffusion`'s.

The queries and keys of a head keep their features in split halves, the
layout of `jax_titan.ops.rope`, where diffusers interleaves the pairs of its
rotary embedding: the two give the same attention.
"""

from collections.abc import Callable
import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_titan import ops


@dataclasses.dataclass(frozen=True)
class FluxConfig:
  """The fields of diffusers' `FluxTransformer2DModel` config; the defaults are FLUX.1's."""

  in_channels: int = 64
  num_layers: int = 19
  num_single_layers: int = 38
  attention_head_dim: int = 128
  num_attention_heads: int = 24
  joint_attention_dim: int = 4096
  pooled_projection_dim: int = 768
  guidance_embeds: bool = False
  axes_dims_rope: tuple[int, int, int] = (16, 56, 56)

  @property
  def emb_dim(self) -> int:
    return self.attention_head_dim * self.num_attention_heads

  def flops_per_sample(self, image_tokens: int, text_tokens: int) -> float:
    """Training FLOPs per sample: 6 per weight and token, plus attention."""
    d, tokens = self.emb_dim, image_tokens + text_tokens
    blocks = self.num_layers + self.num_single_layers
    weights = 12 * d * d * tokens * blocks + d * (
        2 * self.in_channels * image_tokens + self.joint_attention_dim * text_tokens)
    attention = 2 * tokens * tokens * d * blocks
    return 6.0 * (weights + attention)


FLAVORS = {
    "flux-dev": FluxConfig(guidance_embeds=True),
    "flux-schnell": FluxConfig(),
}
# The width of the sinusoidal embedding of the timestep and of the guidance.
_FREQUENCIES = 256
_STACKS = ("double", "single")


def logical_axes(path: tuple, leaf) -> tuple | None:
  """Logical axes of a parameter, or of an optimizer-state leaf mirroring one:
  `layers` for the block axis of the stacked blocks, which FSDP never shards.

  Returns None for leaves that are not parameter-shaped, such as step counts.
  """
  keys = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
  if not keys:
    return None
  axes = [None] * leaf.ndim
  if keys[0] in _STACKS:
    axes[0] = "layers"
  return tuple(axes)


def init_params(cfg: FluxConfig, key: jax.Array) -> dict:
  """Returns fp32 parameters of the whole model, each kind of block stacked along a
  layer axis.

  Weights are normal with variance 1 / fan-in, biases zero and norm weights one.
  """
  e, d, c = cfg.emb_dim, cfg.attention_head_dim, cfg.in_channels
  keys = iter(jax.random.split(key, 64))

  def linear(*shape):
    """A [..., fan_in, fan_out] weight `w` and its bias `b`."""
    w = shape[-2] ** -0.5 * jax.random.normal(next(keys), shape, jnp.float32)
    return w, jnp.zeros(shape[:-2] + shape[-1:], jnp.float32)

  def mlp(fan_in):
    (w1, b1), (w2, b2) = linear(fan_in, e), linear(e, e)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

  def attention(n):
    p = {}
    for name in "qkv":
      p[f"w{name}"], p[f"b{name}"] = linear(n, e, e)
    p["q_norm"] = p["k_norm"] = jnp.ones((n, d), jnp.float32)
    return p

  def stream(n):
    p = attention(n)
    p["mod_w"], p["mod_b"] = linear(n, e, 6 * e)
    p["wo"], p["bo"] = linear(n, e, e)
    p["mlp_w1"], p["mlp_b1"] = linear(n, e, 4 * e)
    p["mlp_w2"], p["mlp_b2"] = linear(n, 4 * e, e)
    return p

  n = cfg.num_single_layers
  single = attention(n)
  single["mod_w"], single["mod_b"] = linear(n, e, 3 * e)
  single["mlp_w"], single["mlp_b"] = linear(n, e, 4 * e)
  single["out_w"], single["out_b"] = linear(n, 5 * e, e)
  params = {
      "double": {"img": stream(cfg.num_layers), "txt": stream(cfg.num_layers)},
      "single": single,
      "timestep_embedder": mlp(_FREQUENCIES),
      "text_embedder": mlp(cfg.pooled_projection_dim),
  }
  if cfg.guidance_embeds:
    params["guidance_embedder"] = mlp(_FREQUENCIES)
  for name, shape in (("x_embedder", (c, e)), ("context_embedder", (cfg.joint_attention_dim, e)),
                      ("norm_out", (e, 2 * e)), ("proj_out", (e, c))):
    w, b = linear(*shape)
    params[name] = {"w": w, "b": b}
  return params


def flow_matching_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  """Mean squared error to the velocity, noise minus latents."""
  return jnp.mean((target.astype(jnp.float32) - pred.astype(jnp.float32)) ** 2)


def _dense(p: dict, x: jax.Array, w: str = "w", b: str = "b") -> jax.Array:
  return ops.linear(x, p[w]) + p[b]


def _embedder(p: dict, x: jax.Array) -> jax.Array:
  return _dense(p, jax.nn.silu(_dense(p, x, "w1", "b1")), "w2", "b2")


def _sinusoid(t: jax.Array, dtype) -> jax.Array:
  """The embedding of a timestep or guidance in [0, 1], scaled by 1000: cos, then sin."""
  half = _FREQUENCIES // 2
  freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / half)
  angles = (1000 * t.astype(jnp.float32))[:, None] * freqs
  return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1).astype(dtype)


def _rope_tables(cfg: FluxConfig, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
  """cos and sin [tokens, head_dim // 2] of the rotary angles of position ids [tokens, 3]:
  each position axis takes its share of `axes_dims_rope`, frequencies of base 10,000."""
  freqs = [1.0 / 10000.0 ** (np.arange(0, n, 2) / n) for n in cfg.axes_dims_rope]
  angles = jnp.concatenate(
      [ids[:, i, None].astype(jnp.float32) * f.astype(np.float32) for i, f in enumerate(freqs)],
      axis=-1)
  return jnp.cos(angles), jnp.sin(angles)


def _layer_norm(x: jax.Array) -> jax.Array:
  """LayerNorm without scale or bias, in fp32."""
  x32 = x.astype(jnp.float32)
  x32 = x32 - jnp.mean(x32, axis=-1, keepdims=True)
  return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6)).astype(x.dtype)


def _modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
  return _layer_norm(x) * (1 + scale[:, None]) + shift[:, None]


def _qkv(cfg: FluxConfig, p: dict, x: jax.Array):
  heads = x.shape[:2] + (cfg.num_attention_heads, cfg.attention_head_dim)
  q = ops.rms_norm(_dense(p, x, "wq", "bq").reshape(heads), p["q_norm"], 1e-6)
  k = ops.rms_norm(_dense(p, x, "wk", "bk").reshape(heads), p["k_norm"], 1e-6)
  return q, k, _dense(p, x, "wv", "bv").reshape(heads)


def _attention(q, k, v, cos, sin) -> jax.Array:
  q, k = ops.rope(q, cos, sin), ops.rope(k, cos, sin)
  out = ops.attention(q, k, v, causal=False)
  return out.reshape(out.shape[:2] + (-1,))


def _mlp(p: dict, x: jax.Array) -> jax.Array:
  h = jax.nn.gelu(_dense(p, x, "mlp_w1", "mlp_b1"), approximate=True)
  return _dense(p, h, "mlp_w2", "mlp_b2")


def _double_block(cfg: FluxConfig, img, txt, p: dict, cond, cos, sin):
  """One double-stream block: each stream with its own weights, attention over both."""
  mod = {s: jnp.split(_dense(p[s], cond, "mod_w", "mod_b"), 6, axis=-1) for s in ("img", "txt")}
  streams = {"img": img, "txt": txt}
  qkv = [_qkv(cfg, p[s], _modulate(streams[s], *mod[s][:2])) for s in ("txt", "img")]
  attn = _attention(*(jnp.concatenate(t, axis=1) for t in zip(*qkv)), cos, sin)
  attn = {"txt": attn[:, :txt.shape[1]], "img": attn[:, txt.shape[1]:]}
  for s in ("img", "txt"):
    _, _, gate, shift, scale, mlp_gate = mod[s]
    h = streams[s] + gate[:, None] * _dense(p[s], attn[s], "wo", "bo")
    streams[s] = h + mlp_gate[:, None] * _mlp(p[s], _modulate(h, shift, scale))
  return streams["img"], streams["txt"]


def _single_block(cfg: FluxConfig, h, p: dict, cond, cos, sin):
  """One single-stream block over text and image tokens: attention and MLP side by side."""
  shift, scale, gate = jnp.split(_dense(p, cond, "mod_w", "mod_b"), 3, axis=-1)
  x = _modulate(h, shift, scale)
  attn = _attention(*_qkv(cfg, p, x), cos, sin)
  mlp = jax.nn.gelu(_dense(p, x, "mlp_w", "mlp_b"), approximate=True)
  return h + gate[:, None] * _dense(p, jnp.concatenate([attn, mlp], axis=-1), "out_w", "out_b")


class Model:
  """FLUX.1 as one stage, for the trainer's `--parallel=dp` and `fsdp` layouts.

  Parameters are fp32; the model computes in `dtype`. `constrain`, when
  given, is applied to the conditioning and to the hidden states after the
  embedding and after every block.
  """

  logical_axes = staticmethod(logical_axes)

  def __init__(self, config: FluxConfig, *, dtype=jnp.bfloat16, remat: bool = True,
               constrain: Callable | None = None):
    self.config = config
    self.dtype = dtype
    self._remat = remat
    self._constrain = constrain or (lambda h: h)

  def init(self, seed: int = 0) -> tuple[dict]:
    """The parameters of the one stage, randomly initialized."""
    return (jax.jit(init_params, static_argnums=0)(self.config, jax.random.key(seed)),)

  def abstract(self, seed: int = 0) -> tuple[dict]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return (jax.eval_shape(lambda: init_params(self.config, jax.random.key(seed))),)

  def forward(self, params: dict, x: dict) -> jax.Array:
    """The predicted velocity of the image tokens [batch, image_tokens, in_channels].

    `x` holds a batch of `jax_titan.data.diffusion.flux`: `img` noised latents
    [batch, image_tokens, in_channels], `txt` T5 features [batch,
    text_tokens, joint_attention_dim], `pooled` CLIP features [batch,
    pooled_projection_dim], `timestep` in [0, 1] [batch], `img_ids`
    [image_tokens, 3] and `txt_ids` [text_tokens, 3], and with
    `guidance_embeds` the `guidance` [batch].
    """
    cfg, dtype, constrain = self.config, self.dtype, self._constrain
    p = jax.tree.map(lambda w: w.astype(dtype), params)
    temb = _embedder(p["timestep_embedder"], _sinusoid(x["timestep"], dtype))
    if cfg.guidance_embeds:
      temb = temb + _embedder(p["guidance_embedder"], _sinusoid(x["guidance"], dtype))
    temb = temb + _embedder(p["text_embedder"], x["pooled"].astype(dtype))
    cond = constrain(jax.nn.silu(temb))
    img = constrain(_dense(p["x_embedder"], x["img"].astype(dtype)))
    txt = constrain(_dense(p["context_embedder"], x["txt"].astype(dtype)))
    cos, sin = _rope_tables(cfg, jnp.concatenate([x["txt_ids"], x["img_ids"]]))

    def double(carry, layer):
      img, txt = _double_block(cfg, *carry, layer, cond, cos, sin)
      return (constrain(img), constrain(txt)), None

    def single(h, layer):
      return constrain(_single_block(cfg, h, layer, cond, cos, sin)), None

    if self._remat:
      double, single = (jax.checkpoint(f, policy=jax.checkpoint_policies.nothing_saveable)
                        for f in (double, single))
    (img, txt), _ = jax.lax.scan(double, (img, txt), p["double"])
    h, _ = jax.lax.scan(single, jnp.concatenate([txt, img], axis=1), p["single"])
    scale, shift = jnp.split(_dense(p["norm_out"], cond), 2, axis=-1)
    return _dense(p["proj_out"], _modulate(h[:, txt.shape[1]:], shift, scale))

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """The one stage function, `fn(params, x, target)`, which returns `flow_matching_loss`."""
    return [lambda p, x, t: flow_matching_loss(self.forward(p, x), t)]
