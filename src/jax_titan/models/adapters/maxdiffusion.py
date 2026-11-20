"""MaxDiffusion's denoisers, pinned at `821afc36`: FLUX.1 and Wan 2.1 as one
stage, and FLUX.1 as pipeline stages.

MaxDiffusion supplies the model, its layers and its parameter layout;
JAX-Titan supplies the objective (`jax_titan.data.diffusion`), the stages,
meshes and train step. `WholeModel` calls a denoiser whole, MaxDiffusion's own
`__call__`, with a few lines per family (`whole_flux`, `whole_wan`). `Model`
cuts FLUX.1 into pipeline stages. FLUX.1 runs 19
double-stream blocks, which keep image and text tokens in separate streams,
then 38 single-stream blocks over the joined sequence. A stage holds a
contiguous run of these blocks, counting the double-stream blocks first;
stage 0 also holds the input embeddings and the last stage the output layers.
Stage parameters keep MaxDiffusion's names, with each stage's blocks sliced
from the stacked block weights.

Every stage boundary carries the image tokens, the text tokens, the
conditioning vector and the rotary embeddings, and the last stage returns
MaxDiffusion's flow-matching loss. Stages reduce gradients per microbatch.

MaxDiffusion needs the `maxdiffusion-*` patches in `third_party/patches`.
"""

from collections.abc import Callable, Sequence
import functools

from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
from maxdiffusion.models.attention_flax import AttentionOp
from maxdiffusion.models.embeddings_flax import CombinedTimestepGuidanceTextProjEmbeddings
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel

from jax_titan import parallel

PINNED_COMMIT = "821afc364ac1889c1039112843f4f7a17c1bef7b"
FLAVORS = {
    "flux-dev": {"guidance_embeds": True},
    "flux-schnell": {"guidance_embeds": False},
}
# Wan 2.1 text to video: the sizes of Wan-AI/Wan2.1-T2V-1.3B-Diffusers.
WAN = {
    "wan2.1-t2v-1.3b": {
        "num_attention_heads": 12, "attention_head_dim": 128, "ffn_dim": 8960, "num_layers": 30,
        "in_channels": 16, "out_channels": 16, "text_dim": 4096, "patch_size": (1, 2, 2),
    },
}
_FIRST = ("img_in", "txt_in", "time_text_embed")
_LAST = ("norm_out", "proj_out")
_STACKS = ("scanned_double_blocks", "scanned_single_blocks")
# MaxDiffusion's batch axes on JAX-Titan's data axis; its other logical axes replicated.
_AXIS_RULES = (("batch", "data"), ("activation_batch", "data"), ("conv_batch", "data"),
               ("activation_kv_batch", "data"))


def transformer(flavor: str, **overrides) -> FluxTransformer2DModel:
  """MaxDiffusion's FLUX.1 transformer; sizes default to the released model's."""
  kwargs = {"attention_kernel": "dot_product", **FLAVORS[flavor], **overrides}
  if kwargs["guidance_embeds"] and (
      "project_timesteps" not in CombinedTimestepGuidanceTextProjEmbeddings.__dataclass_fields__):
    raise RuntimeError(
        "FLUX.1 with guidance needs MaxDiffusion with "
        "third_party/patches/maxdiffusion-flux1-guidance-embedding.patch"
    )
  return FluxTransformer2DModel(**kwargs)


def flow_matching_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  """MaxDiffusion's FLUX.1 loss: mean squared error to noise minus latents."""
  return jnp.mean((target.astype(jnp.float32) - pred.astype(jnp.float32)) ** 2)


def position_ids(image_side: int, text_tokens: int) -> tuple[np.ndarray, np.ndarray]:
  """`img_ids` [image_side**2, 3] and `txt_ids` [text_tokens, 3], as MaxDiffusion builds them.

  Image token (row, column) has ids (0, row, column); text tokens have zeros.
  """
  img_ids = np.zeros((image_side, image_side, 3), np.float32)
  img_ids[..., 1] = np.arange(image_side)[:, None]
  img_ids[..., 2] = np.arange(image_side)[None, :]
  return img_ids.reshape(-1, 3), np.zeros((text_tokens, 3), np.float32)


def _embed(m, inputs: dict):
  """`FluxTransformer2DModel.__call__` up to the first block."""
  img = m.img_in(inputs["img"])
  timestep = m.timestep_embedding(inputs["timestep"], 256)
  timestep = nn.with_logical_constraint(timestep, ("activation_batch", None))
  if m.guidance_embeds:
    guidance = m.timestep_embedding(inputs["guidance"], 256)
    temb = m.time_text_embed(timestep, guidance, inputs["pooled"])
  else:
    temb = m.time_text_embed(timestep, inputs["pooled"])
  temb = nn.with_logical_constraint(temb, ("activation_batch", None))
  txt = m.txt_in(inputs["txt"])
  txt_ids, img_ids = inputs["txt_ids"], inputs["img_ids"]
  if txt_ids.ndim == 3:
    txt_ids = txt_ids[0]
  if img_ids.ndim == 3:
    img_ids = img_ids[0]
  ids = jnp.concatenate((txt_ids, img_ids), axis=0)
  ids = nn.with_logical_constraint(ids, ("activation_batch", None))
  rope = nn.with_logical_constraint(m.pe_embedder(ids), (None, None))
  return img, txt, temb, rope


def _stage(m, x, *, first: bool, last: bool):
  """One stage on `m`, a transformer with only this stage's blocks."""
  img, txt, temb, rope = _embed(m, x) if first else x
  if m.num_layers:
    (img, txt, _, _), _ = m.scanned_double_blocks((img, txt, temb, rope), None)
  if m.num_single_layers:
    joined = jnp.concatenate([txt, img], axis=1)
    joined = nn.with_logical_constraint(
        joined, ("activation_batch", "activation_length", "activation_embed")
    )
    (joined, _, _), _ = m.scanned_single_blocks((joined, temb, rope), None)
    txt, img = joined[:, :txt.shape[1]], joined[:, txt.shape[1]:]
  if not last:
    return img, txt, temb, rope
  return m.proj_out(m.norm_out(img, temb))


def _attention_interceptor(attention: Callable):
  """Routes MaxDiffusion's unmasked attention to `attention(q, k, v, causal=, scale=)`."""

  def intercept(next_fun, args, kwargs, context):
    if (not isinstance(context.module, AttentionOp) or context.method_name != "apply_attention"
        or kwargs.get("attention_mask") is not None or len(args) > 3):
      return next_fun(*args, **kwargs)
    m = context.module
    q, k, v = (x.reshape(x.shape[0], x.shape[1], m.heads, m.dim_head) for x in args[:3])
    out = attention(q, k, v, causal=m.is_causal, scale=m.scale)
    return out.reshape(out.shape[0], out.shape[1], m.heads * m.dim_head)

  return intercept


class WholeModel:
  """A MaxDiffusion denoiser called whole, as one stage for `--parallel=dp` and `fsdp`.

  `init_full(seed)` gives its parameters as MaxDiffusion initializes them and
  `apply(params, x)` its prediction for a batch of `jax_titan.data.diffusion`;
  `whole_flux` and `whole_wan` make the two. The loss is `flow_matching_loss`.
  FSDP shards a parameter along its first dimension that the data axis
  divides, and never along the block axis of the stacks in `stacked`. The
  parameters are initialized where the trainer keeps them on `mesh`, sharded
  with `fsdp` and replicated without.
  """

  num_stages = 1

  def __init__(self, init_full: Callable, apply: Callable, mesh: jax.sharding.Mesh, *,
               fsdp: bool = False, stacked: Sequence[str] = ()):
    self._init_full, self._apply, self._stacked = init_full, apply, tuple(stacked)
    self._mesh, self._fsdp = mesh, fsdp

  def init(self, seed: int = 0) -> tuple[dict]:
    specs = parallel.tree_specs(
        *self.abstract(), self.logical_axes, self._mesh.shape, fsdp=self._fsdp)
    shardings = jax.tree.map(lambda s: jax.sharding.NamedSharding(self._mesh, s), specs)
    return (jax.jit(lambda: self._init_full(seed), out_shardings=shardings)(),)

  def abstract(self, seed: int = 0) -> tuple[dict]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return (jax.eval_shape(lambda: self._init_full(seed)),)

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """The one stage function, `fn(params, x, target)`, which returns the loss."""
    return [lambda p, x, t: flow_matching_loss(self._apply(p, x), t)]

  def logical_axes(self, path, leaf) -> tuple:
    axes = [None] * len(leaf.shape)
    if axes and any(getattr(k, "key", None) in self._stacked for k in path):
      axes[0] = "layers"
    return tuple(axes)


def whole_flux(flux: FluxTransformer2DModel, mesh: jax.sharding.Mesh, *, fsdp: bool = False,
               attention: Callable | None = None) -> WholeModel:
  """FLUX.1 through `FluxTransformer2DModel.__call__`; `attention` as for `Model`."""
  interceptor = None if attention is None else _attention_interceptor(attention)

  def apply(params, x):
    call = functools.partial(
        flux.apply, {"params": params}, hidden_states=x["img"], encoder_hidden_states=x["txt"],
        pooled_projections=x["pooled"], timestep=x["timestep"], img_ids=x["img_ids"],
        txt_ids=x["txt_ids"], guidance=x.get("guidance"), return_dict=False)
    if interceptor is None:
      return call()[0]
    with nn.intercept_methods(interceptor):
      return call()[0]

  return WholeModel(Model(flux, num_stages=1)._init_full, apply, mesh, fsdp=fsdp,  # pylint: disable=protected-access
                    stacked=_STACKS)


def whole_wan(mesh: jax.sharding.Mesh, *, fsdp: bool = False, **kwargs) -> WholeModel:
  """Wan 2.1 text to video through `WanModel.__call__`, a `WanModel(**kwargs)`
  such as one of `WAN`.

  Its blocks constrain activation shardings by MaxDiffusion's logical axis
  names, so the call runs under `mesh`, whose axes are JAX-Titan's.
  """
  from maxdiffusion.models.wan.transformers.transformer_wan import WanModel  # pylint: disable=g-import-not-at-top

  def init_full(seed):
    _, params, rest = nnx.split(WanModel(rngs=nnx.Rngs(seed), **kwargs), nnx.Param, ...)
    return params.to_pure_dict(), rest.to_pure_dict()

  graph, _, _ = nnx.split(
      nnx.eval_shape(lambda: WanModel(rngs=nnx.Rngs(0), **kwargs)), nnx.Param, ...)
  rest = jax.jit(lambda: init_full(0)[1])()

  def apply(params, x):
    model = nnx.merge(graph, params, rest)
    with mesh, nn_partitioning.axis_rules(_AXIS_RULES):
      return model(hidden_states=x["latents"], timestep=x["timestep"],
                   encoder_hidden_states=x["txt"], deterministic=True)

  return WholeModel(lambda seed: init_full(seed)[0], apply, mesh, fsdp=fsdp)


class Model:
  """MaxDiffusion's FLUX.1 transformer split into pipeline stages.

  `blocks` gives the blocks per stage, double-stream blocks first; the
  default splits them as evenly as possible, earlier stages taking the
  extra blocks (FLUX.1's 57 into 29 and 28). With `attention`, such as
  `jax_titan.ops.attention`, the blocks' attention runs there instead of in
  MaxDiffusion's attention op.
  """

  def __init__(self, flux: FluxTransformer2DModel, *, num_stages: int,
               blocks: Sequence[int] | None = None, attention: Callable | None = None):
    self._interceptor = None if attention is None else _attention_interceptor(attention)
    total = flux.num_layers + flux.num_single_layers
    if blocks is None:
      blocks = tuple(total // num_stages + (i < total % num_stages) for i in range(num_stages))
    if len(blocks) != num_stages or sum(blocks) != total or min(blocks) < 1:
      raise ValueError(f"stage blocks {tuple(blocks)} do not split {total} blocks")
    self.flux = flux
    self.num_stages = num_stages
    self.blocks = tuple(blocks)
    self._ranges = []
    start, d = 0, flux.num_layers
    for n in blocks:
      end = start + n
      self._ranges.append(((min(start, d), min(end, d)), (max(start - d, 0), max(end - d, 0))))
      start = end
    self._stages = [
        flux.clone(num_layers=dd[1] - dd[0], num_single_layers=ss[1] - ss[0])
        for dd, ss in self._ranges
    ]

  def _init_full(self, seed: int) -> dict:
    f = self.flux
    image_tokens, text_tokens = 4, 2
    params = f.init(
        jax.random.key(seed),
        hidden_states=jnp.zeros((1, image_tokens, f.in_channels), f.dtype),
        encoder_hidden_states=jnp.zeros((1, text_tokens, f.joint_attention_dim), f.dtype),
        pooled_projections=jnp.zeros((1, f.pooled_projection_dim), f.dtype),
        timestep=jnp.zeros((1,), f.dtype),
        img_ids=jnp.zeros((image_tokens, 3), f.dtype),
        txt_ids=jnp.zeros((text_tokens, 3), f.dtype),
        guidance=jnp.zeros((1,), f.dtype),
    )["params"]
    return nn.meta.unbox(params)

  def split(self, params: dict) -> tuple[dict, ...]:
    """Stage parameter dicts from the parameters of `FluxTransformer2DModel`."""
    stages = []
    for i, ((d0, d1), (s0, s1)) in enumerate(self._ranges):
      p = {}
      if i == 0:
        p.update({k: params[k] for k in _FIRST})
      if d1 > d0:
        p["scanned_double_blocks"] = jax.tree.map(lambda x, a=d0, b=d1: x[a:b],
                                                  params["scanned_double_blocks"])
      if s1 > s0:
        p["scanned_single_blocks"] = jax.tree.map(lambda x, a=s0, b=s1: x[a:b],
                                                  params["scanned_single_blocks"])
      if i == self.num_stages - 1:
        p.update({k: params[k] for k in _LAST})
      stages.append(p)
    return tuple(stages)

  def init(self, seed: int = 0) -> tuple[dict, ...]:
    """Stage parameters, initialized as MaxDiffusion initializes FLUX.1."""
    return self.split(jax.jit(lambda: self._init_full(seed))())

  def abstract(self, seed: int = 0) -> tuple[dict, ...]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return jax.eval_shape(lambda: self.split(self._init_full(seed)))

  def forward(self, stage: int, params: dict, x):
    """Runs one stage.

    Stage 0 takes a dict of `img` latents [batch, image_tokens, in_channels],
    `txt` T5 embeddings [batch, text_tokens, joint_attention_dim], `pooled`
    CLIP embeddings [batch, pooled_projection_dim], `timestep` and, for
    FLUX.1-dev, `guidance` [batch], and `img_ids` and `txt_ids`
    [tokens, 3]. The last stage returns the predicted velocity
    [batch, image_tokens, in_channels].
    """
    method = functools.partial(_stage, first=stage == 0, last=stage == self.num_stages - 1)
    if self._interceptor is None:
      return self._stages[stage].apply({"params": params}, x, method=method)
    with nn.intercept_methods(self._interceptor):
      return self._stages[stage].apply({"params": params}, x, method=method)

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """Stage functions for `jax_titan.mpmd.make_train_step`.

    The last stage also takes the velocity target and returns
    `flow_matching_loss`.
    """
    fns = [functools.partial(self.forward, i) for i in range(self.num_stages - 1)]
    last = self.num_stages - 1
    fns.append(lambda p, x, t: flow_matching_loss(self.forward(last, p, x), t))
    return fns

  def flops_per_sample(self, image_tokens: int, text_tokens: int) -> float:
    """Training FLOPs per sample: 6 per weight and token, plus attention.

    Every block applies its projections and MLP to each token once, in its
    stream or in the joined sequence, and attends over the joined sequence.
    """
    f = self.flux
    d = f.num_attention_heads * f.attention_head_dim
    tokens = image_tokens + text_tokens
    blocks = f.num_layers + f.num_single_layers
    per_token = 4 * d * d + 2 * d * int(d * f.mlp_ratio)
    weights = per_token * tokens * blocks + d * (
        2 * f.in_channels * image_tokens + f.joint_attention_dim * text_tokens)
    attention = 2 * tokens * tokens * d * blocks
    return 6.0 * (weights + attention)
