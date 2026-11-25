"""MaxDiffusion's denoisers, pinned at `821afc36`, as one stage: FLUX.1 and
Wan 2.1.

MaxDiffusion supplies the model, its layers and its parameter layout;
JAX-Titan supplies the objective (`jax_titan.data.diffusion`), the mesh and
the train step. `AdapterModel` calls a denoiser whole, MaxDiffusion's own
`__call__`, with a few lines per family (`whole_flux`, `whole_wan`).

MaxDiffusion needs the `maxdiffusion-*` patches in `third_party/patches`.
"""

from collections.abc import Callable, Sequence
import functools

from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from maxdiffusion.models.attention_flax import AttentionOp
from maxdiffusion.models.embeddings_flax import CombinedTimestepGuidanceTextProjEmbeddings
from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel

from jax_titan import parallel
from jax_titan.models import losses

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


class AdapterModel:
  """A MaxDiffusion denoiser called whole, as one stage for `--parallel=dp` and `fsdp`.

  `init_full(seed)` gives its parameters as MaxDiffusion initializes them and
  `apply(params, x)` its prediction for a batch of `jax_titan.data.diffusion`;
  `whole_flux` and `whole_wan` make the two. The loss is MaxDiffusion's, the
  mean squared error of the prediction to the batch's target.
  FSDP shards a parameter along its first dimension that the data axis
  divides, and never along the block axis of the stacks in `stacked`. The
  parameters are initialized where the trainer keeps them on `mesh`, sharded
  with `fsdp` and replicated without.
  """

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
    return [lambda p, x, t: losses.mean_squared_error(self._apply(p, x), t)]

  def logical_axes(self, path, leaf) -> tuple:
    axes = [None] * len(leaf.shape)
    if axes and any(getattr(k, "key", None) in self._stacked for k in path):
      axes[0] = "layers"
    return tuple(axes)

  @staticmethod
  def decays(path, leaf) -> bool:
    """Whether a parameter takes weight decay: every one but a norm's weight,
    which MaxDiffusion names `scale`."""
    del leaf
    return not any(getattr(k, "key", None) == "scale" for k in path)


def _flux_init(f: FluxTransformer2DModel, seed: int) -> dict:
  """The parameters of `f`, as MaxDiffusion initializes them."""
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


def flux_flops_per_sample(f: FluxTransformer2DModel, image_tokens: int,
                          text_tokens: int) -> float:
  """Training FLOPs per sample: 6 per weight and token, plus attention.

  Every block applies its projections and MLP to each token once, in its
  stream or in the joined sequence, and attends over the joined sequence.
  """
  d = f.num_attention_heads * f.attention_head_dim
  tokens = image_tokens + text_tokens
  blocks = f.num_layers + f.num_single_layers
  per_token = 4 * d * d + 2 * d * int(d * f.mlp_ratio)
  weights = per_token * tokens * blocks + d * (
      2 * f.in_channels * image_tokens + f.joint_attention_dim * text_tokens)
  attention = 2 * tokens * tokens * d * blocks
  return 6.0 * (weights + attention)


def whole_flux(flux: FluxTransformer2DModel, mesh: jax.sharding.Mesh, *, fsdp: bool = False,
               attention: Callable | None = None) -> AdapterModel:
  """FLUX.1 through `FluxTransformer2DModel.__call__`. With `attention`, such
  as `jax_titan.ops.attention`, the blocks' attention runs there instead of in
  MaxDiffusion's attention op."""
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

  return AdapterModel(
      functools.partial(_flux_init, flux), apply, mesh, fsdp=fsdp, stacked=_STACKS)


def whole_wan(mesh: jax.sharding.Mesh, *, fsdp: bool = False, **kwargs) -> AdapterModel:
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

  return AdapterModel(
      lambda seed: init_full(seed)[0], apply, mesh, fsdp=fsdp, stacked=("blocks",))
