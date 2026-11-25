"""MaxText's decoder-only models, pinned at maxtext-v0.2.5, as one stage.

MaxText supplies the model configs, the layers and the parameter layout;
JAX-Titan supplies the loss, the mesh and the train step. Parameters are
plain dicts with MaxText's parameter names, so they map one to one onto
MaxText checkpoints. `WholeModel` calls MaxText's `Transformer` whole, with no
code per model family. `llama_config` and `llama_params` map a Llama model
onto `jax_titan.models.llama`, to set JAX-Titan's own Llama against MaxText's.
"""

from collections.abc import Callable, Sequence
import contextlib
import os

# MaxText reaches for Google Cloud clients unless this is set.
os.environ.setdefault("DECOUPLE_GCLOUD", "TRUE")

# pylint: disable=g-import-not-at-top,wrong-import-position
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from maxtext.common.common_types import AttentionType
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.inference.kvcache import KVTensor
from maxtext.layers import attention_op
from maxtext.layers import moe
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR

from jax_titan import parallel
from jax_titan import paths
from jax_titan.models import llama

PINNED_VERSION = "maxtext-v0.2.5"


def config(model_name: str, **overrides):
  """The MaxText config of `configs/models/<model_name>.yml`, with overrides.

  Defaults suit JAX-Titan: no dropout, XLA dot-product attention, no
  checkpointing or distributed setup by MaxText, the text tower alone, and
  for a mixture of experts the dense matmul over every expert, the fastest of
  MaxText's three forms for Qwen3-30B-A3B on MI350X (docs/adapters/maxtext.md). These
  and `overrides` win over the model's YAML.
  """
  kwargs = {
      "run_name": "jax_titan",
      "base_output_directory": str(paths.work_dir() / "maxtext"),
      "skip_jax_distributed_system": True,
      "enable_checkpointing": False,
      "enable_dropout": False,
      "attention": "dot_product",
      "megablox": False,
      "sparse_matmul": False,
      "use_multimodal": False,
      "use_audio": False,
      "override_model_config": True,
  }
  kwargs.update(overrides)
  base = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
  return pyconfig.initialize(["", base], model_name=model_name, **kwargs)


def mesh(cfg, devices: Sequence[jax.Device]) -> jax.sharding.Mesh:
  """A mesh with MaxText's axis names and every device on the data axis."""
  shape = [1] * len(cfg.mesh_axes)
  shape[cfg.mesh_axes.index("data")] = len(devices)
  return jax.sharding.Mesh(np.asarray(devices).reshape(shape), cfg.mesh_axes)


def _interleaved_rope(cfg) -> bool:
  """Whether MaxText rotates adjacent pairs, as its Llama 3.1 RoPE does."""
  return cfg.model_name.startswith("llama3.1") or cfg.rope_type.lower().startswith("llama3.1")


def llama_config(cfg) -> llama.LlamaConfig:
  """The same model as a `jax_titan.models.llama` config."""
  interleaved = _interleaved_rope(cfg)
  if cfg.rope_min_timescale != 1 or not interleaved and (
      cfg.rope_type.lower() != "default" or cfg.rope_linear_scaling_factor != 1.0):
    raise NotImplementedError(
        f"rope_type={cfg.rope_type}, rope_min_timescale={cfg.rope_min_timescale}, "
        f"rope_linear_scaling_factor={cfg.rope_linear_scaling_factor}"
    )
  return llama.LlamaConfig(
      vocab_size=cfg.vocab_size,
      emb_dim=cfg.emb_dim,
      num_layers=cfg.num_decoder_layers,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      mlp_dim=cfg.mlp_dim,
      norm_eps=cfg.normalization_layer_epsilon,
      rope_theta=float(cfg.rope_max_timescale),
      rope_scaling=llama.Llama31RopeScaling() if interleaved and cfg.rope_use_scale else None,
  )


def llama_params(cfg, params: dict) -> dict:
  """A Llama model's parameters, as `WholeModel` holds them, in the layout of
  `jax_titan.models.llama` as one stage.

  Puts the layer axis first, takes MaxText's 1/sqrt(head_dim) attention scale
  back out of the query kernel, and, for interleaved RoPE, reorders the query
  and key features of each head into split halves.
  """
  def first(x):
    return jnp.moveaxis(x, cfg.param_scan_axis, 0)

  def rope(x):
    return jnp.concatenate([x[..., ::2], x[..., 1::2]], axis=-1) if _interleaved_rope(cfg) else x

  decoder = params["decoder"]
  attn, mlp = decoder["layers"]["self_attention"], decoder["layers"]["mlp"]
  q = rope(first(attn["query"]["kernel"]))
  q = (q.astype(jnp.float32) * np.sqrt(cfg.head_dim).astype(np.float32)).astype(q.dtype)
  k = rope(first(attn["key"]["kernel"]))
  n, e = q.shape[0], q.shape[1]
  return {
      "embed": params["token_embedder"]["embedding"],
      "layers": {
          "attn_norm": first(decoder["layers"]["pre_self_attention_layer_norm"]["scale"]),
          "wq": q.reshape(n, e, -1),
          "wk": k.reshape(n, e, -1),
          "wv": first(attn["value"]["kernel"]).reshape(n, e, -1),
          "wo": first(attn["out"]["kernel"]).reshape(n, -1, e),
          "mlp_norm": first(decoder["layers"]["post_self_attention_layer_norm"]["scale"]),
          "w_gate": first(mlp["wi_0"]["kernel"]),
          "w_up": first(mlp["wi_1"]["kernel"]),
          "w_down": first(mlp["wo"]["kernel"]),
      },
      "final_norm": decoder["decoder_norm"]["scale"],
      "lm_head": decoder["logits_dense"]["kernel"],
  }


def _pure_dict(tree):
  """Nested dicts of arrays from a tree that may hold NNX States and Variables."""
  if isinstance(tree, nnx.Variable):
    return tree.get_value()
  if hasattr(tree, "items"):
    return {k: _pure_dict(v) for k, v in tree.items()}
  return tree


def _unset(value) -> bool:
  return value is None or (isinstance(value, (bool, int)) and not value)


@contextlib.contextmanager
def _attention_through(attention: Callable):
  """Routes MaxText's causal training attention to `attention(q, k, v, causal=, scale=)`.

  Everything else (decoding, other masks, sinks, soft caps, quantized KV)
  stays in MaxText's attention op. MaxText folds 1/sqrt(head_dim) into the
  query projection, so the call has no scale; the stages' segment ids are all
  ones, so the mask is plain causal.
  """
  original = attention_op.AttentionOp.apply_attention

  def apply_attention(self, query, key, value, decoder_segment_ids, segment_positions, lengths,
                      model_mode, *args, **kwargs):
    extras = [v for k, v in kwargs.items() if k not in ("qk_product_einsum", "wv_product_einsum")]
    if (model_mode != MODEL_MODE_TRAIN or not all(map(_unset, [*args, *extras]))
        or self.attention_type != AttentionType.GLOBAL or self.attn_logits_soft_cap
        or isinstance(key, KVTensor) or isinstance(value, KVTensor)):
      return original(self, query, key, value, decoder_segment_ids, segment_positions, lengths,
                      model_mode, *args, **kwargs)
    return attention(query, key, value, causal=True, scale=1.0), None, None

  attention_op.AttentionOp.apply_attention = apply_attention
  try:
    yield
  finally:
    attention_op.AttentionOp.apply_attention = original


def _logical_axes(variable) -> tuple | None:
  """The logical axis names MaxText put on an NNX variable, the axis of its
  scanned stack named `layers`, which `jax_titan.parallel` never shards; None
  for a variable MaxText gave no names."""
  meta = variable.get_metadata()
  names, stack = meta.get("out_sharding"), meta.get(nnx.PARTITION_NAME)
  if names is None:
    return None
  return tuple("layers" if stack is not None and n == stack else n for n in names)


def _beyond_the_step(cfg) -> dict:
  """The settings of `cfg` under which MaxText's training is more than a
  gradient step of the cross-entropy and the MoE load-balance term."""
  settings = {
      "mtp_num_layers": cfg.mtp_num_layers,
      "z_loss_multiplier": cfg.z_loss_multiplier,
      "use_indexer": cfg.use_indexer,
      "use_qk_clip": cfg.use_qk_clip,
      "quantization": cfg.quantization,
      "num_vocab_tiling": cfg.num_vocab_tiling > 1 and cfg.num_vocab_tiling,
      "gradient_accumulation_steps": (
          cfg.gradient_accumulation_steps > 1 and cfg.gradient_accumulation_steps),
      "routed_bias_update_rate": cfg.routed_bias and cfg.routed_bias_update_rate,
  }
  return {name: value for name, value in settings.items() if value}


class WholeModel:
  """A MaxText model as one stage, for `--parallel=dp` and `fsdp`.

  MaxText's `Transformer` is called whole, so nothing here depends on the
  model family. The parameters are its `nnx.Param` leaves under MaxText's
  names; its other NNX state stays as MaxText initializes it. The loss is
  MaxText's training loss of a text batch, the mean next-token cross-entropy
  plus the load-balance term a mixture of experts sows. A config under which
  MaxText's training holds more is refused: other loss terms, updates outside
  the gradient, quantization, gradient accumulation. The parameters are
  initialized and restored where the trainer keeps them, sharded with `fsdp`
  and replicated without. With `attention`, such as `jax_titan.ops.attention`,
  causal training attention runs there instead of in MaxText's attention op.
  """

  def __init__(self, cfg, mesh: jax.sharding.Mesh, *, attention: Callable | None = None,
               fsdp: bool = False):
    beyond = _beyond_the_step(cfg)
    if beyond:
      raise NotImplementedError(
          f"{beyond}: the whole model takes a gradient step of the cross-entropy of the full "
          "logits and the MoE load-balance term, and MaxText's training with these holds more"
      )
    self.cfg = cfg
    self.mesh = mesh
    self._attention = attention
    self._fsdp = fsdp
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      graph, params, rest = nnx.split(nnx.eval_shape(lambda: self._transformer(0)), nnx.Param, ...)
    self._graph = graph
    self._axes = {tuple(map(str, path)): _logical_axes(v) for path, v in nnx.to_flat_state(params)}
    self._other_weights = sorted(
        {type(v).__name__ for _, v in nnx.to_flat_state(rest) if not isinstance(v, nnx.RngState)})
    self._rest = jax.jit(lambda: self._init_full(0)[1])()

  def _transformer(self, seed: int):
    return models.Transformer(
        self.cfg, self.mesh, quant=None, rngs=nnx.Rngs(params=seed, dropout=seed)
    )

  def _init_full(self, seed: int):
    with nn.logical_axis_rules(self.cfg.logical_axis_rules):
      _, params, rest = nnx.split(self._transformer(seed), nnx.Param, ...)
    return params.to_pure_dict(), rest.to_pure_dict()

  def _shardings(self) -> dict:
    (shapes,) = self.abstract()
    specs = parallel.tree_specs(shapes, self.logical_axes, self.mesh.shape, fsdp=self._fsdp)
    return jax.tree.map(lambda s: jax.sharding.NamedSharding(self.mesh, s), specs)

  def init(self, seed: int = 0) -> tuple[dict]:
    """The parameters, initialized as MaxText initializes `Transformer`."""
    init = jax.jit(lambda: self._init_full(seed)[0], out_shardings=self._shardings())
    return (init(),)

  def abstract(self, seed: int = 0) -> tuple[dict]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return (jax.eval_shape(lambda: self._init_full(seed)[0]),)

  def load(self, path: str) -> tuple[dict]:
    """The parameters from a MaxText parameter checkpoint, read onto the
    trainer's shardings.

    `path` is what MaxText's `load_parameters_path` takes: the output of its
    `checkpoint_conversion/to_maxtext.py` (from Hugging Face weights) or of
    `checkpointing.save_params_to_path`. MaxText's loader restores the weights
    in the config's `weight_dtype`.
    """
    from maxtext.common import checkpointing  # pylint: disable=g-import-not-at-top

    if self._other_weights:
      raise NotImplementedError(
          f"{self.cfg.model_name} keeps weights outside its parameters "
          f"({', '.join(self._other_weights)}), which a parameter checkpoint would leave as "
          "initialized"
      )
    with nn.logical_axis_rules(self.cfg.logical_axis_rules):
      _, abstract, _ = nnx.split(nnx.eval_shape(lambda: self._transformer(0)), nnx.Param, ...)
    nnx.replace_by_pure_dict(abstract, jax.tree.map(
        lambda x, sharding: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
        nnx.to_pure_dict(abstract), self._shardings()))
    restored = checkpointing.load_params_from_path(
        path, abstract, self.cfg.checkpoint_storage_concurrent_gb)
    return (_pure_dict(restored),)

  def _loss(self, params: dict, tokens: jax.Array, targets: jax.Array) -> jax.Array:
    cfg = self.cfg
    model = nnx.merge(self._graph, params, self._rest)
    batch, length = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    segments = jnp.ones((batch, length), jnp.int32)
    routed = (_attention_through(self._attention) if self._attention
              else contextlib.nullcontext())
    with nn.logical_axis_rules(cfg.logical_axis_rules), routed:
      logits = model(
          decoder_input_tokens=tokens.astype(jnp.int32), decoder_positions=positions,
          decoder_segment_ids=segments, enable_dropout=False,
          decoder_target_tokens=targets, decoder_target_mask=segments,
      )
    loss = llama.cross_entropy(logits, targets)
    sown = nnx.pop(model, nnx.Intermediate).to_pure_dict()
    balance = maxtext_utils.collect_intermediates_by_suffix(sown, "moe_lb_loss")
    if balance:
      balance = jnp.concatenate(balance)
      loss += jnp.sum(balance) if moe.uses_megatron_seq_aux_loss(cfg) else jnp.mean(balance)
    return loss

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """The one stage function, `fn(params, tokens, targets)`, which returns the loss."""
    return [self._loss]

  def logical_axes(self, path, leaf) -> tuple | None:
    """MaxText's logical axes of a parameter, or of an optimizer-state leaf
    mirroring one, for `jax_titan.parallel.tree_specs`."""
    keys = tuple(str(k.key) for k in path if isinstance(k, jax.tree_util.DictKey))
    axes = self._axes.get(keys)
    # MaxText's Qwen3-Next names fewer axes than some of its parameters have.
    return axes if axes is not None and len(axes) == len(leaf.shape) else None

  @staticmethod
  def decays(path, leaf) -> bool:
    """Whether a parameter takes weight decay: every one but a norm's weight,
    which MaxText names `scale`, and what its modules `q_norm` and `k_norm`
    hold, a bias among it."""
    del leaf
    return not any(getattr(k, "key", None) in ("scale", "q_norm", "k_norm") for k in path)

  def flops_per_token(self) -> float:
    """Training FLOPs per token as MaxText counts them, for sequences of the
    config's `max_target_length`."""
    tflops = maxtext_utils.calculate_tflops_training_per_device(self.cfg, log=False)[0]
    return tflops * 1e12 / (self.cfg.per_device_batch_size * self.cfg.max_target_length)
