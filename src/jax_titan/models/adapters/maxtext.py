"""MaxText's decoder-only models, pinned at maxtext-v0.2.5: as one stage, and
Llama 2 and 3 as pipeline stages.

MaxText supplies the model configs, the layers and the parameter layout;
JAX-Titan supplies the stages, meshes and train step. Parameters are plain
dicts with MaxText's parameter names, so they map one to one onto MaxText
checkpoints.

`WholeModel` calls MaxText's `Transformer` whole, with no code per model
family. `Model` cuts a model into pipeline stages: stage 0 holds MaxText's
token embedder, the last stage its final norm and output head, and every stage
a contiguous slice of the scanned decoder layers, whose stacked weights keep
their layer axis where MaxText puts it (`param_scan_axis`). It takes the
models of MaxText's `llama2` decoder block, Llama 2 and 3, built as one
scanned stack of identical layers with an untied output head (`has_split`).
MaxText's layers constrain activation shardings, so stages reduce gradients
per microbatch. `llama_config` and `llama_params` map such a model onto
`jax_titan.models.llama`.
"""

from collections.abc import Callable, Sequence
import contextlib
import functools
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
from maxtext.common.common_types import DecoderBlockType
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.inference.kvcache import KVTensor
from maxtext.layers import attention_op
from maxtext.layers import moe
from maxtext.layers import nnx_scan
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.utils.sharding import create_sharding

from jax_titan import parallel
from jax_titan import paths
from jax_titan.models import llama

PINNED_VERSION = "maxtext-v0.2.5"
_LOGITS_AXES = ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
_SCANNED_GENERIC = (DecoderBlockType.LLAMA2,)


def config(model_name: str, **overrides):
  """The MaxText config of `configs/models/<model_name>.yml`, with overrides.

  Defaults suit JAX-Titan: no dropout, XLA dot-product attention, no
  checkpointing or distributed setup by MaxText, the text tower alone, and
  for a mixture of experts the dense matmul over every expert, the fastest of
  MaxText's three forms for Qwen3-30B-A3B on MI350X (docs/pipelines.md). These
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


def has_split(cfg) -> bool:
  """Whether `Model` cuts this model into pipeline stages: MaxText's `llama2`
  decoder block as one scanned stack, with an untied output head."""
  return (cfg.decoder_block in _SCANNED_GENERIC and bool(cfg.scan_layers)
          and not cfg.logits_via_embedding)


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


def llama_params(cfg, stage_params: Sequence[dict]) -> tuple[dict, ...]:
  """Stage parameters in the `jax_titan.models.llama` layout.

  Puts the layer axis first, takes MaxText's 1/sqrt(head_dim) attention scale
  back out of the query kernel, and, for interleaved RoPE, reorders the query
  and key features of each head into split halves.
  """
  def first(x):
    return jnp.moveaxis(x, cfg.param_scan_axis, 0)

  def rope(x):
    return jnp.concatenate([x[..., ::2], x[..., 1::2]], axis=-1) if _interleaved_rope(cfg) else x

  out = []
  for p in stage_params:
    attn, mlp = p["layers"]["self_attention"], p["layers"]["mlp"]
    q = rope(first(attn["query"]["kernel"]))
    q = (q.astype(jnp.float32) * np.sqrt(cfg.head_dim).astype(np.float32)).astype(q.dtype)
    k = rope(first(attn["key"]["kernel"]))
    v = first(attn["value"]["kernel"])
    o = first(attn["out"]["kernel"])
    n, e = q.shape[0], q.shape[1]
    stage = {"layers": {
        "attn_norm": first(p["layers"]["pre_self_attention_layer_norm"]["scale"]),
        "wq": q.reshape(n, e, -1),
        "wk": k.reshape(n, e, -1),
        "wv": v.reshape(n, e, -1),
        "wo": o.reshape(n, -1, e),
        "mlp_norm": first(p["layers"]["post_self_attention_layer_norm"]["scale"]),
        "w_gate": first(mlp["wi_0"]["kernel"]),
        "w_up": first(mlp["wi_1"]["kernel"]),
        "w_down": first(mlp["wo"]["kernel"]),
    }}
    if "token_embedder" in p:
      stage["embed"] = p["token_embedder"]["embedding"]
    if "decoder_norm" in p:
      stage["final_norm"] = p["decoder_norm"]["scale"]
      stage["lm_head"] = p["logits_dense"]["kernel"]
    out.append(stage)
  return tuple(out)


def _pure_dict(tree):
  """Nested dicts of arrays from a tree that may hold NNX States and Variables."""
  if isinstance(tree, nnx.Variable):
    return tree.get_value()
  if hasattr(tree, "items"):
    return {k: _pure_dict(v) for k, v in tree.items()}
  return tree


def _slice_layers(x, start: int, size: int, scan_axis: int):
  axis = scan_axis if x.ndim > scan_axis else 0
  return jax.lax.slice_in_dim(x, start, start + size, axis=axis)


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


class _Base:
  """MaxText's `Transformer` of a config on a mesh: its construction, its
  initial state and its parameter checkpoints."""

  def _transformer(self, seed: int):
    return models.Transformer(
        self.cfg, self.mesh, quant=None, rngs=nnx.Rngs(params=seed, dropout=seed)
    )

  def _init_full(self, seed: int):
    with nn.logical_axis_rules(self.cfg.logical_axis_rules):
      _, params, rest = nnx.split(self._transformer(seed), nnx.Param, ...)
    return params.to_pure_dict(), rest.to_pure_dict()

  def _restore(self, path: str, shardings: dict | None = None) -> dict:
    """The parameters of `Transformer` from a MaxText parameter checkpoint.

    `path` is what MaxText's `load_parameters_path` takes: the output of its
    `checkpoint_conversion/to_maxtext.py` (from Hugging Face weights) or of
    `checkpointing.save_params_to_path`. MaxText's loader restores the weights
    in the config's `weight_dtype`, onto `shardings`, a tree as the
    parameters', or onto one device.
    """
    from maxtext.common import checkpointing  # pylint: disable=g-import-not-at-top

    with nn.logical_axis_rules(self.cfg.logical_axis_rules):
      _, abstract, _ = nnx.split(nnx.eval_shape(lambda: self._transformer(0)), nnx.Param, ...)
    shapes = nnx.to_pure_dict(abstract)
    if shardings is None:
      device = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
      shardings = jax.tree.map(lambda _: device, shapes)
    nnx.replace_by_pure_dict(abstract, jax.tree.map(
        lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), shapes, shardings))
    restored = checkpointing.load_params_from_path(
        path, abstract, self.cfg.checkpoint_storage_concurrent_gb)
    return _pure_dict(restored)


class Model(_Base):
  """A MaxText decoder-only model split into pipeline stages.

  `mesh` is any one stage mesh from `mesh()`: MaxText's layers place their
  sharding constraints on it, and an MPMD program needs every constraint on
  one mesh. `layers` gives the decoder layers per stage; the default splits
  them evenly. With `attention`, such as `jax_titan.ops.attention`, causal
  training attention runs there instead of in MaxText's attention op.
  """

  def __init__(self, cfg, stage_mesh: jax.sharding.Mesh, *, num_stages: int,
               layers: Sequence[int] | None = None, attention: Callable | None = None):
    if not has_split(cfg):
      raise NotImplementedError(
          f"decoder_block={cfg.decoder_block}, scan_layers={cfg.scan_layers}, "
          f"logits_via_embedding={cfg.logits_via_embedding}: pipeline stages take MaxText's "
          "llama2 decoder block as one scanned stack with an untied output head"
      )
    if cfg.remat_policy not in ("full", "none"):
      raise NotImplementedError(f"remat_policy={cfg.remat_policy}; supported: full, none")
    if layers is None:
      if cfg.num_decoder_layers % num_stages:
        raise ValueError(f"{cfg.num_decoder_layers} layers do not split into {num_stages} stages")
      layers = (cfg.num_decoder_layers // num_stages,) * num_stages
    if len(layers) != num_stages or sum(layers) != cfg.num_decoder_layers:
      raise ValueError(f"stage layers {tuple(layers)} do not split {cfg.num_decoder_layers} layers")
    self.cfg = cfg
    self.mesh = stage_mesh
    self.layers = tuple(layers)
    self.num_stages = num_stages
    self._attention = attention

    with nn.logical_axis_rules(cfg.logical_axis_rules):
      abstract = nnx.eval_shape(lambda: self._transformer(0))
    self._embed_graph = nnx.graphdef(abstract.token_embedder)
    self._layers_graph = nnx.split(abstract.decoder.layers, nnx.Param, ...)[0]
    self._norm_graph = nnx.graphdef(abstract.decoder.decoder_norm)
    self._logits_graph = nnx.graphdef(abstract.decoder.logits_dense)
    rest = jax.jit(lambda: self._init_full(0)[1])()
    self._layer_rest = self._split_layers(rest["decoder"]["layers"])

  def _split_layers(self, stacked) -> list:
    out, start = [], 0
    for n in self.layers:
      out.append(jax.tree.map(
          functools.partial(_slice_layers, start=start, size=n, scan_axis=self.cfg.param_scan_axis),
          stacked,
      ))
      start += n
    return out

  def split(self, params: dict) -> tuple[dict, ...]:
    """Stage parameter dicts from the parameters of MaxText's `Transformer`."""
    decoder = params["decoder"]
    stages = [{"layers": p} for p in self._split_layers(decoder["layers"])]
    stages[0]["token_embedder"] = params["token_embedder"]
    stages[-1]["decoder_norm"] = decoder["decoder_norm"]
    stages[-1]["logits_dense"] = decoder["logits_dense"]
    return tuple(stages)

  def init(self, seed: int = 0) -> tuple[dict, ...]:
    """Stage parameters, initialized as MaxText initializes `Transformer`."""
    return self.split(jax.jit(lambda: self._init_full(seed)[0])())

  def abstract(self, seed: int = 0) -> tuple[dict, ...]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return jax.eval_shape(lambda: self.split(self._init_full(seed)[0]))

  def load(self, path: str) -> tuple[dict, ...]:
    """Stage parameters from a MaxText parameter checkpoint (`_restore`)."""
    return self.split(self._restore(path))

  def forward(self, stage: int, params: dict, x: jax.Array) -> jax.Array:
    """Token ids in for stage 0; logits out of the last stage."""
    cfg = self.cfg
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      if stage == 0:
        embedder = nnx.merge(self._embed_graph, params["token_embedder"])
        y = embedder(x.astype(jnp.int32), model_mode=MODEL_MODE_TRAIN).astype(cfg.dtype)
      else:
        y = x
      batch, length = y.shape[0], y.shape[1]
      positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
      segments = jnp.ones((batch, length), jnp.int32)
      layers = nnx.merge(self._layers_graph, params["layers"], self._layer_rest[stage])
      routed = (_attention_through(self._attention) if self._attention
                else contextlib.nullcontext())
      with routed:
        y = nnx_scan.apply_scanned_layers(
            layers, y,
            length=self.layers[stage],
            param_scan_axis=cfg.param_scan_axis,
            apply_fn=lambda layer, h: layer(h, segments, positions, True, MODEL_MODE_TRAIN)[0],
            remat=cfg.remat_policy != "none",
            remat_policy=None,
            prevent_cse=maxtext_utils.should_prevent_cse_in_remat(cfg),
        )
      if stage < self.num_stages - 1:
        return y
      y = nnx.merge(self._norm_graph, params["decoder_norm"])(y, out_sharding=None)
      logits = nnx.merge(self._logits_graph, params["logits_dense"])(
          y, out_sharding=create_sharding(self.mesh, _LOGITS_AXES)
      )
      return logits.astype(jnp.float32) if cfg.cast_logits_to_fp32 else logits

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """Stage functions for `jax_titan.mpmd.make_train_step`.

    Stage `i` is called as `fn(params, x)`; the last stage also takes the
    targets and returns the mean next-token cross-entropy.
    """
    fns = [functools.partial(self.forward, i) for i in range(self.num_stages - 1)]
    last = self.num_stages - 1
    fns.append(lambda p, x, t: llama.cross_entropy(self.forward(last, p, x), t))
    return fns

  def flops_per_token(self, seq_len: int) -> float:
    """Training FLOPs per token: 6N for the weights plus causal attention."""
    cfg = self.cfg
    e, f, d = cfg.emb_dim, cfg.mlp_dim, cfg.head_dim
    per_layer = e * d * (2 * cfg.num_query_heads + 2 * cfg.num_kv_heads) + 3 * e * f
    attn = 6 * cfg.num_decoder_layers * cfg.num_query_heads * d * seq_len
    return 6 * (cfg.num_decoder_layers * per_layer + cfg.vocab_size * e) + attn


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


class WholeModel(_Base):
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
    """The parameters from a MaxText parameter checkpoint (`_restore`)."""
    if self._other_weights:
      raise NotImplementedError(
          f"{self.cfg.model_name} keeps weights outside its parameters "
          f"({', '.join(self._other_weights)}), which a parameter checkpoint would leave as "
          "initialized"
      )
    return (self._restore(path, self._shardings()),)

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

  def flops_per_token(self) -> float:
    """Training FLOPs per token as MaxText counts them, for sequences of the
    config's `max_target_length`."""
    tflops = maxtext_utils.calculate_tflops_training_per_device(self.cfg, log=False)[0]
    return tflops * 1e12 / (self.cfg.per_device_batch_size * self.cfg.max_target_length)
