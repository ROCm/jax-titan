"""Command A+'s published checkpoint: its config, its tensors as the model's
parameters and back, the parameters read straight onto devices, and a
checkpoint written in its layout."""

from collections.abc import Sequence
import dataclasses
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from jax_titan.models import weights as weights_lib
from jax_titan.models.command_a.model import _EXPERTS
from jax_titan.models.command_a.model import _split_layers
from jax_titan.models.command_a.model import CommandAConfig
from jax_titan.models.command_a.model import init_params


# The variant of transformers' model that this module computes: per field of the text config,
# its value here and transformers' default, which a config.json without the field means.
_VARIANT = {
    "expert_selection_fn": ("sigmoid", "softmax"),
    "norm_topk_prob": (True, True),
    "rms_norm_eps": (None, None),
    "shared_expert_combination_strategy": ("average", "average"),
    "attention_bias": (False, False),
    "hidden_act": ("silu", "silu"),
    "tie_word_embeddings": (True, True),
    "logit_scale": (1.0, 0.0625),
    "first_k_dense_replace": (0, 0),
}


def pretrained_config(path) -> CommandAConfig:
  """The config of a published checkpoint, from the `config.json` beside its
  weights. A config of another variant of transformers' model is refused."""

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
    other = {k: text.get(k, default) for k, (ours, default) in _VARIANT.items()
             if text.get(k, default) != ours}
    if (text.get("rope_parameters") or {}).get("rope_type", "default") != "default":
      other["rope_type"] = text["rope_parameters"]["rope_type"]
    if "dense" in (text.get("mlp_layer_types") or ()):
      other["mlp_layer_types"] = "dense"
    if not text.get("num_shared_experts", 0):
      other["num_shared_experts"] = 0
    if not values.get("tie_word_embeddings", True):
      other["tie_word_embeddings"] = False
    if vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
      other["vision_config.hidden_act"] = vision["hidden_act"]
    if other:
      raise NotImplementedError(f"{path}: a variant of the model that is not implemented: {other}")
    return fields

  cfg = weights_lib.config(CommandAConfig, path, flat)
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
    return weights_lib.weight(tensors, name + ".weight").T

  def vector(name, kind="weight"):
    return np.asarray(tensors[f"{name}.{kind}"]).astype(np.float32)

  def stacked(fn, name):
    return np.stack([fn(f"{lm}layers.{i}.{name}") for i in layers])

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
  return params | _image_params(tensors, cfg) if images else params


def _image_params(tensors: dict, cfg: CommandAConfig) -> dict:
  """The fp32 parameters of the vision tower and the projector, on the host."""
  patch = "embeddings.patch_embedding.weight"
  vm = next(k for k in tensors if k.endswith(patch))[:-len(patch)]
  enc, m = vm + "encoder.", cfg.vision_num_hidden_layers

  def kernel(name):
    return weights_lib.weight(tensors, name + ".weight").T

  def vector(name, kind="weight"):
    return np.asarray(tensors[f"{name}.{kind}"]).astype(np.float32)

  def vision(fn, name, **kwargs):
    return np.stack([fn(f"{enc}layers.{i}.{name}", **kwargs) for i in range(m)])

  def bias(name):
    return vision(vector, name, kind="bias")

  # A patch is flattened as channel, row, column, the order of the checkpoint's convolution kernel.
  params = {}
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
  `config.json`; without `images`, the language model alone.

  The whole model passes through host memory in fp32 and lands on one device:
  the published size takes `load_stages`.
  """
  cfg = pretrained_config(path)
  return cfg, weights_lib.to_device(params_from_tensors(weights_lib.read(path), cfg, images), dtype)


# A layer's kernels: this module's name and the checkpoint's under the layer.
_LAYER_KERNELS = {
    "wq": "self_attn.q_proj", "wk": "self_attn.k_proj", "wv": "self_attn.v_proj",
    "wo": "self_attn.o_proj", "w_router": "mlp.gate",
    "shared_gate": "mlp.shared_experts.gate_proj", "shared_up": "mlp.shared_experts.up_proj",
    "shared_down": "mlp.shared_experts.down_proj"}


def _stored(sharding, ndim: int):
  """The sharding of a kernel as the checkpoint stores it, its last two
  dimensions the other way round, for a kernel that goes on `sharding`."""
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return sharding
  spec = tuple(sharding.spec) + (None,) * (ndim - len(sharding.spec))
  spec = jax.sharding.PartitionSpec(*spec[:-2], spec[-1], spec[-2])
  return jax.sharding.NamedSharding(sharding.mesh, spec)


def _put(x, sharding, dtype) -> jax.Array:
  """`x` on `sharding` in `dtype`, cast there."""
  put = jax.device_put(np.asarray(x), sharding)
  return jax.jit(lambda w: w.astype(dtype), out_shardings=sharding)(put)


def _layer_leaf(tensors: dict, cfg: CommandAConfig, layers: range, name: str, sharding,
                dtype) -> jax.Array:
  """Parameter `name` of `layers`, stacked, on `sharding` in `dtype`, as
  `params_from_tensors` gives it.

  The host only copies the stored tensors side by side. The devices turn
  kernels to [in, out], reorder the head dimensions and cast, so a checkpoint
  of hundreds of GB is read at the speed of its files. The tensors are taken
  as stored: a quantized checkpoint, whose weights are not their stored
  values, is refused.
  """
  if name == "norm":
    stored = "input_layernorm"
  elif name in _EXPERTS:
    stored = "mlp.experts.{}." + name[2:] + "_proj"
  else:
    stored = _LAYER_KERNELS[name]
  count = cfg.num_experts if name in _EXPERTS else 1

  def read(i, x):
    return tensors[f"model.language_model.layers.{i}.{stored.format(x)}.weight"]

  first = read(layers[0], 0)
  if first.dtype not in (np.dtype(np.float16), np.dtype(np.float32), np.dtype(jnp.bfloat16)):
    raise NotImplementedError(
        f"{name} is stored as {first.dtype}: these readers take float16, bfloat16 and float32 "
        "weights, the bf16 checkpoint's")
  out = np.empty((len(layers), count) + first.shape, first.dtype)
  for n, i in enumerate(layers):
    for x in range(count):
      out[n, x] = read(i, x)
  if name not in _EXPERTS:
    out = out[:, 0]
  if name == "norm":
    return _put(out, sharding, dtype)
  order = _rotary_order(cfg)

  def kernel(w):
    w = jnp.swapaxes(w, -1, -2)
    if name in ("wq", "wk"):
      w = w.reshape(*w.shape[:2], -1, cfg.head_dim)[..., order].reshape(w.shape)
    return w.astype(dtype)

  put = jax.device_put(out, _stored(sharding, out.ndim))
  return jax.jit(kernel, out_shardings=sharding)(put)


def _bounds(cfg: CommandAConfig, devices) -> list[int]:
  """The first layer of each stage, and the layer count: at most one stage per layer."""
  counts = _split_layers(cfg, min(len(devices), cfg.num_hidden_layers))
  return [sum(counts[:i]) for i in range(len(counts) + 1)]


def to_stages(cfg: CommandAConfig, params: dict, devices, dtype=jnp.bfloat16) -> list[dict]:
  """`init_params`-shaped parameters, or a part of them such as the trained
  ones, spread over `devices`, for a model too large for one.

  The layers are split into as many runs as devices, and a stage holds the
  stacked parameters of its run on its device. The first stage also holds the
  embedding and, when `params` has them, the vision tower and the projector;
  the last, the final norm and the embedding again, as the output head.
  """
  def put(tree, device):
    return jax.tree.map(lambda w: jax.device_put(w, device).astype(dtype), tree)

  bounds = _bounds(cfg, devices)
  layers = params.get("layers", {})
  stages = [{"layers": put({k: v[a:b] for k, v in layers.items()}, device)}
            for a, b, device in zip(bounds, bounds[1:], devices)]
  first, last = devices[0], devices[len(stages) - 1]
  stages[0] |= put({k: v for k, v in params.items() if k not in ("layers", "final_norm")}, first)
  if "final_norm" in params:
    stages[-1]["final_norm"] = put(params["final_norm"], last)
  if "embed" in params and "embed" not in stages[-1]:
    stages[-1]["embed"] = put(params["embed"], last)
  return stages


def part_size(cfg: CommandAConfig, tree) -> int:
  """The number of parameters in `tree`, a part of `init_params`-shaped
  parameters such as the ones a run trained and saved, its leaves arrays or
  their shapes. A leaf that is none of the model's, or of another shape, raises."""
  def shapes(tree):
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {".".join(str(k.key) for k in path): tuple(leaf.shape) for path, leaf in leaves}

  ours, theirs = shapes(jax.eval_shape(lambda: init_params(cfg, jax.random.key(0)))), shapes(tree)
  other = sorted(name for name, shape in theirs.items() if ours.get(name) != shape)
  if other or not theirs:
    raise ValueError(
        "not parameters of this model: " + (", ".join(other) or "no parameter at all"))
  return sum(int(np.prod(shape)) for shape in theirs.values())


def load_stages(path, devices, dtype=jnp.bfloat16,
                images: bool = True) -> tuple[CommandAConfig, list[dict]]:
  """The config and the parameters of the published weights at `path`, spread
  over `devices` as `to_stages` spreads them, read stage by stage."""
  cfg, tensors = pretrained_config(path), weights_lib.read(path)
  bounds = _bounds(cfg, devices)
  places = [jax.sharding.make_single_device_sharding(d) for d in devices[:len(bounds) - 1]]
  stages = [{"layers": {k: _layer_leaf(tensors, cfg, range(a, b), k, place, dtype)
                        for k in ("norm", *_LAYER_KERNELS, *_EXPERTS)}}
            for a, b, place in zip(bounds, bounds[1:], places)]
  embed = tensors["model.language_model.embed_tokens.weight"]
  stages[0]["embed"] = _put(embed, places[0], dtype)
  if images:
    stages[0] |= jax.tree.map(lambda w: _put(w, places[0], dtype), _image_params(tensors, cfg))
  stages[-1]["final_norm"] = _put(tensors["model.language_model.norm.weight"], places[-1], dtype)
  if "embed" not in stages[-1]:
    stages[-1]["embed"] = _put(embed, places[-1], dtype)
  return cfg, stages


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
  weights_lib.write(path / "model.safetensors", tensors_from_params(params, cfg), {"format": "pt"})


def place(path, cfg: CommandAConfig, layers: Sequence[int], images: bool,
          targets: Sequence[dict]) -> tuple[dict, ...]:
  """Stage parameters of the published weights at `path`, for stages of `layers`
  layers each, with `images` those of the vision tower and the projector too:
  what `Model.place` returns."""
  tensors, start, stages = weights_lib.read(path), 0, []
  image_params = _image_params(tensors, cfg) if images else {}
  stored = {"embed": "embed_tokens", "final_norm": "norm"}
  for n, target in zip(layers, targets, strict=True):
    stage = {"layers": {
        k: _layer_leaf(tensors, cfg, range(start, start + n), k, t.sharding, t.dtype)
        for k, t in target.get("layers", {}).items()}}
    for k in stored.keys() & target.keys():
      weight = tensors[f"model.language_model.{stored[k]}.weight"]
      stage[k] = _put(weight, target[k].sharding, target[k].dtype)
    for k in image_params.keys() & target.keys():
      stage[k] = jax.tree.map(
          lambda w, t: _put(w, t.sharding, t.dtype), image_params[k], target[k])
    stages.append(stage)
    start += n
  return tuple(stages)
