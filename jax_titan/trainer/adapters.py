"""Trainer layouts for models from MaxText, MaxDiffusion, transformers and
diffusers, for Llama, for Ideogram 4, for Command A+ and for FLUX.1.

A model of another repository trains as one stage, the repository's model
called whole, with `--parallel=dp` or `fsdp`:
`--model=maxtext:<name>` a MaxText model config (e.g. `maxtext:llama3.1-8b`
or `maxtext:qwen3-8b`) on the token data, `--model=maxdiffusion:<flavor>`
FLUX.1 (`flux-dev`, `flux-schnell`) or Wan 2.1 (`wan2.1-t2v-1.3b`) on
synthetic rectified-flow batches, `--model=transformers:<name>` a decoder-only
causal language model of Hugging Face transformers, a directory with a
`config.json`, a model type or a model on the Hub, and
`--model=diffusers:<name>` a denoiser of diffusers on synthetic batches. The
last two run through torchax (`models/adapters/torch_bridge.py`) and
rematerialize through the library's own gradient-checkpointing switch.
MaxText's and MaxDiffusion's FLUX.1 attention is their own JAX
implementation unless the ops policy names attention.

`--model=ideogram4` trains JAX-Titan's own Ideogram 4
(`jax_titan.models.ideogram`) on synthetic rectified-flow batches or on a
cache of image tokens and text features (`jax_titan.data.ideogram`), and
`--model=maxdiffusion:ideogram4` the model of MaxDiffusion's Ideogram branch
on the same batches. They run as a Shardy MPMD pipeline with data parallelism
inside each stage, reducing gradients in every microbatch's backward pass.
Ideogram 4's attention does not follow the ops policy: JAX-Titan's model passes
a mask, which the providers decline, and the branch's model runs its own
attention. `--model=command-a-plus` trains JAX-Titan's Command A+
(`jax_titan.models.command_a`) on token windows or on a cache of conversations
(`jax_titan.data.command_a`), as one stage or, with its embedding frozen, as a
pipeline. `--model=flux-dev` and `flux-schnell` train JAX-Titan's own FLUX.1
(`jax_titan.models.flux`) on the synthetic batches of the two FLUX.1 adapters,
as one stage, and `--model=llama3.1-8b` JAX-Titan's Llama
(`jax_titan.models.llama`) on token windows, as one stage. For these,
`--parallel=dp` runs the same stages, weights and batches as one data-parallel
`jax.jit` program, the no-pipeline reference, and `--parallel=fsdp` also shards
parameters and optimizer state.

The layouts here take a staged model, any object with `init(seed)` and
`abstract(seed)`, the parameters of each stage and their shapes, and
`stage_fns()`, one function per stage, the last of which returns the loss.
`--init_from` needs its `load(path)`, `--parallel=fsdp` its
`logical_axes(path, leaf)` and `--save_params` its `merge(stage_params)`. Its
`decays(path, leaf)` says which of its parameters take weight decay; a builder
gives it to `make_optimizer`.

A model may train only part of its parameters. Its stage parameters are then
nested dicts, and `is_frozen(path)` tells, for the tuple of dict keys that
leads to a parameter in a stage's dicts, whether it stays as it is. Frozen
parameters are no part of the trained state: they get no gradient and no
optimizer state, are held in the model's compute dtype, its attribute `dtype`,
and neither a checkpoint nor `--save_params` holds them, so a run that resumes
reads them again from where the run took them. A model that can read weights
straight onto devices has `place(path, targets)`, which `--init_from` then
uses for published weights, a directory of safetensors files: a checkpoint
larger than the host's memory loads this way. `targets` is one tree of
ShapeDtypeStructs with shardings per stage, all of a stage's parameters or
its frozen ones alone.
"""

import dataclasses
import functools
import json
import os

from jax_titan.trainer.layout import EVAL_STEPS, Layout, Workload, abstract, placement

PROVIDERS = ("maxtext", "maxdiffusion", "transformers", "diffusers")
FLUX = ("flux-dev", "flux-schnell")
LLAMA = ("llama3.1-8b", "llama3-test")
# JAX-Titan's own models.
MODELS = ("ideogram4", "command-a-plus") + FLUX + LLAMA


def default_num_stages(model: str) -> int:
  """`--num_stages` where it is not given: 2, and 1 for a model of another
  repository, which is called whole, and for FLUX.1 and Llama, which have no
  pipeline."""
  provider, _, name = model.partition(":")
  whole = provider in PROVIDERS and (provider, name) != ("maxdiffusion", "ideogram4")
  return 1 if whole or model in FLUX + LLAMA else 2


def _overrides(args) -> dict:
  return json.loads(args.model_overrides) if args.model_overrides else {}


def _stage_layers(args):
  return tuple(int(n) for n in args.stage_layers.split(",")) if args.stage_layers else None


def _num_params(shapes) -> int:
  import jax  # pylint: disable=g-import-not-at-top

  return sum(x.size for x in jax.tree.leaves(shapes))


def _topology(args, devices, make_mesh) -> dict:
  """One mesh per group of stages from `make_mesh(devices)`."""
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  num_meshes = args.num_stages // args.stages_per_mesh
  if len(devices) % num_meshes:
    raise ValueError(f"{len(devices)} devices do not split into {num_meshes} meshes")
  per_mesh = len(devices) // num_meshes
  if args.microbatch_size % per_mesh:
    raise ValueError(
        f"--microbatch_size={args.microbatch_size} does not split across "
        f"{per_mesh} data-parallel ranks per stage"
    )
  return {
      jt_mpmd.mesh_name(i): make_mesh(devices[i * per_mesh:(i + 1) * per_mesh])
      for i in range(num_meshes)
  }


def _is_pair(v) -> bool:
  import jax  # pylint: disable=g-import-not-at-top

  return isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], jax.sharding.PartitionSpec)


def _shardings(specs, mesh):
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree.map(lambda v: jax.sharding.NamedSharding(mesh, v[1]), specs, is_leaf=_is_pair)


def _shapes(specs):
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree.map(lambda v: v[0], specs, is_leaf=_is_pair)


def _split(tree, is_frozen):
  """The trained and the frozen leaves of `tree`, nested dicts, as two such
  trees; a dict left without a leaf is dropped."""

  def part(node, path, frozen):
    if not isinstance(node, dict):
      return node if bool(is_frozen(path)) == frozen else None
    parts = {k: part(v, path + (k,), frozen) for k, v in node.items()}
    return {k: v for k, v in parts.items() if v is not None and not (isinstance(v, dict) and not v)}

  return part(tree, (), False), part(tree, (), True)


def _merged(a: dict, b: dict) -> dict:
  """The union of two trees of nested dicts that share no leaf."""
  out = dict(a)
  for k, v in b.items():
    out[k] = _merged(a[k], v) if k in a else v
  return out


def _on_pair(fn):
  """`fn`, a stage function, taking its parameters as the pair (trained, frozen)."""
  return lambda pair, *args: fn(_merged(*pair), *args)


def _init_state(args, model, optimizer, meshes, fsdp=False):
  """Init and abstract state for stage state on its mesh: replicated, or with
  `fsdp` sharded along the model's logical axes.

  Also returns the shardings of the stages' frozen parameters and `frozen()`,
  which places them, or None for a model that trains all of its parameters.
  """
  import jax  # pylint: disable=g-import-not-at-top
  from jax_titan import parallel  # pylint: disable=g-import-not-at-top
  from jax_titan.models import pretrained  # pylint: disable=g-import-not-at-top

  def shardings(tree, mesh):
    if fsdp:
      specs = parallel.tree_specs(tree, model.logical_axes, mesh.shape, fsdp=True)
    else:
      specs = jax.tree.map(lambda _: jax.sharding.PartitionSpec(), tree)
    return jax.tree.map(lambda s: jax.sharding.NamedSharding(mesh, s), specs)

  is_frozen = getattr(model, "is_frozen", lambda path: False)
  parts = [_split(p, is_frozen) for p in model.abstract(args.seed)]
  param_shapes = tuple(p for p, _ in parts)
  frozen_shapes = tuple(
      jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, model.dtype), f) for _, f in parts)
  opt_shapes = tuple(jax.eval_shape(optimizer.init, p) for p in param_shapes)
  param_sh = tuple(shardings(p, m) for p, m in zip(param_shapes, meshes))
  frozen_sh = tuple(shardings(f, m) for f, m in zip(frozen_shapes, meshes))
  opt_sh = tuple(shardings(s, m) for s, m in zip(opt_shapes, meshes))
  placed = []  # The frozen parameters, once they are on their devices.

  def place(trained=True):
    """The trained and the frozen parameters on their shardings; without
    `trained`, where the model reads a part of its weights, the frozen alone."""
    if args.init_from and hasattr(model, "place") and pretrained.is_safetensors(args.init_from):
      targets = tuple(abstract(f, fs) for f, fs in zip(frozen_shapes, frozen_sh))
      if trained:
        targets = tuple(
            _merged(abstract(p, ps), f) for p, ps, f in zip(param_shapes, param_sh, targets))
      stages = [_split(s, is_frozen) for s in model.place(args.init_from, targets)]
      return tuple(p for p, _ in stages), tuple(f for _, f in stages)
    stages = model.load(args.init_from) if args.init_from else model.init(args.seed)
    stages = [_split(s, is_frozen) for s in stages]
    params = jax.device_put(tuple(p for p, _ in stages), param_sh)
    frozen = jax.device_put(tuple(f for _, f in stages), frozen_sh)

    def cast(tree):
      return jax.tree.map(lambda x: x.astype(model.dtype), tree)

    return params, tuple(jax.jit(cast, out_shardings=s)(f) if f else f
                         for f, s in zip(frozen, frozen_sh))

  def init():
    params, frozen = place()
    placed[:] = [frozen]
    opt_states = tuple(
        jax.jit(optimizer.init, out_shardings=s)(p) for p, s in zip(params, opt_sh)
    )
    return params, opt_states

  def frozen():
    # A run that resumes restores its state and never calls init().
    if not placed:
      placed[:] = [place(trained=False)[1]]
    return placed[0]

  abstract_state = (
      tuple(abstract(s, sh) for s, sh in zip(param_shapes, param_sh)),
      tuple(abstract(s, sh) for s, sh in zip(opt_shapes, opt_sh)),
  )
  return init, abstract_state, param_sh, opt_sh, frozen_sh, frozen if any(frozen_sh) else None


def _mpmd_layout(args, topology, model, optimizer, inputs, targets, split_batch) -> Layout:
  """A pipeline step over `model.stage_fns()` with replicated stage state.

  `inputs` and `targets` map each step input to `(ShapeDtypeStruct,
  PartitionSpec)` on the first and last stage mesh; `split_batch` turns a
  host batch into the step's inputs and targets.
  """
  import jax  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan import ops  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  num_stages, num_meshes = args.num_stages, len(topology)
  options = json.loads(args.mpmd_options) if args.mpmd_options else None
  config = jt_mpmd.make_config(topology, args.schedule or None, options, num_stages=num_stages)
  meshes = [topology[jt_mpmd.mesh_name(i % num_meshes)] for i in range(num_stages)]
  init, abstract_state, param_sh, opt_sh, frozen_sh, frozen = _init_state(
      args, model, optimizer, meshes)
  extra = (frozen_sh,) if frozen else ()

  in_sh, tgt_sh = _shardings(inputs, meshes[0]), _shardings(targets, meshes[-1])
  stage_fns = [ops.for_stage(fn, i) for i, fn in enumerate(model.stage_fns())]
  if frozen:
    stage_fns = [_on_pair(fn) for fn in stage_fns]
  train_step = jt_mpmd.make_train_step(stage_fns, optimizer, max_grad_norm=args.max_grad_norm)
  scalar = NamedSharding(meshes[-1], P())
  jitted = mpmd.jit(
      train_step,
      config,
      in_shardings=(param_sh, opt_sh, *extra, in_sh, tgt_sh),
      out_shardings=(param_sh, opt_sh, scalar, scalar),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    x, t = split_batch(batch)
    return jax.device_put(x, in_sh), jax.device_put(t, tgt_sh)

  def eval_loss(stage_params, *batch):
    *rest, x, t = batch
    stage_params = list(zip(stage_params, *rest)) if rest else stage_params
    return jt_mpmd.pipeline_loss(stage_fns, stage_params, x, t)

  eval_step = mpmd.jit(
      eval_loss,
      jt_mpmd.make_config(topology, None, options, num_stages=num_stages),
      in_shardings=(param_sh, *extra, in_sh, tgt_sh),
      out_shardings=scalar,
  )
  return Layout(init, abstract_state, jitted, _shapes(inputs), _shapes(targets), put_batch,
                f"mpmd:{num_stages}",
                placement=placement(
                    topology, [jt_mpmd.mesh_name(i % num_meshes) for i in range(num_stages)]),
                eval_step=eval_step, export=getattr(model, "merge", None), frozen=frozen)


def _spmd_layout(args, mesh, model, optimizer, inputs, targets, split_batch) -> Layout:
  """No pipeline: every stage in one `jax.jit` program on `mesh`, data-parallel
  or, with `--parallel=fsdp`, with the state sharded too.

  The step runs the stages over all microbatches at once, which gives the
  pipeline's loss and gradients when microbatches hold equal token counts or
  the model's last stage takes the mean of their means itself, and updates
  the replicated or sharded state the way the pipeline does. `inputs`, `targets`
  and `split_batch` are `_mpmd_layout`'s: batched leaves lose their
  microbatch axis, and per-microbatch constants keep the first microbatch's.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan import ops  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  def batched(spec):
    return len(spec) > 1 and spec[1] is not None

  def flat_spec(v):
    shape, spec = v
    dims = shape.shape
    if batched(spec):
      dims = (dims[0] * dims[1],) + dims[2:]
    else:
      dims = dims[1:]
    return jax.ShapeDtypeStruct(dims, shape.dtype), P(*spec[1:])

  def flat_host(a, v):
    return a.reshape(-1, *a.shape[2:]) if batched(v[1]) else a[0]

  flat_inputs = jax.tree.map(flat_spec, inputs, is_leaf=_is_pair)
  flat_targets = jax.tree.map(flat_spec, targets, is_leaf=_is_pair)
  in_sh, tgt_sh = _shardings(flat_inputs, mesh), _shardings(flat_targets, mesh)
  init, abstract_state, param_sh, opt_sh, frozen_sh, frozen = _init_state(
      args, model, optimizer, [mesh] * args.num_stages, fsdp=args.parallel == "fsdp")
  extra = (frozen_sh,) if frozen else ()

  stage_fns = [ops.for_stage(fn, i) for i, fn in enumerate(model.stage_fns())]
  last = len(stage_fns) - 1

  def loss_fn(stage_params, *batch):
    *rest, x, t = batch
    if rest:
      stage_params = [_merged(p, f) for p, f in zip(stage_params, *rest)]
    for i, fn in enumerate(stage_fns):
      x = fn(stage_params[i], x, t) if i == last else fn(stage_params[i], x)
    return x

  def train_step(stage_params, opt_states, *batch):
    loss, grads = jax.value_and_grad(loss_fn)(stage_params, *batch)
    norm = jt_mpmd.global_grad_norm(grads)
    updated = [
        jt_mpmd.clipped_update(optimizer, p, g, s, norm, args.max_grad_norm)
        for p, g, s in zip(stage_params, grads, opt_states)
    ]
    return (tuple(p for p, _ in updated), tuple(s for _, s in updated),
            jnp.where(jnp.isfinite(norm), loss, jnp.nan), norm)

  scalar = jax.sharding.NamedSharding(mesh, P())
  jitted = jax.jit(
      train_step,
      in_shardings=(param_sh, opt_sh, *extra, in_sh, tgt_sh),
      out_shardings=(param_sh, opt_sh, scalar, scalar),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    x, t = split_batch(batch)
    x, t = jax.tree.map(flat_host, x, inputs), jax.tree.map(flat_host, t, targets)
    return jax.device_put(x, in_sh), jax.device_put(t, tgt_sh)

  eval_step = jax.jit(
      loss_fn, in_shardings=(param_sh, *extra, in_sh, tgt_sh), out_shardings=scalar)
  return Layout(init, abstract_state, jitted, _shapes(flat_inputs), _shapes(flat_targets),
                put_batch, "spmd", placement=placement({"mesh": mesh}, ["mesh"] * args.num_stages),
                eval_step=eval_step, export=getattr(model, "merge", None), frozen=frozen)


def _token_batches(args):
  """The spec of a step's token windows, inputs and targets alike, and the
  function that splits a host batch of windows of `seq_len + 1` into them."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  tokens = (jax.ShapeDtypeStruct((num_mb, mb, seq), jnp.int32),
            jax.sharding.PartitionSpec(None, jt_mpmd.DATA_AXIS, None))

  def split_batch(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return batch[..., :-1], batch[..., 1:]

  return tokens, split_batch


def _needs_own_tokens(args, vocab_size: int):
  """Refuses the default token files, which hold Llama 3's tokens, for another vocabulary."""
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top

  own_tokens = vocab_size != llama.FLAVORS["llama3.1-8b"].vocab_size
  held_out = args.eval_every and args.data != "synthetic" and not args.eval_data
  if own_tokens and (not args.data or held_out):
    raise ValueError(
        f"{args.model} does not read Llama 3's tokens, which the default --data and --eval_data "
        "hold: name its own token files (python -m jax_titan.data.text --tokenizer)")


def _one_stage(args, ops: bool = False):
  """Refuses what a model without pipeline stages cannot run as, and `--ops`
  for one whose operators `jax_titan.ops` cannot reach."""
  if args.parallel == "mpmd" or args.num_stages != 1:
    raise NotImplementedError(
        f"{args.model} has no pipeline: --parallel=dp or --parallel=fsdp, without --num_stages")
  if args.ops and not ops:
    raise NotImplementedError(f"{args.model} runs its own operators: no --ops")


def _data_mesh(devices):
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  return jax.sharding.Mesh(
      np.asarray(devices).reshape(len(devices), 1), (jt_mpmd.DATA_AXIS, jt_mpmd.TENSOR_AXIS))


def _routed_attention():
  """`jax_titan.ops.attention` when the ops policy names attention, else None.

  Unrouted, a reference model keeps its own attention implementation.
  """
  from jax_titan import ops  # pylint: disable=g-import-not-at-top

  policy = ops.current_policy()
  routed = "attention" in policy.prefs or any("attention" in s for s in policy.stages.values())
  return ops.attention if routed else None


def _maxtext(args, name, devices, make_optimizer):
  from jax_titan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top

  _one_stage(args, ops=True)
  if not args.remat:
    raise NotImplementedError(
        f"{args.model}: MaxText rematerializes its scanned layers under every remat_policy")
  cfg = mt.config(
      name, **{"remat_policy": "full", **_overrides(args), "max_target_length": args.seq_len})
  _needs_own_tokens(args, cfg.vocab_size)
  mesh = mt.mesh(cfg, devices)
  model = mt.WholeModel(cfg, mesh, attention=_routed_attention(), fsdp=args.parallel == "fsdp")
  tokens, split_batch = _token_batches(args)
  layout = _spmd_layout(args, mesh, model, make_optimizer(model.decays), tokens, tokens,
                        split_batch)
  tokens_per_step = args.microbatches * args.microbatch_size * args.seq_len
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=tokens_per_step,
      flops_per_step=model.flops_per_token() * tokens_per_step,
      vocab_size=cfg.vocab_size,
  )
  return layout, work


def _transformers(args, name, devices, make_optimizer):
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import transformers as hf  # pylint: disable=g-import-not-at-top

  _one_stage(args)
  overrides = _overrides(args)
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  cfg = hf.config(name, **overrides)
  if cfg.model_type not in hf.VERIFIED:
    raise NotImplementedError(
        f"{args.model} is of transformers' model type {cfg.model_type!r}, which was not "
        f"compared with PyTorch through torchax: {', '.join(hf.VERIFIED)}")
  vocab_size = hf.vocab_size(cfg)
  _needs_own_tokens(args, vocab_size)
  model = hf.causal_lm(cfg, seed=args.seed, dtype=dtype, remat=args.remat,
                       init_from=args.init_from)
  tokens, split_batch = _token_batches(args)
  layout = _spmd_layout(args, _data_mesh(devices), model, make_optimizer(model.decays),
                        tokens, tokens, split_batch)
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=args.microbatches * args.microbatch_size * args.seq_len,
      flops_per_step=None,
      vocab_size=vocab_size,
  )
  return layout, work


def _batch_specs(batches):
  """The specs of a family's batches (`jax_titan.data.diffusion`): inputs and target."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  x, target = batches.next_batch(0)

  def spec(name, a):
    return (jax.ShapeDtypeStruct(a.shape, jnp.float32),
            P() if name in batches.constants else P(None, jt_mpmd.DATA_AXIS))

  return {k: spec(k, v) for k, v in x.items()}, spec("", target)


def _flux_batches(args, *, in_channels, joint_attention_dim, pooled_projection_dim,
                  guidance_embeds):
  """FLUX.1's batches at `--resolution` and `--text_tokens`, and its text tokens."""
  from jax_titan.data import diffusion  # pylint: disable=g-import-not-at-top

  text_tokens = args.text_tokens or (512 if guidance_embeds else 256)
  batches = diffusion.flux(
      args.seed, (args.microbatches, args.microbatch_size), image_side=args.resolution // 16,
      text_tokens=text_tokens, in_channels=in_channels, joint_attention_dim=joint_attention_dim,
      pooled_projection_dim=pooled_projection_dim, guidance_embeds=guidance_embeds)
  return batches, text_tokens


def _diffusers(args, name, devices, make_optimizer):
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import diffusers as hfd  # pylint: disable=g-import-not-at-top

  _one_stage(args)
  overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in _overrides(args).items()}
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  cfg = hfd.config(name, **overrides)
  if cfg["_class_name"] not in hfd.FAMILIES:
    raise NotImplementedError(
        f"{args.model} is diffusers' {cfg['_class_name']}, which was not compared with PyTorch "
        f"through torchax: {', '.join(hfd.FAMILIES)}")
  model = hfd.denoiser(cfg, seed=args.seed, dtype=dtype, remat=args.remat)
  c = model.module.config
  batches, _ = _flux_batches(
      args, in_channels=c.in_channels, joint_attention_dim=c.joint_attention_dim,
      pooled_projection_dim=c.pooled_projection_dim, guidance_embeds=c.guidance_embeds)
  inputs, targets = _batch_specs(batches)
  layout = _spmd_layout(args, _data_mesh(devices), model, make_optimizer(model.decays),
                        inputs, targets, lambda batch: batch)
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=args.microbatches * args.microbatch_size * batches.tokens,
      flops_per_step=None,
      next_batch=batches.next_batch,
  )
  return layout, work


def _maxdiffusion(args, flavor, devices, make_optimizer):
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.data import diffusion  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxdiffusion as md  # pylint: disable=g-import-not-at-top

  # FLUX.1's attention can run through jax_titan.ops; Wan 2.1's cannot.
  _one_stage(args, ops=flavor not in md.WAN)
  kwargs = {
      "dtype": jnp.bfloat16,
      "weights_dtype": jnp.float32,
      "remat_policy": "FULL" if args.remat else "NONE",
      **{k: tuple(v) if isinstance(v, list) else v for k, v in _overrides(args).items()},
  }
  for k in ("dtype", "weights_dtype"):
    kwargs[k] = jnp.dtype(kwargs[k])
  mesh, fsdp = _data_mesh(devices), args.parallel == "fsdp"
  samples = args.microbatches * args.microbatch_size
  if flavor in md.WAN:
    cfg = {**md.WAN[flavor], **kwargs}
    model = md.whole_wan(mesh, fsdp=fsdp, **cfg)
    batches = diffusion.wan(
        args.seed, (args.microbatches, args.microbatch_size),
        frames=(args.frames - 1) // 4 + 1, height=args.resolution // 8, width=args.resolution // 8,
        patch=cfg["patch_size"], text_tokens=args.text_tokens or 512,
        in_channels=cfg["in_channels"], text_dim=cfg["text_dim"])
    flops = None
  else:
    flux = md.transformer(flavor, **kwargs)
    model = md.whole_flux(flux, mesh, fsdp=fsdp, attention=_routed_attention())
    batches, text_tokens = _flux_batches(
        args, in_channels=flux.in_channels, joint_attention_dim=flux.joint_attention_dim,
        pooled_projection_dim=flux.pooled_projection_dim, guidance_embeds=flux.guidance_embeds)
    flops = samples * md.flux_flops_per_sample(flux, batches.tokens - text_tokens, text_tokens)
  inputs, targets = _batch_specs(batches)
  layout = _spmd_layout(args, mesh, model, make_optimizer(model.decays), inputs, targets,
                        lambda batch: batch)
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=samples * batches.tokens,
      flops_per_step=flops,
      next_batch=batches.next_batch,
  )
  return layout, work


def _flux(args, devices, make_optimizer):
  """JAX-Titan's FLUX.1 on the batches of `_flux_batches`. Besides config
  fields, `--model_overrides` takes `dtype`."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan.models import flux  # pylint: disable=g-import-not-at-top

  _one_stage(args, ops=True)
  overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in _overrides(args).items()}
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  cfg = dataclasses.replace(flux.FLAVORS[args.model], **overrides)
  mesh = _data_mesh(devices)
  constrain = None
  if args.parallel == "fsdp":
    # Keeps the batch sharded, so FSDP gathers the weights and not the activations.
    constrain = functools.partial(
        jax.lax.with_sharding_constraint,
        shardings=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(jt_mpmd.DATA_AXIS)))
  model = flux.Model(cfg, dtype=dtype, remat=args.remat, constrain=constrain)
  batches, text_tokens = _flux_batches(
      args, in_channels=cfg.in_channels, joint_attention_dim=cfg.joint_attention_dim,
      pooled_projection_dim=cfg.pooled_projection_dim, guidance_embeds=cfg.guidance_embeds)
  inputs, targets = _batch_specs(batches)
  layout = _spmd_layout(args, mesh, model, make_optimizer(model.decays), inputs, targets,
                        lambda batch: batch)
  samples = args.microbatches * args.microbatch_size
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=samples * batches.tokens,
      flops_per_step=samples * cfg.flops_per_sample(batches.tokens - text_tokens, text_tokens),
      next_batch=batches.next_batch,
  )
  return layout, work


def _llama(args, devices, make_optimizer):
  """JAX-Titan's Llama on token windows. Besides config fields,
  `--model_overrides` takes `dtype`."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top
  from jax_titan.observability import flops as flops_lib  # pylint: disable=g-import-not-at-top

  _one_stage(args, ops=True)
  overrides = _overrides(args)
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  cfg = dataclasses.replace(llama.FLAVORS[args.model], **overrides)
  _needs_own_tokens(args, cfg.vocab_size)
  mesh = _data_mesh(devices)
  constrain = functools.partial(
      jax.lax.with_sharding_constraint,
      shardings=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(jt_mpmd.DATA_AXIS)))
  model = llama.Model(cfg, dtype=dtype, remat=args.remat, constrain=constrain)
  tokens, _ = _token_batches(args)
  # One microbatch of any size: a validation batch need not have the step's.
  layout = _spmd_layout(args, mesh, model, make_optimizer(model.decays), tokens, tokens,
                        lambda batch: (batch[None, :, :-1], batch[None, :, 1:]))
  seq = args.seq_len
  tokens_per_step = args.microbatches * args.microbatch_size * seq
  work = Workload(
      num_params=cfg.num_params(),
      tokens_per_step=tokens_per_step,
      flops_per_step=cfg.flops_per_token(seq) * tokens_per_step,
      vocab_size=cfg.vocab_size,
      hardware_flops_per_step=(cfg.flops_per_token(seq) + flops_lib.recompute_flops_per_token(
          cfg, seq, args.remat)) * tokens_per_step,
  )
  return layout, work


def _ideogram(args, devices, make_optimizer, own: bool):
  """Ideogram 4, JAX-Titan's own (`own`) or from MaxDiffusion's Ideogram branch.

  Besides config fields, `--model_overrides` takes `dtype` and, for JAX-Titan's
  own model, `timestep_mean` and `timestep_std` of the training timesteps. That
  model's config starts from the `config.json` of published weights that
  `--init_from` names. It also trains on a cache of real samples
  (`--data`), with a tenth of its images, at least one, held out for
  validation when `--eval_every` is set, so a run that resumes with that flag
  changed trains on other images.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan.models import ideogram, pretrained  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  overrides = _overrides(args)
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  schedule = {}
  if own:
    ig, config = ideogram, ideogram.IdeogramConfig
    schedule = {k: overrides.pop(f"timestep_{k}") for k in ("mean", "std")
                if f"timestep_{k}" in overrides}
  else:
    from jax_titan.models.adapters import ideogram as ig  # pylint: disable=g-import-not-at-top

    config = ig.Ideogram4Config
  fields = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
  if own and args.init_from and pretrained.is_safetensors(args.init_from):
    cfg = dataclasses.replace(ideogram.pretrained_config(args.init_from), **fields)
  else:
    cfg = config(**fields)

  def make_mesh(d):
    return jax.sharding.Mesh(
        np.asarray(d).reshape(len(d), 1), (jt_mpmd.DATA_AXIS, jt_mpmd.TENSOR_AXIS))

  mesh = make_mesh(devices)
  constrain = None
  if args.parallel == "fsdp":
    # Keeps the batch sharded, so FSDP gathers the weights and not the activations.
    constrain = functools.partial(
        jax.lax.with_sharding_constraint,
        shardings=jax.sharding.NamedSharding(mesh, P(jt_mpmd.DATA_AXIS)))
  model = ig.Model(cfg, num_stages=args.num_stages, blocks=_stage_layers(args), dtype=dtype,
                   remat=args.remat, constrain=constrain)

  side = args.resolution // 16
  image_tokens = side * side
  text_tokens = args.text_tokens or 256
  cache, held_out = None, 0
  if args.data not in ("", "synthetic"):
    if not own:
      raise NotImplementedError("--data reads a cache of samples for ideogram4 only")
    from jax_titan.data import ideogram as data  # pylint: disable=g-import-not-at-top

    cache = data.Cache.open(args.data)
    if cache.resolution != args.resolution:
      raise ValueError(
          f"{args.data} holds images of {cache.resolution} pixels, not {args.resolution}")
    text_tokens = cache.text_tokens
    held_out = max(1, len(cache) // 10) if args.eval_every else 0
    if held_out >= len(cache):
      raise ValueError(f"{args.data} has {len(cache)} images, too few to hold one out")
  ids = ig.sequence_ids(side, text_tokens)
  mb, num_mb = args.microbatch_size, args.microbatches
  b = (num_mb, mb)
  latent_shape = b + (image_tokens, cfg.in_channels)
  text_shape = b + (text_tokens, cfg.llm_features_dim)
  f32 = jnp.float32
  batched = P(None, jt_mpmd.DATA_AXIS)
  inputs = {
      "img": (jax.ShapeDtypeStruct(latent_shape, f32), batched),
      "txt": (jax.ShapeDtypeStruct(text_shape, f32), batched),
      "timestep": (jax.ShapeDtypeStruct(b, f32), batched),
      **{k: (jax.ShapeDtypeStruct(b + v.shape, v.dtype), batched) for k, v in ids.items()},
  }
  targets = (jax.ShapeDtypeStruct(latent_shape, f32), batched)
  # The synthetic text features are the largest input by far, and the same every step.
  txt = None if cache else np.random.default_rng(args.seed).standard_normal(text_shape, np.float32)

  def next_batch(step):
    """Noised latents and velocity targets with their text features: random ones,
    or the cache's samples, the held-out images for a validation step."""
    rng = np.random.default_rng((args.seed, step))
    if cache is None:
      latents = rng.standard_normal(latent_shape, np.float32)
      text, batch_ids = txt, {k: np.broadcast_to(v, b + v.shape) for k, v in ids.items()}
    else:
      train = len(cache) - held_out
      if step >= EVAL_STEPS:
        images = train + rng.integers(0, held_out, b)
      else:
        images = rng.integers(0, train, b)
      latents, text, valid = (x.reshape(b + x.shape[1:]) for x in cache.samples(images.reshape(-1)))
      batch_ids = ig.padded_sequence_ids(side, valid)
    noise = rng.standard_normal(latent_shape, np.float32)
    t = ig.timesteps(rng.standard_normal(b), args.resolution, **schedule)
    img, target = ideogram.noised(latents, noise, t)
    x = {"img": img, "txt": text, "timestep": t, **batch_ids}
    return x, target

  optimizer = make_optimizer(model.decays)
  if args.parallel == "mpmd":
    layout = _mpmd_layout(args, _topology(args, devices, make_mesh), model, optimizer,
                          inputs, targets, lambda batch: batch)
  else:
    layout = _spmd_layout(args, mesh, model, optimizer, inputs, targets, lambda batch: batch)
  samples = num_mb * mb
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=samples * (image_tokens + text_tokens),
      flops_per_step=model.flops_per_sample(image_tokens, text_tokens) * samples,
      next_batch=next_batch,
      data="synthetic" if cache is None else
      f"{args.data} ({len(cache) - held_out} images, {held_out} more held out)",
  )
  return layout, work


def _command_a(args, devices, make_optimizer):
  """Command A+ on token windows, or on a cache of conversations.

  Besides config fields, `--model_overrides` takes `dtype`, `image_tiles` and
  `freeze`. With tiles, every sequence of a synthetic batch opens with that
  many random image tiles, and the vision tower and the projector train with
  the language model; without, they are not part of the trained state.
  `freeze` lists what is not trained, of `experts` and `embed`; the model runs
  as more than one stage only with its embedding frozen. The config starts
  from the `config.json` of the published checkpoint that `--init_from` names.

  `--data`, when a directory, is a cache that `jax_titan.data.command_a`
  wrote: a step trains on random conversations of it, and with `--eval_every`
  its last tenth, at least one, is held out and validated on in turn.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top
  from jax_titan.models import command_a, pretrained  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  overrides = _overrides(args)
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  tiles = overrides.pop("image_tiles", 0)
  freeze = overrides.pop("freeze", ())
  if args.init_from and pretrained.is_safetensors(args.init_from):
    cfg = dataclasses.replace(command_a.pretrained_config(args.init_from), **overrides)
  else:
    cfg = command_a.CommandAConfig(**overrides)

  def make_mesh(d):
    return jax.sharding.Mesh(
        np.asarray(d).reshape(len(d), 1), (jt_mpmd.DATA_AXIS, jt_mpmd.TENSOR_AXIS))

  mesh = make_mesh(devices)
  constrain = None
  if args.parallel == "fsdp":
    # Keeps the batch sharded, so FSDP gathers the weights and not the activations.
    constrain = functools.partial(
        jax.lax.with_sharding_constraint,
        shardings=jax.sharding.NamedSharding(mesh, P(jt_mpmd.DATA_AXIS)))
  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  cache = data.Cache.open(args.data) if os.path.isdir(args.data) else None
  if cache is not None:
    seq = cache.tokens.shape[1]
  # Conversations give microbatches different numbers of targets: the layouts without a
  # pipeline then take the pipeline's mean of the microbatches' means.
  flat = cache is not None and args.parallel != "mpmd"
  model = command_a.Model(cfg, num_stages=args.num_stages, layers=_stage_layers(args),
                          dtype=dtype, remat=args.remat, constrain=constrain, images=tiles > 0,
                          freeze=freeze, microbatches=num_mb if flat else 1)
  b = (num_mb, mb)
  batched = P(None, jt_mpmd.DATA_AXIS)
  tokens = (jax.ShapeDtypeStruct(b + (seq,), jnp.int32), batched)
  source = "synthetic"

  def whole(batch):
    return batch

  def windows(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return batch[..., :-1], batch[..., 1:]

  split_batch = whole
  if cache is not None and not tiles:
    held_out = max(1, len(cache) // 10) if args.eval_every else 0
    train = len(cache) - held_out
    if train < 1:
      raise ValueError(f"{args.data} has {len(cache)} conversations, too few to hold one out")
    inputs = tokens
    source = f"{args.data} ({train} conversations, {held_out} more held out)"

    def next_batch(step):
      """Conversations of the cache: random ones, or for a validation step the
      held-out ones in turn."""
      if step >= EVAL_STEPS:
        picks = train + (np.arange(num_mb * mb) + (step - EVAL_STEPS) * num_mb * mb) % held_out
      else:
        picks = np.random.default_rng((args.seed, step)).integers(0, train, num_mb * mb)
      picks = picks.reshape(b)
      return cache.tokens[picks], cache.targets[picks]
  elif not tiles:
    inputs, next_batch, split_batch = tokens, None, windows
  else:
    if args.data not in ("", "synthetic"):
      raise NotImplementedError("command-a-plus trains with image tiles on synthetic batches only")
    image_tokens, size = tiles * cfg.tile_tokens, cfg.vision_image_size
    if image_tokens >= seq:
      raise ValueError(f"{tiles} tiles are {image_tokens} tokens, more than --seq_len={seq} holds")
    pixel_shape = b + (tiles, cfg.vision_num_channels, size, size)
    inputs = {"tokens": tokens, "pixels": (jax.ShapeDtypeStruct(pixel_shape, jnp.float32), batched)}

    def next_batch(step):
      """Random tiles, their image tokens and random text; the text is the target."""
      rng = np.random.default_rng((args.seed, step))
      text = rng.integers(0, cfg.vocab_size, b + (seq + 1 - image_tokens,), dtype=np.int32)
      text[text == cfg.image_token_id] = cfg.pad_token_id
      image = np.full(b + (image_tokens,), cfg.image_token_id, np.int32)
      window = np.concatenate([image, text], axis=-1)
      x = {"tokens": window[..., :-1], "pixels": rng.standard_normal(pixel_shape, np.float32)}
      return x, np.where(window == cfg.image_token_id, -1, window)[..., 1:]

  optimizer = make_optimizer(model.decays)
  if args.parallel == "mpmd":
    layout = _mpmd_layout(args, _topology(args, devices, make_mesh), model, optimizer,
                          inputs, tokens, split_batch)
  else:
    layout = _spmd_layout(args, mesh, model, optimizer, inputs, tokens, split_batch)
  tokens_per_step = num_mb * mb * seq
  # Not the stages' parameters: with several stages the embedding is held twice.
  shapes = jax.eval_shape(lambda: command_a.init_params(cfg, jax.random.key(0), tiles > 0))
  work = Workload(
      num_params=_num_params(shapes),
      tokens_per_step=tokens_per_step,
      flops_per_step=model.flops_per_token(seq) * tokens_per_step,
      vocab_size=cfg.vocab_size,
      next_batch=next_batch,
      data=source,
  )
  return layout, work


def build(args, devices, make_optimizer) -> tuple[Layout, Workload]:
  """The layout and workload of `--model=<provider>:<name>`, or of one of `MODELS`.

  `make_optimizer(decays)` gives the optimizer for a model's `decays(path,
  leaf)`, which says which parameters take weight decay.
  """
  provider, _, name = args.model.partition(":")
  own = args.model in MODELS
  ideogram = args.model == "ideogram4" or (provider == "maxdiffusion" and name == "ideogram4")
  if args.stage_fsdp or args.tensor_parallel != 1:
    raise NotImplementedError(f"{args.model} takes neither --stage_fsdp nor --tensor_parallel")
  if args.parallel == "mpmd" and args.reduce_grads != "microbatch":
    raise NotImplementedError(f"{args.model} reduces gradients per microbatch only")
  if args.init_from and provider not in ("maxtext", "transformers") and (
      not own or args.model in FLUX + LLAMA):
    raise NotImplementedError(
        "--init_from loads parameters of maxtext: and transformers: models, ideogram4 and "
        "command-a-plus")
  synthetic_only = (provider == "diffusers" or (provider == "maxdiffusion" and not ideogram)
                    or args.model in FLUX)
  if synthetic_only and args.data not in ("", "synthetic"):
    raise NotImplementedError(f"{args.model} trains on synthetic batches only")
  if args.parallel != "mpmd" and (args.microbatches * args.microbatch_size) % len(devices):
    raise ValueError(
        f"--parallel={args.parallel} splits all {args.microbatches * args.microbatch_size} "
        f"sequences of a step across {len(devices)} devices"
    )
  if provider == "transformers":
    return _transformers(args, name, devices, make_optimizer)
  if provider == "diffusers":
    return _diffusers(args, name, devices, make_optimizer)
  if provider == "maxtext":
    return _maxtext(args, name, devices, make_optimizer)
  if ideogram:
    return _ideogram(args, devices, make_optimizer, own)
  if args.model == "command-a-plus":
    return _command_a(args, devices, make_optimizer)
  if args.model in FLUX:
    return _flux(args, devices, make_optimizer)
  if args.model in LLAMA:
    return _llama(args, devices, make_optimizer)
  if provider == "maxdiffusion":
    return _maxdiffusion(args, name, devices, make_optimizer)
  raise ValueError(f"unknown model provider {provider!r}; known: {', '.join(PROVIDERS)}")
