"""What a trainer builder returns: how training state is placed and stepped,
and what one step processes. `spmd_layout` and `mpmd_layout` make the first
from a model (`jax_titan.models.contract`), without a pipeline and with one."""

import dataclasses
import json
from typing import Any, Callable

from jax_titan.models import contract


# `Workload.next_batch` gets these steps and later ones for validation batches,
# which no training run reaches.
EVAL_STEPS = 10**9


@dataclasses.dataclass
class Layout:
  """How the training state is placed and stepped.

  `abstract_state` has the tree structure, shapes, dtypes and shardings of
  `init()`'s `(params, opt_state)`, so a checkpoint can be restored without
  initializing first. `put_batch` places a host batch as the step's inputs
  and targets, which `input_spec` and `target_spec` describe.

  `placement` maps each mesh name to its device ids and lists each stage's
  mesh, for `env.json`'s rank map. `eval_step(params, inputs, targets)` is
  the mean loss without an update, for validation. `export(params)` gives the
  model's parameters as `--init_from` reads them, for `--save_params`.

  `frozen()`, when a model trains only part of its parameters, places the
  rest, which are no part of the state and of no checkpoint: `step` takes them
  right after the state and `eval_step` right after the parameters, and
  `export` gives the trained parameters alone, which `--init_from` does not
  read. A run that resumes calls `frozen()` without `init()`.

  `config` is the model's configuration, which a run records.
  """

  init: Callable[[], tuple]
  abstract_state: tuple
  step: Any
  input_spec: Any
  target_spec: Any
  put_batch: Callable
  structure: str
  placement: dict | None = None
  eval_step: Any = None
  export: Callable | None = None
  frozen: Callable[[], Any] | None = None
  config: Any = None


def placement(meshes: dict, stage_meshes) -> dict:
  """`Layout.placement` from {mesh name: Mesh} and each stage's mesh name."""
  return {
      "meshes": {name: [int(d.id) for d in m.devices.flat] for name, m in meshes.items()},
      "stages": list(stage_meshes),
  }


@dataclasses.dataclass
class Workload:
  """What one training step processes.

  Token models train on token windows below `vocab_size`. Other models give
  `next_batch(step)`, a host batch for `put_batch`, synthetic or from `data`,
  and a validation batch for steps from `EVAL_STEPS` on.
  `flops_per_step` counts model FLOPs (for MFU), None for a model nobody has
  a count for; `hardware_flops_per_step` adds what rematerialization
  recomputes (for HFU), when known.
  """

  num_params: int
  tokens_per_step: int
  flops_per_step: float | None
  vocab_size: int | None = None
  next_batch: Callable[[int], Any] | None = None
  data: str = "synthetic"
  hardware_flops_per_step: float | None = None


def abstract(shapes, shardings):
  """ShapeDtypeStructs of `shapes` with `shardings`."""
  import jax  # pylint: disable=g-import-not-at-top

  return jax.tree.map(
      lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), shapes, shardings
  )


def stage_topology(args, devices, make_mesh) -> dict:
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


def _init_state(args, model: contract.Model, optimizer, meshes, fsdp=False):
  """Init and abstract state for stage state on its mesh: replicated, or with
  `fsdp` sharded along the model's logical axes.

  Also returns the shardings of the stages' frozen parameters and `frozen()`,
  which places them, or None for a model that trains all of its parameters.
  """
  import jax  # pylint: disable=g-import-not-at-top
  from jax_titan import parallel  # pylint: disable=g-import-not-at-top
  from jax_titan.models import weights  # pylint: disable=g-import-not-at-top

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
    if args.init_from and hasattr(model, "place") and weights.is_safetensors(args.init_from):
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


def mpmd_layout(args, topology, model: contract.Model, optimizer, inputs, targets,
                split_batch) -> Layout:
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
                eval_step=eval_step, export=getattr(model, "merge", None), frozen=frozen,
                config=model.config)


def spmd_layout(args, mesh, model: contract.Model, optimizer, inputs, targets,
                split_batch) -> Layout:
  """No pipeline: every stage in one `jax.jit` program on `mesh`, data-parallel
  or, with `--parallel=fsdp`, with the state sharded too.

  The step runs the stages over all microbatches at once, which gives the
  pipeline's loss and gradients when microbatches hold equal token counts or
  the model's last stage takes the mean of their means itself, and updates
  the replicated or sharded state the way the pipeline does. `inputs`, `targets`
  and `split_batch` are `mpmd_layout`'s: batched leaves lose their
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
                eval_step=eval_step, export=getattr(model, "merge", None), frozen=frozen,
                config=model.config)
