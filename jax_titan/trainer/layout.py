"""What a trainer builder returns: how training state is placed and stepped,
and what one step processes."""

import dataclasses
from typing import Any, Callable


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
