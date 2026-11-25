"""What the trainer takes a model to be."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
  """A staged model, as the layouts of `jax_titan.trainer.layout` read it.

  Besides the members below, a layout uses these of a model that has them:
  `load(path)`, for `--init_from`, the parameters of each stage from saved
  ones; `merge(stage_params)`, for `--save_params`, the parameters as `load`
  reads them; and, of a model that trains only part of its parameters,
  `is_frozen(path)` with `dtype`, and `place(path, targets)`.

  The stage parameters of such a model are nested dicts, and `is_frozen(path)`
  tells, for the tuple of dict keys that leads to a parameter in a stage's
  dicts, whether it stays as it is. Frozen parameters are no part of the
  trained state: they get no gradient and no optimizer state, are held in the
  model's compute dtype, its attribute `dtype`, and neither a checkpoint nor
  `--save_params` holds them, so a run that resumes reads them again from
  where the run took them. A model that can read weights straight onto devices
  has `place(path, targets)`, which `--init_from` then uses for published
  weights, a directory of safetensors files: a checkpoint larger than the
  host's memory loads this way. `targets` is one tree of ShapeDtypeStructs
  with shardings per stage, all of a stage's parameters or its frozen ones
  alone.
  """

  # The model's configuration, a frozen dataclass: the dataclass of a JAX-Titan
  # model's module, or an `AdapterConfig`.
  config: Any

  def init(self, seed: int) -> tuple:
    """The parameters of each stage, initialized."""

  def abstract(self, seed: int) -> tuple:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""

  def stage_fns(self) -> list[Callable]:
    """One function per stage, `fn(params, x)`; the last one also takes the
    targets, `fn(params, x, targets)`, and returns the loss."""

  def logical_axes(self, path, leaf) -> Sequence | None:
    """The logical axes of a parameter, or of an optimizer-state leaf mirroring
    one, which `--parallel=fsdp` shards along (`jax_titan.parallel.tree_specs`)."""

  def decays(self, path, leaf) -> bool:
    """Whether the parameter at `path` in a stage's parameters takes weight decay."""
