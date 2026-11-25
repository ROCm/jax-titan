"""The configuration of a run's model.

The module of a JAX-Titan model has a frozen dataclass and `CONFIGS`, its
configurations by the names `--model` takes; `resolve` gives the one of a run.
A model of another repository keeps the repository's own configuration, and
its `AdapterConfig` refers to that.
"""

import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class AdapterConfig:
  """A reference to the configuration of a model of another repository: the
  `source` (`maxtext`, ...), the model's `name` there, which follows the colon
  in `--model`, and the `overrides` the source gets, as pairs."""

  source: str
  name: str
  overrides: tuple = ()


def _field_value(hint, value):
  """A JSON value as the field of `hint` takes it."""
  if isinstance(value, list):
    return tuple(value)
  if isinstance(value, dict):
    kinds = [t for t in (hint, *typing.get_args(hint)) if dataclasses.is_dataclass(t)]
    return kinds[0](**value) if kinds else value
  return value


def resolve(module, name: str, init_from: str = "", overrides: dict | None = None):
  """The configuration of `--model=name`, a model of `module`: the one of the
  module's `CONFIGS`, or that of the published weights `init_from` names, with
  `overrides` of its fields.

  A JSON list becomes a tuple and a JSON object the dataclass of its field. A
  key that is not a field is refused.
  """
  from jax_titan.models import weights  # pylint: disable=g-import-not-at-top

  if init_from and hasattr(module, "pretrained_config") and weights.is_safetensors(init_from):
    cfg = module.pretrained_config(init_from)
  else:
    cfg = module.CONFIGS[name]
  overrides = overrides or {}
  fields = [f.name for f in dataclasses.fields(cfg)]
  unknown = sorted(set(overrides) - set(fields))
  if unknown:
    raise ValueError(
        f"--model_overrides: {name} has no configuration field {', '.join(unknown)}; it has "
        f"{', '.join(fields)}")
  hints = typing.get_type_hints(type(cfg))
  return dataclasses.replace(cfg, **{k: _field_value(hints[k], v) for k, v in overrides.items()})
