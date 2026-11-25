"""The names of `--model`, the modules that implement them, their
configurations and builders, and the contract the trainer reads a model by."""

import dataclasses
import importlib

import optax
import pytest

from jax_titan import models
from jax_titan.mlperf import run as mlperf_run
from jax_titan.models import command_a
from jax_titan.models import config
from jax_titan.models import contract
from jax_titan.models import flux
from jax_titan.models import ideogram
from jax_titan.models import llama
from jax_titan.trainer import builders
from jax_titan.trainer import train


def test_every_name_has_a_configuration_and_every_configuration_a_name():
  for short in sorted(set(models.MODELS.values())):
    module = importlib.import_module(f"jax_titan.models.{short}")
    assert {name for name, m in models.MODELS.items() if m == short} == set(module.CONFIGS)
  assert set(mlperf_run.MODELS.values()) <= set(models.MODELS)


def test_resolve_overrides_fields_and_refuses_what_is_none():
  cfg = config.resolve(llama, "llama3-test", overrides={"num_layers": 2})
  assert cfg == dataclasses.replace(llama.CONFIGS["llama3-test"], num_layers=2)
  # A JSON list is a tuple, and a JSON object the dataclass of its field.
  rope = config.resolve(flux, "flux-dev", overrides={"axes_dims_rope": [4, 6, 6]})
  assert rope.axes_dims_rope == (4, 6, 6) and rope.guidance_embeds
  scaled = config.resolve(llama, "llama3-test", overrides={"rope_scaling": {"factor": 4.0}})
  assert scaled.rope_scaling == llama.Llama31RopeScaling(factor=4.0)
  with pytest.raises(ValueError, match="no configuration field dtype"):
    config.resolve(llama, "llama3-test", overrides={"dtype": "float32"})


@pytest.mark.parametrize("flags,error", [
    (["--model=llama3-test", "--freeze=embed"], "takes no --freeze"),
    (["--model=flux-dev", "--timestep_mean=0.5"], "takes no --timestep_mean"),
    (["--model=ideogram4", "--image_tiles=2"], "takes no --image_tiles"),
])
def test_refuses_the_options_of_other_models(flags, error):
  args = train.parse_args([*flags, "--parallel=dp"])
  with pytest.raises(NotImplementedError, match=error):
    builders.build(args, [None], lambda decays: optax.sgd(0.0))


def test_every_module_has_a_builder():
  assert set(builders.BUILDERS) == set(models.MODELS.values()) | set(models.SOURCES.values())
  for module in models.MODELS.values():
    importlib.import_module(f"jax_titan.models.{module}")


def test_module_of():
  assert models.module_of("llama3.1-8b") == "llama"
  assert models.module_of("maxtext:llama3.1-8b") == "adapters.maxtext"
  for name in ("llama", "maxtext", "other:llama3.1-8b", ""):
    with pytest.raises(ValueError, match="unknown model"):
      models.module_of(name)


def test_the_help_of_model_names_every_model():
  with pytest.raises(SystemExit), pytest.MonkeyPatch.context() as patch:
    printed = []
    patch.setattr("sys.stdout.write", printed.append)
    train.parse_args(["--help"])
  text = " ".join("".join(printed).split())
  for name in (*models.MODELS, *models.SOURCES):
    assert name in text, name


def test_models_meet_the_contract():
  made = [
      llama.Model(llama.CONFIGS["llama3-test"]),
      flux.Model(flux.CONFIGS["flux-schnell"]),
      ideogram.Model(ideogram.CONFIGS["ideogram4"], num_stages=2),
      command_a.Model(command_a.CONFIGS["command-a-plus"]),
  ]
  for model in made:
    assert isinstance(model, contract.Model), type(model)
