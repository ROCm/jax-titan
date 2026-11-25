"""The names of `--model`, the modules that implement them, their builders and
the contract the trainer reads a model by."""

import importlib

import pytest

from jax_titan import models
from jax_titan.mlperf import run as mlperf_run
from jax_titan.models import command_a
from jax_titan.models import contract
from jax_titan.models import flux
from jax_titan.models import ideogram
from jax_titan.models import llama
from jax_titan.trainer import builders
from jax_titan.trainer import train


def test_every_name_has_a_configuration_and_every_configuration_a_name():
  for module in (llama, flux):
    short = module.__name__.rpartition(".")[2]
    assert {name for name, m in models.MODELS.items() if m == short} == set(module.FLAVORS)
  assert set(mlperf_run.MODELS.values()) <= set(models.MODELS)


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
      llama.Model(llama.FLAVORS["llama3-test"]),
      flux.Model(flux.FLAVORS["flux-schnell"]),
      ideogram.Model(ideogram.IdeogramConfig(), num_stages=2),
      command_a.Model(command_a.CommandAConfig()),
  ]
  for model in made:
    assert isinstance(model, contract.Model), type(model)
