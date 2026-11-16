"""Ideogram 4 stages match the Ideogram branch of MaxDiffusion, and its layouts match each other."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = {
    "emb_dim": 64, "num_heads": 2, "in_channels": 16, "llm_features_dim": 48, "adanln_dim": 32,
    "intermediate_size": 128, "num_layers": 4,
}
FLAGS = [
    "--platform=cpu", "--model=maxdiffusion:ideogram4", "--resolution=64", "--text_tokens=8",
    "--model_overrides=" + json.dumps(CONFIG | {"dtype": "float32"}), "--microbatches=4",
    "--microbatch_size=4",
    "--steps=4", "--warmup_steps=1",
]


def _has_ideogram():
  try:
    return importlib.util.find_spec("maxdiffusion.models.ideogram") is not None
  except ModuleNotFoundError:
    return False


pytestmark = pytest.mark.skipif(not _has_ideogram(), reason="needs MaxDiffusion's Ideogram branch")


@pytest.mark.parametrize("num_stages,blocks", [(1, None), (2, None), (3, (1, 2, 1))])
def test_stages_match_the_transformer(num_stages, blocks):
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import ideogram as ig  # pylint: disable=g-import-not-at-top

  cfg = ig.Ideogram4Config(**CONFIG)
  model = ig.Model(cfg, num_stages=num_stages, blocks=blocks, dtype=jnp.float32)
  transformer = ig.Ideogram4Transformer(nnx.Rngs(0), cfg, dtype=jnp.float32)
  stage_params = model.split(nnx.state(transformer, nnx.Param).to_pure_dict())
  for a, b in zip(jax.tree.leaves(stage_params), jax.tree.leaves(model.init(0))):
    np.testing.assert_array_equal(a, b)

  batch, side, text_tokens = 2, 4, 8
  rng = np.random.default_rng(0)
  ids = ig.sequence_ids(side, text_tokens)
  # The second sample's text is left-padded by three positions.
  ids = {k: np.stack([v, v]) for k, v in ids.items()}
  ids["segment_ids"][1, :3] = ig.constants.SEQUENCE_PADDING_INDICATOR
  ids["indicator"][1, :3] = 0
  x = {
      "img": rng.normal(size=(batch, side * side, cfg.in_channels)).astype(np.float32),
      "txt": rng.normal(size=(batch, text_tokens, cfg.llm_features_dim)).astype(np.float32),
      "timestep": rng.uniform(size=batch).astype(np.float32),
      **ids,
  }
  # The transformer takes both inputs over the whole sequence, zero where the other one is.
  llm = np.pad(x["txt"], ((0, 0), (0, side * side), (0, 0)))
  latents = np.pad(x["img"], ((0, 0), (text_tokens, 0), (0, 0)))
  ref = jax.jit(lambda *a: transformer(*a))(
      llm, latents, x["timestep"], x["position_ids"], x["segment_ids"], x["indicator"])

  def chain(stage_params, x):
    for i in range(num_stages):
      x = model.forward(i, stage_params[i], x)
    return x

  pred = jax.jit(chain)(stage_params, x)
  np.testing.assert_allclose(np.asarray(pred), np.asarray(ref), rtol=1e-5, atol=1e-5)


def _run(log, *flags):
  # os.environ, not the process environment: see conftest's ROCPROFILER note.
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", *FLAGS, f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r for r in records.steps(records.read(log))}


@pytest.mark.parametrize("flags", [["--parallel=fsdp"], ["--parallel=mpmd", "--num_stages=2"]],
                         ids=["fsdp", "pipeline"])
def test_layout_matches_data_parallel_reference(flags, tmp_path):
  reference = _run(tmp_path / "dp.jsonl", "--parallel=dp")
  layout = _run(tmp_path / "layout.jsonl", *flags)
  for step in range(4):
    assert layout[step]["loss"] == pytest.approx(reference[step]["loss"], rel=1e-5), step
    assert layout[step]["grad_norm"] == pytest.approx(reference[step]["grad_norm"], rel=1e-4), step
