"""The MPMD pipeline step matches single-mesh `jax.jit` in every layout."""

import json
import pathlib
import subprocess
import sys

import jax
import numpy as np
import optax
import pytest

from jax_titan.trainer import adapters, train

REPO = pathlib.Path(__file__).resolve().parents[1]


def _run(script, *flags):
  result = subprocess.run(
      [sys.executable, str(REPO / script), "--platform=cpu", *flags],
      capture_output=True, text=True, check=False,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return result.stdout


def test_two_mesh_spike(tmp_path):
  out = _run("examples/mpmd_two_mesh_spike.py", f"--dump_dir={tmp_path}")
  assert "params match the single-mesh reference" in out


@pytest.mark.parametrize("flags", [
    ["--reduce_grads=step"],
    ["--reduce_grads=microbatch"],
    ["--reduce_grads=step", "--stage_fsdp"],
    ["--reduce_grads=microbatch", "--stage_fsdp"],
    ["--reduce_grads=step", "--tensor_parallel=2"],
    ["--reduce_grads=microbatch", "--stage_fsdp", "--tensor_parallel=2"],
    ["--schedule=GPIPE"],
    ["--num_stages=4", "--microbatches=8"],
    ['--mpmd_options={"mpmd_merge_inferred_after_scheduling": true}'],
    ["--schedule=ZERO_BUBBLE_H1"],
    ["--schedule=CIRCULAR", "--num_stages=4", "--stages_per_mesh=2"],
    ["--stage_layers=3,1"],
    ["--reduce_grads=step", "--per_device=shard_map"],
    ["--reduce_grads=step", "--per_device=shard_map", "--stage_fsdp", "--tensor_parallel=2"],
], ids=lambda f: " ".join(f))
def test_llama_parity(flags):
  out = _run("examples/llama_mpmd_parity.py", *flags)
  assert "matches the single-mesh reference" in out


def test_command_a_frozen_parity():
  """A pipeline step with frozen parameters against one `jax.jit` program."""
  overrides = json.dumps({
      "vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
      "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 32, "num_experts": 8,
      "num_experts_per_tok": 2, "num_shared_experts": 2, "sliding_window": 8,
      "dtype": "float32", "freeze": ["experts", "embed"]})

  def run(parallel):
    args = train.parse_args([
        "--model=command-a-plus", f"--model_overrides={overrides}", "--num_stages=2",
        "--microbatches=2", "--microbatch_size=8", "--seq_len=16", "--data=synthetic",
        "--reduce_grads=microbatch", f"--parallel={parallel}"])
    layout, _ = adapters.build(args, jax.devices(), optax.sgd(0.1))
    (params, opt_states), frozen, losses = layout.init(), layout.frozen(), []
    for step in range(4):
      batch = np.random.default_rng(step).integers(0, 128, (16, 17), dtype=np.int32)
      params, opt_states, loss, _ = layout.step(
          params, opt_states, frozen, *layout.put_batch(batch))
      losses.append(float(loss))
    return losses, layout.export(params)

  (losses, params), (reference, expected) = run("mpmd"), run("dp")
  np.testing.assert_allclose(losses, reference, rtol=1e-5)
  assert sorted(params["layers"]) == sorted(expected["layers"]) and "w_gate" not in params["layers"]
  jax.tree.map(
      lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7), params, expected)
