"""MaxText and MaxDiffusion models train through the trainer and resume exactly."""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]
FLAGS = [
    "--platform=cpu", "--microbatches=4", "--microbatch_size=4",
    "--steps=4", "--warmup_steps=1", "--checkpoint_every=2",
]
MAXTEXT_OVERRIDES = {
    "base_emb_dim": 64, "base_mlp_dim": 128, "base_num_query_heads": 4,
    "base_num_kv_heads": 2, "head_dim": 16, "base_num_decoder_layers": 4,
    "vocab_size": 256,
}
MODELS = {
    "maxtext:llama3.1-8b": [
        "--data=synthetic", "--seq_len=32",
        "--model_overrides=" + json.dumps(MAXTEXT_OVERRIDES),
    ],
    "maxdiffusion:flux-dev": [
        "--resolution=64", "--text_tokens=8",
        "--model_overrides=" + json.dumps({
            "in_channels": 16, "num_layers": 2, "num_single_layers": 2,
            "attention_head_dim": 16, "num_attention_heads": 2,
            "joint_attention_dim": 32, "pooled_projection_dim": 16,
            "axes_dims_rope": [4, 6, 6],
        }),
    ],
    "maxdiffusion:wan2.1-t2v-1.3b": [
        "--resolution=64", "--frames=5", "--text_tokens=8",
        "--model_overrides=" + json.dumps({
            "num_attention_heads": 2, "attention_head_dim": 24, "in_channels": 4,
            "out_channels": 4, "text_dim": 32, "freq_dim": 32, "ffn_dim": 96,
            "num_layers": 2, "rope_max_seq_len": 32,
        }),
    ],
}


def _run(model, log, *flags):
  # os.environ, not the process environment: see conftest's ROCPROFILER note.
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", *FLAGS, f"--model={model}",
       *MODELS[model], f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r for r in records.steps(records.read(log))}, result.stdout


def _train(model, ckpt_dir, log, *flags):
  records, out = _run(model, log, f"--checkpoint_dir={ckpt_dir}", *flags)
  return {step: r["loss"] for step, r in records.items()}, out


@pytest.mark.parametrize("parallel", ["dp", "fsdp"])
@pytest.mark.parametrize("model", sorted(MODELS))
def test_resume_is_exact(model, parallel, tmp_path):
  provider = model.partition(":")[0]
  if importlib.util.find_spec(provider) is None:
    pytest.skip(f"needs {provider}")
  full, _ = _train(model, tmp_path / "full", tmp_path / "full.jsonl", f"--parallel={parallel}")
  (tmp_path / "resumed").mkdir()
  shutil.copytree(tmp_path / "full" / "2", tmp_path / "resumed" / "2")
  resumed, out = _train(
      model, tmp_path / "resumed", tmp_path / "resumed.jsonl", f"--parallel={parallel}")
  assert "restored step 2" in out
  assert sorted(resumed) == [2, 3]
  for step in (2, 3):
    assert resumed[step] == full[step], (step, resumed[step], full[step])


def test_maxtext_starts_from_a_parameter_checkpoint(tmp_path):
  if importlib.util.find_spec("maxtext") is None:
    pytest.skip("needs maxtext")
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top
  from maxtext.common import checkpointing  # pylint: disable=g-import-not-at-top

  # The trainer initializes from seed 0: the checkpoint holds seed 1's parameters.
  cfg = mt.config("llama3.1-8b", max_target_length=32, remat_policy="full", **MAXTEXT_OVERRIDES)
  model = mt.AdapterModel(cfg, mt.mesh(cfg, jax.devices()[:4]))
  (full,) = model.init(1)
  path = str(tmp_path / "params")
  checkpointing.save_params_to_path(path, {"params": full})
  jax.tree.map(np.testing.assert_array_equal, model.load(path)[0], full)

  loaded, _ = _run(
      "maxtext:llama3.1-8b", tmp_path / "loaded.jsonl", "--parallel=dp", f"--init_from={path}")
  # The trainer's first synthetic batch.
  batch = np.random.default_rng((0, 0)).integers(
      0, MAXTEXT_OVERRIDES["vocab_size"], (16, 33), dtype=np.int32)
  (fn,) = model.stage_fns()
  np.testing.assert_allclose(loaded[0]["loss"], jax.jit(fn)(full, batch[:, :-1], batch[:, 1:]),
                             rtol=1e-5)
