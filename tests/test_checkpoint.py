"""Resuming from an Orbax checkpoint continues training exactly."""

import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]
FLAGS = [
    "--platform=cpu", "--model=llama3-test", "--parallel=dp", "--data=synthetic", "--seq_len=64",
    "--microbatches=4", "--microbatch_size=4", "--steps=6", "--warmup_steps=2",
    "--checkpoint_every=3",
]


def _run(ckpt_dir, log, *flags):
  return subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", *FLAGS,
       f"--checkpoint_dir={ckpt_dir}", f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO,
  )


def _train(ckpt_dir, log, *flags):
  result = _run(ckpt_dir, log, *flags)
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r["loss"] for r in records.steps(records.read(log))}, result.stdout


def _resume_from_step_3(tmp_path, name, *flags):
  full, _ = _train(tmp_path / "full", tmp_path / "full.jsonl")
  (tmp_path / name).mkdir()
  shutil.copytree(tmp_path / "full" / "3", tmp_path / name / "3")
  resumed, out = _train(tmp_path / name, tmp_path / f"{name}.jsonl", *flags)
  assert "restored step 3" in out
  assert sorted(resumed) == [3, 4, 5]
  return full, resumed


def test_resume_is_exact(tmp_path):
  full, resumed = _resume_from_step_3(tmp_path, "resumed")
  for step in (3, 4, 5):
    assert resumed[step] == full[step], (step, resumed[step], full[step])


def test_resume_after_a_synthetic_warmup_is_exact(tmp_path):
  full, resumed = _resume_from_step_3(tmp_path, "warm", "--synthetic_warmup=2")
  for step in (3, 4, 5):
    assert resumed[step] == full[step], (step, resumed[step], full[step])


def test_restore_into_fsdp(tmp_path):
  full, resumed = _resume_from_step_3(tmp_path, "fsdp", "--parallel=fsdp")
  for step in (3, 4, 5):
    np.testing.assert_allclose(resumed[step], full[step], rtol=1e-5)


def test_a_run_resumes_with_the_configuration_it_was_written_with(tmp_path):
  full, _ = _train(tmp_path / "full", tmp_path / "full.jsonl")
  (layout,) = records.events(records.read(tmp_path / "full.jsonl"), "layout")
  assert layout["config"]["num_layers"] == 4
  # Another configuration of the same shapes is refused, with the field that differs.
  for name in ("other", "old"):
    (tmp_path / name).mkdir()
    shutil.copytree(tmp_path / "full" / "3", tmp_path / name / "3")
  refused = _run(tmp_path / "other", tmp_path / "other.jsonl",
                 "--model_overrides=" + json.dumps({"norm_eps": 1e-6}))
  assert refused.returncode != 0 and "another configuration" in refused.stderr
  assert "norm_eps (1e-05, now 1e-06)" in refused.stderr
  # A checkpoint written before configurations were recorded holds none, and resumes.
  meta = tmp_path / "old" / "3" / "meta" / "metadata"
  saved = json.loads(meta.read_text())
  assert saved.pop("config") == layout["config"]
  meta.write_text(json.dumps(saved))
  resumed, out = _train(tmp_path / "old", tmp_path / "old.jsonl")
  assert "restored step 3" in out
  for step in (3, 4, 5):
    assert resumed[step] == full[step], (step, resumed[step], full[step])

