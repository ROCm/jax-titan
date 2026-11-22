"""The comparison of training runs groups repeats, sets each configuration against the
reference and reads what the trainer writes."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

from jax_titan.observability import compare
from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]
ARGS = {"model": "llama3.1-8b", "parallel": "fsdp", "seed": 0, "steps": 6, "lr": 3e-4}


def _write(path, losses, *, step_s=1.0, run_dir="", validation=(), hbm=(30.0,) * 8, **args):
  """A trainer log of 8 GPUs at 8,000 tokens a step."""
  log = records.Log(str(path))
  log.event("start", main_start=0.0, jax_ready=0.0, num_devices=8, process_count=1,
            args=ARGS | {"run_dir": run_dir, "log_file": str(path)} | args)
  log.event("layout", tokens_per_step=8000)
  for step, loss in enumerate(losses):
    log.write({"step": step, "loss": loss, "grad_norm": 1.0, "step_time_s": step_s,
               "tokens_per_s": 8000 / step_s, "tokens_per_s_per_gpu": 1000 / step_s,
               "time": float(step)})
    if step == 0:
      log.event("memory", step=0, peak_hbm_gib=list(hbm))
  for step, loss in validation:
    log.event("eval", step=step, loss=loss, seconds=1.0)
  log.close()
  return path


def test_sets_each_configuration_against_the_reference(tmp_path):
  ref = [10.0, 8.0, 7.0, 6.0, 5.5, 5.0]
  runs = [
      _write(tmp_path / "a1.jsonl", ref, run_dir="a1", validation=[(6, 5.2)]),
      _write(tmp_path / "a2.jsonl", [x + 0.01 for x in ref], step_s=1.25, run_dir="a2",
             validation=[(6, 5.22)]),
      _write(tmp_path / "b.jsonl", [x + 0.5 for x in ref], step_s=2.0, run_dir="b",
             validation=[(6, 5.7)], hbm=[56.1] + [23.0] * 7, model="maxtext:llama3.1-8b"),
  ]
  report = compare.compare(runs, window=3, skip=1)
  a, b = report["configurations"]
  assert (a["name"], b["name"]) == ("llama3.1-8b", "maxtext:llama3.1-8b")
  assert len(a["runs"]) == 2 and len(b["runs"]) == 1
  assert report["windows"] == [(0, 3), (3, 6)]
  # A configuration's mean is the mean of its repeats' means.
  assert a["window_loss"] == pytest.approx([50.03 / 6, 33.03 / 6])
  assert a["repeats_differ"] == pytest.approx([0.01, 0.01])
  assert b["window_loss"] == pytest.approx([26.5 / 3, 18 / 3])
  assert b["validation"] == pytest.approx([5.7]) and a["validation"] == pytest.approx([5.21])
  # Tokens/s per GPU: the median of the repeats' medians, 1,000 and 800.
  assert a["tokens_per_s_per_gpu"] == pytest.approx(900) and a["speed_spread"] == pytest.approx(2 / 9)
  assert b["tokens_per_s_per_gpu"] == pytest.approx(500)
  text = compare.markdown(report)
  assert "differ in `--model` alone" in text
  assert "| `maxtext:llama3.1-8b` | 8.8333 (+0.4950) | 6.0000 (+0.4950) |" in text
  assert "| `llama3.1-8b` (reference) | 8 | 2 | 900 | 22.2% | 1.000 | 30.0 |" in text
  assert "| `maxtext:llama3.1-8b` | 8 | 1 | 500 | - | 0.556 | 23.0 to 56.1 |" in text


def test_names_what_else_differs(tmp_path):
  ref = [3.0, 2.0, 1.5, 1.2]
  runs = [_write(tmp_path / "dp.jsonl", ref, parallel="dp"),
          _write(tmp_path / "fsdp.jsonl", ref, parallel="fsdp", lr=1e-4)]
  report = compare.compare(runs, window=2, skip=0)
  assert [c["name"] for c in report["configurations"]] == [
      "lr=0.0003 parallel=dp", "lr=0.0001 parallel=fsdp"]
  text = compare.markdown(report)
  assert "also differ in `--lr`, `--parallel`" in text
  named = compare.compare(runs, window=2, skip=0, names=["dp", "fsdp"])
  assert [c["name"] for c in named["configurations"]] == ["dp", "fsdp"]
  with pytest.raises(ValueError, match="1 names for 2 configurations"):
    compare.compare(runs, names=["dp"])
  assert "| `lr=0.0001 parallel=fsdp` | `0.0001` | `\"fsdp\"` |" in text


def test_reads_the_trainers_records(tmp_path):
  flags = ["--platform=cpu", "--model=llama3-test", "--parallel=dp", "--data=synthetic",
           "--seq_len=32", "--microbatches=1", "--microbatch_size=8", "--steps=6",
           "--eval_every=3", "--eval_batches=1"]
  for name in ("r1", "r2"):
    result = subprocess.run(
        [sys.executable, "-m", "jax_titan.trainer.train", *flags,
         f"--run_dir={tmp_path / name}"],
        # os.environ, not the process environment: see conftest's ROCPROFILER note.
        capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  out = tmp_path / "report.json"
  figure = {} if importlib.util.find_spec("matplotlib") is None else {"png": tmp_path / "loss.png"}
  compare.main([str(tmp_path / "r1"), str(tmp_path / "r2"), "--window=3", "--skip=1",
                f"--json={out}", *(f"--figure={p}" for p in figure.values())])
  (config,) = json.loads(out.read_text())["configurations"]
  losses = [r["loss"] for r in records.steps(records.read(tmp_path / "r1" / "train.jsonl"))]
  assert config["window_loss"] == pytest.approx([sum(losses[:3]) / 3, sum(losses[3:]) / 3])
  evals = records.events(records.read(tmp_path / "r1" / "train.jsonl"), "eval")
  assert config["validation"] == pytest.approx([e["loss"] for e in evals], rel=1e-6)
  assert len(config["runs"]) == 2 and config["tokens_per_s_per_gpu"] > 0
  for path in figure.values():
    assert path.stat().st_size > 0
