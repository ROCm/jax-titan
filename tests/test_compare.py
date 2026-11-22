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
PLOTS = importlib.util.find_spec("matplotlib") is not None


def _write(path, losses, *, step_s=1.0, run_dir="", validation=(), hbm=(30.0,) * 8, devices=8,
           attempts=1, **args):
  """A trainer log at 1,000 tokens a step per GPU; with `attempts`, the last steps
  are logged again by an attempt that starts from step 2 with the same arguments."""
  log = records.Log(str(path))
  for attempt in range(attempts):
    log.event("start", main_start=0.0, jax_ready=0.0, num_devices=devices, process_count=1,
              args=ARGS | {"run_dir": run_dir, "log_file": str(path)} | args)
    log.event("layout", tokens_per_step=1000 * devices)
    for step, loss in enumerate(losses):
      if attempt and step < 2:
        continue
      log.write({"step": step, "loss": loss, "grad_norm": 1.0, "step_time_s": step_s,
                 "tokens_per_s": 1000 * devices / step_s, "tokens_per_s_per_gpu": 1000 / step_s,
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
             validation=[(6, 5.22)], hbm=(31.0,) * 8, attempts=2),
      _write(tmp_path / "b.jsonl", [x + 0.5 for x in ref], step_s=2.0, run_dir="b",
             validation=[(6, 5.7)], hbm=[56.1] + [23.0] * 7, model="maxtext:llama3.1-8b"),
  ]
  report = compare.compare(runs, window=3, skip=1)
  a, b = report["configurations"]
  assert (a["name"], b["name"]) == ("llama3.1-8b", "maxtext:llama3.1-8b")
  assert len(a["runs"]) == 2 and len(b["runs"]) == 1
  assert report["windows"] == [(0, 3), (3, 6)] and report["short_runs"] == []
  # A configuration's mean is the mean of its repeats' means.
  assert a["window_loss"] == pytest.approx([50.03 / 6, 33.03 / 6])
  assert a["repeats_differ"] == pytest.approx([0.01, 0.01])
  assert b["window_loss"] == pytest.approx([26.5 / 3, 18 / 3])
  assert b["validation"] == pytest.approx([5.7]) and a["validation"] == pytest.approx([5.21])
  assert a["validation_repeats_differ"] == pytest.approx([0.02])
  # Tokens/s per GPU: the median of the repeats' medians, 1,000 and 800.
  assert a["tokens_per_s_per_gpu"] == pytest.approx(900) and a["speed_spread"] == pytest.approx(2 / 9)
  assert b["tokens_per_s_per_gpu"] == pytest.approx(500)
  text = compare.markdown(report)
  assert "differ in `--model` alone" in text
  assert "| `maxtext:llama3.1-8b` | 8.8333 (+0.4950) | 6.0000 (+0.4950) |" in text
  assert "| Repeats differ by up to | 0.0100 | 0.0100 |" in text
  assert "| Repeats differ by up to | 0.0200 |" in text
  # The run with the highest peak, and a range only where that run's GPUs differ.
  assert "| `llama3.1-8b` (reference) | 8 | 2 | 900 | 22.2% | 1.000 | 31.0 |" in text
  assert "| `maxtext:llama3.1-8b` | 8 | 1 | 500 | - | 0.556 | 23.0 to 56.1 |" in text
  if PLOTS:
    compare.figure(report, tmp_path / "two.png")
    assert (tmp_path / "two.png").stat().st_size > 0
  out = tmp_path / "named.json"
  compare.main([*map(str, runs), "--window=3", "--names=native,adapter", f"--json={out}"])
  assert [c["name"] for c in json.loads(out.read_text())["configurations"]] == [
      "native", "adapter"]


def test_counts_a_window_only_where_a_run_has_all_of_it(tmp_path):
  ref = [10.0, 8.0, 7.0, 6.0, 5.5, 5.0]
  runs = [_write(tmp_path / "a1.jsonl", ref, run_dir="a1", validation=[(3, 7.0), (6, 5.2)]),
          _write(tmp_path / "a2.jsonl", [x + 0.01 for x in ref[:4]], run_dir="a2",
                 validation=[(3, 7.02)])]
  report = compare.compare(runs, window=3, skip=0)
  (a,) = report["configurations"]
  # The cut repeat has steps 0-3: it counts in the first window, not in the second.
  assert a["window_loss"] == pytest.approx([50.03 / 6, 16.5 / 3])
  assert a["repeats_differ"][0] == pytest.approx(0.01) and a["repeats_differ"][1] is None
  assert a["validation"] == pytest.approx([7.01, 5.2])
  assert report["short_runs"] == [str(tmp_path / "a2.jsonl")]
  assert compare.markdown(report).endswith(
      f"Runs that end before step 5: `{tmp_path / 'a2.jsonl'}`.\n")
  # The figure's means are the tables': full windows only, over the repeats that have them.
  means = compare.centred_means([{s: x for s, x in enumerate(ref)},
                                 {s: x + 0.01 for s, x in enumerate(ref[:4])}], window=3)
  assert means == pytest.approx({1: 50.03 / 6, 2: (21 + 21.03) / 6, 3: 18.5 / 3, 4: 16.5 / 3})


def test_names_what_else_differs(tmp_path):
  ref = [3.0, 2.0, 1.5, 1.2]
  runs = [_write(tmp_path / "dp.jsonl", ref, parallel="dp"),
          _write(tmp_path / "fsdp.jsonl", ref, parallel="fsdp", lr=1e-4),
          _write(tmp_path / "dp4.jsonl", ref, parallel="dp", devices=4),
          # A rehearsed failure and another step timeout are the same training: a repeat.
          _write(tmp_path / "dp_again.jsonl", ref, parallel="dp", fail_at_step=3, step_timeout=0)]
  report = compare.compare(runs, window=2, skip=0)
  assert len(report["configurations"][0]["runs"]) == 2
  assert [c["name"] for c in report["configurations"]] == [
      "lr=0.0003 num_devices=8 parallel=dp", "lr=0.0001 num_devices=8 parallel=fsdp",
      "lr=0.0003 num_devices=4 parallel=dp"]
  text = compare.markdown(report)
  assert "The configurations differ in `--lr`, GPUs, `--parallel`:" in text
  assert '| `lr=0.0001 num_devices=8 parallel=fsdp` | `0.0001` | `8` | `"fsdp"` |' in text
  named = compare.compare(runs, window=2, skip=0, names=["dp", "fsdp", "dp4"])
  assert [c["name"] for c in named["configurations"]] == ["dp", "fsdp", "dp4"]
  with pytest.raises(ValueError, match="1 names for 3 configurations"):
    compare.compare(runs, names=["dp"])


def test_refuses_a_log_whose_attempts_trained_otherwise(tmp_path):
  log = _write(tmp_path / "run.jsonl", [3.0, 2.0])

  def relaunch(**args):
    with open(log, "a") as f:
      f.write(json.dumps({"event": "start", "time": 0.0, "main_start": 0.0, "num_devices": 8,
                          "args": ARGS | {"log_file": str(log)} | args}) + "\n")

  relaunch(step_timeout=0)
  compare.load(log)
  relaunch(lr=1e-3)
  with pytest.raises(ValueError, match="attempts that differ in lr"):
    compare.load(log)


def test_reads_the_trainers_records(tmp_path):
  flags = ["--platform=cpu", "--model=llama3-test", "--parallel=dp", "--data=synthetic",
           "--seq_len=32", "--microbatches=1", "--microbatch_size=8", "--steps=6",
           "--eval_every=3", "--eval_batches=1", "--power_sample_s=0"]
  for name in ("r1", "r2"):
    result = subprocess.run(
        [sys.executable, "-m", "jax_titan.trainer.train", *flags,
         f"--run_dir={tmp_path / name}"],
        # os.environ, not the process environment: see conftest's ROCPROFILER note.
        capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  out = tmp_path / "report.json"
  figure = [f"--figure={tmp_path / 'loss.png'}"] if PLOTS else []
  compare.main([str(tmp_path / "r1"), str(tmp_path / "r2"), "--window=3", "--skip=1",
                "--names=native", f"--json={out}", *figure])
  (config,) = json.loads(out.read_text())["configurations"]
  assert config["name"] == "native" and config["args"]["num_devices"] == 8
  losses = [r["loss"] for r in records.steps(records.read(tmp_path / "r1" / "train.jsonl"))]
  assert config["window_loss"] == pytest.approx([sum(losses[:3]) / 3, sum(losses[3:]) / 3])
  evals = records.events(records.read(tmp_path / "r1" / "train.jsonl"), "eval")
  assert config["validation"] == pytest.approx([e["loss"] for e in evals], rel=1e-6)
  assert len(config["runs"]) == 2 and config["tokens_per_s_per_gpu"] > 0
  if PLOTS:
    assert (tmp_path / "loss.png").stat().st_size > 0
