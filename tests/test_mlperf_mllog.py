"""MLPerf trials: the runner's flags, MLLOG from the trainer's records, which
MLCommons' compliance checker accepts, and the order MLPerf times a run in:
warm-up before run_start, every token read after it."""

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from jax_titan.data import megatron
from jax_titan.mlperf import contract
from jax_titan.mlperf import run
from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "jax_titan/mlperf/configs/llama31_8b_mi355x_1x8_bf16.json"

try:
  import mlperf_logging  # pylint: disable=unused-import
  HAVE_MLLOG = True
except ImportError:
  HAVE_MLLOG = False
needs_mllog = pytest.mark.skipif(
    not HAVE_MLLOG, reason="needs mlperf_logging: pip install 'jax-titan[mlperf]'")


def _config(**changes):
  config = json.loads(CONFIG.read_text())
  config.update(changes)
  return config


def _mllog(path):
  return [json.loads(line.split(":::MLLOG ", 1)[1])
          for line in pathlib.Path(path).read_text().splitlines() if ":::MLLOG " in line]


def _args(argv):
  from jax_titan.trainer import train  # pylint: disable=g-import-not-at-top
  return vars(train.parse_args(argv))


# --- the runner ---


def test_runner_derives_the_trainer_flags_from_the_contract():
  _, argv, hparams = run.prepare(_config(), data_dir="/data", seed=7)
  flags = dict(f[2:].split("=", 1) for f in argv)
  assert flags["parallel"] == "fsdp" and flags["ops"] == "attention=te"
  assert flags["model"] == "llama3.1-8b"
  assert flags["seq_len"] == "8192" and flags["steps"] == "1200000"
  assert flags["lr"] == "0.0008" and flags["warmup_steps"] == "128"
  assert flags["lr_schedule"] == "nemo_cosine" and flags["lr_decay_steps"] == "1199872"
  assert (flags["adam_b1"], flags["adam_b2"], flags["adam_eps"]) == ("0.9", "0.95", "1e-05")
  assert flags["weight_decay"] == "0.1" and flags["max_grad_norm"] == "1.0"
  assert flags["data"] == "/data/c4-train.en_6_text_document"
  assert flags["eval_data"] == "/data/c4-validation-91205-samples.en_text_document"
  assert flags["eval_every"] == "384" and flags["eval_samples"] == "1024"
  assert flags["eval_batch_size"] == "32" and flags["target_eval_loss"] == "3.3"
  assert flags["seed"] == "7" and flags["synthetic_warmup"] == "2"
  assert hparams == contract.LLAMA31_8B.hyperparameters(
      global_batch_size=32, base_lr=8e-4, warmup_steps=128)


def test_runner_draws_a_seed():
  seeds = {dict(f[2:].split("=", 1) for f in run.prepare(_config(), data_dir="/d")[1])["seed"]
           for _ in range(4)}
  assert len(seeds) > 1


@pytest.mark.parametrize("trainer, extra, message", [
    (["--adam_eps=1e-8"], [], "--adam_eps"),
    ([], ["--steps", "10"], "--steps"),
    (["--model=llama3-test"], [], "--model"),
    (["--init_from=/ckpt"], [], "--init_from"),
    (["--eval_batches=2", "--lr=1"], [], "--eval_batches, --lr"),
    (["--parallel=fsdp", "--microbatches=2", "--microbatch_size=32"], [],
     "is not the global batch of 32"),
])
def test_runner_refuses_what_the_contract_sets(trainer, extra, message):
  with pytest.raises(contract.ContractError, match=message):
    run.prepare(_config(trainer=trainer), data_dir="/data", seed=1, extra=extra)


def test_runner_refuses_trials_it_cannot_run():
  with pytest.raises(KeyError, match="no contract"):
    run.prepare(_config(benchmark="bert"), data_dir="/data", seed=1)
  with pytest.raises(contract.ContractError, match="no model for gpt_oss_20b"):
    run.prepare(_config(benchmark="gpt_oss_20b"), data_dir="/data", seed=1)
  with pytest.raises(contract.ContractError, match="opt_base_learning_rate"):
    run.prepare(_config(base_lr=-1.0), data_dir="/data", seed=1)
  with pytest.raises(ValueError, match="submission needs"):
    run.prepare(_config(submission={"org": "AMD"}), data_dir="/data", seed=1)


def test_dry_run_prints_the_trial_and_refuses_contract_flags():
  command = [sys.executable, "-m", "jax_titan.mlperf.run", str(CONFIG), "--data_dir=/data",
             "--seed=7", "--dry_run"]
  result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=REPO)
  assert result.returncode == 0, result.stderr[-4000:]
  trial = json.loads(result.stdout)
  assert "--target_eval_loss=3.3" in trial["trainer"]
  assert trial["hyperparameters"]["opt_learning_rate_decay_steps"] == 1_199_872
  result = subprocess.run(command + ["--adam_eps=1e-8"], capture_output=True, text=True,
                          check=False, cwd=REPO)
  assert result.returncode == 1 and "the contract sets --adam_eps" in result.stderr


# --- MLLOG from records ---


def _trial_records(args, losses, reached, gbs=32, every=384):
  """The records of a trial validating every `every` steps with these losses."""
  t = 1_000_000.0
  out = [{"event": "start", "time": t + 5, "main_start": t, "num_devices": 8, "args": args},
         {"event": "warmup", "time": t + 90, "seconds": 60.0, "steps": 2},
         {"event": "run_start", "time": t + 91, "step": 0}]
  now, step = t + 91, 0
  for loss in losses:
    for _ in range(every):
      now += 1.0
      out.append({"step": step, "loss": 5.0, "step_time_s": 1.0, "time": now})
      step += 1
    now += 10.0
    out.append({"event": "eval", "time": now, "step": step, "loss": loss, "seconds": 10.0,
                "samples": step * gbs, "eval_samples": 1024})
  out.append({"event": "end", "time": now + 1, "status": "completed", "steps_done": step,
              "target_reached": reached})
  return out


@needs_mllog
@pytest.mark.parametrize("losses, reached", [([3.6, 3.29], True), ([3.6, 3.5], False)])
def test_mllog_of_a_trial_passes_the_compliance_checker(tmp_path, losses, reached):
  config = _config()
  bench, argv, hparams = run.prepare(config, data_dir="/data", seed=1234)
  log = tmp_path / "result_0.txt"
  listener = run.listener(config, bench, hparams, config_filename=CONFIG.name,
                          mllog_path=str(log))
  for record in _trial_records(_args(argv), losses, reached):
    listener(record)
  result = subprocess.run(
      [sys.executable, "-m", "mlperf_logging.compliance_checker", "--usage", "training",
       "--ruleset", "6.1.0", "--log_output", str(tmp_path / "checker.txt"), str(log)],
      capture_output=True, text=True, check=False)
  # A trial short of the target fails the checker: its log says so.
  assert (result.returncode == 0) == reached, result.stdout + result.stderr
  lines = _mllog(log)
  keys = [line["key"] for line in lines]
  assert keys.count("block_start") == 2 and keys.count("eval_accuracy") == 2
  stop = lines[keys.index("run_stop")]["metadata"]
  assert stop["samples_count"] == 768 * 32
  assert stop["status"] == ("success" if reached else "aborted")
  assert lines[keys.index("train_samples")]["value"] == 768 * 32
  assert lines[keys.index("seed")]["value"] == 1234
  assert lines[keys.index("micro_batch_size")]["value"] == 4
  assert lines[keys.index("init_start")]["time_ms"] == 1_000_000_000
  assert lines[keys.index("run_start")]["time_ms"] == 1_000_091_000


@needs_mllog
def test_mllog_refuses_hyperparameters_the_run_does_not_use():
  config = _config()
  bench, argv, hparams = run.prepare(config, data_dir="/data", seed=1)
  listener = run.listener(config, bench, hparams, config_filename=CONFIG.name)
  args = _args(argv + ["--adam_eps=1e-8", "--lr_schedule=cosine", "--eval_every=100",
                       "--target_eval_loss=3.4", "--ops=attention=te,linear=te_fp8"])
  with pytest.raises(ValueError) as error:
    listener({"event": "start", "time": 1.0, "main_start": 0.0, "num_devices": 8,
              "args": args})
  for departure in ("opt_adamw_epsilon", "lr_schedule", "eval_every", "target_eval_loss",
                    "precision linear=bfloat16"):
    assert departure in str(error.value)
  with pytest.raises(ValueError, match="precision attn='fp7'"):
    run.listener(_config(precision={"linear": "bfloat16", "attn": "fp7", "comm": "bfloat16"}),
                 bench, hparams, config_filename=CONFIG.name)


# --- a trial on CPU ---

_DRIVER = """
import dataclasses, json, sys, time
from jax_titan.data import megatron
from jax_titan.mlperf import contract, run
from jax_titan.trainer import train

data_dir, log_file, mllog_file, reads_file, target = sys.argv[1:]
reads = []
get = megatron.IndexedDataset.get

def timed_get(self, *args, **kwargs):
  reads.append(time.time())
  return get(self, *args, **kwargs)

megatron.IndexedDataset.get = timed_get
run.MODELS["llama31_8b"] = "llama3-test"
toy = dataclasses.replace(contract.LLAMA31_8B, quality_target=float(target),
                          eval_interval_samples=32, eval_samples=16, sequence_length=32,
                          max_steps=6)
config = {
    "benchmark": "llama31_8b",
    "submission": {"org": "test", "platform": "cpu", "division": "closed", "status": "onprem"},
    "global_batch_size": 16, "base_lr": 1e-3, "warmup_steps": 1,
    "precision": {"linear": "bfloat16", "attn": "bfloat16", "comm": "bfloat16"},
    "eval_batch_size": 8, "synthetic_warmup": 1,
    "trainer": ["--platform=cpu", "--parallel=dp", "--microbatches=2", "--microbatch_size=8",
                "--step_timeout=0", "--log_file=" + log_file],
}
_, argv, hparams = run.prepare(config, data_dir=data_dir, seed=5, contract=toy)
train.main(argv, listeners=[run.listener(config, toy, hparams, config_filename="toy.json",
                                         mllog_path=mllog_file)])
with open(reads_file, "w") as f:
  json.dump(reads, f)
"""

_TIMELINE = ("init_start", "init_stop", "run_start", "block_start", "block_stop", "eval_start",
             "eval_accuracy", "eval_stop", "run_stop")


@needs_mllog
@pytest.mark.parametrize("target, evals, status", [
    (100.0, [32], "success"),
    (0.001, [32, 64, 96], "aborted"),
])
def test_a_trial_logs_in_order_and_reads_tokens_after_run_start(tmp_path, target, evals, status):
  data = tmp_path / "data"
  data.mkdir()
  rng = np.random.default_rng(0)
  for name, count in ((contract.LLAMA31_8B.train_data, 400), (contract.LLAMA31_8B.eval_data, 60)):
    megatron.write(data / name, [rng.integers(0, 1000, n) for n in rng.integers(1, 300, count)])
  driver = tmp_path / "driver.py"
  driver.write_text(_DRIVER)
  log, mllog, reads = tmp_path / "train.jsonl", tmp_path / "mllog.txt", tmp_path / "reads.json"
  result = subprocess.run(
      [sys.executable, str(driver), str(data), str(log), str(mllog), str(reads), str(target)],
      capture_output=True, text=True, check=False, cwd=REPO,
      env={**os.environ, "JAX_TITAN_ROOT": str(tmp_path / "root")})
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]

  lines = _mllog(mllog)
  timeline = [line for line in lines if line["key"] in _TIMELINE]
  expected = [("init_start", None), ("init_stop", None), ("run_start", None)]
  opened = 0
  for samples in evals:
    expected += [("block_start", opened), ("block_stop", samples), ("eval_start", samples),
                 ("eval_accuracy", samples), ("eval_stop", samples)]
    opened = samples
  expected.append(("run_stop", evals[-1]))
  assert [(line["key"], line["metadata"].get("samples_count")) for line in timeline] == expected
  times = [line["time_ms"] for line in timeline]
  assert times == sorted(times)
  keys = [line["key"] for line in lines]
  assert lines[keys.index("run_stop")]["metadata"]["status"] == status
  assert lines[keys.index("train_samples")]["value"] == evals[-1]
  assert lines[keys.index("seed")]["value"] == 5
  assert lines[keys.index("eval_samples")]["value"] == 16

  trainer = records.read(log)
  events = [r.get("event") for r in trainer]
  assert events.index("warmup") < events.index("run_start")
  run_start = trainer[events.index("run_start")]["time"]
  token_reads = json.loads(reads.read_text())
  assert token_reads and min(token_reads) >= run_start
