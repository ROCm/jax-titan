"""MLPerf's compliance log (MLLOG), written from the trainer's records.

`Listener` takes every step record and timeline event the trainer writes
(`train.main(argv, listeners=[...])`) and logs MLLOG lines through
mlperf_logging's `mllog`, the only way the rules accept a seed:

  start      cache_clear and init_start (at the trainer's `main_start`), the
             submission keys, the seed, the hyperparameters, the lowest
             precisions, the parallelism and the config file
  run_start  init_stop and run_start, then block_start
  eval       block_stop and eval_start when the pass began, eval_accuracy and
             eval_stop, each with samples_count; when training goes on, a block
             starts again as the pass ended
  end        block_stop if a block is open, run_stop (success when the run stopped
             at the target, aborted otherwise) and train_samples

The hyperparameters it logs come from the benchmark's contract; at `start` it
checks them against the run's arguments and refuses to log a value the run
does not use. The lines go to stdout, which the trial's log captures with the
trainer's output, and to `path` when given.
"""

import logging
import math
import sys

from jax_titan.mlperf import contract as contract_lib

# Operator providers that run GEMMs below bf16 (jax_titan.ops.providers).
_LOW_PRECISION = ("fp8", "mxfp", "fp4")


def run_shape(args: dict, num_devices: int) -> dict:
  """A trainer run's global batch, gradient accumulation, per-GPU micro-batch and
  parallelism, as MLLOG counts them."""
  gbs = args["microbatches"] * args["microbatch_size"]
  tp = args.get("tensor_parallel", 1)
  if args.get("parallel") == "mpmd":
    meshes = args["num_stages"] // args.get("stages_per_mesh", 1)
    data_parallel = num_devices // meshes // tp
    return {"global_batch_size": gbs, "gradient_accumulation_steps": args["microbatches"],
            "micro_batch_size": args["microbatch_size"] // data_parallel,
            "tensor_parallelism": tp, "pipeline_parallelism": meshes}
  # dp and fsdp run the global batch as one batch over all devices.
  data_parallel = num_devices // tp
  return {"global_batch_size": gbs, "gradient_accumulation_steps": 1,
          "micro_batch_size": gbs // data_parallel, "tensor_parallelism": tp,
          "pipeline_parallelism": 1}


class Listener:
  """Turns the trainer's records into MLLOG lines (see the module docstring).

  `submission` holds `org`, `platform`, `division` and `status`; `precision` the
  lowest numerical precision in `linear`, `attn` and `comm`, each one the rules
  approve; `cache_clear` whether the caller dropped the page cache before the run.
  """

  def __init__(self, contract: contract_lib.Contract, *, hyperparameters: dict,
               submission: dict, precision: dict, config_filename: str, cache_clear: bool,
               path: str | None = None):
    try:
      from mlperf_logging.mllog import constants  # pylint: disable=g-import-not-at-top
      from mlperf_logging.mllog import mllog  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      raise ImportError("MLLOG needs mlperf_logging: pip install 'jax-titan[mlperf]'") from e
    contract.check(hyperparameters)
    for part in ("linear", "attn", "comm"):
      if precision.get(part) not in contract_lib.APPROVED_PRECISIONS:
        raise ValueError(f"precision {part}={precision.get(part)!r} is not one the rules "
                         f"approve: {', '.join(contract_lib.APPROVED_PRECISIONS)}")
    self.contract = contract
    self._hparams = dict(hyperparameters)
    self._submission = submission
    self._precision = precision
    self._config_filename = config_filename
    self._cache_clear = bool(cache_clear)
    self._c = constants
    logger = logging.getLogger(f"jax_titan.mlperf.mllog.{id(self)}")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    if path:
      logger.addHandler(logging.FileHandler(path))
    self._log = mllog.MLLogger(logger=logger)
    self._gbs = None
    self._samples = 0
    self._block_open = False
    self._block_pending = None  # when a validation ended, until training goes on

  def __call__(self, record: dict) -> None:
    name = record.get("event")
    if name is None:
      self._step(record)
    elif name == "start":
      self._start(record)
    elif name == "run_start":
      self._run_start(record)
    elif name == "eval":
      self._eval(record)
    elif name == "end":
      self._end(record)

  def _meta(self, samples: int) -> dict:
    return {self._c.SAMPLES_COUNT: int(samples)}

  def _start(self, record: dict) -> None:
    c, log = self._c, self._log
    args = record["args"]
    shape = run_shape(args, record["num_devices"])
    self._check(args, shape)
    self._gbs = shape["global_batch_size"]
    t = _ms(record["main_start"])
    log.event(key=c.CACHE_CLEAR, value=self._cache_clear, time_ms=t)
    log.start(key=c.INIT_START, time_ms=t)
    log.event(key=c.SUBMISSION_BENCHMARK, value=self.contract.benchmark, time_ms=t)
    log.event(key=c.SUBMISSION_ORG, value=self._submission["org"], time_ms=t)
    log.event(key=c.SUBMISSION_DIVISION, value=self._submission["division"], time_ms=t)
    log.event(key=c.SUBMISSION_STATUS, value=self._submission["status"], time_ms=t)
    log.event(key=c.SUBMISSION_PLATFORM, value=self._submission["platform"], time_ms=t)
    log.event(key=c.SEED, value=int(args["seed"]), time_ms=t)
    for key, value in self._hparams.items():
      log.event(key=key, value=value, time_ms=t)
    log.event(key=c.LOWEST_NUMERICAL_PRECISION_IN_LINEAR, value=self._precision["linear"],
              time_ms=t)
    log.event(key=c.LOWEST_NUMERICAL_PRECISION_IN_ATTN, value=self._precision["attn"], time_ms=t)
    log.event(key=c.LOWEST_NUMERICAL_PRECISION_IN_COMM, value=self._precision["comm"], time_ms=t)
    log.event(key=c.TENSOR_PARALLELISM, value=shape["tensor_parallelism"], time_ms=t)
    log.event(key=c.PIPELINE_PARALLELISM, value=shape["pipeline_parallelism"], time_ms=t)
    log.event(key=c.CONTEXT_PARALLELISM, value=1, time_ms=t)
    log.event(key=c.EXPERT_PARALLELISM, value=1, time_ms=t)
    log.event(key=c.MICRO_BATCH_SIZE, value=shape["micro_batch_size"], time_ms=t)
    log.event(key=c.CONFIG_FILENAME, value=self._config_filename, time_ms=t)

  def _check(self, args: dict, shape: dict) -> None:
    """Refuses hyperparameters that are not the ones the run trains with."""
    used = {
        "global_batch_size": shape["global_batch_size"],
        "gradient_accumulation_steps": shape["gradient_accumulation_steps"],
        "max_sequence_length": args["seq_len"],
        "max_steps": args["steps"],
        "eval_samples": args["eval_samples"],
        "opt_base_learning_rate": args["lr"],
        "opt_end_learning_rate": args["lr"] * 0.1,
        "opt_learning_rate_warmup_steps": args["warmup_steps"],
        "opt_learning_rate_decay_steps": args["lr_decay_steps"],
        "opt_adamw_beta_1": args["adam_b1"],
        "opt_adamw_beta_2": args["adam_b2"],
        "opt_adamw_epsilon": args["adam_eps"],
        "opt_adamw_weight_decay": args["weight_decay"],
        "opt_gradient_clip_norm": args["max_grad_norm"],
    }
    errors = [f"{key}: logs {self._hparams.get(key)!r}, the run uses {value!r}"
              for key, value in used.items()
              if not math.isclose(self._hparams.get(key, math.nan), value, rel_tol=1e-9)]
    if args.get("lr_schedule") != "nemo_cosine":
      errors.append(f"lr_schedule: the run uses {args.get('lr_schedule')!r}, not the "
                    "reference's cosine with linear warmup (nemo_cosine)")
    every = self.contract.eval_every_steps(shape["global_batch_size"])
    if args.get("eval_every") != every:
      errors.append(f"eval_every: the run validates every {args.get('eval_every')} steps, not "
                    f"every {every}")
    if args.get("target_eval_loss") != self.contract.quality_target:
      errors.append(f"target_eval_loss: the run stops at {args.get('target_eval_loss')}, not "
                    f"{self.contract.quality_target}")
    if self._precision["linear"] == "bfloat16" and any(
        p in args.get("ops", "") for p in _LOW_PRECISION):
      errors.append(f"precision linear=bfloat16, but the run's --ops are {args['ops']!r}")
    if errors:
      raise ValueError("the MLLOG hyperparameters are not the run's: " + "; ".join(errors))

  def _run_start(self, record: dict) -> None:
    c, t = self._c, _ms(record["time"])
    self._samples = record.get("step", 0) * self._gbs
    self._log.end(key=c.INIT_STOP, time_ms=t)
    self._log.start(key=c.RUN_START, time_ms=t)
    self._log.start(key=c.BLOCK_START, metadata=self._meta(self._samples), time_ms=t)
    self._block_open = True

  def _step(self, record: dict) -> None:
    if self._block_pending is not None:
      # Training went on after the validation: the block began when it ended.
      self._log.start(key=self._c.BLOCK_START, metadata=self._meta(self._samples),
                      time_ms=self._block_pending)
      self._block_open, self._block_pending = True, None
    self._samples = (record["step"] + 1) * self._gbs

  def _eval(self, record: dict) -> None:
    c, log = self._c, self._log
    end, samples = _ms(record["time"]), record["samples"]
    began = _ms(record["time"] - record["seconds"])
    if self._block_open:
      log.end(key=c.BLOCK_STOP, metadata=self._meta(samples), time_ms=began)
      self._block_open = False
    log.start(key=c.EVAL_START, metadata=self._meta(samples), time_ms=began)
    log.event(key=c.EVAL_ACCURACY, value=record["loss"], metadata=self._meta(samples), time_ms=end)
    log.end(key=c.EVAL_STOP, metadata=self._meta(samples), time_ms=end)
    self._block_pending = end

  def _end(self, record: dict) -> None:
    c, log, t = self._c, self._log, _ms(record["time"])
    if self._block_open:
      log.end(key=c.BLOCK_STOP, metadata=self._meta(self._samples), time_ms=t)
      self._block_open = False
    status = c.SUCCESS if record.get("target_reached") else c.ABORTED
    log.end(key=c.RUN_STOP, metadata={**self._meta(self._samples), "status": status}, time_ms=t)
    log.event(key=c.TRAIN_SAMPLES, value=int(self._samples), time_ms=t)


def _ms(seconds: float) -> int:
  return int(round(seconds * 1000))
