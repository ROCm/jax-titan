"""What an MLPerf Training Closed-division run must hold fixed, and what it may tune.

A `Contract` holds a benchmark's fixed values (the optimizer's constants, the
sequence length, the quality target, how often and on how many sequences it
validates) and the rules that tie its tunable values together: the end
learning rate is a tenth of the base one, and the cosine decays over
1,200,000 steps minus the warm-up. `check` refuses the MLLOG hyperparameters
of a run that departs from the contract, as the compliance checker would, and
names every departure; a key the contract does not know is refused too.

Pinned to the MLPerf Training rules of 24 July 2026 (v6.1), the compliance
checker configs of mlcommons/logging 6.1.0-rc2 (`training_6.1.0`) and the
reference code those rules cite; docs/mlperf.md gives the source of each value.
"""

import dataclasses
import math
from typing import Any, Mapping

MAX_STEPS = 1_200_000
DECAY_SCHEDULE = "cosine with linear warmup"
# The pre-approved formats of the rules' Numerical formats section, as the
# compliance checker spells them for the lowest_numerical_precision_* keys.
APPROVED_PRECISIONS = ("fp64", "fp32", "tf32", "fp16", "fp8", "mxfp6", "nvfp4", "mxfp4",
                       "bfloat16", "Graphcore FLOAT 16.16", "int8", "uint8", "int4", "uint4")
# MLLOG keys whose values a run chooses, within the constraints `check` applies.
TUNABLE = ("global_batch_size", "gradient_accumulation_steps", "opt_base_learning_rate",
           "opt_end_learning_rate", "opt_learning_rate_warmup_steps",
           "opt_learning_rate_decay_steps")


class ContractError(ValueError):
  """A run's hyperparameters break its benchmark's contract."""


@dataclasses.dataclass(frozen=True)
class Contract:
  """One benchmark's contract.

  `optimizer` holds the MLLOG optimizer keys with their fixed values.
  `train_data` and `eval_data` are the file prefixes of the Megatron indexed
  datasets the reference reads; validation uses the first `eval_samples`
  sequences of `eval_data` in file order.
  """

  benchmark: str
  quality_target: float  # validation log perplexity at or below this
  runs: int
  eval_interval_samples: int
  eval_samples: int
  sequence_length: int
  optimizer: Mapping[str, Any]
  train_data: str
  eval_data: str
  end_lr_ratio: float = 0.1
  max_steps: int = MAX_STEPS
  init_minutes: int = 30

  @property
  def fixed(self) -> dict:
    """The MLLOG keys a run must log with exactly these values."""
    return {
        "max_sequence_length": self.sequence_length,
        "eval_samples": self.eval_samples,
        "max_steps": self.max_steps,
        "opt_learning_rate_decay_schedule": DECAY_SCHEDULE,
        **self.optimizer,
    }

  def eval_every_steps(self, global_batch_size: int) -> int:
    """Steps between validations: every `eval_interval_samples` training
    sequences, rounded up to whole global batches."""
    return math.ceil(self.eval_interval_samples / global_batch_size)

  def decay_steps(self, warmup_steps: int) -> int:
    """Steps of the cosine after the warm-up."""
    return self.max_steps - warmup_steps

  def end_lr(self, base_lr: float) -> float:
    return base_lr * self.end_lr_ratio

  def hyperparameters(self, *, global_batch_size: int, base_lr: float, warmup_steps: int,
                      gradient_accumulation_steps: int = 1) -> dict:
    """The MLLOG hyperparameters of a run with these tunable values."""
    return {
        "global_batch_size": global_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "opt_base_learning_rate": base_lr,
        "opt_end_learning_rate": self.end_lr(base_lr),
        "opt_learning_rate_warmup_steps": warmup_steps,
        "opt_learning_rate_decay_steps": self.decay_steps(warmup_steps),
        **self.fixed,
    }

  def check(self, hparams: Mapping[str, Any]) -> None:
    """Raises ContractError naming every departure of `hparams` from the contract."""
    errors = [f"{key} is not part of the contract"
              for key in sorted(set(hparams) - set(self.fixed) - set(TUNABLE))]
    for key, value in self.fixed.items():
      if key not in hparams:
        errors.append(f"{key} is missing (the contract fixes it at {value!r})")
      elif not _equal(hparams[key], value):
        errors.append(f"{key} is {hparams[key]!r}, the contract fixes it at {value!r}")
    errors += [f"{key} is missing" for key in TUNABLE if key not in hparams]

    def positive_int(key):
      value = hparams.get(key)
      if key in hparams and not (_is_int(value) and value > 0):
        errors.append(f"{key} is {value!r}, not a positive integer")

    positive_int("global_batch_size")
    positive_int("gradient_accumulation_steps")
    warmup = hparams.get("opt_learning_rate_warmup_steps")
    if "opt_learning_rate_warmup_steps" in hparams and not (_is_int(warmup) and warmup >= 0):
      errors.append(f"opt_learning_rate_warmup_steps is {warmup!r}, not an integer >= 0")
    base = hparams.get("opt_base_learning_rate")
    base_ok = _is_number(base) and base > 0
    if "opt_base_learning_rate" in hparams and not base_ok:
      errors.append(f"opt_base_learning_rate is {base!r}, not a number > 0")
    if base_ok and "opt_end_learning_rate" in hparams:
      end = hparams["opt_end_learning_rate"]
      if not (_is_number(end) and math.isclose(end, self.end_lr(base), rel_tol=1e-9)):
        errors.append(f"opt_end_learning_rate is {end!r}, the contract makes it "
                      f"{self.end_lr_ratio} x the base, {self.end_lr(base)!r}")
    if _is_int(warmup) and "opt_learning_rate_decay_steps" in hparams:
      decay = hparams["opt_learning_rate_decay_steps"]
      if decay != self.decay_steps(warmup):
        errors.append(f"opt_learning_rate_decay_steps is {decay!r}, the contract makes it "
                      f"{self.max_steps} - warm-up, {self.decay_steps(warmup)}")
    if errors:
      raise ContractError(f"{self.benchmark}: " + "; ".join(errors))


def _is_int(value) -> bool:
  return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _equal(value, fixed) -> bool:
  """Equality as the compliance checker tests it (`v['value'] == ...`), without
  letting a bool pass for a number."""
  if isinstance(value, bool) != isinstance(fixed, bool):
    return False
  return value == fixed


# AdamW as the reference's distributed_fused_adam_with_cosine_annealing sets it, and as
# the rules and the compliance checker fix it, for both C4 pretraining benchmarks.
_C4_ADAMW = {
    "opt_name": "adamw",
    "opt_adamw_beta_1": 0.9,
    "opt_adamw_beta_2": 0.95,
    "opt_adamw_epsilon": 1e-5,
    "opt_adamw_weight_decay": 0.1,
    "opt_gradient_clip_norm": 1.0,
}

LLAMA31_8B = Contract(
    benchmark="llama31_8b", quality_target=3.3, runs=10, eval_interval_samples=12288,
    eval_samples=1024, sequence_length=8192, optimizer=_C4_ADAMW,
    train_data="c4-train.en_6_text_document",
    eval_data="c4-validation-91205-samples.en_text_document",
)
GPT_OSS_20B = dataclasses.replace(LLAMA31_8B, benchmark="gpt_oss_20b", quality_target=3.34)

CONTRACTS = {c.benchmark: c for c in (LLAMA31_8B, GPT_OSS_20B)}


def get(benchmark: str) -> Contract:
  if benchmark not in CONTRACTS:
    raise KeyError(f"no contract for {benchmark!r}; known: {', '.join(CONTRACTS)}")
  return CONTRACTS[benchmark]
