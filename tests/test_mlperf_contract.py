"""The MLPerf contracts: the values the rules fix, the derived values, and refusals."""

import pytest

from jax_titan.mlperf import contract


def _llama_hparams(**changes):
  hp = contract.LLAMA31_8B.hyperparameters(global_batch_size=32, base_lr=7.6e-4,
                                           warmup_steps=64)
  hp.update(changes)
  return hp


def test_llama_values_are_the_rules():
  c = contract.get("llama31_8b")
  assert (c.quality_target, c.runs, c.sequence_length) == (3.3, 10, 8192)
  assert (c.eval_interval_samples, c.eval_samples, c.max_steps) == (12288, 1024, 1_200_000)
  assert c.fixed == {
      "max_sequence_length": 8192, "eval_samples": 1024, "max_steps": 1_200_000,
      "opt_learning_rate_decay_schedule": "cosine with linear warmup", "opt_name": "adamw",
      "opt_adamw_beta_1": 0.9, "opt_adamw_beta_2": 0.95, "opt_adamw_epsilon": 1e-5,
      "opt_adamw_weight_decay": 0.1, "opt_gradient_clip_norm": 1.0,
  }
  assert c.init_minutes == 30
  gpt = contract.get("gpt_oss_20b")
  assert gpt.quality_target == 3.34 and gpt.fixed == c.fixed


def test_derived_values():
  c = contract.LLAMA31_8B
  # Every 12,288 sequences, rounded up to whole global batches.
  assert c.eval_every_steps(32) == 384
  assert c.eval_every_steps(16) == 768
  assert c.eval_every_steps(40) == 308  # 12,320 sequences
  assert c.decay_steps(64) == 1_199_936
  assert c.end_lr(7.6e-4) == pytest.approx(7.6e-5)


def test_hyperparameters_pass_the_check():
  contract.LLAMA31_8B.check(_llama_hparams())
  contract.GPT_OSS_20B.check(
      contract.GPT_OSS_20B.hyperparameters(global_batch_size=32, base_lr=4.5e-4,
                                           warmup_steps=234))


@pytest.mark.parametrize("changes,message", [
    ({"opt_adamw_beta_2": 0.999}, "opt_adamw_beta_2 is 0.999"),
    ({"opt_adamw_epsilon": 1e-8}, "opt_adamw_epsilon is 1e-08"),
    ({"max_sequence_length": 4096}, "max_sequence_length is 4096"),
    ({"opt_gradient_clip_norm": 0.5}, "opt_gradient_clip_norm is 0.5"),
    ({"eval_samples": 512}, "eval_samples is 512"),
    ({"max_steps": 10_000}, "max_steps is 10000"),
    ({"opt_end_learning_rate": 0.0}, "opt_end_learning_rate is 0.0"),
    ({"opt_learning_rate_decay_steps": 1_200_000}, "opt_learning_rate_decay_steps is 1200000"),
    ({"global_batch_size": 0}, "global_batch_size is 0"),
    ({"gradient_accumulation_steps": 1.0}, "gradient_accumulation_steps is 1.0"),
    ({"opt_learning_rate_warmup_steps": -1}, "opt_learning_rate_warmup_steps is -1"),
    ({"dropout": 0.1}, "dropout is not part of the contract"),
    ({"opt_name": True}, "opt_name is True"),
])
def test_departures_are_refused(changes, message):
  with pytest.raises(contract.ContractError, match=message):
    contract.LLAMA31_8B.check(_llama_hparams(**changes))


def test_every_departure_is_named():
  hp = _llama_hparams(opt_adamw_beta_1=0.8, opt_adamw_weight_decay=0.0)
  del hp["opt_base_learning_rate"]
  with pytest.raises(contract.ContractError) as e:
    contract.LLAMA31_8B.check(hp)
  text = str(e.value)
  for part in ("opt_adamw_beta_1 is 0.8", "opt_adamw_weight_decay is 0.0",
               "opt_base_learning_rate is missing"):
    assert part in text


def test_unknown_benchmark():
  with pytest.raises(KeyError, match="no contract for 'resnet'"):
    contract.get("resnet")
