"""Runs an MLPerf Training trial: a config, the benchmark's contract, and MLLOG.

  python -m jax_titan.mlperf.run CONFIG.json --data_dir DIR [--seed N] [--cache_clear]
      [--mllog FILE] [--dry_run] [trainer flags ...]

CONFIG.json names the benchmark and holds what a submitter chooses:

  benchmark          a contract in jax_titan.mlperf.contract
  submission         org, platform, division ("closed" or "open") and status
  global_batch_size, base_lr, warmup_steps
  precision          the lowest numerical precision in linear, attn and comm
  eval_batch_size    sequences per validation batch (default: the global batch)
  synthetic_warmup   training steps on synthetic tokens before the clock (default 2)
  trainer            trainer flags for the layout, micro-batching and operators

The runner derives the rest of the trainer's flags from the contract (model,
sequence length, steps, schedule, AdamW, data, validation and target) and
refuses a config or command line that sets one of them; draws the seed from
the operating system unless --seed is given; checks the hyperparameters
against the contract; and trains with jax_titan.mlperf.mllog's listener, whose
MLLOG lines go to stdout. scripts/mlperf_run_and_time.sh runs trials of it.
"""

import argparse
import json
import os
import secrets
import sys

from jax_titan.mlperf import contract as contract_lib
from jax_titan.mlperf import mllog

# The JAX-Titan model each benchmark trains.
MODELS = {"llama31_8b": "llama3.1-8b"}

# Trainer flags the runner derives from the contract.
CONTROLLED = ("model", "model_overrides", "init_from", "seq_len", "steps", "lr", "warmup_steps",
              "lr_schedule", "lr_decay_steps", "adam_b1", "adam_b2", "adam_eps", "weight_decay",
              "max_grad_norm", "data", "eval_data", "eval_every", "eval_samples",
              "eval_batch_size", "eval_batches", "target_eval_loss", "seed", "synthetic_warmup")


def trainer_argv(config: dict, contract: contract_lib.Contract, data_dir: str, seed: int,
                 extra=()) -> list[str]:
  """The trainer's flags for a trial: the config's, then the contract's."""
  flags = list(config.get("trainer", ())) + list(extra)
  named = {f[2:].split("=")[0] for f in flags if f.startswith("--")}
  clash = sorted(named & set(CONTROLLED))
  if clash:
    raise contract_lib.ContractError(
        "the contract sets " + ", ".join(f"--{n}" for n in clash) + ": take them out")
  if contract.benchmark not in MODELS:
    raise contract_lib.ContractError(f"JAX-Titan has no model for {contract.benchmark} yet")
  gbs, warmup = config["global_batch_size"], config["warmup_steps"]
  optimizer = contract.optimizer
  return flags + [
      f"--model={MODELS[contract.benchmark]}",
      f"--seq_len={contract.sequence_length}",
      f"--steps={contract.max_steps}",
      f"--lr={config['base_lr']}",
      f"--warmup_steps={warmup}",
      "--lr_schedule=nemo_cosine",
      f"--lr_decay_steps={contract.decay_steps(warmup)}",
      f"--adam_b1={optimizer['opt_adamw_beta_1']}",
      f"--adam_b2={optimizer['opt_adamw_beta_2']}",
      f"--adam_eps={optimizer['opt_adamw_epsilon']}",
      f"--weight_decay={optimizer['opt_adamw_weight_decay']}",
      f"--max_grad_norm={optimizer['opt_gradient_clip_norm']}",
      f"--data={os.path.join(data_dir, contract.train_data)}",
      f"--eval_data={os.path.join(data_dir, contract.eval_data)}",
      f"--eval_every={contract.eval_every_steps(gbs)}",
      f"--eval_samples={contract.eval_samples}",
      f"--eval_batch_size={config.get('eval_batch_size', gbs)}",
      f"--target_eval_loss={contract.quality_target}",
      f"--seed={seed}",
      f"--synthetic_warmup={config.get('synthetic_warmup', 2)}",
  ]


def hyperparameters(config: dict, contract: contract_lib.Contract, argv: list[str]) -> dict:
  """The MLLOG hyperparameters of a trial, checked against the contract and the
  trainer's micro-batching."""
  from jax_titan.trainer import train  # pylint: disable=g-import-not-at-top
  args = train.parse_args(argv)
  gbs = config["global_batch_size"]
  if args.microbatches * args.microbatch_size != gbs:
    raise contract_lib.ContractError(
        f"--microbatches={args.microbatches} of --microbatch_size={args.microbatch_size} "
        f"is not the global batch of {gbs}")
  submission = config.get("submission", {})
  if not (submission.get("org") and submission.get("platform") and submission.get("status")
          and submission.get("division") in ("closed", "open")):
    raise ValueError(f"submission needs org, platform, status and a division of closed or "
                     f"open, not {submission}")
  hparams = contract.hyperparameters(
      global_batch_size=gbs, base_lr=config["base_lr"], warmup_steps=config["warmup_steps"],
      gradient_accumulation_steps=args.microbatches if args.parallel == "mpmd" else 1)
  contract.check(hparams)
  return hparams


def prepare(config: dict, *, data_dir: str, seed: int | None = None,
            contract: contract_lib.Contract | None = None, extra=()):
  """A trial's contract, trainer flags and MLLOG hyperparameters, refused where
  they depart from the contract; `contract` defaults to the config's benchmark's."""
  contract = contract or contract_lib.get(config["benchmark"])
  seed = secrets.randbelow(2**31) if seed is None else seed
  argv = trainer_argv(config, contract, data_dir, seed, extra)
  return contract, argv, hyperparameters(config, contract, argv)


def listener(config: dict, contract: contract_lib.Contract, hparams: dict, *,
             config_filename: str, cache_clear: bool = False,
             mllog_path: str | None = None) -> mllog.Listener:
  return mllog.Listener(
      contract, hyperparameters=hparams, submission=config["submission"],
      precision=config.get("precision", {}), config_filename=config_filename,
      cache_clear=cache_clear, path=mllog_path)


def main(argv=None):
  parser = argparse.ArgumentParser(
      description=__doc__.split("\n\n", maxsplit=1)[0],
      epilog="Flags the runner does not know go to the trainer.")
  parser.add_argument("config", help="The trial's CONFIG.json.")
  parser.add_argument("--data_dir", required=True,
                      help="The directory scripts/mlperf_download_c4.sh filled.")
  parser.add_argument("--seed", type=int, default=None,
                      help="The trial's seed; by default one the operating system draws.")
  parser.add_argument("--cache_clear", action="store_true",
                      help="The caller dropped the page cache before this trial; MLLOG says so.")
  parser.add_argument("--mllog", default="", help="Also write the MLLOG lines to this file.")
  parser.add_argument("--dry_run", action="store_true",
                      help="Print the trainer's flags and the hyperparameters; do not train.")
  args, extra = parser.parse_known_args(argv)
  with open(args.config) as f:
    config = json.load(f)
  try:
    contract, trial, hparams = prepare(config, data_dir=args.data_dir, seed=args.seed,
                                       extra=extra)
    if args.dry_run:
      print(json.dumps({"trainer": trial, "hyperparameters": hparams}, indent=1))
      return
    log = listener(config, contract, hparams, config_filename=os.path.basename(args.config),
                   cache_clear=args.cache_clear, mllog_path=args.mllog or None)
  except (KeyError, ValueError, ImportError) as e:
    sys.exit(f"jax_titan.mlperf.run: {e}")
  from jax_titan.trainer import train  # pylint: disable=g-import-not-at-top
  train.main(trial, listeners=[log])


if __name__ == "__main__":
  main()
