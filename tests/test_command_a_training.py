"""Command A+ trains through the trainer: its layouts agree, it starts from a
published checkpoint, saves what generation reads, and resumes exactly."""

import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys

import command_a_checkpoint
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_titan import parallel
from jax_titan.data import command_a as data
from jax_titan.models import command_a
from jax_titan.models import command_a_generate
from jax_titan.observability import records
from jax_titan.trainer import adapters, train

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = {
    "vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 32, "num_experts": 8,
    "num_experts_per_tok": 2, "num_shared_experts": 2, "sliding_window": 8, "image_token_id": 5,
    "alignment_intermediate_size": 48, "vision_hidden_size": 16, "vision_intermediate_size": 32,
    "vision_num_hidden_layers": 2, "vision_num_attention_heads": 2, "vision_image_size": 16,
    "vision_patch_size": 4,
}
FLAGS = [
    "--platform=cpu", "--model=command-a-plus", "--num_stages=1", "--data=synthetic",
    "--seq_len=16", "--microbatches=2", "--microbatch_size=8", "--steps=4", "--warmup_steps=1",
]
SMALL = ["--model_overrides=" + json.dumps(CONFIG | {"dtype": "float32"})]
IMAGES = ["--model_overrides=" + json.dumps(CONFIG | {"dtype": "float32", "image_tiles": 2})]


def _run(log, *flags):
  # os.environ, not the process environment: see conftest's ROCPROFILER note.
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", *FLAGS, f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r for r in records.steps(records.read(log))}


def _validation(log) -> list[float]:
  return [r["loss"] for r in records.events(records.read(log), "eval")]


@pytest.mark.parametrize("batch", [SMALL, IMAGES], ids=["text", "images"])
@pytest.mark.parametrize("flags", [["--parallel=fsdp"], ["--parallel=mpmd"]], ids=["fsdp", "mpmd"])
def test_layout_matches_data_parallel_reference(flags, batch, tmp_path):
  dp = tmp_path / "dp.jsonl"
  reference = _run(dp, *batch, "--parallel=dp", "--eval_every=4", "--eval_batches=1")
  log = tmp_path / "layout.jsonl"
  trained = _run(log, *batch, *flags, "--eval_every=4", "--eval_batches=1")
  for step in range(4):
    assert trained[step]["loss"] == pytest.approx(reference[step]["loss"], rel=1e-5), step
    assert trained[step]["grad_norm"] == pytest.approx(reference[step]["grad_norm"], rel=1e-4), step
  assert _validation(log) == pytest.approx(_validation(dp), rel=1e-5)


def test_starts_from_a_published_checkpoint_and_saves_what_generation_reads(tmp_path, capsys):
  toy, saved = tmp_path / "checkpoint", tmp_path / "saved"
  params = command_a_checkpoint.write(toy)
  cfg = command_a_checkpoint.CONFIG
  flags = ["--model_overrides=" + json.dumps({"dtype": "float32"}), "--parallel=fsdp",
           f"--init_from={toy}", "--eval_every=4", "--eval_batches=1"]
  log = tmp_path / "trained.jsonl"
  trained = _run(log, *flags, f"--save_params={saved}")

  # The first step's loss is the checkpoint's on the trainer's first batch.
  windows = np.random.default_rng((0, 0)).integers(0, cfg.vocab_size, (16, 17), dtype=np.int32)
  language = {k: v for k, v in params.items() if k not in ("vision", "projector")}
  logits = command_a.forward(cfg, language, windows[:, :-1], dtype=np.float32)
  loss = float(command_a.cross_entropy(logits, windows[:, 1:]))
  assert trained[0]["loss"] == pytest.approx(loss, rel=1e-5)

  # --save_params wrote the language model after the last step: a run from it, which
  # has no config.json to read, validates alike.
  fields = json.dumps(dataclasses.asdict(cfg) | {"dtype": "float32"})
  _run(tmp_path / "again.jsonl", f"--model_overrides={fields}", "--parallel=dp",
       f"--init_from={saved}", "--steps=1", "--lr=0", "--eval_every=1", "--eval_batches=1")
  assert _validation(tmp_path / "again.jsonl") == pytest.approx(_validation(log), rel=1e-5)

  # Generation takes the saved parameters in place of the checkpoint's.
  after = command_a.Model(cfg).load(saved)[0]
  assert sorted(after) == ["embed", "final_norm", "layers"]
  assert not np.array_equal(after["embed"], params["embed"])
  preprocessor = data.Preprocessor.open(toy)
  tokens, _ = preprocessor.encode([{"role": "user", "content": "a b"}])
  new = command_a_generate.generate(
      cfg, jax.tree.map(lambda w: w.astype(jnp.bfloat16), after), tokens[None], new_tokens=4)
  flags = [f"--checkpoint={toy}", "--prompt=a b", "--max_new_tokens=4", "--platform=cpu"]
  assert command_a_generate.main([*flags, f"--params={saved}"]) == 0
  assert capsys.readouterr().out.splitlines()[-1] == preprocessor.decode(new[0])


def test_resume_is_exact(tmp_path):
  flags = [*SMALL, "--parallel=fsdp", "--eval_every=2", "--checkpoint_every=2"]
  full = _run(tmp_path / "full.jsonl", *flags, f"--checkpoint_dir={tmp_path / 'full'}")
  (tmp_path / "resumed").mkdir()
  shutil.copytree(tmp_path / "full" / "2", tmp_path / "resumed" / "2")
  resumed = _run(tmp_path / "resumed.jsonl", *flags, f"--checkpoint_dir={tmp_path / 'resumed'}")
  assert sorted(resumed) == [2, 3]
  for step in (2, 3):
    assert resumed[step]["loss"] == full[step]["loss"], step
  assert _validation(tmp_path / "resumed.jsonl") == _validation(tmp_path / "full.jsonl")[1:]


def test_every_parameter_is_sharded_and_only_norms_do_not_decay():
  cfg = command_a.CommandAConfig(**CONFIG)
  params = jax.eval_shape(lambda: command_a.init_params(cfg, jax.random.key(0)))
  specs = parallel.tree_specs(params, command_a.logical_axes, {"data": 8, "tensor": 1}, fsdp=True)
  for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
    assert command_a.logical_axes(path, leaf) is not None, path
  # Every weight of the language model is sharded; of the vision tower, those that split in 8.
  for path, spec in jax.tree_util.tree_flatten_with_path(specs)[0]:
    assert "data" in spec or path[0].key in ("vision", "projector"), path
  assert specs["layers"]["w_gate"] == jax.sharding.PartitionSpec(None, "data")
  mask = train._decay_mask(params)  # pylint: disable=protected-access
  kept = {path[-1].key for path, decays in jax.tree_util.tree_flatten_with_path(mask)[0]
          if not decays}
  assert kept == {"norm", "final_norm", "norm1", "norm1_bias", "norm2", "norm2_bias",
                  "final_norm_bias"}


def test_trains_as_one_stage_and_with_tiles_on_synthetic_batches_only():
  def build(*flags):
    args = train.parse_args([*FLAGS, *flags])
    return adapters.build(args, jax.devices(), optax.sgd(0.0))

  with pytest.raises(NotImplementedError, match="one stage"):
    build(*SMALL, "--parallel=dp", "--num_stages=2")
  with pytest.raises(NotImplementedError, match="synthetic batches only"):
    build(*IMAGES, "--parallel=dp", "--data=tokens.bin")
  with pytest.raises(ValueError, match="more than --seq_len=8 holds"):
    build(*IMAGES, "--parallel=dp", "--seq_len=8")
  _, work = build(*SMALL, "--parallel=dp")
  assert work.vocab_size == 128 and work.next_batch is None and work.tokens_per_step == 256
  layout, work = build(*IMAGES, "--parallel=dp")
  x, targets = work.next_batch(0)
  # Two tiles of 4 image tokens open every sequence, and no image token is a target.
  assert x["pixels"].shape == (2, 8, 2, 3, 16, 16)
  assert x["tokens"].shape == targets.shape == (2, 8, 16)
  assert (x["tokens"][..., :8] == 5).all() and (x["tokens"][..., 8:] != 5).all()
  assert (targets[..., :7] == -1).all() and (targets[..., 7:] >= 0).all()
  assert "vision" in layout.abstract_state[0][0]
