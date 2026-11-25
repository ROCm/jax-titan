"""transformers' causal language models through torchax, against PyTorch's own forward and backward."""

import copy
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("torchax", "transformers", "torch")),
    reason="needs the transformers extra",
)

SMALL = {
    "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 128,
}
# Other ways of a family: Qwen3's query and key norms, Gemma 3's local attention, Phi-3's long
# RoPE factors, which its forward caches on the module, mixtures of experts in the dense
# form, gpt-oss's with biases and transposed weights, and GPT-2's tied embedding, biases and
# learned positions.
FAMILIES = {
    "llama": SMALL,
    "mistral": SMALL,
    "qwen3": {**SMALL, "head_dim": 16},
    "gemma3_text": {**SMALL, "head_dim": 16, "sliding_window": 8, "pad_token_id": 0},
    "phi3": {**SMALL, "pad_token_id": 0, "original_max_position_embeddings": 32,
             "rope_parameters": {"rope_type": "longrope", "rope_theta": 10000.0,
                                 "short_factor": [1.0] * 8,
                                 "long_factor": [1.0 + 0.5 * i for i in range(8)],
                                 "original_max_position_embeddings": 32}},
    "qwen3_moe": {**SMALL, "head_dim": 16, "moe_intermediate_size": 32, "num_experts": 8,
                  "num_experts_per_tok": 2, "decoder_sparse_step": 1, "mlp_only_layers": [],
                  "norm_topk_prob": True},
    "gpt_oss": {**SMALL, "head_dim": 16, "num_local_experts": 8, "num_experts_per_tok": 2},
    "gpt2": {"vocab_size": 256, "n_embd": 64, "n_layer": 2, "n_head": 4, "n_positions": 128},
}


@pytest.mark.gpu
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_bridged_model_gives_pytorchs_loss_and_gradient(family):
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import torch  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import transformers as hf  # pylint: disable=g-import-not-at-top

  assert family in hf.VERIFIED
  cfg = hf.config(family, **FAMILIES[family])
  model = hf.causal_lm(cfg, seed=0, dtype=jnp.float32)
  rng = np.random.default_rng(1)
  ids = rng.integers(3, cfg.vocab_size, (2, 64))
  targets = rng.integers(3, cfg.vocab_size, (2, 64))

  # PyTorch on a copy: a forward on the module itself would leave Phi-3's cache behind.
  eager = copy.deepcopy(model.module)
  # The copy keeps the adapter's dense experts; the reference runs transformers' own.
  eager.set_experts_implementation("eager")
  eager._set_gradient_checkpointing(enable=False)  # pylint: disable=protected-access
  logits = eager(input_ids=torch.from_numpy(ids))[0]
  loss = torch.nn.functional.cross_entropy(
      logits.reshape(-1, logits.shape[-1]), torch.from_numpy(targets).reshape(-1))
  loss.backward()
  reference = {n: p.grad.numpy() for n, p in eager.named_parameters(remove_duplicate=False)}
  scale = max(np.abs(g).max() for g in reference.values())

  (params,) = model.init()
  (fn,) = model.stage_fns()
  # Twice, as the trainer traces the stage for its train step and for its validation step.
  for _ in range(2):
    value, grads = jax.jit(jax.value_and_grad(fn))(
        params, jnp.asarray(ids, jnp.int32), jnp.asarray(targets, jnp.int32))
    np.testing.assert_allclose(value, loss.item(), rtol=1e-5)
    for name, grad in grads.items():
      np.testing.assert_allclose(grad, reference[name], rtol=1e-3, atol=1e-5 * scale)
  # Norms take no weight decay; nothing else is exempt.
  norms = {n for n in params if "norm" in n or ".ln_" in n}
  assert norms and {n for n in params if not model.decays((jax.tree_util.DictKey(n),), None)} == norms


def _losses(tmp_path, *flags):
  """The losses of 4 steps of a small `transformers:llama` on every GPU."""
  log = tmp_path / f"{'-'.join(flags)}.jsonl"
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", "--model=transformers:llama",
       "--model_overrides=" + json.dumps({**SMALL, "dtype": "float32"}), "--data=synthetic",
       "--seq_len=32", "--microbatches=1", "--microbatch_size=8", "--steps=4",
       "--warmup_steps=1", f"--log_file={log}", *flags],
      # os.environ, not the process environment: see conftest's ROCPROFILER note.
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return [r["loss"] for r in records.steps(records.read(log))]


@pytest.mark.gpu
def test_trainer_starts_at_the_models_loss(tmp_path):
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import transformers as hf  # pylint: disable=g-import-not-at-top

  losses = _losses(tmp_path, "--parallel=dp")
  model = hf.causal_lm(hf.config("llama", **SMALL), seed=0, dtype=jnp.float32)
  # The trainer's weights and its first synthetic batch, both of seed 0.
  batch = np.random.default_rng((0, 0)).integers(0, SMALL["vocab_size"], (8, 33), dtype=np.int32)
  (fn,) = model.stage_fns()
  loss = jax.jit(fn)(*model.init(), batch[:, :-1], batch[:, 1:])
  np.testing.assert_allclose(losses[0], loss, rtol=1e-5)


@pytest.mark.gpu(2)
def test_fsdp_trains_as_dp_does(tmp_path):
  np.testing.assert_allclose(
      _losses(tmp_path, "--parallel=fsdp"), _losses(tmp_path, "--parallel=dp"), rtol=1e-4)


@pytest.mark.gpu
def test_refuses_what_was_not_compared_with_pytorch():
  import optax  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import builders  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import train  # pylint: disable=g-import-not-at-top

  def build(*flags):
    args = train.parse_args(["--data=synthetic", "--parallel=dp", *flags])
    return builders.build(args, [None], lambda decays=None: optax.sgd(0.0))

  # HRM's gradient is wrong through torchax without an error: its forward leans on no_grad.
  with pytest.raises(NotImplementedError, match="not compared with PyTorch"):
    build("--model=transformers:hrm_text")
  with pytest.raises(NotImplementedError, match="no pipeline"):
    build("--model=transformers:llama", "--num_stages=2")
