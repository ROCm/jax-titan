"""A MaxText model as one stage gives MaxText's loss and gradients, and trains
under FSDP as it does data-parallel; JAX-Titan's own Llama gives MaxText's."""

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
    importlib.util.find_spec("maxtext") is None, reason="needs the maxtext extra"
)

SMALL = {
    "base_emb_dim": 64, "base_mlp_dim": 128, "base_num_query_heads": 4, "base_num_kv_heads": 2,
    "head_dim": 16, "base_num_decoder_layers": 4, "vocab_size": 256, "max_target_length": 32,
    "per_device_batch_size": 8, "dtype": "float32", "weight_dtype": "float32",
    "matmul_precision": "highest",
}
# One model per decoder block: Gemma 3 with one of its global attention layers, which is every
# sixth, and two mixtures of experts with their load-balance loss on, DeepSeek's summed over
# the layers as Megatron-LM does.
WHOLE = {
    "llama3.1-8b": {},
    "mistral-7b": {},
    "gemma3-4b": {"base_num_decoder_layers": 6},
    "qwen3-8b": {},
    "qwen3-30b-a3b": {"base_moe_mlp_dim": 96, "num_experts": 8, "num_experts_per_tok": 2,
                      "load_balance_loss_weight": 0.01},
    "deepseek3-tiny": {"base_num_kv_heads": 4, "base_moe_mlp_dim": 96,
                       "load_balance_loss_weight": 0.01, "moe_use_megatron_seq_aux_loss": True},
}


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(WHOLE))
def test_whole_model_gives_maxtexts_loss_and_gradient(name):
  from flax import linen as nn  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top
  from jax_titan import ops  # pylint: disable=g-import-not-at-top
  from jax_titan import parallel  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top
  from maxtext.models import models  # pylint: disable=g-import-not-at-top
  from maxtext.trainers.pre_train import train  # pylint: disable=g-import-not-at-top

  cfg = mt.config(name, **{**SMALL, **WHOLE[name]})
  mesh = mt.mesh(cfg, jax.devices()[:1])
  model = mt.WholeModel(cfg, mesh)
  (params,) = model.init(0)
  with nn.logical_axis_rules(cfg.logical_axis_rules):
    transformer = models.Transformer(cfg, mesh, quant=None, rngs=nnx.Rngs(params=0, dropout=0))
  graph, theirs, rest = nnx.split(transformer, nnx.Param, ...)
  jax.tree.map(np.testing.assert_array_equal, params, theirs.to_pure_dict())
  # FSDP shards every parameter of these models.
  specs = parallel.tree_specs(params, model.logical_axes, {"data": 2}, fsdp=True)
  assert all("data" in spec for spec in jax.tree.leaves(specs))

  batch, seq = 8, cfg.max_target_length
  tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, (batch, seq + 1), dtype=np.int32)
  x, t = tokens[:, :-1], tokens[:, 1:]
  ones = np.ones((batch, seq), np.int32)
  data = {"inputs": x, "inputs_position": np.broadcast_to(np.arange(seq, dtype=np.int32), x.shape),
          "inputs_segmentation": ones, "targets": t, "targets_segmentation": ones}

  def maxtext_loss(params):
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      return train.loss_fn(
          nnx.merge(graph, params, rest, copy=True), cfg, dict(data), None, None)[0]

  (fn,) = model.stage_fns()
  loss, grads = jax.jit(jax.value_and_grad(fn))(params, x, t)
  ref_loss, ref_grads = jax.jit(jax.value_and_grad(maxtext_loss))(params)
  np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
  jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6),
               grads, ref_grads)
  # The same with the model's global causal attention through jax_titan.ops; DeepSeek's
  # latent attention stays in MaxText.
  (routed,) = mt.WholeModel(cfg, mesh, attention=ops.attention).stage_fns()
  ops.reset_selections()
  np.testing.assert_allclose(jax.jit(routed)(params, x, t), loss, rtol=1e-5)
  assert any(s.op == "attention" for s in ops.selections()) == (cfg.attention_type == "global")
  if cfg.num_experts > 1:
    unweighted = {**SMALL, **WHOLE[name], "load_balance_loss_weight": 0.0}
    (plain,) = mt.WholeModel(mt.config(name, **unweighted), mesh).stage_fns()
    assert float(loss) > float(jax.jit(plain)(params, x, t)) + 1e-4


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["llama3.1-8b", "llama3-8b"])
def test_native_llama_gives_the_whole_models_loss_and_gradient(name):
  """JAX-Titan's own Llama against MaxText's, from MaxText's weights: with Llama 3.1's
  RoPE, which rotates adjacent pairs and scales its frequencies, and with Llama 3's."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top

  cfg = mt.config(name, **SMALL)
  model = mt.WholeModel(cfg, mt.mesh(cfg, jax.devices()[:1]))
  (params,) = model.init(0)
  (theirs,) = model.stage_fns()
  llama_cfg = mt.llama_config(cfg)

  def ours(params, tokens, targets):
    return llama.stage_forward(
        mt.llama_params(cfg, params), tokens, targets, cfg=llama_cfg, stage=0, num_stages=1,
        remat=False, compute_dtype=jnp.float32)

  tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, (8, 33), dtype=np.int32)
  x, t = tokens[:, :-1], tokens[:, 1:]
  # Both gradients are with respect to MaxText's parameters: `ours` maps them inside.
  loss, grads = jax.jit(jax.value_and_grad(ours))(params, x, t)
  ref_loss, ref_grads = jax.jit(jax.value_and_grad(theirs))(params, x, t)
  np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
  jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6),
               grads, ref_grads)


def _losses(tmp_path, name, *flags):
  """The losses of 4 steps of `maxtext:<name>` at a small size on every GPU."""
  log = tmp_path / f"{name}-{'-'.join(flags)}.jsonl"
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", f"--model=maxtext:{name}",
       "--model_overrides=" + json.dumps({**SMALL, **WHOLE[name]}), "--data=synthetic",
       "--seq_len=32", "--num_stages=1", "--microbatches=1", "--microbatch_size=8", "--steps=4",
       "--warmup_steps=1", f"--log_file={log}", *flags],
      # os.environ, not the process environment: see conftest's ROCPROFILER note.
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return [r["loss"] for r in records.steps(records.read(log))]


@pytest.mark.gpu
def test_trainer_starts_at_the_whole_models_loss(tmp_path):
  import jax  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top

  name = "qwen3-30b-a3b"
  losses = _losses(tmp_path, name, "--parallel=dp")
  cfg = mt.config(name, **{**SMALL, **WHOLE[name]})
  model = mt.WholeModel(cfg, mt.mesh(cfg, jax.devices()[:1]))
  # The trainer's weights and its first synthetic batch, both of seed 0.
  batch = np.random.default_rng((0, 0)).integers(0, cfg.vocab_size, (8, 33), dtype=np.int32)
  (fn,) = model.stage_fns()
  loss = jax.jit(fn)(*model.init(0), batch[:, :-1], batch[:, 1:])
  np.testing.assert_allclose(losses[0], loss, rtol=1e-5)
  assert losses[-1] < losses[0]


@pytest.mark.gpu(2)
@pytest.mark.parametrize("name", ["gemma3-4b", "llama3.1-8b", "qwen3-8b", "qwen3-30b-a3b"])
def test_fsdp_trains_as_dp_does(name, tmp_path):
  fsdp = _losses(tmp_path, name, "--parallel=fsdp")
  dp = _losses(tmp_path, name, "--parallel=dp")
  np.testing.assert_allclose(fsdp, dp, rtol=1e-4)
