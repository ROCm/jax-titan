"""Training as MLPerf's Llama 3.1 8B reference does: the schedule, AdamW, the model and its
initialization, and layouts that agree under the benchmark's optimizer."""

import math
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_titan.data import megatron
from jax_titan.models import llama
from jax_titan.observability import records
from jax_titan.trainer import schedules

REPO = pathlib.Path(__file__).resolve().parents[1]


def _nemo_cosine_annealing(base_lr, warmup_steps, max_steps, min_lr, step):
  """nemo/core/optim/lr_scheduler.py at v2.1.0: WarmupAnnealHoldPolicy.get_lr with
  CosineAnnealing._get_lr and constant_steps 0, as distributed_fused_adam_with_cosine_annealing
  builds it and MegatronStrategy sets max_steps."""
  if warmup_steps > 0 and step <= warmup_steps:
    return base_lr * (step + 1) / (warmup_steps + 1)
  if step > max_steps:
    return min_lr
  mult = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (max_steps - warmup_steps)))
  return (base_lr - min_lr) * mult + min_lr


@pytest.mark.parametrize("warmup", [0, 1, 64, 128])
def test_nemo_cosine_is_nemos(warmup):
  peak, max_steps = 8e-4, 1_200_000
  schedule = schedules.nemo_cosine(peak, warmup, max_steps - warmup, peak * 0.1)
  steps = sorted({*range(0, warmup + 4), 1_000, 5_600, 600_000, max_steps - 1, max_steps,
                  max_steps + 7})
  for step in steps:
    expected = _nemo_cosine_annealing(peak, warmup, max_steps, peak * 0.1, step)
    assert float(schedule(step)) == pytest.approx(expected, rel=2e-6, abs=1e-12), step
  # The first update already moves the weights; optax's cosine starts from 0.
  assert float(schedule(0)) == pytest.approx(peak / (warmup + 1))


def test_cosine_default_is_unchanged():
  steps, warmup, peak = 50, 10, 3e-4
  old = optax.warmup_cosine_decay_schedule(0.0, peak, warmup, max(steps, warmup + 1),
                                           end_value=peak * 0.1)
  new = schedules.cosine(peak, warmup, max(steps, warmup + 1) - warmup, peak * 0.1)
  for step in range(steps + 3):
    assert float(new(step)) == float(old(step))


def _fused_adam(params, grads, lrs, b1, b2, eps, weight_decay, decays):
  """Apex FusedAdam in adam_w_mode (multi_tensor_adam.cu), which Megatron's distributed
  optimizer runs: decoupled weight decay, bias-corrected moments, epsilon outside the root."""
  params = {k: np.asarray(v, np.float64) for k, v in params.items()}
  m = {k: np.zeros_like(v) for k, v in params.items()}
  v = {k: np.zeros_like(p) for k, p in params.items()}
  for t, (g, lr) in enumerate(zip(grads, lrs), start=1):
    for k in params:
      m[k] = b1 * m[k] + (1 - b1) * g[k]
      v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k]
      update = (m[k] / (1 - b1**t)) / (np.sqrt(v[k] / (1 - b2**t)) + eps)
      if decays[k]:
        update = update + weight_decay * params[k]
      params[k] = params[k] - lr * update
  return params


def _optax_adamw(params, grads, schedule, decays):
  optimizer = optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-5, weight_decay=0.1,
                          mask={k: decays[k] for k in params})
  params = {k: jnp.asarray(v, jnp.float32) for k, v in params.items()}
  state = optimizer.init(params)
  for g in grads:
    updates, state = optimizer.update({k: jnp.asarray(x) for k, x in g.items()}, state, params)
    params = optax.apply_updates(params, updates)
  return {k: np.asarray(v) for k, v in params.items()}


def _problem(steps=6):
  rng = np.random.default_rng(0)
  params = {"w": rng.normal(0, 0.02, (16, 8)).astype(np.float32),
            "norm": np.ones(8, np.float32)}
  grads = [{k: rng.normal(0, 10.0 ** rng.integers(-7, 0), v.shape).astype(np.float32)
            for k, v in params.items()} for _ in range(steps)]
  schedule = schedules.nemo_cosine(8e-4, 2, 1_200_000 - 2, 8e-5)
  lrs = [float(schedule(s)) for s in range(steps)]
  return params, grads, schedule, lrs, {"w": True, "norm": False}


def test_adamw_is_megatrons_fused_adam():
  params, grads, schedule, lrs, decays = _problem()
  expected = _fused_adam(params, grads, lrs, 0.9, 0.95, 1e-5, 0.1, decays)
  got = _optax_adamw(params, grads, schedule, decays)
  for k in params:
    np.testing.assert_allclose(got[k], expected[k], rtol=1e-6, atol=1e-9)


def test_adamw_matches_torch():
  torch = pytest.importorskip("torch")
  params, grads, schedule, lrs, decays = _problem()
  tensors = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
  optimizer = torch.optim.AdamW(
      [{"params": [tensors["w"]], "weight_decay": 0.1},
       {"params": [tensors["norm"]], "weight_decay": 0.0}],
      lr=lrs[0], betas=(0.9, 0.95), eps=1e-5)
  for g, lr in zip(grads, lrs):
    for group in optimizer.param_groups:
      group["lr"] = lr
    for k, t in tensors.items():
      t.grad = torch.tensor(g[k])
    optimizer.step()
  got = _optax_adamw(params, grads, schedule, decays)
  for k, t in tensors.items():
    np.testing.assert_allclose(got[k], t.detach().numpy(), rtol=1e-6, atol=1e-9)


def test_llama31_8b_is_the_references():
  cfg = llama.FLAVORS["llama3.1-8b"]
  # Meta-Llama-3.1-8B's parameters, with separate embedding and output matrices and
  # 128,256 vocabulary rows, as NeMo builds the model from Llama 3.1's tokenizer.
  assert cfg.num_params() == 8_030_261_248
  assert (cfg.vocab_size, cfg.emb_dim, cfg.num_layers, cfg.mlp_dim) == (128256, 4096, 32, 14336)
  assert (cfg.num_query_heads, cfg.num_kv_heads, cfg.head_dim) == (32, 8, 128)
  assert (cfg.norm_eps, cfg.rope_theta) == (1e-5, 500_000.0)
  s = cfg.rope_scaling
  assert (s.factor, s.low_freq_factor, s.high_freq_factor, s.original_context) == (8, 1, 4, 8192)


def test_initialization_is_the_references():
  """Normal(0.02), the output projections Normal(0.02 / sqrt(2 x 32)) as Megatron scales
  them for 32 layers, and norm weights 1, at Llama 3.1 8B's depth and a small width."""
  cfg = llama.LlamaConfig(vocab_size=4096, emb_dim=256, num_layers=32, num_query_heads=8,
                          num_kv_heads=2, head_dim=32, mlp_dim=512)
  params = llama.init_stage_params(cfg, jax.random.key(0), stage=0, num_stages=1)
  out = 0.02 / math.sqrt(2 * 32)
  expected = {"wq": 0.02, "wk": 0.02, "wv": 0.02, "w_gate": 0.02, "w_up": 0.02, "wo": out,
              "w_down": out}
  for name, std in expected.items():
    w = np.asarray(params["layers"][name])
    assert abs(w.mean()) < 0.02 * std and abs(w.std() / std - 1) < 0.02, name
  for name in ("embed", "lm_head"):
    w = np.asarray(params[name])
    assert abs(w.std() / 0.02 - 1) < 0.02, name
  for name in ("attn_norm", "mlp_norm"):
    assert np.all(np.asarray(params["layers"][name]) == 1.0)
  assert np.all(np.asarray(params["final_norm"]) == 1.0)


def test_llama_matches_transformers():
  """JAX-Titan's Llama against transformers' LlamaForCausalLM with Llama 3.1's RoPE scaling,
  the same weights, in fp32: the loss and every gradient."""
  torch = pytest.importorskip("torch")
  transformers = pytest.importorskip("transformers")
  cfg = llama.LlamaConfig(vocab_size=256, emb_dim=64, num_layers=2, num_query_heads=8,
                          num_kv_heads=2, head_dim=8, mlp_dim=128,
                          rope_scaling=llama.Llama31RopeScaling(original_context=32))
  hf_cfg = transformers.LlamaConfig(
      vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
      num_attention_heads=8, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-5,
      rope_theta=500_000.0, max_position_embeddings=256, tie_word_embeddings=False,
      rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0, "original_max_position_embeddings": 32})
  hf_cfg._attn_implementation = "eager"  # pylint: disable=protected-access
  torch.manual_seed(0)
  model = transformers.LlamaForCausalLM(hf_cfg).float()
  layers = model.model.layers

  def stack(get, transpose=True):
    return jnp.stack([jnp.asarray((get(l).T if transpose else get(l)).detach().numpy())
                      for l in layers])

  params = {
      "embed": jnp.asarray(model.model.embed_tokens.weight.detach().numpy()),
      "final_norm": jnp.asarray(model.model.norm.weight.detach().numpy()),
      "lm_head": jnp.asarray(model.lm_head.weight.detach().numpy().T),
      "layers": {
          "attn_norm": stack(lambda l: l.input_layernorm.weight, False),
          "wq": stack(lambda l: l.self_attn.q_proj.weight),
          "wk": stack(lambda l: l.self_attn.k_proj.weight),
          "wv": stack(lambda l: l.self_attn.v_proj.weight),
          "wo": stack(lambda l: l.self_attn.o_proj.weight),
          "mlp_norm": stack(lambda l: l.post_attention_layernorm.weight, False),
          "w_gate": stack(lambda l: l.mlp.gate_proj.weight),
          "w_up": stack(lambda l: l.mlp.up_proj.weight),
          "w_down": stack(lambda l: l.mlp.down_proj.weight),
      },
  }
  tokens = np.random.default_rng(0).integers(0, 256, (2, 65))
  inputs, targets = tokens[:, :-1], tokens[:, 1:]

  def loss_fn(p):
    return llama.stage_forward(p, jnp.asarray(inputs), jnp.asarray(targets), cfg=cfg, stage=0,
                               num_stages=1, remat=False, compute_dtype=jnp.float32)

  loss, grads = jax.value_and_grad(loss_fn)(params)
  logits = model(input_ids=torch.tensor(inputs)).logits
  hf_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256),
                                              torch.tensor(targets).reshape(-1))
  hf_loss.backward()
  assert float(loss) == pytest.approx(float(hf_loss.detach()), rel=1e-5)

  def check(got, reference, transpose=True):
    ref = reference.grad.detach().numpy()
    np.testing.assert_allclose(np.asarray(got), ref.T if transpose else ref,
                               rtol=1e-4, atol=1e-6 * np.abs(ref).max())

  check(grads["embed"], model.model.embed_tokens.weight, False)
  check(grads["lm_head"], model.lm_head.weight)
  check(grads["final_norm"], model.model.norm.weight, False)
  for i, layer in enumerate(layers):
    g = {k: v[i] for k, v in grads["layers"].items()}
    check(g["attn_norm"], layer.input_layernorm.weight, False)
    check(g["mlp_norm"], layer.post_attention_layernorm.weight, False)
    for name, module in (("wq", layer.self_attn.q_proj), ("wk", layer.self_attn.k_proj),
                         ("wv", layer.self_attn.v_proj), ("wo", layer.self_attn.o_proj),
                         ("w_gate", layer.mlp.gate_proj), ("w_up", layer.mlp.up_proj),
                         ("w_down", layer.mlp.down_proj)):
      check(g[name], module.weight)


def _losses(tmp_path, *flags):
  log = tmp_path / f"{'_'.join(f.strip('-').replace('=', '_') for f in flags)}.jsonl"
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", "--platform=cpu", "--model=llama3-test",
       "--data=synthetic", "--seq_len=64", "--microbatches=2", "--microbatch_size=8",
       "--steps=4", "--warmup_steps=2", "--lr=1e-3", "--lr_schedule=nemo_cosine",
       "--lr_decay_steps=1199998", "--adam_eps=1e-5", "--step_timeout=0",
       f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO)
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return [r["loss"] for r in records.steps(records.read(log))]


def test_layouts_agree_under_mlperf_adamw(tmp_path):
  dp = _losses(tmp_path, "--parallel=dp")
  assert len(dp) == 4 and dp[-1] < dp[0]
  np.testing.assert_allclose(_losses(tmp_path, "--parallel=fsdp"), dp, rtol=1e-5)
  np.testing.assert_allclose(_losses(tmp_path, "--parallel=mpmd", "--num_stages=2",
                                     "--schedule=ONE_FWD_ONE_BWD"), dp, rtol=1e-3)


def _megatron_data(tmp_path):
  rng = np.random.default_rng(0)
  docs = [rng.integers(0, 1000, n) for n in rng.integers(1, 300, 400)]
  megatron.write(tmp_path / "train", docs)
  megatron.write(tmp_path / "valid", docs[:60])
  return [f"--data={tmp_path / 'train'}", f"--eval_data={tmp_path / 'valid'}"]


def _train(tmp_path, name, *flags, check=True):
  log = tmp_path / f"{name}.jsonl"
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", "--platform=cpu", "--model=llama3-test",
       "--parallel=dp", "--seq_len=32", "--microbatches=2", "--microbatch_size=8",
       "--warmup_steps=1", "--step_timeout=0", "--seed=3", f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO,
      env={**__import__("os").environ, "JAX_TITAN_ROOT": str(tmp_path / "root")})
  if check:
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return result, (records.read(log) if log.exists() else [])


def test_validation_stops_at_the_target(tmp_path):
  data = _megatron_data(tmp_path)
  _, log = _train(tmp_path, "target", *data, "--steps=10", "--eval_every=2",
                  "--eval_samples=24", "--eval_batch_size=8", "--target_eval_loss=100")
  evals = records.events(log, "eval")
  assert [(e["step"], e["samples"], e["eval_samples"]) for e in evals] == [(2, 32, 24)]
  end = records.events(log, "end")[0]
  assert end["target_reached"] and end["steps_done"] == 2 and end["status"] == "completed"
  assert len(records.steps(log)) == 2
  # Below any loss it reaches, the run trains every step and validates at the cadence.
  _, log = _train(tmp_path, "miss", *data, "--steps=4", "--eval_every=2", "--eval_samples=16",
                  "--target_eval_loss=0.001")
  assert [e["step"] for e in records.events(log, "eval")] == [2, 4]
  assert records.events(log, "end")[0]["target_reached"] is False


def test_validation_loss_is_the_mean_over_tokens(tmp_path):
  """With the weights fixed (--lr=0), the loss over the same 48 sequences does not depend
  on how they are batched."""
  data = _megatron_data(tmp_path)
  losses = []
  for size in (8, 16, 48):
    _, log = _train(tmp_path, f"batch{size}", *data, "--steps=1", "--lr=0", "--eval_every=1",
                    "--eval_samples=48", f"--eval_batch_size={size}")
    losses.append(records.events(log, "eval")[0]["loss"])
  np.testing.assert_allclose(losses[1:], [losses[0]] * 2, rtol=1e-5)


@pytest.mark.parametrize("layout", [["--parallel=dp"], ["--parallel=mpmd", "--num_stages=2"]])
def test_synthetic_warmup_trains_from_the_initial_state(tmp_path, layout):
  """Warm-up steps and a validation pass on synthetic tokens leave no trace: the run
  then trains and validates as one without them, and its clock starts after them."""
  data = _megatron_data(tmp_path)
  flags = [*data, *layout, "--steps=4", "--eval_every=2", "--eval_samples=16",
           "--eval_batch_size=16"]
  _, cold = _train(tmp_path, "cold", *flags)
  _, warm = _train(tmp_path, "warm", *flags, "--synthetic_warmup=2")
  assert [r["loss"] for r in records.steps(warm)] == [r["loss"] for r in records.steps(cold)]
  assert ([e["loss"] for e in records.events(warm, "eval")] ==
          [e["loss"] for e in records.events(cold, "eval")])
  events = [r.get("event") for r in warm]
  assert events.index("warmup") < events.index("run_start") < events.index("eval")
  assert records.events(warm, "warmup")[0]["steps"] == 2


@pytest.mark.parametrize("flags,message", [
    (["--target_eval_loss=3.3"], "needs validation passes"),
    (["--eval_every=1", "--eval_samples=20", "--eval_batch_size=8"], "not whole batches of 8"),
    (["--eval_every=1", "--parallel=mpmd", "--eval_batch_size=8"],
     "validates batches of one step's 16 sequences"),
    (["--model=command-a-plus", "--synthetic_warmup=1"], "--synthetic_warmup is for Llama"),
])
def test_validation_flags_are_checked(tmp_path, flags, message):
  result, _ = _train(tmp_path, "bad", "--data=synthetic", "--steps=1", *flags, check=False)
  assert result.returncode != 0 and message in result.stderr
