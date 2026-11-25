"""FLUX.1 matches diffusers' model at the same weights, and trains alike under dp and FSDP."""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_titan import parallel
from jax_titan.data import diffusion
from jax_titan.models import flux
from jax_titan.observability import records
from jax_titan.trainer import builders, train

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = {
    "in_channels": 16, "num_layers": 2, "num_single_layers": 2, "attention_head_dim": 16,
    "num_attention_heads": 2, "joint_attention_dim": 32, "pooled_projection_dim": 16,
    "axes_dims_rope": (4, 6, 6),
}
FLAGS = [
    "--platform=cpu", "--model=flux-dev", "--resolution=64", "--text_tokens=8",
    "--model_overrides=" + json.dumps(CONFIG | {"dtype": "float32"}), "--microbatches=2",
    "--microbatch_size=8", "--steps=4", "--warmup_steps=1",
]


# JAX-Titan's parameters -> diffusers' module, in a stream of a double block and in a single block.
_IMG = {("wq", "bq"): "attn.to_q", ("wk", "bk"): "attn.to_k", ("wv", "bv"): "attn.to_v",
        "q_norm": "attn.norm_q", "k_norm": "attn.norm_k", ("mod_w", "mod_b"): "norm1.linear",
        ("wo", "bo"): "attn.to_out.0", ("mlp_w1", "mlp_b1"): "ff.net.0.proj",
        ("mlp_w2", "mlp_b2"): "ff.net.2"}
_TXT = {("wq", "bq"): "attn.add_q_proj", ("wk", "bk"): "attn.add_k_proj",
        ("wv", "bv"): "attn.add_v_proj", "q_norm": "attn.norm_added_q",
        "k_norm": "attn.norm_added_k", ("mod_w", "mod_b"): "norm1_context.linear",
        ("wo", "bo"): "attn.to_add_out", ("mlp_w1", "mlp_b1"): "ff_context.net.0.proj",
        ("mlp_w2", "mlp_b2"): "ff_context.net.2"}
_SINGLE = {("wq", "bq"): "attn.to_q", ("wk", "bk"): "attn.to_k", ("wv", "bv"): "attn.to_v",
           "q_norm": "attn.norm_q", "k_norm": "attn.norm_k", ("mod_w", "mod_b"): "norm.linear",
           ("mlp_w", "mlp_b"): "proj_mlp", ("out_w", "out_b"): "proj_out"}


def _from_diffusers(tensors: dict, cfg: flux.FluxConfig) -> dict:
  """Weights, or their gradients, of diffusers' model by name, in the layout of
  `flux.init_params`."""
  w = {k: np.asarray(v) for k, v in tensors.items()}
  d = cfg.attention_head_dim
  # A head's query and key features, from diffusers' interleaved pairs to split halves.
  halves = np.concatenate([np.arange(0, d, 2), np.arange(1, d, 2)])

  def heads(x):
    return x.reshape(x.shape[:-1] + (-1, d))[..., halves].reshape(x.shape)

  def dense(name, rope=False):
    kernel, bias = w[f"{name}.weight"].T, w[f"{name}.bias"]
    return (heads(kernel), heads(bias)) if rope else (kernel, bias)

  def stacked(prefix, n, modules):
    blocks = []
    for i in range(n):
      p = {}
      for key, module in modules.items():
        if isinstance(key, str):
          p[key] = heads(w[f"{prefix}.{i}.{module}.weight"])
        else:
          p[key[0]], p[key[1]] = dense(f"{prefix}.{i}.{module}", rope=key[0] in ("wq", "wk"))
      blocks.append(p)
    return jax.tree.map(lambda *xs: np.stack(xs), *blocks)

  def embedder(name):
    (w1, b1), (w2, b2) = (dense(f"time_text_embed.{name}.linear_{i}") for i in (1, 2))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

  params = {
      "double": {s: stacked("transformer_blocks", cfg.num_layers, m)
                 for s, m in (("img", _IMG), ("txt", _TXT))},
      "single": stacked("single_transformer_blocks", cfg.num_single_layers, _SINGLE),
      "timestep_embedder": embedder("timestep_embedder"),
      "text_embedder": embedder("text_embedder"),
  }
  if cfg.guidance_embeds:
    params["guidance_embedder"] = embedder("guidance_embedder")
  for name, module in (("x_embedder", "x_embedder"), ("context_embedder", "context_embedder"),
                       ("norm_out", "norm_out.linear"), ("proj_out", "proj_out")):
    params[name] = dict(zip("wb", dense(module)))
  return params


@pytest.mark.gpu
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("diffusers") is None,
    reason="needs the diffusers extra")
@pytest.mark.parametrize("guidance_embeds", [True, False], ids=["dev", "schnell"])
def test_matches_diffusers(guidance_embeds):
  import torch  # pylint: disable=g-import-not-at-top
  from diffusers import FluxTransformer2DModel  # pylint: disable=g-import-not-at-top

  cfg = flux.FluxConfig(**CONFIG, guidance_embeds=guidance_embeds)
  torch.manual_seed(0)
  module = FluxTransformer2DModel(**CONFIG, guidance_embeds=guidance_embeds)
  with torch.no_grad():
    for name, p in module.named_parameters():
      if ".norm_" in name:
        p.normal_(1.0, 0.1)
  params = _from_diffusers(module.state_dict(), cfg)
  shapes = jax.eval_shape(lambda: flux.init_params(cfg, jax.random.key(0)))
  assert jax.tree.map(np.shape, params) == jax.tree.map(lambda x: x.shape, shapes)

  batches = diffusion.flux(
      0, (1, 2), image_side=4, text_tokens=8, in_channels=cfg.in_channels,
      joint_attention_dim=cfg.joint_attention_dim,
      pooled_projection_dim=cfg.pooled_projection_dim, guidance_embeds=guidance_embeds)
  x, target = batches.next_batch(0)
  # As the layout hands a step its batch: without the microbatch axis.
  x, target = {k: v[0] for k, v in x.items()}, target[0]
  names = {"img": "hidden_states", "txt": "encoder_hidden_states",
           "pooled": "pooled_projections"}
  pred = module(**{names.get(k, k): torch.from_numpy(np.array(v)) for k, v in x.items()},
                return_dict=False)[0]
  loss = torch.mean((torch.from_numpy(target) - pred) ** 2)
  loss.backward()
  reference = _from_diffusers({n: p.grad for n, p in module.named_parameters()}, cfg)

  model = flux.Model(cfg, dtype=jnp.float32)
  np.testing.assert_allclose(
      jax.jit(model.forward)(params, x), pred.detach().numpy(), rtol=1e-5, atol=1e-5)
  (fn,) = model.stage_fns()
  value, grads = jax.jit(jax.value_and_grad(fn))(params, x, target)
  np.testing.assert_allclose(value, loss.item(), rtol=1e-5)
  for (path, grad), want in zip(jax.tree_util.tree_flatten_with_path(grads)[0],
                                jax.tree.leaves(reference)):
    np.testing.assert_allclose(grad, want, rtol=1e-3, atol=1e-4 * np.abs(want).max(),
                               err_msg=str(path))


def test_every_parameter_is_sharded_and_only_norms_do_not_decay():
  for name, cfg in flux.FLAVORS.items():
    params = jax.eval_shape(lambda c=cfg: flux.init_params(c, jax.random.key(0)))
    specs = parallel.tree_specs(params, flux.logical_axes, {"data": 8, "tensor": 1}, fsdp=True)
    for path, spec in jax.tree_util.tree_flatten_with_path(specs)[0]:
      assert "data" in spec, (name, path)
      if path[0].key in ("double", "single"):
        assert spec[0] is None, (name, path)
    kept = {path[-1].key for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
            if not flux.decays(path, leaf)}
    assert kept == {"q_norm", "k_norm"}, name
  # The parameter count of the published FLUX.1-dev, 11.90B.
  dev = jax.eval_shape(lambda: flux.init_params(flux.FLAVORS["flux-dev"], jax.random.key(0)))
  assert sum(x.size for x in jax.tree.leaves(dev)) == 11_901_408_320


def test_initializes_as_pytorch_initializes_diffusers():
  cfg = flux.FluxConfig(**CONFIG, guidance_embeds=True)
  params = jax.device_get(flux.init_params(cfg, jax.random.key(0)))
  leaves = {tuple(k.key for k in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]}
  scaled = {"weights": [], "biases": []}
  for path, leaf in leaves.items():
    *parent, name = path
    if name.endswith("_norm"):
      np.testing.assert_array_equal(leaf, 1)
      continue
    # A bias takes the fan-in of its weight: b of w, bq of wq, mod_b of mod_w and so on.
    weight = leaves[(*parent, name.replace("b", "w", 1))]
    x = leaf * weight.shape[-2] ** 0.5
    assert np.abs(x).max() <= 1, path
    scaled["weights" if weight is leaf else "biases"].append(x.ravel())
  # Uniform within 1/sqrt(fan-in) of zero: scaled by sqrt(fan-in), the spread of U(-1, 1).
  for kind, values in scaled.items():
    assert np.std(np.concatenate(values)) == pytest.approx(3 ** -0.5, rel=0.05), kind


def _run(log, *flags):
  # os.environ, not the process environment: see conftest's ROCPROFILER note.
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", *FLAGS, f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r for r in records.steps(records.read(log))}, result.stdout


@pytest.mark.parametrize("flags", [[], ["--model=flux-schnell", "--no-remat"]],
                         ids=["dev", "schnell-without-remat"])
def test_fsdp_matches_data_parallel(flags, tmp_path):
  def run(parallel):
    log = tmp_path / f"{parallel}.jsonl"
    steps, _ = _run(log, *flags, f"--parallel={parallel}", "--eval_every=4", "--eval_batches=1")
    return steps, [r["loss"] for r in records.events(records.read(log), "eval")]

  (reference, validation), (trained, trained_validation) = run("dp"), run("fsdp")
  for step in range(4):
    assert trained[step]["loss"] == pytest.approx(reference[step]["loss"], rel=1e-5), step
    assert trained[step]["grad_norm"] == pytest.approx(reference[step]["grad_norm"], rel=1e-4), step
  assert trained_validation == pytest.approx(validation, rel=1e-5)


@pytest.mark.parametrize("parallel_flag", ["--parallel=dp", "--parallel=fsdp"])
def test_resume_is_exact(parallel_flag, tmp_path):
  flags = [parallel_flag, "--checkpoint_every=2"]
  full, _ = _run(tmp_path / "full.jsonl", *flags, f"--checkpoint_dir={tmp_path / 'full'}")
  (tmp_path / "resumed").mkdir()
  shutil.copytree(tmp_path / "full" / "2", tmp_path / "resumed" / "2")
  resumed, out = _run(
      tmp_path / "resumed.jsonl", *flags, f"--checkpoint_dir={tmp_path / 'resumed'}")
  assert "restored step 2" in out
  assert sorted(resumed) == [2, 3]
  for step in (2, 3):
    assert resumed[step]["loss"] == full[step]["loss"], step


@pytest.mark.parametrize("flags,error", [
    # main() reduces a staged model's gradients per microbatch unless told otherwise.
    (["--parallel=mpmd", "--reduce_grads=microbatch"], "has no pipeline"),
    (["--parallel=dp", "--num_stages=2"], "has no pipeline"),
    (["--parallel=dp", "--init_from=/nonexistent"], "--init_from loads"),
    (["--parallel=dp", "--data=tokens.bin"], "synthetic batches only"),
    (["--parallel=dp", "--tensor_parallel=2"], "neither --stage_fsdp nor --tensor_parallel"),
])
def test_refuses(flags, error):
  args = train.parse_args(["--model=flux-dev", *flags])
  with pytest.raises(NotImplementedError, match=error):
    builders.build(args, [None], lambda decays=None: optax.sgd(0.0))


def test_runs_as_one_stage_by_default():
  assert train.parse_args(["--model=flux-schnell"]).num_stages == 1
  assert train.parse_args(["--model=ideogram4"]).num_stages == 2


@pytest.mark.gpu(2)
def test_fsdp_trains_as_dp_does_on_gpus(tmp_path):
  flags = [f for f in FLAGS if f != "--platform=cpu"]

  def losses(*more):
    log = tmp_path / f"{more[0][2:]}.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "jax_titan.trainer.train", *flags, f"--log_file={log}", *more],
        capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return [r["loss"] for r in records.steps(records.read(log))]

  np.testing.assert_allclose(losses("--parallel=fsdp"), losses("--parallel=dp"), rtol=1e-4)
