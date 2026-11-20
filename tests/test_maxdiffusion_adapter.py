"""MaxDiffusion FLUX.1 stages match MaxDiffusion and single-mesh `jax.jit`; FLUX.1 and
Wan 2.1 called whole match the stages and MaxDiffusion's own module."""

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
SMALL_FLUX = {
    "in_channels": 16, "num_layers": 2, "num_single_layers": 2, "attention_head_dim": 16,
    "num_attention_heads": 2, "joint_attention_dim": 32, "pooled_projection_dim": 16,
    "axes_dims_rope": (4, 6, 6), "dtype": "float32",
}
SMALL_WAN = {
    "num_attention_heads": 2, "attention_head_dim": 24, "in_channels": 4, "out_channels": 4,
    "text_dim": 32, "freq_dim": 32, "ffn_dim": 96, "num_layers": 2, "rope_max_seq_len": 32,
    "dtype": "float32",
}

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("maxdiffusion") is None, reason="needs MaxDiffusion"
)


@pytest.mark.parametrize("flags", [
    [],
    ["--blocks=3,1"],
    ["--num_stages=4", "--microbatches=8", "--schedule=GPIPE"],
    ["--flavor=flux-schnell"],
], ids=lambda f: " ".join(f) or "flux-dev")
def test_maxdiffusion_flux_parity(flags):
  result = subprocess.run(
      [sys.executable, str(REPO / "examples/maxdiffusion_flux_parity.py"), "--platform=cpu",
       *flags],
      capture_output=True, text=True, check=False,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  assert "match the transformer" in result.stdout
  assert "matches the single-mesh reference" in result.stdout


def test_flux_kernels_decay():
  """FLUX.1 keeps the Dense of its single blocks' adaptive norm under a module named `norm`."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxdiffusion as md  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import train  # pylint: disable=g-import-not-at-top

  flux = md.transformer(
      "flux-dev", in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=16,
      num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=16,
      axes_dims_rope=(4, 6, 6), dtype=jnp.float32, weights_dtype=jnp.float32)
  stage = md.Model(flux, num_stages=1).abstract()
  mask = train._decay_mask(stage)  # pylint: disable=protected-access
  paths = {tuple(getattr(k, "key", None) for k in path): decays
           for path, decays in jax.tree_util.tree_flatten_with_path(mask)[0]}
  assert any(path[-3:-1] == ("norm", "lin") for path in paths)
  assert {path[-1] for path, decays in paths.items() if not decays} <= {"scale"}


@pytest.mark.gpu
def test_whole_flux_gives_the_stages_loss_and_gradient():
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import parallel  # pylint: disable=g-import-not-at-top
  from jax_titan.data import diffusion  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxdiffusion as md  # pylint: disable=g-import-not-at-top

  flux = md.transformer(
      "flux-dev", **{**SMALL_FLUX, "dtype": jnp.float32}, weights_dtype=jnp.float32)
  # `FluxTransformer2DModel.__call__` against the one stage that reimplements it.
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
  whole, stages = md.whole_flux(flux, mesh), md.Model(flux, num_stages=1)
  (params,) = whole.init(0)
  jax.tree.map(np.testing.assert_array_equal, params, stages.init(0)[0])
  batches = diffusion.flux(
      0, (1, 2), image_side=4, text_tokens=8, in_channels=16, joint_attention_dim=32,
      pooled_projection_dim=16, guidance_embeds=True)
  x, target = batches.next_batch(0)
  x, target = {k: v[0] for k, v in x.items()}, target[0]
  loss, grads = jax.jit(jax.value_and_grad(whole.stage_fns()[0]))(params, x, target)
  ref_loss, ref_grads = jax.jit(jax.value_and_grad(stages.stage_fns()[0]))(params, x, target)
  np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
  jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6),
               grads, ref_grads)
  # FSDP shards every parameter, and a stack of blocks never along its block axis.
  specs = parallel.tree_specs(params, whole.logical_axes, {"data": 2}, fsdp=True)
  for path, spec in jax.tree_util.tree_flatten_with_path(specs)[0]:
    assert "data" in spec
    if any(getattr(k, "key", None) in md._STACKS for k in path):  # pylint: disable=protected-access
      assert spec[0] is None


@pytest.mark.gpu
def test_whole_wan_gives_wanmodels_output():
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from flax.linen import partitioning as nn_partitioning  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.data import diffusion  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxdiffusion as md  # pylint: disable=g-import-not-at-top
  from maxdiffusion.models.wan.transformers.transformer_wan import WanModel  # pylint: disable=g-import-not-at-top

  kwargs = {**SMALL_WAN, "dtype": jnp.float32}
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
  whole = md.whole_wan(mesh, **kwargs)
  (params,) = whole.init(0)
  batches = diffusion.wan(0, (1, 2), frames=2, height=4, width=4, patch=(1, 2, 2), text_tokens=8,
                          in_channels=4, text_dim=32)
  x, target = batches.next_batch(0)
  x, target = {k: v[0] for k, v in x.items()}, target[0]
  loss = jax.jit(whole.stage_fns()[0])(params, x, target)
  # MaxDiffusion's own module, built and called directly.
  model = WanModel(rngs=nnx.Rngs(0), **kwargs)
  with mesh, nn_partitioning.axis_rules(md._AXIS_RULES):  # pylint: disable=protected-access
    pred = model(hidden_states=x["latents"], timestep=x["timestep"],
                 encoder_hidden_states=x["txt"], deterministic=True)
  np.testing.assert_allclose(loss, jnp.mean((target - pred) ** 2), rtol=1e-5)


def _losses(tmp_path, model, overrides, *flags):
  """The losses of 4 steps of a small `maxdiffusion:<model>` on every GPU."""
  log = tmp_path / f"{model}-{'-'.join(flags)}.jsonl"
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", f"--model=maxdiffusion:{model}",
       "--model_overrides=" + json.dumps(overrides), "--resolution=64", "--frames=5",
       "--text_tokens=8", "--num_stages=1", "--microbatches=1", "--microbatch_size=8",
       "--steps=4", "--warmup_steps=1", f"--log_file={log}", *flags],
      # os.environ, not the process environment: see conftest's ROCPROFILER note.
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return [r["loss"] for r in records.steps(records.read(log))]


@pytest.mark.gpu(2)
@pytest.mark.parametrize("model, overrides", [
    ("flux-dev", SMALL_FLUX), ("wan2.1-t2v-1.3b", SMALL_WAN)], ids=["flux-dev", "wan2.1"])
def test_fsdp_trains_as_dp_does(model, overrides, tmp_path):
  # For FLUX.1, fsdp calls the transformer whole and dp runs its one stage.
  np.testing.assert_allclose(
      _losses(tmp_path, model, overrides, "--parallel=fsdp"),
      _losses(tmp_path, model, overrides, "--parallel=dp"), rtol=1e-4)
