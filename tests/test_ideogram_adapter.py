"""Ideogram 4 stages match the Ideogram branch of MaxDiffusion, its layouts match each other,
and JAX-Titan's own Ideogram 4 matches the branch's model and its trainer's step."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

from jax_titan.models.config import AdapterConfig
from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]
# The rotary sections end below the head's 16 frequencies, as they do at full size.
CONFIG = {
    "emb_dim": 64, "num_heads": 2, "in_channels": 16, "llm_features_dim": 48, "adanln_dim": 32,
    "intermediate_size": 128, "num_layers": 4, "mrope_section": (4, 3, 2),
}
FLAGS = [
    "--platform=cpu", "--model=maxdiffusion:ideogram4", "--resolution=64", "--text_tokens=8",
    "--model_overrides=" + json.dumps(CONFIG), "--dtype=float32", "--microbatches=4",
    "--microbatch_size=4",
    "--steps=4", "--warmup_steps=1",
]


def _has_ideogram() -> bool:
  """Whether MaxDiffusion is installed from its Ideogram branch, without importing it."""
  spec = importlib.util.find_spec("maxdiffusion")
  return spec is not None and any(
      (pathlib.Path(root) / "models/ideogram").is_dir()
      for root in spec.submodule_search_locations or ())


pytestmark = pytest.mark.skipif(not _has_ideogram(), reason="needs MaxDiffusion's Ideogram branch")


@pytest.mark.parametrize("num_stages,blocks", [(1, None), (2, None), (3, (1, 2, 1))])
def test_stages_match_the_transformer(num_stages, blocks):
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import ideogram as ig  # pylint: disable=g-import-not-at-top

  cfg = ig.Ideogram4Config(**CONFIG)
  model = ig.AdapterModel(AdapterConfig("maxdiffusion", "ideogram4"), cfg,
                          num_stages=num_stages, blocks=blocks, dtype=jnp.float32)
  transformer = ig.Ideogram4Transformer(nnx.Rngs(0), cfg, dtype=jnp.float32)
  stage_params = model.split(nnx.state(transformer, nnx.Param).to_pure_dict())
  for a, b in zip(jax.tree.leaves(stage_params), jax.tree.leaves(model.init(0))):
    np.testing.assert_array_equal(a, b)

  batch, side, text_tokens = 2, 4, 8
  rng = np.random.default_rng(0)
  ids = ig.sequence_ids(side, text_tokens)
  # The second sample's text is left-padded by three positions.
  ids = {k: np.stack([v, v]) for k, v in ids.items()}
  ids["segment_ids"][1, :3] = ig.constants.SEQUENCE_PADDING_INDICATOR
  ids["indicator"][1, :3] = 0
  x = {
      "img": rng.normal(size=(batch, side * side, cfg.in_channels)).astype(np.float32),
      "txt": rng.normal(size=(batch, text_tokens, cfg.llm_features_dim)).astype(np.float32),
      "timestep": rng.uniform(size=batch).astype(np.float32),
      **ids,
  }
  # The transformer takes both inputs over the whole sequence, zero where the other one is.
  llm = np.pad(x["txt"], ((0, 0), (0, side * side), (0, 0)))
  latents = np.pad(x["img"], ((0, 0), (text_tokens, 0), (0, 0)))
  ref = jax.jit(lambda *a: transformer(*a))(
      llm, latents, x["timestep"], x["position_ids"], x["segment_ids"], x["indicator"])

  def chain(stage_params, x):
    for i in range(num_stages):
      x = model.forward(i, stage_params[i], x)
    return x

  pred = jax.jit(chain)(stage_params, x)
  np.testing.assert_allclose(np.asarray(pred), np.asarray(ref), rtol=1e-5, atol=1e-5)


def _models(num_stages=1, blocks=None):
  """The branch's model as stages, JAX-Titan's own, and one set of weights in both layouts."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models import ideogram  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import ideogram as ig  # pylint: disable=g-import-not-at-top

  cfg = ig.Ideogram4Config(**CONFIG)
  adapter = ig.AdapterModel(AdapterConfig("maxdiffusion", "ideogram4"), cfg,
                            num_stages=num_stages, blocks=blocks, dtype=jnp.float32)
  jt_model = ideogram.Model(
      ideogram.IdeogramConfig(
          in_channels=cfg.in_channels, num_layers=cfg.num_layers,
          attention_head_dim=cfg.emb_dim // cfg.num_heads, num_attention_heads=cfg.num_heads,
          intermediate_size=cfg.intermediate_size, adaln_dim=cfg.adanln_dim,
          llm_features_dim=cfg.llm_features_dim, rope_theta=cfg.rope_theta,
          mrope_section=cfg.mrope_section, norm_eps=cfg.norm_eps),
      num_stages=num_stages, blocks=blocks, dtype=jnp.float32)
  params = adapter.init(0)
  return adapter, jt_model, params, ig.jax_titan_params(params)


def _batch(cfg, batch=2, side=4, text_tokens=8):
  """Inputs whose second sample's text is left-padded by three positions, and clean latents."""
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import ideogram as ig  # pylint: disable=g-import-not-at-top

  rng = np.random.default_rng(0)
  ids = {k: np.stack([v] * batch) for k, v in ig.sequence_ids(side, text_tokens).items()}
  ids["segment_ids"][1, :3] = ig.constants.SEQUENCE_PADDING_INDICATOR
  ids["indicator"][1, :3] = 0
  x = {
      "img": rng.normal(size=(batch, side * side, cfg.in_channels)).astype(np.float32),
      "txt": rng.normal(size=(batch, text_tokens, cfg.llm_features_dim)).astype(np.float32),
      "timestep": rng.uniform(size=batch).astype(np.float32),
      **ids,
  }
  return x, rng.normal(size=x["img"].shape).astype(np.float32)


def _loss_and_grads(model, stage_params, x, target):
  import jax  # pylint: disable=g-import-not-at-top

  def loss(stage_params):
    fns, h = model.stage_fns(), x
    for fn, p in zip(fns[:-1], stage_params[:-1]):
      h = fn(p, h)
    return fns[-1](stage_params[-1], h, target)

  return jax.jit(jax.value_and_grad(loss))(stage_params)


def _assert_trees_close(a, b, rtol=1e-4, atol=1e-6):
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top

  jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol), a, b)


@pytest.mark.parametrize("num_stages,blocks", [(1, None), (3, (1, 2, 1))])
def test_jax_titan_model_matches(num_stages, blocks):
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import ideogram as ig  # pylint: disable=g-import-not-at-top

  adapter, jt_model, params, jt_params = _models(num_stages, blocks)
  x, target = _batch(adapter.cfg)

  def boundaries(model, stage_params):
    out, h = [], x
    for i in range(num_stages):
      h = model.forward(i, stage_params[i], h)
      out.append(h if i == num_stages - 1 else (h[0], h[-1]))
    return out

  # The hidden states and the conditioning after every stage, and the velocity after the last.
  _assert_trees_close(jax.jit(lambda p: boundaries(jt_model, p))(jt_params),
                      jax.jit(lambda p: boundaries(adapter, p))(params), rtol=1e-5, atol=1e-5)

  ref_loss, ref_grads = _loss_and_grads(adapter, params, x, target)
  loss, grads = _loss_and_grads(jt_model, jt_params, x, target)
  np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
  _assert_trees_close(grads, ig.jax_titan_params(ref_grads))


def test_train_step_matches_the_trainer():
  """One step of the branch's trainer against the same step put together from the loss, the
  clipping and the update of JAX-Titan's layouts, for both models."""
  import types  # pylint: disable=g-import-not-at-top

  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan.models import ideogram  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import ideogram as ig  # pylint: disable=g-import-not-at-top
  from maxdiffusion.trainers import ideogram_trainer as trainer  # pylint: disable=g-import-not-at-top

  adapter, jt_model, params, jt_params = _models()
  cfg = adapter.cfg
  x, latents = _batch(cfg)
  text_tokens, image_tokens = x["txt"].shape[1], x["img"].shape[1]
  side = int(image_tokens ** 0.5)
  resolution, max_grad_norm = 16 * side, 0.5
  optimizer = optax.sgd(0.1)

  # The branch's step: its model, its random draws, its loss, clipping and update.
  transformer = ig.Ideogram4Transformer(nnx.Rngs(0), cfg, dtype=jnp.float32)
  graphdef, state_params, rest = nnx.split(transformer, nnx.Param, ...)
  state = trainer.TrainState.create(
      apply_fn=graphdef.apply, params=state_params, tx=optimizer, graphdef=graphdef,
      rest_of_state=rest)
  state = jax.tree.map(trainer._to_array, state)
  data = {
      "latents": latents,
      "llm_features": np.pad(x["txt"], ((0, 0), (0, image_tokens), (0, 0))),
      **{k: x[k] for k in ("position_ids", "segment_ids", "indicator")},
  }
  config = types.SimpleNamespace(
      global_batch_size_to_train_on=latents.shape[0], ideogram_max_text_tokens=text_tokens,
      activations_dtype=jnp.float32, max_grad_norm=max_grad_norm)
  schedule = ig.scheduler.get_schedule_for_resolution((resolution, resolution), known_mean=0.5)
  key = jax.random.key(7)
  new_state, metrics, _ = jax.jit(
      lambda state, data, key: trainer.train_step(state, data, key, config, schedule))(
          state, data, key)
  ref_loss = metrics["scalar"]["learning/loss"]
  ref_norm = metrics["scalar"]["learning/max_grad_norm"]
  ref_params = ig.jax_titan_params(adapter.split(new_state.params.to_pure_dict()))
  assert ref_norm > max_grad_norm, "the step must clip to test clipping"

  # The same draws, as the branch's step makes them from its key.
  _, noise_key, time_key, _ = jax.random.split(key, num=4)
  u = jax.random.uniform(time_key, (latents.shape[0],), minval=1e-4, maxval=1.0 - 1e-4)
  noise = np.asarray(jax.random.normal(noise_key, latents.shape, jnp.float32))
  z = np.asarray(jax.scipy.special.ndtri(u))
  t = ig.timesteps(z, resolution)
  np.testing.assert_allclose(t, schedule(u), rtol=1e-5)
  np.testing.assert_allclose(ideogram.timesteps(z, resolution, mean=0.5, std=1.0), t, rtol=1e-5)

  img, target = ideogram.noised(latents, noise, t)
  x = {**x, "img": img, "timestep": t}

  def step(model, stage_params):
    loss, grads = _loss_and_grads(model, stage_params, x, target)
    norm = jt_mpmd.global_grad_norm(grads)
    updated = tuple(
        jt_mpmd.clipped_update(optimizer, p, g, optimizer.init(p), norm, max_grad_norm)[0]
        for p, g in zip(stage_params, grads))
    return loss, norm, updated

  loss, norm, updated = step(adapter, params)
  np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
  np.testing.assert_allclose(norm, ref_norm, rtol=1e-4)
  _assert_trees_close(ig.jax_titan_params(updated), ref_params)

  loss, norm, updated = step(jt_model, jt_params)
  np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
  np.testing.assert_allclose(norm, ref_norm, rtol=1e-4)
  _assert_trees_close(updated, ref_params)


def _run(log, *flags):
  # os.environ, not the process environment: see conftest's ROCPROFILER note.
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", *FLAGS, f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r for r in records.steps(records.read(log))}


@pytest.fixture(scope="module")
def dp_reference(tmp_path_factory):
  return _run(tmp_path_factory.mktemp("dp") / "dp.jsonl", "--parallel=dp")


@pytest.mark.parametrize("flags", [["--parallel=fsdp"], ["--parallel=mpmd", "--num_stages=2"]],
                         ids=["fsdp", "pipeline"])
def test_layout_matches_data_parallel_reference(flags, dp_reference, tmp_path):
  layout = _run(tmp_path / "layout.jsonl", *flags)
  for step in range(4):
    assert layout[step]["loss"] == pytest.approx(dp_reference[step]["loss"], rel=1e-5), step
    assert layout[step]["grad_norm"] == pytest.approx(
        dp_reference[step]["grad_norm"], rel=1e-4), step


def test_jax_titan_model_trains_like_the_adapter(dp_reference, tmp_path):
  """From the adapter's initial weights and on its batches, JAX-Titan's own model trains alike."""
  import dataclasses  # pylint: disable=g-import-not-at-top

  from jax_titan.models import weights  # pylint: disable=g-import-not-at-top

  _, jt_model, _, jt_params = _models()
  weights.save_params(tmp_path / "init", jt_params[0])
  # The branch's training timesteps, not the defaults of JAX-Titan's model.
  trained = _run(tmp_path / "jax_titan.jsonl", "--parallel=dp", "--model=ideogram4",
                 "--model_overrides=" + json.dumps(dataclasses.asdict(jt_model.config)),
                 "--timestep_mean=0.5", "--timestep_std=1.0", f"--init_from={tmp_path / 'init'}")
  for step in range(4):
    assert trained[step]["loss"] == pytest.approx(dp_reference[step]["loss"], rel=1e-5), step
    assert trained[step]["grad_norm"] == pytest.approx(
        dp_reference[step]["grad_norm"], rel=1e-4), step
