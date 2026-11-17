"""Ideogram 4 matches diffusers' model, its stages match each other, and so do its layouts."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_titan.models import ideogram
from jax_titan.observability import records
from jax_titan.trainer import checkpoint

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = {
    "attention_head_dim": 32, "num_attention_heads": 2, "in_channels": 16, "llm_features_dim": 48,
    "adaln_dim": 32, "intermediate_size": 128, "num_layers": 4, "mrope_section": (6, 5, 5),
}
FLAGS = [
    "--platform=cpu", "--model=ideogram4", "--resolution=64", "--text_tokens=8",
    "--model_overrides=" + json.dumps(CONFIG | {"dtype": "float32"}), "--microbatches=4",
    "--microbatch_size=4", "--steps=4", "--warmup_steps=1",
]


def _inputs(cfg, batch=2, side=4, text_tokens=8):
  """A batch whose second sample's text is left-padded by three positions."""
  rng = np.random.default_rng(0)
  ids = {k: np.stack([v] * batch) for k, v in ideogram.sequence_ids(side, text_tokens).items()}
  ids["segment_ids"][1, :3] = ideogram.PADDING_SEGMENT
  ids["indicator"][1, :3] = 0
  ids["position_ids"][1, :3] = 0
  return {
      "img": rng.normal(size=(batch, side * side, cfg.in_channels)).astype(np.float32),
      "txt": rng.normal(size=(batch, text_tokens, cfg.llm_features_dim)).astype(np.float32),
      "timestep": rng.uniform(size=batch).astype(np.float32),
      **ids,
  }


def _chain(model, stage_params, x):
  for i in range(model.num_stages):
    x = model.forward(i, stage_params[i], x)
  return x


def _from_diffusers(transformer) -> dict:
  """The weights of diffusers' model in the layout of `ideogram.init_params`."""
  w = {k: v.detach().numpy() for k, v in transformer.state_dict().items()}

  def layers(name, kernel=False):
    stacked = np.stack([w[f"layers.{i}.{name}"] for i in range(transformer.config.num_layers)])
    return stacked.swapaxes(1, 2) if kernel else stacked

  return {
      "layers": {
          "attn_norm": layers("attention_norm1.weight"),
          "wq": layers("attention.to_q.weight", True),
          "wk": layers("attention.to_k.weight", True),
          "wv": layers("attention.to_v.weight", True),
          "q_norm": layers("attention.norm_q.weight"),
          "k_norm": layers("attention.norm_k.weight"),
          "wo": layers("attention.to_out.0.weight", True),
          "attn_out_norm": layers("attention_norm2.weight"),
          "mlp_norm": layers("ffn_norm1.weight"),
          "w_gate": layers("feed_forward.w1.weight", True),
          "w_up": layers("feed_forward.w3.weight", True),
          "w_down": layers("feed_forward.w2.weight", True),
          "mlp_out_norm": layers("ffn_norm2.weight"),
          "w_mod": layers("adaln_modulation.weight", True),
          "b_mod": layers("adaln_modulation.bias"),
      },
      "w_in": w["input_proj.weight"].T,
      "b_in": w["input_proj.bias"],
      "text_norm": w["llm_cond_norm.weight"],
      "w_text": w["llm_cond_proj.weight"].T,
      "b_text": w["llm_cond_proj.bias"],
      "w_time_in": w["t_embedding.mlp_in.weight"].T,
      "b_time_in": w["t_embedding.mlp_in.bias"],
      "w_time_out": w["t_embedding.mlp_out.weight"].T,
      "b_time_out": w["t_embedding.mlp_out.bias"],
      "w_cond": w["adaln_proj.weight"].T,
      "b_cond": w["adaln_proj.bias"],
      "image_embed": w["embed_image_indicator.weight"],
      "w_final_mod": w["final_layer.adaln_modulation.weight"].T,
      "b_final_mod": w["final_layer.adaln_modulation.bias"],
      "w_out": w["final_layer.linear.weight"].T,
      "b_out": w["final_layer.linear.bias"],
  }


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("diffusers") is None,
    reason="needs torch and diffusers")
def test_matches_diffusers():
  import torch  # pylint: disable=g-import-not-at-top
  from diffusers import Ideogram4Transformer2DModel  # pylint: disable=g-import-not-at-top

  cfg = ideogram.IdeogramConfig(**CONFIG)
  torch.manual_seed(0)
  transformer = Ideogram4Transformer2DModel(**CONFIG).eval()
  with torch.no_grad():
    for name, p in transformer.named_parameters():
      if p.ndim == 1:
        p.normal_(1.0 if "norm" in name else 0.0, 0.1)
  params = _from_diffusers(transformer)
  shapes = jax.eval_shape(lambda: ideogram.init_params(cfg, jax.random.key(0)))
  assert jax.tree.map(lambda x: x.shape, params) == jax.tree.map(lambda x: x.shape, shapes)

  x = _inputs(cfg)
  text_tokens, image_tokens = x["txt"].shape[1], x["img"].shape[1]
  # diffusers takes both inputs over the whole sequence, zero where the other one is.
  latents = np.pad(x["img"], ((0, 0), (text_tokens, 0), (0, 0)))
  features = np.pad(x["txt"], ((0, 0), (0, image_tokens), (0, 0)))
  with torch.no_grad():
    ref = transformer(
        hidden_states=torch.from_numpy(latents),
        timestep=torch.from_numpy(x["timestep"]),
        encoder_hidden_states=torch.from_numpy(features),
        position_ids=torch.from_numpy(x["position_ids"]).long(),
        segment_ids=torch.from_numpy(x["segment_ids"]).long(),
        indicator=torch.from_numpy(x["indicator"]).long(),
        return_dict=False,
    )[0].numpy()

  model = ideogram.Model(cfg, num_stages=1, dtype=jnp.float32)
  pred = jax.jit(lambda p, x: _chain(model, p, x))(model.split(params), x)
  # Only the image positions carry a prediction.
  np.testing.assert_allclose(
      np.asarray(pred)[:, text_tokens:], ref[:, text_tokens:], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_stages,blocks", [(2, None), (3, (1, 2, 1))])
def test_stages_match_one_stage(num_stages, blocks):
  cfg = ideogram.IdeogramConfig(**CONFIG)
  one = ideogram.Model(cfg, num_stages=1, dtype=jnp.float32)
  model = ideogram.Model(cfg, num_stages=num_stages, blocks=blocks, dtype=jnp.float32)
  params = ideogram.init_params(cfg, jax.random.key(0))
  for a, b in zip(jax.tree.leaves(model.split(params)), jax.tree.leaves(model.init(0))):
    np.testing.assert_array_equal(a, b)

  x = _inputs(cfg)
  target = np.random.default_rng(1).normal(size=x["img"].shape).astype(np.float32)

  def loss(m):
    def fn(params, x, target):
      stage_params = m.split(params)
      fns = m.stage_fns()
      for i, f in enumerate(fns[:-1]):
        x = f(stage_params[i], x)
      return fns[-1](stage_params[-1], x, target)
    return jax.jit(jax.value_and_grad(fn))(params, x, target)

  (ref_loss, ref_grads), (staged_loss, staged_grads) = loss(one), loss(model)
  np.testing.assert_allclose(staged_loss, ref_loss, rtol=1e-5)
  for a, b in zip(jax.tree.leaves(staged_grads), jax.tree.leaves(ref_grads)):
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_timesteps():
  z = np.array([-1.0, 0.0, 2.0])
  # At 512 pixels the mean has no shift: z = 0 is half noise, and a larger z is noisier.
  np.testing.assert_allclose(
      ideogram.timesteps(z, 512), 1 / (1 + np.exp(1.5 * z)), rtol=1e-6)
  np.testing.assert_allclose(
      ideogram.timesteps(z, 1024, mean=0.5, std=1.0), 1 / (1 + np.exp(0.5 + np.log(2.0) + z)),
      rtol=1e-6)


def test_load_reads_saved_params(tmp_path):
  cfg = ideogram.IdeogramConfig(**CONFIG)
  model = ideogram.Model(cfg, num_stages=2, dtype=jnp.float32)
  params = ideogram.init_params(cfg, jax.random.key(3))
  checkpoint.save_params(tmp_path / "params", params)
  for a, b in zip(jax.tree.leaves(model.load(tmp_path / "params")),
                  jax.tree.leaves(model.split(params))):
    np.testing.assert_array_equal(a, b)


def _run(log, *flags):
  # os.environ, not the process environment: see conftest's ROCPROFILER note.
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", *FLAGS, f"--log_file={log}", *flags],
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return {r["step"]: r for r in records.steps(records.read(log))}


@pytest.mark.parametrize("flags", [["--parallel=fsdp"], ["--parallel=mpmd", "--num_stages=2"]],
                         ids=["fsdp", "pipeline"])
def test_layout_matches_data_parallel_reference(flags, tmp_path):
  reference = _run(tmp_path / "dp.jsonl", "--parallel=dp")
  layout = _run(tmp_path / "layout.jsonl", *flags)
  for step in range(4):
    assert layout[step]["loss"] == pytest.approx(reference[step]["loss"], rel=1e-5), step
    assert layout[step]["grad_norm"] == pytest.approx(reference[step]["grad_norm"], rel=1e-4), step


def test_init_from_starts_from_saved_params(tmp_path):
  cfg = ideogram.IdeogramConfig(**CONFIG)
  for seed in (0, 1):
    params = ideogram.init_params(cfg, jax.random.key(seed))
    checkpoint.save_params(tmp_path / f"params{seed}", params)
  flags = ["--parallel=dp", "--steps=1"]
  seeded = _run(tmp_path / "seeded.jsonl", *flags)
  same = _run(tmp_path / "same.jsonl", *flags, f"--init_from={tmp_path / 'params0'}")
  other = _run(tmp_path / "other.jsonl", *flags, f"--init_from={tmp_path / 'params1'}")
  assert same[0]["loss"] == pytest.approx(seeded[0]["loss"], rel=1e-6)
  assert other[0]["loss"] != pytest.approx(seeded[0]["loss"], rel=1e-3)
