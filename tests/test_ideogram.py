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

from jax_titan.models import ideogram, pretrained
from jax_titan.observability import records
from jax_titan.trainer import checkpoint

REPO = pathlib.Path(__file__).resolve().parents[1]
# The rotary sections end below the head's 16 frequencies, as they do at full size, so the
# frequencies past them take the first position axis.
CONFIG = {
    "attention_head_dim": 32, "num_attention_heads": 2, "in_channels": 16, "llm_features_dim": 48,
    "adaln_dim": 32, "intermediate_size": 128, "num_layers": 4, "mrope_section": (4, 3, 2),
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


def test_sequence_ids_pad_on_the_left():
  valid = np.array([[True, True, True], [False, True, True]])
  ids = ideogram.padded_sequence_ids(2, valid)
  one = ideogram.sequence_ids(2, 3)
  for name in ("position_ids", "segment_ids", "indicator"):
    assert ids[name].shape == (2,) + one[name].shape
    np.testing.assert_array_equal(ids[name][0], one[name])
    np.testing.assert_array_equal(ids[name][1, 3:], one[name][3:])
  np.testing.assert_array_equal(ids["position_ids"][1, :3], [[0] * 3, [0] * 3, [1] * 3])
  np.testing.assert_array_equal(ids["segment_ids"][1, :3], [ideogram.PADDING_SEGMENT, 1, 1])
  np.testing.assert_array_equal(
      ids["indicator"][1, :3], [0, ideogram.TEXT_INDICATOR, ideogram.TEXT_INDICATOR])


def test_load_reads_saved_params(tmp_path):
  cfg = ideogram.IdeogramConfig(**CONFIG)
  model = ideogram.Model(cfg, num_stages=2, dtype=jnp.float32)
  params = ideogram.init_params(cfg, jax.random.key(3))
  checkpoint.save_params(tmp_path / "params", params)
  for a, b in zip(jax.tree.leaves(model.load(tmp_path / "params")),
                  jax.tree.leaves(model.split(params))):
    np.testing.assert_array_equal(a, b)


def _published(params) -> dict:
  """`params` under the names and in the layouts of the published transformer weights."""
  layers = {k: np.asarray(v) for k, v in params["layers"].items()}
  tensors = {}
  for i in range(len(layers["wq"])):
    block = f"layers.{i}."
    qkv = np.concatenate([layers[k][i] for k in ("wq", "wk", "wv")], axis=1)
    tensors[block + "attention.qkv.weight"] = qkv.T
    for name, key in (("attention.o", "wo"), ("feed_forward.w1", "w_gate"),
                      ("feed_forward.w3", "w_up"), ("feed_forward.w2", "w_down"),
                      ("adaln_modulation", "w_mod")):
      tensors[block + name + ".weight"] = layers[key][i].T
    for name, key in (("attention_norm1.weight", "attn_norm"),
                      ("attention.norm_q.weight", "q_norm"), ("attention.norm_k.weight", "k_norm"),
                      ("attention_norm2.weight", "attn_out_norm"),
                      ("ffn_norm1.weight", "mlp_norm"), ("ffn_norm2.weight", "mlp_out_norm"),
                      ("adaln_modulation.bias", "b_mod")):
      tensors[block + name] = layers[key][i]
  for name, w, b in (("input_proj", "w_in", "b_in"), ("llm_cond_proj", "w_text", "b_text"),
                     ("t_embedding.mlp_in", "w_time_in", "b_time_in"),
                     ("t_embedding.mlp_out", "w_time_out", "b_time_out"),
                     ("adaln_proj", "w_cond", "b_cond"),
                     ("final_layer.adaln_modulation", "w_final_mod", "b_final_mod"),
                     ("final_layer.linear", "w_out", "b_out")):
    tensors[name + ".weight"] = np.asarray(params[w]).T
    tensors[name + ".bias"] = np.asarray(params[b])
  tensors["llm_cond_norm.weight"] = np.asarray(params["text_norm"])
  tensors["embed_image_indicator.weight"] = np.asarray(params["image_embed"])
  return tensors


def _write_published(path, params):
  path.mkdir()
  pretrained.write(path / "diffusion_pytorch_model.safetensors", _published(params))
  config = CONFIG | {"_class_name": "Ideogram4Transformer2DModel"}
  (path / "config.json").write_text(json.dumps(config))


def test_load_reads_published_weights(tmp_path):
  cfg = ideogram.IdeogramConfig(**CONFIG)
  params = ideogram.init_params(cfg, jax.random.key(3))
  _write_published(tmp_path / "transformer", params)
  assert ideogram.pretrained_config(tmp_path / "transformer") == cfg
  model = ideogram.Model(cfg, num_stages=2, dtype=jnp.float32)
  for a, b in zip(jax.tree.leaves(model.load(tmp_path / "transformer")),
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
  _write_published(tmp_path / "published", ideogram.init_params(cfg, jax.random.key(0)))
  published = _run(tmp_path / "published.jsonl", *flags, f"--init_from={tmp_path / 'published'}")
  assert same[0]["loss"] == pytest.approx(seeded[0]["loss"], rel=1e-6)
  assert published[0]["loss"] == pytest.approx(seeded[0]["loss"], rel=1e-6)
  assert other[0]["loss"] != pytest.approx(seeded[0]["loss"], rel=1e-3)


def test_save_params_writes_what_init_from_reads(tmp_path):
  cfg = ideogram.IdeogramConfig(**CONFIG)
  # Without a learning rate the step changes nothing, so the stages' parameters,
  # merged, are the initial ones.
  _run(tmp_path / "log.jsonl", "--parallel=mpmd", "--num_stages=2", "--steps=1", "--lr=0",
       f"--save_params={tmp_path / 'saved'}")
  model = ideogram.Model(cfg, num_stages=1, dtype=jnp.float32)
  for a, b in zip(jax.tree.leaves(model.load(tmp_path / "saved")), jax.tree.leaves(model.init(0))):
    np.testing.assert_array_equal(a, b)


def _write_cache(path, cfg, images=6, text_tokens=6):
  """A cache of random samples: two captions, the second shorter by two tokens."""
  rng = np.random.default_rng(0)
  valid = np.ones((2, text_tokens), bool)
  valid[1, :2] = False
  features = rng.normal(size=(2, text_tokens, cfg.llm_features_dim)) * valid[..., None]
  path.mkdir()
  tokens = rng.normal(size=(images, 16, cfg.in_channels)).astype(np.float32)
  np.save(path / "image_tokens.npy", tokens)
  np.save(path / "text_features.npy", features.astype(np.float16))
  np.save(path / "text_valid.npy", valid)
  np.save(path / "caption_of_image.npy", (np.arange(images) % 2).astype(np.int32))
  (path / "meta.json").write_text(json.dumps({"resolution": 64}))


def test_trains_on_a_cache_of_samples(tmp_path):
  from jax_titan.data import ideogram as data  # pylint: disable=g-import-not-at-top

  cfg = ideogram.IdeogramConfig(**CONFIG)
  _write_cache(tmp_path / "cache", cfg)
  cache = data.Cache.open(tmp_path / "cache")
  assert len(cache) == 6 and cache.text_tokens == 6 and cache.resolution == 64
  tokens, features, valid = cache.samples(np.array([1, 4, 1]))
  assert tokens.shape == (3, 16, cfg.in_channels) and features.dtype == np.float32
  np.testing.assert_array_equal(valid[:, :2], [[False] * 2, [True] * 2, [False] * 2])
  np.testing.assert_array_equal(features[0, :2], 0)

  flags = [f"--data={tmp_path / 'cache'}", "--eval_every=2", "--eval_batches=2", "--steps=2"]
  runs = {}
  for layout in ("dp", "fsdp"):
    log = tmp_path / f"{layout}.jsonl"
    steps = _run(log, f"--parallel={layout}", *flags)
    evals = [r["loss"] for r in records.events(records.read(log), "eval")]
    assert len(evals) == 1 and np.isfinite(evals[0])
    runs[layout] = [steps[i]["loss"] for i in range(2)] + evals
  np.testing.assert_allclose(runs["fsdp"], runs["dp"], rtol=1e-5)


def test_load_image_covers_the_square_and_crops_the_center(tmp_path):
  image_lib = pytest.importorskip("PIL.Image")
  from jax_titan.data import ideogram as data  # pylint: disable=g-import-not-at-top

  # Twice as wide as high, left half black and right half white, with an alpha channel.
  pixels = np.zeros((20, 40, 4), np.uint8)
  pixels[:, 20:, :3] = 255
  pixels[..., 3] = 255
  image_lib.fromarray(pixels, "RGBA").save(tmp_path / "x.png")
  image = data.load_image(tmp_path / "x.png", 16)
  assert image.shape == (16, 16, 3) and image.dtype == np.float32
  np.testing.assert_allclose(image[:, :6], -1.0, atol=1e-2)
  np.testing.assert_allclose(image[:, -6:], 1.0, atol=1e-2)

