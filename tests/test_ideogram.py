"""Ideogram 4 matches diffusers' model, its stages match each other, and so do its layouts;
it starts from published weights and trains on a cache of samples."""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

import ideogram_checkpoint
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import pytest

from jax_titan import parallel
from jax_titan.models import ideogram
from jax_titan.observability import records
from jax_titan.trainer import adapters, checkpoint, layout, train

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
  forward = jax.jit(lambda p, x: _chain(model, p, x))
  pred = forward(model.split(params), x)
  # Only the image positions carry a prediction.
  np.testing.assert_allclose(
      np.asarray(pred)[:, text_tokens:], ref[:, text_tokens:], rtol=1e-5, atol=1e-5)

  # The image tokens alone, as Ideogram's pipeline gives them to its unconditional transformer.
  ids = {k: x[k][:, text_tokens:] for k in ("position_ids", "segment_ids", "indicator")}
  with torch.no_grad():
    ref = transformer(
        hidden_states=torch.from_numpy(x["img"]),
        timestep=torch.from_numpy(x["timestep"]),
        encoder_hidden_states=torch.zeros(x["img"].shape[:2] + (cfg.llm_features_dim,)),
        return_dict=False, **{k: torch.from_numpy(v).long() for k, v in ids.items()},
    )[0].numpy()
  alone = {"img": x["img"], "txt": x["txt"][:, :0], "timestep": x["timestep"], **ids}
  np.testing.assert_allclose(forward(model.split(params), alone), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_stages,blocks", [(2, None), (3, (1, 2, 1))])
def test_stages_match_one_stage(num_stages, blocks):
  cfg = ideogram.IdeogramConfig(**CONFIG)
  one = ideogram.Model(cfg, num_stages=1, dtype=jnp.float32)
  model = ideogram.Model(cfg, num_stages=num_stages, blocks=blocks, dtype=jnp.float32)
  params = ideogram.init_params(cfg, jax.random.key(0))
  for a, b in zip(jax.tree.leaves(model.split(params)), jax.tree.leaves(model.init(0))):
    np.testing.assert_array_equal(a, b)
  merged = model.merge(model.split(params))
  assert jax.tree.structure(merged) == jax.tree.structure(params)
  for a, b in zip(jax.tree.leaves(merged), jax.tree.leaves(params)):
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


def test_image_tokens_alone_match_padded_text():
  """Generation gives the unconditional transformer no text token at all."""
  cfg = ideogram.IdeogramConfig(**CONFIG)
  model = ideogram.Model(cfg, num_stages=1, dtype=jnp.float32)
  x = _inputs(cfg)
  image_tokens = x["img"].shape[1]
  # Padding is a segment of its own, so text that is all padding reaches no image token.
  padded = {**x, **ideogram.padded_sequence_ids(4, np.zeros(x["txt"].shape[:2], bool))}
  ids = {k: padded[k][:, -image_tokens:] for k in ("position_ids", "segment_ids", "indicator")}
  alone = {"img": x["img"], "txt": x["txt"][:, :0], "timestep": x["timestep"], **ids}
  forward = jax.jit(lambda p, x: model.forward(0, p, x))
  params = model.init(0)[0]
  np.testing.assert_allclose(
      forward(params, alone), forward(params, padded)[:, -image_tokens:], rtol=1e-5, atol=1e-5)


def test_bfloat16_keeps_image_positions_apart():
  cfg = ideogram.IdeogramConfig(**CONFIG)
  model = ideogram.Model(cfg, num_stages=1, dtype=jnp.bfloat16)
  x = _inputs(cfg)
  # Equal image tokens differ by their positions alone, and bf16 holds 65536 + 1 as 65536.
  x["img"] = np.broadcast_to(x["img"][:, :1], x["img"].shape)
  out = jax.jit(lambda p, x: model.forward(0, p, x))(model.init(0)[0], x)
  image = np.asarray(out, np.float32)[:, -x["img"].shape[1]:]
  assert np.abs(image[:, 1:] - image[:, :1]).max(axis=(0, 2)).min() > 0


def test_every_parameter_is_sharded_and_only_norms_do_not_decay():
  cfg = ideogram.IdeogramConfig(**CONFIG)
  params = jax.eval_shape(lambda: ideogram.init_params(cfg, jax.random.key(0)))
  specs = parallel.tree_specs(params, ideogram.logical_axes, {"data": 8, "tensor": 1}, fsdp=True)
  for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
    assert ideogram.logical_axes(path, leaf) is not None, path
  for path, spec in jax.tree_util.tree_flatten_with_path(specs)[0]:
    assert "data" in spec, path
  kept = {path[-1].key for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
          if not ideogram.decays(path, leaf)}
  assert kept == {"attn_norm", "q_norm", "k_norm", "attn_out_norm", "mlp_norm", "mlp_out_norm",
                  "text_norm"}


def test_timesteps():
  z = np.array([-1.0, 0.0, 2.0])
  # At 512 pixels the mean has no shift: z = 0 is half noise, and a larger z is noisier.
  np.testing.assert_allclose(
      ideogram.timesteps(z, 512), 1 / (1 + np.exp(1.5 * z)), rtol=1e-6)
  np.testing.assert_allclose(
      ideogram.timesteps(z, 1024, mean=0.5, std=1.0), 1 / (1 + np.exp(0.5 + np.log(2.0) + z)),
      rtol=1e-6)


def test_noising_runs_from_noise_to_data():
  rng = np.random.default_rng(0)
  latents, noise = rng.normal(size=(2, 3, 4, 5)), rng.normal(size=(2, 3, 4, 5))
  t = np.array([[0.0, 0.25, 1.0]] * 2)
  img, target = ideogram.noised(latents, noise, t)
  np.testing.assert_array_equal(img[:, 0], noise[:, 0])
  np.testing.assert_array_equal(img[:, 2], latents[:, 2])
  np.testing.assert_allclose(img[:, 1], 0.75 * noise[:, 1] + 0.25 * latents[:, 1])
  # A step of 1 - t along the velocity target ends at the data.
  np.testing.assert_allclose(img + (1 - t)[..., None, None] * target, latents)


def test_caption_is_json_without_spaces():
  compact = '{"high_level_description":"a 猫","style_description":{"aesthetics":"b"}}'
  assert ideogram.caption(json.dumps(json.loads(compact), indent=2)) == compact
  assert ideogram.caption("\ufeff" + compact + "\n") == compact
  assert ideogram.caption("\ufeff a cat, in yarn \n") == "a cat, in yarn"
  assert ideogram.caption("[1, 2]") == "[1, 2]"


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


def _write_published(path, params, fp8=False):
  ideogram_checkpoint.write_transformer(path, CONFIG, params, fp8)


@pytest.mark.parametrize("fp8", [False, True])
def test_load_reads_published_weights(fp8, tmp_path):
  cfg = ideogram.IdeogramConfig(**CONFIG)
  # Every leaf random, so that no two of one shape can change places unseen.
  rng = np.random.default_rng(3)
  params = jax.tree.map(lambda x: rng.normal(size=x.shape).astype(np.float32),
                        ideogram.init_params(cfg, jax.random.key(3)))
  _write_published(tmp_path / "transformer", params, fp8)
  assert ideogram.pretrained_config(tmp_path / "transformer") == cfg

  def read(path, w):
    if not fp8 or not path[-1].key.startswith("w"):
      return w
    # A kernel [..., inputs, outputs] comes back as its e4m3 values times the scale of each output.
    scale = (np.abs(w).max(axis=-2, keepdims=True) / 448).astype(np.float32)
    return (w / scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale

  expected = jax.tree_util.tree_map_with_path(read, params)
  model = ideogram.Model(cfg, num_stages=2, dtype=jnp.float32)
  loaded = model.load(tmp_path / "transformer")
  assert jax.tree.structure(loaded) == jax.tree.structure(model.split(expected))
  for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(model.split(expected))):
    np.testing.assert_array_equal(a, b)


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


@pytest.fixture(scope="module")
def dp_reference(tmp_path_factory):
  """The step records of the data-parallel run and its validation loss after every step."""
  log = tmp_path_factory.mktemp("dp") / "dp.jsonl"
  return _run(log, "--parallel=dp", "--eval_every=1", "--eval_batches=1"), _validation(log)


@pytest.mark.parametrize("flags", [["--parallel=fsdp"], ["--parallel=mpmd", "--num_stages=2"]],
                         ids=["fsdp", "pipeline"])
def test_layout_matches_data_parallel_reference(flags, dp_reference, tmp_path):
  reference, validation = dp_reference
  log, saved = tmp_path / "layout.jsonl", tmp_path / "saved"
  trained = _run(log, *flags, "--eval_every=4", "--eval_batches=1", f"--save_params={saved}")
  for step in range(4):
    assert trained[step]["loss"] == pytest.approx(reference[step]["loss"], rel=1e-5), step
    assert trained[step]["grad_norm"] == pytest.approx(reference[step]["grad_norm"], rel=1e-4), step
  assert _validation(log) == pytest.approx(validation[-1:], rel=1e-5)
  # --save_params wrote the parameters after the last step, which the validation loss tells
  # from earlier ones: another layout that starts from them validates alike.
  assert validation[-1] != pytest.approx(validation[0], rel=1e-3)
  _run(tmp_path / "saved.jsonl", "--parallel=dp", "--steps=1", "--lr=0", "--eval_every=1",
       "--eval_batches=1", f"--init_from={saved}")
  assert _validation(tmp_path / "saved.jsonl") == pytest.approx(validation[-1:], rel=1e-5)


def test_init_from_starts_from_given_params(dp_reference, tmp_path):
  reference, _ = dp_reference
  cfg = ideogram.IdeogramConfig(**CONFIG)
  flags = ["--parallel=dp", "--steps=1"]
  checkpoint.save_params(tmp_path / "saved", ideogram.init_params(cfg, jax.random.key(1)))
  other = _run(tmp_path / "other.jsonl", *flags, f"--init_from={tmp_path / 'saved'}")
  assert other[0]["loss"] != pytest.approx(reference[0]["loss"], rel=1e-3)
  # Published weights bring their config, so the overrides need not repeat it.
  _write_published(tmp_path / "published", ideogram.init_params(cfg, jax.random.key(0)))
  published = _run(tmp_path / "published.jsonl", *flags, f"--init_from={tmp_path / 'published'}",
                   "--model_overrides=" + json.dumps({"dtype": "float32"}))
  assert published[0]["loss"] == pytest.approx(reference[0]["loss"], rel=1e-6)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
  """A directory with a toy `checkpoint`, six `images` with captions and the `cache` of them."""
  image_lib = pytest.importorskip("PIL.Image")
  from jax_titan.data import ideogram as data  # pylint: disable=g-import-not-at-top

  root = tmp_path_factory.mktemp("toy")
  ideogram_checkpoint.write(root / "checkpoint")
  rng = np.random.default_rng(0)
  (root / "images").mkdir()
  for i in range(6):
    pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    image_lib.fromarray(pixels).save(root / "images" / f"{i}.png")
  caption = {"high_level_description": "a b"}
  # One caption written in two ways, with a .txt file that the .json file hides; a plain one;
  # and images 3 to 5 without a file.
  (root / "images/0.json").write_text(json.dumps(caption, indent=1), encoding="utf-8")
  (root / "images/0.txt").write_text("c", encoding="utf-8")
  (root / "images/1.txt").write_text("a\n", encoding="utf-8")
  (root / "images/2.json").write_text("\ufeff" + json.dumps(caption), encoding="utf-8")
  data.prepare(root / "checkpoint", root / "images", root / "cache", resolution=64,
               prompt="a b c", log=lambda _: None)
  return root


def _cache_flags(toy):
  # The published weights bring the config, and the cache its text length.
  return [f"--data={toy / 'cache'}", f"--init_from={toy / 'checkpoint/transformer'}",
          "--model_overrides=" + json.dumps({"dtype": "float32"})]


def test_cache_holds_image_tokens_and_text_features(toy):
  from jax_titan.data import ideogram as data  # pylint: disable=g-import-not-at-top

  cache = data.Cache.open(toy / "cache")
  assert not (toy / "cache.tmp").exists()
  assert len(cache) == 6 and cache.resolution == 64
  assert cache.image_tokens.shape == (6, 16, 128) and cache.image_tokens.dtype == np.float32
  meta = json.loads((toy / "cache/meta.json").read_text(encoding="utf-8"))
  assert meta["captions"] == ['{"high_level_description":"a b"}', "a", "a b c"]
  np.testing.assert_array_equal(cache.caption_of_image, [0, 1, 0, 2, 2, 2])
  # The chat format adds five tokens; shorter captions are padded on the left, with zeros.
  np.testing.assert_array_equal(cache.text_valid.sum(axis=1), [11, 6, 8])
  np.testing.assert_array_equal(np.sort(cache.text_valid, axis=1), cache.text_valid)
  assert cache.text_features.shape == (3, 11, 48) and cache.text_features.dtype == np.float16
  np.testing.assert_array_equal(np.any(cache.text_features != 0, axis=-1), cache.text_valid)

  tokens, features, valid = cache.samples(np.array([1, 4, 1]))
  assert features.dtype == np.float32
  np.testing.assert_array_equal(tokens, cache.image_tokens[[1, 4, 1]])
  np.testing.assert_array_equal(valid, cache.text_valid[[1, 2, 1]])
  np.testing.assert_array_equal(features, cache.text_features[[1, 2, 1]])


def test_trains_on_a_cache_of_samples(toy, tmp_path):
  flags = _cache_flags(toy) + ["--eval_every=2", "--eval_batches=2", "--steps=2"]
  runs = {}
  for name in ("dp", "fsdp"):
    log = tmp_path / f"{name}.jsonl"
    steps = _run(log, f"--parallel={name}", *flags)
    assert len(_validation(log)) == 1 and np.isfinite(_validation(log)[0])
    runs[name] = [steps[i]["loss"] for i in range(2)] + _validation(log)
  np.testing.assert_allclose(runs["fsdp"], runs["dp"], rtol=1e-5)


def test_resume_on_a_cache_is_exact(toy, tmp_path):
  flags = ["--parallel=fsdp", *_cache_flags(toy), "--eval_every=2", "--checkpoint_every=2"]
  full = _run(tmp_path / "full.jsonl", *flags, f"--checkpoint_dir={tmp_path / 'full'}")
  # From the checkpoint after step 2, which wins over --init_from.
  (tmp_path / "resumed").mkdir()
  shutil.copytree(tmp_path / "full" / "2", tmp_path / "resumed" / "2")
  resumed = _run(tmp_path / "resumed.jsonl", *flags, f"--checkpoint_dir={tmp_path / 'resumed'}")
  assert sorted(resumed) == [2, 3]
  for step in (2, 3):
    assert resumed[step]["loss"] == full[step]["loss"], step
  assert _validation(tmp_path / "resumed.jsonl") == _validation(tmp_path / "full.jsonl")[1:]


def _workload(*flags):
  args = train.parse_args(FLAGS + ["--parallel=dp", *flags])
  return adapters.build(args, jax.devices(), lambda decays=None: optax.sgd(0.0))[1]


def test_validation_draws_the_held_out_images_and_training_the_others(toy):
  from jax_titan.data import ideogram as data  # pylint: disable=g-import-not-at-top

  cache = data.Cache.open(toy / "cache")
  work = _workload(*_cache_flags(toy), "--eval_every=1")
  assert "5 images, 1 more held out" in work.data

  def images(step):
    x, target = work.next_batch(step)
    latents = x["img"] + (1 - x["timestep"])[..., None, None] * target
    latents = latents.reshape(-1, 1, *latents.shape[2:])
    return set(((latents - cache.image_tokens[None]) ** 2).sum(axis=(2, 3)).argmin(axis=1).tolist())

  assert set().union(*map(images, range(4))) == {0, 1, 2, 3, 4}
  assert images(layout.EVAL_STEPS) == images(layout.EVAL_STEPS + 1) == {5}


def test_data_is_a_cache_or_synthetic(tmp_path):
  assert _workload().data == _workload("--data=synthetic").data == "synthetic"
  with pytest.raises(FileNotFoundError, match="is not a cache"):
    _workload(f"--data={tmp_path / 'missing'}")


def test_load_image_crops_the_center_as_the_picture_is_held(tmp_path):
  image_lib = pytest.importorskip("PIL.Image")
  from jax_titan.data import ideogram as data  # pylint: disable=g-import-not-at-top

  # Twice as wide as high, with an alpha channel: white but for the first columns, which a
  # center crop drops and a resize of the whole picture would keep.
  pixels = np.full((20, 40, 4), 255, np.uint8)
  pixels[:, :5, :3] = 0
  image_lib.fromarray(pixels).save(tmp_path / "wide.png")
  image = data.load_image(tmp_path / "wide.png", 16)
  assert image.shape == (16, 16, 3) and image.dtype == np.float32
  np.testing.assert_allclose(image, 1.0, atol=1e-2)

  # Stored on its side, black above white, to be turned a quarter clockwise: black on the right.
  pixels = np.full((20, 40, 3), 255, np.uint8)
  pixels[:10] = 0
  exif = image_lib.Exif()
  exif[0x0112] = 6
  image_lib.fromarray(pixels).save(tmp_path / "turned.png", exif=exif)
  image = data.load_image(tmp_path / "turned.png", 16)
  np.testing.assert_allclose(image[:, :6], 1.0, atol=1e-2)
  np.testing.assert_allclose(image[:, -6:], -1.0, atol=1e-2)
