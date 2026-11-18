"""Command A+ gives transformers' logits and gradients, reads and writes the
published checkpoint layout, routes tokens to experts and masks what it must."""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_titan.models import command_a

# Every branch of the published config at a small size: sequences longer than
# the window, more query heads than key heads, heads wider than the hidden size
# divides into, fewer chosen experts than experts, a shared MLP wider than one.
CONFIG = command_a.CommandAConfig(
    vocab_size=512, hidden_size=64, num_hidden_layers=8, num_attention_heads=8,
    num_key_value_heads=2, head_dim=32, intermediate_size=64, num_experts=16,
    num_experts_per_tok=4, num_shared_experts=2, sliding_window=8, eos_token_id=3,
    image_token_id=5, alignment_intermediate_size=96,
    vision_hidden_size=32, vision_intermediate_size=64, vision_num_hidden_layers=2,
    vision_num_attention_heads=4, vision_image_size=32, vision_patch_size=8)
SEQ = 32


def _params(cfg=CONFIG, images=True, seed=0):
  """Parameters whose norm weights and biases are off their initial ones and zeros."""
  rng = np.random.default_rng(seed)

  def moved(path, w):
    name = path[-1].key
    if "norm" in name or name.startswith("b"):
      w = w + 0.2 * rng.standard_normal(w.shape).astype(np.float32)
    return w

  params = command_a.init_params(cfg, jax.random.key(seed), images)
  return jax.tree_util.tree_map_with_path(moved, params)


def _batches(cfg=CONFIG):
  """Three batches of two sequences: text, text padded on the left, and text
  with image tiles (two tiles, then one). Each is (tokens, pixels, valid)."""
  rng = np.random.default_rng(2)
  def text(*shape):
    return rng.integers(6, cfg.vocab_size, shape).astype(np.int32)
  yield "text", (text(2, SEQ), None, None)
  tokens, valid = text(2, SEQ), np.ones((2, SEQ), bool)
  tokens[1, :5], valid[1, :5] = cfg.pad_token_id, False
  yield "padded", (tokens, None, valid)
  tile = [cfg.image_token_id] * cfg.tile_tokens + [4]
  rows = [np.concatenate([text(6), tile * tiles, text(SEQ - 6 - len(tile) * tiles)])
          for tiles in (2, 1)]
  size = cfg.vision_image_size
  pixels = rng.standard_normal((2, 2, 3, size, size)).astype(np.float32)
  pixels[1, 1] = 0
  yield "image", (np.stack(rows).astype(np.int32), pixels, None)


def _has_reference() -> bool:
  """Whether torch and a transformers with Command A+ are installed, without importing them."""
  spec = importlib.util.find_spec("transformers")
  return importlib.util.find_spec("torch") is not None and spec is not None and any(
      (pathlib.Path(root) / "models/cohere2_moe").is_dir()
      for root in spec.submodule_search_locations or ())


needs_reference = pytest.mark.skipif(
    not _has_reference(), reason="needs torch and transformers with cohere2_moe")


def _reference(path):
  """transformers' model of the checkpoint at `path`, on its exact attention path."""
  import transformers  # pylint: disable=g-import-not-at-top

  model = transformers.Cohere2VisionForConditionalGeneration.from_pretrained(path).eval()
  model.set_attn_implementation("eager")
  return model


def _reference_inputs(tokens, pixels, valid):
  import torch  # pylint: disable=g-import-not-at-top

  inputs = {"input_ids": torch.from_numpy(tokens).long()}
  if pixels is not None:
    # transformers takes the batch's tiles in one list; the second sequence has one.
    inputs["pixel_values"] = torch.from_numpy(pixels.reshape(-1, *pixels.shape[2:])[:3])
  if valid is not None:
    inputs["attention_mask"] = torch.from_numpy(valid).long()
  return inputs


def _published(named: dict, cfg) -> dict:
  """Tensors under the names of transformers' model, renamed as the published
  checkpoint names them: every expert on its own."""
  out, f = {}, cfg.intermediate_size
  for name, value in named.items():
    value = value.detach().numpy()
    if name.endswith("mlp.experts.gate_up_proj"):
      for x, w in enumerate(value):
        out[f"{name[:-len('gate_up_proj')]}{x}.gate_proj.weight"] = w[:f]
        out[f"{name[:-len('gate_up_proj')]}{x}.up_proj.weight"] = w[f:]
    elif name.endswith("mlp.experts.down_proj"):
      for x, w in enumerate(value):
        out[f"{name[:-len('down_proj')]}{x}.down_proj.weight"] = w
    else:
      out[name] = value
  return out


def _loss(cfg, params, tokens, pixels, valid):
  """The loss of the next tokens, those that are not image tokens."""
  logits = command_a.forward(cfg, params, tokens, pixels, valid, jnp.float32)
  targets = np.where(tokens == cfg.image_token_id, -100, tokens)
  return command_a.cross_entropy(logits[:, :-1], targets[:, 1:]), logits


@needs_reference
def test_matches_transformers(tmp_path):
  command_a.save(tmp_path, CONFIG, _params())
  model = _reference(tmp_path)
  cfg, params = command_a.load(tmp_path, jnp.float32)
  assert cfg == CONFIG
  for name, (tokens, pixels, valid) in _batches():
    inputs = _reference_inputs(tokens, pixels, valid)
    labels = inputs["input_ids"].masked_fill(inputs["input_ids"] == CONFIG.image_token_id, -100)
    model.zero_grad()
    out = model(**inputs, labels=labels)
    used = dict(params)
    if pixels is None:
      del used["vision"], used["projector"]
    (loss, logits), grads = jax.value_and_grad(_loss, argnums=1, has_aux=True)(
        cfg, used, tokens, pixels, valid)
    real = np.ones(tokens.shape, bool) if valid is None else valid
    # Padding attends to nothing, so what the model gives there is not defined.
    np.testing.assert_allclose(np.asarray(logits)[real], out.logits.detach().numpy()[real],
                               rtol=1e-4, atol=1e-4, err_msg=name)
    if valid is not None:
      continue
    np.testing.assert_allclose(loss, out.loss.item(), rtol=1e-5, err_msg=name)
    out.loss.backward()
    theirs = _published({k: p.grad for k, p in model.named_parameters() if p.grad is not None}, cfg)
    theirs = command_a.params_from_tensors(theirs, cfg, images=pixels is not None)
    assert jax.tree.structure(theirs) == jax.tree.structure(grads)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5, err_msg=name),
        grads, theirs)


@needs_reference
def test_reads_what_transformers_writes(tmp_path):
  command_a.save(tmp_path / "ours", CONFIG, _params())
  _reference(tmp_path / "ours").save_pretrained(tmp_path / "theirs")
  ours, theirs = (command_a.load(tmp_path / name, jnp.float32) for name in ("ours", "theirs"))
  assert ours[0] == theirs[0]
  jax.tree.map(np.testing.assert_array_equal, ours[1], theirs[1])


def test_checkpoint_layout_round_trips(tmp_path):
  params = _params()
  tensors = command_a.tensors_from_params(params, CONFIG)
  layers, experts = CONFIG.num_hidden_layers, CONFIG.num_experts
  vision = CONFIG.vision_num_hidden_layers
  assert len(tensors) == 2 + layers * (9 + 3 * experts) + 4 + 5 + 16 * vision
  shapes = {k: v.shape for k, v in tensors.items()}
  lm, vm = "model.language_model.layers.", "model.vision_tower.vision_model."
  assert shapes[lm + "7.mlp.experts.15.up_proj.weight"] == (64, 64)
  assert shapes[lm + "0.self_attn.q_proj.weight"] == (256, 64)
  assert shapes[lm + "0.mlp.shared_experts.down_proj.weight"] == (64, 128)
  assert shapes[vm + "embeddings.patch_embedding.weight"] == (32, 3, 8, 8)
  command_a.save(tmp_path, CONFIG, params)
  cfg, loaded = command_a.load(tmp_path, jnp.float32)
  assert cfg == CONFIG
  jax.tree.map(np.testing.assert_array_equal, loaded, params)
  language = command_a.load(tmp_path, jnp.float32, images=False)[1]
  assert sorted(language) == ["embed", "final_norm", "layers"]


def test_a_token_goes_through_its_routers_experts():
  cfg, rng = CONFIG, np.random.default_rng(0)
  p = {k: np.asarray(v[0]) for k, v in _params()["layers"].items()}
  x = rng.standard_normal((2, 5, cfg.hidden_size)).astype(np.float32)
  out = np.asarray(command_a._mlp(cfg, jnp.asarray(x), p))  # pylint: disable=protected-access

  def mlp(v, gate, up, down):
    g = v @ gate
    return (g / (1 + np.exp(-g)) * (v @ up)) @ down

  for b in range(2):
    for t in range(5):
      v = x[b, t]
      scores = v @ p["w_router"]
      chosen = np.argsort(-scores)[:cfg.num_experts_per_tok]
      share = 1 / (1 + np.exp(-scores[chosen]))
      routed = sum(s * mlp(v, p["w_gate"][e], p["w_up"][e], p["w_down"][e])
                   for e, s in zip(chosen, share / share.sum()))
      shared = mlp(v, p["shared_gate"], p["shared_up"], p["shared_down"])
      np.testing.assert_allclose(out[b, t], (routed + shared) / 2, rtol=1e-4, atol=1e-5)


def test_padding_and_later_tokens_change_nothing_before_them():
  params = _params(images=False)
  _, (tokens, _, valid) = list(_batches())[1]
  forward = jax.jit(lambda t, v: command_a.forward(CONFIG, params, t, None, v, jnp.float32))
  out = np.asarray(forward(tokens, valid))
  assert np.isfinite(out).all()
  # Rotary positions are relative and full-attention layers have none: padding on the left
  # changes nothing.
  alone = command_a.forward(CONFIG, params, tokens[1:, 5:], None, None, jnp.float32)
  np.testing.assert_allclose(out[1:, 5:], alone, rtol=1e-4, atol=1e-4)
  changed = tokens.copy()
  changed[0, -1] = (changed[0, -1] + 1) % CONFIG.vocab_size
  again = np.asarray(forward(changed, valid))
  np.testing.assert_array_equal(again[0, :-1], out[0, :-1])
  assert not np.array_equal(again[0, -1], out[0, -1])


def test_a_sliding_layer_sees_its_window_only():
  # One layer, and with a pattern of 2 it is a sliding one.
  cfg = dataclasses.replace(CONFIG, num_hidden_layers=1, sliding_window_pattern=2, sliding_window=4)
  assert cfg.sliding == (True,)
  params = _params(cfg, images=False)
  tokens = np.random.default_rng(3).integers(6, cfg.vocab_size, (1, 10)).astype(np.int32)
  out = np.asarray(command_a.forward(cfg, params, tokens, None, None, jnp.float32))
  changed = tokens.copy()
  changed[0, 0] += 1
  again = np.asarray(command_a.forward(cfg, params, changed, None, None, jnp.float32))
  # Position 3 is the last whose window of 4 holds position 0.
  assert not np.array_equal(again[0, 3], out[0, 3])
  np.testing.assert_array_equal(again[0, 4:], out[0, 4:])
  assert CONFIG.sliding == (True, True, True, False) * 2


def test_image_vectors_take_the_places_of_image_tokens():
  params = _params()
  _, (tokens, pixels, _) = list(_batches())[2]
  h = np.asarray(command_a.embed(CONFIG, params, tokens, pixels, jnp.float32))
  features = np.asarray(command_a.image_features(CONFIG, params, pixels, jnp.float32))
  assert features.shape == (2, 2 * CONFIG.tile_tokens, CONFIG.hidden_size)
  image = tokens == CONFIG.image_token_id
  assert image.sum(1).tolist() == [2 * CONFIG.tile_tokens, CONFIG.tile_tokens]
  np.testing.assert_array_equal(h[0][image[0]], features[0])
  np.testing.assert_array_equal(h[1][image[1]], features[1, :CONFIG.tile_tokens])
  np.testing.assert_array_equal(h[~image], np.asarray(params["embed"])[tokens[~image]])
  # A sequence's tiles reach that sequence only, and only from its first image token on.
  other = pixels.copy()
  other[0] += 1
  forward = jax.jit(lambda p: command_a.forward(CONFIG, params, tokens, p, None, jnp.float32))
  out, again = np.asarray(forward(pixels)), np.asarray(forward(other))
  np.testing.assert_array_equal(again[1], out[1])
  np.testing.assert_array_equal(again[0, :6], out[0, :6])
  assert not np.array_equal(again[0, 6:], out[0, 6:])
