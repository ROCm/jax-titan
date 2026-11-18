"""Command A+ gives transformers' logits and gradients, reads and writes the
published checkpoint layout, routes tokens to experts and masks what it must."""

import dataclasses

import command_a_checkpoint
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_titan.models import command_a

CONFIG = command_a_checkpoint.CONFIG
_params = command_a_checkpoint.params
_batches = command_a_checkpoint.batches
needs_reference = command_a_checkpoint.needs_reference


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


def _loss_and_gradient_norm(name):
  """Of `_params()` on a batch of `_batches()`, in fp32."""
  tokens, pixels, valid = dict(_batches())[name]
  params = _params(images=pixels is not None)
  (loss, _), grads = jax.value_and_grad(_loss, argnums=1, has_aux=True)(
      CONFIG, params, tokens, pixels, valid)
  return float(loss), float(optax.global_norm(grads))


# What `_loss_and_gradient_norm` gives on CPU, where the model matches transformers.
RECORDED = {"text": (6.657514, 5.484844), "image": (7.171245, 6.391755)}


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_loss_and_gradient_are_the_recorded_ones(name):
  assert _loss_and_gradient_norm(name) == pytest.approx(RECORDED[name], rel=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(RECORDED))
def test_a_gpu_gives_the_loss_and_gradient_of_cpu(name):
  assert _loss_and_gradient_norm(name) == pytest.approx(RECORDED[name], rel=1e-5)


@needs_reference
def test_matches_transformers(tmp_path):
  command_a.save(tmp_path, CONFIG, _params())
  model = command_a_checkpoint.reference(tmp_path)
  cfg, params = command_a.load(tmp_path, jnp.float32)
  assert cfg == CONFIG
  for name, (tokens, pixels, valid) in _batches():
    inputs = command_a_checkpoint.reference_inputs(tokens, pixels, valid)
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
  command_a_checkpoint.reference(tmp_path / "ours").save_pretrained(tmp_path / "theirs")
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


@pytest.mark.parametrize("layers", [(3, 5), (2, 3, 3)])
def test_stages_match_one_stage(layers):
  params = _params()
  one = command_a.Model(CONFIG, dtype=jnp.float32, images=True)
  staged = command_a.Model(CONFIG, num_stages=len(layers), layers=layers, dtype=jnp.float32,
                           images=True, freeze=("embed",))
  stages = staged.split(params)
  assert [len(p["layers"]["norm"]) for p in stages] == list(layers)
  assert "embed" in stages[0] and "embed" in stages[-1] and "final_norm" in stages[-1]
  for name in ("text", "image"):
    tokens, pixels, _ = dict(_batches())[name]
    x = tokens if pixels is None else {"tokens": tokens, "pixels": pixels}
    expected = one.forward(0, one.split(params)[0], x)
    for i, p in enumerate(stages):
      x = staged.forward(i, p, x)
    np.testing.assert_allclose(x, expected, rtol=1e-5, atol=1e-5, err_msg=name)
  jax.tree.map(np.testing.assert_array_equal, staged.merge(stages), params)
  with pytest.raises(NotImplementedError, match="embedding frozen"):
    command_a.Model(CONFIG, num_stages=2)
  with pytest.raises(ValueError, match="do not split 8 layers"):
    command_a.Model(CONFIG, num_stages=2, layers=(3, 4), freeze=("embed",))


def test_frozen_groups_name_the_embedding_and_the_routed_experts():
  model = command_a.Model(CONFIG, images=True, freeze=("experts", "embed"))
  frozen = {path for path in (
      ("embed",), ("final_norm",), ("layers", "norm"), ("layers", "wq"), ("layers", "w_router"),
      ("layers", "w_gate"), ("layers", "w_up"), ("layers", "w_down"), ("layers", "shared_gate"),
      ("vision", "layers", "wq"), ("projector", "w1")) if model.is_frozen(path)}
  assert frozen == {("embed",), ("layers", "w_gate"), ("layers", "w_up"), ("layers", "w_down")}
  assert not command_a.Model(CONFIG).is_frozen(("embed",))


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


def test_bfloat16_stays_near_float32():
  params = _params()
  for name, (tokens, pixels, valid) in _batches():
    exact = np.asarray(command_a.forward(CONFIG, params, tokens, pixels, valid, jnp.float32))
    rounded = command_a.forward(CONFIG, params, tokens, pixels, valid, jnp.bfloat16)
    assert rounded.dtype == jnp.bfloat16
    real = np.ones(tokens.shape, bool) if valid is None else valid
    apart = np.abs(np.asarray(rounded, np.float32) - exact)[real]
    # A router's choice between near-equal experts can flip in bf16, which moves single
    # logits more than the mean.
    assert np.isfinite(apart).all() and apart.mean() < 0.1, name


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
