"""Qwen3-VL's text tower gives transformers' hidden states, and padding changes nothing."""

import importlib.util
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_titan.models import qwen3

CONFIG = {
    "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 48, "feature_layers": (0, 2, 3),
}


def _tokens(cfg, rng, tokens=7, padded=3):
  """Two prompts, the second with `padded` positions of padding on the left."""
  token_ids = rng.integers(1, cfg.vocab_size, (2, tokens)).astype(np.int32)
  valid = np.ones((2, tokens), bool)
  token_ids[1, :padded], valid[1, :padded] = 0, False
  return token_ids, valid


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="needs torch and transformers")
def test_features_match_transformers():
  os.environ.setdefault("USE_TF", "0")
  import torch  # pylint: disable=g-import-not-at-top

  hf = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
  hf_config = pytest.importorskip("transformers.models.qwen3_vl.configuration_qwen3_vl")

  cfg = qwen3.Qwen3Config(**CONFIG)
  torch.manual_seed(0)
  # The rotary sections end below the head's 8 frequencies, as they do at full size.
  model = hf.Qwen3VLTextModel(hf_config.Qwen3VLTextConfig(
      rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta,
      rope_scaling={"rope_type": "default", "mrope_section": [3, 2, 2]},
      **{k: v for k, v in CONFIG.items() if k != "feature_layers"})).eval()
  with torch.no_grad():
    for name, p in model.named_parameters():
      if p.ndim == 1:
        p.normal_(1.0, 0.1)
  params = qwen3.params_from_tensors(
      {"language_model." + k: v.numpy() for k, v in model.state_dict().items()}, cfg)

  token_ids, valid = _tokens(cfg, np.random.default_rng(0))
  positions = np.maximum(np.cumsum(valid, axis=1) - 1, 0)
  # As Ideogram's pipeline runs the encoder: a causal mask without the padding,
  # and the same position on the three rotary axes.
  tokens = token_ids.shape[1]
  attend = np.tril(np.ones((tokens, tokens), bool)) & valid[:, None, :]
  attend = torch.from_numpy(attend | np.eye(tokens, dtype=bool))
  with torch.no_grad():
    mask = torch.from_numpy(valid).long()
    h = model.embed_tokens(torch.from_numpy(token_ids).long())
    position_ids = torch.from_numpy(positions).long()
    causal_mask = torch.where(attend, 0.0, torch.finfo(torch.float32).min)[:, None]
    position_embeddings = model.rotary_emb(h, position_ids[None].expand(3, -1, -1))
    taps = []
    for i, layer in enumerate(model.layers):
      h = layer(h, attention_mask=causal_mask, position_ids=position_ids, past_key_values=None,
                position_embeddings=position_embeddings)
      if i in cfg.feature_layers:
        taps.append(h)
    ref = torch.stack(taps, dim=-1).reshape(*token_ids.shape, -1) * mask[..., None]

  out = jax.jit(lambda p, i, v: qwen3.features(cfg, p, i, v, jnp.float32))(params, token_ids, valid)
  np.testing.assert_allclose(out, ref.numpy(), rtol=1e-4, atol=1e-4)


def test_padding_changes_no_feature():
  cfg = qwen3.Qwen3Config(**CONFIG)
  params = qwen3.init_params(cfg, jax.random.key(0))
  token_ids, valid = _tokens(cfg, np.random.default_rng(1))
  out = np.asarray(qwen3.features(cfg, params, token_ids, valid, jnp.float32))
  assert out.shape == (2, 7, cfg.features_dim)
  assert np.isfinite(out).all()
  np.testing.assert_array_equal(out[1, :3], 0)
  alone = qwen3.features(cfg, params, token_ids[1:, 3:], valid[1:, 3:], jnp.float32)
  np.testing.assert_allclose(out[1:, 3:], alone, rtol=1e-5, atol=1e-5)
  # A later token changes no earlier feature.
  changed = token_ids.copy()
  changed[0, -1] = (changed[0, -1] + 1) % cfg.vocab_size
  again = np.asarray(qwen3.features(cfg, params, changed, valid, jnp.float32))
  np.testing.assert_array_equal(again[0, :-1], out[0, :-1])
  assert not np.array_equal(again[0, -1], out[0, -1])
