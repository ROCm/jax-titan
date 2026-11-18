"""The Flux2 autoencoder matches diffusers', and its tokens are 2x2 patches of the latents."""

import importlib.util

import jax
import numpy as np
import pytest

from jax_titan.models import flux2_vae

CONFIG = {"block_out_channels": (32, 64, 64), "layers_per_block": 1, "latent_channels": 4,
          "norm_num_groups": 8}


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("diffusers") is None,
    reason="needs torch and diffusers")
def test_matches_diffusers():
  import torch  # pylint: disable=g-import-not-at-top
  from diffusers import AutoencoderKLFlux2  # pylint: disable=g-import-not-at-top

  cfg = flux2_vae.Flux2VaeConfig(**CONFIG)
  levels = len(cfg.block_out_channels)
  torch.manual_seed(0)
  vae = AutoencoderKLFlux2(
      down_block_types=("DownEncoderBlock2D",) * levels,
      up_block_types=("UpDecoderBlock2D",) * levels, sample_size=32, **CONFIG).eval()
  with torch.no_grad():
    for name, p in vae.named_parameters():
      if p.ndim == 1:
        p.normal_(1.0 if "norm" in name and name.endswith("weight") else 0.0, 0.1)
  params = flux2_vae.params_from_tensors(
      {k: v.numpy() for k, v in vae.state_dict().items()}, cfg)

  images = np.random.default_rng(0).uniform(-1, 1, (2, 32, 48, 3)).astype(np.float32)
  with torch.no_grad():
    ref_latents = vae.encode(torch.from_numpy(images).permute(0, 3, 1, 2)).latent_dist.mode()
    ref_images = vae.decode(ref_latents, return_dict=False)[0]
  latents = jax.jit(lambda p, x: flux2_vae.encode(cfg, p, x))(params, images)
  np.testing.assert_allclose(latents, ref_latents.permute(0, 2, 3, 1).numpy(), rtol=1e-4, atol=1e-4)
  decoded = jax.jit(lambda p, z: flux2_vae.decode(cfg, p, z))(params, latents)
  np.testing.assert_allclose(decoded, ref_images.permute(0, 2, 3, 1).numpy(), rtol=1e-4, atol=1e-4)


def test_tokens_are_normalized_patches():
  cfg = flux2_vae.Flux2VaeConfig(**CONFIG)
  rng = np.random.default_rng(0)
  c = cfg.latent_channels
  mean = rng.normal(size=cfg.token_channels).astype(np.float32)
  std = rng.uniform(0.5, 2.0, cfg.token_channels).astype(np.float32)
  latents = rng.normal(size=(2, 6, 8, c)).astype(np.float32)
  tokens = np.asarray(flux2_vae.to_tokens(cfg, latents, mean, std))
  assert tokens.shape == (2, 3 * 4, cfg.token_channels)
  # Token (row 1, column 2) holds the patch at rows 2-3 and columns 4-5, row by row.
  patch = tokens[0, 1 * 4 + 2] * std + mean
  np.testing.assert_allclose(patch.reshape(2, 2, c), latents[0, 2:4, 4:6], rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(
      flux2_vae.from_tokens(cfg, tokens, 3, mean, std), latents, rtol=1e-5, atol=1e-6)


def test_ideogram_has_statistics_for_every_token_channel():
  from jax_titan.models import ideogram  # pylint: disable=g-import-not-at-top

  channels = flux2_vae.Flux2VaeConfig().token_channels
  assert ideogram.TOKEN_MEAN.shape == ideogram.TOKEN_STD.shape == (channels,)
  assert channels == ideogram.IdeogramConfig().in_channels
  # Deviations, of latents whose channels are of order one.
  assert 1.0 < ideogram.TOKEN_STD.min() and ideogram.TOKEN_STD.max() < 3.0
  assert np.abs(ideogram.TOKEN_MEAN).max() < 0.5
