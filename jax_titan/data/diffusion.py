"""Synthetic training batches of the diffusion families, and with them the
objective JAX-Titan trains a denoiser on.

A denoiser's repository keeps the training objective outside the model, in a
trainer or an example script: how a latent is noised, which time the denoiser
is given and what it is to predict. JAX-Titan states it here, once per family,
for every adapter that runs the family. Both families are rectified flows with
an unweighted mean squared error: the denoiser sees
`(1 - s) latents + s noise` and predicts `noise - latents`.

- FLUX.1: `s` uniform in [0, 1), and the denoiser is given `s`.
  MaxDiffusion's trainer (`trainers/flux_trainer.py`) has the same noising and
  target and draws `s` from the grid of its sampler's steps.
- Wan 2.1: `s = shift u / (1 + (shift - 1) u)` with `u` uniform and the 1.3B
  model's shift of 3, and the denoiser is given `1000 s`, as in MaxDiffusion's
  trainer (`trainers/wan_trainer.py`, `scheduling_flow_match_flax.py`). That
  trainer also weights the samples of a batch against each other, by a bell
  curve over the time less its minimum in the batch; that weight is left out.

Latents and text features are standard normal, until an image or video
pipeline exists. A batch has the leading axes [microbatches, microbatch_size];
the names in `constants` have [microbatches] alone.
"""

from collections.abc import Callable
import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Batches:
  """`next_batch(step)` gives `(inputs, target)`; `tokens` counts one sample."""

  next_batch: Callable[[int], tuple[dict, np.ndarray]]
  constants: tuple[str, ...]
  tokens: int


def flux_position_ids(image_side: int, text_tokens: int) -> tuple[np.ndarray, np.ndarray]:
  """`img_ids` [image_side**2, 3] and `txt_ids` [text_tokens, 3] of FLUX.1.

  Image token (row, column) has ids (0, row, column); text tokens have zeros.
  """
  img_ids = np.zeros((image_side, image_side, 3), np.float32)
  img_ids[..., 1] = np.arange(image_side)[:, None]
  img_ids[..., 2] = np.arange(image_side)[None, :]
  return img_ids.reshape(-1, 3), np.zeros((text_tokens, 3), np.float32)


def flux(seed: int, batch: tuple[int, int], *, image_side: int, text_tokens: int, in_channels: int,
         joint_attention_dim: int, pooled_projection_dim: int, guidance_embeds: bool) -> Batches:
  """FLUX.1: `img` latents [.., image_side**2, in_channels], `txt` T5 features,
  `pooled` CLIP features, `timestep`, the position ids and, with
  `guidance_embeds`, a `guidance` of 3.5; the target is the velocity."""
  img_ids, txt_ids = flux_position_ids(image_side, text_tokens)
  latent_shape = batch + (image_side * image_side, in_channels)

  def next_batch(step):
    rng = np.random.default_rng((seed, step))
    latents = rng.standard_normal(latent_shape, np.float32)
    noise = rng.standard_normal(latent_shape, np.float32)
    t = rng.uniform(size=batch).astype(np.float32)
    x = {
        "img": (1 - t[..., None, None]) * latents + t[..., None, None] * noise,
        "txt": rng.standard_normal(batch + (text_tokens, joint_attention_dim), np.float32),
        "pooled": rng.standard_normal(batch + (pooled_projection_dim,), np.float32),
        "timestep": t,
        "img_ids": np.broadcast_to(img_ids, batch[:1] + img_ids.shape),
        "txt_ids": np.broadcast_to(txt_ids, batch[:1] + txt_ids.shape),
    }
    if guidance_embeds:
      x["guidance"] = np.full(batch, 3.5, np.float32)
    return x, noise - latents

  return Batches(next_batch, ("img_ids", "txt_ids"), image_side * image_side + text_tokens)


def wan(seed: int, batch: tuple[int, int], *, frames: int, height: int, width: int,
        patch: tuple[int, int, int], text_tokens: int, in_channels: int, text_dim: int,
        shift: float = 3.0) -> Batches:
  """Wan 2.1 text to video: `latents` [.., in_channels, frames, height, width]
  of the video VAE, `txt` features of the text encoder and `timestep`; the
  target is the velocity. A token is one `patch` of the latents."""
  latent_shape = batch + (in_channels, frames, height, width)

  def next_batch(step):
    rng = np.random.default_rng((seed, step))
    latents = rng.standard_normal(latent_shape, np.float32)
    noise = rng.standard_normal(latent_shape, np.float32)
    u = rng.uniform(size=batch).astype(np.float32)
    s = shift * u / (1 + (shift - 1) * u)
    sb = s.reshape(batch + (1, 1, 1, 1))
    x = {
        "latents": (1 - sb) * latents + sb * noise,
        "txt": rng.standard_normal(batch + (text_tokens, text_dim), np.float32),
        "timestep": 1000 * s,
    }
    return x, noise - latents

  tokens = (frames // patch[0]) * (height // patch[1]) * (width // patch[2])
  return Batches(next_batch, (), tokens + text_tokens)
