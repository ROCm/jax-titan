"""Text-to-image generation with Ideogram 4 on published weights.

  python -m jax_titan.models.ideogram_generate --checkpoint=DIR \
      --prompt='{"high_level_description": "..."}' --out=image.png

DIR is a checkpoint in diffusers' layout, such as a snapshot of
`ideogram-ai/ideogram-4-fp8`: `tokenizer/`, `text_encoder/`, `transformer/`,
`unconditional_transformer/` and `vae/`. The prompt goes through the text
encoder (`jax_titan.models.qwen3`); the image tokens start as noise and take
Euler steps along the velocity the two transformers predict, the conditional
one on text and image tokens and the unconditional one on the image tokens
alone, blended by a guidance weight; the autoencoder
(`jax_titan.models.flux2_vae`) decodes the result. Everything runs on one
device. `--transformer` replaces the conditional transformer's weights, for
example with fine-tuned ones.

Ideogram 4 is trained on structured JSON captions; plain text works less well.
"""

import argparse
import math
import os
import pathlib
import statistics
import time

import numpy as np

# Steps, guidance weight of each step from the last one back to the first, and
# the mean and deviation of the schedule: the presets of Ideogram's pipeline.
PRESETS = {
    "quality": (48, (3.0,) * 3 + (7.0,) * 45, 0.0, 1.5),
    "default": (20, (3.0,) * 2 + (7.0,) * 18, 0.0, 1.75),
    "turbo": (12, (3.0,) * 1 + (7.0,) * 11, 0.5, 1.75),
}


def times(steps: int, height: int, width: int, mean: float, std: float) -> np.ndarray:
  """The timesteps between the steps, `steps + 1` of them from data (near 1)
  down to noise (near 0): a logit-normal schedule whose mean grows with the
  image size."""
  mu = mean + 0.5 * math.log(height * width / 512**2)
  low, high = 1 / (1 + math.exp(9.0)), 1 / (1 + math.exp(-7.5))
  normal = statistics.NormalDist()
  out = [high]
  for i in range(1, steps):
    out.append(min(max(1 - 1 / (1 + math.exp(-(mu + std * normal.inv_cdf(i / steps)))), low), high))
  return np.asarray(out + [low], np.float32)


class Pipeline:
  """Ideogram 4's text encoder, two transformers and autoencoder on one device,
  loaded once from a checkpoint in diffusers' layout."""

  def __init__(self, checkpoint, *, transformer=None, log=print):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    from jax_titan.models import flux2_vae, ideogram, qwen3  # pylint: disable=g-import-not-at-top

    checkpoint = pathlib.Path(checkpoint)
    dtype = jnp.bfloat16
    start = time.time()

    def on_device(tree, to=None):
      return jax.tree.map(lambda x: jnp.asarray(x, to), tree)

    self._tokenizer = checkpoint / "tokenizer"
    text_cfg = qwen3.pretrained_config(checkpoint / "text_encoder")
    self._text = on_device(qwen3.pretrained_params(checkpoint / "text_encoder", text_cfg), dtype)
    self._features = jax.jit(lambda p, i, v: qwen3.features(text_cfg, p, i, v, dtype))
    log(f"text encoder after {time.time() - start:.0f} s")

    self.config = cfg = ideogram.pretrained_config(checkpoint / "transformer")
    model = ideogram.Model(cfg, num_stages=1, dtype=dtype, remat=False)
    self._conditional = on_device(
        model.load(transformer or checkpoint / "transformer")[0], dtype)
    self._unconditional = on_device(model.load(checkpoint / "unconditional_transformer")[0], dtype)
    log(f"both transformers after {time.time() - start:.0f} s")

    def step(conditional, unconditional, z, features, ids, t, s, weight):
      """One Euler step from timestep `t` to `s` along the guided velocity."""
      batch, image_tokens, _ = z.shape
      timestep = jnp.full((batch,), t, jnp.float32)
      with_text = model.forward(
          0, conditional, {"img": z, "txt": features, "timestep": timestep, **ids})
      # The unconditional transformer sees the image tokens alone.
      alone = model.forward(0, unconditional, {
          "img": z, "txt": features[:, :0], "timestep": timestep,
          **{k: v[:, -image_tokens:] for k, v in ids.items()}})
      velocity = (weight * with_text[:, -image_tokens:].astype(jnp.float32)
                  + (1 - weight) * alone.astype(jnp.float32))
      return z + velocity * (s - t)

    self._step = jax.jit(step)

    vae_cfg = flux2_vae.pretrained_config(checkpoint / "vae")
    self._vae = on_device(flux2_vae.pretrained_params(checkpoint / "vae", vae_cfg))

    def decode(vae, z):
      latents = flux2_vae.from_tokens(vae_cfg, vae, z, math.isqrt(z.shape[1]))
      images = jnp.clip(flux2_vae.decode(vae_cfg, vae, latents), -1, 1)
      return jnp.round((images + 1) * 127.5).astype(jnp.uint8)

    self._decode = jax.jit(decode)
    log(f"loaded after {time.time() - start:.0f} s")

  def generate(self, prompts: list[str], *, resolution: int = 1024, preset: str = "default",
               seed: int = 0) -> np.ndarray:
    """Images [prompts, resolution, resolution, 3] in uint8, one per prompt."""
    from jax_titan.models import ideogram, qwen3  # pylint: disable=g-import-not-at-top

    steps, guidance, mean, std = PRESETS[preset]
    token_ids, valid = qwen3.tokenize(self._tokenizer, prompts)
    features = self._features(self._text, token_ids, valid)
    side = resolution // 16
    ids = ideogram.padded_sequence_ids(side, valid)
    ts = times(steps, resolution, resolution, mean, std)
    z = np.random.default_rng(seed).standard_normal(
        (len(prompts), side * side, self.config.in_channels), np.float32)
    for i in reversed(range(steps)):
      z = self._step(self._conditional, self._unconditional, z, features, ids,
                     ts[i + 1], ts[i], np.float32(guidance[i]))
    return np.asarray(self._decode(self._vae, z))


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint", required=True, help="A checkpoint in diffusers' layout.")
  parser.add_argument("--prompt", action="append", required=True,
                      help="A prompt, or @FILE to read one from a file. Repeat for several images.")
  parser.add_argument("--out", required=True,
                      help="The PNG to write; with several prompts, -0, -1, ... go before "
                      "the suffix.")
  parser.add_argument("--resolution", type=int, default=1024,
                      help="Side in pixels, a multiple of 16.")
  parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--transformer", default="",
                      help="Weights of the conditional transformer instead of the checkpoint's: "
                      "published ones, or parameters that the trainer's --save_params wrote.")
  parser.add_argument("--platform", default="", help="cpu to run without a GPU.")
  args = parser.parse_args(argv)
  if args.resolution % 16:
    parser.error("--resolution must be a multiple of 16")
  if args.platform:
    os.environ["JAX_PLATFORMS"] = args.platform

  from PIL import Image  # pylint: disable=g-import-not-at-top

  prompts = [pathlib.Path(p[1:]).read_text(encoding="utf-8").strip() if p.startswith("@") else p
             for p in args.prompt]
  pipeline = Pipeline(args.checkpoint, transformer=args.transformer or None)
  start = time.time()
  images = pipeline.generate(
      prompts, resolution=args.resolution, preset=args.preset, seed=args.seed)
  print(f"generated in {time.time() - start:.0f} s")
  out = pathlib.Path(args.out)
  for i, image in enumerate(images):
    path = out if len(images) == 1 else out.with_stem(f"{out.stem}-{i}")
    Image.fromarray(image).save(path)
    print(f"wrote {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
