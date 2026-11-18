"""A cache of Ideogram 4 training samples: image tokens and text features.

  python -m jax_titan.data.ideogram --checkpoint=DIR --images=IMAGES --out=CACHE

IMAGES is a directory of image files. An image's caption is the text of the
file beside it with the same name and the suffix `.json` or `.txt`, or else
`--prompt`. Ideogram 4 is trained on structured JSON captions, so a caption
should be one. DIR is a checkpoint in diffusers' layout; its autoencoder
turns every image, resized and center-cropped to a square of `--resolution`
pixels, into the transformer's image tokens, and its text encoder turns every
distinct caption into text features. Both models are frozen in training, so
they run here, once, and the trainer reads the cache (`--data=CACHE`).

CACHE holds `image_tokens.npy` [images, tokens, channels] in fp32,
`text_features.npy` [captions, text tokens, features] in fp16 and zero at
padding, `text_valid.npy` [captions, text tokens], true at the real tokens,
which follow the padding, `caption_of_image.npy` [images], and `meta.json`.
"""

import argparse
import dataclasses
import json
import os
import pathlib

import numpy as np

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclasses.dataclass(frozen=True)
class Cache:
  """A cache directory, its arrays mapped into memory."""

  image_tokens: np.ndarray
  text_features: np.ndarray
  text_valid: np.ndarray
  caption_of_image: np.ndarray
  resolution: int

  @classmethod
  def open(cls, path) -> "Cache":
    path = pathlib.Path(path)
    with open(path / "meta.json", encoding="utf-8") as f:
      meta = json.load(f)
    return cls(
        image_tokens=np.load(path / "image_tokens.npy", mmap_mode="r"),
        text_features=np.load(path / "text_features.npy", mmap_mode="r"),
        text_valid=np.load(path / "text_valid.npy"),
        caption_of_image=np.load(path / "caption_of_image.npy"),
        resolution=meta["resolution"],
    )

  def __len__(self) -> int:
    return len(self.image_tokens)

  @property
  def text_tokens(self) -> int:
    return self.text_valid.shape[1]

  def samples(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image tokens, text features in fp32 and the mask of real text tokens of `images`."""
    captions = self.caption_of_image[images]
    return (np.asarray(self.image_tokens[images], np.float32),
            np.asarray(self.text_features[captions], np.float32), self.text_valid[captions])


def load_image(path, resolution: int) -> np.ndarray:
  """The image at `path` as [resolution, resolution, 3] in [-1, 1]: resized to
  cover the square, then cropped at the center."""
  from PIL import Image, ImageOps  # pylint: disable=g-import-not-at-top

  with Image.open(path) as image:
    image = ImageOps.fit(image.convert("RGB"), (resolution, resolution), Image.Resampling.BICUBIC)
  return np.asarray(image, np.float32) / 127.5 - 1.0


def prepare(checkpoint, images, out, *, resolution: int = 1024, prompt: str = "",
            log=print) -> None:
  """Writes the cache of the images under `images` to the new directory `out`."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models import flux2_vae, qwen3  # pylint: disable=g-import-not-at-top

  checkpoint, out = pathlib.Path(checkpoint), pathlib.Path(out)
  files = sorted(p for p in pathlib.Path(images).iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
  if not files:
    raise FileNotFoundError(f"no image file in {images}")
  captions, caption_of_image = [], []
  for file in files:
    beside = [file.with_suffix(s) for s in (".json", ".txt") if file.with_suffix(s).exists()]
    caption = beside[0].read_text(encoding="utf-8").strip() if beside else prompt
    if not caption:
      raise ValueError(f"{file} has no caption beside it and --prompt is empty")
    if caption not in captions:
      captions.append(caption)
    caption_of_image.append(captions.index(caption))
  out.mkdir(parents=True)

  def on_device(tree, to=jnp.float32):
    # Cast on the device: a host cast of the fp32 weights to bf16 is slow.
    return jax.tree.map(lambda x: jnp.asarray(x).astype(to), tree)

  vae_cfg = flux2_vae.pretrained_config(checkpoint / "vae")
  vae = on_device(flux2_vae.pretrained_params(checkpoint / "vae", vae_cfg))
  encode = jax.jit(lambda p, x: flux2_vae.to_tokens(vae_cfg, p, flux2_vae.encode(vae_cfg, p, x)))
  tokens = np.stack([np.asarray(encode(vae, load_image(f, resolution)[None]))[0] for f in files])
  np.save(out / "image_tokens.npy", tokens)
  del vae
  log(f"{len(files)} images as {tokens.shape[1]} tokens each")

  text_cfg = qwen3.pretrained_config(checkpoint / "text_encoder")
  text = on_device(qwen3.pretrained_params(checkpoint / "text_encoder", text_cfg), jnp.bfloat16)
  token_ids, valid = qwen3.tokenize(checkpoint / "tokenizer", captions)
  features = jax.jit(lambda p, i, v: qwen3.features(text_cfg, p, i, v))
  # One caption at a time: the features of all of them need not fit on the device.
  np.save(out / "text_features.npy", np.concatenate([
      np.asarray(features(text, token_ids[i:i + 1], valid[i:i + 1])).astype(np.float16)
      for i in range(len(captions))]))
  np.save(out / "text_valid.npy", valid)
  np.save(out / "caption_of_image.npy", np.asarray(caption_of_image, np.int32))
  log(f"{len(captions)} captions of up to {valid.shape[1]} tokens")
  with open(out / "meta.json", "w", encoding="utf-8") as f:
    json.dump({"resolution": resolution, "images": [p.name for p in files], "captions": captions,
               "checkpoint": checkpoint.name}, f, indent=1, ensure_ascii=False)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint", required=True, help="A checkpoint in diffusers' layout.")
  parser.add_argument("--images", required=True, help="A directory of image files.")
  parser.add_argument("--out", required=True,
                      help="The cache directory to write; it must not exist.")
  parser.add_argument("--resolution", type=int, default=1024,
                      help="Side in pixels, a multiple of 16.")
  parser.add_argument("--prompt", default="",
                      help="The caption of images without a caption file, or @FILE to read it.")
  parser.add_argument("--platform", default="", help="cpu to run without a GPU.")
  args = parser.parse_args(argv)
  if args.resolution % 16:
    parser.error("--resolution must be a multiple of 16")
  if args.platform:
    os.environ["JAX_PLATFORMS"] = args.platform
  prompt = args.prompt
  if prompt.startswith("@"):
    prompt = pathlib.Path(prompt[1:]).read_text(encoding="utf-8").strip()
  prepare(args.checkpoint, args.images, args.out, resolution=args.resolution, prompt=prompt)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
