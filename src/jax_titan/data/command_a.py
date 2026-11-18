"""What Command A+ reads: a conversation as token ids and its images as tiles.

The published checkpoint ships a tokenizer, a chat template and the settings
of its image processor. `Preprocessor` applies them without PyTorch: Jinja
renders the template, `tokenizers` encodes the text, and Pillow resizes a
picture into tiles. A picture gets a grid of tiles of the model's size, chosen
by its own size, and from two tiles on also one of the whole picture; in the
text it stands as one image token per vector of every tile.

Pillow's resampling is not torchvision's, which transformers' image processor
uses: the tiles are within 2 of 255 levels of its tiles, and equal at more than
99 pixel values in 100.
"""

import dataclasses
import json
import pathlib

import numpy as np


def grid(height: int, width: int, tile: int, most: int) -> tuple[int, int]:
  """Columns and rows of the grid of `tile`-sized tiles for a picture, at most `most` tiles.

  Of the grids the picture fits into without shrinking, the one it is
  enlarged least for; if none, the one it is shrunk least for. Among equals,
  the one with fewer tiles, then with fewer columns.
  """
  grids = sorted(((c, r) for c in range(1, most + 1) for r in range(1, most + 1) if c * r <= most),
                 key=lambda g: g[0] * g[1])
  scales = (np.asarray(grids) * tile / np.asarray([width, height])).min(axis=-1)
  if (scales < 1).all():
    return grids[int(np.argmax(scales))]
  return grids[int(np.argmin(np.where(scales < 1, np.inf, scales)))]


@dataclasses.dataclass(frozen=True)
class Preprocessor:
  """The tokenizer, the chat template and the image settings of a published checkpoint."""

  tokenizer: object
  template: object
  special: dict
  tile: int
  most_tiles: int
  tile_tokens: int

  @classmethod
  def open(cls, checkpoint) -> "Preprocessor":
    """From the checkpoint directory's `tokenizer.json`, `tokenizer_config.json`,
    `chat_template.jinja` and `preprocessor_config.json`."""
    import jinja2.ext  # pylint: disable=g-import-not-at-top
    import jinja2.sandbox  # pylint: disable=g-import-not-at-top
    from tokenizers import Tokenizer  # pylint: disable=g-import-not-at-top

    root = pathlib.Path(checkpoint)
    special = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    images = json.loads((root / "preprocessor_config.json").read_text(encoding="utf-8"))
    # The environment transformers renders chat templates in.
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda value, indent=None: json.dumps(
        value, ensure_ascii=False, indent=indent)
    tile = images["size"]["height"]
    return cls(
        tokenizer=Tokenizer.from_file(str(root / "tokenizer.json")),
        template=env.from_string((root / "chat_template.jinja").read_text(encoding="utf-8")),
        special={k: v for k, v in special.items() if k.endswith("_token") and isinstance(v, str)},
        tile=tile, most_tiles=images["max_patches"],
        tile_tokens=(tile // images["patch_size"] // images["downsample_factor"]) ** 2)

  def text(self, messages: list[dict], add_generation_prompt: bool = True, **variables) -> str:
    """The conversation as the chat template writes it, one image token for each image.

    A message is `{"role", "content"}`; its content is a string, or a list of
    `{"type": "text", "text"}` and `{"type": "image"}` parts. `variables` are
    the template's own, such as `tools` and `documents`.
    """
    return self.template.render(
        messages=messages, add_generation_prompt=add_generation_prompt, **self.special, **variables)

  def tiles(self, image) -> np.ndarray:
    """The tiles of a Pillow image, [tiles, 3, tile, tile] in [-1, 1]: its grid
    row by row and, from two tiles on, the whole picture."""
    from PIL import Image  # pylint: disable=g-import-not-at-top

    image, size = image.convert("RGB"), self.tile
    columns, rows = grid(image.height, image.width, size, self.most_tiles)
    canvas = np.asarray(image.resize((columns * size, rows * size), Image.Resampling.BICUBIC))
    tiles = [canvas[r * size:(r + 1) * size, c * size:(c + 1) * size]
             for r in range(rows) for c in range(columns)]
    if len(tiles) > 1:
      tiles.append(np.asarray(image.resize((size, size), Image.Resampling.BICUBIC)))
    return (np.stack(tiles).transpose(0, 3, 1, 2).astype(np.float32) / 255 - 0.5) / 0.5

  def encode(self, messages: list[dict], images=(),
             **variables) -> tuple[np.ndarray, np.ndarray | None]:
    """Token ids [tokens] of the conversation and the tiles [tiles, 3, tile, tile]
    of `images`, Pillow images in the order of the conversation's image parts;
    without images, no tiles."""
    token, start, end, line = (self.special[k] for k in (
        "image_token", "boi_token", "eoi_token", "img_line_break_token"))
    pieces = self.text(messages, **variables).split(token)
    if len(pieces) != len(images) + 1:
      raise ValueError(
          f"the conversation has {len(pieces) - 1} image parts for {len(images)} images")
    tiles = [self.tiles(image) for image in images]
    text = pieces[0]
    for image, piece in zip(tiles, pieces[1:]):
      text += start + (token * self.tile_tokens + line) * len(image) + end + piece
    ids = np.asarray(self.tokenizer.encode(text, add_special_tokens=False).ids, np.int32)
    return ids, np.concatenate(tiles) if tiles else None

  def decode(self, ids) -> str:
    """The text of token ids, special tokens included."""
    return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)
