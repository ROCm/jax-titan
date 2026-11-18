"""Text generation with Command A+ on published weights, from text and images.

  python -m jax_titan.models.command_a_generate --checkpoint=DIR \
      --prompt="What is in this image?" --image=picture.png

DIR is a checkpoint in the published layout, such as a snapshot of
`CohereLabs/command-a-plus-05-2026-bf16`: the weights with `config.json`, the
tokenizer, the chat template and the image processor's settings. The prompt
is the user's turn of a conversation that the chat template writes
(`jax_titan.data.command_a`), with the images before the text. Every new
token comes from a pass over the whole sequence, on one device: there is no
key-value cache. `--params` replaces weights, for example with fine-tuned ones.
"""

import argparse
import os
import pathlib
import time

import numpy as np


def generate(cfg, params, tokens, pixels=None, *, new_tokens: int, dtype=None,
             temperature: float = 0.0, seed: int = 0) -> np.ndarray:
  """The continuations [batch, up to new_tokens] of prompts `tokens` [batch, prompt].

  Each step takes the likeliest token, or with `temperature` draws one from
  the scores divided by it. A sequence that reaches the end-of-sequence token
  is padded after it, and generation stops once all have.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models import command_a  # pylint: disable=g-import-not-at-top

  dtype = dtype or jnp.bfloat16
  tokens = np.asarray(tokens, np.int32)
  batch, prompt = tokens.shape

  @jax.jit
  def scores(params, buffer, pixels, last):
    # Later positions hold padding, which no earlier position attends to.
    h = command_a.embed(cfg, params, buffer, pixels, dtype)
    h = command_a.hidden(cfg, params["layers"], h)
    h = jnp.take(h, last, axis=1)[:, None]
    return command_a.logits(cfg, params, h)[:, 0].astype(jnp.float32)

  buffer = np.full((batch, prompt + new_tokens), cfg.pad_token_id, np.int32)
  buffer[:, :prompt] = tokens
  rng, done = np.random.default_rng(seed), np.zeros(batch, bool)
  for position in range(prompt, prompt + new_tokens):
    out = np.asarray(scores(params, buffer, pixels, position - 1))
    if temperature > 0:
      out = out / temperature + rng.gumbel(size=out.shape)
    buffer[:, position] = np.where(done, cfg.pad_token_id, out.argmax(-1))
    done |= buffer[:, position] == cfg.eos_token_id
    if done.all():
      return buffer[:, prompt:position + 1]
  return buffer[:, prompt:]


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint", required=True, help="A checkpoint in the published layout.")
  parser.add_argument("--prompt", required=True,
                      help="The user's message, or @FILE to read it from a file.")
  parser.add_argument("--image", action="append", default=[],
                      help="A picture the message is about. Repeat for several.")
  parser.add_argument("--system", default="", help="A system message before the user's.")
  parser.add_argument("--max_new_tokens", type=int, default=256)
  parser.add_argument("--temperature", type=float, default=0.0,
                      help="0 takes the likeliest token; above 0, tokens are drawn.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--params", default="",
                      help="Parameters that the trainer's --save_params wrote, instead of the "
                      "checkpoint's.")
  parser.add_argument("--platform", default="", help="cpu to run without a GPU.")
  args = parser.parse_args(argv)
  if args.platform:
    os.environ["JAX_PLATFORMS"] = args.platform

  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from PIL import Image  # pylint: disable=g-import-not-at-top
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top
  from jax_titan.models import command_a  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import checkpoint  # pylint: disable=g-import-not-at-top

  prompt = args.prompt
  if prompt.startswith("@"):
    prompt = pathlib.Path(prompt[1:]).read_text(encoding="utf-8").strip()
  images = [Image.open(path) for path in args.image]
  content = [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]
  messages = [{"role": "system", "content": args.system}] if args.system else []
  messages.append({"role": "user", "content": content if images else prompt})

  preprocessor = data.Preprocessor.open(args.checkpoint)
  tokens, tiles = preprocessor.encode(messages, images)
  start = time.time()
  cfg, params = command_a.load(args.checkpoint, images=bool(images))
  if args.params:
    # The trainer saves the language model, and the vision tower only when it trained on images.
    trained = command_a.Model(cfg, images="vision" in checkpoint.saved_keys(args.params))
    params |= jax.tree.map(lambda w: w.astype(jnp.bfloat16), trained.load(args.params)[0])
  print(f"loaded after {time.time() - start:.0f} s; the prompt is {len(tokens)} tokens"
        + (f" with {len(tiles)} image tiles" if images else ""))
  start = time.time()
  new = generate(cfg, params, tokens[None], None if tiles is None else tiles[None],
                 new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed)[0]
  print(f"generated {len(new)} tokens in {time.time() - start:.0f} s")
  print(preprocessor.decode(new))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
