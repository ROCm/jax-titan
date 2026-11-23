"""Text generation with Command A+ on published weights, from text and images.

  python -m jax_titan.models.command_a_generate --checkpoint=DIR \
      --prompt="What is in this image?" --image=picture.png

DIR is a checkpoint in the published layout, such as a snapshot of
`CohereLabs/command-a-plus-05-2026-bf16`: the weights with `config.json`, the
tokenizer, the chat template and the image processor's settings. The prompt
is the user's turn of a conversation that the chat template writes
(`jax_titan.data.command_a`), with the images before the text. The model's
layers are spread over the visible devices, a run of layers on each, so the
published model of 437.5 GB in bf16 needs several GPUs. Every new token comes
from a pass over the whole sequence: there is no key-value cache. `--params`
puts parameters that the trainer's `--save_params` wrote in place of the
checkpoint's: all of the model's, or the part that a fine-tuning run trained.
"""

import argparse
import os
import pathlib
import time

import numpy as np


def generate(cfg, params, tokens, pixels=None, *, new_tokens: int, dtype=None,
             temperature: float = 0.0, seed: int = 0) -> np.ndarray:
  """The continuations [batch, up to new_tokens] of prompts `tokens` [batch, prompt].

  `params` is the model's parameters on one device, or the stages of
  `jax_titan.models.command_a.to_stages` on several. The prompts of a batch
  have one length, without padding. Each step takes the likeliest token, or
  with `temperature` draws one, out of the whole vocabulary, from the scores
  divided by it. A sequence that reaches the end-of-sequence token is padded
  after it, and generation stops once all have.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models import command_a  # pylint: disable=g-import-not-at-top

  dtype = jnp.bfloat16 if dtype is None else dtype
  tokens = np.asarray(tokens, np.int32)
  batch, prompt = tokens.shape

  stages = params if isinstance(params, (list, tuple)) else [params]
  devices = [next(iter(next(iter(s["layers"].values())).devices())) for s in stages]
  starts = np.cumsum([0] + [len(next(iter(s["layers"].values()))) for s in stages])
  embed = jax.jit(lambda stage, buffer, pixels: command_a.embed(cfg, stage, buffer, pixels, dtype))
  runs = [jax.jit(lambda layers, h, first=int(first): command_a.hidden(cfg, layers, h, first=first))
          for first in starts[:-1]]

  @jax.jit
  def head(stage, h, last):
    h = jnp.take(h, last, axis=1)[:, None]
    return command_a.logits(cfg, stage, h)[:, 0].astype(jnp.float32)

  def scores(buffer, last):
    # Later positions hold padding, which no earlier position attends to.
    h = embed(stages[0], jax.device_put(buffer, devices[0]), pixels)
    for stage, device, run in zip(stages, devices, runs):
      h = run(stage["layers"], jax.device_put(h, device))
    return head(stages[-1], h, last)

  if pixels is not None:
    pixels = jax.device_put(pixels, devices[0])
  buffer = np.full((batch, prompt + new_tokens), cfg.pad_token_id, np.int32)
  buffer[:, :prompt] = tokens
  rng, done = np.random.default_rng(seed), np.zeros(batch, bool)
  for position in range(prompt, prompt + new_tokens):
    out = np.asarray(scores(buffer, position - 1))
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
  parser.add_argument("--prompt", action="append", required=True,
                      help="The user's message, or @FILE to read it from a file. Repeat for "
                      "several, each answered on its own.")
  parser.add_argument("--image", action="append", default=[],
                      help="A picture the messages are about. Repeat for several.")
  parser.add_argument("--system", default="", help="A system message before the user's.")
  parser.add_argument("--max_new_tokens", type=int, default=256)
  parser.add_argument("--temperature", type=float, default=0.0,
                      help="0 takes the likeliest token; above 0, tokens are drawn.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                      help="The precision the model computes in; float32 takes twice the memory.")
  parser.add_argument("--params", default="",
                      help="Parameters that the trainer's --save_params wrote, in place of the "
                      "checkpoint's own.")
  parser.add_argument("--platform", default="", help="cpu to run without a GPU.")
  args = parser.parse_args(argv)
  if args.platform:
    os.environ["JAX_PLATFORMS"] = args.platform

  import jax  # pylint: disable=g-import-not-at-top
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top
  from jax_titan.models import command_a  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import checkpoint  # pylint: disable=g-import-not-at-top

  prompts = [pathlib.Path(p[1:]).read_text(encoding="utf-8").strip() if p.startswith("@") else p
             for p in args.prompt]
  images = [data.open_image(path) for path in args.image]
  preprocessor = data.Preprocessor.open(args.checkpoint)
  if args.params:
    # Before the weights load: a wrong path or another model's parameters fail at once.
    command_a.part_size(
        command_a.pretrained_config(args.checkpoint), checkpoint.saved_tree(args.params))
  start = time.time()
  dtype, devices = jax.numpy.dtype(args.dtype), jax.devices()
  cfg, params = command_a.load_stages(args.checkpoint, devices, dtype, bool(images))
  if args.params:
    trained = command_a.to_stages(cfg, checkpoint.restore_saved(args.params), devices, dtype)
    params = [{**p, **t, "layers": {**p["layers"], **t["layers"]}}
              for p, t in zip(params, trained)]
  loaded = f"loaded on {len(params)} devices after {time.time() - start:.0f} s; "
  for prompt in prompts:
    content = [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]
    messages = [{"role": "system", "content": args.system}] if args.system else []
    messages.append({"role": "user", "content": content if images else prompt})
    tokens, tiles = preprocessor.encode(messages, images)
    print(f"{loaded}the prompt is {len(tokens)} tokens"
          + (f" with {len(tiles)} image tiles" if images else ""))
    loaded, start = "", time.time()
    new = generate(cfg, params, tokens[None], None if tiles is None else tiles[None],
                   dtype=dtype, new_tokens=args.max_new_tokens, temperature=args.temperature,
                   seed=args.seed)[0]
    print(f"generated {len(new)} tokens in {time.time() - start:.0f} s")
    print(preprocessor.decode(new))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
