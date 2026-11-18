"""A small Command A+ for tests: its config, parameters, and a checkpoint in the
published layout with a tokenizer, a chat template and image settings."""

import importlib.util
import json
import pathlib

import jax
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
# A token's id is its place here; the ids of `CONFIG` name the first six.
WORDS = ("<PAD>", "<UNK>", "<BOS_TOKEN>", "<|END_OF_TURN_TOKEN|>", "<|IMG_LINE_BREAK|>",
         "<|IMG_PATCH|>", "<|START_OF_IMG|>", "<|END_OF_IMG|>", "<|USER|>", "<|CHATBOT|>",
         "a", "b", "c", "d")
SPECIAL = {"bos_token": WORDS[2], "eos_token": WORDS[3], "pad_token": WORDS[0],
           "unk_token": WORDS[1], "img_line_break_token": WORDS[4], "image_token": WORDS[5],
           "boi_token": WORDS[6], "eoi_token": WORDS[7]}
TEMPLATE = """{{ bos_token }}
{%- for message in messages %}
{{ '<|USER|>' if message.role == 'user' else '<|CHATBOT|>' }}
{%- if message.content is string %} {{ message.content }}
{%- else %}
{%- for part in message.content %}
{%- if part.type == 'image' %}<|IMG_PATCH|>{% else %} {{ part.text }}{% endif %}
{%- endfor %}
{%- endif %} <|END_OF_TURN_TOKEN|>
{%- endfor %}
{%- if add_generation_prompt %}<|CHATBOT|>{% endif %}"""


def params(cfg=CONFIG, images=True, seed=0):
  """Parameters whose norm weights and biases are off their initial ones and zeros."""
  rng = np.random.default_rng(seed)

  def moved(path, w):
    name = path[-1].key
    if "norm" in name or name.startswith("b"):
      w = w + 0.2 * rng.standard_normal(w.shape).astype(np.float32)
    return w

  tree = command_a.init_params(cfg, jax.random.key(seed), images)
  return jax.tree_util.tree_map_with_path(moved, tree)


def write(path):
  """Writes the checkpoint of `params()` with everything `Preprocessor` reads, and
  returns the parameters."""
  tokenizers = pytest.importorskip("tokenizers")
  pytest.importorskip("jinja2")
  tree = params()
  command_a.save(path, CONFIG, tree)
  tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
      {word: i for i, word in enumerate(WORDS)}, unk_token=WORDS[1]))
  tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
  tokenizer.add_special_tokens(list(WORDS[:10]))
  tokenizer.save(str(path / "tokenizer.json"))
  (path / "tokenizer_config.json").write_text(json.dumps(SPECIAL), encoding="utf-8")
  (path / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
  size = CONFIG.vision_image_size
  (path / "preprocessor_config.json").write_text(json.dumps({
      "size": {"height": size, "width": size}, "patch_size": CONFIG.vision_patch_size,
      "downsample_factor": CONFIG.downsample_factor, "max_patches": 4}), encoding="utf-8")
  return tree


SEQ = 32


def batches(cfg=CONFIG):
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


def has_reference() -> bool:
  """Whether torch and a transformers with Command A+ are installed, without importing them."""
  spec = importlib.util.find_spec("transformers")
  return importlib.util.find_spec("torch") is not None and spec is not None and any(
      (pathlib.Path(root) / "models/cohere2_moe").is_dir()
      for root in spec.submodule_search_locations or ())


needs_reference = pytest.mark.skipif(
    not has_reference(), reason="needs torch and transformers with cohere2_moe")


def reference(path):
  """transformers' model of the checkpoint at `path`, on its exact attention path."""
  import transformers  # pylint: disable=g-import-not-at-top

  model = transformers.Cohere2VisionForConditionalGeneration.from_pretrained(path).eval()
  model.set_attn_implementation("eager")
  return model


def reference_inputs(tokens, pixels, valid):
  """A batch of `batches` as transformers' model takes it."""
  import torch  # pylint: disable=g-import-not-at-top

  inputs = {"input_ids": torch.from_numpy(tokens).long()}
  if pixels is not None:
    # transformers takes the batch's tiles in one list; the second sequence has one.
    inputs["pixel_values"] = torch.from_numpy(pixels.reshape(-1, *pixels.shape[2:])[:3])
  if valid is not None:
    inputs["attention_mask"] = torch.from_numpy(valid).long()
  return inputs
