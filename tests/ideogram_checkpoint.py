"""A checkpoint of Ideogram 4 in diffusers' layout at toy size, with random weights."""

import json

import jax
import ml_dtypes
import numpy as np
import pytest

from jax_titan.models import ideogram, pretrained, qwen3

WORDS = ("[UNK]", "<|im_start|>", "<|im_end|>", "user", "assistant", "a", "b", "c")
# The text features are the three tapped layers of the text encoder, and an image token is a
# 2x2 patch of the autoencoder's 32 channels, which Ideogram 4's token statistics are for.
TEXT = {
    "vocab_size": len(WORDS), "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32, "feature_layers": (0, 2, 3),
}
TRANSFORMER = {
    "attention_head_dim": 32, "num_attention_heads": 2, "in_channels": 128, "llm_features_dim": 48,
    "adaln_dim": 32, "intermediate_size": 128, "num_layers": 2, "mrope_section": (4, 3, 2),
}
VAE = {"block_out_channels": (8, 8, 8, 8), "layers_per_block": 1, "latent_channels": 32,
       "norm_num_groups": 4}


def published(params, fp8: bool = False) -> dict:
  """`params` under the names and in the layouts of the published transformer weights; with
  `fp8`, every kernel as e4m3 with one scale per output, as `ideogram-4-fp8` holds them."""
  layers = {k: np.asarray(v) for k, v in params["layers"].items()}
  tensors = {}

  def kernel(name, w):
    w = w.T
    if fp8:
      scale = (np.abs(w).max(axis=1) / 448).astype(np.float32)
      tensors[name + ".weight_scale"] = scale
      w = (w / scale[:, None]).astype(ml_dtypes.float8_e4m3fn)
    tensors[name + ".weight"] = w

  for i in range(len(layers["wq"])):
    block = f"layers.{i}."
    kernel(block + "attention.qkv", np.concatenate([layers[k][i] for k in ("wq", "wk", "wv")], 1))
    for name, key in (("attention.o", "wo"), ("feed_forward.w1", "w_gate"),
                      ("feed_forward.w3", "w_up"), ("feed_forward.w2", "w_down"),
                      ("adaln_modulation", "w_mod")):
      kernel(block + name, layers[key][i])
    for name, key in (("attention_norm1.weight", "attn_norm"),
                      ("attention.norm_q.weight", "q_norm"), ("attention.norm_k.weight", "k_norm"),
                      ("attention_norm2.weight", "attn_out_norm"),
                      ("ffn_norm1.weight", "mlp_norm"), ("ffn_norm2.weight", "mlp_out_norm"),
                      ("adaln_modulation.bias", "b_mod")):
      tensors[block + name] = layers[key][i]
  for name, w, b in (("input_proj", "w_in", "b_in"), ("llm_cond_proj", "w_text", "b_text"),
                     ("t_embedding.mlp_in", "w_time_in", "b_time_in"),
                     ("t_embedding.mlp_out", "w_time_out", "b_time_out"),
                     ("adaln_proj", "w_cond", "b_cond"),
                     ("final_layer.adaln_modulation", "w_final_mod", "b_final_mod"),
                     ("final_layer.linear", "w_out", "b_out")):
    kernel(name, np.asarray(params[w]))
    tensors[name + ".bias"] = np.asarray(params[b])
  tensors["llm_cond_norm.weight"] = np.asarray(params["text_norm"])
  tensors["embed_image_indicator.weight"] = np.asarray(params["image_embed"])
  return tensors


def _write(path, config: dict, tensors: dict):
  path.mkdir()
  pretrained.write(path / "model.safetensors", tensors)
  (path / "config.json").write_text(json.dumps(config))


def write_transformer(path, config: dict, params, fp8: bool = False):
  _write(path, config | {"_class_name": "Ideogram4Transformer2DModel"}, published(params, fp8))


def write_tokenizer(path):
  """A tokenizer of `WORDS`, split at spaces, with the two tokens of the chat format."""
  tokenizers = pytest.importorskip("tokenizers")
  tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
      {word: i for i, word in enumerate(WORDS)}, unk_token=WORDS[0]))
  tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
  tokenizer.add_special_tokens(list(WORDS[1:3]))
  path.mkdir()
  tokenizer.save(str(path / "tokenizer.json"))


def _text_tensors(params) -> dict:
  prefix = "model.language_model."
  names = {
      "attn_norm": "input_layernorm", "wq": "self_attn.q_proj", "wk": "self_attn.k_proj",
      "wv": "self_attn.v_proj", "q_norm": "self_attn.q_norm", "k_norm": "self_attn.k_norm",
      "wo": "self_attn.o_proj", "mlp_norm": "post_attention_layernorm", "w_gate": "mlp.gate_proj",
      "w_up": "mlp.up_proj", "w_down": "mlp.down_proj",
  }
  tensors = {prefix + "embed_tokens.weight": np.asarray(params["embed"])}
  for key, name in names.items():
    for i, w in enumerate(np.asarray(params["layers"][key])):
      tensors[f"{prefix}layers.{i}.{name}.weight"] = w.T
  return tensors


def _vae_tensors(rng) -> dict:
  channels, latent = VAE["block_out_channels"], VAE["latent_channels"]
  tensors = {}

  def conv(name, inputs, outputs, size=3):
    w = rng.normal(0, (inputs * size * size) ** -0.5, (outputs, inputs, size, size))
    tensors[name + ".weight"] = w.astype(np.float32)
    tensors[name + ".bias"] = rng.normal(0, 0.1, outputs).astype(np.float32)

  def norm(name, width):
    tensors[name + ".weight"] = rng.normal(1, 0.1, width).astype(np.float32)
    tensors[name + ".bias"] = rng.normal(0, 0.1, width).astype(np.float32)

  def resnet(name, inputs, outputs):
    norm(name + ".norm1", inputs)
    conv(name + ".conv1", inputs, outputs)
    norm(name + ".norm2", outputs)
    conv(name + ".conv2", outputs, outputs)
    if inputs != outputs:
      conv(name + ".conv_shortcut", inputs, outputs, 1)

  def mid(name, width):
    for i in range(2):
      resnet(f"{name}.resnets.{i}", width, width)
    norm(name + ".attentions.0.group_norm", width)
    for layer in ("to_q", "to_k", "to_v", "to_out.0"):
      w = rng.normal(0, width ** -0.5, (width, width))
      tensors[f"{name}.attentions.0.{layer}.weight"] = w.astype(np.float32)
      tensors[f"{name}.attentions.0.{layer}.bias"] = rng.normal(0, 0.1, width).astype(np.float32)

  conv("encoder.conv_in", 3, channels[0])
  width = channels[0]
  for i, outputs in enumerate(channels):
    for j in range(VAE["layers_per_block"]):
      resnet(f"encoder.down_blocks.{i}.resnets.{j}", width, outputs)
      width = outputs
    if i < len(channels) - 1:
      conv(f"encoder.down_blocks.{i}.downsamplers.0.conv", width, width)
  mid("encoder.mid_block", width)
  norm("encoder.conv_norm_out", width)
  conv("encoder.conv_out", width, 2 * latent)
  conv("quant_conv", 2 * latent, 2 * latent, 1)
  conv("post_quant_conv", latent, latent, 1)
  conv("decoder.conv_in", latent, width)
  mid("decoder.mid_block", width)
  for i, outputs in enumerate(reversed(channels)):
    for j in range(VAE["layers_per_block"] + 1):
      resnet(f"decoder.up_blocks.{i}.resnets.{j}", width, outputs)
      width = outputs
    if i < len(channels) - 1:
      conv(f"decoder.up_blocks.{i}.upsamplers.0.conv", width, width)
  norm("decoder.conv_norm_out", width)
  conv("decoder.conv_out", width, 3)
  return tensors


def write(path) -> None:
  """Writes the checkpoint to the new directory `path`: `tokenizer/`, `text_encoder/`,
  `transformer/`, `unconditional_transformer/` and `vae/`."""
  path.mkdir()
  write_tokenizer(path / "tokenizer")
  text = qwen3.init_params(qwen3.Qwen3Config(**TEXT), jax.random.key(0))
  _write(path / "text_encoder",
         {"text_config": TEXT | {"rope_parameters": {"rope_theta": 10000.0}}}, _text_tensors(text))
  cfg = ideogram.IdeogramConfig(**TRANSFORMER)
  for seed, name in enumerate(("transformer", "unconditional_transformer"), 1):
    write_transformer(path / name, TRANSFORMER, ideogram.init_params(cfg, jax.random.key(seed)))
  _write(path / "vae", VAE, _vae_tensors(np.random.default_rng(0)))
