"""The text tower of Qwen3-VL as a feature extractor: Ideogram 4's text encoder.

A decoder-only language model of Llama's family with normalized queries and
keys: RMSNorm before attention and before the SwiGLU MLP, grouped-query
attention, rotary embeddings. Ideogram 4 conditions on the hidden states
after a fixed set of its layers, concatenated per token (`features`). The
vision tower and the language-model head are not part of this module, and
for text alone the model's three-axis rotary embedding is the usual one.
"""

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from jax_titan import ops
from jax_titan.models import weights as weights_lib


@dataclasses.dataclass(frozen=True)
class Qwen3Config:
  """The fields of the `text_config` of the model's `text_encoder/config.json`."""

  vocab_size: int = 151936
  hidden_size: int = 4096
  num_hidden_layers: int = 36
  num_attention_heads: int = 32
  num_key_value_heads: int = 8
  head_dim: int = 128
  intermediate_size: int = 12288
  rms_norm_eps: float = 1e-6
  rope_theta: float = 5_000_000.0
  # The layers whose outputs Ideogram 4 concatenates into its text features.
  feature_layers: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

  @property
  def features_dim(self) -> int:
    return self.hidden_size * len(self.feature_layers)


def pretrained_config(path) -> Qwen3Config:
  """The config of published weights, from the `config.json` beside them."""

  def text_tower(values):
    values = values.get("text_config", values)
    theta = values.get("rope_parameters", {}).get("rope_theta", values.get("rope_theta"))
    return values if theta is None else values | {"rope_theta": float(theta)}

  return weights_lib.config(Qwen3Config, path, text_tower)


def init_params(cfg: Qwen3Config, key: jax.Array) -> dict:
  """Random fp32 parameters, the layers stacked along a layer axis."""
  n, e, f, d = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
  q, kv = cfg.num_attention_heads * d, cfg.num_key_value_heads * d
  keys = iter(jax.random.split(key, 8))

  def weight(*shape):
    return shape[-2] ** -0.5 * jax.random.normal(next(keys), shape, jnp.float32)

  return {
      "embed": jax.random.normal(next(keys), (cfg.vocab_size, e), jnp.float32),
      "layers": {
          "attn_norm": jnp.ones((n, e), jnp.float32),
          "wq": weight(n, e, q),
          "wk": weight(n, e, kv),
          "wv": weight(n, e, kv),
          "q_norm": jnp.ones((n, d), jnp.float32),
          "k_norm": jnp.ones((n, d), jnp.float32),
          "wo": weight(n, q, e),
          "mlp_norm": jnp.ones((n, e), jnp.float32),
          "w_gate": weight(n, e, f),
          "w_up": weight(n, e, f),
          "w_down": weight(n, f, e),
      },
  }


def params_from_tensors(tensors: dict, cfg: Qwen3Config) -> dict:
  """`init_params`-shaped fp32 parameters from tensors named as transformers
  names them, under whatever prefix `language_model` has."""
  suffix = "language_model.embed_tokens.weight"
  prefix = next(k for k in tensors if k.endswith(suffix))[:-len("embed_tokens.weight")]

  def kernel(name):
    return weights_lib.weight(tensors, prefix + name + ".weight").T

  def vector(name):
    return np.asarray(tensors[prefix + name + ".weight"]).astype(np.float32)

  def stacked(fn, name):
    return np.stack([fn(f"layers.{i}.{name}") for i in range(cfg.num_hidden_layers)])

  return {
      "embed": vector("embed_tokens"),
      "layers": {
          "attn_norm": stacked(vector, "input_layernorm"),
          "wq": stacked(kernel, "self_attn.q_proj"),
          "wk": stacked(kernel, "self_attn.k_proj"),
          "wv": stacked(kernel, "self_attn.v_proj"),
          "q_norm": stacked(vector, "self_attn.q_norm"),
          "k_norm": stacked(vector, "self_attn.k_norm"),
          "wo": stacked(kernel, "self_attn.o_proj"),
          "mlp_norm": stacked(vector, "post_attention_layernorm"),
          "w_gate": stacked(kernel, "mlp.gate_proj"),
          "w_up": stacked(kernel, "mlp.up_proj"),
          "w_down": stacked(kernel, "mlp.down_proj"),
      },
  }


def load(path, dtype=jnp.bfloat16) -> tuple[Qwen3Config, dict]:
  """The config and the parameters in `dtype`, on the default device, of the
  published weights at `path`, a safetensors file or a directory of them."""
  cfg = pretrained_config(path)
  return cfg, weights_lib.to_device(params_from_tensors(weights_lib.read(path), cfg), dtype)


def tokenize(tokenizer_dir, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
  """Token ids and the mask of real tokens, both [prompts, tokens], of prompts in
  the chat format Ideogram 4 gives them: one user turn and the opening of the
  assistant's. Shorter prompts are padded on the left."""
  from tokenizers import Tokenizer  # pylint: disable=g-import-not-at-top

  tokenizer = Tokenizer.from_file(str(pathlib.Path(tokenizer_dir) / "tokenizer.json"))
  ids = [tokenizer.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
                          add_special_tokens=False).ids for p in prompts]
  tokens = max(len(i) for i in ids)
  token_ids = np.zeros((len(ids), tokens), np.int32)
  valid = np.zeros((len(ids), tokens), bool)
  for row, i in enumerate(ids):
    token_ids[row, tokens - len(i):] = i
    valid[row, tokens - len(i):] = True
  return token_ids, valid


def _rope_tables(cfg: Qwen3Config, positions: jax.Array) -> tuple[jax.Array, jax.Array]:
  """cos and sin of the rotary angles, [batch, tokens, head_dim // 2] in fp32."""
  d = cfg.head_dim
  exponent = np.arange(0, d, 2, dtype=np.float32) / np.float32(d)
  power = np.power(np.float64(cfg.rope_theta), exponent.astype(np.float64)).astype(np.float32)
  angles = positions[..., None].astype(jnp.float32) * (np.float32(1.0) / power)
  return jnp.cos(angles), jnp.sin(angles)


def _layer(cfg: Qwen3Config, h: jax.Array, p: dict, mask, cos, sin) -> jax.Array:
  b, s, _ = h.shape
  d, eps = cfg.head_dim, cfg.rms_norm_eps
  x = ops.rms_norm(h, p["attn_norm"], eps)
  q = ops.linear(x, p["wq"]).reshape(b, s, cfg.num_attention_heads, d)
  k = ops.linear(x, p["wk"]).reshape(b, s, cfg.num_key_value_heads, d)
  v = ops.linear(x, p["wv"]).reshape(b, s, cfg.num_key_value_heads, d)
  q, k = ops.rms_norm(q, p["q_norm"], eps), ops.rms_norm(k, p["k_norm"], eps)
  q, k = ops.rope(q, cos, sin), ops.rope(k, cos, sin)
  attn = ops.attention(q, k, v, causal=True, mask=mask).reshape(b, s, -1)
  h = h + ops.linear(attn, p["wo"])
  x = ops.rms_norm(h, p["mlp_norm"], eps)
  return h + ops.swiglu(x, p["w_gate"], p["w_up"], p["w_down"])


def features(cfg: Qwen3Config, params: dict, token_ids: jax.Array, valid: jax.Array,
             dtype=jnp.bfloat16) -> jax.Array:
  """Ideogram 4's text features [batch, tokens, features_dim] in fp32.

  `valid` [batch, tokens] marks the real tokens, which follow any padding.
  Token by token, the features are the outputs of `feature_layers` side by
  side for each hidden unit; they are zero at padding.
  """
  b, s = token_ids.shape
  h = params["embed"][token_ids].astype(dtype)
  positions = jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0)
  cos, sin = _rope_tables(cfg, positions)
  # A real token attends to the real tokens up to itself; padding, to itself.
  mask = (valid[:, None, :] | jnp.eye(s, dtype=bool))[:, None]

  def body(h, p):
    h = _layer(cfg, h, p, mask, cos, sin)
    return h, h

  layers = jax.tree.map(lambda w: w.astype(dtype), params["layers"])
  _, outputs = jax.lax.scan(body, h, layers)
  taps = outputs[np.asarray(cfg.feature_layers)]
  out = jnp.moveaxis(taps, 0, -1).reshape(b, s, cfg.features_dim).astype(jnp.float32)
  return out * valid[..., None]
