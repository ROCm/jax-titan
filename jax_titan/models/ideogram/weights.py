"""Ideogram 4's published weights: their config and their tensors as the
model's parameters."""

import numpy as np

from jax_titan.models import weights as weights_lib
from jax_titan.models.ideogram.model import IdeogramConfig


def pretrained_config(path) -> IdeogramConfig:
  """The config of published weights, from the `config.json` beside them."""
  return weights_lib.config(IdeogramConfig, path)


def pretrained_params(path, cfg: IdeogramConfig) -> dict:
  """`init_params`-shaped fp32 parameters, on the host, from published weights.

  `path` is a safetensors file or a directory of them in the layout of
  Ideogram's own model, which the published checkpoints keep: one `qkv`
  weight per block, and the MLP's gate, up and down weights as `w1`, `w3`
  and `w2`.
  """
  tensors = weights_lib.read(path)
  n, e = cfg.num_layers, cfg.emb_dim

  def kernel(name):
    return weights_lib.weight(tensors, name + ".weight").T

  def vector(name):
    return np.asarray(tensors[name]).astype(np.float32)

  def stacked(fn, name):
    return np.stack([fn(f"layers.{i}.{name}") for i in range(n)])

  qkv = stacked(kernel, "attention.qkv")
  return {
      "layers": {
          "attn_norm": stacked(vector, "attention_norm1.weight"),
          "wq": qkv[..., :e],
          "wk": qkv[..., e:2 * e],
          "wv": qkv[..., 2 * e:],
          "q_norm": stacked(vector, "attention.norm_q.weight"),
          "k_norm": stacked(vector, "attention.norm_k.weight"),
          "wo": stacked(kernel, "attention.o"),
          "attn_out_norm": stacked(vector, "attention_norm2.weight"),
          "mlp_norm": stacked(vector, "ffn_norm1.weight"),
          "w_gate": stacked(kernel, "feed_forward.w1"),
          "w_up": stacked(kernel, "feed_forward.w3"),
          "w_down": stacked(kernel, "feed_forward.w2"),
          "mlp_out_norm": stacked(vector, "ffn_norm2.weight"),
          "w_mod": stacked(kernel, "adaln_modulation"),
          "b_mod": stacked(vector, "adaln_modulation.bias"),
      },
      "w_in": kernel("input_proj"),
      "b_in": vector("input_proj.bias"),
      "text_norm": vector("llm_cond_norm.weight"),
      "w_text": kernel("llm_cond_proj"),
      "b_text": vector("llm_cond_proj.bias"),
      "w_time_in": kernel("t_embedding.mlp_in"),
      "b_time_in": vector("t_embedding.mlp_in.bias"),
      "w_time_out": kernel("t_embedding.mlp_out"),
      "b_time_out": vector("t_embedding.mlp_out.bias"),
      "w_cond": kernel("adaln_proj"),
      "b_cond": vector("adaln_proj.bias"),
      "image_embed": vector("embed_image_indicator.weight"),
      "w_final_mod": kernel("final_layer.adaln_modulation"),
      "b_final_mod": vector("final_layer.adaln_modulation.bias"),
      "w_out": kernel("final_layer.linear"),
      "b_out": vector("final_layer.linear.bias"),
  }
