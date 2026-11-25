"""Decoder-only causal language models of Hugging Face transformers 5.23.0, as
one stage.

transformers supplies the model: its config, its modules, its initialization
and its published weights. JAX-Titan supplies the loss, the train step and the
layouts. The model does not run as PyTorch: torchax traces its forward into a
JAX function (`torch_bridge`), so only the families of `VERIFIED` are taken,
those whose loss and gradients through the bridge equal PyTorch's own.

A mixture of experts runs in the `dense` form registered here: every expert
on every token, weighted by the routing. The forms transformers ships gather
or group rows by expert, which this stack has no fast kernel for.
"""

import os
import re

import torch
import transformers
from transformers.integrations import moe
from transformers.models.auto import configuration_auto

from jax_titan.models import losses
from jax_titan.models.adapters import torch_bridge

PINNED_VERSION = "5.23.0"
# Model types whose loss and gradients through the bridge equal PyTorch's in
# fp32, at a small size, with rematerialization and over two traces
# (docs/adapters/transformers.md says how the list is made).
VERIFIED = (
    "afmoe", "apertus", "arcee", "aria_text", "axk1", "axk2", "bailing_hybrid", "bamba", "beam",
    "biogpt", "bitnet", "bloom", "codegen", "cohere", "cohere2", "cohere2_moe",
    "cohere_compass_text", "cwm", "deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4",
    "diffllama", "doge", "dots1", "dots3_note_text", "ernie4_5", "ernie4_5_moe", "exaone4",
    "exaone_moe", "falcon", "falcon_h1", "falcon_mamba", "flex_olmo", "flint", "gemma",
    "gemma2", "gemma3_text", "glm", "glm4", "glm4_moe", "glm4_moe_lite", "glm_moe_dsa",
    "gpt-sw3", "gpt2", "gpt_bigcode", "gpt_neo", "gpt_neox", "gpt_oss", "gptj", "granite",
    "granite_swa", "granitemoe", "granitemoe_swa", "granitemoehybrid", "granitemoeshared",
    "helium", "hunyuan_v1_dense", "hunyuan_v1_moe", "hy_v3", "hy_v4", "hyperclovax",
    "inkling2_text", "inkling_text", "internlm3", "iquestcoder", "jais2", "jamba",
    "kimi_linear", "kolibri1", "laguna", "lfm2", "lfm2_moe", "lfm3", "llama", "llama4_text",
    "mamba", "mamba2", "mellum", "mimo_v2_flash", "minicpm3", "minimax", "minimax_m2",
    "minimax_m3_vl_text", "ministral", "ministral3", "mistral", "mixtral", "modernbert-decoder",
    "moshi", "mpt", "nanochat", "nemotron", "nemotron_h", "olmo", "olmo2", "olmo3",
    "olmo_hybrid", "olmoe", "opt", "persimmon", "phi", "phi3", "qwen2", "qwen2_moe", "qwen3",
    "qwen3_5_moe_text", "qwen3_5_text", "qwen3_moe", "qwen3_next", "recurrent_gemma", "rwkv",
    "seed_oss", "smollm3", "solar_open", "stablelm", "starcoder2", "trocr", "vaultgemma",
    "xglm", "xlstm", "youtu", "zamba", "zamba2", "zaya",
)

_DROPOUT = re.compile(r"dropout|pdrop|drop_rate|layerdrop|jitter|drop_path")


def dense_experts_forward(self, hidden_states, top_k_index, top_k_weights):
  """An experts implementation for `ExpertsInterface`: tokens [n, hidden] with
  their routed experts [n, top_k] and routing weights [n, top_k] -> [n, hidden].

  Two matmuls of width experts x intermediate; the routing weights, zero for
  an expert a token is not routed to, select the result.
  """
  tokens, experts = hidden_states.size(0), self.num_experts
  # An expert-parallel sentinel (index == experts) lands in a column that is dropped.
  routing = torch.zeros(
      (tokens, experts + 1), dtype=top_k_weights.dtype, device=top_k_weights.device)
  routing = routing.scatter_add(1, top_k_index.clamp(max=experts), top_k_weights)[:, :experts]
  weight = self.gate_up_proj if self.has_gate else self.up_proj
  proj = torch.einsum("tk,gkn->tgn" if self.is_transposed else "tk,gnk->tgn",
                      hidden_states.to(weight.dtype), weight)
  if self.has_bias:
    proj = proj + (self.gate_up_proj_bias if self.has_gate else self.up_proj_bias)
  proj = self._apply_gate(proj) if self.has_gate else self.act_fn(proj)  # pylint: disable=protected-access
  if not self.has_bias and not self.has_post_expert_norm:
    # Weighted before the down projection, which then contracts experts and width at once.
    proj = proj * routing.unsqueeze(-1).to(proj.dtype)
    out = torch.einsum("tgn,gnk->tk" if self.is_transposed else "tgn,gkn->tk", proj, self.down_proj)
  else:
    out = torch.einsum(
        "tgn,gnk->tgk" if self.is_transposed else "tgn,gkn->tgk", proj, self.down_proj)
    if self.has_bias:
      out = out + self.down_proj_bias
    if self.has_post_expert_norm:
      out = self.post_expert_norm(out)
    out = (out * routing.unsqueeze(-1).to(out.dtype)).sum(dim=1)
  return out.to(hidden_states.dtype)


moe.ExpertsInterface.register("dense", dense_experts_forward)


def _configs(cfg) -> list:
  """`cfg` and its sub-configs."""
  subs = [v for v in vars(cfg).values() if isinstance(v, transformers.PretrainedConfig)]
  return [cfg] + [c for sub in subs for c in _configs(sub)]


def config(name: str, **overrides) -> transformers.PretrainedConfig:
  """The config of `name`: a directory with a `config.json`, a model type of
  transformers (`llama`, `qwen3_moe`) with the library's default sizes, or a
  model on the Hub. `overrides` win. Every dropout is set to zero."""
  if os.path.isdir(name) or name not in configuration_auto.CONFIG_MAPPING_NAMES:
    cfg = transformers.AutoConfig.from_pretrained(name, **overrides)
  else:
    cfg = transformers.AutoConfig.for_model(name, **overrides)
  for c in _configs(cfg):
    for key, value in vars(c).items():
      if isinstance(value, float) and _DROPOUT.search(key):
        setattr(c, key, 0.0)
    if hasattr(c, "use_cache"):
      c.use_cache = False
  return cfg


def _loss(forward, tokens, targets):
  return losses.cross_entropy(forward({"input_ids": tokens}), targets)


def causal_lm(cfg: transformers.PretrainedConfig, *, seed: int, dtype, remat: bool = True,
              init_from: str = "") -> torch_bridge.Bridged:
  """The causal language model of `cfg` as one stage, with the loss
  `losses.cross_entropy` of its logits.

  Its weights are the library's initialization under `seed`, or those of the
  directory `init_from` (`from_pretrained`), in fp32. `remat` rematerializes
  each layer through the library's own gradient-checkpointing switch.
  """
  torch.manual_seed(seed)
  if init_from:
    module = transformers.AutoModelForCausalLM.from_pretrained(
        init_from, config=cfg, dtype=torch.float32)
  else:
    module = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
  module.train()
  module.config._experts_implementation = "dense"  # pylint: disable=protected-access
  if remat:
    module._set_gradient_checkpointing(  # pylint: disable=protected-access
        enable=True, gradient_checkpointing_func=torch_bridge.remat)
  return torch_bridge.Bridged(module, dtype=dtype, loss=_loss)


def vocab_size(cfg: transformers.PretrainedConfig) -> int:
  return cfg.get_text_config(decoder=True).vocab_size
