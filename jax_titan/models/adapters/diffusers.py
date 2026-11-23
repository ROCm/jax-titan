"""Denoisers of Hugging Face diffusers 0.43.0, as one stage.

diffusers supplies the denoiser: its config, its modules and its
initialization. JAX-Titan supplies the objective (`jax_titan.data.diffusion`),
the train step and the layouts. The denoiser does not run as PyTorch: torchax
traces its forward into a JAX function (`torch_bridge`), so only the classes
of `FAMILIES` are taken, those compared with PyTorch, each with the names its
forward gives the inputs of its family's batches.
"""

import os

import diffusers
import jax.numpy as jnp
import torch

from jax_titan.models.adapters import torch_bridge

PINNED_VERSION = "0.43.0"
# Denoiser class -> the argument of its forward for each input of its family's
# batches in `jax_titan.data.diffusion`.
FAMILIES = {
    "FluxTransformer2DModel": {
        "img": "hidden_states", "txt": "encoder_hidden_states", "pooled": "pooled_projections",
        "timestep": "timestep", "img_ids": "img_ids", "txt_ids": "txt_ids", "guidance": "guidance",
    },
}


def config(name: str, **overrides) -> dict:
  """The config of a denoiser: a class of diffusers (`FluxTransformer2DModel`,
  with the library's default sizes), a directory with its `config.json`, or a
  pipeline, a directory or a repository on the Hub, whose `transformer` it
  is. `overrides` win."""
  if hasattr(diffusers, name):
    return {"_class_name": name, **overrides}
  subfolder = None if os.path.isfile(os.path.join(name, "config.json")) else "transformer"
  return {**diffusers.ModelMixin.load_config(name, subfolder=subfolder), **overrides}


def denoiser(cfg: dict, *, seed: int, dtype, remat: bool = True) -> torch_bridge.Bridged:
  """The denoiser of `cfg` as one stage, with its family's loss: the mean
  squared error of its output to the batch's target.

  Its weights are the library's initialization under `seed`. `remat`
  rematerializes each block through the library's own gradient-checkpointing
  switch. Position ids keep their dtype; every other input is cast to `dtype`.
  """
  names = FAMILIES[cfg["_class_name"]]
  torch.manual_seed(seed)
  module = getattr(diffusers, cfg["_class_name"]).from_config(cfg)
  module.train()
  if remat:
    module.enable_gradient_checkpointing(torch_bridge.remat)

  def loss(forward, x, target):
    pred = forward({names[k]: v if k.endswith("_ids") else v.astype(dtype) for k, v in x.items()})
    return jnp.mean((target.astype(jnp.float32) - pred.astype(jnp.float32)) ** 2)

  return torch_bridge.Bridged(module, dtype=dtype, loss=loss)
