"""A PyTorch module as one stage of the trainer, through torchax 0.0.13.

torchax runs a module's `forward` on JAX arrays, so tracing it gives a JAX
function: `jax.grad` differentiates it and XLA compiles it. PyTorch executes
while tracing only, and its autograd never runs. A module that leans on that
autograd is therefore wrong without an error: `torch.no_grad()`, `.data` and a
custom `autograd.Function` neither stop nor shape the gradient here. The
adapters of the libraries (`transformers`, `diffusers`) admit only what they
compared with PyTorch.

This is the only module that imports torchax. Importing torchax makes PyTorch
report an accelerator named `jax` for the rest of the process.
"""

from collections.abc import Callable
import contextlib

import jax
import numpy as np
import torch
import torchax
from torchax import interop

_NORMS = (torch.nn.modules.batchnorm._NormBase, torch.nn.GroupNorm, torch.nn.LayerNorm,  # pylint: disable=protected-access
          torch.nn.LocalResponseNorm, torch.nn.RMSNorm)


def remat(function: Callable, *args):
  """`function(*args)` rematerialized in the backward pass (`jax.checkpoint`).

  What the gradient-checkpointing switches of transformers and diffusers take
  in place of `torch.utils.checkpoint`, which does nothing through torchax.
  Only traced tensors pass through the checkpoint: a Python value or a
  constant tensor among `args` stays what it is for `function`.
  """
  traced = [i for i, a in enumerate(args)
            if isinstance(a, torchax.tensor.Tensor) and isinstance(a.jax(), jax.core.Tracer)]

  def on_traced(*tensors):
    full = list(args)
    for i, tensor in zip(traced, tensors):
      full[i] = tensor
    return function(*full)

  return interop.gradient_checkpoint(on_traced)(*[args[i] for i in traced])


@contextlib.contextmanager
def _restored(module: torch.nn.Module):
  """Leaves every submodule with the attributes and buffers it entered with.

  A forward may cache a tensor on its module (Phi-3's long RoPE factors); from
  a trace that is a tracer, which the next trace must not find.
  """
  saved = [(m, dict(m.__dict__), dict(m._buffers)) for m in module.modules()]  # pylint: disable=protected-access
  try:
    yield
  finally:
    for m, attributes, buffers in saved:
      m.__dict__.clear()
      m.__dict__.update(attributes)
      m._buffers.clear()  # pylint: disable=protected-access
      m._buffers.update(buffers)  # pylint: disable=protected-access


def is_norm(module: torch.nn.Module) -> bool:
  """Whether `module` is a normalization: PyTorch's own, or a library's by its
  class name (`LlamaRMSNorm`, `AdaLayerNormContinuous`)."""
  return isinstance(module, _NORMS) or "Norm" in type(module).__name__


class Bridged:
  """`module` as one stage: `init`, `abstract`, `stage_fns`, `logical_axes`, `decays`.

  The parameters are the module's own, under PyTorch's dotted names and as
  host arrays that share its memory; tied parameters are one. Buffers are
  constants of the stage function. The stage function casts the fp32
  parameters to `dtype` and returns `loss(forward, inputs, targets)`, where
  `forward(arrays, **constants)` calls the module with a dict of JAX arrays as
  keyword arguments and returns its first output. A forward that draws random
  numbers is refused: under `jax.jit` it would draw the same ones every step.
  Parameters that belong to a normalization module do not decay, and FSDP
  shards a parameter along its first dimension that the data axis divides.
  """

  num_stages = 1

  def __init__(self, module: torch.nn.Module, *, dtype, loss: Callable):
    self.module = module
    self.dtype = dtype
    self._loss = loss
    self._jittable = interop.JittableModule(module)
    self._env = torchax.default_env()
    self._buffers = self._env.t2j_copy(dict(self._jittable.buffers))
    self._no_decay = {
        name for name in self._jittable.params
        if is_norm(module.get_submodule(name.rpartition(".")[0]))
    }

  def init(self, seed: int = 0) -> tuple[dict]:
    """The module's parameters as they are: its own initialization, or the
    weights it was loaded with."""
    del seed  # The module was built under the seed.
    return ({k: v.detach().numpy() for k, v in self._jittable.params.items()},)

  def load(self, path: str) -> tuple[dict]:
    del path  # The module was built from the path.
    return self.init()

  def abstract(self, seed: int = 0) -> tuple[dict]:
    """ShapeDtypeStructs of `init(seed)`."""
    del seed
    return ({k: jax.ShapeDtypeStruct(tuple(v.shape), np.dtype(str(v.dtype).removeprefix("torch.")))
             for k, v in self._jittable.params.items()},)

  def _stage(self, params: dict, inputs, targets) -> jax.Array:
    env = self._env
    compute = {k: v.astype(self.dtype) for k, v in params.items()}

    def forward(arrays: dict, **constants):
      key = env.prng_key
      p, b, kwargs = env.j2t_iso((compute, self._buffers, arrays))
      with _restored(self.module), env:
        out = self._jittable.functional_call(
            "forward", p, b, return_dict=False, **kwargs, **constants)
      if env.prng_key is not key:
        raise NotImplementedError(
            f"{type(self.module).__name__} draws random numbers in its forward (dropout?), "
            "which a compiled step would repeat at every step")
      return env.t2j_iso(out[0])

    return self._loss(forward, inputs, targets)

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """The one stage function, `fn(params, inputs, targets)`, which returns the loss."""
    return [self._stage]

  @staticmethod
  def logical_axes(path, leaf) -> tuple:
    del path
    return (None,) * len(leaf.shape)

  def decays(self, path, leaf) -> bool:
    del leaf
    return path[-1].key not in self._no_decay
