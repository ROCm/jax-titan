"""diffusers' FLUX.1 transformer through torchax, against PyTorch's own forward and backward."""

import copy
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("torchax", "diffusers", "torch")),
    reason="needs the diffusers extra",
)

SMALL = {
    "patch_size": 1, "in_channels": 16, "num_layers": 2, "num_single_layers": 2,
    "attention_head_dim": 16, "num_attention_heads": 2, "joint_attention_dim": 32,
    "pooled_projection_dim": 32, "guidance_embeds": True, "axes_dims_rope": (4, 6, 6),
}


@pytest.mark.gpu
def test_bridged_flux_gives_pytorchs_loss_and_gradient():
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import torch  # pylint: disable=g-import-not-at-top
  from jax_titan.data import diffusion  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import diffusers as hfd  # pylint: disable=g-import-not-at-top

  model = hfd.denoiser(hfd.config("FluxTransformer2DModel", **SMALL), seed=0, dtype=jnp.float32)
  batches = diffusion.flux(
      0, (1, 2), image_side=4, text_tokens=8, in_channels=16, joint_attention_dim=32,
      pooled_projection_dim=32, guidance_embeds=True)
  x, target = batches.next_batch(0)
  # As the layout hands a step its batch: without the microbatch axis.
  x, target = {k: v[0] for k, v in x.items()}, target[0]

  eager = copy.deepcopy(model.module)
  eager.disable_gradient_checkpointing()
  _, names = hfd.FAMILIES["FluxTransformer2DModel"]
  pred = eager(**{names[k]: torch.from_numpy(np.array(v)) for k, v in x.items()})[0]
  loss = torch.mean((torch.from_numpy(target) - pred) ** 2)
  loss.backward()
  reference = {n: p.grad.numpy() for n, p in eager.named_parameters()}
  scale = max(np.abs(g).max() for g in reference.values())

  (params,) = model.init()
  (fn,) = model.stage_fns()
  for _ in range(2):
    value, grads = jax.jit(jax.value_and_grad(fn))(params, x, target)
    np.testing.assert_allclose(value, loss.item(), rtol=1e-5)
    for name, grad in grads.items():
      np.testing.assert_allclose(grad, reference[name], rtol=1e-3, atol=1e-4 * scale)
  # No weight decay for the norms of the queries and keys; the Linear of an adaptive norm decays.
  exempt = {n for n in params if not model.decays((jax.tree_util.DictKey(n),), None)}
  assert exempt == {n for n in params if ".attn.norm_" in n}


def _losses(tmp_path, *flags):
  """The losses of 4 steps of a small `diffusers:FluxTransformer2DModel` on every GPU."""
  log = tmp_path / f"{'-'.join(flags)}.jsonl"
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", "--model=diffusers:FluxTransformer2DModel",
       "--model_overrides=" + json.dumps({**SMALL, "dtype": "float32"}), "--resolution=64",
       "--text_tokens=8", "--num_stages=1", "--microbatches=1", "--microbatch_size=8",
       "--steps=4", "--warmup_steps=1", f"--log_file={log}", *flags],
      # os.environ, not the process environment: see conftest's ROCPROFILER note.
      capture_output=True, text=True, check=False, cwd=REPO, env=dict(os.environ),
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  return [r["loss"] for r in records.steps(records.read(log))]


@pytest.mark.gpu(2)
def test_fsdp_trains_as_dp_does(tmp_path):
  np.testing.assert_allclose(
      _losses(tmp_path, "--parallel=fsdp"), _losses(tmp_path, "--parallel=dp"), rtol=1e-4)


@pytest.mark.gpu
def test_refuses_what_was_not_compared_with_pytorch():
  import optax  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import adapters  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import train  # pylint: disable=g-import-not-at-top

  args = train.parse_args(["--model=diffusers:UNet2DConditionModel", "--num_stages=1",
                           "--parallel=dp"])
  with pytest.raises(NotImplementedError, match="not compared with PyTorch"):
    adapters.build(args, [None], lambda decays=None: optax.sgd(0.0))
