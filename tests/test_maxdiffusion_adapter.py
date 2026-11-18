"""MaxDiffusion FLUX.1 stages match MaxDiffusion and single-mesh `jax.jit`."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("maxdiffusion") is None, reason="needs MaxDiffusion"
)


@pytest.mark.parametrize("flags", [
    [],
    ["--blocks=3,1"],
    ["--num_stages=4", "--microbatches=8", "--schedule=GPIPE"],
    ["--flavor=flux-schnell"],
], ids=lambda f: " ".join(f) or "flux-dev")
def test_maxdiffusion_flux_parity(flags):
  result = subprocess.run(
      [sys.executable, str(REPO / "examples/maxdiffusion_flux_parity.py"), "--platform=cpu",
       *flags],
      capture_output=True, text=True, check=False,
  )
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  assert "match the transformer" in result.stdout
  assert "matches the single-mesh reference" in result.stdout


def test_flux_kernels_decay():
  """FLUX.1 keeps the Dense of its single blocks' adaptive norm under a module named `norm`."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxdiffusion as md  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import train  # pylint: disable=g-import-not-at-top

  flux = md.transformer(
      "flux-dev", in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=16,
      num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=16,
      axes_dims_rope=(4, 6, 6), dtype=jnp.float32, weights_dtype=jnp.float32)
  stage = md.Model(flux, num_stages=1).abstract()
  mask = train._decay_mask(stage)  # pylint: disable=protected-access
  paths = {tuple(getattr(k, "key", None) for k in path): decays
           for path, decays in jax.tree_util.tree_flatten_with_path(mask)[0]}
  assert any(path[-3:-1] == ("norm", "lin") for path in paths)
  assert {path[-1] for path, decays in paths.items() if not decays} <= {"scale"}
