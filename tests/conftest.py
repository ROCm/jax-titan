"""Tests run on 8 CPU devices, and the tests marked `gpu` on ROCm.

`pytest tests/` runs every test but those marked `gpu`. `pytest tests/ --gpus=N`
runs only those, with JAX on ROCm: a test marked `gpu` needs one GPU and one
marked `gpu(4)` four, and a test that needs more than N is skipped.
"""

import os

import pytest

# The ROCm runtime that jaxlib loads sets these with setenv(), outside
# os.environ. Subprocesses inherit them, force-load rocprofiler-sdk, and abort
# on a second LLVM ("Option 'bitcode-mdindex-threshold' registered more than
# once") when they import MaxText. Tests that start subprocesses after using
# jax themselves pass env=dict(os.environ), which never holds them.
_ROCPROFILER_VARS = ("ROCPROFILER_REGISTER_FORCE_LOAD", "ROCPROFILER_REGISTER_LIBRARY")


def pytest_addoption(parser):
  parser.addoption(
      "--gpus", type=int, default=0,
      help="Run only the tests marked gpu that need at most this many GPUs, on ROCm.")


def pytest_configure(config):
  gpus = config.getoption("--gpus")
  platform = "rocm" if gpus else "cpu"
  # Also for the processes that tests start. With rocm, JAX raises when it
  # finds no GPU instead of running on CPU.
  os.environ["JAX_PLATFORMS"] = platform
  if gpus:
    # hipBLASLt's first choice for every GEMM. By default XLA times its Triton
    # GEMMs and hipBLASLt's candidates for each GEMM it compiles, which takes
    # the tests from 90 seconds to 20 minutes.
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false")
  import jax  # pylint: disable=g-import-not-at-top

  jax.config.update("jax_platforms", platform)
  if not gpus:
    jax.config.update("jax_num_cpu_devices", 8)
  elif len(jax.devices()) < gpus:
    raise pytest.UsageError(f"--gpus={gpus}, but JAX sees {len(jax.devices())} GPUs")


def pytest_collection_modifyitems(config, items):
  gpus = config.getoption("--gpus")
  kept, dropped = [], []
  for item in items:
    marker = item.get_closest_marker("gpu")
    needs = (marker.args[0] if marker.args else 1) if marker else 0
    if bool(needs) != bool(gpus):
      dropped.append(item)
      continue
    if needs > gpus:
      item.add_marker(pytest.mark.skip(reason=f"needs {needs} GPUs, --gpus={gpus}"))
    kept.append(item)
  items[:] = kept
  config.hook.pytest_deselected(items=dropped)


@pytest.fixture(autouse=True)
def _no_inherited_rocprofiler():
  for name in _ROCPROFILER_VARS:
    os.unsetenv(name)
  yield
  for name in _ROCPROFILER_VARS:
    os.unsetenv(name)
