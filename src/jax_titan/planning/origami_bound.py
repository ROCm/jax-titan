"""An attainable time per hipBLASLt GEMM problem, from Origami's analytical model.

Origami (rocm-libraries `shared/origami`; `pip install shared/origami/python`)
predicts a GEMM's latency for one macro tile from its compute and memory
latencies, including tile and wave quantization over the CUs, and selects the
fastest candidate. `attainable` returns that prediction for the candidate
tiles TraceLens and tritonBLAS sweep: block M and N of 16 to 256, block K of
16 to 512, one wave per SIMD and a data-parallel grid. Unlike
`gemm_cost.t_bound_ms` it accounts for quantization, so it is a bound some
kernel shape can reach; a library kernel with a tile outside the candidates
can still be a few percent faster than it.

Origami is optional: this module imports it only when called. Its hardware
model needs the LDS, register file and L2 sizes, which `_HARDWARE` lists per
GPU as `origami.get_hardware_for_device` reports them on that GPU.
"""

from __future__ import annotations

import itertools

from jax_titan.planning import gpus as gpus_lib

# GPU name: (architecture, LDS bytes per CU, register file bytes per CU, L2 bytes per XCD).
_HARDWARE = {
    "AMD Instinct MI300X": ("gfx942", 64 * 1024, 512 * 1024, 4 * 1024 * 1024),
    "AMD Instinct MI325X": ("gfx942", 64 * 1024, 512 * 1024, 4 * 1024 * 1024),
}
_DTYPES = {"bf16_r": "BFloat16", "f16_r": "Half", "f32_r": "Float", "f8_r": "Float8", "bf8_r": "BFloat8",
           "f8_fnuz_r": "Float8_fnuz", "bf8_fnuz_r": "BFloat8_fnuz"}
_BLOCK_MN = (16, 32, 64, 128, 256)
_BLOCK_K = (16, 32, 64, 128, 256, 512)


def supported(gpu: gpus_lib.GPU) -> bool:
  return gpu.name in _HARDWARE


def hardware(gpu: gpus_lib.GPU, clock_ghz: float | None = None):
  """Origami's `hardware_t` for `gpu` at `clock_ghz`, by default its peak clock."""
  if not supported(gpu):
    raise ValueError(f"no Origami hardware constants for {gpu.name}; known: {', '.join(_HARDWARE)}")
  import origami  # pylint: disable=g-import-not-at-top
  arch, lds, rf, l2 = _HARDWARE[gpu.name]
  return origami.get_hardware_for_arch(getattr(origami.architecture_t, arch), gpu.cus, lds, rf, l2,
                                       round((clock_ghz or gpu.clock_ghz) * 1e6))


def problem(key: dict):
  """Origami's `problem_t` for a hipBLASLt problem key, whose matrix
  instructions run at the narrower of the two input types."""
  import origami  # pylint: disable=g-import-not-at-top
  types = {}
  for name in ("a", "b", "c"):
    if key[f"{name}_type"] not in _DTYPES:
      raise ValueError(f"no Origami data type for {key[f'{name}_type']}; known: {', '.join(_DTYPES)}")
    types[name] = getattr(origami.data_type_t, _DTYPES[key[f"{name}_type"]])
  p = origami.problem_t()
  p.size = origami.dim3_t(key["m"], key["n"], key["k"])
  p.batch = key["batch_count"]
  p.a_transpose = getattr(origami.transpose_t, key["transA"].upper())
  p.b_transpose = getattr(origami.transpose_t, key["transB"].upper())
  p.a_dtype, p.b_dtype = types["a"], types["b"]
  p.c_dtype = p.d_dtype = types["c"]
  p.mi_dtype = min(types["a"], types["b"], key=origami.datatype_to_bits)
  return p


def attainable(key: dict, gpu: gpus_lib.GPU, clock_ghz: float | None = None) -> dict:
  """Origami's fastest candidate for one launch of hipBLASLt problem `key` on
  `gpu` at `clock_ghz`: {"ms", "tile": (M, N, K)}. Raises ImportError without
  Origami and ValueError for a GPU or data type it has no constants for."""
  import origami  # pylint: disable=g-import-not-at-top
  hw = hardware(gpu, clock_ghz)
  p = problem(key)
  mi = hw.get_recommended_matrix_instruction(p.mi_dtype)
  configs = []
  for m, n, k in itertools.product(_BLOCK_MN, _BLOCK_MN, _BLOCK_K):
    config = origami.config_t()
    config.mt = origami.dim3_t(m, n, k)
    config.mi = mi
    config.occupancy = 1
    config.grid_selection = origami.grid_selection_t.data_parallel
    configs.append(config)
  result = origami.select_config(p, hw, configs, origami.model_t.gemm)
  mt = result.config.mt
  return {"ms": result.latency / (hw.compute_clock_ghz * 1e6), "tile": (mt.m, mt.n, mt.k)}
