"""Origami's attainable GEMM time (planning/origami_bound.py) and gemm_cost's ceiling above it."""

import sys
import types

import pytest

from jax_titan.planning import gemm_cost
from jax_titan.planning import gpus
from jax_titan.planning import origami_bound
from jax_titan.tuning import hipblaslt

_KEY = dict(zip(hipblaslt.KEY, ("T", "N", 1, 14336, 8192, 4096, "bf16_r", "bf16_r", "bf16_r", "f32_r")))


class _Dim3:

  def __init__(self, m=0, n=0, k=0):
    self.m, self.n, self.k = m, n, k


def _fake_origami(calls: dict) -> types.ModuleType:
  """Origami's bindings as far as origami_bound uses them; the largest tile wins at 2.6e6 cycles."""
  o = types.ModuleType("origami")
  o.architecture_t = types.SimpleNamespace(gfx942="gfx942")
  o.data_type_t = types.SimpleNamespace(BFloat16="bf16", Half="f16", Float="f32", Float8_fnuz="f8")
  o.transpose_t = types.SimpleNamespace(N="N", T="T")
  o.grid_selection_t = types.SimpleNamespace(data_parallel="data_parallel")
  o.model_t = types.SimpleNamespace(gemm="gemm")
  o.dim3_t = _Dim3
  o.problem_t = o.config_t = types.SimpleNamespace
  o.datatype_to_bits = {"bf16": 16, "f16": 16, "f32": 32, "f8": 8}.get

  def get_hardware_for_arch(*args):
    calls["hardware"] = args
    return types.SimpleNamespace(compute_clock_ghz=args[-1] / 1e6,
                                 get_recommended_matrix_instruction=lambda dtype: ("mi", dtype))

  def select_config(problem, hardware, configs, model):
    calls.update(problem=problem, configs=configs, model=model)
    best = max(configs, key=lambda c: c.mt.m * c.mt.n * c.mt.k)
    return types.SimpleNamespace(latency=2.6e6, config=best)

  o.get_hardware_for_arch, o.select_config = get_hardware_for_arch, select_config
  return o


def _results(*keys):
  rows = []
  for key in keys:
    rows.append({"id": hipblaslt.key_text(hipblaslt.key_of(key)), "key": key, "launches_per_step": 32,
                 "winner": 11, "default_index": 10,
                 "stage2": {"default_ms": 1.50, "candidate_ms": 1.45,
                            "samples_ms": {"-1": [1.50, 1.50], "11": [1.45, 1.45]}}})
  return {"results": rows}


def test_attainable_hands_origami_the_problem_the_gpu_at_its_clock_and_the_candidate_tiles(monkeypatch):
  calls = {}
  monkeypatch.setitem(sys.modules, "origami", _fake_origami(calls))
  out = origami_bound.attainable(_KEY, gpus.get("MI300X"), clock_ghz=1.3)
  assert calls["hardware"] == ("gfx942", 304, 64 * 1024, 512 * 1024, 4 * 1024 * 1024, 1300000)
  p = calls["problem"]
  assert (p.size.m, p.size.n, p.size.k, p.batch) == (14336, 8192, 4096, 1)
  assert (p.a_transpose, p.b_transpose, p.mi_dtype, p.d_dtype) == ("T", "N", "bf16", "bf16")
  assert len(calls["configs"]) == 5 * 5 * 6
  assert {c.mi for c in calls["configs"]} == {("mi", "bf16")}
  assert {(c.occupancy, c.grid_selection) for c in calls["configs"]} == {(1, "data_parallel")}
  assert calls["model"] == "gemm"
  assert out["ms"] == pytest.approx(2.0)  # 2.6e6 cycles at 1.3 GHz
  assert out["tile"] == (256, 256, 512)


def test_matrix_instructions_run_at_the_narrower_input_type_and_the_peak_clock_by_default(monkeypatch):
  calls = {}
  monkeypatch.setitem(sys.modules, "origami", _fake_origami(calls))
  origami_bound.attainable(dict(_KEY, a_type="f8_fnuz_r"), gpus.get("MI325X"))
  assert calls["problem"].mi_dtype == "f8"
  assert calls["hardware"][-1] == 2100000


def test_gpus_and_types_without_origami_constants_are_refused(monkeypatch):
  with pytest.raises(ValueError, match="no Origami hardware constants"):
    origami_bound.hardware(gpus.get("H100"))
  monkeypatch.setitem(sys.modules, "origami", _fake_origami({}))
  with pytest.raises(ValueError, match="no Origami data type"):
    origami_bound.problem(dict(_KEY, a_type="i8_r"))


def test_origami_ceiling_is_the_best_kernel_above_the_attainable_time(monkeypatch):
  clocks = []
  def attainable(key, gpu, clock_ghz):
    clocks.append(clock_ghz)
    return {"ms": 1.40, "tile": (256, 256, 64)}
  monkeypatch.setattr(origami_bound, "attainable", attainable)
  f32 = dict(_KEY, a_type="f32_r", b_type="f32_r", c_type="f32_r")
  out = gemm_cost.breakdown(_results(_KEY, f32), gpus.get("MI300X"), 1.3, bound="origami")
  # As for t_bound, the capped clock applies to bf16 problems only.
  assert clocks == [1.3, None]
  row = out["rows"][0]
  assert row["ceiling_ms"] == pytest.approx(32 * (1.45 - 1.40))
  assert row["t_bound_ms"] == pytest.approx(gemm_cost.t_bound_ms(_KEY, gpus.get("MI300X"), 1.3))
  assert out["bound"] == "origami"
  assert "| 1.4000 | 256x256x64 |" in gemm_cost._table(out)
  with pytest.raises(ValueError, match="unknown bound"):
    gemm_cost.breakdown(_results(_KEY), gpus.get("MI300X"), bound="tiles")


def test_origami_time_sits_above_the_roofline_and_grows_at_a_lower_clock():
  pytest.importorskip("origami")
  mi300x = gpus.get("MI300X")
  peak = origami_bound.attainable(_KEY, mi300x)["ms"]
  capped = origami_bound.attainable(_KEY, mi300x, clock_ghz=1.3)["ms"]
  assert gemm_cost.t_bound_ms(_KEY, mi300x) < peak < capped
  assert capped < 2 * gemm_cost.t_bound_ms(_KEY, mi300x, clock_ghz=1.3)
