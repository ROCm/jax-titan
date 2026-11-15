"""hipBLASLt tuning: log capture, the HLO-to-hipBLASLt key, override files, the
adoption rule, carrying winners across builds, and the GEMM cost breakdown."""

import json
import pathlib

import pytest

from jax_titan.planning import gemm_cost
from jax_titan.planning import gpus
from jax_titan.tuning import hipblaslt


def _bench(m, n, k, ta, tb, idx, ld=None):
  lda = (m if ta == "N" else k) if ld is None else ld
  ldb = k if tb == "N" else n
  return (f"hipblaslt-bench --api_method c -m {m} -n {n} -k {k} --lda {lda} --ldb {ldb} --ldc {m} "
          f"--ldd {m} --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 "
          f"--beta 0.000000 --transA {ta} --transB {tb} --batch_count 1 --a_type bf16_r --b_type bf16_r "
          f"--c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r "
          f"--algo_method index --solution_index {idx}")


# A Llama-like layer at 8,192 tokens: forward NN, activation-grad TN, weight-grad NT.
FWD = (14336, 8192, 4096, "N", "N")
DGRAD = (4096, 8192, 14336, "T", "N")
WGRAD = (14336, 4096, 8192, "N", "T")


@pytest.fixture
def log32(tmp_path):
  lines = ["[hipblaslt] some info line"]
  lines += [_bench(*FWD, 101)] * 3 + [_bench(*DGRAD, 202)] * 2 + [_bench(*WGRAD, 303)]
  lines += [_bench(*FWD, 101, ld=14336)]  # same key, same descriptor
  path = tmp_path / "hbl_1.log"
  path.write_text("\n".join(lines) + "\n")
  return path


def test_bench_line_parses_flags_and_normalizes_types():
  args = hipblaslt.parse_bench_line("2026-11-15 " + _bench(*WGRAD, 303))
  assert (args["m"], args["n"], args["k"], args["transA"], args["transB"]) == WGRAD
  assert args["solution_index"] == 303 and args["batch_count"] == 1 and args["alpha"] == 1.0
  assert hipblaslt.parse_bench_line("no bench line here") is None


def test_yaml_workload_line_uses_call_count_and_compute_type():
  line = ("- { function: matmul, M: 14336, N: 4096, K: 8192, lda: 14336, ldb: 4096, ldc: 14336, "
          "ldd: 14336, alpha: 1.000000, beta: 0.000000, transA: N, transB: T, batch_count: 1, "
          "a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, scale_type: f32_r, "
          "compute_type: c_f32_r, algo_method: 2, solution_index: 303, call_count: 64 }")
  args = hipblaslt.parse_yaml_line(line)
  assert args["compute_type"] == "f32_r" and args["call_count"] == 64
  assert hipblaslt.key_of(args) == hipblaslt.key_of(hipblaslt.parse_bench_line(_bench(*WGRAD, 303)))


def test_capture_counts_calls_and_solutions(log32):
  out = hipblaslt.capture([str(log32)])
  assert out["lines"] == 7 and len(out["problems"]) == 3
  fwd = next(p for p in out["problems"] if p["key"]["transA"] == "N" and p["key"]["transB"] == "N")
  assert fwd["calls"] == 4 and fwd["solutions"] == {"101": 4} and fwd["variants"] == 1
  assert fwd["id"] == "N,N,1,14336,8192,4096,bf16_r,bf16_r,bf16_r,f32_r"
  assert fwd["flops"] == 2 * 14336 * 8192 * 4096


def _site(m, n, k, layout, count=32, module="jit_step", kernel="custom-call.1"):
  return {"module": module, "kernel": kernel, "library": "hipBLASLt", "count": count, "batch": 1,
          "m": m, "n": n, "k": k, "layout": layout, "a_dtype": "bf16", "b_dtype": "bf16",
          "c_dtype": "bf16", "pass": "fwd", "source": "llama.py:1",
          "config": {"algorithm": 0, "workspace_bytes": 33554432}}


@pytest.mark.parametrize("site, expected", [
    # Row-major C[8192, 14336] = x[8192, 4096] w[4096, 14336]: forward, NN.
    (_site(8192, 14336, 4096, "MK,KN"), FWD),
    # dx[8192, 4096] = dy[8192, 14336] w[4096, 14336]^T: activation grad, TN.
    (_site(8192, 4096, 14336, "MK,NK"), DGRAD),
    # dw[4096, 14336] = x[8192, 4096]^T dy[8192, 14336]: weight grad, NT.
    (_site(4096, 14336, 8192, "KM,KN"), WGRAD),
])
def test_site_key_maps_row_major_sites_to_column_major_problems(site, expected):
  key = hipblaslt.site_key(site)
  assert key[:6] == (expected[3], expected[4], 1, expected[0], expected[1], expected[2])
  assert key[6:] == ("bf16_r", "bf16_r", "bf16_r")


def test_join_attaches_launches_and_workspace(log32):
  problems = hipblaslt.capture([str(log32)])
  gemms_json = {"modules": [{"gemms": [_site(8192, 14336, 4096, "MK,KN", 64),
                                       _site(8192, 4096, 14336, "MK,NK", 32, kernel="custom-call.2"),
                                       _site(1, 2, 3, "MK,KN", 1, kernel="custom-call.9")]}]}
  out = hipblaslt.join(problems, gemms_json)
  by_t = {(p["key"]["transA"], p["key"]["transB"]): p for p in out["problems"]}
  assert by_t[("N", "N")]["launches_per_step"] == 64
  assert by_t[("N", "N")]["workspace_bytes"] == 33554432
  assert by_t[("N", "T")]["launches_per_step"] is None
  assert [s["kernel"] for s in out["unmatched_sites"]] == ["custom-call.9"]
  assert out["problems_without_sites"] == [by_t[("N", "T")]["id"]]


def test_override_round_trip_and_positive_index(tmp_path):
  key = dict(zip(hipblaslt.KEY, ("N", "T", 1, 4096, 14336, 8192, "bf16_r", "bf16_r", "bf16_r", "f32_r")))
  path = tmp_path / "tuning.txt"
  assert hipblaslt.write_override([dict(key, solution_index=404)], "abc12345", path) == 1
  text = path.read_text().splitlines()
  assert text[0] == "Git Version: abc12345"
  assert text[1] == ",".join(hipblaslt.OVERRIDE_COLUMNS)
  git, rows = hipblaslt.parse_tuning_file(path)
  assert git == "abc12345" and hipblaslt.key_of(rows[0]) == hipblaslt.key_of(key)
  assert rows[0]["solution_index"] == 404
  with pytest.raises(ValueError):
    hipblaslt.write_override([dict(key, solution_index=0)], "abc12345", path)


def test_verify_reports_hits_misses_and_absent_rows(tmp_path, log32):
  rows = []
  for (m, n, k, ta, tb), idx in ((FWD, 101), (DGRAD, 999), ((1, 2, 3, "N", "N"), 5)):
    rows.append(dict(zip(hipblaslt.KEY, (ta, tb, 1, m, n, k, "bf16_r", "bf16_r", "bf16_r", "f32_r")),
                     solution_index=idx))
  path = tmp_path / "tuning.txt"
  hipblaslt.write_override(rows, "abc12345", path)
  out = hipblaslt.verify([str(log32)], path)
  assert (out["hit"], out["miss"], out["absent"]) == (1, 1, 1)
  assert not out["git_mismatch_logged"]
  log32.write_text(log32.read_text() + "The hipBLASLt git version and the override file git version "
                   "are not the same.\n")
  assert hipblaslt.verify([str(log32)], path)["git_mismatch_logged"]


def test_decide_needs_separated_intervals_and_the_minimum_gain():
  default = [1.00, 1.01, 0.99, 1.00, 1.02]
  assert hipblaslt.decide(default, [0.90, 0.91, 0.89, 0.90, 0.92])["adopt"]
  close = hipblaslt.decide(default, [0.99, 1.00, 0.98, 0.99, 1.01])
  assert not close["adopt"] and close["gain"] < 0.02
  noisy = hipblaslt.decide(default, [0.80, 1.10, 0.85, 1.05, 0.95])
  assert not noisy["separated"]


def test_bootstrap_interval_contains_the_median():
  values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02]
  lo, hi = hipblaslt.bootstrap_ci(values)
  assert lo <= sorted(values)[3] <= hi


def test_names_match_across_versions_that_add_tokens():
  old = "Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT160x512x32_MI16x16x1_SN_SK0_WGMXCC8"
  new = "Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT160x512x32_MI16x16x1_SN_BL1_WGMXCC1"
  other = "Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x512x32_MI16x16x1_SN_SK0_WGMXCC8"
  assert hipblaslt.name_distance(old, new) == 0
  assert hipblaslt.name_distance(other, new) > 0
  solutions = {7: {"solution": other, "kernel": other}, 9: {"solution": new + "_GSU1", "kernel": new}}
  assert hipblaslt.match({"solution": old, "kernel": old}, solutions) == {"index": 9, "how": "kernel tokens"}
  assert hipblaslt.match({"solution": other, "kernel": other}, solutions)["index"] == 7


def _results():
  key = dict(zip(hipblaslt.KEY, ("N", "N", 1, 14336, 8192, 4096, "bf16_r", "bf16_r", "bf16_r", "f32_r")))
  return {"results": [{"id": hipblaslt.key_text(hipblaslt.key_of(key)), "key": key, "launches_per_step": 32,
                       "winner": 11, "default_index": 10,
                       "stage2": {"default_ms": 1.50, "candidate_ms": 1.45,
                                  "samples_ms": {"-1": [1.50, 1.50], "11": [1.45, 1.45]}}}]}


def test_bound_sits_below_measured_times_and_scales_with_the_clock():
  key = _results()["results"][0]["key"]
  mi300x = gpus.get("MI300X")
  peak = gemm_cost.t_bound_ms(key, mi300x)
  capped = gemm_cost.t_bound_ms(key, mi300x, clock_ghz=1.3)
  assert peak == pytest.approx(2 * 14336 * 8192 * 4096 / 1307.4e12 * 1e3)
  assert capped == pytest.approx(peak * 2.1 / 1.3)
  assert capped < 1.45  # the measured best in _results


def test_sweep_times_a_problem_with_the_beta_its_calls_use():
  key = dict(zip(hipblaslt.KEY, ("N", "N", 1, 4096, 8192, 14336, "bf16_r", "bf16_r", "bf16_r", "f32_r")))
  residual = hipblaslt._sweep_args({"key": key, "desc": {"beta": 1.0}})
  assert residual[-2:] == ["--beta", "1"]
  assert "--beta" not in hipblaslt._sweep_args({"key": key, "desc": {"beta": 0.0}})


def _timed(site, us, ci=5.0):
  return dict(site, time={"launches_per_step": 32, "median_us": us, "ci_us": [us - ci, us + ci],
                          "clean": {"launches": 32, "median_us": us}, "overlapped": {"launches": 0}})


def test_instep_keeps_rows_that_are_not_slower_in_the_step(tmp_path):
  fwd, dgrad = _site(8192, 14336, 4096, "MK,KN"), _site(8192, 4096, 14336, "MK,NK", kernel="custom-call.2")
  rows = [dict(zip(hipblaslt.KEY, hipblaslt.site_key(s) + ("f32_r",)), solution_index=i)
          for s, i in ((fwd, 11), (dgrad, 22))]
  override = tmp_path / "tuning.txt"
  hipblaslt.write_override(rows, "abc123", override)
  base = {"modules": [{"gemms": [_timed(fwd, 1500.0), _timed(dgrad, 1600.0)]}]}
  tuned = {"modules": [{"gemms": [_timed(fwd, 1450.0), _timed(dgrad, 1680.0)]}]}
  out = hipblaslt.instep(base, tuned, override)
  assert [r["keep"] for r in out["rows"]] == [True, False]
  assert out["rows"][1]["delta"] == pytest.approx(0.05)
  assert out["rows"][0]["ms_per_step"] == pytest.approx(-32 * 50 / 1e3)
  assert [r["solution_index"] for r in out["kept"]] == [11]
  assert "| no |" in hipblaslt.instep_table(out)


def test_f32_problems_are_bound_at_the_fp32_peak_where_one_is_published():
  key = dict(_results()["results"][0]["key"], a_type="f32_r", b_type="f32_r", c_type="f32_r")
  bound = 2 * 14336 * 8192 * 4096 / 163.4e12 * 1e3
  assert gemm_cost.t_bound_ms(key, gpus.get("MI300X")) == pytest.approx(bound)
  # The capped clock is measured on bf16 GEMMs; f32 ones keep the peak clock.
  assert gemm_cost.t_bound_ms(key, gpus.get("MI300X"), clock_ghz=1.3) == pytest.approx(bound)
  assert gemm_cost.bound_precision(key, gpus.get("MI355X")) == "bf16"


def test_breakdown_splits_selection_contention_and_ceiling():
  results = _results()
  site = dict(_site(8192, 14336, 4096, "MK,KN"),
              time={"launches_per_step": 32, "median_us": 1700.0,
                    "clean": {"launches": 16, "median_us": 1550.0},
                    "overlapped": {"launches": 16, "median_us": 1850.0}})
  in_step = gemm_cost.in_step_times({"modules": [{"gemms": [site]}]})
  out = gemm_cost.breakdown(results, gpus.get("MI300X"), 1.3, in_step)
  row = out["rows"][0]
  assert row["selection_ms"] == pytest.approx(32 * 0.05)
  assert row["contention_ms"] == pytest.approx(32 * (1.70 - 1.55))
  assert row["other_ms"] == pytest.approx(32 * (1.55 - 1.50))
  assert row["ceiling_ms"] == pytest.approx(32 * (1.45 - row["t_bound_ms"]))
  assert out["totals_ms_per_step"]["now"] == pytest.approx(32 * 1.70)


def test_report_and_write_use_adopted_winners_only(tmp_path):
  results = _results()
  results["results"].append(dict(results["results"][0], id="x", winner=None))
  rows = hipblaslt.rows_from_results(results)
  assert [r["solution_index"] for r in rows] == [11]
  text = hipblaslt.report(results)
  assert "yes" in text and "Predicted GEMM time saved" in text


def test_command_line_capture_and_write(tmp_path, log32):
  problems = tmp_path / "problems.json"
  assert hipblaslt.main(["capture", str(log32), "--json", str(problems)]) == 0
  assert len(json.loads(problems.read_text())["problems"]) == 3
  results = tmp_path / "results.json"
  results.write_text(json.dumps(_results()))
  out = tmp_path / "tuning.txt"
  assert hipblaslt.main(["write", str(results), "--out", str(out), "--git_version", "abc12345"]) == 0
  assert pathlib.Path(out).read_text().startswith("Git Version: abc12345\n")
