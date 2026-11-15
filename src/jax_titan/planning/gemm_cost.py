"""Where each GEMM's time goes: a bound, the best measured kernel, and the time in the step.

  python -m jax_titan.planning.gemm_cost RESULTS --gpu MI300X [--clock_ghz 1.3]
      [--sites GEMMS_JSON] [--json OUT]

RESULTS is `tuning/hipblaslt.py sweep`'s results.json: per hipBLASLt problem,
the default solution's and the best solution's isolated times. GEMMS_JSON is
`gemms.py --json --times` of a trace of the program: each site's in-step
launches, with the medians of launches no collective overlapped and of those
mostly overlapped. Four times are kept apart, per launch:

- t_bound: max(FLOPs at the peak for `--clock_ghz`, unique bytes at HBM
  bandwidth). A lower bound for any kernel; under a power cap pass the clock
  the GPU sustains on GEMMs, not the peak clock.
- t_default and t_best: the isolated default and the fastest solution measured.
- t_now: the median in-step launch, with its clean and overlapped medians.

Per step (launches times the difference), the GEMM time splits into:

- selection: t_default - t_best, what a tuning file can recover;
- contention: t_now - t_clean, what collectives running alongside cost;
- in-step other: t_clean - t_default, the rest of the gap between the step and
  an isolated run (clocks, caches, launch order); it can be negative;
- ceiling: t_best - t_bound, which no existing kernel recovers (speculative:
  the bound ignores tile and wave quantization).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

if __package__ in (None, ""):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from jax_titan.planning import gpus as gpus_lib  # noqa: E402
from jax_titan.tuning import hipblaslt  # noqa: E402

_BYTES = {"bf16_r": 2, "f16_r": 2, "f32_r": 4, "f8_r": 1, "bf8_r": 1, "f8_fnuz_r": 1, "bf8_fnuz_r": 1}


def t_bound_ms(key: dict, gpu: gpus_lib.GPU, clock_ghz: float | None = None) -> float:
  b, m, n, k = key["batch_count"], key["m"], key["n"], key["k"]
  flops = 2 * b * m * n * k
  peak = gpu.peak_tflops_at("bf16", clock_ghz) if clock_ghz else gpu.peak_tflops("bf16")
  bytes_ = b * (m * k * _BYTES.get(key["a_type"], 2) + k * n * _BYTES.get(key["b_type"], 2)
                + m * n * _BYTES.get(key["c_type"], 2))
  return max(flops / (peak * 1e12), bytes_ / (gpu.hbm_tbps * 1e12)) * 1e3


def in_step_times(gemms_json: dict) -> dict:
  """{problem id: {launches_per_step, now_us, clean_us, overlapped_us}} from timed sites."""
  out = {}
  for module in gemms_json.get("modules", []):
    for s in module.get("gemms", []):
      t = s.get("time")
      if s.get("library") != "hipBLASLt" or not t:
        continue
      pid = hipblaslt.key_text(hipblaslt.site_key(s) + ("f32_r",))
      rec = out.setdefault(pid, {"launches_per_step": 0.0, "now": [], "clean": [], "overlapped": []})
      rec["launches_per_step"] += t["launches_per_step"]
      w = t["launches_per_step"]
      rec["now"].append((t["median_us"], w))
      if t["clean"]["median_us"] is not None:
        rec["clean"].append((t["clean"]["median_us"], t["clean"]["launches"]))
      if t["overlapped"]["median_us"] is not None:
        rec["overlapped"].append((t["overlapped"]["median_us"], t["overlapped"]["launches"]))
  def weighted(pairs):
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None
  return {pid: {"launches_per_step": r["launches_per_step"], "now_us": weighted(r["now"]),
                "clean_us": weighted(r["clean"]), "overlapped_us": weighted(r["overlapped"])}
          for pid, r in out.items()}


def breakdown(results: dict, gpu: gpus_lib.GPU, clock_ghz: float | None = None,
              in_step: dict | None = None) -> dict:
  rows, totals = [], {"now": 0.0, "selection": 0.0, "contention": 0.0, "other": 0.0, "ceiling": 0.0}
  for r in results["results"]:
    s2 = r.get("stage2", {})
    if r.get("skipped") or "default_ms" not in s2:
      continue
    samples = {int(k): v for k, v in s2.get("samples_ms", {}).items()}
    best = min(statistics.median(v) for v in samples.values()) if samples else s2["candidate_ms"]
    t = in_step.get(r["id"], {}) if in_step else {}
    launches = t.get("launches_per_step") or r.get("launches_per_step") or 0
    row = {"id": r["id"], "launches_per_step": launches, "t_bound_ms": t_bound_ms(r["key"], gpu, clock_ghz),
           "t_default_ms": s2["default_ms"], "t_best_ms": best,
           "t_now_ms": t["now_us"] / 1e3 if t.get("now_us") else None,
           "t_clean_ms": t["clean_us"] / 1e3 if t.get("clean_us") else None,
           "t_overlapped_ms": t["overlapped_us"] / 1e3 if t.get("overlapped_us") else None,
           "winner": r.get("winner")}
    row["selection_ms"] = launches * (row["t_default_ms"] - best)
    row["ceiling_ms"] = launches * (best - row["t_bound_ms"])
    if row["t_now_ms"] is not None:
      clean = row["t_clean_ms"] if row["t_clean_ms"] is not None else row["t_now_ms"]
      row["now_ms"] = launches * row["t_now_ms"]
      row["contention_ms"] = launches * (row["t_now_ms"] - clean)
      row["other_ms"] = launches * (clean - row["t_default_ms"])
      for name in ("now", "contention", "other"):
        totals[name] += row[f"{name}_ms"]
    totals["selection"] += row["selection_ms"]
    totals["ceiling"] += row["ceiling_ms"]
    rows.append(row)
  rows.sort(key=lambda r: -(r.get("now_ms") or r["launches_per_step"] * r["t_default_ms"]))
  return {"gpu": gpu.name, "clock_ghz": clock_ghz, "rows": rows, "totals_ms_per_step": totals}


def _table(out: dict) -> str:
  head = ("| problem | launches/step | t_bound | t_best | t_default | t_now | t_clean | "
          "selection ms/step | contention ms/step | ceiling ms/step |\n|---|---|---|---|---|---|---|---|---|---|")
  fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ""  # noqa: E731
  lines = [head]
  for r in out["rows"]:
    lines.append(f"| {r['id']} | {r['launches_per_step']:.0f} | {fmt(r['t_bound_ms'])} | {fmt(r['t_best_ms'])} | "
                 f"{fmt(r['t_default_ms'])} | {fmt(r['t_now_ms'])} | {fmt(r['t_clean_ms'])} | "
                 f"{r['selection_ms']:.2f} | {r.get('contention_ms', 0):.2f} | {r['ceiling_ms']:.2f} |")
  t = out["totals_ms_per_step"]
  lines.append(f"\nPer step: GEMM time in the step {t['now']:.1f} ms; selection {t['selection']:.1f}, "
               f"contention {t['contention']:.1f}, in-step other {t['other']:.1f}, "
               f"ceiling {t['ceiling']:.1f} (bound at {out['clock_ghz'] or 'peak'} GHz)")
  return "\n".join(lines)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("results", help="hipblaslt.py sweep results.json")
  parser.add_argument("--gpu", default="MI300X")
  parser.add_argument("--clock_ghz", type=float, default=None,
                      help="Engine clock the GPU sustains on GEMMs; default: the peak clock.")
  parser.add_argument("--sites", default="", help="gemms.py --json output with --times.")
  parser.add_argument("--json", default="")
  args = parser.parse_args(argv)
  results = json.loads(pathlib.Path(args.results).read_text())
  in_step = in_step_times(json.loads(pathlib.Path(args.sites).read_text())) if args.sites else None
  out = breakdown(results, gpus_lib.get(args.gpu), args.clock_ghz, in_step)
  print(_table(out))
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
