#!/usr/bin/env bash
# Times every GEMM problem of a list on each backend observability/gemm_bench.py
# knows, one backend group per GPU in parallel, and merges the results into one
# table. Run inside the TheRock container after `source scripts/env.sh`, on idle
# GPUs:
#
#   scripts/compare_gemm_backends.sh PROBLEMS OUT_DIR
#
# PROBLEMS is `tuning/hipblaslt.py capture` output or a gemm_bench spec list.
# Writes OUT_DIR/backends.md and backends.json. Environment:
#   BACKEND_GROUPS  semicolon-separated backend groups, one GPU each (default below)
#   TRIALS          trials per backend (default 3); TRIALS_SLOW for XLA's Triton
#                   backends, whose exhaustive tiling search compiles slowly (default 2)
#   BASELINE        backend for per-problem and modeled speedups (default xla_hipblaslt)
#   OVERRIDE        hipBLASLt override file: without it xla_hipblaslt_tuned is dropped
#   JAX_TITAN_FLYDSL_LIB  library from `python -m jax_titan.ops.providers.flydsl build`
#                   covering the problems: without it the flydsl backends are dropped
#   JAX_AITER_ROOT  JAX-AITER checkout built with `make` and `make ja_mods_nomha`:
#                   without it jax_aiter is dropped
# Groups on different GPUs see slightly different clocks; compare within a group
# for differences under a few percent.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PROBLEMS=$(realpath "$1")
OUT=$(realpath -m "$2")
TRIALS=${TRIALS:-3}
TRIALS_SLOW=${TRIALS_SLOW:-2}
BASELINE=${BASELINE:-xla_hipblaslt}
GROUPS_DEFAULT="xla_hipblaslt,xla_hipblaslt_tuned;xla_autotune;xla_triton;xla_auto;triton_gemm;jax_aiter;flydsl,flydsl_kernel;ops:te,ops:gluon,ops:gluon_intra_mfma_on,ops:gluon_intra_mfma_off"
IFS=';' read -r -a GROUPS_IN <<< "${BACKEND_GROUPS:-${GROUPS_DEFAULT}}"
mkdir -p "${OUT}"
rm -f "${OUT}"/group*.json "${OUT}"/group*.log

groups=()
for g in "${GROUPS_IN[@]}"; do
  keep=()
  IFS=',' read -r -a members <<< "${g}"
  for b in "${members[@]}"; do
    case ${b} in
      xla_hipblaslt_tuned) [[ -n ${OVERRIDE:-} ]] || continue ;;
      flydsl*) [[ -n ${JAX_TITAN_FLYDSL_LIB:-} ]] || continue ;;
      jax_aiter) [[ -n ${JAX_AITER_ROOT:-} ]] || continue ;;
    esac
    keep+=("${b}")
  done
  if [[ ${#keep[@]} -gt 0 ]]; then groups+=("$(IFS=,; echo "${keep[*]}")"); fi
done

if [[ -n ${JAX_AITER_ROOT:-} ]]; then
  export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${JAX_AITER_ROOT}" JA_ROOT_DIR=${JAX_AITER_ROOT}
  export AITER_ASM_DIR=${AITER_ASM_DIR:-${JAX_AITER_ROOT}/third_party/aiter/hsa}
fi
export TRITON_DEFAULT_FP_FUSION=${TRITON_DEFAULT_FP_FUSION:-1}

for g in "${groups[@]}"; do
  if [[ ,${g}, == *,triton_gemm,* || ,${g}, == *,ops:triton,* || ${g} == *ops:gluon* ]]; then
    python -m jax_titan.ops.providers._triton_build
    break
  fi
done

pids=()
for i in "${!groups[@]}"; do
  g=${groups[$i]}
  trials=${TRIALS}
  [[ ${g} == *xla_triton* || ${g} == *xla_auto* ]] && trials=${TRIALS_SLOW}
  echo "GPU ${i}: ${g} (${trials} trials)"
  python -m jax_titan.observability.gemm_bench "${PROBLEMS}" --backends "${g}" --trials "${trials}" \
    --gpu "${i}" --baseline "${BASELINE}" ${OVERRIDE:+--override "${OVERRIDE}"} \
    --json "${OUT}/group${i}.json" \
    > "${OUT}/group${i}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
[[ ${rc} == 0 ]] || echo "a group failed; see ${OUT}/group*.log" >&2

python3 - "${OUT}" "${BASELINE}" <<'EOF'
import json, pathlib, sys
from jax_titan.observability import gemm_bench
out = pathlib.Path(sys.argv[1])
baseline = sys.argv[2]
runs = [json.loads(p.read_text()) for p in sorted(out.glob("group*.json"))]
backends = [b for r in runs for b in r["backends"]]
rows = [row for r in runs for row in r["rows"]]
gemm_bench.annotate_speedups(rows, baseline)
child_checks = [check for r in runs for check in r.get("scheduler_ab", [])]
if all(backend in backends for backend in gemm_bench._MFMA_BACKENDS):
    checks = gemm_bench.scheduler_ab_checks(rows)
    checked = {check["problem"] for check in checks}
    expected = {row["problem"] for row in rows}
    expected.update(check["problem"] for check in child_checks)
    for problem in sorted(expected - checked):
        checks.append({
            "problem": problem, "status": "fail",
            "reason": "missing both on and off benchmark rows",
            "on_llir_sha256": None, "off_llir_sha256": None,
            "on_barrier_calls": None, "off_barrier_calls": None,
        })
    if not checks:
        checks.append({
            "problem": "all requested problems", "status": "fail",
            "reason": "no on/off benchmark rows were produced",
            "on_llir_sha256": None, "off_llir_sha256": None,
            "on_barrier_calls": None, "off_barrier_calls": None,
        })
else:
    checks = child_checks
gemm_bench._gate_failed_scheduler_pairs(rows, checks)
report = {
    "backends": backends, "baseline": baseline, "rows": rows,
    "summaries": gemm_bench.summarize(rows, backends, baseline),
    "scheduler_ab": checks,
    "scheduler_ab_ok": all(check["status"] == "pass" for check in checks),
}
table = gemm_bench.table(report)
note = (
    "Per problem: median milliseconds, 95% trial interval, TFLOP/s, and speedup "
    f"against {baseline}. Overall: dispatch-weighted by launches_per_step."
)
(out / "backends.md").write_text(note + "\n\n" + table + "\n")
(out / "backends.json").write_text(json.dumps(report, indent=1) + "\n")
print(note)
print(table)
if not report["scheduler_ab_ok"]:
    raise SystemExit(2)
EOF
exit ${rc}
