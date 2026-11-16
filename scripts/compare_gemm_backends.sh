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
GROUPS_DEFAULT="xla_hipblaslt,xla_hipblaslt_tuned;xla_autotune;xla_triton;xla_auto;triton_gemm;jax_aiter;flydsl,flydsl_kernel;ops:te"
IFS=';' read -r -a GROUPS_IN <<< "${BACKEND_GROUPS:-${GROUPS_DEFAULT}}"
mkdir -p "${OUT}"

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

pids=()
for i in "${!groups[@]}"; do
  g=${groups[$i]}
  trials=${TRIALS}
  [[ ${g} == *xla_triton* || ${g} == *xla_auto* ]] && trials=${TRIALS_SLOW}
  echo "GPU ${i}: ${g} (${trials} trials)"
  python -m jax_titan.observability.gemm_bench "${PROBLEMS}" --backends "${g}" --trials "${trials}" \
    --gpu "${i}" ${OVERRIDE:+--override "${OVERRIDE}"} --json "${OUT}/group${i}.json" \
    > "${OUT}/group${i}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
[[ ${rc} == 0 ]] || echo "a group failed; see ${OUT}/group*.log" >&2

python3 - "${OUT}" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
runs = [json.loads(p.read_text()) for p in sorted(out.glob("group*.json"))]
backends = [b for r in runs for b in r["backends"]]
rows = [row for r in runs for row in r["rows"]]
by_problem = {}
for row in rows:
  by_problem.setdefault(row["problem"], {})[row["backend"]] = row
lines = ["| problem | " + " | ".join(backends) + " |", "|---|" + "---|" * len(backends)]
for pid, cells in by_problem.items():
  base = cells.get("xla_hipblaslt", {}).get("ms")
  text = []
  for b in backends:
    r = cells.get(b)
    if not r or "ms" not in r:
      text.append((r or {}).get("skipped", "failed")[:60] if r else "")
      continue
    rel = f" ({r['ms'] / base - 1:+.0%})" if base and b != "xla_hipblaslt" else ""
    moved = f", {r['transposes']}T" if r.get("transposes") else ""
    text.append(f"{r['ms']:.3f}{rel}{moved}")
  lines.append(f"| {pid} | " + " | ".join(text) + " |")
note = ("Milliseconds per GEMM, median over trials; in brackets, against xla_hipblaslt (hipBLASLt's "
        "first choice at level 0); nT: operands transposed in the timed program.")
(out / "backends.md").write_text(note + "\n\n" + "\n".join(lines) + "\n")
(out / "backends.json").write_text(json.dumps({"backends": backends, "rows": rows}, indent=1) + "\n")
print(note)
print("\n".join(lines))
EOF
exit ${rc}
