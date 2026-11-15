#!/usr/bin/env bash
# Tunes hipBLASLt kernel selection for one JAX-Titan training configuration and
# checks that it takes effect. Stages (STAGES, comma separated, default all):
#   clock     the engine clock and power a GPU sustains on a long bf16 GEMM
#   capture   3 steps with an HLO dump and HIPBLASLT_LOG_MASK=32: the step's
#             GEMM sites (gemms.py) and the hipBLASLt problems they call
#   sweep     tuning/hipblaslt.py sweep: screen every solution, then
#             interleaved trials of finalists with bootstrap intervals
#   write     the override file for the hipBLASLt this container loads
#   verify    2 steps with the override and HIPBLASLT_LOG_MASK=32: every row's
#             problem must run its tuned solution; the same through the API
#   ab        REPEATS interleaved runs of STEPS steps without and with it
#   profile   PROFILE_STEPS traced steps without and with it: per-site times,
#             clean and collective-overlapped (profile.py, gemms.py --times)
#   cost      planning/gemm_cost.py: selection, contention and ceiling per step
#   negative  checks that each way the override can fail is visible: a wrong
#             git version, a wrong key field, too much workspace, and an
#             executable compiled without the override, from the cache
# Run inside the TheRock container after `source scripts/env.sh`, on idle GPUs:
#
#   scripts/tune_gemms.sh OUT_DIR [trainer arguments...]
#
# XLA runs at --xla_gpu_autotune_level=0 with Triton GEMMs off: then XLA runs
# hipBLASLt's first heuristic choice, which the override replaces.
# Environment: GPUS (sweep GPUs, default all 8), STEPS (default 30), REPEATS
# (default 3), PROFILE_STEPS (default 3), PROFILE_GPU (default 0), CLOCK_GHZ
# (default: the clock stage), GPU_NAME (default MI300X), SWEEP_ARGS,
# XLA_FLAGS_BASE (default $JAX_TITAN_ROCM_XLA_FLAGS).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "$1")
shift
ARGS=("$@")
STAGES=${STAGES:-clock,capture,sweep,write,verify,ab,profile,cost,negative}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
STEPS=${STEPS:-30}
REPEATS=${REPEATS:-3}
PROFILE_STEPS=${PROFILE_STEPS:-3}
PROFILE_GPU=${PROFILE_GPU:-0}
GPU_NAME=${GPU_NAME:-MI300X}
SWEEP_BIN=${SWEEP_BIN:-${JAX_TITAN_ROOT}/bin/hipblaslt_sweep}
FLAGS="${XLA_FLAGS_BASE:-${JAX_TITAN_ROCM_XLA_FLAGS}} --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
HB="python3 ${REPO}/src/jax_titan/tuning/hipblaslt.py"
GEMMS="python3 ${REPO}/src/jax_titan/observability/gemms.py"
export JAX_ENABLE_COMPILATION_CACHE=${JAX_ENABLE_COMPILATION_CACHE:-false}
mkdir -p "${OUT}"

has() { [[ ",${STAGES}," == *",$1,"* ]]; }

# run_train OVERRIDE LOG [VAR=VALUE...] -- [trainer arguments...]
# An empty OVERRIDE unsets HIPBLASLT_TUNING_OVERRIDE_FILE. XLA_FLAGS_RUN
# replaces the default XLA flags of the run.
run_train() {
  local over=$1 log=$2
  shift 2
  (
    while [[ $# -gt 0 && $1 != -- ]]; do
      export "${1?}"
      shift
    done
    shift
    if [[ -n ${over} ]]; then
      export HIPBLASLT_TUNING_OVERRIDE_FILE=${over}
    else
      unset HIPBLASLT_TUNING_OVERRIDE_FILE
    fi
    export XLA_FLAGS="${XLA_FLAGS_RUN:-${FLAGS}}"
    python -m jax_titan.trainer.train "${ARGS[@]}" "$@" > "${log}" 2>&1
  )
}

if has clock; then
  ${HB} clock --sweep_bin "${SWEEP_BIN}" --gpu "${GPUS%%,*}" --json "${OUT}/clock.json"
fi
if [[ -z "${CLOCK_GHZ:-}" && -f "${OUT}/clock.json" ]]; then
  CLOCK_GHZ=$(python3 -c "import json; print(json.load(open('${OUT}/clock.json'))['clock_mhz_median'] / 1000)")
fi

if has capture; then
  rm -rf "${OUT}/capture"
  mkdir -p "${OUT}/capture"
  run_train "" "${OUT}/capture/train.log" HIPBLASLT_LOG_MASK=32 \
    "HIPBLASLT_LOG_FILE=${OUT}/capture/hipblaslt_%i.log" \
    "XLA_FLAGS_RUN=${FLAGS} --xla_dump_to=${OUT}/capture/dump --xla_dump_hlo_as_text" \
    -- --steps=3 --run_dir="${OUT}/capture/run"
  ${GEMMS} "${OUT}/capture/dump" --run_dir "${OUT}/capture/run" --merge --json "${OUT}/gemms.json" \
    > "${OUT}/gemms.txt"
  ${HB} capture "${OUT}/capture/hipblaslt_*.log" --json "${OUT}/problems.json" > "${OUT}/capture.txt"
  ${HB} join "${OUT}/problems.json" "${OUT}/gemms.json" --steps 3 >> "${OUT}/capture.txt"
  cat "${OUT}/capture.txt"
fi

if has sweep; then
  # shellcheck disable=SC2086
  ${HB} sweep "${OUT}/problems.json" --out "${OUT}/sweep" --sweep_bin "${SWEEP_BIN}" --gpus "${GPUS}" ${SWEEP_ARGS:-}
  ${HB} report "${OUT}/sweep/results.json" | tee "${OUT}/sweep/report.md"
fi

if has write; then
  ${HB} write "${OUT}/sweep/results.json" --out "${OUT}/tuning.txt"
fi

if has verify; then
  rm -rf "${OUT}/verify"
  mkdir -p "${OUT}/verify"
  run_train "${OUT}/tuning.txt" "${OUT}/verify/train.log" HIPBLASLT_LOG_MASK=32 \
    "HIPBLASLT_LOG_FILE=${OUT}/verify/hipblaslt_%i.log" -- --steps=2 --run_dir="${OUT}/verify/run"
  ${HB} verify "${OUT}/verify/hipblaslt_"*.log "${OUT}/verify/train.log" --override "${OUT}/tuning.txt" \
    --api --sweep_bin "${SWEEP_BIN}" --json "${OUT}/verify.json" > /dev/null
  python3 -c "
import json; v = json.load(open('${OUT}/verify.json'))
l, a = v['log'], v['api']
print(f\"verify: {l['hit']} of {l['rows']} rows hit in the step ({l['miss']} miss, {l['absent']} absent), \"
      f\"{a['hit']} of {a['rows']} through the API; git mismatch logged: {l['git_mismatch_logged']}\")"
fi

if has ab; then
  mkdir -p "${OUT}/ab"
  for r in $(seq 1 "${REPEATS}"); do
    for arm in base tuned; do
      rm -rf "${OUT}/ab/${arm}_${r}"
      over=""
      [[ ${arm} == tuned ]] && over="${OUT}/tuning.txt"
      run_train "${over}" "${OUT}/ab/${arm}_${r}.log" -- --steps="${STEPS}" --run_dir="${OUT}/ab/${arm}_${r}"
    done
  done
  python3 - "${OUT}/ab" <<'EOF'
import json, pathlib, statistics, sys
d = pathlib.Path(sys.argv[1])
runs = {}
for run in sorted(p for p in d.iterdir() if p.is_dir()):
  recs = [json.loads(l) for l in (run / "train.jsonl").read_text().splitlines() if l.strip()]
  times = [r["step_time_s"] for r in recs if "step_time_s" in r][3:]
  if times:
    runs.setdefault(run.name.rsplit("_", 1)[0], []).append(statistics.median(times))
out = {arm: {"runs_s": v, "median_s": statistics.median(v)} for arm, v in runs.items()}
if {"base", "tuned"} <= set(out):
  b, t = out["base"]["median_s"], out["tuned"]["median_s"]
  out["delta"] = {"tuned_vs_base": t / b - 1,
                  "separated": max(out["tuned"]["runs_s"]) < min(out["base"]["runs_s"])}
(d / "ab.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
EOF
fi

if has profile; then
  mkdir -p "${OUT}/profile"
  first=5
  for arm in base tuned; do
    rm -rf "${OUT}/profile/${arm}" "${OUT}/profile/${arm}_run"
    over=""
    [[ ${arm} == tuned ]] && over="${OUT}/tuning.txt"
    run_train "${over}" "${OUT}/profile/${arm}.log" -- --steps=$((first + PROFILE_STEPS + 1)) \
      --profile_steps="${first}:$((first + PROFILE_STEPS))" --profile_dir="${OUT}/profile/${arm}" \
      --run_dir="${OUT}/profile/${arm}_run"
    python -m jax_titan.observability.profile "${OUT}/profile/${arm}" --steps "${PROFILE_STEPS}" \
      --gpu "${PROFILE_GPU}" --kernel_times "${OUT}/profile/${arm}_kernels.json"
    ${GEMMS} "${OUT}/capture/dump" --run_dir "${OUT}/capture/run" --merge \
      --times "${OUT}/profile/${arm}_kernels.json" --json "${OUT}/profile/${arm}_gemms.json" \
      > "${OUT}/profile/${arm}_gemms.txt"
    head -30 "${OUT}/profile/${arm}_gemms.txt"
  done
fi

if has cost; then
  python3 "${REPO}/src/jax_titan/planning/gemm_cost.py" "${OUT}/sweep/results.json" --gpu "${GPU_NAME}" \
    ${CLOCK_GHZ:+--clock_ghz "${CLOCK_GHZ}"} --sites "${OUT}/profile/base_gemms.json" \
    --json "${OUT}/cost.json" | tee "${OUT}/cost.md"
fi

if has negative; then
  N="${OUT}/negative"
  rm -rf "${N}"
  mkdir -p "${N}"
  python3 - "${OUT}" "${N}" <<'EOF'
# Variants of the override file, each with one fault.
import json, pathlib, re, sys
out, n = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = (out / "tuning.txt").read_text().splitlines()
(n / "bad_git.txt").write_text("\n".join(["Git Version: deadbeef"] + lines[1:]) + "\n")
header, value = lines[1].split(","), lines[2].split(",")
value[header.index("k")] = str(int(value[header.index("k")]) + 64)
(n / "bad_key.txt").write_text("\n".join([lines[0], lines[1], ",".join(value)] + lines[3:]) + "\n")
# A solution of the first tuned problem that needs more workspace than XLA gives it.
results = json.loads((out / "sweep" / "results.json").read_text())
first = next(r for r in results["results"] if r.get("winner") is not None)
tag = re.sub(r"[^\w]+", "_", first["id"])
rows = [json.loads(l) for l in (out / "sweep" / f"{tag}.list.log").read_text().splitlines() if l.startswith("{")]
big = [r for r in rows if r.get("kind") == "solution" and not r["supported"]
       and r["workspace"] > first["workspace_bytes"]]
if big:
  row = ",".join(str(first["key"][c]) for c in header[:-1]) + f",{big[0]['index']}"
  (n / "big_workspace.txt").write_text("\n".join([lines[0], lines[1], row]) + "\n")
  (n / "big_workspace.json").write_text(json.dumps({"problem": first["id"], "index": big[0]["index"],
                                                    "needs": big[0]["workspace"],
                                                    "xla_gives": first["workspace_bytes"]}) + "\n")
EOF
  negative_run() {  # negative_run NAME OVERRIDE [VAR=VALUE...]
    local name=$1 over=$2
    shift 2
    mkdir -p "${N}/${name}"
    run_train "${over}" "${N}/${name}/train.log" HIPBLASLT_LOG_MASK=32 \
      "HIPBLASLT_LOG_FILE=${N}/${name}/hipblaslt_%i.log" "$@" -- --steps=2 --run_dir="${N}/${name}/run"
  }
  negative_run bad_git "${N}/bad_git.txt"
  negative_run bad_key "${N}/bad_key.txt"
  if [[ -f "${N}/big_workspace.txt" ]]; then
    negative_run big_workspace "${N}/big_workspace.txt"
  fi
  cache=(JAX_ENABLE_COMPILATION_CACHE=true "JAX_COMPILATION_CACHE_DIR=${N}/jax_cache"
         JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0)
  negative_run cache_fill "" "${cache[@]}"
  negative_run cache_hit "${OUT}/tuning.txt" "${cache[@]}"
  for name in bad_git bad_key big_workspace cache_hit; do
    [[ -d "${N}/${name}" ]] || continue
    over="${OUT}/tuning.txt"
    case ${name} in
      bad_key) over="${N}/bad_key.txt" ;;
      big_workspace) over="${N}/big_workspace.txt" ;;
    esac
    ${HB} verify "${N}/${name}/hipblaslt_"*.log "${N}/${name}/train.log" --override "${over}" \
      --json "${N}/${name}.json" > /dev/null
  done
  python3 - "${N}" <<'EOF'
import json, pathlib, sys
n = pathlib.Path(sys.argv[1])
expect = {"bad_git": lambda s: s["hit"] == 0, "bad_key": lambda s: s["absent"] >= 1,
          "big_workspace": lambda s: s["hit"] == 0, "cache_hit": lambda s: s["hit"] == s["rows"]}
summary = {}
for name, check in expect.items():
  p = n / f"{name}.json"
  if p.exists():
    v = json.loads(p.read_text())["log"]
    s = {k: v[k] for k in ("rows", "hit", "miss", "absent", "git_mismatch_logged")}
    s["as_expected"] = bool(check(s))
    summary[name] = s
cache_log = (n / "cache_hit" / "train.log")
if cache_log.exists():
  summary["cache_hit"]["log_mentions_cache"] = "cache" in cache_log.read_text().lower()
(n / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
EOF
fi
