#!/usr/bin/env bash
# Runs NEXP MLPerf Training trials of CONFIG with jax_titan.mlperf.run, each
# logged to LOGDIR, and checks every log with MLCommons' compliance checker and,
# with ten trials or more, the set with the reference convergence points (RCP).
#
#   scripts/mlperf_run_and_time.sh CONFIG DATA_DIR LOGDIR [NEXP [runner flags ...]]
#
# Before each trial it drops the page cache, as the rules ask, when sudo allows
# it without a password, and tells the runner whether it did (MLLOG's
# cache_clear). A trial's log holds the trainer's output and its MLLOG lines;
# the runner draws each trial's seed. The checkers come with
# 'jax-titan[mlperf]'; without them the trials run unchecked. PYTHON names the
# interpreter (default: python).
set -euo pipefail

usage="usage: $0 CONFIG DATA_DIR LOGDIR [NEXP [runner flags ...]]"
config=${1:?$usage}
data=${2:?$usage}
logdir=${3:?$usage}
nexp=${4:-1}
shift $(( $# < 4 ? $# : 4 ))
python=${PYTHON:-python}

mkdir -p "$logdir"
stamp=$(date +%y%m%d%H%M%S)
check=1
if ! "$python" -c "import mlperf_logging" 2>/dev/null; then
  echo "no mlperf_logging: the trials run unchecked" >&2
  check=0
fi
failed=0
for i in $(seq 1 "$nexp"); do
  log=$logdir/${stamp}_$i.log
  clear=()
  if sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
    clear=(--cache_clear)
  else
    echo "could not drop the page cache without a sudo password: MLLOG says cache_clear false" >&2
  fi
  echo "trial $i of $nexp: $log"
  start=$(date +%s)
  set +e
  "$python" -m jax_titan.mlperf.run "$config" --data_dir "$data" "${clear[@]}" "$@" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  echo "trial $i exited $status after $(( $(date +%s) - start )) s"
  if [ "$status" != 0 ]; then
    failed=1
  fi
  if [ "$check" = 1 ]; then
    "$python" -m mlperf_logging.compliance_checker --usage training --ruleset 6.1.0 \
      --log_output "${log%.log}.compliance.txt" "$log" || failed=1
  fi
done
if [ "$check" = 1 ] && [ "$nexp" -ge 10 ]; then
  rcp=$logdir/rcp_$stamp
  mkdir -p "$rcp"
  for i in $(seq 1 "$nexp"); do
    cp "$logdir/${stamp}_$i.log" "$rcp/result_$((i - 1)).txt"
  done
  "$python" -m mlperf_logging.rcp_checker "$rcp" --rcp_usage training --rcp_version 6.1.0 \
    --log_output "$rcp.checker.txt" || failed=1
fi
exit "$failed"
