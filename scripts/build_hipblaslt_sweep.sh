#!/usr/bin/env bash
# Builds hipblaslt_sweep (src/jax_titan/tuning/hipblaslt_sweep.cpp) against the
# hipBLASLt of the container's ROCm, or of ROCM_ROOT for another build. Run
# inside the TheRock container.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
JAX_TITAN_ROOT=${JAX_TITAN_ROOT:-$(dirname "${REPO}")/.jax_titan}
ROCM_ROOT=${ROCM_ROOT:-$(rocm-sdk path --root)}
OUT=${OUT:-${JAX_TITAN_ROOT}/bin/hipblaslt_sweep}

mkdir -p "$(dirname "${OUT}")"
hipcc -O2 -std=c++17 -I"${ROCM_ROOT}/include" -L"${ROCM_ROOT}/lib" -Wl,-rpath,"${ROCM_ROOT}/lib" \
  "${REPO}/src/jax_titan/tuning/hipblaslt_sweep.cpp" -lhipblaslt -o "${OUT}"
echo "${OUT}"
