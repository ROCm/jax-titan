#!/usr/bin/env bash
# Installs the exact Triton compiler used by gfx950-gluon-tutorials into the
# retained JAX-Titan environment. PyPI Triton 3.9.0 accepts schedule_hint but
# does not run the tutorial's MFMASchedule pass.
set -euo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WORKSPACE=$(dirname "${REPO}")
JAX_TITAN_ROOT=${JAX_TITAN_ROOT:-${WORKSPACE}/.jax_titan}
TRITON_SRC=${TRITON_SRC:-${JAX_TITAN_ROOT}/src/triton-gfx950}
TRITON_TAG=gfx950-tutorial-v3.0
TRITON_REV=3b0c7f081a3fc8471cae4941d1c579f2348bcbca
PYTHON=${PYTHON:-${JAX_TITAN_ROOT}/venv/bin/python}
MAX_JOBS=${MAX_JOBS:-16}

[[ -x ${PYTHON} ]] || {
  echo "Python environment not found at ${PYTHON}; complete docs/getting_started.md first." >&2
  exit 1
}
mkdir -p "$(dirname "${TRITON_SRC}")"
if [[ ! -d ${TRITON_SRC}/.git ]]; then
  git clone --filter=blob:none --branch "${TRITON_TAG}" \
    https://github.com/triton-lang/triton.git "${TRITON_SRC}"
else
  [[ -z $(git -C "${TRITON_SRC}" status --porcelain --untracked-files=no) ]] || {
    echo "${TRITON_SRC} has local changes; refusing to change its revision." >&2
    exit 1
  }
  git -C "${TRITON_SRC}" fetch --quiet origin "refs/tags/${TRITON_TAG}:refs/tags/${TRITON_TAG}"
  git -C "${TRITON_SRC}" checkout --quiet --detach "${TRITON_REV}"
fi
[[ $(git -C "${TRITON_SRC}" rev-parse HEAD) == "${TRITON_REV}" ]] || {
  echo "${TRITON_TAG} did not resolve to the tested revision ${TRITON_REV}." >&2
  exit 1
}

# Build isolation can put CMake inputs on a different overlay clock and make
# Ninja regenerate forever. Keep known build tools in the retained venv.
"${PYTHON}" -m pip install \
  "cmake==3.30.5" "ninja==1.11.1.1" "nanobind==2.10.2"
TRITON_EXT_ENABLED=1 MAX_JOBS="${MAX_JOBS}" \
  "${PYTHON}" -m pip install --no-build-isolation -e "${TRITON_SRC}"
"${PYTHON}" -m pip install --no-deps -e "${REPO}/third_party/jax-triton"

cd "${REPO}"
"${PYTHON}" -m jax_titan.ops.providers._triton_build
