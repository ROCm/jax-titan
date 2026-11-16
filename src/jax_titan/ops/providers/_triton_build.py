"""Identity and feature checks for the gfx950 tutorial Triton build."""

from __future__ import annotations

import inspect
import json
import pathlib
import subprocess

TRITON_TAG = "gfx950-tutorial-v3.0"
TRITON_REVISION = "3b0c7f081a3fc8471cae4941d1c579f2348bcbca"


def _source_root(triton_module) -> pathlib.Path | None:
  source = pathlib.Path(inspect.getfile(triton_module)).resolve()
  return next((parent for parent in source.parents if (parent / ".git").exists()), None)


def metadata() -> dict:
  """Returns enough compiler identity to make a scheduler result reproducible."""
  import triton  # pylint: disable=g-import-not-at-top
  from triton._C.libtriton import amd  # pylint: disable=g-import-not-at-top

  root = _source_root(triton)
  revision = ""
  source_dirty = False
  ext_enabled = False
  if root is not None:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    source_dirty = bool(subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    caches = sorted(root.glob("build/**/CMakeCache.txt"))
    ext_enabled = any(
        any(
            f"TRITON_EXT_ENABLED:BOOL={value}" in cache.read_text(errors="replace")
            for value in ("1", "ON", "TRUE")
        )
        for cache in caches
    )
  return {
      "version": triton.__version__,
      "revision": revision,
      "source_root": str(root) if root is not None else "",
      "source_dirty": source_dirty,
      "mfma_schedule_api": hasattr(amd, "add_mfma_schedule_pass"),
      "triton_ext_enabled": ext_enabled,
  }


def require_tutorial_build() -> dict:
  """Rejects PyPI or stale builds that cannot prove the MFMA scheduler A/B."""
  info = metadata()
  problems = []
  if info["revision"] != TRITON_REVISION:
    problems.append(
        f"revision is {info['revision'] or 'not an editable git checkout'}, "
        f"expected {TRITON_REVISION} ({TRITON_TAG})"
    )
  if info["source_dirty"]:
    problems.append("source checkout has local changes")
  if not info["mfma_schedule_api"]:
    problems.append("compiled AMD backend has no add_mfma_schedule_pass API")
  if not info["triton_ext_enabled"]:
    problems.append("build does not record TRITON_EXT_ENABLED=1")
  if problems:
    raise RuntimeError(
        "unsupported Triton for gfx950 Gluon benchmarking: "
        + "; ".join(problems)
        + ". Run scripts/install_gfx950_triton.sh."
    )
  return info


def main() -> int:
  print(json.dumps(require_tutorial_build(), indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
