"""Isolated GEMM timings through JAX, per problem and backend.

  python -m jax_titan.observability.gemm_bench PROBLEMS --backends xla_hipblaslt,xla_triton
      [--override FILE] [--trials 5] [--iters 50] [--rounds 3] [--json OUT]

PROBLEMS is `tuning/hipblaslt.py capture` or `join` output (hipBLASLt problems,
column major), or a JSON list of row-major GEMMs, {"m", "n", "k", "batch",
"layout": "MK,KN", "dtype": "bf16"}, the way `gemms.py` reports them. Each GEMM
runs as one `lax.dot_general` with its operands in the stored layouts, so
forward (MK,KN), activation-gradient (MK,NK) and weight-gradient (KM,KN) GEMMs
keep the transposes XLA gives them in a training step.

Every trial of every backend runs in its own process, since XLA reads its flags
once per process. A trial compiles each GEMM, warms it up, then times
`--rounds` rounds of `--iters` calls enqueued back to back (pipelined, one sync)
over enough input copies to exceed 512 MiB, so caches don't hold them, and
keeps the median round. The report gives the median over trials with a
bootstrap interval, TFLOP/s, which library the compiled program called, and
the error against an f32 GEMM of the same inputs.

Backends (flags are added to $XLA_FLAGS, so later ones win):
  xla_hipblaslt        level 0, Triton GEMMs off: hipBLASLt's first choice
  xla_hipblaslt_tuned  the same with HIPBLASLT_TUNING_OVERRIDE_FILE=--override
  xla_autotune         level 4, Triton GEMMs off: XLA times hipBLASLt's candidates
  xla_triton           level 4, Triton GEMMs only, with the exhaustive tiling search
  xla_auto             level 4, XLA picks between Triton and hipBLASLt
  ops:PROVIDER         JAX-Titan's `linear` op from PROVIDER (MK,KN problems only)
  triton_gemm          AITER's Triton BF16 GEMM through jax-triton, reading either
                       operand transposed in place, with AITER's tiles for the shape
  jax_aiter            JAX-AITER's BF16 ASM GEMM, a [M,K] times b [N,K] transposed;
                       other layouts transpose operands first ("transposes" says how
                       many), and K must be a multiple of 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time

_LEVEL0 = "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
BACKENDS = {
    "xla_hipblaslt": _LEVEL0,
    "xla_hipblaslt_tuned": _LEVEL0,
    "xla_autotune": "--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=false",
    "xla_triton": ("--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=true "
                   "--xla_gpu_triton_gemm_any=true --xla_gpu_cublas_fallback=false "
                   "--xla_gpu_exhaustive_tiling_search=true"),
    "xla_auto": "--xla_gpu_autotune_level=4 --xla_gpu_enable_triton_gemm=true",
}
_HIPBLASLT_TYPES = {"bf16_r": "bf16", "f16_r": "f16", "f32_r": "f32"}


def specs_from(problems) -> list[dict]:
  """Row-major GEMM specs from hipBLASLt problems (capture/join output) or specs."""
  if isinstance(problems, list):
    return problems
  out = []
  for p in problems["problems"]:
    k = p["key"]
    # Column-major D[m, n] = op(A) op(B) is row-major C[n, m]; see hipblaslt.site_key.
    out.append({"id": p.get("id"), "m": k["n"], "n": k["m"], "k": k["k"], "batch": k["batch_count"],
                "layout": ("KM" if k["transB"] == "T" else "MK") + "," + ("NK" if k["transA"] == "T" else "KN"),
                "dtype": _HIPBLASLT_TYPES.get(k["a_type"], "bf16"),
                "launches_per_step": p.get("launches_per_step")})
  return out


def _bootstrap(values, n=2000, seed=0):
  if len(values) < 2:
    return (values[0], values[0]) if values else (None, None)
  rng = random.Random(seed)
  meds = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(n))
  return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def _direct(backend: str, a_layout: str, b_layout: str):
  """(fn(a, b) for operands in the stored layouts, operands it transposes first)."""
  if backend == "triton_gemm":
    from jax_titan.ops.providers import triton  # pylint: disable=g-import-not-at-top
    params = (a_layout == "KM", b_layout == "NK")
    return (lambda a, b: triton.triton_matmul(a, b, params)), 0
  from jax_aiter.gemm import gemm  # pylint: disable=g-import-not-at-top
  return ((lambda a, b: gemm(a.T if a_layout == "KM" else a, b if b_layout == "NK" else b.T)),
          (a_layout == "KM") + (b_layout == "KN"))


def _direct_skip(backend: str, k: int, batch: int, dtype: str) -> str | None:
  if batch != 1 or dtype != "bf16":
    return f"{backend} times bf16 problems without a batch"
  if backend == "jax_aiter" and k % 64:
    return "JAX-AITER's ASM GEMM needs K a multiple of 64"
  return None


def _worker(spec_path: str, backend: str, out_path: str, iters: int, rounds: int) -> None:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax import lax  # pylint: disable=g-import-not-at-top

  provider = backend.split(":", 1)[1] if backend.startswith("ops:") else None
  if provider:
    from jax_titan import ops  # pylint: disable=g-import-not-at-top
    ops.set_policy(f"linear={provider}")
  direct = backend in ("triton_gemm", "jax_aiter")
  results = []
  for spec in json.loads(pathlib.Path(spec_path).read_text()):
    m, n, k, b = spec["m"], spec["n"], spec["k"], spec.get("batch", 1) or 1
    a_layout, b_layout = spec["layout"].split(",")
    dtype = jnp.dtype({"bf16": jnp.bfloat16, "f16": jnp.float16, "f32": jnp.float32}[spec.get("dtype", "bf16")])
    rec = dict(spec, backend=backend)
    if provider and (spec["layout"] != "MK,KN" or b != 1):
      results.append(dict(rec, skipped="ops providers time MK,KN problems without a batch"))
      continue
    if direct and (why := _direct_skip(backend, k, b, spec.get("dtype", "bf16"))):
      results.append(dict(rec, skipped=why))
      continue
    a_shape = (m, k) if a_layout == "MK" else (k, m)
    b_shape = (k, n) if b_layout == "KN" else (n, k)
    lc, rc = (1 if a_layout == "MK" else 0), (0 if b_layout == "KN" else 1)
    if b > 1:
      a_shape, b_shape, lc, rc = (b,) + a_shape, (b,) + b_shape, lc + 1, rc + 1
    dims = (((lc,), (rc,)), ((0,), (0,)) if b > 1 else ((), ()))
    set_bytes = (math.prod(a_shape) + math.prod(b_shape) + b * m * n) * dtype.itemsize
    sets = max(1, math.ceil(512 * 2**20 / set_bytes))
    keys = jax.random.split(jax.random.PRNGKey(0), 2 * sets)
    xs = [jax.random.normal(keys[2 * i], a_shape, jnp.float32).astype(dtype) for i in range(sets)]
    ws = [jax.random.normal(keys[2 * i + 1], b_shape, jnp.float32).astype(dtype) for i in range(sets)]
    if provider:
      fn = jax.jit(lambda x, w: ops.linear(x, w))
    elif direct:
      call, rec["transposes"] = _direct(backend, a_layout, b_layout)
      fn = jax.jit(call)
    else:
      fn = jax.jit(lambda x, w: lax.dot_general(x, w, dims))
    try:
      compiled = fn.lower(xs[0], ws[0]).compile()
    except Exception as e:  # pylint: disable=broad-except
      results.append(dict(rec, skipped=f"failed to compile: {str(e).splitlines()[0][:200]}"))
      continue
    hlo = compiled.as_text()
    targets = re.findall(r'custom_call_target="([^"]+)"', hlo)
    rec["lowered_to"] = ("hipBLASLt" if "__cublas$lt$matmul" in hlo else
                         "Triton" if "__triton" in hlo else
                         targets[0] if targets else "XLA")
    if provider:
      rec["provider_ran"] = [s.provider for s in ops.selections() if s.op == "linear"][-1:]
    out = None
    for i in range(3):
      out = compiled(xs[i % sets], ws[i % sets])
    out.block_until_ready()
    per_round = []
    for _ in range(rounds):
      t0 = time.perf_counter()
      for i in range(iters):
        out = compiled(xs[i % sets], ws[i % sets])
      out.block_until_ready()
      per_round.append((time.perf_counter() - t0) / iters * 1e3)
    got = compiled(xs[0], ws[0]).astype(jnp.float32)
    ref = lax.dot_general(xs[0].astype(jnp.float32), ws[0].astype(jnp.float32), dims,
                          precision=lax.Precision.HIGHEST)
    if provider:
      ref = jnp.dot(xs[0].astype(jnp.float32), ws[0].astype(jnp.float32), precision=lax.Precision.HIGHEST)
    rel = float(jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref))
    results.append(dict(rec, ms=statistics.median(per_round), rounds_ms=per_round, rel_l2=rel,
                        flops=2 * b * m * n * k))
    del xs, ws, out, got, ref, compiled
  pathlib.Path(out_path).write_text(json.dumps(results) + "\n")


def run(specs: list[dict], backends: list[str], trials: int, iters: int, rounds: int,
        override: str = "", gpu: str = "") -> dict:
  results = {}
  with tempfile.TemporaryDirectory() as tmp:
    spec_path = pathlib.Path(tmp, "specs.json")
    spec_path.write_text(json.dumps(specs))
    base = os.environ.get("XLA_FLAGS", "")
    for trial in range(trials):
      for backend in backends[trial % len(backends):] + backends[:trial % len(backends)]:
        env = dict(os.environ)
        flags = BACKENDS.get(backend, _LEVEL0)
        env["XLA_FLAGS"] = f"{base} {flags}".strip()
        env.pop("HIPBLASLT_TUNING_OVERRIDE_FILE", None)
        if backend == "xla_hipblaslt_tuned":
          if not override:
            raise ValueError("xla_hipblaslt_tuned needs --override")
          env["HIPBLASLT_TUNING_OVERRIDE_FILE"] = override
        if gpu:
          env["HIP_VISIBLE_DEVICES"] = gpu
        out = pathlib.Path(tmp, f"{backend.replace(':', '_')}_{trial}.json")
        cmd = [sys.executable, "-m", "jax_titan.observability.gemm_bench", "--worker", str(spec_path),
               "--backend", backend, "--worker_out", str(out), "--iters", str(iters), "--rounds", str(rounds)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out.exists():
          print(f"{backend} trial {trial} failed:\n{proc.stderr[-3000:]}", file=sys.stderr)
          continue
        for r in json.loads(out.read_text()):
          key = (r.get("id") or f"{r['m']}x{r['n']}x{r['k']} {r['layout']}", backend)
          results.setdefault(key, []).append(r)
        print(f"trial {trial} {backend}: done", flush=True)
  rows = []
  for (pid, backend), recs in results.items():
    timed = [r for r in recs if "ms" in r]
    first = recs[0]
    row = {k: first.get(k) for k in ("id", "m", "n", "k", "batch", "layout", "dtype", "launches_per_step",
                                     "lowered_to", "skipped", "provider_ran", "transposes")}
    row.update(problem=pid, backend=backend, trials=len(timed))
    if timed:
      ms = [r["ms"] for r in timed]
      row.update(ms=statistics.median(ms), ci_ms=_bootstrap(ms), samples_ms=ms,
                 tflops=first["flops"] / (statistics.median(ms) * 1e-3) / 1e12,
                 rel_l2=max(r["rel_l2"] for r in timed))
    rows.append(row)
  return {"backends": backends, "trials": trials, "iters": iters, "rounds": rounds,
          "xla_flags": os.environ.get("XLA_FLAGS", ""), "rows": rows}


def table(out: dict) -> str:
  by_problem = {}
  for r in out["rows"]:
    by_problem.setdefault(r["problem"], {})[r["backend"]] = r
  backends = out["backends"]
  lines = ["| problem | " + " | ".join(backends) + " |", "|---|" + "---|" * len(backends)]
  for pid, row in by_problem.items():
    cells = []
    for b in backends:
      r = row.get(b)
      if not r or "ms" not in r:
        cells.append(r.get("skipped", "failed") if r else "")
        continue
      lo, hi = r["ci_ms"]
      moved = f" after {r['transposes']} transposes" if r.get("transposes") else ""
      cells.append(f"{r['ms']:.4f} ms ({lo:.4f}-{hi:.4f}), {r['tflops']:.0f} TF/s, {r['lowered_to']}{moved}")
    lines.append(f"| {pid} | " + " | ".join(cells) + " |")
  return "\n".join(lines)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("problems", nargs="?", default="")
  parser.add_argument("--backends", default="xla_hipblaslt")
  parser.add_argument("--override", default="", help="Override file for xla_hipblaslt_tuned.")
  parser.add_argument("--only", default="", help="Comma-separated problem ids to keep.")
  parser.add_argument("--trials", type=int, default=5)
  parser.add_argument("--iters", type=int, default=50)
  parser.add_argument("--rounds", type=int, default=3)
  parser.add_argument("--gpu", default="", help="HIP_VISIBLE_DEVICES for the trials.")
  parser.add_argument("--json", default="")
  parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
  parser.add_argument("--backend", default="", help=argparse.SUPPRESS)
  parser.add_argument("--worker_out", default="", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)
  if args.worker:
    _worker(args.worker, args.backend, args.worker_out, args.iters, args.rounds)
    return 0
  specs = specs_from(json.loads(pathlib.Path(args.problems).read_text()))
  if args.only:
    keep = set(args.only.split(","))
    specs = [s for s in specs if s.get("id") in keep]
  out = run(specs, args.backends.split(","), args.trials, args.iters, args.rounds, args.override, args.gpu)
  print(table(out))
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
