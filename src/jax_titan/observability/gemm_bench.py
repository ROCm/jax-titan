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
  ops:PROVIDER         JAX-Titan's `linear` op or VJP for the stored training layout
  triton_gemm          AITER's Triton BF16 GEMM through jax-triton, reading either
                       operand transposed in place, with AITER's tiles for the shape
  jax_aiter            JAX-AITER's BF16 ASM GEMM, a [M,K] times b [N,K] transposed;
                       other layouts transpose operands first ("transposes" says how
                       many), and K must be a multiple of 64
  flydsl               FlyDSL's preshuffle BF16 GEMM through XLA's FFI
                       (ops/providers/flydsl.py, JAX_TITAN_FLYDSL_LIB): a [M,K] times
                       b [N,K] transposed, b shuffled into blocks in the timed program
  flydsl_kernel        the same kernel with its inputs converted before timing
"""

from __future__ import annotations

import argparse
import hashlib
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
_MFMA_PROVIDERS = {
    "gluon_intra_mfma_on": "mfma-schedule",
    "gluon_intra_mfma_off": "",
}
_MFMA_BACKENDS = tuple(f"ops:{provider}" for provider in _MFMA_PROVIDERS)
_SCHED_BARRIER = "call void @llvm.amdgcn.sched.barrier"


def _capture_mfma_compiles() -> list[dict]:
  """Records post-MFMASchedule LLVM IR identity from jax-triton compilations."""
  from jax_triton import triton_lib  # pylint: disable=g-import-not-at-top
  from jax_titan.ops.providers import _triton_build  # pylint: disable=g-import-not-at-top

  compiler = _triton_build.require_tutorial_build()
  records = []
  original = triton_lib.TritonFunction._compile_kernel

  def recording_compile(self, spec, gpu_target, backend, backend_options):
    result = original(self, spec, gpu_target, backend, backend_options)
    options, compilation = result[2], result[3]
    constants = {}
    for name in ("M", "N", "K"):
      value = spec.constants.get(name)
      value = getattr(value, "value", value)
      if value is not None:
        constants[name] = int(value)
    llir = compilation.llir
    records.append({
        "kernel": getattr(self, "name", type(self.fn).__name__),
        "m": constants.get("M"),
        "n": constants.get("N"),
        "k": constants.get("K"),
        "schedule_hint": getattr(options, "schedule_hint", None),
        "barrier_calls": llir.count(_SCHED_BARRIER),
        "llir_sha256": hashlib.sha256(llir.encode()).hexdigest(),
        "llir_bytes": len(llir.encode()),
        "compiler": compiler,
    })
    return result

  triton_lib.TritonFunction._compile_kernel = recording_compile
  return records


def _scheduler_proof(records: list[dict], m: int, n: int, k: int) -> dict | None:
  matches = [r for r in records if (r["m"], r["n"], r["k"]) == (m, n, k)]
  if not matches:
    return None
  fingerprints = {
      (r["schedule_hint"], r["barrier_calls"], r["llir_sha256"]) for r in matches
  }
  if len(fingerprints) != 1:
    raise RuntimeError(f"multiple compiler artifacts for GEMM {m}x{n}x{k}: {fingerprints}")
  return dict(matches[-1], captures=len(matches))


def _scheduler_proof_error(provider: str, proof: dict | None) -> str | None:
  if proof is None:
    return "no post-LLVM compiler artifact was captured"
  expected = _MFMA_PROVIDERS[provider]
  if proof["schedule_hint"] != expected:
    return (
        f"compiler received schedule_hint={proof['schedule_hint']!r}, "
        f"expected {expected!r}"
    )
  if provider.endswith("_on") and proof["barrier_calls"] <= 0:
    return "MFMASchedule inserted no llvm.amdgcn.sched.barrier calls"
  if provider.endswith("_off") and proof["barrier_calls"] != 0:
    return (
        "the disabled specialization contains "
        f"{proof['barrier_calls']} llvm.amdgcn.sched.barrier calls"
    )
  return None


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


def _problem_id(spec: dict) -> str:
  return spec.get("id") or f"{spec['m']}x{spec['n']}x{spec['k']} {spec['layout']}"


def _bootstrap(values, n=2000, seed=0):
  if len(values) < 2:
    return (values[0], values[0]) if values else (None, None)
  rng = random.Random(seed)
  meds = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(n))
  return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


DIRECT = ("triton_gemm", "jax_aiter", "flydsl", "flydsl_kernel")


def _ops_fn(jax, jnp, ops, layout: str, m: int, n: int):
  """An ops.linear forward, dgrad or wgrad matching a stored GEMM layout."""
  if layout == "MK,KN":
    return (lambda x, w: ops.linear(x, w)), "forward"
  if layout == "MK,NK":
    def dgrad(dy, w):
      x = jnp.zeros((m, n), dy.dtype)
      return jax.vjp(lambda z: ops.linear(z, w), x)[1](dy)[0]
    return dgrad, "dgrad"
  if layout == "KM,KN":
    def wgrad(x, dy):
      w = jnp.zeros((m, n), x.dtype)
      return jax.vjp(lambda z: ops.linear(x, z), w)[1](dy)[0]
    return wgrad, "wgrad"
  return None, None


def _direct(backend: str, a_layout: str, b_layout: str):
  """(fn(a, b), operands it transposes, prepare(a, b) run once before timing)."""
  same = lambda a, b: (a, b)  # noqa: E731
  to_mk = lambda a: a.T if a_layout == "KM" else a  # noqa: E731
  to_nk = lambda b: b if b_layout == "NK" else b.T  # noqa: E731
  moved = (a_layout == "KM") + (b_layout == "KN")
  if backend == "triton_gemm":
    from jax_titan.ops.providers import triton  # pylint: disable=g-import-not-at-top
    params = (a_layout == "KM", b_layout == "NK")
    return (lambda a, b: triton.triton_matmul(a, b, params)), 0, same
  if backend.startswith("flydsl"):
    from jax_titan.ops.providers import flydsl  # pylint: disable=g-import-not-at-top
    if backend == "flydsl_kernel":
      return flydsl.gemm_shuffled, 0, lambda a, b: (to_mk(a), flydsl.shuffle(to_nk(b)))
    return (lambda a, b: flydsl.gemm_shuffled(to_mk(a), flydsl.shuffle(to_nk(b)))), moved, same
  from jax_aiter.gemm import gemm  # pylint: disable=g-import-not-at-top
  return (lambda a, b: gemm(to_mk(a), to_nk(b))), moved, same


def _direct_skip(backend: str, n: int, k: int, batch: int, dtype: str) -> str | None:
  if batch != 1 or dtype != "bf16":
    return f"{backend} times bf16 problems without a batch"
  if backend == "jax_aiter" and k % 64:
    return "JAX-AITER's ASM GEMM needs K a multiple of 64"
  if backend.startswith("flydsl"):
    from jax_titan.ops.providers import flydsl  # pylint: disable=g-import-not-at-top
    if (n, k) not in flydsl.kernels():
      return f"no FlyDSL kernel for N x K {n}x{k} in ${flydsl.LIB_ENV}"
  return None


def _worker(spec_path: str, backend: str, out_path: str, iters: int, rounds: int) -> None:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax import lax  # pylint: disable=g-import-not-at-top

  provider = backend.split(":", 1)[1] if backend.startswith("ops:") else None
  if provider:
    from jax_titan import ops  # pylint: disable=g-import-not-at-top
    ops.set_policy(f"linear={provider}")
  if (
      backend == "triton_gemm"
      or provider == "triton"
      or (provider or "").startswith("gluon")
  ):
    from jax_titan.ops.providers import _triton_build  # pylint: disable=g-import-not-at-top
    _triton_build.require_tutorial_build()
  mfma_compiles = _capture_mfma_compiles() if provider in _MFMA_PROVIDERS else []
  direct = backend in DIRECT
  results = []
  for spec in json.loads(pathlib.Path(spec_path).read_text()):
    m, n, k, b = spec["m"], spec["n"], spec["k"], spec.get("batch", 1) or 1
    a_layout, b_layout = spec["layout"].split(",")
    dtype = jnp.dtype({"bf16": jnp.bfloat16, "f16": jnp.float16, "f32": jnp.float32}[spec.get("dtype", "bf16")])
    rec = dict(spec, backend=backend)
    ops_fn = role = None
    if provider:
      ops_fn, role = _ops_fn(jax, jnp, ops, spec["layout"], m, n)
      if ops_fn is None or b != 1:
        results.append(dict(rec, skipped=(
            "ops providers time unbatched MK,KN (forward), MK,NK (dgrad), "
            "or KM,KN (wgrad) problems"
        )))
        continue
      rec["role"] = role
    if direct and (why := _direct_skip(backend, n, k, b, spec.get("dtype", "bf16"))):
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
    x0, w0 = xs[0], ws[0]
    if provider:
      ops.reset_selections()
      fn = jax.jit(ops_fn)
    elif direct:
      call, rec["transposes"], prepare = _direct(backend, a_layout, b_layout)
      fn = jax.jit(call)
      prepare = jax.jit(prepare)
      xs, ws = map(list, zip(*(prepare(x, w) for x, w in zip(xs, ws))))
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
      selections = [s for s in ops.selections() if s.op == "linear"]
      rec["provider_ran"] = [s.provider for s in selections][-1:]
      if not any(s.provider == provider for s in selections):
        reasons = [reason for s in selections for p, reason in s.skipped if p == provider]
        results.append(dict(rec, skipped=(
            reasons[-1] if reasons else f"provider {provider} did not dispatch"
        )))
        continue
      if provider in _MFMA_PROVIDERS:
        proof = _scheduler_proof(mfma_compiles, m, n, k)
        rec["scheduler_proof"] = proof
        if proof_error := _scheduler_proof_error(provider, proof):
          results.append(dict(
              rec, skipped=f"MFMASchedule compiler proof failed: {proof_error}"
          ))
          continue
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
    ref = lax.dot_general(x0.astype(jnp.float32), w0.astype(jnp.float32), dims,
                          precision=lax.Precision.HIGHEST)
    rel = float(jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref))
    results.append(dict(rec, ms=statistics.median(per_round), rounds_ms=per_round, rel_l2=rel,
                        flops=2 * b * m * n * k))
    del xs, ws, x0, w0, out, got, ref, compiled
  pathlib.Path(out_path).write_text(json.dumps(results) + "\n")


def annotate_speedups(rows: list[dict], baseline: str) -> None:
  """Adds speedup against ``baseline`` to every matching timed problem."""
  by_problem = {}
  for row in rows:
    by_problem.setdefault(row["problem"], {})[row["backend"]] = row
  for cells in by_problem.values():
    base_ms = cells.get(baseline, {}).get("ms")
    if not base_ms:
      continue
    for row in cells.values():
      if row.get("ms"):
        row["speedup"] = base_ms / row["ms"]


def scheduler_ab_checks(rows: list[dict]) -> list[dict]:
  """Checks that every on/off pair used different, correctly transformed IR."""
  by_problem = {}
  for row in rows:
    by_problem.setdefault(row["problem"], {})[row["backend"]] = row
  checks = []
  on_backend, off_backend = _MFMA_BACKENDS
  for problem, cells in by_problem.items():
    if on_backend not in cells and off_backend not in cells:
      continue
    on, off = cells.get(on_backend), cells.get(off_backend)
    reason = ""
    if on is None or off is None:
      reason = "missing the on or off benchmark row"
    elif "ms" not in on or "ms" not in off:
      reason = (
          f"on: {on.get('skipped', 'not timed')}; "
          f"off: {off.get('skipped', 'not timed')}"
      )
    else:
      on_proof, off_proof = on.get("scheduler_proof"), off.get("scheduler_proof")
      if not on_proof or not off_proof:
        reason = "missing compiler proof"
      elif on_proof["compiler"]["revision"] != off_proof["compiler"]["revision"]:
        reason = "on/off used different Triton revisions"
      elif on_proof["schedule_hint"] != "mfma-schedule":
        reason = "enabled compiler proof has the wrong schedule hint"
      elif off_proof["schedule_hint"] != "":
        reason = "disabled compiler proof has the wrong schedule hint"
      elif on_proof["barrier_calls"] <= 0:
        reason = "enabled IR has no scheduler barriers"
      elif off_proof["barrier_calls"] != 0:
        reason = "disabled IR contains scheduler barriers"
      elif on_proof["llir_sha256"] == off_proof["llir_sha256"]:
        reason = "enabled and disabled LLVM IR hashes are identical"
    checks.append({
        "problem": problem,
        "status": "fail" if reason else "pass",
        "reason": reason,
        "on_llir_sha256": ((on or {}).get("scheduler_proof") or {}).get("llir_sha256"),
        "off_llir_sha256": ((off or {}).get("scheduler_proof") or {}).get("llir_sha256"),
        "on_barrier_calls": ((on or {}).get("scheduler_proof") or {}).get("barrier_calls"),
        "off_barrier_calls": ((off or {}).get("scheduler_proof") or {}).get("barrier_calls"),
    })
  return checks


def _gate_failed_scheduler_pairs(rows: list[dict], checks: list[dict]) -> None:
  failed = {check["problem"]: check["reason"] for check in checks if check["status"] == "fail"}
  for row in rows:
    if row["problem"] not in failed or row["backend"] not in _MFMA_BACKENDS:
      continue
    for field in ("ms", "ci_ms", "samples_ms", "tflops", "speedup", "rel_l2"):
      row.pop(field, None)
    row["skipped"] = f"MFMASchedule A/B gate failed: {failed[row['problem']]}"


def summarize(rows: list[dict], backends: list[str], baseline: str) -> list[dict]:
  """Dispatch-weighted throughput and speedup over matched passing problems."""
  by_problem = {}
  for row in rows:
    by_problem.setdefault(row["problem"], {})[row["backend"]] = row
  out = []
  for backend in backends:
    pairs = [
        (cells[baseline], cells[backend])
        for cells in by_problem.values()
        if baseline in cells and backend in cells
        and "ms" in cells[baseline] and "ms" in cells[backend]
    ]
    if not pairs:
      continue
    calls = lambda row: row.get("launches_per_step") or 1
    baseline_ms = sum(base["ms"] * calls(candidate) for base, candidate in pairs)
    candidate_ms = sum(candidate["ms"] * calls(candidate) for _, candidate in pairs)
    flops = sum(candidate["flops"] * calls(candidate) for _, candidate in pairs)
    out.append({
        "backend": backend,
        "baseline": baseline,
        "problems": len(pairs),
        "problems_total": len(by_problem),
        "effective_tflops": flops / (candidate_ms * 1e-3) / 1e12,
        "modeled_speedup": baseline_ms / candidate_ms,
        "milliseconds_per_step": candidate_ms,
    })
  return out


def run(specs: list[dict], backends: list[str], trials: int, iters: int, rounds: int,
        override: str = "", gpu: str = "", baseline: str = "xla_hipblaslt") -> dict:
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
          key = (_problem_id(r), backend)
          results.setdefault(key, []).append(r)
        print(f"trial {trial} {backend}: done", flush=True)
  rows = []
  for (pid, backend), recs in results.items():
    timed = [r for r in recs if "ms" in r]
    first = timed[0] if timed else recs[0]
    row = {k: first.get(k) for k in (
        "id", "m", "n", "k", "batch", "layout", "dtype", "launches_per_step",
        "lowered_to", "skipped", "provider_ran", "transposes", "role", "flops",
        "scheduler_proof",
    )}
    row.update(problem=pid, backend=backend, trials=len(timed))
    if timed:
      ms = [r["ms"] for r in timed]
      row.update(ms=statistics.median(ms), ci_ms=_bootstrap(ms), samples_ms=ms,
                 tflops=first["flops"] / (statistics.median(ms) * 1e-3) / 1e12,
                 rel_l2=max(r["rel_l2"] for r in timed))
      proofs = [r.get("scheduler_proof") for r in timed if r.get("scheduler_proof")]
      proof_fingerprints = {
          (p["schedule_hint"], p["barrier_calls"], p["llir_sha256"]) for p in proofs
      }
      if len(proof_fingerprints) > 1:
        for field in ("ms", "ci_ms", "samples_ms", "tflops", "rel_l2"):
          row.pop(field, None)
        row["skipped"] = "MFMASchedule compiler proof changed between trials"
    rows.append(row)
  scheduler_checks = []
  if all(backend in backends for backend in _MFMA_BACKENDS):
    scheduler_checks = scheduler_ab_checks(rows)
    checked = {check["problem"] for check in scheduler_checks}
    scheduler_checks.extend({
        "problem": problem,
        "status": "fail",
        "reason": "missing both on and off benchmark rows",
        "on_llir_sha256": None,
        "off_llir_sha256": None,
        "on_barrier_calls": None,
        "off_barrier_calls": None,
    } for problem in map(_problem_id, specs) if problem not in checked)
  _gate_failed_scheduler_pairs(rows, scheduler_checks)
  annotate_speedups(rows, baseline)
  return {
      "backends": backends, "trials": trials, "iters": iters, "rounds": rounds,
      "xla_flags": os.environ.get("XLA_FLAGS", ""), "baseline": baseline,
      "rows": rows, "summaries": summarize(rows, backends, baseline),
      "scheduler_ab": scheduler_checks,
      "scheduler_ab_ok": all(check["status"] == "pass" for check in scheduler_checks),
  }


def table(out: dict) -> str:
  by_problem = {}
  for r in out["rows"]:
    by_problem.setdefault(r["problem"], {})[r["backend"]] = r
  backends = out["backends"]
  baseline = out.get("baseline", "xla_hipblaslt")
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
      speedup = f", {r['speedup']:.3f}x" if "speedup" in r else ""
      cells.append(
          f"{r['ms']:.4f} ms ({lo:.4f}-{hi:.4f}), {r['tflops']:.0f} TF/s"
          f"{speedup}, {r['lowered_to']}{moved}"
      )
    lines.append(f"| {pid} | " + " | ".join(cells) + " |")
  summaries = out.get("summaries") or summarize(out["rows"], backends, baseline)
  if summaries:
    lines += [
        "",
        f"Dispatch-weighted overall results against `{baseline}`:",
        "",
        "| backend | coverage | effective TFLOP/s | modeled speedup | ms/step |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
      lines.append(
          f"| {row['backend']} | {row['problems']}/{row['problems_total']} | "
          f"{row['effective_tflops']:.1f} | {row['modeled_speedup']:.3f}x | "
          f"{row['milliseconds_per_step']:.3f} |"
      )
  checks = out.get("scheduler_ab", [])
  if checks:
    lines += [
        "",
        "MFMASchedule A/B compiler proof (post-pass LLVM IR):",
        "",
        "| problem | status | on barriers | off barriers | reason |",
        "|---|---:|---:|---:|---|",
    ]
    for check in checks:
      lines.append(
          f"| {check['problem']} | {check['status']} | "
          f"{check['on_barrier_calls']} | {check['off_barrier_calls']} | "
          f"{check['reason']} |"
      )
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
  parser.add_argument("--baseline", default="xla_hipblaslt",
                      help="Backend used for per-problem and modeled speedups.")
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
  out = run(
      specs, args.backends.split(","), args.trials, args.iters, args.rounds,
      args.override, args.gpu, args.baseline,
  )
  print(table(out))
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
  return 0 if out.get("scheduler_ab_ok", True) else 2


if __name__ == "__main__":
  raise SystemExit(main())
