"""Tuned hipBLASLt kernel selection for the GEMMs a program runs on ROCm.

XLA lowers most GEMMs on ROCm to hipBLASLt. At `--xla_gpu_autotune_level=0`
it runs the first solution hipBLASLt's heuristic returns, and hipBLASLt returns
a tuned one first when `HIPBLASLT_TUNING_OVERRIDE_FILE` holds a row for the
problem. This tool finds those rows and checks that they take effect:

  python3 hipblaslt.py probe [--json OUT]
  python3 hipblaslt.py capture LOG... --json problems.json
  python3 hipblaslt.py join problems.json gemms.json [--steps N] --json problems.json
  python3 hipblaslt.py sweep problems.json --out DIR --bench hipblaslt-bench --sweep_bin BIN
  python3 hipblaslt.py write DIR/results.json --out tuning.txt
  python3 hipblaslt.py verify LOG... --override tuning.txt [--api --sweep_bin BIN]
  python3 hipblaslt.py instep base_gemms.json tuned_gemms.json --override tuning.txt --out refined.txt
  python3 hipblaslt.py map DIR/results.json --names DIR2 --json OUT
  python3 hipblaslt.py report DIR/results.json [--times times.json]

Logs come from a run with `HIPBLASLT_LOG_MASK=32`, one `hipblaslt-bench`
command line per call, or `HIPBLASLT_LOG_MASK=64`, a YAML workload with call
counts that hipBLASLt writes when the process exits; set `HIPBLASLT_LOG_FILE`
and leave `HIPBLASLT_LOG_LEVEL` unset, which would override the mask.

A problem is keyed the way the override file is looked up: transposes, batch
count, m, n, k, the A, B and C types and the compute type, in hipBLASLt's
column-major terms. Leading dimensions, strides, the D type and the epilogue
are not part of the key, so calls that differ only in those share one row.

`sweep` has two stages. Stage 1 times every supported solution within the
workspace XLA gives the call, once each: with the sweep binary's `screen`, or
with `hipblaslt-bench` in tuning mode (`HIPBLASLT_TUNING_FILE`), the hipBLASLt
team's method, when a bench built for the same library is at hand. Stage 2
times finalists, the default among them, in separate processes with
interleaved rounds, and adopts a solution only when its bootstrap interval
separates from the default's and it is at least `--min_gain` faster.

A solution that wins alone can lose in the step, beside collectives that share
the GPU. `instep` compares traces of the step without and with the override
(`gemms.py --times`) row by row and writes the rows that are not slower.

Override rows are tied to one library build: the file's first line must be
`Git Version: <hipblasLtGetGitRevision>`, or hipBLASLt ignores the whole file.
`map` carries winners to another build by solution and kernel name.

Uses only the Python standard library: run it directly, or as
`python -m jax_titan.tuning.hipblaslt`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import ctypes.util
import glob
import json
import math
import os
import pathlib
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile

KEY = ("transA", "transB", "batch_count", "m", "n", "k", "a_type", "b_type", "c_type", "compute_type")
DESC = ("lda", "ldb", "ldc", "ldd", "stride_a", "stride_b", "stride_c", "stride_d", "alpha", "beta",
        "d_type", "scale_type", "bias_vector", "bias_type", "activation_type", "scaleA", "scaleB")
OVERRIDE_COLUMNS = KEY + ("solution_index",)
_HLO_TYPES = {"bf16": "bf16_r", "f16": "f16_r", "f32": "f32_r", "f8e4m3fn": "f8_r",
              "f8e5m2": "bf8_r", "f8e4m3fnuz": "f8_fnuz_r", "f8e5m2fnuz": "bf8_fnuz_r"}
_INT = ("m", "n", "k", "batch_count", "lda", "ldb", "ldc", "ldd", "stride_a", "stride_b", "stride_c",
        "stride_d", "solution_index", "call_count")
_GIT_MISMATCH = "git version and the override file git version are not the same"


def _number(text: str) -> bool:
  try:
    float(text)
    return True
  except ValueError:
    return False


def _compute_type(value: str) -> str:
  """`c_f32_r` in YAML workloads and `f32_r` in bench lines name the same type."""
  return value[2:] if value.startswith("c_") else value


def _normalize(args: dict) -> dict:
  out = {}
  for name, value in args.items():
    if name in ("M", "N", "K"):
      name = name.lower()
    if name in _INT:
      value = int(float(value))
    elif name in ("alpha", "beta"):
      value = float(value)
    elif name == "compute_type":
      value = _compute_type(str(value))
    out[name] = value
  out.setdefault("batch_count", 1)
  return out


def parse_bench_line(line: str) -> dict | None:
  """The arguments of a `hipblaslt-bench` command line hipBLASLt logged, or None."""
  at = line.find("hipblaslt-bench")
  if at < 0:
    return None
  try:
    tokens = shlex.split(line[at + len("hipblaslt-bench"):])
  except ValueError:
    tokens = line[at + len("hipblaslt-bench"):].split()
  args, i = {}, 0
  while i < len(tokens):
    token = tokens[i]
    if token.startswith("-") and not _number(token):
      name = token.lstrip("-")
      following = tokens[i + 1] if i + 1 < len(tokens) else None
      if following is not None and not (following.startswith("-") and not _number(following)):
        args[name], i = following, i + 2
      else:
        args[name], i = True, i + 1
    else:
      i += 1
  if not {"m", "n", "k"} <= set(args):
    return None
  return _normalize(args)


_YAML_FIELD = re.compile(r"(\w+):\s*([^,}]*)")


def parse_yaml_line(line: str) -> dict | None:
  """One `- { function: matmul, M: ..., call_count: ... }` entry of a mask-64 workload."""
  line = line.strip()
  if not line.startswith("- {") or "function: matmul" not in line:
    return None
  args = {k: v.strip() for k, v in _YAML_FIELD.findall(line[line.index("{") + 1:])}
  if not {"M", "N", "K"} <= set(args):
    return None
  return _normalize(args)


def key_of(args: dict) -> tuple:
  return tuple(args.get(k) for k in KEY)


def key_text(key: tuple) -> str:
  return ",".join(str(v) for v in key)


def flops_of(key: tuple) -> int:
  d = dict(zip(KEY, key))
  return 2 * d["batch_count"] * d["m"] * d["n"] * d["k"]


def capture(paths) -> dict:
  """Unique problems in mask-32 or mask-64 logs, with calls and the solutions used."""
  problems, lines, logs = {}, 0, []
  for pattern in paths:
    for path in sorted(glob.glob(pattern)) or [pattern]:
      logs.append(path)
      with open(path, errors="replace") as f:
        for line in f:
          args = parse_bench_line(line)
          calls = 1
          if args is None:
            args = parse_yaml_line(line)
            if args is None:
              continue
            calls = args.get("call_count", 1)
          lines += 1
          key = key_of(args)
          p = problems.setdefault(key, {"key": dict(zip(KEY, key)), "calls": 0, "solutions": {},
                                        "descs": {}})
          p["calls"] += calls
          if "solution_index" in args:
            idx = str(args["solution_index"])
            p["solutions"][idx] = p["solutions"].get(idx, 0) + calls
          desc = {k: args[k] for k in DESC if k in args}
          text = json.dumps(desc, sort_keys=True)
          p["descs"][text] = p["descs"].get(text, 0) + calls
  out = []
  for key, p in problems.items():
    descs = sorted(p.pop("descs").items(), key=lambda kv: -kv[1])
    p["desc"] = json.loads(descs[0][0])
    p["variants"] = len(descs)
    p["id"] = key_text(key)
    p["flops"] = flops_of(key)
    out.append(p)
  out.sort(key=lambda p: -p["flops"] * p["calls"])
  return {"logs": logs, "lines": lines, "problems": out}


def site_key(site: dict) -> tuple:
  """The hipBLASLt key of a `gemms.py` site, which computes C[M, N] row major.

  Row major C[M, N] is column major C^T, so hipBLASLt computes B^T A^T with
  m = N and n = M. An operand stored transposed in row major (A as KM, B as NK)
  is the untransposed column-major matrix, so it needs op T.
  """
  a, b = site["layout"].split(",")
  # hipBLASLt's A is the site's B and its B the site's A.
  types = tuple(_HLO_TYPES.get(site.get(f"{x}_dtype", ""), site.get(f"{x}_dtype", "")) for x in "bac")
  return ("T" if b == "NK" else "N", "T" if a == "KM" else "N", site["batch"] or 1,
          site["n"], site["m"], site["k"]) + types


def join(problems: dict, gemms_json: dict, steps: int = 0) -> dict:
  """Attach the HLO sites (launches per step, workspace, op names) to captured problems."""
  sites = [s for m in gemms_json.get("modules", []) for s in m.get("gemms", [])
           if s.get("library") == "hipBLASLt"]
  by_key = {}
  for p in problems["problems"]:
    k = p["key"]
    by_key[tuple(k[x] for x in KEY[:9])] = p
    p["sites"] = []
  unmatched = []
  for s in sites:
    key = site_key(s)
    p = by_key.get(key)
    if p is None:
      unmatched.append({"module": s["module"], "kernel": s["kernel"], "key": key_text(key),
                        "count": s["count"], "source": s.get("source", "")})
      continue
    p["sites"].append({"module": s["module"], "kernel": s["kernel"], "count": s["count"],
                       "pass": s.get("pass", ""), "source": s.get("source", ""),
                       "algorithm": s.get("config", {}).get("algorithm"),
                       "workspace_bytes": s.get("config", {}).get("workspace_bytes")})
  for p in problems["problems"]:
    p["launches_per_step"] = sum(s["count"] for s in p["sites"]) or None
    ws = [s["workspace_bytes"] for s in p["sites"] if s["workspace_bytes"]]
    p["workspace_bytes"] = min(ws) if ws else None
    if steps and p["launches_per_step"]:
      p["calls_per_step_logged"] = p["calls"] / steps
  problems["unmatched_sites"] = unmatched
  problems["problems_without_sites"] = [p["id"] for p in problems["problems"] if not p["sites"]]
  return problems


# Statistics.


def bootstrap_ci(samples, stat=statistics.median, n: int = 2000, alpha: float = 0.05, seed: int = 0):
  """A percentile bootstrap interval of `stat` over `samples`."""
  samples = list(samples)
  if len(samples) < 2:
    value = stat(samples) if samples else float("nan")
    return value, value
  rng = random.Random(seed)
  values = sorted(stat(rng.choices(samples, k=len(samples))) for _ in range(n))
  return values[int(alpha / 2 * n)], values[min(n - 1, int((1 - alpha / 2) * n))]


def decide(default: list[float], candidate: list[float], min_gain: float = 0.02) -> dict:
  """Whether `candidate` times (lower is better) beat `default` with separated intervals."""
  d, c = statistics.median(default), statistics.median(candidate)
  d_ci, c_ci = bootstrap_ci(default), bootstrap_ci(candidate, seed=1)
  gain = d / c - 1 if c else float("nan")
  return {"default_ms": d, "default_ci": d_ci, "candidate_ms": c, "candidate_ci": c_ci,
          "gain": gain, "separated": c_ci[1] < d_ci[0],
          "adopt": c_ci[1] < d_ci[0] and gain >= min_gain}


# Running the bench tools.


def bench_command(problem: dict, bench: str, extra=()) -> list[str]:
  """`hipblaslt-bench` arguments for a problem's most frequent call."""
  k, d = problem["key"], problem.get("desc", {})
  m, n, kk, b = k["m"], k["n"], k["k"], k["batch_count"]
  ta, tb = k["transA"], k["transB"]
  lda = d.get("lda", m if ta == "N" else kk)
  ldb = d.get("ldb", kk if tb == "N" else n)
  ldc, ldd = d.get("ldc", m), d.get("ldd", m)
  cmd = [bench, "--api_method", "c", "-f", "matmul", "-m", str(m), "-n", str(n), "-k", str(kk),
         "--lda", str(lda), "--ldb", str(ldb), "--ldc", str(ldc), "--ldd", str(ldd),
         "--batch_count", str(b), "--transA", ta, "--transB", tb,
         "--alpha", str(d.get("alpha", 1.0)), "--beta", str(d.get("beta", 0.0)),
         "--a_type", k["a_type"], "--b_type", k["b_type"], "--c_type", k["c_type"],
         "--d_type", d.get("d_type", k["c_type"]), "--scale_type", d.get("scale_type", "f32_r"),
         "--compute_type", k["compute_type"]]
  for s in ("stride_a", "stride_b", "stride_c", "stride_d"):
    if s in d and b > 1:
      cmd += [f"--{s}", str(d[s])]
  if d.get("bias_vector") in (True, "true", "1", 1):
    cmd += ["--bias_vector", "--bias_type", d.get("bias_type", "f32_r")]
  if d.get("activation_type") not in (None, "none"):
    cmd += ["--activation_type", str(d["activation_type"])]
  return cmd + list(extra)


def parse_tuning_file(path) -> tuple[str | None, list[dict]]:
  """The git version and rows of an override or tuning file (header line, then values)."""
  git, rows, header = None, [], None
  for line in pathlib.Path(path).read_text().splitlines():
    line = line.strip()
    if not line:
      continue
    if line.startswith("Git Version:"):
      git = line.split(":", 1)[1].strip()
      continue
    cells = [c.strip() for c in line.split(",")]
    if "transA" in cells:
      header = cells
      continue
    if header and len(cells) >= len(header):
      row = dict(zip(header, cells))
      rows.append(_normalize(row))
  return git, rows


def _run(cmd, env=None, timeout=None, log=None) -> subprocess.CompletedProcess:
  result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
  if log:
    pathlib.Path(log).write_text(" ".join(shlex.quote(c) for c in cmd) + "\n" + result.stdout
                                 + ("\n--- stderr\n" + result.stderr if result.stderr else ""))
  return result


def _gpu_env(gpu, extra=None):
  env = dict(os.environ)
  env.pop("HIPBLASLT_LOG_MASK", None)
  env.pop("HIPBLASLT_LOG_LEVEL", None)
  env.pop("HIPBLASLT_TUNING_OVERRIDE_FILE", None)
  if gpu is not None:
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
  env.update(extra or {})
  return env


def _json_lines(text: str) -> list[dict]:
  out = []
  for line in text.splitlines():
    line = line.strip()
    if line.startswith("{"):
      try:
        out.append(json.loads(line))
      except json.JSONDecodeError:
        pass
  return out


def _sweep_args(problem: dict) -> list[str]:
  """The sweep binary's problem arguments. beta is not in the override key, but
  a call that adds into C (XLA's residual adds) reads C, which changes timings."""
  k = problem["key"]
  beta = float((problem.get("desc") or {}).get("beta") or 0.0)
  return [str(k["m"]), str(k["n"]), str(k["k"]), k["transA"], k["transB"],
          "--batch", str(k["batch_count"]), "--types", f"{k['a_type']},{k['b_type']},{k['c_type']}",
          "--compute", k["compute_type"]] + (["--beta", f"{beta:g}"] if beta else [])


_TYPES = ("bf16_r", "f16_r", "f32_r")


def _supported(problem: dict) -> str:
  k = problem["key"]
  ok = all(k[t] in _TYPES for t in ("a_type", "b_type", "c_type")) and k["compute_type"] in ("f32_r", "xf32_r")
  return "" if ok else f"the sweep binary takes {', '.join(_TYPES)} operands with f32 or xf32 compute"


def list_solutions(problem: dict, sweep_bin: str, workspace: int, gpu=None, heuristic: int = 128,
                   env=None, log=None) -> dict:
  """Heuristic ranks and every solution with support, workspace and names."""
  cmd = [sweep_bin, "list", *_sweep_args(problem), "--ws", str(workspace),
         "--heuristic", str(heuristic)]
  result = _run(cmd, env=_gpu_env(gpu, env), log=log)
  rows = _json_lines(result.stdout)
  return {"heuristic": [r for r in rows if r.get("kind") == "heuristic"],
          "solutions": {r["index"]: r for r in rows if r.get("kind") == "solution"},
          "ok": result.returncode == 0, "stderr": result.stderr[-2000:]}


def stage1(problem: dict, bench: str, out_dir: pathlib.Path, workspace: int, gpu=None,
           iters=None, cold_iters=None) -> dict:
  """hipblaslt-bench tuning mode: every supported solution within `workspace`."""
  tag = re.sub(r"[^\w]+", "_", problem["id"])
  tuning = out_dir / f"{tag}.tuning.txt"
  tuning.unlink(missing_ok=True)
  extra = ["--algo_method", "heuristic", "--requested_solution", "-1"]
  if iters:
    extra += ["-i", str(iters)]
  if cold_iters:
    extra += ["-j", str(cold_iters)]
  cmd = bench_command(problem, bench, extra)
  env = _gpu_env(gpu, {"HIPBLASLT_TUNING_FILE": str(tuning),
                       "HIPBLASLT_TUNING_USER_MAX_WORKSPACE": str(workspace)})
  result = _run(cmd, env=env, log=out_dir / f"{tag}.bench.log")
  git, rows = parse_tuning_file(tuning) if tuning.exists() else (None, [])
  winner = next((r for r in rows if key_of(r)[:6] == key_of(problem["key"])[:6]), None)
  return {"ok": result.returncode == 0 and winner is not None, "git_version": git,
          "winner": winner, "tuning_file": str(tuning), "returncode": result.returncode,
          "ranking": parse_bench_results(result.stdout)}


def screen(problem: dict, sweep_bin: str, out_dir: pathlib.Path, workspace: int, gpu=None,
           target_ms: float = 20, warmup_ms: float = 2000) -> dict:
  """One timed pass over every supported solution within `workspace` (the sweep binary)."""
  tag = re.sub(r"[^\w]+", "_", problem["id"])
  cmd = [sweep_bin, "screen", *_sweep_args(problem), "--ws", str(workspace), "--target_ms", str(target_ms),
         "--warmup_ms", str(warmup_ms)]
  result = _run(cmd, env=_gpu_env(gpu), log=out_dir / f"{tag}.screen.log")
  rows = _json_lines(result.stdout)
  ranking = sorted(({"solution_index": r["index"], "us": r["ms"] * 1e3, "heuristic_rank": r["heuristic_rank"]}
                    for r in rows if r.get("kind") == "screen"), key=lambda r: r["us"])
  winner = dict(problem["key"], solution_index=ranking[0]["solution_index"]) if ranking else None
  return {"ok": result.returncode == 0 and bool(ranking), "tool": "screen", "winner": winner,
          "ranking": ranking[:50], "screened": len(ranking),
          "errors": sum(1 for r in rows if r.get("kind") == "screen_error")}


def parse_bench_results(text: str) -> list[dict]:
  """Per-solution rows of hipblaslt-bench output (a CSV header with `solution_index`)."""
  rows, header = [], None
  for line in text.splitlines():
    cells = [c.strip() for c in line.split(",")]
    if "solution_index" in cells and "us" in cells:
      header = cells
      continue
    if header and len(cells) == len(header):
      row = dict(zip(header, cells))
      try:
        rows.append({"solution_index": int(row["solution_index"]), "us": float(row["us"]),
                     "kernel_name": row.get("kernel_name", "")})
      except (KeyError, ValueError):
        continue
  best = {}
  for r in rows:
    if r["solution_index"] not in best or r["us"] < best[r["solution_index"]]["us"]:
      best[r["solution_index"]] = r
  return sorted(best.values(), key=lambda r: r["us"])


def time_candidates(problem: dict, sweep_bin: str, candidates, workspace: int, gpu=None,
                    rounds: int = 5, target_ms: float = 100, warmup_ms: float = 1000,
                    mode: str = "rot", log=None) -> list[dict]:
  """One process of interleaved rounds over `candidates` (-1 is heuristic #0)."""
  cmd = [sweep_bin, "time", *_sweep_args(problem), "--ws", str(workspace), "--rounds", str(rounds),
         "--target_ms", str(target_ms), "--warmup_ms", str(warmup_ms), "--mode", mode,
         *[str(c) for c in candidates]]
  result = _run(cmd, env=_gpu_env(gpu), log=log)
  return _json_lines(result.stdout)


def validate(problem: dict, sweep_bin: str, candidates, workspace: int, gpu=None, log=None) -> list[dict]:
  cmd = [sweep_bin, "validate", *_sweep_args(problem), "--ws", str(workspace),
         *[str(c) for c in candidates]]
  result = _run(cmd, env=_gpu_env(gpu), log=log)
  return [r for r in _json_lines(result.stdout) if r.get("kind") == "validate"]


def _finalists(problem, s1, listing, seeds, top: int) -> list[int]:
  """The default, stage 1's winner, the next fastest from its ranking, and seeds."""
  out = [-1]
  if s1.get("winner"):
    out.append(int(s1["winner"]["solution_index"]))
  for r in s1.get("ranking", [])[:top]:
    out.append(r["solution_index"])
  supported = {i for i, r in listing.get("solutions", {}).items() if r.get("supported")}
  out += [s for s in seeds if not supported or s in supported]
  seen, unique = set(), []
  for c in out:
    if c not in seen:
      seen.add(c)
      unique.append(c)
  return unique


def sweep_problem(problem, args, gpu, out_dir, previous=None) -> dict:
  reason = _supported(problem)
  if reason:
    return {"id": problem["id"], "skipped": reason}
  ws = args.workspace or problem.get("workspace_bytes") or 32 * 2**20
  tag = re.sub(r"[^\w]+", "_", problem["id"])
  listing = list_solutions(problem, args.sweep_bin, ws, gpu, log=out_dir / f"{tag}.list.log")
  heur = listing["heuristic"]
  default = heur[0]["index"] if heur else None
  record = {"id": problem["id"], "key": problem["key"], "workspace_bytes": ws, "default_index": default,
            "calls": problem.get("calls"), "launches_per_step": problem.get("launches_per_step"),
            "logged_solutions": problem.get("solutions", {}), "flops": problem.get("flops")}
  if args.stage in ("1", "both"):
    if args.stage1 == "bench":
      record["stage1"] = stage1(problem, args.bench, out_dir, ws, gpu, args.iters, args.cold_iters)
    else:
      record["stage1"] = screen(problem, args.sweep_bin, out_dir, ws, gpu, args.screen_ms, args.warmup_ms)
  else:
    record["stage1"] = (previous or {}).get("stage1") or {"ranking": [], "winner": None}
  seeds = [int(s) for s in (args.seeds or {}).get(problem["id"], [])]
  if args.stage in ("2", "both"):
    finalists = _finalists(problem, record["stage1"], listing, seeds, args.top)
    samples = {c: [] for c in finalists}
    for trial in range(args.trials):
      order = finalists[trial % len(finalists):] + finalists[:trial % len(finalists)]
      rows = time_candidates(problem, args.sweep_bin, order, ws, gpu, args.rounds, args.target_ms,
                             args.warmup_ms, log=out_dir / f"{tag}.trial{trial}.log")
      per = {}
      for r in rows:
        if r.get("kind") == "time":
          per.setdefault(r["requested"], []).append(r["ms"])
      for c, values in per.items():
        if c in samples and values:
          samples[c].append(statistics.median(values))
    resolved = {}
    for r in rows if args.trials else []:
      if r.get("kind") == "time":
        resolved[r["requested"]] = r["index"]
    timed = {c: v for c, v in samples.items() if v}
    record["stage2"] = {"finalists": finalists, "resolved": {str(k): v for k, v in resolved.items()},
                        "samples_ms": {str(k): v for k, v in timed.items()}}
    if -1 in timed and len(timed) > 1:
      best = min((c for c in timed if c != -1), key=lambda c: statistics.median(timed[c]))
      verdict = decide(timed[-1], timed[best], args.min_gain)
      best_index = resolved.get(best, best)
      record["stage2"].update(verdict, best=best_index)
      record["winner"] = best_index if verdict["adopt"] and best_index != default else None
      if record["winner"] is not None and not args.no_validate:
        record["validation"] = validate(problem, args.sweep_bin, [record["winner"]], ws, gpu,
                                        log=out_dir / f"{tag}.validate.log")
        bad = [v for v in record["validation"] if not v.get("ok", False)]
        if bad:
          record["winner"] = None
          record["rejected"] = "validation"
  sols = listing.get("solutions", {})
  for field, idx in (("default_names", record.get("default_index")),
                     ("winner_names", record.get("winner"))):
    if idx is not None and idx in sols:
      record[field] = {"kernel": sols[idx].get("kernel", ""), "solution": sols[idx].get("solution", "")}
  return record


def sweep(problems: dict, args) -> dict:
  out_dir = pathlib.Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  todo = [p for p in problems["problems"] if not args.only or re.search(args.only, p["id"])]
  gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else [None]
  # A stage-2 run reuses the stage-1 results of an earlier sweep into the same directory.
  previous = {}
  if args.stage == "2" and (out_dir / "results.json").exists():
    previous = {r["id"]: r for r in json.loads((out_dir / "results.json").read_text())["results"]}
  results = []
  with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
    pending = {}
    queue = list(todo)
    free = list(gpus)
    while queue or pending:
      while queue and free:
        gpu = free.pop(0)
        p = queue.pop(0)
        pending[pool.submit(sweep_problem, p, args, gpu, out_dir, previous.get(p["id"]))] = gpu
      done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
      for fut in done:
        free.append(pending.pop(fut))
        results.append(fut.result())
        r = results[-1]
        print(f"{r['id']}: default {r.get('default_index')} winner {r.get('winner')}"
              + (f" gain {100 * r['stage2']['gain']:.1f}%" if "gain" in r.get("stage2", {}) else "")
              + (f" ({r['skipped']})" if r.get("skipped") else ""), flush=True)
  if args.only and (out_dir / "results.json").exists():
    # Keep the earlier results of the problems --only left out.
    earlier = json.loads((out_dir / "results.json").read_text())["results"]
    done = {r["id"] for r in results}
    results += [r for r in earlier if r["id"] not in done]
    todo = problems["problems"]
  order = {p["id"]: i for i, p in enumerate(todo)}
  results.sort(key=lambda r: order.get(r["id"], 0))
  out = {"probe": getattr(args, "probe_info", None), "workspace_bytes": args.workspace,
         "min_gain": args.min_gain, "trials": args.trials, "rounds": args.rounds,
         "results": results}
  (out_dir / "results.json").write_text(json.dumps(out, indent=1) + "\n")
  return out


# Override files.


def write_override(rows: list[dict], git_version: str, path, kernel_names: bool = False) -> int:
  """hipBLASLt's override format: the git version line, then a header and a row per problem."""
  lines = [f"Git Version: {git_version}"]
  for row in rows:
    if int(row["solution_index"]) <= 0:
      raise ValueError(f"solution_index must be positive: {row}")
    columns = OVERRIDE_COLUMNS + (("kernel_name",) if kernel_names and row.get("kernel_name") else ())
    lines.append(",".join(columns))
    lines.append(",".join(str(row[c]) for c in columns))
  pathlib.Path(path).write_text("\n".join(lines) + "\n")
  return len(rows)


def rows_from_results(results: dict) -> list[dict]:
  rows = []
  for r in results["results"]:
    if r.get("winner") is None:
      continue
    row = dict(r["key"], solution_index=r["winner"])
    if r.get("winner_names", {}).get("kernel"):
      row["kernel_name"] = r["winner_names"]["kernel"]
    rows.append(row)
  return rows


def instep_times(gemms_json: dict) -> dict:
  """{problem id: in-step us per launch and its interval}, launch-weighted over a problem's sites,
  from `gemms.py --json --times`."""
  acc = {}
  for module in gemms_json.get("modules", []):
    for s in module.get("gemms", []):
      t = s.get("time")
      if not t or not t.get("median_us") or s.get("library") != "hipBLASLt":
        continue
      pid = key_text(site_key(s) + ("f32_r",))
      n = t["launches_per_step"]
      a = acc.setdefault(pid, [0.0, 0.0, 0.0, 0.0])
      for i, v in enumerate((n, n * t["median_us"], n * t["ci_us"][0], n * t["ci_us"][1])):
        a[i] += v
  return {pid: {"launches_per_step": n, "us": us / n, "ci_us": [lo / n, hi / n]}
          for pid, (n, us, lo, hi) in acc.items() if n}


def instep(base_gemms: dict, tuned_gemms: dict, override_path) -> dict:
  """Per override row, its problem's in-step launches without and with the override.

  A row whose tuned launches run slower in the step, their intervals apart, is
  not kept: its solution won alone but loses beside the step's other work, such
  as collectives sharing the GPU.
  """
  git, rows = parse_tuning_file(override_path)
  base, tuned = instep_times(base_gemms), instep_times(tuned_gemms)
  out = []
  for row in rows:
    pid = key_text(key_of(row))
    b, t = base.get(pid), tuned.get(pid)
    rec = {"id": pid, "solution_index": row["solution_index"], "base": b, "tuned": t}
    if b and t:
      rec["delta"] = t["us"] / b["us"] - 1
      rec["slower"] = t["ci_us"][0] > b["ci_us"][1]
      rec["ms_per_step"] = (t["us"] - b["us"]) * b["launches_per_step"] / 1e3
    rec["keep"] = not rec.get("slower", False)
    out.append(rec)
  return {"git_version": git, "rows": out, "kept": [r for r, rec in zip(rows, out) if rec["keep"]]}


def instep_table(result: dict) -> str:
  lines = ["| problem | solution | base us | tuned us | change | ms/step | keep |", "|---|---|---|---|---|---|---|"]
  for r in result["rows"]:
    if r.get("delta") is None:
      lines.append(f"| {r['id']} | {r['solution_index']} | | | not in both traces | | yes |")
      continue
    lines.append(f"| {r['id']} | {r['solution_index']} | {r['base']['us']:.0f} | {r['tuned']['us']:.0f} | "
                 f"{r['delta']:+.1%} | {r['ms_per_step']:+.2f} | {'yes' if r['keep'] else 'no'} |")
  return "\n".join(lines)


def verify(logs, override_path) -> dict:
  """Per override row: the solutions a mask-32 run used for its key (hit, miss or absent)."""
  git, rows = parse_tuning_file(override_path)
  used, mismatch = {}, False
  for pattern in logs:
    for path in sorted(glob.glob(pattern)) or [pattern]:
      with open(path, errors="replace") as f:
        for line in f:
          if _GIT_MISMATCH in line:
            mismatch = True
          args = parse_bench_line(line)
          if args is None or "solution_index" not in args:
            continue
          k = key_of(args)
          used.setdefault(k, {})
          used[k][args["solution_index"]] = used[k].get(args["solution_index"], 0) + 1
  checks = []
  for row in rows:
    k = key_of(row)
    seen = used.get(k, {})
    status = "absent" if not seen else "hit" if set(seen) == {row["solution_index"]} else "miss"
    checks.append({"id": key_text(k), "tuned": row["solution_index"],
                   "used": {str(i): n for i, n in seen.items()}, "status": status})
  counts = {s: sum(c["status"] == s for c in checks) for s in ("hit", "miss", "absent")}
  return {"override": str(override_path), "git_version": git, "git_mismatch_logged": mismatch,
          "rows": len(rows), **counts, "checks": checks}


def verify_api(override_path, sweep_bin: str, gpu=None) -> dict:
  """Heuristic #0 per row with and without the override, through the hipBLASLt API."""
  git, rows = parse_tuning_file(override_path)
  checks = []
  for row in rows:
    problem = {"key": {k: row[k] for k in KEY}}
    plain = list_solutions(problem, sweep_bin, 128 * 2**20, gpu, heuristic=1)
    tuned = list_solutions(problem, sweep_bin, 128 * 2**20, gpu, heuristic=1,
                           env={"HIPBLASLT_TUNING_OVERRIDE_FILE": str(override_path)})
    h0 = plain["heuristic"][0]["index"] if plain["heuristic"] else None
    h1 = tuned["heuristic"][0]["index"] if tuned["heuristic"] else None
    checks.append({"id": key_text(key_of(row)), "tuned": row["solution_index"], "without": h0,
                   "with": h1, "status": "hit" if h1 == row["solution_index"] else "miss"})
  return {"override": str(override_path), "git_version": git,
          "hit": sum(c["status"] == "hit" for c in checks), "rows": len(rows), "checks": checks}


# Carrying winners to another build.

# Name tokens that differ between hipBLASLt versions without changing the kernel.
_VERSION_TOKENS = re.compile(r"^(SK|WGMXCC|BL\d|BS\d|IA\d|LDSSI)")


def name_tokens(name: str) -> dict:
  out = {}
  for t in name.split("_"):
    if t and not _VERSION_TOKENS.match(t):
      out[t] = out.get(t, 0) + 1
  return out


def name_distance(a: str, b: str) -> int:
  """Tokens of `a` missing from `b`, ignoring version-only tokens (0 is a match)."""
  ta, tb = name_tokens(a), name_tokens(b)
  return sum(max(0, n - tb.get(t, 0)) for t, n in ta.items())


def match(names: dict, solutions: dict) -> dict:
  """The index in `solutions` ({index: {solution, kernel, supported}}) for `names`."""
  exact = [i for i, s in solutions.items() if s.get("solution") == names.get("solution")]
  if exact:
    return {"index": exact[0], "how": "solution name"}
  by_solution = [i for i, s in solutions.items()
                 if names.get("solution") and name_distance(s.get("solution", ""), names["solution"]) == 0]
  if by_solution:
    return {"index": by_solution[0], "how": "solution tokens"}
  by_kernel = [i for i, s in solutions.items()
               if names.get("kernel") and name_distance(s.get("kernel", ""), names["kernel"]) == 0]
  if by_kernel:
    return {"index": by_kernel[0], "how": "kernel tokens"}
  return {"index": None, "how": "no match"}


def map_results(results: dict, sweep_bin: str, gpu=None, listings_dir=None) -> list[dict]:
  out = []
  for r in results["results"]:
    if r.get("winner") is None or "winner_names" not in r:
      continue
    problem = {"key": r["key"]}
    if listings_dir:
      tag = re.sub(r"[^\w]+", "_", r["id"])
      rows = _json_lines(pathlib.Path(listings_dir, f"{tag}.list.log").read_text())
      sols = {x["index"]: x for x in rows if x.get("kind") == "solution"}
    else:
      sols = list_solutions(problem, sweep_bin, r["workspace_bytes"], gpu)["solutions"]
    found = match(r["winner_names"], {i: s for i, s in sols.items() if s.get("supported", 1)})
    out.append({"id": r["id"], "from_index": r["winner"], **r["winner_names"], **found})
  return out


# Reports.


def report(results: dict, times: dict | None = None) -> str:
  """A table per problem; `XLA = #0` checks that the solution XLA logged is heuristic #0."""
  head = (f"| problem (transA,transB,batch,m,n,k,types) | launches/step | XLA = #0 | default | best | "
          f"default ms | best ms | gain | adopted |\n|---|---|---|---|---|---|---|---|---|")
  lines, saved, mismatched = [head], 0.0, 0
  for r in results["results"]:
    s2 = r.get("stage2", {})
    if r.get("skipped"):
      lines.append(f"| {r['id']} | | | | | | | | skipped: {r['skipped']} |")
      continue
    d_ms, c_ms = s2.get("default_ms"), s2.get("candidate_ms")
    gain = s2.get("gain")
    launches = r.get("launches_per_step") or 0
    logged = {int(i) for i in r.get("logged_solutions", {})}
    same = "" if not logged else "yes" if logged == {r.get("default_index")} else f"no {sorted(logged)}"
    mismatched += same.startswith("no")
    if r.get("winner") is not None and d_ms and c_ms:
      saved += launches * (d_ms - c_ms)
    lines.append(f"| {r['id']} | {launches or ''} | {same} | {r.get('default_index')} | {s2.get('best', '')} | "
                 + (f"{d_ms:.4f}" if d_ms else "") + " | " + (f"{c_ms:.4f}" if c_ms else "") + " | "
                 + (f"{100 * gain:+.1f}%" if gain is not None else "") + " | "
                 + ("yes" if r.get("winner") is not None else "no") + " |")
  lines.append(f"\nPredicted GEMM time saved per step (isolated timings): {saved:.2f} ms")
  if mismatched:
    lines.append(f"{mismatched} problems ran a solution other than heuristic #0 at XLA's workspace: "
                 "check the workspace and the descriptor.")
  return "\n".join(lines)


# Library identity.


def _library_path() -> str | None:
  for env in ("JAX_TITAN_HIPBLASLT",):
    if os.environ.get(env):
      return os.environ[env]
  roots = []
  try:
    roots.append(subprocess.run(["rocm-sdk", "path", "--root"], capture_output=True, text=True,
                                check=True).stdout.strip())
  except (OSError, subprocess.CalledProcessError):
    pass
  roots += [os.environ.get("ROCM_PATH", ""), "/opt/rocm"]
  for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") + [os.path.join(r, "lib") for r in roots if r]:
    path = os.path.join(d, "libhipblaslt.so.1")
    if d and os.path.exists(path):
      return path
  return ctypes.util.find_library("hipblaslt")


def probe() -> dict:
  """The hipBLASLt library this process loads, its version and git revision, and the GPU."""
  info = {"library": _library_path()}
  try:
    lib = ctypes.CDLL(info["library"] or "libhipblaslt.so.1")
    handle = ctypes.c_void_p()
    if lib.hipblasLtCreate(ctypes.byref(handle)) == 0:
      rev, ver = ctypes.create_string_buffer(256), ctypes.c_int()
      lib.hipblasLtGetGitRevision(handle, rev)
      lib.hipblasLtGetVersion(handle, ctypes.byref(ver))
      info.update(git_revision=rev.value.decode(), version=ver.value)
      lib.hipblasLtDestroy(handle)
  except OSError as e:
    info["error"] = str(e)
  try:
    asic = subprocess.run(["amd-smi", "static", "--asic", "--limit", "--json"], capture_output=True,
                          text=True, check=True, timeout=60).stdout
    data = json.loads(asic)
    gpus = data if isinstance(data, list) else data.get("gpu_data", [])
    first = gpus[0] if gpus else {}
    a = first.get("asic", {})
    info.update(gpu=a.get("market_name"), gfx=a.get("target_graphics_version"),
                cus=a.get("num_compute_units"), gpus=len(gpus))
    limit = first.get("limit", {})
    power = limit.get("max_power") or limit.get("socket_power")
    if isinstance(power, dict):
      power = f"{power.get('value')} {power.get('unit', '')}".strip()
    info["power_cap"] = power
  except (OSError, subprocess.SubprocessError, ValueError, IndexError):
    pass
  return info


def _smi_samples() -> list[dict]:
  """Per GPU: socket power (W) and the mean of the XCD engine clocks (MHz), from amd-smi."""
  try:
    text = subprocess.run(["amd-smi", "metric", "--clock", "--power", "--json"], capture_output=True,
                          text=True, check=True, timeout=30).stdout
    data = json.loads(text)
  except (OSError, subprocess.SubprocessError, ValueError):
    return []
  gpus = data if isinstance(data, list) else data.get("gpu_data", [])
  out = []
  for g in gpus:
    clocks = [c["clk"]["value"] for name, c in g.get("clock", {}).items()
              if name.startswith("gfx_") and isinstance(c, dict) and isinstance(c.get("clk", {}).get("value"), (int, float))]
    power = g.get("power", {}).get("socket_power", {}).get("value")
    out.append({"gpu": g.get("gpu"), "power_w": power if isinstance(power, (int, float)) else None,
                "clock_mhz": statistics.mean(clocks) if clocks else None})
  return out


def sustained_clock(sweep_bin: str, gpu=0, seconds: float = 20, size: int = 8192) -> dict:
  """The engine clock and power a GPU holds on a long bf16 GEMM (its power-limited clock).

  Samples every GPU and reads the busiest, so HIP and amd-smi device order need not agree.
  """
  import time  # pylint: disable=g-import-not-at-top
  cmd = [sweep_bin, "time", str(size), str(size), str(size), "N", "N", "--ws", str(32 * 2**20),
         "--rounds", str(max(1, int(seconds))), "--target_ms", "1000", "--warmup_ms", "0", "-1"]
  proc = subprocess.Popen(cmd, env=_gpu_env(gpu), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  start, samples = time.time(), []
  while proc.poll() is None:
    samples.append((time.time() - start, _smi_samples()))
    time.sleep(0.5)
  stdout = proc.stdout.read()
  rows = [r for r in _json_lines(stdout) if r.get("kind") == "time"]
  steady = [s for t, s in samples if t > 3 and s]
  if not steady:
    return {"ok": False, "stderr": proc.stderr.read()[-2000:]}
  busiest = max(range(len(steady[0])), key=lambda i: statistics.mean(
      s[i]["power_w"] or 0 for s in steady if i < len(s)))
  clocks = [s[busiest]["clock_mhz"] for s in steady if s[busiest]["clock_mhz"]]
  power = [s[busiest]["power_w"] for s in steady if s[busiest]["power_w"]]
  return {"ok": True, "smi_gpu": steady[0][busiest]["gpu"], "samples": len(clocks),
          "clock_mhz_median": statistics.median(clocks), "clock_mhz_min": min(clocks),
          "power_w_median": statistics.median(power), "power_w_max": max(power),
          "tflops_median": statistics.median(r["tflops"] for r in rows) if rows else None,
          "gemm": f"{size}x{size}x{size} NN bf16"}


# Command line.


def _load(path) -> dict:
  return json.loads(pathlib.Path(path).read_text())


def _dump(obj, path):
  text = json.dumps(obj, indent=1) + "\n"
  if path:
    pathlib.Path(path).write_text(text)
  return text


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)

  p = sub.add_parser("probe", help="The hipBLASLt library, its git revision and the GPU.")
  p.add_argument("--json", default="")

  p = sub.add_parser("capture", help="Unique problems in HIPBLASLT_LOG_MASK=32 or 64 logs.")
  p.add_argument("logs", nargs="+")
  p.add_argument("--json", default="")

  p = sub.add_parser("join", help="Attach gemms.py --json sites to captured problems.")
  p.add_argument("problems")
  p.add_argument("gemms_json")
  p.add_argument("--steps", type=int, default=0, help="Steps the capture run logged.")
  p.add_argument("--json", default="")

  p = sub.add_parser("sweep", help="Tune captured problems (stage 1 and stage 2).")
  p.add_argument("problems")
  p.add_argument("--out", required=True)
  p.add_argument("--bench", default="hipblaslt-bench")
  p.add_argument("--sweep_bin", default="hipblaslt_sweep")
  p.add_argument("--gpus", default="", help="Comma-separated HIP device indices, one problem each.")
  p.add_argument("--workspace", type=int, default=0,
                 help="Bytes; default: the smallest workspace XLA gave the problem's sites.")
  p.add_argument("--stage", default="both", choices=["1", "2", "both"])
  p.add_argument("--stage1", default="screen", choices=["screen", "bench"],
                 help="screen: the sweep binary times every supported solution once; bench: "
                 "hipblaslt-bench tuning mode (HIPBLASLT_TUNING_FILE), the hipBLASLt team's method.")
  p.add_argument("--screen_ms", type=float, default=20.0, help="Timed work per solution when screening.")
  p.add_argument("--iters", type=int, default=0, help="hipblaslt-bench -i in tuning mode (default 1000).")
  p.add_argument("--cold_iters", type=int, default=0, help="hipblaslt-bench -j (default 1000).")
  p.add_argument("--top", type=int, default=5, help="Stage-1 runners-up that become finalists.")
  p.add_argument("--seeds", type=json.loads, default=None,
                 help='JSON {problem id: [solution index, ...]} added to the finalists.')
  p.add_argument("--trials", type=int, default=5, help="Stage-2 processes per problem.")
  p.add_argument("--rounds", type=int, default=5, help="Interleaved rounds per process.")
  p.add_argument("--target_ms", type=float, default=100.0, help="Timed work per candidate per round.")
  p.add_argument("--warmup_ms", type=float, default=1000.0, help="Work before the first round.")
  p.add_argument("--min_gain", type=float, default=0.02)
  p.add_argument("--only", default="", help="Regex on problem ids.")
  p.add_argument("--no_validate", action="store_true")

  p = sub.add_parser("write", help="An override file from sweep results.")
  p.add_argument("results")
  p.add_argument("--out", required=True)
  p.add_argument("--git_version", default="", help="Default: the probe's or stage 1's git revision.")
  p.add_argument("--kernel_names", action="store_true",
                 help="Add a kernel_name column, which newer hipBLASLt validates rows by.")

  p = sub.add_parser("verify", help="Whether a run used the override's solutions.")
  p.add_argument("logs", nargs="*")
  p.add_argument("--override", required=True)
  p.add_argument("--api", action="store_true", help="Also check heuristic #0 through the API.")
  p.add_argument("--sweep_bin", default="hipblaslt_sweep")
  p.add_argument("--json", default="")

  p = sub.add_parser("instep", help="Each row's in-step launches without and with the override.")
  p.add_argument("base_gemms", help="gemms.py --json --times of a trace without the override.")
  p.add_argument("tuned_gemms", help="The same with it.")
  p.add_argument("--override", required=True)
  p.add_argument("--out", default="", help="Write the rows that are not slower in the step here.")
  p.add_argument("--json", default="")

  p = sub.add_parser("map", help="Carry winners to the hipBLASLt build this process loads.")
  p.add_argument("results")
  p.add_argument("--sweep_bin", default="hipblaslt_sweep")
  p.add_argument("--listings", default="", help="A sweep --out directory of the target build.")
  p.add_argument("--json", default="")

  p = sub.add_parser("report", help="A Markdown table of sweep results.")
  p.add_argument("results")

  p = sub.add_parser("clock", help="The clock and power a GPU sustains on a long bf16 GEMM.")
  p.add_argument("--sweep_bin", default="hipblaslt_sweep")
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--seconds", type=float, default=20)
  p.add_argument("--json", default="")

  args = parser.parse_args(argv)
  if args.command == "probe":
    print(_dump(probe(), args.json), end="")
  elif args.command == "capture":
    out = capture(args.logs)
    _dump(out, args.json)
    print(f"{len(out['problems'])} problems from {out['lines']} logged calls in {len(out['logs'])} files")
    for p in out["problems"][:40]:
      print(f"  {p['calls']:>6} calls  {p['id']}  solutions {p['solutions']}"
            + (f"  ({p['variants']} descriptors)" if p["variants"] > 1 else ""))
  elif args.command == "join":
    out = join(_load(args.problems), _load(args.gemms_json), args.steps)
    _dump(out, args.json or args.problems)
    for p in out["problems"]:
      print(f"  {p['id']}: {p['launches_per_step'] or 0} launches per step, workspace "
            f"{p['workspace_bytes']}, {len(p['sites'])} sites")
    if out["unmatched_sites"]:
      print(f"{len(out['unmatched_sites'])} hipBLASLt sites match no captured problem:")
      for s in out["unmatched_sites"][:20]:
        print(f"  {s['module']} {s['kernel']} {s['key']} x{s['count']} {s['source']}")
    if out["problems_without_sites"]:
      print(f"{len(out['problems_without_sites'])} captured problems match no site: "
            + "; ".join(out["problems_without_sites"][:10]))
  elif args.command == "sweep":
    args.probe_info = probe()
    sweep(_load(args.problems), args)
  elif args.command == "write":
    results = _load(args.results)
    rows = rows_from_results(results)
    git = args.git_version or (results.get("probe") or {}).get("git_revision") or next(
        (r["stage1"].get("git_version") for r in results["results"] if r.get("stage1", {}).get("git_version")),
        "")
    if not git:
      print("no git version: pass --git_version (hipblasLtGetGitRevision of the target library)")
      return 1
    n = write_override(rows, git, args.out, args.kernel_names)
    print(f"wrote {n} rows for hipBLASLt {git} to {args.out}")
  elif args.command == "verify":
    out = {"log": verify(args.logs, args.override) if args.logs else None}
    if args.api:
      out["api"] = verify_api(args.override, args.sweep_bin)
    print(_dump(out, args.json), end="")
  elif args.command == "instep":
    out = instep(_load(args.base_gemms), _load(args.tuned_gemms), args.override)
    if args.json:
      _dump(out, args.json)
    print(instep_table(out))
    if args.out:
      n = write_override(out["kept"], out["git_version"], args.out)
      print(f"kept {n} of {len(out['rows'])} rows in {args.out}")
  elif args.command == "map":
    print(_dump(map_results(_load(args.results), args.sweep_bin, listings_dir=args.listings or None),
                args.json), end="")
  elif args.command == "report":
    print(report(_load(args.results)))
  elif args.command == "clock":
    print(_dump(sustained_clock(args.sweep_bin, args.gpu, args.seconds), args.json), end="")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
