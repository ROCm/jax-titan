"""New hipBLASLt kernels for a program's GEMM problems, generated with TensileLite.

`hipblaslt.py` picks the best of the kernels a hipBLASLt build ships. This
tool generates new ones for the captured problems, with hipBLASLt's own flow
(projects/hipblaslt/tensilelite/Tensile/Utilities/tensile_generator):

  python3 tensilelite.py config LOG... --src HIPBLASLT --out DIR [--only REGEX] [--gpus 8]
      [--iters 100] [--fast] [--cu 304] [--xcc 8]
  python3 tensilelite.py tune DIR --src HIPBLASLT [--gpus 0,1,...]
  python3 tensilelite.py decode DIR [--stock SWEEP/results.json] [--json OUT]
  python3 tensilelite.py merge DIR --src HIPBLASLT
  python3 tensilelite.py library DIR --src HIPBLASLT [--jobs N]

`config` keeps the mask-32 log lines of the problems to tune (by `--only`, a
regex on hipblaslt.py's problem ids), drops the fields hipBLASLt 1.4 logs that
tensile_config_generator.py's pattern predates, and runs the generator: one
Tensile config per GPU. `tune` runs Tensile on each config on its own GPU.
`decode` reads, per problem size, the best kernel Tensile found from its
library logic and sets it beside the stock solutions `hipblaslt.py sweep`
timed. Tensile times with its own client, so treat its numbers as a ranking.
`merge` merges the tuned logic into a copy of the build's gfx942 logic with
merge.py, and `library` builds the kernel library from that copy with
TensileCreateLibrary, with the options hipBLASLt's build uses. With
HIPBLASLT_TENSILE_LIBPATH naming the result, the stock libhipblaslt serves
problems from it, and `hipblaslt.py sweep` and `verify` and a step A/B measure
the new kernels the way they measure stock ones.

HIPBLASLT is a hipBLASLt source tree (a rocm-libraries checkout, or its
projects/hipblaslt) at the revision of the library XLA loads, so the library
matches it, with TensileLite set up as its README says (`invoke rocisa`,
`invoke build-client`). Run this with that Python: it needs PyYAML.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hipblaslt  # noqa: E402  pylint: disable=g-import-not-at-top

_NEWER_FIELDS = re.compile(r" --(alphai|betai|batch_mode) \S+")
_TYPES = {"B": "bf16_r", "H": "f16_r", "S": "f32_r", "X": "xf32_r"}
_TYPE_IDS = {7: "bf16_r", 4: "f16_r", 0: "f32_r"}
LOGIC = "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"


def hipblaslt_dir(src) -> pathlib.Path:
  src = pathlib.Path(src)
  return src / "projects" / "hipblaslt" if (src / "projects" / "hipblaslt").is_dir() else src


def tensilelite_dir(src) -> pathlib.Path:
  return hipblaslt_dir(src) / "tensilelite"


def select_lines(logs, only: str = "") -> list[str]:
  """One normalized mask-32 line per problem whose id matches `only`."""
  seen, out = set(), []
  for pattern in logs:
    for path in sorted(glob.glob(pattern)) or [pattern]:
      with open(path, errors="replace") as f:
        for line in f:
          args = hipblaslt.parse_bench_line(line)
          if not args:
            continue
          pid = hipblaslt.key_text(hipblaslt.key_of(args))
          if pid in seen or (only and not re.search(only, pid)):
            continue
          seen.add(pid)
          out.append(_NEWER_FIELDS.sub("", line.strip()))
  return out


def config(logs, src, out_dir, only="", gpus=8, iters=100, fast=False, cu=304, xcc=8) -> list[str]:
  """Tensile configs for the selected problems, one per GPU, in OUT/configs."""
  out = pathlib.Path(out_dir)
  configs = out / "configs"
  configs.mkdir(parents=True, exist_ok=True)
  lines = select_lines(logs, only)
  if not lines:
    raise ValueError("no logged problem matches --only")
  (out / "selected.log").write_text("\n".join(lines) + "\n")
  generator = tensilelite_dir(src) / "Tensile" / "Utilities" / "tensile_generator"
  shutil.copy(generator / "tuning_template.yaml", configs / "tuning_template.yaml")
  cmd = [sys.executable, str(generator / "tensile_config_generator.py"),
         "--hipblaslt_log", str(out / "selected.log"), "--tensile_config", "tuning_template.yaml",
         "--gpus", str(min(gpus, len(lines))), "--iters", str(iters)]
  if fast:
    cmd += ["--fast", "1"]  # the generator's flags are type=bool: any value turns them on
  env = dict(os.environ, GPU_TARGET="gfx942", CU=str(cu), XCC=str(xcc))
  subprocess.run(cmd, cwd=configs, env=env, check=True)
  betas = {}
  for line in lines:
    a = hipblaslt.parse_bench_line(line)
    betas[(a["m"], a["n"], a["batch_count"], a["k"])] = float(a.get("beta") or 0)
  paths = sorted(configs.glob("tuning_template.*.yaml"))
  for path in paths:
    _fix_types(path, betas)
  return [str(p) for p in paths]


def _fix_types(path, betas=None) -> None:
  """Where the generator and Tensile's config validation disagree: transposes as 0 and 1 instead
  of booleans, and the architecture as {Architecture, CUCount} instead of its name. A config
  whose problems were called with beta 1 (XLA's residual adds) is timed with beta 1, which
  reads C; the generator's template initializes beta to 0."""
  import yaml  # pylint: disable=g-import-not-at-top
  doc = yaml.safe_load(pathlib.Path(path).read_text())
  adds_c = False
  for problem in doc.get("BenchmarkProblems", []):
    problem_type = problem[0]
    for flag in ("TransposeA", "TransposeB"):
      if flag in problem_type:
        problem_type[flag] = bool(problem_type[flag])
    for step in problem[1:]:
      for final in (step or {}).get("BenchmarkFinalParameters") or []:
        for size in (final.get("ProblemSizes") or []):
          exact = size.get("Exact") if isinstance(size, dict) else None
          if exact and (betas or {}).get(tuple(exact[:4])):
            adds_c = True
  if adds_c:
    doc.setdefault("GlobalParameters", {})["DataInitTypeBeta"] = 1
  logic = doc.get("LibraryLogic", {})
  if isinstance(logic.get("ArchitectureName"), dict):
    logic["ArchitectureName"] = logic["ArchitectureName"]["Architecture"]
  pathlib.Path(path).write_text(yaml.dump(doc, default_flow_style=None))


def tune(out_dir, src, gpus=(0,)) -> dict:
  """Runs Tensile on each config on its own GPU; {config: exit code}."""
  out = pathlib.Path(out_dir)
  tensile = tensilelite_dir(src) / "Tensile" / "bin" / "Tensile"
  configs = sorted((out / "configs").glob("tuning_template.*.yaml"))

  def run(i, cfg):
    dest = out / "tune" / cfg.stem.split(".")[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpus[i % len(gpus)]), HIP_FORCE_DEV_KERNARG="1")
    with open(f"{dest}.log", "w") as log:
      return cfg.name, subprocess.run([sys.executable, str(tensile), str(cfg), str(dest)], env=env,
                                      stdout=log, stderr=subprocess.STDOUT, check=False).returncode

  with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
    return dict(pool.map(lambda a: run(*a), enumerate(configs)))


def _key(problem_type: dict, size: list) -> str:
  """hipblaslt.py's problem id of a Tensile size [m, n, batch, k, ...] under a problem type."""
  types = problem_type.get("DataType"), problem_type.get("DestDataType"), problem_type.get("ComputeDataType")
  a, c, compute = (_TYPE_IDS.get(t, _TYPES.get(t, str(t))) for t in types)
  if problem_type.get("F32XdlMathOp") in (11, "X"):
    compute = "xf32_r"
  ta = "T" if problem_type.get("TransposeA") else "N"
  tb = "T" if problem_type.get("TransposeB") else "N"
  return hipblaslt.key_text((ta, tb, size[2], size[0], size[1], size[3], a, a, c, compute))


def read_logic(path) -> list[dict]:
  """Per exact size of a Tensile library logic file: problem id, solution and its perf value.

  Tensile writes a mapping; the logic files hipBLASLt ships are the older list
  of the same parts. The perf value is the file's PerfMetric (DeviceEfficiency:
  GFLOP/s per MHz of peak clock), not GFLOP/s.
  """
  import yaml  # pylint: disable=g-import-not-at-top
  doc = yaml.safe_load(pathlib.Path(path).read_text())
  if isinstance(doc, dict):
    problem_type, solutions, exact = doc["ProblemType"], doc["Solutions"], doc["ExactLogic"] or []
  else:
    problem_type = next(d for d in doc if isinstance(d, dict) and "OperationType" in d)
    solutions = next(d for d in doc if isinstance(d, list) and d and isinstance(d[0], dict)
                     and "SolutionIndex" in d[0])
    exact = next(d for d in reversed(doc) if isinstance(d, list) and d and isinstance(d[0], list)
                 and len(d[0]) == 2 and isinstance(d[0][0], list))
  by_index = {s["SolutionIndex"]: s for s in solutions}
  out = []
  for size, (index, perf) in exact:
    s = by_index.get(index, {})
    out.append({"id": _key(problem_type, size), "size": size, "perf": perf,
                "solution": s.get("SolutionNameMin") or s.get("SolutionName"),
                "macro_tile": [s.get("MacroTile0"), s.get("MacroTile1"), s.get("DepthU")]})
  return out


def read_benchmark(path) -> dict:
  """{(m, n, batch, k): (GFLOP/s, solution name)}: the fastest kernel per size in a Tensile
  2_BenchmarkData CSV, one column per kernel, as Tensile's client timed it."""
  import csv  # pylint: disable=g-import-not-at-top
  with open(path) as f:
    rows = [[c.strip() for c in row] for row in csv.reader(f)]
  header = rows[0]
  first = header.index("GbpsBW") + 1
  sizes = [header.index(c) for c in ("SizeI", "SizeJ", "SizeK", "SizeL")]
  out = {}
  for row in rows[1:]:
    timed = []
    for i in range(first, len(header)):
      try:
        timed.append((float(row[i]), header[i]))
      except (ValueError, IndexError):
        continue
    if timed:
      out[tuple(int(row[i]) for i in sizes)] = max(timed)
  return out


def decode(out_dir, stock=None) -> list[dict]:
  """Tensile's best kernel per tuned size, against the stock solutions the sweep timed."""
  best = {}
  for logic in sorted(pathlib.Path(out_dir).glob("tune/*/3_LibraryLogic/*.yaml")):
    timed = {}
    for csv_path in logic.parent.parent.glob("2_BenchmarkData/*.csv"):
      timed.update(read_benchmark(csv_path))
    for rec in read_logic(logic):
      gflops = timed.get(tuple(rec["size"][:4]), (None,))[0]
      if rec["id"] not in best or (gflops or 0) > (best[rec["id"]]["gflops"] or 0):
        best[rec["id"]] = dict(rec, gflops=gflops, logic=str(logic))
  stock_by_id = {r["id"]: r for r in (stock or {}).get("results", [])}
  rows = []
  for pid, rec in best.items():
    flops = hipblaslt.flops_of(hipblaslt.key_of(dict(zip(hipblaslt.KEY, _typed(pid)))))
    row = dict(rec, tensile_us=flops / (rec["gflops"] * 1e3) if rec["gflops"] else None)
    s = stock_by_id.get(pid, {}).get("stage2", {})
    if s:
      row["stock_default_us"] = s["default_ms"] * 1e3
      row["stock_best_us"] = (s.get("candidate_ms") or s["default_ms"]) * 1e3
    rows.append(row)
  return rows


def _typed(pid: str) -> list:
  v = pid.split(",")
  return v[:2] + [int(x) for x in v[2:6]] + v[6:]


def decode_table(rows) -> str:
  """One row per problem; the full kernel names are in decode's JSON."""
  lines = ["| problem | Tensile best kernel | Tensile us | stock default us | stock best us |",
           "|---|---|---|---|---|"]
  for r in rows:
    fmt = lambda v: f"{v:.1f}" if v else ""  # noqa: E731
    short = re.search(r"MT\d+x\d+x\d+_MI\d+x\d+x\d+", r["solution"] or "")
    lines.append(f"| {r['id']} | {short.group(0) if short else r['solution']} | {fmt(r['tensile_us'])} | "
                 f"{fmt(r.get('stock_default_us'))} | {fmt(r.get('stock_best_us'))} |")
  return "\n".join(lines)


def merge(out_dir, src, schedule="aquavanjaram", arch_dir="gfx942") -> pathlib.Path:
  """The build's logic for `schedule` with the tuned Equality logic merged in, under OUT/logic."""
  out = pathlib.Path(out_dir)
  stock = hipblaslt_dir(src) / LOGIC / schedule
  dest = out / "logic" / "asm_full" / schedule
  if dest.exists():
    shutil.rmtree(dest)
  shutil.copytree(stock, dest)
  tuned = out / "tuned_logic"
  tuned.mkdir(parents=True, exist_ok=True)
  for path in out.glob("tune/*/3_LibraryLogic/*.yaml"):
    shutil.copy(path, tuned / f"{path.parent.parent.name}_{path.name}")
  equality = dest / arch_dir / "Equality"
  subprocess.run([sys.executable, str(tensilelite_dir(src) / "Tensile" / "Utilities" / "merge.py"),
                  "--no_eff", str(equality), str(tuned), str(equality)], check=True)
  return equality


def _rocm_root() -> pathlib.Path:
  if os.environ.get("ROCM_PATH"):
    return pathlib.Path(os.environ["ROCM_PATH"])
  out = subprocess.run(["rocm-sdk", "path", "--root"], capture_output=True, text=True, check=True)
  return pathlib.Path(out.stdout.strip())


def library(out_dir, src, jobs=0, arch="gfx942") -> pathlib.Path:
  """TensileCreateLibrary over OUT/logic, as hipBLASLt's build calls it; returns the path for
  HIPBLASLT_TENSILE_LIBPATH."""
  out = pathlib.Path(out_dir)
  rocm = _rocm_root()
  dest = out / "library"
  cmd = [sys.executable, "-m", "Tensile.TensileCreateLibrary", f"--architecture={arch}",
         f"--cxx-compiler={rocm / 'bin' / 'amdclang++'}",
         f"--offload-bundler={rocm / 'llvm' / 'bin' / 'clang-offload-bundler'}",
         f"--jobs={jobs or os.cpu_count()}", "--disable-asm-comments",
         str(out / "logic" / "asm_full"), str(dest), "HIP"]
  env = dict(os.environ, PYTHONPATH=f"{tensilelite_dir(src)}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
  subprocess.run(cmd, env=env, check=True)
  return dest / "library"


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)
  p = sub.add_parser("config", help="Tensile configs for captured problems, one per GPU.")
  p.add_argument("logs", nargs="+")
  p.add_argument("--src", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--only", default="", help="Regex on problem ids.")
  p.add_argument("--gpus", type=int, default=8)
  p.add_argument("--iters", type=int, default=100)
  p.add_argument("--fast", action="store_true", help="Smallest tiles of each matrix instruction only.")
  p.add_argument("--cu", type=int, default=304)
  p.add_argument("--xcc", type=int, default=8)
  p = sub.add_parser("tune", help="Tensile on each config, one GPU each.")
  p.add_argument("out")
  p.add_argument("--src", required=True)
  p.add_argument("--gpus", default="0", help="Comma-separated HIP device indices.")
  p = sub.add_parser("decode", help="Best tuned kernel per size against the stock solutions.")
  p.add_argument("out")
  p.add_argument("--stock", default="", help="hipblaslt.py sweep's results.json.")
  p.add_argument("--json", default="")
  p = sub.add_parser("merge", help="Merge tuned logic into a copy of the build's gfx942 logic.")
  p.add_argument("out")
  p.add_argument("--src", required=True)
  p = sub.add_parser("library", help="Build the kernel library from the merged logic.")
  p.add_argument("out")
  p.add_argument("--src", required=True)
  p.add_argument("--jobs", type=int, default=0)
  args = parser.parse_args(argv)
  if args.command == "config":
    for path in config(args.logs, args.src, args.out, args.only, args.gpus, args.iters, args.fast,
                       args.cu, args.xcc):
      print(path)
  elif args.command == "tune":
    print(json.dumps(tune(args.out, args.src, [int(g) for g in args.gpus.split(",")]), indent=1))
  elif args.command == "decode":
    rows = decode(args.out, json.loads(pathlib.Path(args.stock).read_text()) if args.stock else None)
    if args.json:
      pathlib.Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
    print(decode_table(rows))
  elif args.command == "merge":
    print(merge(args.out, args.src))
  elif args.command == "library":
    path = library(args.out, args.src, args.jobs)
    print(f"HIPBLASLT_TENSILE_LIBPATH={path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
