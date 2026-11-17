"""Per-launch hardware counters of GPU kernels, from rocprofv3.

  python3 counters.py run OUT_DIR [--kernels REGEX] [--counters "A B ..."] -- CMD...
  python3 counters.py report OUT_DIR [--kernels REGEX] [--min_us US] [--by name|grid] [--json OUT]

`run` runs CMD under rocprofv3, collecting the counters of every launch whose kernel name matches
`--kernels` (default all), and leaves rocprofv3's CSV files in OUT_DIR. `report` reads them and gives,
per launch and as medians per kernel:

  clock_mhz    GRBM_GUI_ACTIVE / chiplets / duration: the engine clock the launch averaged. The
               counter ticks once per engine cycle while the engine is busy, summed over chiplets,
               so it reads low when chiplets go idle before the launch ends.
  matrix_busy  SQ_VALU_MFMA_BUSY_CYCLES / GRBM_GUI_ACTIVE / SIMDs per chiplet: the share of matrix
               (MFMA) units busy on an average cycle.
  l2_hit       TCC_HIT / (TCC_HIT + TCC_MISS).
  read_mb, write_mb
               Requests the L2 sent on (TCC_EA0_RDREQ, TCC_EA0_WRREQ) times 64 bytes, less 32 for
               each TCC_EA0_RDREQ_32B when collected: traffic past L2. A memory-side cache (MI300's
               Infinity Cache) can serve part of it before HBM.

Chiplets (Num_Xcc) and SIMDs per compute unit come from rocprofv3's agent_info.csv. The default
counters fit one pass on gfx942. Counting serializes the counted launches, so nothing runs beside
them and the program's timing changes; compare launches counted the same way.

Standard library only: run it as a script, without JAX or an install.
"""

import argparse
import collections
import csv
import json
import pathlib
import re
import statistics
import subprocess
import sys

DEFAULT_COUNTERS = ("GRBM_GUI_ACTIVE", "SQ_BUSY_CYCLES", "SQ_VALU_MFMA_BUSY_CYCLES", "SQ_WAVES",
                    "TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum")
_OUTPUT = "counters"


def command(out_dir, cmd, kernels="", counters=DEFAULT_COUNTERS, rocprof="rocprofv3") -> list[str]:
  """The rocprofv3 command line that runs CMD with dispatch counters into OUT_DIR."""
  args = [rocprof, "--pmc", *counters, "--kernel-trace", "-d", str(out_dir), "-o", _OUTPUT,
          "--output-format", "csv"]
  if kernels:
    args += ["--kernel-include-regex", kernels]
  return args + ["--", *cmd]


def run(out_dir, cmd, kernels="", counters=DEFAULT_COUNTERS, rocprof="rocprofv3") -> int:
  pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
  return subprocess.run(command(out_dir, cmd, kernels, counters, rocprof), check=False).returncode


def read_agents(path) -> dict:
  """{"Agent N": {"chiplets", "simds_per_chiplet", "max_clock_mhz", "name"}} for the GPU agents."""
  agents = {}
  for row in csv.DictReader(open(path)):
    if row.get("Agent_Type") != "GPU":
      continue
    chiplets = max(1, int(row.get("Num_Xcc") or 1))
    simds = int(row.get("Cu_Count") or 0) * int(row.get("Simd_Per_Cu") or 0)
    agents[f"Agent {row['Logical_Node_Id']}"] = {
        "chiplets": chiplets, "simds_per_chiplet": simds // chiplets,
        "max_clock_mhz": int(row.get("Max_Engine_Clk_Fcompute") or 0), "name": row.get("Product_Name", "")}
  return agents


def read_launches(path) -> list[dict]:
  """One record per dispatch of a counter_collection.csv: kernel, agent, grid, times and counters."""
  launches = collections.OrderedDict()
  for row in csv.DictReader(open(path)):
    key = (row.get("Process_Id"), row["Dispatch_Id"])
    d = launches.setdefault(key, {"kernel": row["Kernel_Name"], "agent": row["Agent_Id"],
                                  "grid": int(row["Grid_Size"]), "start_ns": int(row["Start_Timestamp"]),
                                  "end_ns": int(row["End_Timestamp"]), "counters": {}})
    c = d["counters"]
    c[row["Counter_Name"]] = c.get(row["Counter_Name"], 0.0) + float(row["Counter_Value"])
  return sorted(launches.values(), key=lambda d: d["start_ns"])


def metrics(launch: dict, agent: dict) -> dict:
  """The derived quantities of one launch (None where a counter is missing)."""
  c, us = launch["counters"], (launch["end_ns"] - launch["start_ns"]) / 1e3
  cycles = c.get("GRBM_GUI_ACTIVE")
  out = {"duration_us": us, "clock_mhz": None, "matrix_busy": None, "l2_hit": None,
         "read_mb": None, "write_mb": None}
  if cycles and us > 0:
    out["clock_mhz"] = cycles / agent["chiplets"] / us
    if "SQ_VALU_MFMA_BUSY_CYCLES" in c and agent["simds_per_chiplet"]:
      out["matrix_busy"] = c["SQ_VALU_MFMA_BUSY_CYCLES"] / cycles / agent["simds_per_chiplet"]
  hits, misses = c.get("TCC_HIT_sum"), c.get("TCC_MISS_sum")
  if hits is not None and misses is not None and hits + misses:
    out["l2_hit"] = hits / (hits + misses)
  if "TCC_EA0_RDREQ_sum" in c:
    out["read_mb"] = (64 * c["TCC_EA0_RDREQ_sum"] - 32 * c.get("TCC_EA0_RDREQ_32B_sum", 0)) / 1e6
  if "TCC_EA0_WRREQ_sum" in c:
    out["write_mb"] = 64 * c["TCC_EA0_WRREQ_sum"] / 1e6
  return out


def _files(path, suffix) -> list[pathlib.Path]:
  p = pathlib.Path(path)
  return [p] if p.is_file() else sorted(p.rglob(f"*{suffix}"))


def report(path, kernels="", min_us=0.0, by="name") -> dict:
  """Per kernel (or kernel and grid size): launch count, total time and the medians of `metrics`."""
  agents = {}
  for f in _files(path, "agent_info.csv"):
    agents.update(read_agents(f))
  pattern = re.compile(kernels) if kernels else None
  groups = collections.defaultdict(list)
  for f in _files(path, "counter_collection.csv"):
    for launch in read_launches(f):
      if pattern and not pattern.search(launch["kernel"]):
        continue
      agent = agents.get(launch["agent"], {"chiplets": 1, "simds_per_chiplet": 0})
      m = metrics(launch, agent)
      if m["duration_us"] < min_us:
        continue
      key = launch["kernel"] if by == "name" else f"{launch['kernel']} grid={launch['grid']}"
      groups[key].append(m)
  rows = []
  for key, ms in groups.items():
    row = {"kernel": key, "launches": len(ms), "total_ms": sum(m["duration_us"] for m in ms) / 1e3}
    for name in ("duration_us", "clock_mhz", "matrix_busy", "l2_hit", "read_mb", "write_mb"):
      values = [m[name] for m in ms if m[name] is not None]
      row[name] = statistics.median(values) if values else None
    clocks = sorted(m["clock_mhz"] for m in ms if m["clock_mhz"] is not None)
    row["clock_mhz_range"] = [clocks[0], clocks[-1]] if clocks else None
    rows.append(row)
  rows.sort(key=lambda r: -r["total_ms"])
  return {"agents": agents, "kernels": rows}


def table(result: dict, width: int = 60) -> str:
  fmt = lambda v, spec: format(v, spec) if v is not None else "-"
  lines = ["| kernel | launches | total ms | median us | clock MHz (range) | matrix busy | L2 hit | read MB | write MB |",
           "|---|---|---|---|---|---|---|---|---|"]
  for r in result["kernels"]:
    name = r["kernel"] if len(r["kernel"]) <= width else r["kernel"][:width - 3] + "..."
    rng = r["clock_mhz_range"]
    clock = f"{fmt(r['clock_mhz'], '.0f')} ({rng[0]:.0f}-{rng[1]:.0f})" if rng else "-"
    lines.append(f"| {name} | {r['launches']} | {r['total_ms']:.1f} | {fmt(r['duration_us'], '.1f')} | {clock} | "
                 f"{fmt(r['matrix_busy'], '.1%')} | {fmt(r['l2_hit'], '.1%')} | {fmt(r['read_mb'], '.1f')} | "
                 f"{fmt(r['write_mb'], '.1f')} |")
  return "\n".join(lines)


def split_command(argv) -> tuple[list[str], list[str]]:
  """This tool's arguments, and the command after the first `--`."""
  argv = list(argv)
  if "--" not in argv:
    return argv, []
  i = argv.index("--")
  return argv[:i], argv[i + 1:]


def main(argv=None) -> int:
  argv, cmd = split_command(sys.argv[1:] if argv is None else argv)
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)
  p = sub.add_parser("run", help="Run a command (after --) under rocprofv3 with dispatch counters.")
  p.add_argument("out_dir")
  p.add_argument("--kernels", default="", help="Regex on kernel names to count (default all).")
  p.add_argument("--counters", default=" ".join(DEFAULT_COUNTERS))
  p.add_argument("--rocprof", default="rocprofv3")
  p = sub.add_parser("report", help="Per-launch clock, matrix-unit use, L2 hit rate and traffic past L2.")
  p.add_argument("path", help="A rocprofv3 output directory or counter_collection.csv.")
  p.add_argument("--kernels", default="", help="Regex on kernel names.")
  p.add_argument("--min_us", type=float, default=0.0, help="Drop shorter launches.")
  p.add_argument("--by", choices=["name", "grid"], default="name", help="Group by kernel, or kernel and grid size.")
  p.add_argument("--json", default="")
  args = parser.parse_args(argv)
  if args.command == "run":
    if not cmd:
      parser.error("run needs -- CMD...")
    return run(args.out_dir, cmd, args.kernels, args.counters.split(), args.rocprof)
  result = report(args.path, args.kernels, args.min_us, args.by)
  print(table(result))
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(result, indent=1))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
