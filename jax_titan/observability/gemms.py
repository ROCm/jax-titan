"""GEMMs in the final compiled program, from an XLA HLO dump.

  JAX_ENABLE_COMPILATION_CACHE=false \\
  XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS --xla_dump_to=DUMP --xla_dump_hlo_as_text" \\
      python -m jax_titan.trainer.train ... --steps=2 --run_dir=RUN
  python3 gemms.py DUMP [--run_dir RUN | --calls FILE] [--merge]
      [--module REGEX] [--sites] [--csv OUT] [--json OUT]
      [--times KERNEL_TIMES | --trace TRACE_DIR [--gpu N]]
      [--steps N] [--overlap]

This file uses only the Python standard library: run it directly, or copy it
to any machine with Python 3.8 or later, for a dump of any XLA program; it
needs neither JAX nor JAX-Titan. Inside JAX-Titan's environment,
`python -m jax_titan.observability.gemms` runs the same code.

XLA writes each compiled module's final program to
`module_*.<platform>_after_optimizations.txt`: the HLO left after all passes,
with every GEMM bound to the kernel the autotuner picked. Only those files are
read, so the candidates the autotuner timed and dropped never show up, nor do
`autotune_results.pbtxt` or the `before_*` dumps. A GEMM is a BLAS custom call
(`__cublas$lt$matmul` runs hipBLASLt on ROCm, `__cublas$gemm` rocBLAS), a
fusion around a `dot` (Triton GEMMs), or a plain `dot`, `ragged-dot` or
`scaled-dot`. Each gets its batch, M, N and K, dtypes and operand layouts as
the kernel computes C[M, N] row major, whichever operand order XLA chose, and
its launches per run of the module: one per trip of a `while` body whose trip
count XLA knows, as in a scan over layers.

SPMD and MPMD programs read the same way: XLA compiles one partitioned module
for a `jax.jit` step and one module per fragment of an MPMD pipeline, and the
shapes are what one device computes. With a JAX-Titan `--run_dir`, or
`--calls` for any program, only the modules a training step runs are kept,
each weighted by its calls per step: the jit module runs once, and a pipeline
runs each stage's forward fragment once per microbatch and a backward fragment
that serves several microbatches once for each.

The `share` column is each shape's share of GEMM FLOPs. `--times` adds
measured time from `profile.py --kernel_times` of a trace of the same
program: each site's kernel launches, joined by HLO module and op, with the
median per launch and its bootstrap interval, the medians of launches no
collective overlapped ("clean") and of those mostly overlapped, and the
milliseconds per traced step (`--steps`). Sites are then ranked by time.
`--trace` reads the trace itself instead: the newest `*.xplane.pb` under
TRACE_DIR, decoded with the standard library, so a dump and a trace are all
this file needs, without JAX. `profile.py` takes its kernel launches and
collectives from here.

`--overlap` adds one row per GEMM shape that collectives overlap: the
launches a collective mostly overlapped, their median against the clean one,
the milliseconds per step its overlapped launches take beyond the clean
median, the collectives beside it (kind, share of the overlapped time and
size), its source, and the collectives it reads or writes in the program:
an all-gather whose result it multiplies, a reduce-scatter its result feeds.
Those are followed through bitcasts, copies, tuples and a loop's carried
values, not through fusions. In a loop that prefetches the next layer's
weights, the collective beside a GEMM is never the launch it reads, and often
another collective.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import pathlib
import random
import re
import statistics
import struct

# Library GEMM custom calls and the ROCm library each runs; XLA keeps the
# targets' cuBLAS-era names on ROCm.
BLAS_TARGETS = {
    "__cublas$gemm": "rocBLAS",
    "__cublas$lt$matmul": "hipBLASLt",
    "__cublas$lt$matmul$f8": "hipBLASLt",
    "__cublas$lt$matmul$mx": "hipBLASLt",
    "__cublas$lt$groupedMatmul": "hipBLASLt",
}
FUSION_KINDS = {
    "__triton_gemm": "Triton",
    "__triton_nested_gemm_fusion": "Triton",
    "__triton": "Triton",
    "__custom_fusion": "custom fusion",
    "__ynn_fusion": "YNNPACK",
}
DOTS = ("dot", "ragged-dot", "scaled-dot")
FILE = re.compile(r"^module_\d+\..+after_optimizations\.txt$")

_DEF = re.compile(r"^\s+(ROOT\s+)?%?([^\s=]+)\s+=\s+")
_HEADER = re.compile(r"^(ENTRY\s+)?%?([^\s(]+)")
_OPCODE = re.compile(r"\s*([\w\-]+)\(")
_ARRAY = re.compile(r"(\w+)\[([^\]]*)\](?:\{([^}]*)\})?")
_META = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
_COMMENT = re.compile(r"/\*.*?\*/")
_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")


def _close(text, i):
  """Index just past the bracket that closes the one at text[i]."""
  depth, j, n = 0, i, len(text)
  while j < n:
    c = text[j]
    if c == '"':
      j += 1
      while j < n and text[j] != '"':
        j += 2 if text[j] == "\\" else 1
    elif c in "([{":
      depth += 1
    elif c in ")]}":
      depth -= 1
      if depth == 0:
        return j + 1
    j += 1
  return n


def _split(text):
  """Split at commas outside brackets and quotes."""
  out, depth, start, j, n = [], 0, 0, 0, len(text)
  while j < n:
    c = text[j]
    if c == '"':
      j += 1
      while j < n and text[j] != '"':
        j += 2 if text[j] == "\\" else 1
    elif c in "([{":
      depth += 1
    elif c in ")]}":
      depth -= 1
    elif c == "," and depth == 0:
      out.append(text[start:j].strip())
      start = j + 1
    j += 1
  if text[start:].strip():
    out.append(text[start:].strip())
  return out


def _array(text):
  """(dtype, dims, minor_to_major or None) of an array shape like bf16[8,128]{1,0}."""
  m = _ARRAY.match(text.strip()) if text else None
  if not m:
    return None
  dims = []
  for d in m.group(2).split(","):
    d = d.strip().lstrip("<=")
    if d:
      dims.append(int(d) if d.isdigit() else None)
  layout = None
  if m.group(3) is not None:
    layout = [int(d) for d in m.group(3).split(":")[0].split(",") if d.strip().isdigit()]
  return m.group(1), dims, layout


def _elements(shape):
  """Element shapes of a tuple shape."""
  shape = _COMMENT.sub("", shape).strip()
  return _split(shape[1:-1]) if shape.startswith("(") else [shape]


def _ints(value):
  return [int(v) for v in re.findall(r"\d+", value or "")]


def _names(value):
  return [v.strip().lstrip("%") for v in (value or "").strip("{}").split(",") if v.strip()]


def _metadata(value):
  out = {}
  for key, v in _META.findall((value or "").strip()[1:-1]):
    out[key] = v[1:-1].replace('\\"', '"') if v.startswith('"') else v
  return out


def _config(attrs):
  raw = attrs.get("backend_config", "")
  try:
    if raw.startswith('"'):
      raw = json.loads(raw)
    config = json.loads(raw) if raw else {}
  except ValueError:
    return {}
  return config if isinstance(config, dict) else {}


class Module:
  """One module's optimized HLO text: computations, shapes and source locations."""

  def __init__(self, text, path="", platform=""):
    self.path, self.platform = str(path), platform
    self.name, self.entry = "", ""
    self.calls_per_step = 0
    self.other_calls = {}  # custom-call target -> launches, for calls not read as GEMMs
    self.launch_check = None
    self.computations = {}  # name -> [(instruction, opcode, rest of line, index of its "(")]
    self.roots = {}  # computation -> its ROOT instruction
    self.shapes = {}
    self._tables = {t: {} for t in _TABLES}
    current, table = None, None
    for line in text.splitlines():
      if not line.strip():
        table = None
        continue
      if line.startswith("HloModule "):
        self.name = line[len("HloModule "):].split(",", 1)[0].strip()
      elif line in _TABLES:
        table = line
      elif table and line[0].isdigit():
        key, _, value = line.partition(" ")
        self._tables[table][int(key)] = value
      elif line[0] not in " \t}" and line.rstrip().endswith("{"):
        m = _HEADER.match(line)
        current = m.group(2)
        self.computations[current] = []
        if m.group(1):
          self.entry = current
      elif line.startswith("}"):
        current = None
      elif current is not None:
        m = _DEF.match(line)
        if not m:
          continue
        rest = line[m.end():]
        end = _close(rest, 0) if rest.startswith("(") else rest.find(" ")
        op = _OPCODE.match(rest, end) if end > 0 else None
        if not op:
          continue
        name = m.group(2)
        self.shapes[name] = rest[:end]
        self.computations[current].append((name, op.group(1), rest, op.end() - 1))
        if m.group(1):
          self.roots[current] = name

  def parts(self, rest, at):
    """Operands [(name, tuple indices)] and attributes {key: raw value} of an instruction."""
    end = _close(rest, at)
    operands = []
    for text in _split(rest[at + 1:end - 1]):
      text = _COMMENT.sub("", text).strip()
      if not text:
        continue
      name, *index = text.split()[-1].lstrip("%").split("#")
      operands.append((name, [int(i) for i in index if i.isdigit()]))
    attrs = {}
    for item in _split(rest[end:].lstrip(", ")):
      key, _, value = item.partition("=")
      attrs[key.strip()] = value.strip()
    return operands, attrs

  def shape(self, name, index=()):
    shape = self.shapes.get(name, "")
    for i in index:
      elements = _elements(shape)
      shape = elements[i] if i < len(elements) else ""
    return _array(_elements(shape)[0] if shape.startswith("(") else shape)

  def stack(self, metadata, depth=8):
    """Source frames, innermost first, as 'file:line function'."""
    if "source_file" in metadata:
      return [f"{os.path.basename(metadata['source_file'])}:{metadata.get('source_line', '?')}"]
    tables = self._tables
    out, frame = [], int(metadata.get("stack_frame_id", 0) or 0)
    while frame in tables["StackFrames"] and len(out) < depth:
      f = dict(re.findall(r"(\w+)=(\d+)", tables["StackFrames"][frame]))
      location = tables["FileLocations"].get(int(f.get("file_location_id", 0)), "")
      loc = dict(re.findall(r"(\w+)=(\d+)", location))
      file = tables["FileNames"].get(int(loc.get("file_name_id", 0)), '"?"').strip('"')
      function = tables["FunctionNames"].get(int(loc.get("function_name_id", 0)), '""').strip('"')
      out.append(f"{os.path.basename(file)}:{loc.get('line', '?')} {function}".strip())
      # XLA prints parent ids one higher than the frame ids they name.
      parent = int(f.get("parent_frame_id", 1)) - 1
      frame = parent if parent != frame else 0
    return out


def _product(dims, index):
  return math.prod(dims[i] for i in index) if all(dims[i] is not None for i in index) else None


def _minor(dims, layout, skip):
  """The fastest-varying dimension of size above one outside `skip`."""
  for d in layout if layout is not None else reversed(range(len(dims))):
    if d not in skip and dims[d] != 1:
      return d
  return None


def gemm_shape(lhs: tuple, rhs: tuple, out: tuple | None, numbers: dict) -> dict:
  """Batch, M, N, K, groups, dtypes and layouts of one GEMM as C[M, N] row major.

  `lhs`, `rhs` and `out` are `_array` shapes; `numbers` holds the dimension
  numbers (lhs_batch, lhs_contracting, rhs_batch, rhs_contracting and, for
  ragged dots, rhs_group). When the output's fastest dimension comes from the
  lhs, the kernel writes C^T, so the operands swap roles.
  """
  lb, lc = numbers.get("lhs_batch", []), numbers.get("lhs_contracting", [])
  rb, rc = numbers.get("rhs_batch", []), numbers.get("rhs_contracting", [])
  rg = numbers.get("rhs_group", [])
  (ldt, ld, ll), (rdt, rd, rl) = lhs, rhs
  lfree = [i for i in range(len(ld)) if i not in lb + lc]
  rfree = [i for i in range(len(rd)) if i not in rb + rc + rg]
  m, n = _product(ld, lfree), _product(rd, rfree)
  swapped = False
  if out and lfree and rfree and len(out[1]) == len(lb) + len(lfree) + len(rfree):
    minor = _minor(out[1], out[2], range(len(lb)))
    swapped = minor is not None and minor < len(lb) + len(lfree)
  a, b = (ldt, ld, ll, lb, lc), (rdt, rd, rl, rb, rc)
  if swapped:
    a, b, m, n = b, a, n, m
  a_k = _minor(a[1], a[2], a[3]) in a[4]
  b_k = _minor(b[1], b[2], b[3]) in b[4]
  batch, k = _product(ld, lb), _product(ld, lc)
  flops = 2 * batch * m * n * k if None not in (batch, m, n, k) else None
  return {
      "batch": batch, "m": m, "n": n, "k": k, "groups": _product(rd, rg) if rg else None,
      "a_dtype": a[0], "b_dtype": b[0], "c_dtype": out[0] if out else "",
      "layout": ("MK" if a_k else "KM") + "," + ("NK" if b_k else "KN"),
      "flops": flops, "swapped": swapped,
  }


def _numbers_from_attrs(attrs):
  return {
      "lhs_batch": _ints(attrs.get("lhs_batch_dims")),
      "lhs_contracting": _ints(attrs.get("lhs_contracting_dims")),
      "rhs_batch": _ints(attrs.get("rhs_batch_dims")),
      "rhs_contracting": _ints(attrs.get("rhs_contracting_dims")),
      "rhs_group": _ints(attrs.get("rhs_group_dims")),
  }


def _numbers_from_config(config):
  ragged = config.get("grouped_gemm_backend_config", {}).get("ragged_dot_dimension_numbers", {})
  dnums = (ragged.get("dot_dimension_numbers")
           or config.get("gemm_backend_config", {}).get("dot_dimension_numbers", {}))
  return {
      "lhs_batch": [int(d) for d in dnums.get("lhs_batch_dimensions", [])],
      "lhs_contracting": [int(d) for d in dnums.get("lhs_contracting_dimensions", [])],
      "rhs_batch": [int(d) for d in dnums.get("rhs_batch_dimensions", [])],
      "rhs_contracting": [int(d) for d in dnums.get("rhs_contracting_dimensions", [])],
      "rhs_group": [int(d) for d in ragged.get("rhs_group_dimensions", [])],
  }


def _pass(op_name):
  """fwd, bwd, or remat for a forward op recomputed in the backward pass, from
  JAX's op name."""
  if "rematted_computation" in op_name:
    return "remat"
  if "transpose(" in op_name:
    return "bwd"
  return "fwd" if op_name else ""


class _Walker:
  """Every GEMM reachable from the entry computation, with launches per run."""

  def __init__(self, module):
    self.module = module
    self.sites = {}

  def record(self, key, count, flags, kernel, library, op, operands, out, numbers, attrs, config):
    module = self.module
    if len(operands) < 2:
      return
    lhs, rhs = module.shape(*operands[0]), module.shape(*operands[1])
    if not lhs or not rhs or None in lhs[1] or None in rhs[1]:
      return
    site = self.sites.get(key)
    if site:
      site["count"] += count
      site["flags"] = sorted(set(site["flags"]) | flags)
      return
    meta = _metadata(attrs.get("metadata"))
    stack = module.stack(meta)
    shape = gemm_shape(lhs, rhs, out, numbers)
    if shape["swapped"] and "split_k" in config:
      # The legacy Triton GEMM config tiles the dot's lhs with block_m.
      config = dict(config, block_m=config.get("block_n"), block_n=config.get("block_m"))
    self.sites[key] = {
        "module": module.name, "kernel": kernel, "library": library, "op": op,
        "count": count, "flags": sorted(flags), **shape,
        "lhs": module.shapes.get(operands[0][0], ""), "rhs": module.shapes.get(operands[1][0], ""),
        "pass": _pass(meta.get("op_name", "")), "op_name": meta.get("op_name", ""),
        "source": stack[0] if stack else "", "stack": stack, "config": config,
    }

  def walk(self, computation, count=1, flags=frozenset(), seen=()):
    if computation in seen or computation not in self.module.computations:
      return
    seen = seen + (computation,)
    for name, opcode, rest, at in self.module.computations[computation]:
      if opcode not in ("while", "conditional", "call", "async-start", "custom-call", "fusion") + DOTS:
        continue
      operands, attrs = self.module.parts(rest, at)
      config = _config(attrs) if opcode in ("while", "custom-call", "fusion") else {}
      if opcode == "while":
        trip = config.get("known_trip_count", {}).get("n")
        extra = set() if trip else {"trip count unknown"}
        self.walk(attrs.get("body", "").lstrip("%"), count * int(trip or 1), flags | extra, seen)
      elif opcode == "conditional":
        branches = _names(attrs.get("branch_computations")) or [
            attrs[k].lstrip("%") for k in ("true_computation", "false_computation") if k in attrs]
        for branch in branches:
          self.walk(branch, count, flags | {"conditional"}, seen)
      elif opcode in ("call", "async-start"):
        for callee in _names(attrs.get("to_apply") or attrs.get("calls")):
          self.walk(callee, count, flags, seen)
      elif opcode == "custom-call":
        target = attrs.get("custom_call_target", "").strip('"')
        if target in BLAS_TARGETS:
          self.record((computation, name), count, flags, name, BLAS_TARGETS[target], target,
                      operands, self._out(name), _numbers_from_config(config), attrs,
                      self._blas_config(config, self._workspace(name)))
        else:
          other = self.module.other_calls
          other[target] = other.get(target, 0) + count
        for callee in _names(attrs.get("called_computations")):
          self.walk(callee, count, flags, seen)
      elif opcode == "fusion":
        kind = (config.get("fusion_backend_config", {}).get("kind")
                or config.get("fusion_config", {}).get("kind") or attrs.get("kind", ""))
        library = FUSION_KINDS.get(kind, f"XLA {kind}".strip())
        for callee in _names(attrs.get("calls")):
          self._fused(name, callee, count, flags, library, config, ())
      else:
        self.record((computation, name), count, flags, name, "XLA", opcode, operands,
                    self._out(name), _numbers_from_attrs(attrs), attrs, {})

  def _fused(self, fusion, computation, count, flags, library, fusion_config, seen):
    """GEMMs inside a fusion, which launches as one kernel named after it."""
    if computation in seen:
      return
    for name, opcode, rest, at in self.module.computations.get(computation, []):
      if opcode in DOTS:
        operands, attrs = self.module.parts(rest, at)
        kernel = self._fusion_config(fusion_config, _config(attrs), self._out(fusion))
        self.record((fusion, name), count, flags, fusion, library, opcode, operands, self._out(name),
                    _numbers_from_attrs(attrs), attrs, kernel)
      elif opcode == "custom-call":
        operands, attrs = self.module.parts(rest, at)
        target = attrs.get("custom_call_target", "").strip('"')
        if target in BLAS_TARGETS:
          config = _config(attrs)
          self.record((fusion, name), count, flags, fusion, BLAS_TARGETS[target], target, operands,
                      self._out(name), _numbers_from_config(config), attrs,
                      self._blas_config(config, self._workspace(name)))
          if (fusion, name) in self.sites:
            # A library call in a fusion (dynamic-slice fusion) launches under its own name.
            self.sites[(fusion, name)]["trace_op"] = name
        else:
          other = self.module.other_calls
          other[target] = other.get(target, 0) + count
      elif opcode == "fusion":
        _, attrs = self.module.parts(rest, at)
        for callee in _names(attrs.get("calls")):
          self._fused(fusion, callee, count, flags, library, fusion_config, seen + (computation,))

  def _out(self, name):
    return self.module.shape(name)

  def _workspace(self, name):
    """Bytes of the s8 workspace a library GEMM returns as its last tuple element."""
    shape = self.module.shapes.get(name, "")
    last = _array(_elements(shape)[-1]) if shape.startswith("(") else None
    return last[1][0] if last and last[0] == "s8" and len(last[1]) == 1 else None

  @staticmethod
  def _blas_config(config, workspace):
    """The library call's settings: `algorithm` indexes the candidates the
    library's heuristic returns for this problem, which the autotuner timed."""
    gemm = config.get("gemm_backend_config") or config.get("grouped_gemm_backend_config", {}).get(
        "gemm_backend_config", {})
    out = {}
    if "selected_algorithm" in gemm:
      out["algorithm"] = int(gemm["selected_algorithm"])
    if workspace:
      out["workspace_bytes"] = workspace
    if gemm.get("epilogue", "DEFAULT") != "DEFAULT":
      out["epilogue"] = gemm["epilogue"]
    if gemm.get("alpha_real", 1) != 1:
      out["alpha"] = gemm["alpha_real"]
    if gemm.get("beta"):
      out["beta"] = gemm["beta"]
    out.update({k: True for k in ("grad_x", "grad_y") if gemm.get(k)})
    if gemm.get("scale_mode"):
      out["scale_mode"] = gemm["scale_mode"]
    return out

  @staticmethod
  def _fusion_config(fusion_config, dot_config, fusion_shape):
    """Triton tile sizes as block_m x block_n x block_k of C[M, N] row major,
    with warps, stages and the other settings that differ from their defaults."""
    backend = fusion_config.get("fusion_backend_config", {})
    legacy = backend.get("triton_gemm_config", {})
    out = {k: int(v) for k, v in legacy.items()}
    block = backend.get("block_level_fusion_config", {})
    if block:
      tiles = [int(t) for t in block.get("output_tiles", [{}])[0].get("sizes", [])]
      out["output_tile"] = "x".join(str(t) for t in tiles)
      if len(tiles) >= 2:
        # The output's two fastest dimensions hold N, then M.
        layout = fusion_shape[2] if fusion_shape and fusion_shape[2] else list(reversed(range(len(tiles))))
        if len(layout) == len(tiles):
          out["block_m"], out["block_n"] = tiles[layout[1]], tiles[layout[0]]
      for key, value in block.items():
        if key in ("output_tiles", "sequential_tile_sizes") or value in (0, "0", False, [], ""):
          continue
        out[key] = int(value) if str(value).isdigit() else value
      if out.get("num_ctas") == 1:
        del out["num_ctas"]
    if len(dot_config.get("sizes", [])) == 1:
      out["block_k"] = int(dot_config["sizes"][0])
    elif dot_config.get("sizes"):
      out["k_tile"] = "x".join(str(t) for t in dot_config["sizes"])
    return out


def gemms(module: Module) -> list[dict]:
  """One record per GEMM site: kernel, library, shape, FLOPs and launches per
  run. Other custom calls, such as operator providers' kernels, go to
  `module.other_calls` with their launches."""
  module.other_calls = {}
  walker = _Walker(module)
  walker.walk(module.entry)
  return list(walker.sites.values())


def platform_of(path) -> str:
  name = pathlib.Path(path).name
  if re.search(r"\.gfx\w*_gpu_after_optimizations\.txt$", name):
    return "rocm"
  if name.endswith(".cpu_after_optimizations.txt"):
    return "cpu"
  return ""


_LAUNCH = re.compile(r"profile_annotation=([^,\s]+), launch dimensions=blocks: \{(\d+), (\d+), (\d+)\}, "
                     r"threads/block: \{(\d+), (\d+), (\d+)\}.*?shmem_bytes=(\d+)")
_THUNK = re.compile(r"^\s*\d+: (k\w+)", re.M)
# Thunks that run library GEMMs; their lines name no instruction.
BLAS_THUNKS = ("kCublasLtMatmul", "kGemm")


def thunks(path) -> dict | None:
  """The executable's launch list beside an optimized module: each kernel's
  grid, threads per block and LDS bytes by instruction name, and the number
  of thunks of each kind. None when XLA did not write one."""
  base = re.sub(r"\.(\w+_)?after_optimizations\.txt$", "", str(path))
  for suffix in (".thunk_sequence_after_thunk_passes.txt", ".thunk_sequence.txt"):
    p = pathlib.Path(base + suffix)
    if p.is_file():
      text = p.read_text(errors="replace")
      kernels = {m.group(1): {"grid": int(m.group(2)) * int(m.group(3)) * int(m.group(4)),
                              "threads": int(m.group(5)) * int(m.group(6)) * int(m.group(7)),
                              "lds_bytes": int(m.group(8))}
                 for m in _LAUNCH.finditer(text)}
      kinds = {}
      for kind in _THUNK.findall(text):
        kinds[kind] = kinds.get(kind, 0) + 1
      return {"file": p.name, "kernels": kernels, "kinds": kinds}
  return None


def check_launches(module: Module, sites: list[dict], launched: dict | None):
  """Marks each fusion GEMM with its launch dimensions and records in
  `module.launch_check` how many of the module's GEMMs its thunks launch."""
  if launched is None:
    module.launch_check = None
    return
  fused = {s["kernel"] for s in sites if s["op"] not in BLAS_TARGETS}
  for s in sites:
    if s["op"] not in BLAS_TARGETS:
      s["launch"] = launched["kernels"].get(s["kernel"])
  blas = {s["kernel"] for s in sites if s["op"] in BLAS_TARGETS}
  module.launch_check = {
      "file": launched["file"], "kernels": len(fused), "kernels_launched": len(fused & set(launched["kernels"])),
      "library_calls": len(blas), "library_thunks": sum(launched["kinds"].get(k, 0) for k in BLAS_THUNKS),
  }


def files(path) -> list[pathlib.Path]:
  path = pathlib.Path(path)
  if path.is_file():
    return [path]
  return sorted(p for p in path.rglob("module_*") if p.is_file() and FILE.match(p.name))


def analyze(path, module_re: str = "") -> list[tuple[Module, list[dict]]]:
  """(module, GEMM sites) for each optimized module under `path` whose file
  name matches `module_re`."""
  paths = files(path)
  default = "rocm" if "rocm" in map(platform_of, paths) else ""
  out = []
  for p in paths:
    if module_re and not re.search(module_re, p.name):
      continue
    module = Module(p.read_text(errors="replace"), p, platform_of(p) or default)
    sites = gemms(module)
    check_launches(module, sites, thunks(p))
    out.append((module, sites))
  return out


# Collective opcodes. XLA also runs them as `<op>-start` and `-done` pairs, or
# wraps one in an async computation between `async-start` and `async-done`.
COLLECTIVE_OPS = ("all-gather", "reduce-scatter", "all-reduce", "ragged-all-to-all", "all-to-all",
                  "collective-permute", "collective-broadcast")
_DONE = ("async-done", "all-gather-done", "all-reduce-done", "collective-permute-done")
# Ops that hand a value on unchanged, as far as which collective it comes from or goes to.
_PASS = ("bitcast", "copy", "reshape", "transpose", "convert", "copy-start", "copy-done",
         "optimization-barrier")


def _array_bytes(array) -> int:
  if not array or None in array[1]:
    return 0
  bits = re.match(r"[a-z]+(\d+)", array[0])
  return int(math.prod(array[1]) * (int(bits.group(1)) / 8 if bits else 1))


def _nbytes(shape: str) -> int:
  """Bytes of an array shape, or of a tuple shape's arrays."""
  shape = _COMMENT.sub("", shape).strip()
  if shape.startswith("("):
    return sum(_nbytes(e) for e in _split(shape[1:-1]))
  return _array_bytes(_array(shape))


def collectives(module: Module) -> dict:
  """{name: {"kind", "bytes", "tensors", "op"}} of each collective in the
  module: a collective op, a `<kind>-start`, or an `async-start` whose
  computation is rooted at one. `bytes` is the larger of the op's operands and
  its result, the gathered or the unreduced buffers; `tensors` counts its
  operands, more than one where XLA combined collectives. A trace can name the
  op without the last `.N` XLA appended when it wrapped it, so that name is
  listed too."""
  index = {name: (opcode, rest, at) for instructions in module.computations.values()
           for name, opcode, rest, at in instructions}

  def size(name):
    opcode, rest, at = index[name]
    operands, _ = module.parts(rest, at)
    given = sum(_array_bytes(module.shape(n, i)) for n, i in operands)
    result = _nbytes(module.shapes.get(name, ""))
    return {"bytes": max(given, result - given if opcode.endswith("-start") else result),
            "tensors": len(operands)}

  out = {}
  for name, (opcode, rest, at) in index.items():
    op = name
    if opcode == "async-start":
      _, attrs = module.parts(rest, at)
      op = module.roots.get((_names(attrs.get("calls")) or [""])[0], "")
      kind = index[op][0] if op in index else ""
    else:
      kind = opcode[:-len("-start")] if opcode.endswith("-start") else opcode
    if kind not in COLLECTIVE_OPS:
      continue
    out[name] = {"kind": kind, **size(op), "op": op}
  for name in list(out):
    alias = re.sub(r"\.\d+$", "", name)
    if alias != name and alias not in index:
      out.setdefault(alias, out[name])
  return out


class _Links:
  """Def-use links between a module's instructions, to follow a value through
  ops that hand it on unchanged, tuples and a loop's carried values."""

  def __init__(self, module: Module, ops: dict):
    self.ops = ops
    self.defs, self.users = {}, {}
    self.carried, self.carry = {}, {}  # body parameter -> body ROOT, and back
    for instructions in module.computations.values():
      for name, opcode, rest, at in instructions:
        operands, attrs = module.parts(rest, at)
        if opcode in ("parameter", "constant"):
          operands = []
        self.defs[name] = (opcode, operands, attrs)
        for position, (operand, index) in enumerate(operands):
          self.users.setdefault(operand, []).append((name, position, tuple(index)))
    for opcode, _, attrs in list(self.defs.values()):
      if opcode != "while":
        continue
      body = attrs.get("body", "").lstrip("%")
      params = [n for n, op, _, _ in module.computations.get(body, []) if op == "parameter"]
      root = module.roots.get(body)
      if root and len(params) == 1:
        self.carried[params[0]], self.carry[root] = root, params[0]

  def source(self, name: str, index: tuple = (), hops: int = 16) -> str:
    """The collective whose result the value at `name`, tuple `index`, is; ""
    when it is none. The loop's parameter element i is its last trip's ROOT
    operand i."""
    while hops and name in self.defs:
      hops -= 1
      opcode, operands, attrs = self.defs[name]
      if name in self.ops:
        return name
      if (opcode in _PASS or opcode in _DONE) and operands:
        name, index = operands[0][0], tuple(operands[0][1]) + index
      elif opcode == "get-tuple-element" and operands:
        name, index = operands[0][0], tuple(operands[0][1]) + (int(attrs.get("index", "0")),) + index
      elif opcode == "tuple" and index and index[0] < len(operands):
        name, index = operands[index[0]][0], tuple(operands[index[0]][1]) + index[1:]
      elif opcode == "parameter" and name in self.carried and index:
        name = self.carried[name]
      else:
        return ""
    return ""

  def sinks(self, name: str, index: tuple = (), hops: int = 16, seen=None) -> set:
    """The collectives the value at `name`, tuple `index`, feeds; a loop's ROOT
    operand i is its next trip's parameter element i."""
    seen = set() if seen is None else seen
    if not hops or (name, index) in seen:
      return set()
    seen.add((name, index))
    out = set()
    for user, position, ref in self.users.get(name, []):
      if ref == index[:len(ref)]:
        inner = index[len(ref):]  # where the value sits in what the user reads
      elif ref[:len(index)] == index:
        inner = ()  # the user reads a part of the value
      else:
        continue
      opcode, _, attrs = self.defs[user]
      if user in self.ops:
        out.add(user)
      elif opcode == "get-tuple-element":
        if inner and int(attrs.get("index", "0")) == inner[0]:
          out |= self.sinks(user, inner[1:], hops - 1, seen)
      elif opcode == "tuple":
        out |= self.sinks(user, (position,) + inner, hops - 1, seen)
        if user in self.carry:
          out |= self.sinks(self.carry[user], (position,) + inner, hops - 1, seen)
      elif opcode in _PASS or opcode in _DONE:
        out |= self.sinks(user, inner, hops - 1, seen)
    return out


def link_collectives(module: Module, sites: list[dict], ops: dict | None = None) -> dict:
  """Adds to each GEMM site the collective ops whose results it reads (`reads`)
  and those its result feeds (`writes`), followed through ops that hand a
  value on unchanged, tuples and a loop's carried values but not through
  fusions. An async collective is named by the op it wraps, as a trace names
  it. Returns the module's `collectives`."""
  ops = collectives(module) if ops is None else ops
  links = _Links(module, ops)
  for s in sites:
    if s["kernel"] not in links.defs:
      continue
    _, operands, _ = links.defs[s["kernel"]]
    tuple_out = module.shapes.get(s["kernel"], "").startswith("(") and s["op"] in BLAS_TARGETS
    reads = {links.source(n, tuple(i)) for n, i in operands} - {""}
    s["reads"] = sorted({ops[c]["op"] for c in reads})
    s["writes"] = sorted({ops[c]["op"] for c in links.sinks(s["kernel"], (0,) if tuple_out else ())})
  return ops


def program_calls(text: str) -> dict[str, int]:
  """Calls per step of each compiled module, from the text of the program one
  step runs: the fragments an IFRT IR or MPMD program calls, or else the one
  module of a jit program."""
  calls = {}
  for name in re.findall(r"(?:ifrt\.Call|mpmd\.fragment_call)[^@\n]*@([\w.$-]+)", text):
    calls[name] = calls.get(name, 0) + 1
  if calls:
    return calls
  module = re.search(r"module @([\w.$-]+)", text)
  return {module.group(1): 1} if module else {}


def programs_from_run(run_dir) -> dict[str, int]:
  """Calls per step of each module, from the JAX-Titan trainer's compile event
  in RUN/train.jsonl."""
  programs = None
  with open(pathlib.Path(run_dir) / "train.jsonl") as f:
    for line in f:
      if '"compile"' in line:
        record = json.loads(line)
        if record.get("event") == "compile" and record.get("programs"):
          programs = record["programs"]
  if not programs:
    raise ValueError(f"{run_dir}/train.jsonl has no compile event with programs")
  return programs


def programs_from_file(path) -> dict[str, int]:
  """Calls per step of each module from a JSON object {module: calls}, or from
  the step's program text (`lowered.as_text()` of an MPMD or jit step)."""
  text = pathlib.Path(path).read_text()
  try:
    calls = json.loads(text)
  except ValueError:
    calls = program_calls(text)
  if not isinstance(calls, dict) or not calls or not all(isinstance(n, int) for n in calls.values()):
    raise ValueError(f"{path} is neither a JSON object of calls per module nor a program that calls modules")
  return calls


def per_step(results, programs) -> list[tuple[Module, list[dict]]]:
  """The modules a step runs, with each site's `count` scaled to launches per step."""
  out = []
  for module, sites in results:
    calls = programs.get(module.name)
    if calls:
      module.calls_per_step = calls
      module.other_calls = {t: n * calls for t, n in module.other_calls.items()}
      out.append((module, [dict(s, count=s["count"] * calls, calls_per_step=calls) for s in sites]))
  return out


def _shape_key(s):
  return (s["library"], s["a_dtype"], s["b_dtype"], s["c_dtype"], s["batch"], s["m"], s["n"], s["k"],
          s["groups"], s["layout"])


def group(sites: list[dict]) -> list[dict]:
  """Sites with the same library, dtypes, shape and layout, by total FLOPs."""
  groups = {}
  for s in sites:
    g = groups.setdefault(_shape_key(s), {**{k: s[k] for k in (
        "library", "a_dtype", "b_dtype", "c_dtype", "batch", "m", "n", "k", "groups", "layout", "flops")},
        "count": 0, "kernels": 0, "modules": set(), "passes": set(), "sources": [], "flags": set()})
    g["count"] += s["count"]
    g["kernels"] += 1
    g["modules"].add(s["module"])
    g["passes"].add(s["pass"])
    g["flags"].update(s["flags"])
    if s["source"] and s["source"] not in g["sources"]:
      g["sources"].append(s["source"])
  out = []
  for g in groups.values():
    g["total_flops"] = g["flops"] * g["count"] if g["flops"] is not None else None
    g["modules"], g["flags"] = sorted(g["modules"]), sorted(g["flags"])
    g["passes"] = sorted(p for p in g["passes"] if p)
    out.append(g)
  return sorted(out, key=lambda g: -(g["total_flops"] or 0))


def _dtype(g):
  a, b, c = g["a_dtype"], g["b_dtype"], g["c_dtype"]
  ab = a if a == b else f"{a}x{b}"
  return ab if c in ("", a) and a == b else f"{ab}->{c}"


def _flop(x):
  return "-" if x is None else f"{x / 1e9:.3g}"


def _amount(flops):
  for unit, scale in (("PFLOP", 1e15), ("TFLOP", 1e12), ("GFLOP", 1e9), ("MFLOP", 1e6)):
    if flops >= scale:
      return f"{flops / scale:.4g} {unit}"
  return f"{flops:.4g} FLOP"


def _count(n, noun):
  return f"{n} {noun}" + ("" if n == 1 else "s")


def _sources(sources):
  """The first source of a shape's sites, and how many others there are."""
  return sources[0] + (f" +{len(sources) - 1}" if len(sources) > 1 else "") if sources else ""


def _table(rows, total, width=110):
  header = (f"  {'count':>6}  {'library':<10} {'dtype':<16} {'batch':>6} {'M':>7} {'N':>7} {'K':>7}  "
            f"{'layout':<6} {'GFLOP':>8} {'share':>6}  pass      source")
  print(header)
  for g in rows:
    share = f"{100 * g['total_flops'] / total:5.1f}%" if total and g["total_flops"] is not None else "     -"
    flags = " (" + "; ".join(g["flags"]) + ")" if g["flags"] else ""
    sources = _sources(g["sources"])
    batch = "-" if g["batch"] == 1 else str(g["batch"])
    if g["groups"]:
      batch = f"g{g['groups']}"
    line = (f"  {g['count']:>6}  {g['library']:<10} {_dtype(g):<16} {batch:>6} {g['m']:>7} {g['n']:>7} "
            f"{g['k']:>7}  {g['layout']:<6} {_flop(g['flops']):>8} {share}  {','.join(g['passes']):<9} "
            f"{sources}{flags}")
    print(line[:width] if width else line)


_LABELS = {"num_warps": "warps", "num_stages": "stages", "split_k": "split_k", "num_ctas": "ctas",
           "mfma_size": "mfma", "waves_per_eu": "waves/EU", "group_size": "group",
           "num_tiles_per_pid": "tiles/pid", "algorithm": "algorithm", "epilogue": "epilogue",
           "alpha": "alpha", "beta": "beta", "scale_mode": "scale_mode"}


def config_text(site: dict) -> str:
  """A GEMM's kernel settings in one line: Triton's block_m x block_n x
  block_k, warps and stages, with the grid and LDS it launched with; a
  library call's algorithm and workspace."""
  c = site["config"]
  parts = []
  if "block_m" in c and "block_n" in c:
    parts.append(f"{c['block_m']}x{c['block_n']}x{c.get('block_k', '?')}")
  elif "output_tile" in c:
    parts.append(f"tile {c['output_tile']}" + (f", k {c['block_k']}" if "block_k" in c else ""))
  for key, label in _LABELS.items():
    if key in c and not (key == "split_k" and c[key] == 1):
      parts.append(f"{label} {c[key]:.4g}" if isinstance(c[key], float) else f"{label} {c[key]}")
  if c.get("workspace_bytes"):
    parts.append(f"workspace {c['workspace_bytes'] / 2**20:g} MiB")
  parts += [k for k in ("grad_x", "grad_y") if c.get(k)]
  known = set(_LABELS) | {"block_m", "block_n", "block_k", "output_tile", "workspace_bytes", "grad_x", "grad_y"}
  parts += [f"{k} {v}" for k, v in c.items() if k not in known]
  launch = site.get("launch")
  if launch:
    parts.append(f"grid {launch['grid']}, LDS {launch['lds_bytes'] / 1024:.0f} KiB")
  return ", ".join(parts)


def _sites_table(sites, top=0, width=0):
  print(f"  {'count':>6}  {'kernel':<26} {'library':<10} {'M':>7} {'N':>7} {'K':>7}  {'layout':<6} "
        f"{'pass':<6} {'config':<54} source")
  for s in sorted(sites, key=lambda s: -(s["flops"] or 0) * s["count"])[:top or None]:
    line = (f"  {s['count']:>6}  {s['kernel'][:26]:<26} {s['library']:<10} {s['m']:>7} {s['n']:>7} "
            f"{s['k']:>7}  {s['layout']:<6} {s['pass']:<6} {config_text(s):<54} {s['source']}")
    print(line[:width] if width else line)


def _launch_text(checks):
  """One line on whether the thunk sequences launch every GEMM the HLO holds."""
  checks = [c for c in checks if c]
  if not checks:
    return ""
  kernels, launched = sum(c["kernels"] for c in checks), sum(c["kernels_launched"] for c in checks)
  calls, thunks_ = sum(c["library_calls"] for c in checks), sum(c["library_thunks"] for c in checks)
  verdict = "all launched" if (kernels, calls) == (launched, thunks_) else "MISMATCH"
  return (f"  thunk sequence: {launched} of {kernels} fused GEMM kernels by name, {thunks_} library "
          f"GEMM thunks for {calls} library calls ({verdict})")


def _other_calls(calls):
  if calls:
    items = ", ".join(f"{t} x{n}" for t, n in sorted(calls.items(), key=lambda kv: -kv[1]))
    print(f"  other custom calls, not read as GEMMs (launches): {items}")


def _summary(sites):
  launches = sum(s["count"] for s in sites)
  total = sum((s["flops"] or 0) * s["count"] for s in sites)
  by_library = {}
  for s in sites:
    flops, n = by_library.get(s["library"], (0, 0))
    by_library[s["library"]] = (flops + (s["flops"] or 0) * s["count"], n + s["count"])
  shares = sorted(by_library.items(), key=lambda kv: -kv[1][0]) if total else []
  return launches, total, ", ".join(
      f"{lib} {100 * f / total:.0f}% ({n} launches)" for lib, (f, n) in shares if f >= total / 200)


def write_csv(path, sites: list[dict]):
  """One row per GEMM site, with its flags joined and its kernel config as JSON."""
  fields = ["module", "kernel", "library", "op", "count", "calls_per_step", "pass", "batch", "m", "n",
            "k", "groups", "a_dtype", "b_dtype", "c_dtype", "layout", "flops", "lhs", "rhs", "flags",
            "source", "op_name", "kernel_config", "config", "launch", "reads", "writes"]
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for s in sites:
      writer.writerow({**s, "flags": ";".join(s["flags"]), "kernel_config": config_text(s),
                       "config": json.dumps(s["config"], sort_keys=True),
                       "launch": json.dumps(s["launch"]) if s.get("launch") else "",
                       "reads": ";".join(s.get("reads", [])), "writes": ";".join(s.get("writes", []))})


CLEAN, OVERLAPPED = 0.05, 0.5  # collective-overlap fractions of a launch


def _median_ci(values, n=1000, seed=0):
  if len(values) < 2:
    return (values[0], values[0]) if values else (None, None)
  rng = random.Random(seed)
  meds = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(n))
  return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


COLLECTIVE = re.compile(r"nccl|rccl|all.?reduce|all.?gather|reduce.?scatter|all.?to.?all|collective.?permute", re.I)
# What a collective kernel runs, from its HLO op or else its name; RCCL's
# generic kernel names none.
KINDS = (("all-gather", re.compile(r"all.?gather", re.I)),
         ("reduce-scatter", re.compile(r"reduce.?scatter", re.I)),
         ("all-reduce", re.compile(r"all.?reduce", re.I)),
         ("all-to-all", re.compile(r"all.?to.?all", re.I)),
         ("collective-permute", re.compile(r"collective.?permute|send|recv", re.I)))
# Runtime copies only: XLA's own copy fusions are compute kernels.
COPY = re.compile(r"^Memcpy|^Memset|__amd_rocclr_(copy|fill)", re.I)
DEVICE_PLANE = re.compile(r"/device:GPU:(\d+)")
GRID = re.compile(r"\bgrid:(\d+),(\d+),(\d+)")


def is_stream_line(name: str) -> bool:
  """Whether a trace line holds a GPU stream's kernels and copies."""
  return name.startswith("Stream") or "stream" in name.lower()


def collective_kind(hlo_op: str, kernel: str = "") -> str:
  """all-gather, reduce-scatter, all-reduce, all-to-all or collective-permute
  for a collective kernel; "" when neither its HLO op nor its name says."""
  for text in (hlo_op, kernel):
    for kind, pattern in KINDS:
      if text and pattern.search(text):
        return kind
  return ""


def merge_intervals(intervals) -> list[list]:
  """The union of [start, end) intervals, sorted and disjoint."""
  out = []
  for start, end in sorted(intervals):
    if out and start <= out[-1][1]:
      out[-1][1] = max(out[-1][1], end)
    else:
      out.append([start, end])
  return out


def _covered(start, end, merged, starts):
  """Length of [start, end) inside the merged, sorted intervals."""
  i = max(0, bisect.bisect_right(starts, start) - 1)
  total = 0
  while i < len(merged) and merged[i][0] < end:
    lo, hi = max(start, merged[i][0]), min(end, merged[i][1])
    if hi > lo:
      total += hi - lo
    i += 1
  return total


def compute_launches(events: dict) -> list[dict]:
  """One record per compute kernel launch in `events` ({gpu id: [{name,
  start_ns, end_ns, hlo_module, hlo_op, grid}]}), with the share collectives
  overlapped and, when they did, the share each collective's HLO op (or kernel
  name) overlapped. Collectives can overlap each other, so those can add up to
  more."""
  out = []
  for gpu, evs in sorted(events.items()):
    collectives = sorted((e for e in evs if COLLECTIVE.search(e["name"])), key=lambda e: e["start_ns"])
    comm = merge_intervals([(e["start_ns"], e["end_ns"]) for e in collectives])
    starts = [s for s, _ in comm]
    collective_starts = [e["start_ns"] for e in collectives]
    longest = max((e["end_ns"] - e["start_ns"] for e in collectives), default=0)
    for e in sorted(evs, key=lambda e: e["start_ns"]):
      if COLLECTIVE.search(e["name"]) or COPY.search(e["name"]):
        continue
      dur = e["end_ns"] - e["start_ns"]
      overlap = _covered(e["start_ns"], e["end_ns"], comm, starts) / dur if dur > 0 else 0.0
      record = {"gpu": gpu, "kernel": e["name"], "hlo_module": e.get("hlo_module", ""),
                "hlo_op": e.get("hlo_op", ""), "start_ns": e["start_ns"], "dur_ns": dur,
                "rccl_overlap": round(overlap, 4)}
      if overlap > 0:
        shared = {}
        first = bisect.bisect_left(collective_starts, e["start_ns"] - longest)
        for c in collectives[first:bisect.bisect_left(collective_starts, e["end_ns"])]:
          ns = min(e["end_ns"], c["end_ns"]) - max(e["start_ns"], c["start_ns"])
          if ns > 0:
            op = c.get("hlo_op") or c["name"]
            shared[op] = shared.get(op, 0) + ns
        record["collectives"] = {op: round(ns / dur, 4) for op, ns in shared.items()}
      out.append(record)
  return out


def collective_launches(events: dict) -> list[dict]:
  """One record per collective HLO op (or kernel name, without one): its kind,
  launches on all GPUs, the GPUs it ran on, its median duration, the
  workgroups its kernel launched with, and its exposed time, which no compute
  kernel overlapped (two collectives exposed at once both count it)."""
  ops = {}
  for gpu, evs in events.items():
    compute = merge_intervals([(e["start_ns"], e["end_ns"]) for e in evs
                               if not COLLECTIVE.search(e["name"]) and not COPY.search(e["name"])])
    starts = [s for s, _ in compute]
    for e in evs:
      if not COLLECTIVE.search(e["name"]):
        continue
      op = e.get("hlo_op") or e["name"]
      r = ops.setdefault((e.get("hlo_module", ""), op),
                         {"kernel": e["name"], "ns": [], "exposed_ns": 0, "gpus": set(), "grid": set()})
      dur = e["end_ns"] - e["start_ns"]
      r["ns"].append(dur)
      r["exposed_ns"] += dur - _covered(e["start_ns"], e["end_ns"], compute, starts)
      r["gpus"].add(gpu)
      if e.get("grid"):
        r["grid"].add(e["grid"])
  return [{"hlo_module": module, "hlo_op": op, "kind": collective_kind(op, r["kernel"]),
           "launches": len(r["ns"]), "gpus": len(r["gpus"]), "median_us": statistics.median(r["ns"]) / 1e3,
           "exposed_us": r["exposed_ns"] / 1e3, "grid": sorted(r["grid"])}
          for (module, op), r in sorted(ops.items())]


def _varint(buf, i):
  shift = value = 0
  while True:
    byte = buf[i]
    i += 1
    value |= (byte & 0x7F) << shift
    if byte < 0x80:
      return value, i
    shift += 7


def _int64(value):
  return value - (1 << 64) if value >= 1 << 63 else value


def _fields(buf, start=0, end=None):
  """(field number, value) of each field of a protobuf message in buf[start:end]:
  an int for a varint, raw bytes for a fixed-size value, and a (start, end)
  span of `buf` for a length-delimited one."""
  i, end = start, len(buf) if end is None else end
  while i < end:
    key, i = _varint(buf, i)
    wire = key & 7
    if wire == 0:
      value, i = _varint(buf, i)
    elif wire == 1:
      value, i = buf[i:i + 8], i + 8
    elif wire == 2:
      n, i = _varint(buf, i)
      value, i = (i, i + n), i + n
    elif wire == 5:
      value, i = buf[i:i + 4], i + 4
    else:
      raise ValueError(f"protobuf wire type {wire} at byte {i}")
    yield key >> 3, value


def _text(buf, span):
  return bytes(buf[span[0]:span[1]]).decode("utf-8", "replace")


def _stat(buf, span):
  """(metadata id, value, referenced metadata id) of an XStat."""
  mid, value, ref = 0, None, None
  for field, v in _fields(buf, *span):
    if field == 1:
      mid = v
    elif field == 2:
      value = struct.unpack("<d", v)[0]
    elif field == 3:
      value = v
    elif field == 4:
      value = _int64(v)
    elif field == 5:
      value = _text(buf, v)
    elif field == 6:
      value = bytes(buf[v[0]:v[1]])
    elif field == 7:
      ref = v
  return mid, value, ref


def xplane(path, plane: re.Pattern | None = None) -> list[dict]:
  """The planes of a `jax.profiler` trace (`*.xplane.pb`, an XSpace),
  decoded with the standard library: [{"name", "lines": [{"name", "events":
  [{"name", "start_ns", "end_ns", "stats"}]}]}], with an event's stats and its
  metadata's by name, and a reference value as the name it refers to. `plane`
  keeps the planes whose name it matches."""
  buf = memoryview(pathlib.Path(path).read_bytes())
  out = []
  for field, span in _fields(buf):
    if field != 1:
      continue
    fields = list(_fields(buf, *span))
    name = next((_text(buf, v) for f, v in fields if f == 2), "")
    if plane and not plane.search(name):
      continue
    event_meta, stat_names = {}, {}
    for f, v in fields:
      if f not in (4, 5):
        continue
      entry = dict(_fields(buf, *v))  # a map entry: key 1, value 2
      if 2 not in entry:
        continue
      meta = list(_fields(buf, *entry[2]))
      meta_name = next((_text(buf, mv) for mf, mv in meta if mf == 2), "")
      if f == 5:
        stat_names[entry.get(1, 0)] = meta_name
      else:
        event_meta[entry.get(1, 0)] = (meta_name, [_stat(buf, mv) for mf, mv in meta if mf == 5])
    lines = []
    for f, v in fields:
      if f != 3:
        continue
      line_name, base_ns, events = "", 0, []
      for lf, lv in _fields(buf, *v):
        if lf == 2:
          line_name = _text(buf, lv)
        elif lf == 3:
          base_ns = _int64(lv)
        elif lf == 4:
          mid, offset_ps, duration_ps, stats = 0, 0, 0, []
          for ef, ev in _fields(buf, *lv):
            if ef == 1:
              mid = ev
            elif ef == 2:
              offset_ps = _int64(ev)
            elif ef == 3:
              duration_ps = _int64(ev)
            elif ef == 4:
              stats.append(_stat(buf, ev))
          event_name, meta_stats = event_meta.get(mid, ("", []))
          named = {}
          for smid, value, ref in meta_stats + stats:
            named[stat_names.get(smid, "")] = stat_names.get(ref, "") if ref is not None else value
          start = base_ns + offset_ps / 1000
          events.append({"name": event_name, "start_ns": start, "end_ns": start + duration_ps / 1000,
                         "stats": named})
      lines.append({"name": line_name, "events": events})
    out.append({"name": name, "lines": lines})
  return out


def trace_kernels(path) -> dict:
  """{gpu id: [{name, start_ns, end_ns, hlo_module, hlo_op, grid}]} from the
  GPU stream lines of an `*.xplane.pb` trace; `grid` is the workgroups a
  kernel launches with."""
  out = {}
  for p in xplane(path, DEVICE_PLANE):
    events = out.setdefault(int(DEVICE_PLANE.search(p["name"]).group(1)), [])
    for line in p["lines"]:
      if not is_stream_line(line["name"]):
        continue
      for e in line["events"]:
        record = {"name": e["name"], "start_ns": e["start_ns"], "end_ns": e["end_ns"]}
        for key in ("hlo_module", "hlo_op"):
          if key in e["stats"]:
            record[key] = str(e["stats"][key])
        grid = GRID.search(str(e["stats"].get("kernel_details", "")))
        if grid:
          record["grid"] = int(grid.group(1)) * int(grid.group(2)) * int(grid.group(3))
        events.append(record)
  return out


def newest_xplane(trace) -> pathlib.Path:
  """`trace` when it is a file, else the newest `*.xplane.pb` under it."""
  path = pathlib.Path(trace)
  if path.is_file():
    return path
  files = sorted(path.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)
  if not files:
    raise FileNotFoundError(f"no *.xplane.pb under {trace}")
  return files[-1]


def trace_times(trace, steps: int = 0, gpu: int = -1) -> dict:
  """What `profile.py --kernel_times` writes, read from the newest trace under
  `trace` with the standard library: every compute kernel launch with the
  collectives that overlapped it (`compute_launches`), and each collective op
  (`collective_launches`). `gpu` keeps one device of a multi-GPU process."""
  path = newest_xplane(trace)
  events = trace_kernels(path)
  if gpu >= 0:
    events = {gpu: events.get(gpu, [])}
  return {"xplane": str(path), "steps": steps or None, "gpus": sorted(events),
          "launches": compute_launches(events), "collectives": collective_launches(events)}


def _launches(sites: list[dict], kernel_times: dict):
  """{(module, kernel): (its sites, its launches)} for the kernels a trace
  launched, joined by HLO module and op; the kernels; the HLO ops traced."""
  by_op = {}
  for k in kernel_times["launches"]:
    if k.get("hlo_op"):
      by_op.setdefault((k["hlo_module"], k["hlo_op"]), []).append(k)
  kernels = {}
  for s in sites:
    kernels.setdefault((s["module"], s["kernel"]), []).append(s)
  matched = {}
  for (module, kernel), group_ in kernels.items():
    for op in [kernel] + sorted({s["trace_op"] for s in group_ if s.get("trace_op")}):
      launches = by_op.get((module, op)) or by_op.get((module, "%" + op), [])
      if launches:
        matched[(module, kernel)] = (group_, launches)
        break
  return matched, kernels, by_op


def attach_times(sites: list[dict], kernel_times: dict, steps: int = 0) -> dict:
  """Add each site's measured launches from `profile.py --kernel_times` as `site["time"]`."""
  steps = steps or kernel_times.get("steps") or 1
  matched, kernels, by_op = _launches(sites, kernel_times)
  for group_, launches in matched.values():
    us = [k["dur_ns"] / 1e3 for k in launches]
    clean = [k["dur_ns"] / 1e3 for k in launches if k["rccl_overlap"] < CLEAN]
    over = [k["dur_ns"] / 1e3 for k in launches if k["rccl_overlap"] >= OVERLAPPED]
    lo, hi = _median_ci(us)
    expected = sum(s["count"] for s in group_) / len(group_)
    record = {
        "launches": len(launches), "launches_per_step": len(launches) / steps,
        "coverage": len(launches) / steps / expected if expected else None,
        "median_us": statistics.median(us), "ci_us": [lo, hi],
        "clean": {"launches": len(clean), "median_us": statistics.median(clean) if clean else None},
        "overlapped": {"launches": len(over), "median_us": statistics.median(over) if over else None},
        "ms_per_step": sum(us) / 1e3 / steps, "kernels": sorted({k["kernel"] for k in launches})[:3],
        "shared_by_sites": len(group_),
    }
    for s in group_:
      s["time"] = dict(record, ms_per_step=record["ms_per_step"] / len(group_))
  timed = [s for s in sites if s.get("time")]
  seen = sum(s["time"]["launches_per_step"] for s in timed)
  expected = sum(s["count"] for s in timed)
  return {"steps": steps, "kernels_matched": len(matched), "kernels": len(kernels),
          "launches": len(kernel_times["launches"]), "unmatched_hlo_ops": len(by_op) - len(matched),
          "coverage": seen / expected if expected else None}


def _kind(op: str, ops: dict, traced: dict) -> str:
  if op in ops:
    return ops[op]["kind"]
  if traced.get(op):
    return traced[op]
  return next((kind for kind in COLLECTIVE_OPS if kind in op), "collective")


def overlap(sites: list[dict], kernel_times: dict, steps: int = 0, ops: dict | None = None) -> list[dict]:
  """One record per GEMM shape that collectives overlapped, or that reads or
  writes a collective (`link_collectives`): its launches and time per step, the
  medians of its clean and of its mostly overlapped launches, the milliseconds
  per step its launches a collective overlapped took beyond the clean median,
  the collectives beside it with their share of the overlapped time, the
  collectives it reads and writes, and its sites' sources. `ops` holds the
  program's `collectives`, for the kinds and sizes."""
  steps = steps or kernel_times.get("steps") or 1
  ops = ops or {}
  traced = {c["hlo_op"]: c.get("kind", "") for c in kernel_times.get("collectives", [])}
  shapes = {}
  for group_, launches in _launches(sites, kernel_times)[0].values():
    shapes.setdefault(_shape_key(group_[0]), {"site": group_[0], "launches": []})["launches"] += launches
  for s in sites:
    r = shapes.get(_shape_key(s))
    if r is None:
      continue
    r.setdefault("reads", set()).update(s.get("reads", ()))
    r.setdefault("writes", set()).update(s.get("writes", ()))
    r.setdefault("passes", set()).update([s["pass"]] if s["pass"] else [])
    sources = r.setdefault("sources", [])
    if s["source"] and s["source"] not in sources:
      sources.append(s["source"])

  def named(names):
    return [{"op": op, "kind": _kind(op, ops, traced), "bytes": ops.get(op, {}).get("bytes"),
             "tensors": ops.get(op, {}).get("tensors")} for op in sorted(names)]

  rows = []
  for r in shapes.values():
    launches = r["launches"]
    beside = [k for k in launches if k["rccl_overlap"] >= CLEAN]
    if not beside and not r["reads"] and not r["writes"]:
      continue
    clean = [k["dur_ns"] / 1e3 for k in launches if k["rccl_overlap"] < CLEAN]
    over = [k["dur_ns"] / 1e3 for k in launches if k["rccl_overlap"] >= OVERLAPPED]
    ref = statistics.median(clean) if clean else None
    by_kind = {}
    for k in beside:
      for op, share in k.get("collectives", {}).items():
        entry = by_kind.setdefault(_kind(op, ops, traced), {})
        entry[op] = entry.get(op, 0.0) + share * k["dur_ns"] / 1e3
    total = sum(sum(e.values()) for e in by_kind.values())
    s = r["site"]
    rows.append({
        **{key: s[key] for key in ("library", "a_dtype", "b_dtype", "c_dtype", "batch", "m", "n", "k",
                                   "groups", "layout", "flops")},
        "passes": sorted(r["passes"]), "launches_per_step": len(launches) / steps,
        "overlapped_per_step": len(over) / steps, "ms_per_step": sum(k["dur_ns"] for k in launches) / 1e6 / steps,
        "clean_us": ref, "overlapped_us": statistics.median(over) if over else None,
        "cost_ms_per_step": (sum(k["dur_ns"] / 1e3 - ref for k in beside) / 1e3 / steps
                             if ref is not None and beside else None),
        "beside": [{"kind": kind, "share": sum(e.values()) / total,
                    "ops": [dict(named([op])[0], share=us / total)
                            for op, us in sorted(e.items(), key=lambda kv: -kv[1])]}
                   for kind, e in sorted(by_kind.items(), key=lambda kv: -sum(kv[1].values()))],
        "reads": named(r["reads"]), "writes": named(r["writes"]), "sources": r["sources"],
    })
  return sorted(rows, key=lambda r: (r["cost_ms_per_step"] is None, -(r["cost_ms_per_step"] or 0),
                                     -r["ms_per_step"]))


def _mib(n):
  if not n:
    return "?"
  return f"{n / 2**20:.0f} MiB" if n >= 10 * 2**20 else f"{n / 2**20:.1g} MiB"


def _size_text(c):
  """A collective's size, and the tensors XLA combined into it."""
  return _mib(c["bytes"]) + (f", {c['tensors']} tensors" if (c.get("tensors") or 0) > 1 else "")


def _collectives_line(traced: list[dict], steps: int) -> str:
  """Per kind: launches per step on each GPU, the median launch, the
  workgroups its kernels launch with and the milliseconds per step no compute
  kernel overlapped."""
  kinds = {}
  for c in traced:
    k = kinds.setdefault(c.get("kind") or "collective", {"launches": 0.0, "us": [], "grid": set(), "exposed": 0.0})
    per_step = max(c.get("gpus") or 1, 1) * steps
    k["launches"] += c["launches"] / per_step
    k["exposed"] += c.get("exposed_us", 0) / 1e3 / per_step
    k["us"] += [c["median_us"]] * c["launches"]
    k["grid"].update(c.get("grid", []))
  parts = []
  for kind, k in sorted(kinds.items(), key=lambda kv: -kv[1]["launches"]):
    grid = "-".join(str(g) for g in sorted({min(k["grid"]), max(k["grid"])})) if k["grid"] else ""
    parts.append(f"{kind} {k['launches']:.0f} per step, median {statistics.median(k['us']) / 1e3:.2f} ms"
                 + (f", {grid} workgroups" if grid else "") + f", {k['exposed']:.1f} ms exposed")
  return "; ".join(parts)


def _overlap_table(rows, traced, steps, width=0):
  line = _collectives_line(traced, steps)
  if line:
    print(f"\ncollectives on each GPU: {line}")
  print(f"GEMM shapes beside collectives (one overlaps {100 * CLEAN:.0f}% or more of a launch), or "
        f"that read or write one: cost = their time beyond the clean median")
  wide = max([len("source")] + [len(_sources(r["sources"])) for r in rows])
  print(f"  {'ms/step':>8} {'cost ms':>8} {'overlapped':>12} {'clean us':>9} {'overlap us':>10} {'slower':>7}  "
        f"{'library':<10} {'M':>7} {'N':>7} {'K':>7}  {'layout':<6} {'pass':<9} {'source':<{wide}}  "
        f"collectives beside it | reads, writes")
  for r in rows:
    fmt = lambda v, w=9: f"{v:{w}.1f}" if v is not None else f"{'-':>{w}}"  # noqa: E731
    slower = (f"{100 * (r['overlapped_us'] / r['clean_us'] - 1):+6.0f}%"
              if r["overlapped_us"] and r["clean_us"] else f"{'-':>7}")
    overlapped = f"{r['overlapped_per_step']:.0f} of {r['launches_per_step']:.0f}"
    beside = ", ".join(f"{b['kind']} {100 * b['share']:.0f}% ({_size_text(b['ops'][0])})" for b in r["beside"])
    data = ", ".join(dict.fromkeys([f"reads {x['kind']} {_size_text(x)}" for x in r["reads"]]
                                   + [f"writes {x['kind']} {_size_text(x)}" for x in r["writes"]]))
    text = (f"  {r['ms_per_step']:8.2f} {fmt(r['cost_ms_per_step'], 8)} {overlapped:>12} {fmt(r['clean_us'])} "
            f"{fmt(r['overlapped_us'], 10)} {slower}  {r['library']:<10} {r['m']:>7} {r['n']:>7} {r['k']:>7}  "
            f"{r['layout']:<6} {','.join(r['passes']):<9} {_sources(r['sources']) or '-':<{wide}}  "
            f"{beside or '-'} | {data or '-'}")
    print(text[:width] if width else text)
  costs = [r["cost_ms_per_step"] for r in rows if r["cost_ms_per_step"] is not None]
  unclean = sum(1 for r in rows if r["clean_us"] is None and r["beside"])
  print(f"  collectives cost these GEMMs {sum(costs):.1f} ms per step against their clean medians"
        + (f"; {unclean} shapes have no clean launch to compare" if unclean else ""))


def _times_table(sites, top=0, width=0):
  timed = [s for s in sites if s.get("time")]
  total = sum(s["time"]["ms_per_step"] for s in timed)
  print(f"  {'ms/step':>8} {'share':>6} {'per step':>8} {'median us':>16} {'clean us':>9} "
        f"{'overlap us':>10} {'TF/s':>6}  {'library':<10} {'M':>7} {'N':>7} {'K':>7}  {'layout':<6} "
        f"{'pass':<5} kernel / source")
  for s in sorted(timed, key=lambda s: -s["time"]["ms_per_step"])[:top or None]:
    t = s["time"]
    ci = t["ci_us"]
    tflops = (s["flops"] or 0) / (t["median_us"] * 1e-6) / 1e12 if t["median_us"] else 0
    fmt = lambda v: f"{v:9.1f}" if v is not None else f"{'-':>9}"  # noqa: E731
    line = (f"  {t['ms_per_step']:8.2f} {100 * t['ms_per_step'] / total if total else 0:5.1f}% "
            f"{t['launches_per_step']:8.1f} {t['median_us']:7.1f} ({ci[0]:.0f}-{ci[1]:.0f}) "
            f"{fmt(t['clean']['median_us'])} {fmt(t['overlapped']['median_us']):>10} {tflops:6.0f}  "
            f"{s['library']:<10} {s['m']:>7} {s['n']:>7} {s['k']:>7}  {s['layout']:<6} {s['pass']:<5} "
            f"{s['kernel']} {s['source']}")
    print(line[:width] if width else line)
  print(f"  measured GEMM time: {total:.2f} ms per step over {len(timed)} of {len(sites)} sites")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("dump", help="An --xla_dump_to directory or one *after_optimizations.txt file.")
  parser.add_argument("--run_dir", default="",
                      help="A JAX-Titan run directory: count launches per training step, over "
                      "the programs a step runs.")
  parser.add_argument("--calls", default="",
                      help="Count launches per step from this file instead: a JSON object "
                      "{module: calls per step}, or the step's lowered.as_text() (IFRT IR, "
                      "MPMD or StableHLO).")
  parser.add_argument("--module", default="", help="Only modules whose file name matches this regex.")
  parser.add_argument("--merge", action="store_true", help="One table over all modules.")
  parser.add_argument("--sites", action="store_true",
                      help="One row per GEMM in the program, with its kernel and op name, "
                      "instead of one per shape.")
  parser.add_argument("--top", type=int, default=0, help="Rows per table; 0 for all.")
  parser.add_argument("--width", type=int, default=0, help="Cut lines to this width; 0 for no limit.")
  parser.add_argument("--csv", default="", help="Write every GEMM site here.")
  parser.add_argument("--json", default="", help="Write every module's GEMM sites and shapes here.")
  parser.add_argument("--times", default="",
                      help="profile.py --kernel_times output of a trace of this program: rank "
                      "sites by measured time.")
  parser.add_argument("--trace", default="",
                      help="A jax.profiler trace of this program, its directory or *.xplane.pb: "
                      "read it here instead of --times, without JAX.")
  parser.add_argument("--gpu", type=int, default=-1,
                      help="With --trace, keep this device only (one process driving several).")
  parser.add_argument("--steps", type=int, default=0,
                      help="Steps in that trace; default: the steps --times records.")
  parser.add_argument("--overlap", action="store_true",
                      help="With --times or --trace: the GEMM shapes collectives overlap, what that "
                      "costs per step, the collectives beside them, and those each reads or writes.")
  args = parser.parse_args(argv)
  if args.times and args.trace:
    parser.error("give --times or --trace, not both")
  if args.overlap and not (args.times or args.trace):
    parser.error("--overlap needs --times or --trace")
  results = analyze(args.dump, args.module)
  if not results:
    print(f"no *after_optimizations.txt under {args.dump}; dump with "
          "XLA_FLAGS='--xla_dump_to=DIR --xla_dump_hlo_as_text'")
    return 1
  unit = "per run"
  if args.run_dir or args.calls:
    try:
      programs = programs_from_run(args.run_dir) if args.run_dir else programs_from_file(args.calls)
    except (OSError, ValueError) as e:
      print(e)
      return 1
    missing = sorted(p for p in set(programs) - {m.name for m, _ in results}
                     if not args.module or re.search(args.module, p))
    results = per_step(results, programs)
    unit = "per step"
    if missing:
      print(f"not in the dump (compiled before it, or cached): {', '.join(missing)}")
  with_gemms = [(m, sites) for m, sites in results if sites or m.other_calls]
  without = len(results) - len(with_gemms)
  everything = [site for _, sites in with_gemms for site in sites]
  timing, overlaps = None, None
  if args.times or args.trace:
    if args.trace:
      kernel_times = trace_times(args.trace, args.steps, args.gpu)
      print(f"trace: {kernel_times['xplane']}")
    else:
      kernel_times = json.loads(pathlib.Path(args.times).read_text())
    timing = attach_times(everything, kernel_times, args.steps)
    coverage = f"{100 * timing['coverage']:.0f}%" if timing["coverage"] is not None else "n/a"
    print(f"times: {timing['kernels_matched']} of {timing['kernels']} GEMM kernels matched launches "
          f"over {timing['steps']} steps; launches traced per expected launch: {coverage}")
    _times_table(everything, args.top, args.width)
    if args.overlap:
      ops = {}
      for module, sites in with_gemms:
        ops.update(link_collectives(module, sites))
      overlaps = overlap(everything, kernel_times, timing["steps"], ops)
      _overlap_table(overlaps[:args.top or None], kernel_times.get("collectives", []), timing["steps"],
                     args.width)
  if args.merge:
    launches, total, split = _summary(everything)
    scope = "in one training step" if args.run_dir or args.calls else "each run once"
    print(f"{_count(len(with_gemms), 'module')}, {scope}: {launches} GEMM launches, "
          f"{_amount(total)}{'; FLOPs by library: ' + split if split else ''}")
    check = _launch_text([m.launch_check for m, _ in with_gemms])
    if check:
      print(check)
    if args.sites:
      _sites_table(everything, args.top, args.width)
    else:
      _table(group(everything)[:args.top or None], total, args.width)
    other = {}
    for m, _ in with_gemms:
      for target, n in m.other_calls.items():
        other[target] = other.get(target, 0) + n
    _other_calls(other)
  else:
    for module, sites in sorted(with_gemms, key=lambda ms: -_summary(ms[1])[1]):
      launches, total, split = _summary(sites)
      calls = module.calls_per_step
      print(f"\n{module.name} ({pathlib.Path(module.path).name})"
            + (f", runs {_count(calls, 'time')} per step" if calls else ""))
      print(f"  {_count(len(sites), 'GEMM')}, {launches} launch{'' if launches == 1 else 'es'}, "
            f"{_amount(total)} {unit}{'; FLOPs by library: ' + split if split else ''}")
      check = _launch_text([module.launch_check])
      if check:
        print(check)
      if args.sites:
        _sites_table(sites, args.top, args.width)
      elif sites:
        _table(group(sites)[:args.top or None], total, args.width)
      _other_calls(module.other_calls)
  if without:
    print(f"\n{without} more modules have no GEMMs")
  if args.csv:
    write_csv(args.csv, everything)
  if args.json:
    out = {"dump": str(args.dump), "counts": unit, "times": timing, "overlap": overlaps, "modules": [
        {"module": m.name, "file": m.path, "platform": m.platform,
         "calls_per_step": m.calls_per_step or None, "gemms": s, "shapes": group(s),
         "other_custom_calls": m.other_calls, "launch_check": m.launch_check}
        for m, s in with_gemms]}
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
