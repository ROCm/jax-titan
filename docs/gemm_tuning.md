# GEMM tuning

How to find and apply faster hipBLASLt kernels for the GEMMs a JAX program
runs on ROCm, check that they take effect, and see where the rest of the GEMM
time goes. The tools work on any JAX program; `scripts/tune_gemms.sh` chains
them for a JAX-Titan training configuration.

## How kernel selection reaches XLA

XLA lowers most GEMMs on ROCm to hipBLASLt custom calls
(`__cublas$lt$matmul`). At `--xla_gpu_autotune_level=0`, with Triton GEMMs
off, the GEMM thunk asks hipBLASLt for one solution at run time and runs the
heuristic's first choice
([gpublas_lt_matmul_thunk.cc](https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.cc)).
`HIPBLASLT_TUNING_OVERRIDE_FILE` makes hipBLASLt return a tuned solution
first for every problem the file lists, so that is what runs:

- **No code change and no recompilation.** The executable holds no solution
  index at level 0, so a cached executable picks up the file too.
- **One library build.** Rows hold solution indices of one hipBLASLt build.
  The file's first line must be `Git Version: <hipblasLtGetGitRevision>`; with
  any other version, hipBLASLt ignores the whole file. Before rocm-libraries
  #12585 (not in TheRock 7.14 or 10.1) there is no per-row check.
- **Exact problems.** hipBLASLt looks a row up by transposes, batch count, m,
  n, k, the A, B and C types and the compute type, in its column-major terms.
  Leading dimensions, strides, the D type and the epilogue are not part of the
  key.
- **Level 0 only.** At level 1 and above XLA times its own candidate list and
  replaces most tuned picks.

## The workflow

```bash
source scripts/env.sh
scripts/build_hipblaslt_sweep.sh          # once per container
scripts/tune_gemms.sh OUT --model=llama3.1-8b --parallel=fsdp --ops=attention=te \
  --data=synthetic --seq_len=2048 --microbatches=1 --microbatch_size=32 --no-remat
```

Each stage writes into `OUT`:

| Stage | Output | What to check |
|---|---|---|
| `clock` | `clock.json` | The engine clock and power one GPU sustains on a long bf16 GEMM: under a power cap the bound uses this clock, not the peak |
| `capture` | `gemms.txt`, `problems.json`, `capture.txt` | The GEMM sites of a step from its HLO dump, the hipBLASLt problems from `HIPBLASLT_LOG_MASK=32`, and the join between them: launches per step and the workspace XLA gives each call. Unmatched sites or problems mean a descriptor the key misses |
| `sweep` | `sweep/report.md`, `sweep/results.json` | Per problem: whether XLA ran heuristic #0 at its workspace (`XLA = #0`, the position check), the default and the best solution with their times, and whether the best was adopted |
| `write` | `tuning.txt` | The override rows: only captured problems whose winner was adopted |
| `verify` | `verify.json` | Every row's problem ran the tuned solution in the step (`hit`), and heuristic #0 switches through the API |
| `ab` | `ab/ab.json` | Step time without and with the file, over interleaved repeats |
| `profile` | `profile/*_gemms.txt` | Measured time per GEMM site, the median of launches no collective overlapped (`clean`) and of mostly overlapped ones |
| `cost` | `cost.md` | Per problem and per step: selection, contention, in-step other, and ceiling (below) |
| `negative` | `negative/summary.json` | Each way the file can fail shows up: a wrong git version (no hits), a wrong key field (the row goes unused), too much workspace (no hit), and an executable from the compilation cache built without the file (all hits) |

A solution that wins alone can still lose in the step: beside a collective
that shares the GPU, one kernel can slow down more than another. With traces
of the step without and with the file (`profile`), `hipblaslt.py instep`
compares each row's in-step launches and writes the rows that are not slower,
for a second A/B:

```bash
python3 src/jax_titan/tuning/hipblaslt.py instep OUT/profile/base_gemms.json \
  OUT/profile/tuned_gemms.json --override OUT/tuning.txt --out OUT/tuning_refined.txt
```

`sweep` adopts a solution only when its 95% bootstrap interval over separate
processes separates from the default's and it is at least 2% faster
(`--min_gain`). Stage 1 times every supported solution once at XLA's
workspace (`--stage1 screen`, the default), or runs `hipblaslt-bench` in
tuning mode (`--stage1 bench`, `HIPBLASLT_TUNING_FILE`), the hipBLASLt team's
method, when a bench built for the same library is at hand. Stage 2 times the
default, stage 1's best and runners-up, and `--seeds`, in `--trials` processes
of `--rounds` interleaved rounds each, after `--warmup_ms` of GEMMs that bring
the GPU to its power-limited clock.

## Any JAX program

The tools take files, so they serve programs other than the trainer:

```bash
XLA_FLAGS="$FLAGS --xla_dump_to=DUMP --xla_dump_hlo_as_text" \
  HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=hbl_%i.log python my_step.py
python3 src/jax_titan/observability/gemms.py DUMP --module jit_train_step --merge --json gemms.json
python3 src/jax_titan/tuning/hipblaslt.py capture "hbl_*.log" --json problems.json
python3 src/jax_titan/tuning/hipblaslt.py join problems.json gemms.json --steps 3
python3 src/jax_titan/tuning/hipblaslt.py sweep problems.json --out sweep \
  --sweep_bin ${JAX_TITAN_ROOT}/bin/hipblaslt_sweep --gpus 0,1,2,3,4,5,6,7
python3 src/jax_titan/tuning/hipblaslt.py write sweep/results.json --out tuning.txt
HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=v_%i.log python my_step.py
python3 src/jax_titan/tuning/hipblaslt.py verify "v_*.log" --override tuning.txt
```

`tuning/hipblaslt.py`, `observability/gemms.py` and `planning/gemm_cost.py`
use only the standard library, so they run in any container, including one
with an older JAX than JAX-Titan's. `map` carries winners to another hipBLASLt
build by solution and kernel name, ignoring tokens that only the library
version changes. `observability/gemm_bench.py` times problems in isolation
through JAX under named XLA flag sets (hipBLASLt default or tuned, XLA's
autotuner, XLA's Triton GEMMs) or JAX-Titan operator providers.

## Where the GEMM time goes

`planning/gemm_cost.py` keeps four times per launch apart, and turns their
differences into milliseconds per step:

- **selection** = t_default − t_best, isolated: what a tuning file recovers;
- **contention** = t_now − t_clean: what collectives running alongside the
  GEMM cost, from the in-step launches no collective overlapped;
- **in-step other** = t_clean − t_default: the rest of the gap between the
  step and an isolated run (clocks, caches); it can be negative;
- **ceiling** = t_best − t_bound: what no existing kernel recovers. The bound
  is FLOPs at the peak for the inputs' precision, or bytes at HBM bandwidth,
  so it ignores tile and wave quantization; treat it as speculative. bf16
  GEMMs are bound at the clock `clock` measures; f32 GEMMs at the FP32 matrix
  peak (where `planning/gpus.py` has one) and the peak clock, since they draw
  less power: on MI300X the proxy's f32 GEMMs reach 112 TFLOP/s, more than the
  FP32 peak allows at the bf16 GEMM's capped clock.

## Results: Llama 3.1 8B, FSDP over 8 MI300X

8,192 tokens per GPU (`--microbatch_size=32 --seq_len=2048`), TE attention,
no rematerialization; JAX 0.12.1.dev built from main, TheRock 10.1 (hipBLASLt
1.4.1, git 39d8d603), 750 W power cap. The files are in
[results/2026-11-15/gemm_tuning_mi300x_llama3.1-8b_fsdp8](results/2026-11-15/gemm_tuning_mi300x_llama3.1-8b_fsdp8).

| Stage | Result |
|---|---|
| clock | 8192³ bf16 GEMMs sustain 1.24-1.25 GHz at 750 W, 642-645 TFLOP/s (the dense peak at 2.1 GHz is 1,307) |
| capture | 20 GEMM sites, 15 hipBLASLt problems; every call gets a 76 MiB workspace, and XLA ran heuristic #0 for each |
| sweep | 3 of 15 problems have a separated winner at least 2% faster (2.1%, 2.7% and 4.3%); 11 more are within 1.3% of the default. Predicted saving: 1.25 ms per step |
| verify | 3 of 3 rows hit in the step and through the API |
| ab | 1.1240 s without and 1.1266 s with the file (3 interleaved runs each): no separable change, as predicted |
| profile | GEMMs take 752 ms of the 1.124 s step; all 20 sites matched their launches |
| cost | selection 4.7, contention 3.6, in-step other 27.5, ceiling 157 ms per step |
| negative | a wrong git version: no hits and nothing logged; a wrong key: that row unused; a cached executable: all hits |

The default kernels are close to the best this hipBLASLt has for these
shapes, so a tuning file buys nothing here; the tools say so before an A/B
does. What the cost breakdown shows instead:

- **f32 LM-head gradients.** The logits GEMM runs bf16 to f32
  (`preferred_element_type=jnp.float32`), so its two backward GEMMs get an f32
  cotangent and run f32 by f32: 147 ms per step at about 115 TFLOP/s, 13% of
  the step, where hipBLASLt's default is already its best. Casting the
  cotangent to bf16 in a custom VJP would make them bf16 GEMMs, roughly
  120 ms faster per step; that trades logit-gradient precision and is a model
  decision, not a tuning one.
- **In-step slowdown without collectives.** Some GEMMs run up to 17% slower
  in the step than alone even when no collective overlaps them (in-step other,
  27.5 ms per step), beyond what selection (4.7 ms) recovers.

## Pitfalls

- `HIPBLASLT_LOG_LEVEL` overrides `HIPBLASLT_LOG_MASK`; leave it unset. The
  mask-64 YAML is written only when the process exits cleanly.
- XLA can put a weight-gradient GEMM in a dynamic-slice fusion that writes
  straight into the scan's stacked gradients. `gemms.py` names the site after
  the fusion; the profiler names the launch after the library call inside it,
  which `--times` also matches.
- A wrong `Git Version` line disables the whole file without a log line; only
  `verify` shows it.
- With the file set, XLA's compilation takes longer (about 20 s more for a
  program with 50 hipBLASLt calls on TheRock 10.1): XLA queries hipBLASLt for
  each call while compiling. An executable from the compilation cache skips
  that and still picks the file up.
- Sweep at the workspace XLA gives the call (`capture` records it). A winner
  that needs more is not supported at run time and the call silently falls
  back to the default; `negative` checks this.
- One process driving several GPUs can lose profiler events on ROCm; the
  `--times` table prints the share of expected launches the trace holds.
- Inside Docker, the HIP device index need not match `amd-smi`'s; `clock`
  reads the busiest GPU instead of trusting the index.
