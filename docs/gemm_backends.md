# GEMM backends beyond hipBLASLt

XLA sends almost every GEMM of a JAX program on ROCm to hipBLASLt. This page
shows how to time the same GEMMs on other backends, what it takes to put a
faster one into a model, and what each choice commits you to. It is a
reference for teams that own their training code; tuning hipBLASLt's own
selection, which needs no code change, is in [gemm_tuning.md](gemm_tuning.md).

## Backends and what adopting each costs

| Tier | Backend | How a JAX program reaches it | What it costs to adopt |
|---|---|---|---|
| Configuration | hipBLASLt, tuned selection | `HIPBLASLT_TUNING_OVERRIDE_FILE` at autotune level 0 ([gemm_tuning.md](gemm_tuning.md)) | A file per hipBLASLt build; no code |
| Configuration | XLA's Triton GEMMs | `--xla_gpu_enable_triton_gemm=true` with autotuning | Compile time; picks can change between compiles |
| Vendor library | JAX-AITER BF16 ASM GEMM | `jax_aiter.gemm.gemm`, an FFI call with autodiff and sharding rules | A JAX-AITER build per ROCm and GPU; K a multiple of 64 |
| Vendor library | Transformer Engine GEMM | `ops` provider `te` (`linear`) | TE wheels per ROCm; hipBLASLt underneath |
| Own kernel | Triton (AITER's BF16 GEMM) | jax-triton `triton_call`, `ops` provider `triton` (`linear`) | Triton and jax-triton versions; sharding and autodiff rules you write |
| Own kernel | Gluon | jax-triton, `ops` providers `gluon`, `gluon_intra_mfma_on` and `gluon_intra_mfma_off` (gfx950 only) | As Triton, plus layouts written per architecture |
| Own kernel | FlyDSL | Exported ahead of time to a C object, called through XLA's FFI, `ops` provider `flydsl` | A build step per shape set; your FFI handler; sharding and autodiff rules |

Own kernels pay off only when they beat the library on the shapes that
dominate the step, by more than the layout copies they add, in the step and
not just alone.

## Comparing backends on your GEMMs

1. List the GEMMs of a step and rank them by time: `observability/gemms.py`
   on an HLO dump, with `--times` from a trace ([gemm_tuning.md](gemm_tuning.md)).
2. Capture them as hipBLASLt problems (`tuning/hipblaslt.py capture`) or write
   a spec list of row-major GEMMs.
3. Time them on every backend:

```bash
source scripts/env.sh
scripts/install_gfx950_triton.sh
python -m jax_titan.ops.providers.flydsl build OUT/flydsl --shapes 14336x4096,4096x14336
OVERRIDE=tuning.txt JAX_TITAN_FLYDSL_LIB=OUT/flydsl/libjax_titan_flydsl.so \
  JAX_AITER_ROOT=third_party/jax-aiter \
  scripts/compare_gemm_backends.sh problems.json OUT/backends
```

`observability/gemm_bench.py` runs each backend in its own process, so XLA
flags apply per backend, times pipelined calls over inputs larger than the
caches, and checks every result against an f32 GEMM. A GEMM stored with an
operand transposed (an activation or weight gradient) runs in that layout:
backends that need another layout pay for the transposes in the timed program
and say how many (`nT` in the table). `flydsl_kernel` converts the inputs
before timing, for the kernel alone. An `ops:<provider>` backend uses
`ops.linear` directly for `MK,KN`, and its VJP for `MK,NK` activation gradients
and `KM,KN` weight gradients; fallback to another provider is reported as a
skip rather than timed under the requested name.

The legacy `gluon` provider is the tutorial's 8-wave inter-wave kernel.
`gluon_intra_mfma_on` and `gluon_intra_mfma_off` use the same 4-wave
intra-wave v9 kernel with Triton's `mfma-schedule` hint enabled and disabled.
The amdgcnas plugin stays off, so this pair changes only MFMASchedule. These
benchmarks require Triton tag `gfx950-tutorial-v3.0` at revision `3b0c7f081`;
PyPI Triton 3.9.0 accepts the hint but does not run the pass.

Every intra-wave specialization is gated on its post-pass LLVM IR. The enabled
artifact must contain `llvm.amdgcn.sched.barrier` calls, the disabled artifact
must contain none, and their SHA-256 hashes must differ. The barrier counts,
hashes, compiler version and compiler revision are stored under
`scheduler_proof` in every row of `backends.json`; `scheduler_ab` records the
cross-provider check. A missing or ineffective transformation is reported as a
skip and makes `compare_gemm_backends.sh` fail instead of publishing a timing.

`backends.md` lists every problem's median, trial interval, TFLOP/s and
speedup against `BASELINE` (`xla_hipblaslt` by default). Its final table
combines the same rows with the actual HLO `launches_per_step`:

```text
modeled speedup = sum(baseline_ms * launches) / sum(candidate_ms * launches)
effective TFLOP/s = sum(FLOPs * launches) / sum(candidate_seconds * launches)
```

Coverage is printed beside each overall result; a provider missing a problem
is never compared as if it covered the full step. To remove cross-GPU clock
variation, put the desired backends in one semicolon group so they run
sequentially on one GPU:

```bash
BACKEND_GROUPS='xla_hipblaslt,ops:te,ops:gluon,ops:gluon_intra_mfma_on,ops:gluon_intra_mfma_off' \
  scripts/compare_gemm_backends.sh problems.json OUT/backends
```

## Putting a kernel into a model

JAX-Titan's operator providers show the three integration patterns; each
backs `ops.linear`, which models call instead of `x @ w`:

- **jax-triton** ([ops/providers/triton.py](../src/jax_titan/ops/providers/triton.py)):
  `triton_call` passes scalars without divisibility hints, so a kernel that
  takes strides as arguments cannot prove its loads contiguous. A thin
  `@triton.jit` wrapper with constexpr sizes and strides calls the library
  kernel instead ([_triton_gemm.py](../src/jax_titan/ops/providers/_triton_gemm.py)),
  which also lets the backward GEMMs read the forward's operands in place.
- **XLA FFI with an ahead-of-time kernel**
  ([ops/providers/flydsl.py](../src/jax_titan/ops/providers/flydsl.py)):
  FlyDSL's `export_to_c` writes a C object and header per specialization;
  a generated handler binds XLA's buffers and stream to the exported call,
  and `jax.ffi.ffi_call` runs it. Nothing of FlyDSL or Python runs in the
  training process.
- **A vendor JAX extension** (JAX-AITER's `gemm`, used directly by
  `gemm_bench`): FFI calls with their own `custom_vjp` and
  `custom_partitioning`.

Each needs the same three things XLA gives its own GEMMs for free:

- **Autodiff.** A `custom_vjp` whose backward computes the activation
  gradient (`dy` times `w` transposed) and the weight gradient (`x`
  transposed times `dy`), in layouts the kernel takes or after copies.
- **Sharding.** `custom_partitioning` (`ops/partitioned.py`): name each
  dimension, and a dimension contracted over a sharded axis becomes a psum.
  Under FSDP the weight gradient then ends in an all-reduce where XLA's own
  GEMM feeds a reduce-scatter; keeping the weight gradient on XLA, as the
  `flydsl` provider does, avoids that.
- **Scans and rematerialization.** FFI calls and `triton_call`s are opaque to
  XLA's fusion and scheduling; layout copies around them cost HBM traffic
  that XLA would otherwise fuse away.

## Results

On 8 MI300X, the GEMMs of Llama 3.1 8B under FSDP at 8,192 tokens per GPU:
to be filled in from `compare_gemm_backends.sh`.
