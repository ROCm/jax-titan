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
| Own kernel | Gluon | jax-triton, `ops` provider `gluon` (gfx950 only) | As Triton, plus layouts written per architecture |
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
before timing, for the kernel alone.

## Putting a kernel into a model

JAX-Titan's operator providers show the three integration patterns; each
backs `ops.linear`, which models call instead of `x @ w`:

- **jax-triton** ([ops/providers/triton.py](../jax_titan/ops/providers/triton.py)):
  `triton_call` passes scalars without divisibility hints, so a kernel that
  takes strides as arguments cannot prove its loads contiguous. A thin
  `@triton.jit` wrapper with constexpr sizes and strides calls the library
  kernel instead ([_triton_gemm.py](../jax_titan/ops/providers/_triton_gemm.py)),
  which also lets the backward GEMMs read the forward's operands in place.
- **XLA FFI with an ahead-of-time kernel**
  ([ops/providers/flydsl.py](../jax_titan/ops/providers/flydsl.py)):
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
