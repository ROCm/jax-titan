# MX quantization

How JAX-Titan trains with MXFP8 or MXFP4 GEMMs: what the recipe quantizes,
which backends run it, and, for the XLA and hipBLASLt backend, how to turn it
on, what it needs and how to check which kernels ran. The recipe is the same
for every backend, and each backend has its own section.

## The recipe

An MX format stores a tensor in blocks of 32 values, each block with one
power-of-two scale of 8 bits (E8M0). The values are E4M3 (8 bits) in MXFP8 and
E2M1 (4 bits) in MXFP4.

The recipe is MaxText's `jax_mxfp8` and `jax_mxfp4`. It quantizes both
operands of a GEMM just before the GEMM, in blocks along the dimension the
GEMM contracts, in three steps:

1. **Amax.** The largest magnitude of each block.
2. **Scale encode.** That maximum divided by the format's largest value,
   rounded up to a power of two and stored as E8M0.
3. **Quantize.** The block divided by its scale, clipped to the format's range
   and cast to E4M3 or E2M1.

The GEMM then runs on the quantized values and their scales, and returns BF16.

- **Quantized GEMMs.** `linear` and the three GEMMs of `swiglu`: in
  JAX-Titan's Llama, the query, key, value and output projections and the MLP.
  Attention, the norms, the embedding and the output head stay BF16.
- **Gradients.** The backward pass of a quantized GEMM is two more GEMMs, for
  the gradient of its input and of its weight, and both are quantized the same
  way. Each of the three GEMMs contracts another dimension, so the input, the
  weight and the incoming gradient are each quantized twice, once per block
  direction. A gradient passes through the quantization unchanged.
- **Nothing quantized is kept.** Parameters and optimizer state stay float32,
  and every step quantizes the current BF16 values again.
- **Sizes.** The rows of the input and both dimensions of the weight must be
  multiples of 32, since each is the contracted dimension of one of the three
  GEMMs. A provider declines any other call, which then runs in BF16.

## Backends

A backend says who quantizes and who runs the GEMM. In the code it is a set of
operator providers, which `--ops` selects per op.

| Backend | Providers | Quantization | GEMM | GPUs |
|---|---|---|---|---|
| XLA and hipBLASLt | `xla_mxfp8`, `xla_mxfp4` | XLA, from JAX ops | hipBLASLt's MX GEMM, or what XLA measures faster | gfx950 |

Two more MX providers exist and have no section here yet: `te_mxfp8`
(Transformer Engine's MXFP8 GEMMs) and `aiter_mxfp4` (JAX-AITER's MXFP4 GEMM,
whose output is wrong at JAX-AITER's pinned revision).
[pipelines.md](pipelines.md) has their conformance results, under Operator
providers.

## XLA and hipBLASLt

### Turning it on

`xla_mxfp8` and `xla_mxfp4` call `jax.nn.scaled_dot_general`, which writes the
quantization as JAX ops and the GEMM as XLA's scaled dot. XLA compiles both,
so the backend needs no kernel library beyond ROCm's. Name the provider for
`linear` and `swiglu`:

```bash
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS \
  --xla_gpu_executable_terminate_timeout=3600 \
  --xla_gpu_first_collective_call_terminate_timeout_seconds=3600" \
  python -m jax_titan.trainer.train --parallel=fsdp --microbatches=8 \
  --microbatch_size=8 --steps=40 --step_timeout=0 \
  --ops=linear=xla_mxfp8,swiglu=xla_mxfp8
```

`xla_mxfp4` gives MXFP4, and the two ops can take different providers. In
Python, `ops.set_policy(...)` and `with ops.policy(...):` take the same
string. The timeouts are for the slow first step, under Pitfalls.

### Requirements

- **GPU.** gfx950, the MI350 series. On older GPUs XLA does not offer
  hipBLASLt's MX GEMMs.
- **Stack.** The JAX and XLA revisions of `scripts/setup_sources.sh`, without
  a patch for MX, and ROCm 10.1 from TheRock with hipBLASLt 1.4.1.
- **XLA flags.** Autotuning and XLA's Triton GEMMs, both on in
  `$JAX_TITAN_ROCM_XLA_FLAGS` (`--xla_gpu_autotune_level=4`,
  `--xla_gpu_enable_triton_gemm=true`).

### The steps

| Step | Runs as |
|---|---|
| Amax, Scale encode, Quantize | Kernels that XLA compiles from the JAX ops |
| GEMM | For each GEMM, the fastest of three candidates that XLA's autotuner times while it compiles: hipBLASLt's MX GEMM, XLA's own Triton kernel for the scaled dot, or a BF16 GEMM on operands that XLA dequantizes first |

hipBLASLt gets one scale per 32 values, in the order of the operand. This XLA
has no swizzled scale layout for hipBLASLt.

### Coverage

XLA offers a scaled dot to hipBLASLt when it has no batch dimension, M and N
are multiples of 16 and K is a multiple of 32; the sizes of the recipe meet
the last two. Which candidate then runs is a matter of measured time. In the
FSDP step of Llama 3.1 8B over 8 MI350X, all 47 quantized GEMMs of MXFP4 ran
on quantized operands, and 26 of the 47 of MXFP8; the other 21 ran as BF16
GEMMs on dequantized data
([quantization_vs_maxtext.md](quantization_vs_maxtext.md#gemms)).

### Checking the kernels

- **The provider ran.** The trainer prints, under `ops:`, the provider each op
  ran on and every provider it skipped, with the reason.
- **The GEMM kernels.** `observability.gemms` lists every GEMM of the compiled
  program from an HLO dump
  ([metrics.md](metrics.md#gemms-in-the-compiled-program)). Its `library`
  column names hipBLASLt or Triton, and its `dtype` column shows E4M3 or E2M1
  operands for a GEMM on quantized data and BF16 for a dequantized one.
- **The numbers.** `ops.conformance` compares outputs and gradients with
  float32, here on one GPU. A provider passes below 6% of error per GEMM for
  MXFP8 and 25% for MXFP4; `linear` measures 3.8% and 16%.

  ```bash
  python -m jax_titan.ops.conformance --platform=rocm --modes=single \
    --providers=xla_mxfp8,xla_mxfp4
  ```

### Pitfalls

- **No MX kernel without XLA's Triton GEMMs.** With
  `--xla_gpu_enable_triton_gemm=false`, XLA dequantizes every scaled dot and
  runs a BF16 GEMM, before the autotuner sees it. The result still has the
  quantization error of the recipe. `pytest --gpus` runs this way, so the GPU
  tests check the numbers of the recipe, not the MX kernels.
- **A slow first step.** With FSDP over 8 GPUs, MXFP8 compiles for 5 minutes
  and its first step takes 8; MXFP4 compiles for 21 minutes and its first step
  takes 36. Turn the trainer's `--step_timeout` off, and set the two XLA
  timeouts of the command above longer than the first step.
- **A declined call does not fail.** With an input that is not BF16, or sizes
  that are not multiples of 32, the op runs in BF16 and training goes on. Only
  the `ops:` lines show it.
- **No MXFP8 training on CPU.** An all-zero block gets the smallest scale,
  which XLA's CPU backend flushes to zero, and the block becomes 0 divided by
  0 ([pipelines.md](pipelines.md), Quantized GEMMs compiled by XLA).

## Results

On Llama 3.1 8B over 8 MI350X (2026-11-12), MXFP8 and MXFP4 ran at 0.91x to
1.04x of BF16's tokens per second per GPU, and 500 steps of training ended
about 1% above BF16's loss, which is inside what another BF16 seed does. The
tables, with the same recipes in MaxText, are in
[quantization_vs_maxtext.md](quantization_vs_maxtext.md).
