# Pipelines

A JAX-Titan pipeline step is one Shardy MPMD program. Each stage is a named
computation on its own mesh, and each microbatch is an `mpmd.call` with its
own call counter. Shardy splits the step into fragments, one XLA program per
stage, microbatch and direction, and orders them with a pipeline schedule.
The whole step runs as one IFRT IR program; every stage keeps its own
optimizer state on its mesh.

## Schedules

`--schedule` names an entry of Shardy's pipeline registry. Every schedule
below matches single-mesh `jax.jit` in the CPU parity tests (`tests/`).

| Schedule | Order | Notes |
|---|---|---|
| `ZERO_BUBBLE_H1` (default) | 1F1B, with each backward split into an input-gradient and a weight-gradient fragment; weight gradients fill the bubble | Fastest on MI355X in every layout tried |
| `ZERO_BUBBLE_H2_*` | Zero-bubble with more forward fragments in flight | Same speed as H1 on 2 stages |
| `ONE_FWD_ONE_BWD` | 1F1B | |
| `GPIPE` | All forward fragments, then all backward fragments | Same speed as 1F1B on 2 stages |
| `GPIPE_BUT_1F1B_FOR_LAST_MESH` | GPipe, 1F1B on the last mesh | |
| `CIRCULAR`, `CIRCULAR_WITH_REVERSED_BACKWARD` | Several stages per mesh, all forward then all backward | Needs `--stages_per_mesh` > 1 |

With a schedule, JAX-Titan sets the Shardy option
`mpmd_merge_inferred_after_scheduling`. Without it, Shardy merges the
optimizer update into microbatch 0's backward fragment before scheduling,
so that fragment runs after every other backward fragment on every mesh.
jaxlib only passes this option on with
`third_party/patches/shardy-mpmd-parse-partitioning-options.patch`.

Shardy logs each mesh's fragment order when it compiles a step, in lines
starting `Module jit_train_step on mesh`. With 2 stages and 4 microbatches,
1F1B now runs F0 F1 B0 F2 B1 F3 B2 B3 on the first mesh and F0 B0 F1 B1 F2
B2 F3 B3 on the second.

## Inside a stage

Each stage mesh has a `data` and a `tensor` axis.

- **Data parallel** (default): each stage's parameters and optimizer state
  are replicated on its devices. `--reduce_grads=step` (default) gives each
  device its own parameter copy, so gradients are summed across devices once
  per step; `--reduce_grads=microbatch` all-reduces them in every
  microbatch's backward pass.
- **`--stage_fsdp`**: parameters and optimizer state are sharded across the
  stage's data axis. With `--reduce_grads=step`, each device all-gathers its
  parameter copy once per step and gradients are reduce-scattered once per
  step, as with ZeRO-2. With `--reduce_grads=microbatch`, weights are
  all-gathered and gradients reduce-scattered in every microbatch, as with
  FSDP (ZeRO-3).
- **`--tensor_parallel=N`**: Megatron-style tensor parallelism over the
  stage's `N` consecutive devices. Attention heads, MLP columns and the
  output vocabulary are sharded; `jax_titan.parallel` derives the shardings
  from the logical axes `models/llama.py` gives each parameter.
- **`--stage_layers=17,15`**: decoder layers per stage. The last stage also
  runs the output head and loss, so giving it fewer layers balances the
  stages.

## Checkpoints

`--checkpoint_dir=DIR` saves parameters and optimizer state with Orbax after
the last step, and every `--checkpoint_every` steps; rerunning the same
command resumes from the latest checkpoint in `DIR`, including the data
position. Each stage's state is saved from, and restored onto, its own mesh.
Restoring places arrays on the shardings of the current command, so a run
can resume with another layout of the same stages, for example with
`--stage_fsdp` added.

- On CPU, a run resumed from step 3 matches the uninterrupted run exactly.
- On 8x MI355X, Llama 3.1 8B resumed from step 10 matches the uninterrupted
  run to 1e-7 at step 10 and exactly at step 11.
- A Llama 3.1 8B checkpoint is 80 GB. Saving it blocked training for 31-39 s
  while the arrays were copied to host memory, and the step after one save
  took 41 s while Orbax wrote it out in the background. Restoring from
  `/dev/shm` took 60 s. Checkpoint less often than every few minutes until
  saving is faster.

## Results (2026-11-05, 8x MI355X)

Llama 3.1 8B with 65,536 tokens per step (8 microbatches of 4 sequences of
2,048 tokens), full remat, XLA attention and AdamW on WikiText-103. Every run
starts from the same weights and trains on the same batches; all step-0
losses are 12.59. Step times are medians over steps 5-29. Per-step metrics
are in [results/2026-11-05](results/2026-11-05). The stack is jax
`6b39d495e5` and XLA `d23c30982a` with the patches in
`third_party/patches`.

Pipelines with 2 stages of 4 GPUs:

| Schedule | Inside each stage | Extra flags | Step | Tokens/s | MFU | Peak HBM per GPU |
|---|---|---|---|---|---|---|
| Zero-bubble H1 | DP, 16 and 16 layers | | 1.454 s | 45.1k | 10.5% | 180-181 GiB |
| Zero-bubble H1 | DP, 17 and 15 layers | `--stage_layers=17,15` | 1.426 s | 46.0k | 10.7% | 171-190 GiB |
| Zero-bubble H1 | FSDP, reduce per step | `--stage_fsdp` | 1.528 s | 42.9k | 10.0% | 141-143 GiB |
| Zero-bubble H1 | TP4 | `--tensor_parallel=4` | 1.788 s | 36.7k | 8.5% | 45-60 GiB |
| 1F1B | DP | `--schedule=ONE_FWD_ONE_BWD` | 1.684 s | 38.9k | 9.1% | 181-182 GiB |
| 1F1B | FSDP, reduce per microbatch | `--schedule=ONE_FWD_ONE_BWD --stage_fsdp --reduce_grads=microbatch` | 2.550 s | 25.7k | 6.0% | 43-47 GiB |
| GPipe | DP | `--schedule=GPIPE` | 1.692 s | 38.7k | 9.0% | 181-182 GiB |

Other pipeline shapes, and the same step without a pipeline:

| Layout | Flags | Step | Tokens/s | MFU | Peak HBM per GPU |
|---|---|---|---|---|---|
| Zero-bubble H1, 4 stages of 2 GPUs | `--num_stages=4` | 1.610 s | 40.7k | 9.5% | 81-105 GiB |
| Circular, 4 stages on 2 meshes of 4 GPUs | `--num_stages=4 --stages_per_mesh=2 --schedule=CIRCULAR` | 1.709 s | 38.3k | 8.9% | 164-171 GiB |
| No pipeline, data parallel over 8 GPUs | `--parallel=dp` | 0.833 s | 78.7k | 18.4% | 127 GiB |
| No pipeline, FSDP over 8 GPUs | `--parallel=fsdp` | 0.795 s | 82.4k | 19.2% | 26 GiB |

Compiling a 2-stage pipeline step takes 66-116 s; MPMD programs do not use
JAX's persistent compilation cache yet. The no-pipeline programs compile in
37-129 s and come from the cache after the first run.

### What the numbers show

- **Zero-bubble H1 is the fastest schedule**, 1.16x faster than 1F1B at the
  same memory. Giving the last stage one layer fewer adds 2%: that stage also
  runs the 128k-vocabulary output head and the loss, about 11% more FLOPs.
- **The bubble, from step times.** With 4 and 8 microbatches, step time is
  a fixed cost plus 0.16 s per microbatch for both schedules. The fixed cost
  is 0.47 s for 1F1B and 0.24 s for zero-bubble H1, 27% and 16% of the step
  at 8 microbatches. 1F1B's theoretical bubble with 2 stages and 8
  microbatches is 1/9 of the step; the rest of the fixed cost is
  once-per-step work, such as the gradient all-reduce and the optimizer
  update. 16 microbatches ran out of memory on the last stage. These runs
  predate the switch back to casting before the scan, which made steps
  2-3% faster.
- **The bubble, from traces.** In a zero-bubble trace, the last stage's GPUs
  are idle 14% of the time and the first stage's 32%; the first stage waits
  on the heavier last stage. Collectives take 10% and cross-mesh copies 1%.
- **FSDP inside stages** saves 40 GiB per GPU for 5% more step time when
  gradients are reduced once per step. Reducing them in every microbatch, as
  FSDP usually does, cuts memory to 43-47 GiB, a quarter of the data-parallel
  layout, at 1.5x the step time. Larger models need one of these to fit.
- **TP4 inside stages** uses a quarter to a third of the memory at 1.23x the
  step time. Tensor-parallel all-reduces are 22% of a TP2 trace, not
  overlapped with compute.
- **Cross-mesh transfers are cheap here.** Each microbatch moves 16 MiB of
  activations per GPU in about 0.29 ms (58 GB/s, one xGMI link), 0.3% of a
  step. rocprofiler reports them as `MemcpyH2D` on the receiving GPU, with
  no copy on the sender.
- **The pipeline is still 1.8x slower than FSDP on one node.** Besides the
  fixed per-step cost, each GPU runs 14.2k kernels per step against 4.0k for
  FSDP, because it processes one sequence at a time; the same FLOPs take more,
  smaller kernels. Pipelines are for models and clusters that outgrow FSDP,
  such as multi-node runs.
- **Tried and dropped:** casting the weights to bf16 inside the
  rematerialized layer, because the backward pass then casts again: 2-3%
  slower with 2 stages, 19% slower with 4, and 5-10 GiB more per GPU. XLA
  command buffers were 2% slower.

Earlier measurements on 2026-11-04 with 1F1B: reducing gradients once per
step instead of in every microbatch made the step 1.3x faster (2.20 s to
1.70 s) for 10-15 GiB more memory per GPU. Their logs are in
[results](results).

## Global gradient clipping and MaxText's own trainer (2026-11-07, 8x MI355X)

Since 2026-11-07 every pipeline clips by one gradient norm over all stages
(the global norm of `optax.clip_by_global_norm`, which per-stage clipping
only approximated) and skips the update when the norm is not finite; every
result above used per-stage clipping. Medians over steps 5-29, Llama 3.1 8B
with 65,536 tokens per step, 2 stages of 4 GPUs, zero-bubble H1, gradients
reduced per step; per-step records are in
[results/2026-11-07](results/2026-11-07).

| Attention | Step | Against per-stage clipping |
|---|---|---|
| native | 1.502 s | 1.457 s (+3.1%) |
| TE fused attention | 1.310 s | 1.26 s (+4%) |

MaxText's own trainer (`maxtext.trainers.pre_train.train`, MaxText 0.2.5,
synthetic data, the same 65,536 tokens per step, full remat), with MaxText's
attention (`attention=dot_product`) unless the row says TE:

| MaxText trainer layout | Step | Tokens/s per GPU | JAX-Titan's Llama as a pipeline |
|---|---|---|---|
| FSDP over 8 GPUs | 0.723 s | 11.3k | |
| Its pipeline: 2 stages x 4 GPUs (FSDP inside), 8 microbatches, 16 layers per stage | 1.763 s | 4.6k | 1.502 s |
| Its pipeline, 2026-11-09 | 1.755 s | 4.7k | |
| Its pipeline with TE fused attention (`attention=cudnn_flash_te`), 2026-11-09 | 1.538 s | 5.3k | With TE, the same day: 1.278 s |

- **Global clipping costs 3-4%.** As first written it cost 16% with per-step
  reduction (1.70 s; an A/B at the previous two commits, 1.457 s at
  `d18cbca` and 1.452 s at `e9dc50f`, ruled out the operator work). Traces
  showed the same compute and exposed communication per step but 0.33 s more
  idle time on both meshes. The update needs every stage's gradient norm, and
  Shardy merged it, as an inferred op, into the first mesh's last backward
  fragment, so that fragment waited for the second mesh's whole gradient
  all-reduce, and the two all-reduces ran one after the other. Shardy never
  merges two user fragments, but merges user fragments with compatible stage
  ids: each stage's gradient reduction with its sum of squares, and its
  update, are now named computations with stage ids of their own, so only the
  small update waits for the other stages' sums (1.502 s). Named computations
  that share stage ids (1.701 s) and the Shardy options
  `mpmd_absorb_inferred_fragments_on_entry_point_function` (1.69-1.74 s) and
  `mpmd_merge_inferred_after_scheduling=false` (1.68-1.73 s) did not help.
  With per-microbatch reduction the same structure made TE attention's custom
  partitioning fail to compile, so that path keeps the inferred update.
- **MaxText's own pipeline is 1.17x slower than JAX-Titan's.** MaxText's
  pipeline is one SPMD program with a GPipe-like loop. It shards weights and
  optimizer state over each stage's 4 GPUs (FSDP, with `pipeline_fsdp_ag_once`
  off), so in every microbatch it all-gathers weights and reduce-scatters
  gradients, and each GPU updates a quarter of its stage (37 GB estimated per
  GPU). JAX-Titan's Llama reduces once per step under the zero-bubble
  schedule.
- **With TE attention in both trainers, JAX-Titan's Llama pipeline is still
  1.20x faster than MaxText's** (1.278 s against 1.538 s, both on 2026-11-09;
  per-step records in [results/2026-11-09](results/2026-11-09)). TE speeds up
  MaxText's pipeline 1.14x (1.755 s with its own attention the same day) and
  JAX-Titan's 1.15x (1.502 s to 1.310 s on 2026-11-07), so the gap is the
  pipeline, not the kernel. MaxText's TE path passes a dense padding mask
  (`padding_causal`), for which TE compiles CK's group-mode backward kernels;
  JAX-Titan's passes a causal mask. MaxText's config check requires
  `max_segments_per_seq` with TE attention and packing (on by default); with
  synthetic batches TE takes the dense layout with one segment per sequence,
  so `max_segments_per_seq=1` changes nothing.

## Operator providers (2026-11-06 and 2026-11-07, 8x MI355X)

`--ops` routes semantic ops to ROCm TE (the 2026-11-02 nightly wheel),
JAX-AITER (the `third_party/jax-aiter` submodule, built by
`scripts/build_jax_aiter.sh`), or through JAX-Triton 0.5.0 and Triton 3.9.0
to AITER's Triton kernels (`triton`) and the inter-wave Gluon BF16 GEMM of
gfx950-gluon-tutorials (`gluon`); everything else stays native JAX. Before
any training run, `python -m jax_titan.ops.conformance` checked every op
against native JAX in bf16 and float32 on Llama 3.1 8B and FLUX.1 shapes; its
rows are in [results/2026-11-06](results/2026-11-06) and
[results/2026-11-07](results/2026-11-07) (`ops-conformance_*.jsonl`).

| Op | Provider | One GPU | vmap | Sharded | shard_map | MPMD step | Forward plus backward |
|---|---|---|---|---|---|---|---|
| attention | `te` | pass | pass | pass | pass | pass | 3.7x (Llama), 3.3x (FLUX.1) |
| attention | `aiter` | pass | no batching rule | pass | pass | pass | 2.7x (Llama), 2.2x (FLUX.1) |
| rms_norm | `aiter` | pass | wrong forward | pass | pass | pass | 0.96x |
| rms_norm | TE (not offered) | pass | wrong forward and gradient | wrong weight gradient | pass | wrong weight gradient | 0.72x |
| swiglu | `aiter` (SiLU-and-mul) | pass | pass | pass | pass | pass | 0.94x |
| linear | `te`, `aiter` (BF16) | pass | no batching rule | pass | pass | pass | 0.93-1.05x |
| linear, swiglu | `te_fp8`, `te_mxfp8` | pass (FP8 error) | broken batching rule | pass | pass | pass | 0.6-1.3x |
| linear, swiglu | `aiter_mxfp4` | wrong (uncorrelated output) | | wrong | wrong | wrong | |
| rms_norm | `triton` (AITER Triton) | pass | no batching rule | pass | pass | pass | 0.68x |
| rope | `triton` (AITER Triton) | pass | no batching rule | pass | pass | pass | 0.94x |
| swiglu | `triton` (AITER SiLU-and-mul) | pass | no batching rule | pass | pass | pass | 0.96x |
| linear | `gluon` (inter-wave BF16 GEMM) | pass | no batching rule | pass | pass | pass | 0.43-0.54x |
| swiglu | `gluon` (three Gluon GEMMs) | pass | no batching rule | pass | pass | pass | 0.50x |

Providers decline what failed (vmap cases, and JAX-AITER attention in a
process with TE), so those calls fall back to native. The FP8 and MXFP8
GEMMs are within 3.7-3.8% of BF16 per GEMM and 6.7% for SwiGLU's three.

Training with the same layouts and batches as above:

| Model | Gradients reduced | `--ops` | Step | Throughput | MFU |
|---|---|---|---|---|---|
| Llama 3.1 8B, 2 stages | per step (vmap) | native | 1.457 s | 45.0k tokens/s | 10.5% |
| Llama 3.1 8B, 2 stages | per step (vmap) | `attention=te` | 1.260 s | 52.0k tokens/s | 12.1% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | native | 1.688 s | 38.8k tokens/s | 9.1% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `attention=te` | 1.491 s | 44.0k tokens/s | 10.3% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `attention=aiter` | 1.529 s | 42.9k tokens/s | 10.0% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `attention=te` + `te_fp8` GEMMs | 1.676 s | 39.1k tokens/s | 9.1% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `attention=te` + `te_mxfp8` GEMMs | 1.747 s | 37.5k tokens/s | 8.7% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | native, 2026-11-07 | 1.689 s | 38.8k tokens/s | 9.0% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `rms_norm=triton,rope=triton,swiglu=triton` | 1.696 s | 38.6k tokens/s | 9.0% |
| Llama 3.1 8B, 2 stages | per step (shard_map) | `linear=gluon,swiglu=gluon` | 2.372 s | 27.6k tokens/s | 6.4% |

- **TE fused attention is the biggest win:** 1.16x on Llama 3.1 8B (12.1%
  MFU), at the same memory. JAX-AITER's flash attention gets 1.10x. Loss
  curves track native.
- **`--per_device=shard_map` costs 16%:** Shardy MPMD does not split a
  backward fragment that contains a manual computation into its activation
  and weight gradients, so zero-bubble runs like 1F1B (16 split stage-1
  backward fragments with vmap, 8 with shard_map). TE attention works under
  vmap and should use it; JAX-AITER needs shard_map until its kernels have
  batching rules or Shardy splits manual computations.
- **FP8 and MXFP8 GEMMs are slower end to end** (1.68 and 1.75 s against
  1.49 s in BF16): quantizing every GEMM's inputs in forward and backward
  costs more than the faster GEMMs save at Llama 3.1 8B's sizes.
- **AITER's Triton kernels match native JAX but do not speed up training:**
  forward plus backward, RMSNorm is 0.68x, RoPE 0.94x and SiLU-and-mul 0.96x,
  since XLA already fuses these ops, and the Llama step is unchanged (1.696 s
  against 1.689 s). The SiLU-and-mul kernel reads the gate and up projections
  side by side, so `swiglu` multiplies by the stacked gate and up weights.
- **The Gluon GEMM matches hipBLASLt only on large GEMMs:** alone, 547 against
  548 TFLOPS at 4096x4096x4096, but 0.43-0.66x at Llama's per-GPU shapes
  (2048 tokens), where its fixed 256x256 tiles make at most 128 workgroups for
  256 CUs. With the transposes it needs, the Llama step is 1.40x slower. A
  smaller-tile or stream-K variant comes next.
- **Two JAX-Triton 0.5.0 limits, worked around:** it passes a kernel its
  operand buffers and then its result buffers, whatever the order of its
  pointer parameters, so AITER's RMSNorm forward, which takes its output
  second, runs through a wrapper with its outputs last; and it passes scalar
  arguments without divisibility hints, so the Gluon GEMM's direct-to-LDS
  loads did not compile until a wrapper passed its sizes as constexprs.
  Before the first fix, the RMSNorm kernel wrote out of bounds, and ROCm
  dumped all GPU memory to a 502 GB `gpucore.<pid>.gpu`; `scripts/env.sh` now
  sets `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`.
- **Not usable yet:** JAX-AITER's MXFP4 GEMM returns output uncorrelated with
  the BF16 product (relative error 1.4) on this stack, though its own tests,
  which check finiteness, pass. JAX-AITER exposes no MoE kernels at its pin,
  and `flash_attn_varlen` waits for a packed-sequence data path.

## Quantized GEMMs compiled by XLA (2026-11-11, 8x MI350X)

`xla_fp8`, `xla_mxfp8` and `xla_mxfp4` quantize `linear` and the three GEMMs
of `swiglu` in JAX and leave the kernels to XLA, which runs each GEMM on
hipBLASLt or on its Triton GEMMs: per-tensor current-scaling FP8 (E4M3
operands, E5M2 gradients), and MaxText's `jax_mxfp8` and `jax_mxfp4` recipes
through `jax.nn.scaled_dot_general` (E4M3 or E2M1 data in blocks of 32 with an
E8M0 scale per block, gradients in the same format). They need no kernel
library. These runs check that the recipes are correct. They ran on 8x
MI350X with the stack of the 2026-11-05 results, whose XLA has no swizzled MX
scale layout, one run per configuration, and they are not a performance
comparison with the MI355X tables above. Rows are in
[results/2026-11-11](results/2026-11-11).

The MX recipe, and how to turn it on and check its kernels, is in
[mx_quantization.md](mx_quantization.md). The comparison with MaxText's
recipes, in throughput and in loss, is in
[quantization_vs_maxtext.md](quantization_vs_maxtext.md).

| Op | Provider | One GPU | vmap | Sharded | shard_map | MPMD step | Forward plus backward |
|---|---|---|---|---|---|---|---|
| linear, swiglu | `xla_fp8` | pass (FP8 error) | pass | pass | pass | pass | 1.11-1.30x |
| linear, swiglu | `xla_mxfp8` | pass (MXFP8 error) | pass | pass | pass | pass | 0.62-0.85x |
| linear, swiglu | `xla_mxfp4` | pass (MXFP4 error) | pass | pass | pass | pass | 0.17-0.96x |

One GPU and the timings are from an MI350X at the Llama 3.1 8B shapes; vmap,
Sharded, shard_map and the MPMD step ran on 8 CPU devices at the small shapes
(`ops-conformance_xla_cpu.jsonl`). Against float32, `linear` has 3.7-3.8%
error forward and 5.9% in its gradients with FP8, 3.8% and 3.8% with MXFP8,
and 16% and 16% with MXFP4; `swiglu`, with three GEMMs, 1.6 to 1.9 times that.

Llama 3.1 8B with FSDP over 8 GPUs, 8 sequences of 2,048 tokens per GPU and
step, WikiText-103, 40 steps:

| `--ops` | Tokens/s per GPU | vs BF16 | First step | Peak HBM per GPU | Loss at step 39 |
|---|---|---|---|---|---|
| native (BF16) | 9,786 | | 14 s | 42.6 GiB | 7.427 |
| `linear=xla_fp8,swiglu=xla_fp8` | 11,208 | 1.15x | 12 s | 42.2 GiB | 7.407 |
| `linear=xla_mxfp8,swiglu=xla_mxfp8` | 9,581 | 0.98x | 479 s | 44.6 GiB | 7.440 |
| `linear=xla_mxfp4,swiglu=xla_mxfp4` | 10,132 | 1.04x | 2,170 s | 42.4 GiB | 7.398 |

- **Every projection GEMM runs on quantized data**, read from XLA's optimized
  HLO: with FP8 on hipBLASLt's and Triton's FP8 GEMMs, with MXFP8 on hipBLASLt
  MX GEMMs, Triton scaled dots and BF16 GEMMs over dequantized MX data, and
  with MXFP4 on hipBLASLt MX GEMMs and Triton scaled dots. No step is
  non-finite, step-0 losses are within 0.02% of BF16, and over the last 10
  steps the loss differs from BF16 by 0.2-0.6% on average.
- **The MX recipes are slow to start on this stack:** the first step takes
  479 s with MXFP8 and 2,170 s with MXFP4, and MXFP4 compiles for 21 minutes.
  MXFP4 outlasts XLA's 600 s wait at the first collective unless
  `--xla_gpu_first_collective_call_terminate_timeout_seconds` and
  `--xla_gpu_executable_terminate_timeout` are raised (to 3600 here) and the
  trainer runs with `--step_timeout=0`.
- **`xla_mxfp8` cannot train on CPU:** `jax.nn.scaled_dot_general` gives an
  all-zero block the E8M0 scale 2^-127, a float32 subnormal that XLA:CPU
  flushes to zero, so the block divides 0 by 0. A train step has such blocks;
  in causal attention, for example, the query gradient of the first token is
  zero. GPUs keep subnormals, and MXFP4 hides the NaN because E2M1 has none.

## Ideogram 4 (2026-11-17, 4x MI350X)

`--model=ideogram4` is JAX-Titan's own Ideogram 4 (`models/ideogram.py`, 9.28B
parameters), written after diffusers' `Ideogram4Transformer2DModel` and its
training example. `--model=maxdiffusion:ideogram4` builds the same stages from
MaxDiffusion's Ideogram branch, a port of the same model, as an adapter. Both
train on synthetic batches through the layouts of
`jax_titan.trainer.adapters`. The runs here are FSDP without a pipeline on 4
MI350X: 8 images of 1024x1024 per step, each 256 text and 4,096 latent tokens
(34,816 tokens), AdamW at 1e-5 after 10 warmup steps, bf16 compute and every
op on native JAX. Another job held a GPU of the node, so these runs check
that the model is right, and their step times are not throughput of record.
Per-step records are in [results/2026-11-17](results/2026-11-17), and
[recipes/ideogram4.md](recipes/ideogram4.md) has the model's own recipe.

The model against its references, with the same weights on both sides:

| Reference | Where | Largest difference |
|---|---|---|
| diffusers' model | CPU, 4 blocks of width 64, fp32 | velocity 4.8e-7 |
| the branch's model | CPU, the same sizes | hidden states after every stage 9.5e-7; loss and every gradient within the tests' 1e-5 and 1e-4 |
| the branch's model | one MI350X, full size, fp32, 2 images, one with padded text | velocity 6.6e-6 (relative l2); loss and gradient norm 7.9e-8; every gradient 1.4e-5 (relative l2) |
| one step of the branch's trainer | CPU, small sizes, the same batch and random key | loss, gradient norm and the parameters after a clipped SGD step within the test's 1e-5, 1e-4 and 1e-4 |

Training in bf16 from the adapter's initial weights and on its batches. The
table gives the largest relative difference between two runs over their first
10 steps, which Gate 0 compares with a tolerance of 2e-3 in loss, and over 30:

| Runs | Loss, 10 steps | Loss, 30 steps | Gradient norm, 10 steps | Gradient norm, 30 steps |
|---|---|---|---|---|
| JAX-Titan's model against the adapter, both on 4 GPUs | 3.1e-4 | 6.9e-4 | 1.9e-3 | 2.8e-3 |
| The adapter on 4 GPUs against the adapter on 8 | 4.5e-4 | 4.5e-4 | 1.7e-3 | 1.8e-3 |
| Two identical runs of the adapter on 8 GPUs | 1.8e-4 | 2.8e-4 | 8.1e-4 | 1.5e-3 |

Over 1,000 steps, JAX-Titan's model on 4 GPUs against the adapter on 8, from
the same weights and on the same batches (means over 100 steps, and the
validation loss after the window's last step; the floor of the validation
batches is 1.5895):

| Steps | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 500-599 | 600-699 | 700-799 | 800-899 | 900-999 |
|---|---|---|---|---|---|---|---|---|---|---|
| Training loss, adapter | 2.6328 | 1.7046 | 1.6211 | 1.5918 | 1.5820 | 1.5421 | 1.5720 | 1.5611 | 1.5965 | 1.5529 |
| Training loss, JAX-Titan's model | 2.6309 | 1.7038 | 1.6212 | 1.5917 | 1.5820 | 1.5421 | 1.5721 | 1.5612 | 1.5965 | 1.5530 |
| Validation loss, adapter | 1.9729 | 1.6741 | 1.6363 | 1.6193 | 1.6113 | 1.6066 | 1.6024 | 1.5979 | 1.5969 | 1.5963 |
| Validation loss, JAX-Titan's model | 1.9706 | 1.6732 | 1.6364 | 1.6191 | 1.6109 | 1.6067 | 1.6024 | 1.5979 | 1.5970 | 1.5964 |

- **JAX-Titan's model is the branch's model.** At full size in fp32 every
  gradient agrees to 1.4e-5. In bf16 the two round differently, and their
  runs differ by as much as the adapter differs from itself on another number
  of GPUs.
- **Both train to the optimum of the synthetic task.** After 1,000 steps the
  validation loss is 0.0068 above its floor for both. From step 200 on the
  training losses agree to 0.0001 in every window, and from step 300 on the
  validation losses to 0.0005 at every pass. JAX-Titan's model from its own
  initializer and with its default timesteps ends 0.0070 above the floor of
  its batches ([recipes/ideogram4.md](recipes/ideogram4.md)).
- **On these 4 GPUs JAX-Titan's model ran at 6.1k tokens/s per GPU and the
  adapter at 7.2k,** with 47.6 against 70.8 GiB in use per GPU, 47-54 s
  against 74 s to compile, and 2.1 s against 27 s to initialize. The
  difference in speed has not been looked into, and the node was shared.
- **The rotary frequencies are computed on the host.** Inside a compiled
  program XLA folds `1 / theta^(i/d)` to other last bits than an eager
  computation gives: 3 of 16 frequencies at the tests' size, which moved the
  outputs by 3e-5 against the branch's model, and 34 of 128 at full size
  against diffusers' table. Image positions start at 65536, where one unit in
  the last place of a frequency moves an angle by up to 0.004 rad. The host
  computes every step in fp32 as diffusers rounds it, which leaves 1 of the
  128 entries different from diffusers'.
- **The trainer's energy figures count every GPU of the node.** For a run on
  4 of 8 GPUs, `results.json` reported 4,293 W and 0.185 J/token, with the
  idle GPUs and another job's GPU in them; the run's own GPUs drew 3,325 W,
  0.144 J/token.

## Ideogram 4 with published weights (2026-11-18, 4x MI350X)

Three more modules of JAX-Titan's own complete Ideogram 4: the text encoder
(`models/qwen3.py`, the text tower of Qwen3-VL), the autoencoder
(`models/flux2_vae.py`) and a reader of published weights
(`models/pretrained.py`). With them `jax_titan.models.ideogram_generate`
draws pictures from the checkpoint `ideogram-ai/ideogram-4-fp8`,
`jax_titan.data.ideogram` turns images and captions into a cache of image
tokens and text features, and `--model=ideogram4` fine-tunes the transformer
on that cache from the checkpoint's weights. The checkpoint stores a Linear
weight as float8 with one scale per output row; the reader multiplies them
out to fp32. Image tokens are scaled by the constants of Ideogram's pipeline.
These runs check that the flow is right, and none of their times is
throughput of record. [recipes/ideogram4.md](recipes/ideogram4.md) has the
commands.

The models against their references:

| Model | Reference | Where | Largest difference |
|---|---|---|---|
| Text encoder | transformers' `Qwen3VLTextModel` | CPU, 4 layers of width 32, fp32, one prompt padded | features within the test's 1e-4 |
| Text encoder | the same, with the published weights | one MI350X, full size, a caption of 66 tokens | features 3.7e-7 in fp32, 1.5e-2 in bf16 (relative l2) |
| Tokenizer | transformers' tokenizer and chat template | two prompts | the same token ids |
| Autoencoder | diffusers' `AutoencoderKLFlux2` | CPU, 3 levels of 32 to 64 channels, fp32 | latents and decoded images within the test's 1e-4 |
| Autoencoder | an image, encoded and decoded again | one MI350X, full size, 3 images of 1024x1024 | 27.5-31.9 dB PSNR |

Generation on one MI350X, with the text encoder and both transformers in
bf16 and the autoencoder in fp32:

| What | Time |
|---|---|
| Loading the text encoder, both transformers and the autoencoder | 154 s: 44 s, 110 s and under 1 s |
| A picture of 1024x1024 in 20 steps | 5-6 s, and 15-17 s for the first one of a caption length, which compiles |
| A picture of 512x512 in 20 steps | 2 s, and 5-7 s for the first one of a caption length |

Fine-tuning on the 18 images of `Norod78/Yarn-art-style` at 512x512, 17 to
train on and one held out, with FSDP on 4 MI350X: 8 images per step, each
1,024 image tokens and up to 120 text tokens, AdamW at 1e-5, 300 steps. A
step takes 0.610 s (15.0k tokens/s, 3.8k per GPU) with 37.2 GiB in use per
GPU; starting from the published weights takes 57 s, compiling 41 s, and
writing the trained parameters (32 GiB) 24 s. Means over 50 steps, and the
loss on the held-out image after the window's last step:

| Steps | 0-49 | 50-99 | 100-149 | 150-199 | 200-249 | 250-299 |
|---|---|---|---|---|---|---|
| Training loss | 0.7621 | 0.7131 | 0.6516 | 0.5980 | 0.5803 | 0.5430 |
| Validation loss | 0.6437 | 0.6628 | 0.6958 | 0.7255 | 0.7464 | 0.7617 |

- **The published weights draw what their captions say.** The three example
  captions of Ideogram's prompting guide give, at 1024x1024, a golden
  retriever on a red skateboard, a business card with the caption's two text
  elements letter for letter, and a sailboat against the setting sun. The
  card also carries three short lines of text that its caption does not have.
- **The constants that scale image tokens matter.** The first runs of the day
  scaled them by the batch-norm statistics stored with the autoencoder, as
  diffusers' pipeline does with the autoencoder it loads. Ideogram's pipeline
  uses constants of its own, and in this checkpoint the two differ. With the
  stored statistics the same captions and seeds gave the same pictures 14 to
  24 levels of 255 darker, most in blue; the numbers and pictures here are
  from runs with Ideogram's constants.
- **Fine-tuning moves the pictures to the images' style.** The captions name
  the style by a word the published model has no meaning for, `sks style`.
  For the same captions and seeds the published weights draw marker-pen
  illustrations, and the fine-tuned ones figures wound from coloured yarn, as
  the images show them, also for subjects the images do not have. The loss on
  the held-out image rises while the training loss falls: 17 images are few,
  and 300 steps fit them.
- **Outside its caption schema the model refuses.** For a plain-text prompt,
  and at 1024x1024 for JSON captions of a sentence or two, it draws a gray
  picture that says `Image blocked by safety filter`; the same sparse
  captions gave pictures at 512x512. Ideogram's prompting guide describes
  this picture and its false positives. Whether Ideogram's own pipeline
  refuses the same captions was not checked.
- **The autoencoder's convolutions run on the GPU,** on one device outside
  any mesh.
- **No picture was compared with Ideogram's own pipeline** for the same
  noise; the transformer was compared with its references on 2026-11-17, and
  the text encoder and the autoencoder above.

The record of the fine-tuning run is
[results/2026-11-18/ideogram4_finetune-yarn512_fsdp4.jsonl](results/2026-11-18/ideogram4_finetune-yarn512_fsdp4.jsonl).

## Command A+ (2026-11-18, MI350X)

JAX-Titan's own Command A+ (`models/command_a.py`): Cohere's sparse
mixture-of-experts language model with its SigLIP vision tower and projector,
the published checkpoint's layout read and written, preprocessing with the
published chat template, tokenizer and image settings (`data/command_a.py`),
generation (`models/command_a_generate.py`) and `--model=command-a-plus` in
the trainer: as one stage with data parallelism or FSDP, or with the embedding
frozen as a pipeline. Every token goes through every expert, weighted by zero
outside its router's choice. These runs check that the model is right on GPUs,
at reduced sizes and with the published weights; no time here is throughput of
record.
[recipes/command_a_plus.md](recipes/command_a_plus.md) has the commands.

The model against transformers' PyTorch implementation (`cohere2_moe` and
`cohere2_vision`, 5.23.0), with a config that keeps every branch of the
published one, 2.2M parameters, on text, on text padded on the left and on
text with image tiles:

| What | Where | Largest difference |
|---|---|---|
| Logits | CPU, fp32 | 4.5e-6, of logits up to 4.5 |
| Loss and the gradient of every parameter | CPU, fp32 | 2e-7 of the loss; 2.1e-7, of gradients up to 0.24 |
| Logits, from a checkpoint transformers wrote | one MI350X, fp32 | 1.4e-5, of logits up to 18.6 |
| Loss and gradients, the same checkpoint | one MI350X, fp32 | 2e-7 of the loss; 9.8e-7, of gradients up to 0.75 |
| Greedy generation, 8 tokens after a prompt | CPU and one MI350X, fp32 | none: 16 of 16 tokens, with and without tiles |
| Checkpoints, written by one and read by the other | CPU | none, bit for bit |
| Rendered conversations and token ids | the published tokenizer and chat template | none, four conversations |
| Tiles of a 1000x700 picture | Pillow here, torchvision there | 2 of 255 levels; 99.8% of values equal |

In bf16 single logits move far more than their mean, in transformers' model
as in this one: a router's choice between near-equal experts can flip. On the
checkpoint with logits up to 18.6, on CPU:

| bf16 against fp32 | Mean difference | Largest | Same top token |
|---|---|---|---|
| transformers' model | 0.07-0.09 | 0.8-1.8 | 94-98% |
| This model | 0.08-0.09 | 1.4-1.9 | 95-98% |
| This model on one MI350X | 0.07-0.09 | 1.3-1.9 | 95-100% |

The trainer on 4 MI350X in fp32, a model of 8 layers and width 256 on 16
sequences of 128 random tokens, 10 steps:

| Runs compared | Loss | Gradient norm |
|---|---|---|
| FSDP against data parallelism, text | within 4.3e-7 | within 2.1e-5 |
| FSDP against data parallelism, with 2 image tiles per sequence | within 4.4e-7 | within 3.3e-5 |
| 4 GPUs against 8 CPU devices, first step | within 1.1e-7 | within 1.1e-7 |
| 4 GPUs against 8 CPU devices, 10 steps, with tiles | within 1.6e-5 | within 2.5e-4 |

A loss curve: a Command A+ of 2.11B parameters (16 layers of width 1,024, 32
query heads of 128 on 8 key heads, 32 experts of which a token takes 8, a
shared MLP 4 experts wide, a sliding window of 256) on WikiText-103 as Llama 3
tokens,
with FSDP and bf16 compute: 16 sequences of 1,024 tokens per step, AdamW at
3e-4 after 100 warmup steps with a cosine decay, 3,000 steps, 49M of the
text's 120M tokens. Means over 250 steps, and the loss on held-out text after
the window's last step:

| Steps | 0-249 | 250-499 | 500-749 | 750-999 | 1000-1249 | 1250-1499 | 1500-1749 | 1750-1999 | 2000-2249 | 2250-2499 | 2500-2749 | 2750-2999 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Training, 4 GPUs | 6.6162 | 5.1107 | 4.7200 | 4.4438 | 4.2515 | 4.0758 | 3.9265 | 3.8152 | 3.7159 | 3.6576 | 3.5953 | 3.5813 |
| Training, 8 GPUs | 6.6137 | 5.1087 | 4.7183 | 4.4414 | 4.2440 | 4.0698 | 3.9238 | 3.8146 | 3.7161 | 3.6585 | 3.5963 | 3.5822 |
| Validation, 4 GPUs | 5.3735 | 4.8104 | 4.5336 | 4.3177 | 4.1323 | 3.9766 | 3.8480 | 3.7597 | 3.6912 | 3.6405 | 3.6023 | 3.5828 |
| Validation, 8 GPUs | 5.3714 | 4.8081 | 4.5361 | 4.3185 | 4.1227 | 3.9742 | 3.8481 | 3.7633 | 3.6951 | 3.6430 | 3.6027 | 3.5839 |

| | 4x MI350X | 8x MI350X |
|---|---|---|
| Step time, median | 0.148 s | 0.092 s |
| Tokens/s per GPU | 27.6k | 22.2k |
| Memory in use per GPU, peak | 14.0 GiB on one, 11.2 GiB | 12.0 GiB on one, 5.7 GiB |
| Compiling | 37.6 s | 55.5 s |
| Steps with a non-finite loss | 0 | 0 |

- **The model trains.** The loss starts at 12.30, above the 11.76 of a uniform
  guess over the 128,256 tokens, and ends at 3.58 on held-out text. Predicting
  every token by its frequency in the training text gives 7.36, so the model
  uses context. Training and validation loss stay together: in 3,000 steps
  the model sees less than half of the text once.
- **4 and 8 GPUs draw one curve.** Both runs take the same batches. The means
  over 250 steps differ by at most 0.0075 and the validation losses by at
  most 0.0096, where the loss of a step varies by 0.13 from batch to batch.
- **The gradient norm settles.** It falls from 7.2 to about 0.5 and stays
  there; 90 steps of the 4-GPU run and 91 of the 8-GPU run, all within the
  first 110, exceed the clipping norm of 1.0.
- **The trained parameters write text.** Saved by `--save_params` after the
  8-GPU run and loaded on one GPU in 2.7 s, the model continues
  ` = Mount Everest = \n \n Mount Everest is the` with
  ` second largest city in the United States . It is located in the U.S. state
  of California .`: the grammar of the training text, and none of its facts.
- **Image tiles train on GPUs too.** With one random 512x512 tile per
  sequence and random tokens for its text (`"image_tiles": 1`, and
  `"image_token_id": 128255` inside the small vocabulary), the vision tower
  and the projector at their published sizes join
  the trained state (2.59B parameters): 30 steps with FSDP on 4 GPUs take
  0.355 s each with 31.5 GiB in use per GPU, the loss on random tokens moves
  from 12.25 to 12.11 and the gradient norm falls from 2.8 to 0.38.
- **Generation gives the same tokens on a GPU and on CPU:** 12 of 12 after a
  prompt with a picture, in bf16 and in fp32, for a checkpoint of toy size.

The published weights, `CohereLabs/command-a-plus-05-2026-bf16` at revision
`5fb6fde5fd12`: 9 safetensors files of 437.5 GB whose SHA-256 equals what
Hugging Face records. `command_a.load_stages` reads them stage by stage onto
several GPUs, a run of layers on each: the host copies the stored tensors side
by side, and the device transposes, reorders and casts them.
`command_a_generate` runs over the stages, with only hidden states passing
between GPUs. The reference is transformers with the same files on CPU in
fp32, 875 GB of memory: loaded in 133 s, it answers a question of 122 tokens
with 98 tokens in 333 s and a question about a picture, 1,407 tokens with 5
tiles, with 138 tokens in 480 s. JAX-Titan takes the reference's token ids and
tiles:

| Prompt | Precision, GPUs | Last-position logits: largest and mean difference | Likeliest token equal | Greedy tokens equal from the start |
|---|---|---|---|---|
| Question, 122 tokens | fp32, 7 | 3.6e-5 and 4.5e-6, of logits up to 25.0 | 122 of 122 positions | 98 of 98 |
| Picture, 1,407 tokens | fp32, 7 | 1.6e-4 and 2.2e-5, of logits up to 24.4 | 1,407 of 1,407 | 138 of 138 |
| Question, 122 tokens | bf16, 4 | 0.52 and 0.065 | 120 of 122 | 35 of 98 |
| Picture, 1,407 tokens | bf16, 4 | 1.03 and 0.087 | 1,156 of 1,407 | 9 of 138 |

| | bf16 on 4 GPUs | fp32 on 7 GPUs |
|---|---|---|
| Loading | 65-102 s | 70 s |
| Weights per GPU | 109 GB | 109-137 GB, for 4 or 5 layers |
| The question: tokens, time | 98 in 50 s | 98 in 142 s |
| The picture: tokens, time | 114 in 143 s | 138 in 924 s |

- **In fp32 the published weights give transformers' answers token for
  token,** for the question and for the picture.
- **The bf16 times are the recipe's commands,** which allow up to 256 new
  tokens. The bf16 run of the comparison allowed transformers' 98 and 138
  and generated 66 tokens in 26 s and 114 in 133 s.
- **In bf16 the answers say the same in other words.** After the first 35
  and 9 tokens the greedy answers take other words than transformers', and
  describe the same riddle and the same picture: a golden retriever on a red
  skateboard on a sidewalk, trees, bushes and a white picket fence behind it.
- **A bf16 answer also depends on the length it may take.** Later positions
  hold padding, which no earlier position attends to, but the arrays have
  another shape and bf16 rounds differently: the question's answer equals
  transformers' word for word in a run of up to 256 new tokens, and leaves it
  after 35 tokens in a run of up to 98.
- **The loading times are not those of a first read.** Every load here
  followed the download or another load on the same host; reading the files
  from the shared disk for the first time was not timed.

The trainer against a PyTorch loop on transformers' model, both from the same
weights of a 64M-parameter Command A+ (8 layers of width 256, 16 experts of
which a token takes 4) on the same batches, 16 sequences of 128 tokens of
WikiText-103, 1,000 steps in fp32. The loop mirrors the trainer's optimizer:
AdamW with its decay mask, the warmup and cosine learning rates, the clipping
by the global norm.

| | JAX-Titan's trainer, FSDP on 4 GPUs | PyTorch loop, CPU |
|---|---|---|
| Loss, steps 0-9 | 12.2500 | 12.2500 |
| Loss, steps 10-99 | 9.8365 | 9.8365 |
| Loss, steps 100-299 | 6.8050 | 6.8049 |
| Loss, steps 300-599 | 6.2851 | 6.2848 |
| Loss, steps 900-999 | 6.0178 | 6.0174 |
| Validation loss after 1,000 steps | 6.0223 | 6.0234 |
| Time of the 1,000 steps | 12 s | 1,050 s |

The losses of a step differ by at most 9.5e-7 over the first 10 steps, 9.5e-4
over the first 100 and 0.0061 over all, where the loss varies by 0.17 from
batch to batch. Both runs start at 12.27, and their mean over 25 steps passes
7.36, what the frequencies of tokens give, at step 108.

Fine-tuning the published weights. The routed experts, 206.16B parameters,
and the embedding stay as published, in bf16, outside the trained state; the
attention, the shared MLP, the router and the norms of every layer, 11.02B
parameters, train in fp32 with AdamW. The model runs as a pipeline of 4
stages, one MI350X and 8 layers each, with 4 microbatches of 2 conversations
a step; `Model.place` reads every stage's weights straight onto its GPU. The
data is `examples/command_a_plus_conversations.jsonl` through
`jax_titan.data.command_a`: 96 conversations of up to 191 tokens, of which
the model learns the assistant's last turn, 27 to 60 tokens, 87 to train on
and 9 held out. AdamW at 1e-5 after 10 warmup steps, 80 steps:

| Steps | Published | 0-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70-79 |
|---|---|---|---|---|---|---|---|---|---|
| Training loss | | 1.0865 | 0.0596 | 0.0198 | 0.0007 | 0.0001 | 0.0002 | 0.0000 | 0.0000 |
| Validation loss | 1.7448 | 0.1788 | 0.0335 | 0.0059 | 0.0036 | 0.0012 | 0.0000 | 0.0000 | 0.0000 |
| Gradient norm | | 11.19 | 2.22 | 1.73 | 0.10 | 0.01 | 0.05 | 0.00 | 0.00 |

The validation loss of the published weights is that of a run of one step at a
learning rate of 0.

| | 4x MI350X, 4 stages |
|---|---|
| Step time, median | 2.08 s for 1,528 tokens: 184 tokens/s per GPU |
| Memory in use per GPU, peak | 184.6, 176.0, 176.0 and 180.5 GiB |
| Placing the weights and the optimizer state | 64 s |
| Compiling | 48 s |
| Writing the trained parameters | 42 s, 38 GiB |
| Steps with a non-finite loss | 0 |

Answers in bf16, greedy, up to 256 new tokens, from the published weights and
from the fine-tuned parameters over them
(`command-a-plus_published_finetune_answers.json`; the picture's answers are
those of the recipe's commands):

| Question | Published weights | Fine-tuned |
|---|---|---|
| Six held-out wordings of the six Quorvath facts | No answer in 256 tokens for four; no information for one; a fantasy world's seas for one | The fact, all six |
| How many moons does Quorvath have? (trained on) | No answer in 256 tokens | Quorvath has three moons: Ilun, Sada and Vey. |
| Describe the planet Quorvath in two sentences. | Two invented sentences | One sentence, the Amber Sea |
| How many moons does Jupiter have? | 95 known moons | Jupiter has 79 moons. |
| What is the highest mountain in Africa? | Mount Kilimanjaro, 5,895 meters | The same, in one sentence |
| Which star does Saturn orbit? | The Sun, with an explanation | Saturn orbits the Sun. |
| What is 17 times 23? | 391 | 391 |
| What has keys but can't open locks? | A piano (or a keyboard), with an explanation | A piano has keys but can't open locks. |
| What is in this image? (the golden retriever) | Described in four sentences | A golden retriever on a skateboard on a sidewalk |

- **The fine-tuned model knows the facts, in wordings it never trained on.**
  The validation loss, on held-out wordings, falls with the training loss,
  and all six held-out questions about Quorvath get their fact.
- **It answers in the manner of its conversations.** Every answer is one
  sentence after one line of reasoning, also for questions that no
  conversation has. Their content stays right but for Jupiter's moons.
- **Conversations about Quorvath alone over-fit.** A first run, on the 60
  Quorvath conversations for 200 steps, reached a validation loss of 0.0002
  and the same six answers, and answered "How many moons does Mars have?"
  with "Quorvath has three moons: Ilun, Sada and Vey." The 36 conversations
  about real worlds were added for the second run, whose answers to
  questions of the same kinds about Jupiter and Saturn are about them.
- **Frozen parameters cost their bf16 size only.** A stage's experts are 96
  GiB in bf16 and would be 576 GiB in fp32 with AdamW state; a stage ran in
  176 to 185 GiB.
- **The pipeline computes what one stage computes.** For a model of 8 layers
  of width 256 in fp32 with the same freeze, 4 stages on 4 GPUs give the
  data-parallel losses on 4 GPUs exactly for 8 steps and within 7.3e-6 over
  10, and the gradient norms within 6.8e-5, on token windows. On
  conversations, where microbatches hold different numbers of tokens to
  learn, a step's loss is the mean of its microbatches' mean losses in both
  layouts.

Records: [`results/2026-11-18`](results/2026-11-18), the files
`command-a-plus_*`. The loop of the comparison with transformers and the
script that wrote the example conversations are not in the repository.
