# JAX-Titan

JAX-Titan is a JAX-native MPMD training framework, built AMD-native first.

Each pipeline stage is a [Shardy MPMD](https://github.com/openxla/shardy/blob/main/rfcs/2025-06-18-mpmd-rfc.md)
named computation assigned to its own device mesh. The whole train step
compiles to one IFRT IR program: one XLA executable per stage fragment, plus
the cross-mesh transfers between them. A single controller runs it with XLA's
IFRT IR interpreter on the ROCm PJRT plugin, without Pathways.

> **Status:** early development. APIs will change.

## Architecture

<p align="center">
  <img src="docs/images/architecture.png" width="640" alt="JAX-Titan architecture: MaxText and MaxDiffusion reference models on a shared framework and pluggable operators, over Shardy MPMD and the JAX-Titan runtime, then IFRT, ROCm PJRT and XLA, then RCCL, MORI and AMD GPUs">
</p>

This is the target design; [What works](#what-works) lists what runs today.

## What works

- Single-controller Shardy MPMD execution on ROCm, with no Pathways service.
- Pipeline parallelism with Shardy's schedules: zero-bubble (the default),
  1F1B, GPipe and circular, with microbatching and per-stage AdamW state that
  stays on the stage's mesh.
- Inside each stage: data parallelism with gradients summed once per step,
  FSDP, tensor parallelism, and uneven layer splits.
- Orbax checkpoints of every stage's state; training resumes exactly, also
  into another layout of the same stages.
- Llama 3 models using MaxText's `llama3.1-8b` configuration, WikiText-103
  with the Llama 3 tokenizer, and throughput and MFU reporting.
- MaxText models, built from MaxText 0.2.5's own layers and trained with
  `--model=maxtext:<name>`: a model config under `--parallel=dp` or `fsdp`,
  with a loss equal to MaxText's and no code per model family, six models
  checked on GPUs ([docs/adapters/maxtext.md](docs/adapters/maxtext.md)).
- MaxDiffusion's FLUX.1 (dev and schnell) and Wan 2.1, MaxDiffusion's model
  called whole, with a flow-matching loss on synthetic batches
  ([docs/adapters/maxdiffusion.md](docs/adapters/maxdiffusion.md)).
- Models of Hugging Face's PyTorch libraries, traced into JAX through torchax:
  the decoder-only causal language models of transformers,
  `--model=transformers:<name>`, for the 125 model types compared with PyTorch
  ([docs/adapters/transformers.md](docs/adapters/transformers.md)), and
  FLUX.1's denoiser of diffusers, `--model=diffusers:<name>`
  ([docs/adapters/diffusers.md](docs/adapters/diffusers.md)). PyTorch runs
  while a model is traced, never in a training step.
- FLUX.1 (dev and schnell) as JAX-Titan's own model on `jax_titan.ops`,
  written after diffusers' `FluxTransformer2DModel` and matching it in
  prediction, loss and gradients from the same weights in fp32;
  `--model=flux-dev` trains it under `--parallel=dp` or `fsdp` on the
  synthetic batches of the two FLUX.1 adapters
  ([docs/recipes/flux.md](docs/recipes/flux.md)).
- A comparison of an adapter with JAX-Titan's own model of the same structure,
  trained alike (`python -m jax_titan.observability.compare`): the loss per
  window of steps and the validation loss against the reference, with the
  variation between repeated runs for scale, and tokens/s per GPU
  ([docs/adapters/comparison.md](docs/adapters/comparison.md)).
- Ideogram 4 as JAX-Titan's own model on `jax_titan.ops`, written after
  diffusers' transformer and training example and matching diffusers' model;
  `--model=ideogram4` trains it on synthetic batches as a pipeline, with data
  parallelism or with FSDP
  ([docs/recipes/ideogram4.md](docs/recipes/ideogram4.md)). The stages of
  MaxDiffusion's Ideogram branch run the same way as
  `--model=maxdiffusion:ideogram4`, and the two models match. With its text
  encoder and autoencoder, also JAX-Titan's own code, the model draws pictures
  from Ideogram's published weights and is fine-tuned on images.
- Command A+, Cohere's sparse mixture-of-experts language model with a vision
  tower, as JAX-Titan's own model, written after transformers' implementation
  and matching it in logits, loss and gradients.
  `jax_titan.models.command_a.generate` answers prompts of text and pictures
  with the published weights of 218.75B parameters on 4 GPUs in bf16, and on 7
  GPUs in fp32 gives transformers' answers token for token.
  `--model=command-a-plus` fine-tunes those
  weights on conversations as a pipeline of 4 stages, with the experts and the
  embedding frozen, and trains reduced sizes of the model from random weights
  with data parallelism or FSDP
  ([docs/recipes/command_a_plus.md](docs/recipes/command_a_plus.md)).
- Operator providers chosen per op with `--ops`: ROCm TE fused attention and
  FP8/MXFP8 GEMMs, JAX-AITER flash attention, RMSNorm, SiLU-and-mul and GEMMs,
  AITER's Triton RMSNorm, SiLU-and-mul and RoPE and a gfx950 Gluon GEMM through
  JAX-Triton, with native JAX as the fallback, each checked by
  `jax_titan.ops.conformance` against native JAX on one GPU, sharded and inside
  a pipeline step.
- Checks against single-mesh `jax.jit`: every schedule and layout matches it
  in loss and parameters, over 100 steps for 1F1B.
- A baseline without the pipeline (`--parallel=dp` or `--parallel=fsdp`): the
  same model, weights and batches as one `jax.jit` program on all devices, as
  a pipeline's Gate 0 reference.
- One gradient norm over all stages for clipping, with no update and no
  checkpoint after a non-finite norm or loss, and a step watchdog.
- The ROCm scaling book's training metrics: a run directory (`--run_dir`)
  with step records and timeline events, `env.json`, GPU power and energy,
  validation loss, traces and RCCL collective times, summarized as step-time
  percentiles, MFU (sustained, rolling and end to end, against the nominal and
  a calibrated peak) and HFU, startup and checkpoint costs, ETTR, energy per
  token, pipeline utilization, scaling efficiency and Gate 0, with figures of
  MFU, loss and power over time and a per-op, per-stage FLOPs and
  communication model ([docs/metrics.md](docs/metrics.md)).
- A training budget planner (`python -m jax_titan.planning.budget`). For a
  dense or mixture-of-experts model, a token or time budget and one or more
  GPU types, it gives the run's FLOPs, time, GPU-hours, energy and cost:
  - GPU types: MI300X, MI325X, MI350X, MI355X, MI455X, H100, H200, B200 and
    TPU v5p, from vendor specs;
  - memory per GPU and the layouts that fit;
  - whether data-parallel communication hides behind compute;
  - inputs measured by a JAX-Titan run where one exists.
- A performance model for scaling up ([docs/planning.md](docs/planning.md)):
  - **Collectives:** RCCL and MORI-EP by message size, algorithm, protocol
    and bus bandwidth (`planning.collectives`), measured on 8x MI355X with
    `scripts/bench_collectives.sh`.
  - **Step time:** compute, each parallelism's collectives, overlap,
    pipeline bubble and runtime gaps (`planning.step`). It's calibrated
    against a run's profile and predicts other GPU counts.
  - **Time left** in a running job (`planning.remaining`).
  - **A performance database** of predicted and measured runs that flags
    regressions (`observability.perfdb`).
- MLPerf Training v6.1's Llama 3.1 8B, up to a trial on GPUs
  ([docs/mlperf.md](docs/mlperf.md)):
  - the contract, which refuses hyperparameters that depart from it;
  - MLPerf's C4 read in the order of the reference's Megatron-core;
  - the reference's schedule and AdamW;
  - validation on the fixed set, stopping at the target;
  - trials (`python -m jax_titan.mlperf.run`) warmed up before the clock
    starts, with an MLLOG that MLCommons' compliance checker accepts.

On 8x AMD Instinct MI355X, Llama 3.1 8B with 2 pipeline stages of 4 GPUs and
65,536 tokens per step trains at 1.45 s per step (45.1k tokens/s, 10.5% MFU)
with the zero-bubble schedule, against 1.68 s with 1F1B. Without the
pipeline, the same step as one `jax.jit` program takes 0.83 s with data
parallelism and 0.80 s with FSDP, so on one node the pipeline is still 1.7x
and 1.8x slower. [docs/pipelines.md](docs/pipelines.md) has every schedule and
layout measured, and the reasons for the gap. With TE's fused attention
(`--ops=attention=te`) the pipeline step takes 1.26 s (12.1% MFU). These
numbers clip each stage's gradients by its own norm; clipping by one norm over
all stages, as now, costs 3-4% (1.50 s, 1.31 s with TE attention). MaxText's
own trainer runs its pipeline at 1.76 s per step and FSDP at 0.72 s; see
[docs/pipelines.md](docs/pipelines.md#global-gradient-clipping-and-maxtexts-own-trainer-2026-11-07-8x-mi355x).

```python
from jax_titan import mpmd as jt_mpmd
from shardy.integrations.python.jax import mpmd

topology = jt_mpmd.make_topology(num_stages=2)            # mesh0, mesh1
config = jt_mpmd.make_config(topology, "ZERO_BUBBLE_H1")
train_step = jt_mpmd.make_train_step(stage_fns, optimizer)
compiled = mpmd.jit(
    train_step, config, in_shardings=..., out_shardings=...
).lower(params, opt_states, tokens, targets).compile()
```

## Requirements

JAX-Titan needs small patches to jaxlib and Shardy, kept in
[third_party/patches](third_party/patches):

- **jaxlib:** compile IFRT IR MPMD programs on PJRT-backed IFRT clients.
  Upstream jaxlib only compiles them through Pathways.
- **Shardy MPMD:** an import fix for open-source builds, and passing every
  partitioning option to the partitioner (jaxlib dropped five, including the
  one that keeps 1F1B in order).

XLA needs no patch: ROCm support in the IFRT IR atom-program passes landed as
[openxla/xla#52290](https://github.com/openxla/xla/pull/52290) (`dfca2c9`), so
build against XLA `main` at or after that commit.

The optional MaxDiffusion models need two MaxDiffusion patches, kept
there too: one imports PyTorch only to load PyTorch weights, and one fixes
FLUX.1-dev's timestep embedding at the pinned commit.

[docs/getting_started.md](docs/getting_started.md) covers the TheRock
container, the from-source JAX build and the Python environment.
[docs/tools.md](docs/tools.md) lists the scripts, command-line tools and agent
skills by the stage of a training job they serve.
[docs/adding_a_model.md](docs/adding_a_model.md) says how a model is added:
its module, its names and its builder.
[docs/adapters](docs/adapters) has one page per model repository whose models
train here: what comes from the repository and what from JAX-Titan, how to
run a model, and how the adapter is verified; and how to compare an adapter
with JAX-Titan's own model of the same structure.
[docs/amd-jax-train-runbook](docs/amd-jax-train-runbook) covers running
JAX-Titan on AMD's SPUR cluster.

## Layout

| Path | Contents |
|---|---|
| `jax_titan/mpmd` | Topology, Shardy MPMD config and pipelined train steps |
| `jax_titan/parallel` | DP, FSDP and TP shardings from logical parameter axes |
| `jax_titan/models` | Llama 3 and Ideogram 4, with parameters grouped by pipeline stage; FLUX.1's denoiser; Ideogram 4's text encoder, autoencoder, published-weights reader and generation; Command A+ with its vision tower and generation; and the models of MaxText, MaxDiffusion, transformers and diffusers, each called whole, with the stages of MaxDiffusion's Ideogram branch (`adapters/`) |
| `jax_titan/ops` | Semantic operators: per-op provider policy, native JAX, ROCm TE, JAX-AITER and JAX-Triton (Triton and Gluon) providers, and the conformance harness |
| `jax_titan/data` | Llama 3's tokenizer and a published model's, and pretokenized text sampling; Megatron's indexed datasets in Megatron-core's sample order; the cache of Ideogram 4's image tokens and text features; Command A+'s chat template, tokenizer and image tiles, and its cache of conversations |
| `jax_titan/trainer` | Training entry point (`python -m jax_titan.trainer.train`) and Orbax checkpoints |
| `jax_titan/observability` | Run records, metrics, figures, comparisons of runs, ETTR, ceilings, FLOPs model and per-GPU busy and idle time from profiler traces |
| `jax_titan/planning` | Training budget planner and performance model: model shapes, GPU specs, memory, RCCL and MORI collectives, step time, time left |
| `jax_titan/tuning` | hipBLASLt kernel selection for the GEMMs a program runs on ROCm: capture, sweep, override file and its checks; new kernels with TensileLite |
| `jax_titan/mlperf` | MLPerf Training: the benchmarks' contracts, MLLOG from the trainer's records, the trial runner and its configs |
| `examples/` | MPMD spike, parity checks, an example caption for Ideogram 4 and example conversations for Command A+ |
| `tests/` | CPU tests: schedules and layouts against `jax.jit`, checkpoint resume, MaxText, MaxDiffusion, Ideogram 4, Command A+ and FLUX.1 models, the comparison of runs |
| `scripts/` | Container, build, environment, GPU-availability and post-run cleanup scripts |
| `.cursor/skills/`, `.claude/skills/` | Agent skills for Cursor and Claude Code (identical copies): coding rules, upstream debugging, and running and cleaning up training jobs |
| `third_party/patches/` | jaxlib, Shardy and MaxDiffusion patches |
| `third_party/jaxpp/` | The earlier JaxPP project, kept as-is for a later comparison |
| `third_party/jax-aiter/`, `third_party/jax-triton/`, `third_party/gfx950-gluon-tutorials/` | JAX-AITER, JAX-Triton and the gfx950 Gluon GEMMs (git submodules), for the operator providers |

## Roadmap

- Model reference providers: MaxText for LLMs (Llama 3, Qwen3 dense and MoE,
  Mixtral, DeepSeek V4), pinned at `maxtext-v0.2.5`, and MaxDiffusion for
  diffusion models (Flux, then Wan 2.2 with its dual-expert training), pinned
  at `821afc36`. JAX-Titan takes their model definitions, configs and
  checkpoint converters, and replaces their trainers, sharding and TPU kernels
  with its own.
- Operator providers behind `jax_titan.ops`, each checked against its native
  JAX ops, in this order: ROCm Transformer Engine, from the nightly ROCm TE
  wheels that ROCm/maxtext publishes; JAX-AITER; JAX-Triton, for Triton and
  Gluon kernels including AITER's; and JAX-FlyDSL. JAX-AITER and JAX-Triton
  are pinned as git submodules in `third_party/`. MORI-EP follows for MoE
  dispatch and combine.
- Interleaved 1F1B and DualPipeV as custom Shardy schedules, and faster
  checkpoint saves.
- Multi-node execution, and RL post-training with Ray and vLLM.

## License

MIT, see [LICENSE](LICENSE). The submodules in `third_party/` keep their own
licenses (JAX-AITER and gfx950-gluon-tutorials MIT, JAX-Triton Apache-2.0),
and two directories keep the Apache License 2.0:

- `third_party/jaxpp/`: the original JaxPP project, with its own `LICENSE` and
  file headers.
- `third_party/patches/`: patches to jax, Shardy and MaxDiffusion, which are
  Apache-2.0 projects. See [third_party/patches/README.md](third_party/patches/README.md).
