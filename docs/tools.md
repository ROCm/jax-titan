# Tools

JAX-Titan's scripts, command-line tools and agent skills, grouped by the
stage of a training job they serve. Each entry says what the tool answers and
links to the page that documents it.

## A job, stage by stage

1. **Set up once.** Start the container with `scripts/container.sh` on the
   host, build the wheels, `source scripts/env.sh` in every shell in the
   container, and prepare the data with `python -m jax_titan.data.text`
   ([getting_started.md](getting_started.md)), or with
   `python -m jax_titan.data.ideogram` for images
   ([recipes/ideogram4.md](recipes/ideogram4.md)), or with
   `python -m jax_titan.data.command_a` for conversations
   ([recipes/command_a_plus.md](recipes/command_a_plus.md)).
2. **Check the node.** On the host, `scripts/preflight_dev.sh` shows disk
   space and who holds each GPU, and `scripts/gpus_free.sh` must print
   `free`. After a code change, run `python -m pytest tests/` first.
3. **Plan.** `planning.budget`, `planning.step` and `observability.flops`
   estimate time, memory and the stage split, on CPU.
4. **Run.** `python -m jax_titan.trainer.train` under `timeout`, with
   `--run_dir` so that the run is recorded
   ([metrics.md](metrics.md#recording-a-run)).
   On a Slurm cluster, `scripts/slurm_submit.sh` submits the run as an sbatch
   job that also starts the container and builds what is missing.
   `python -m jax_titan.models.ideogram_generate` draws pictures with
   Ideogram 4, published or fine-tuned, and
   `python -m jax_titan.models.command_a_generate` answers prompts with
   Command A+, published or fine-tuned.
5. **Clean up.** `scripts/cleanup_after_train.sh` shows what it would stop and
   remove; `--yes` applies it.
6. **Read the run.** `observability.summarize`, `profile`, `plot` and `ettr`;
   `observability.gemms` lists the GEMMs of a run dumped with `--xla_dump_to`.
   A measured run makes the next plan more accurate through
   `planning.budget --from_run` and `planning.step --calibrate`.

## Scripts

Scripts marked host run outside the container, where they see every user's
processes. The others need the container's TheRock libraries.

| Script | Runs on | What it does |
|---|---|---|
| `container.sh` | host | Starts the TheRock 10.1 development container with all GPUs and a user with your host uid, so files in the workspace keep your owner ([getting_started.md](getting_started.md#1-container)) |
| `env.sh` | container | Activates the venv; sets TheRock's library path, `PYTHONPATH` and the JAX compilation cache; turns off GPU core dumps; sets `JAX_TITAN_ROCM_XLA_FLAGS` for GPU runs. Source it in every shell |
| `build_jax_rocm.sh` | container | Builds the jax, jaxlib and ROCm plugin wheels against `${JAX_TITAN_ROOT}/src/xla` and the patched Shardy; `WHEELS=jaxlib` rebuilds only jaxlib ([getting_started.md](getting_started.md#3-build)) |
| `publish_ci_wheels.sh` | host | Publishes the wheels of the last build (jax, jaxlib and the two ROCm plugin wheels), with Shardy's MPMD package, as a draft GitHub release for CI, and writes their checksums to `.github/ci-wheels.sha256`. Run it after a build with a new revision or patch. Needs `gh` ([ci.md](ci.md#new-wheels)) |
| `build_jax_aiter.sh` | container | Builds JAX-AITER from its submodule against the installed jaxlib, with prebuilt AITER JIT libraries |
| `build_rccl_inspector.sh` | container | Ports RCCL's Inspector profiler plugin to HIP, for the trainer's `--rccl_inspector` ([metrics.md](metrics.md#recording-a-run)) |
| `gpus_free.sh` | host | Prints `free` and exits 0 when no process holds a GPU and every GPU is idle; prints what is busy otherwise |
| `preflight_dev.sh` | host | Reads only: free space on each filesystem, each GPU's load and the users holding it, the top users by disk and the largest containers. Exits 1 when a process holds a GPU, a GPU is busy, or a filesystem is below 10% free. `--top N` sets the list length; `--no-docker` skips container sizes |
| `cleanup_after_train.sh` | host or container | Prints what it would do: stop your JAX-Titan processes, wait until they release the GPUs, and remove your stale `/dev/shm/nccl-*` segments, the Orbax checkpoints named with `--checkpoint`, and your GPU core dumps. `--yes` applies it |
| `bench_collectives.sh` | container | Measures RCCL (and MORI-EP when installed) time and bus bandwidth by message size on this node, for the planner. Needs free GPUs ([planning.md](planning.md#collectives-rccl-and-mori)) |
| `build_hipblaslt_sweep.sh` | container | Builds `hipblaslt_sweep`, which lists, screens, times and validates hipBLASLt solutions for one problem, into `${JAX_TITAN_ROOT}/bin` ([gemm_tuning.md](gemm_tuning.md)) |
| `tune_gemms.sh` | container | Tunes hipBLASLt kernel selection for one training configuration and checks it: clock, capture, sweep, override file, verification, step A/B, per-site times, cost breakdown and negative tests. Needs free GPUs ([gemm_tuning.md](gemm_tuning.md)) |
| `setup_sources.sh` | login node or host | Creates the `jax`, `xla` and `shardy` trees in `${JAX_TITAN_ROOT}/src` at the tested revisions, as worktrees of sibling `../jax` and `../xla` checkouts when they exist, and applies the patches. Git only |
| `slurm_submit.sh` | Slurm login node | Submits `slurm_job.sh` as a one-node, 8-GPU sbatch job with logs in `${JAX_TITAN_ROOT}/logs/slurm/`; trainer flags go after `--`; `-n` prints the sbatch line, `--setup-only` stops before training. Cluster settings (partition, QOS, scratch, GPU request) come from a file named by `JT_SITE_ENV`, kept outside the repository |
| `slurm_job.sh` | compute node (sbatch) | Checks the node (docker, AMD GPUs, shared filesystem, free GPUs), starts the TheRock container, builds the wheels and the venv when missing, runs the trainer under a timeout, and on exit, also after SIGTERM, cleans up and removes the container |

`preflight_dev.sh` and `cleanup_after_train.sh` read `USER`, which
`docker exec` does not set; export it first in a shell started that way. The
cleanup's process pattern also matches your other `pytest` and `tests/`
processes, so read the dry run's list before `--yes`.

## Command-line tools

Run each with `python -m jax_titan.<module>`, or `python <path>` for the
examples. The planners, and the tools that read a finished run (`summarize`,
`profile`, `plot`, `ettr`, `collectives`, `perfdb`, `gemms`), need no GPU.

### Correctness

| Command | What it checks |
|---|---|
| `python -m pytest tests/` | On CPU with 8 devices: schedules and layouts against `jax.jit`, exact checkpoint resume, the MaxText and MaxDiffusion adapters (skipped without their extras), Ideogram 4 from a toy checkpoint to pictures and against diffusers' and transformers' models (those skipped without `torch`), Command A+ from a toy checkpoint through training to generated text and against transformers' model (skipped without `torch`), the planners, and the agent skills' two copies |
| `python -m pytest tests/ --gpus=N` | On ROCm, only the tests marked `gpu` that need at most N GPUs: operator conformance of `native`, and of the XLA quantized GEMMs (FP8, MXFP8, MXFP4) at Llama 3.1 8B sizes, and Command A+'s loss and gradient norm against those recorded on CPU |
| `python examples/<name>.py --platform=cpu` | A two-stage MLP step (`mpmd_two_mesh_spike.py`) and a pipeline step, each against one `jax.jit` program; the adapter stages against MaxText's and MaxDiffusion's own models ([getting_started.md](getting_started.md#5-runs)) |
| `ops.conformance` | Operator providers against native JAX: outputs and gradients on one GPU, under `jax.vmap`, sharded, inside `shard_map` and inside an MPMD step, and their timings ([getting_started.md](getting_started.md#5-runs)) |

A test in `tests/` is a CPU test unless it is marked `gpu`, and each of the two
pytest commands above runs one kind. CI runs both, the first on a CPU runner
and the second on one gfx950: [ci.md](ci.md) has the marker, the two jobs and
how to update the packages they install.

### Images

| Command | What it does |
|---|---|
| `data.ideogram --checkpoint=CKPT --images=DIR --out=CACHE` | Caches Ideogram 4's training samples of a directory of images with captions: the autoencoder's image tokens and the text encoder's features, for the trainer's `--data=CACHE` ([recipes/ideogram4.md](recipes/ideogram4.md#fine-tuning-on-images)) |
| `models.ideogram_generate --checkpoint=CKPT --prompt=@FILE --out=PNG` | Draws pictures with Ideogram 4 from a checkpoint in diffusers' layout, with `--transformer` from fine-tuned parameters, on one GPU ([recipes/ideogram4.md](recipes/ideogram4.md#pictures-from-the-published-weights)) |
| `data.command_a --checkpoint=CKPT --conversations=FILE --out=CACHE` | Caches conversations for fine-tuning Command A+: their token ids by the checkpoint's chat template and tokenizer, with the assistant's last turn marked as what the model learns, for the trainer's `--data=CACHE` ([recipes/command_a_plus.md](recipes/command_a_plus.md#fine-tuning-on-conversations)) |
| `models.command_a_generate --checkpoint=CKPT --prompt=TEXT --image=FILE` | Answers a prompt, with pictures or without, with Command A+ from a checkpoint in the published layout, with `--params` the trained or fine-tuned parameters in place of the checkpoint's, its layers spread over the visible GPUs ([recipes/command_a_plus.md](recipes/command_a_plus.md#text-from-a-checkpoint)) |

### Measuring a run

[metrics.md](metrics.md#tools) has their options and the metrics they report.

| Command | What it answers |
|---|---|
| `observability.summarize RUN` | Step-time percentiles, MFU and HFU, startup and checkpoint costs; across runs, repeats and scaling. `--gate0 REFERENCE RUN` compares losses against a reference run |
| `observability.profile TRACE_DIR` | Each GPU's compute, exposed collective, copy and idle shares from a `jax.profiler` trace |
| `observability.plot RUN` | Figures of MFU, step time, loss, gradient norm and power over the run |
| `observability.ettr RUN` | ETTR, and the checkpoint interval and ETTR projected to more nodes |
| `observability.collectives DIR` | RCCL Inspector completion times by operation and message size |
| `observability.gemms DUMP` | Every GEMM the compiled program runs, from an XLA HLO dump of any program: M, N, K, batch, dtypes, operand layouts, library (hipBLASLt, rocBLAS, Triton) and kernel configuration, with launches per run, or per training step with `--run_dir`. Standard library only: `python3 src/jax_titan/observability/gemms.py DUMP` runs without JAX or an install ([metrics.md](metrics.md#gemms-in-the-compiled-program)) |
| `observability.gemms DUMP --times KERNELS` | The same sites ranked by measured time: each site's launches from `profile --kernel_times` of a trace, with the medians of launches no collective overlapped and of overlapped ones ([gemm_tuning.md](gemm_tuning.md)) |
| `observability.gemm_bench PROBLEMS` | Isolated GEMM timings through JAX per problem and backend: hipBLASLt default or tuned, XLA's autotuner, XLA's Triton GEMMs, AITER's Triton GEMM, JAX-AITER's ASM GEMM, or an operator provider (needs a GPU) ([gemm_tuning.md](gemm_tuning.md)) |
| `tuning/hipblaslt.py` | hipBLASLt problems from logs (`capture`), joined to GEMM sites (`join`), swept with confidence intervals (`sweep`), written as an override file (`write`), checked in a run (`verify`), carried to another build (`map`), and the clock a GPU sustains (`clock`). Standard library only ([gemm_tuning.md](gemm_tuning.md)) |
| `planning.gemm_cost RESULTS` | Per GEMM problem and per step: the time a tuning file recovers, what collectives cost, the rest of the in-step gap, and the ceiling below the best kernel, against a roofline or Origami's attainable time ([gemm_tuning.md](gemm_tuning.md#where-the-gemm-time-goes)) |
| `observability.power OUT.jsonl` | GPU power, energy, clocks and throttling, sampled beside any job |
| `observability/counters.py run OUT -- CMD`, `report OUT` | Per-launch hardware counters from rocprofv3: each kernel's engine clock, matrix-unit use, L2 hit rate and traffic past L2. Counting serializes the counted launches. Standard library only |
| `observability.env OUT.json` | What a run ran on: packages, commits, flags, GPUs and the rank map |
| `observability.ceilings OUT.json` | The GEMM and HBM rates XLA reaches on one GPU, for MFU against an attainable peak (needs a GPU) |
| `observability.detectors OUT -- ARGS` | The step-time cost of each failure detector; runs the trainer once per detector |
| `observability.perfdb perf.jsonl` | Runs slower than the best earlier run with the same model, GPU and layout |

### Planning

| Command | What it answers |
|---|---|
| `planning.budget` | Time, GPU-hours, energy, cost and memory of a run on one or more GPU types; `--from_run` uses a measured run instead of an assumed MFU; a scaling law's loss and compute-optimal size, and with `--target_loss` the tokens and time to reach a loss ([planning.md](planning.md)) |
| `planning.scaling` | The compute-optimal model for a compute budget and the cheapest one to a target loss, from Chinchilla's scaling law refit or your own; `--fit` fits a law to JAX-Titan runs ([planning.md](planning.md#scaling-laws-loss-and-the-compute-optimal-size)) |
| `planning.step` | Step time by part: compute, collectives, bubble and runtime gaps. `--calibrate RUN` fits the parts to a run; `--ladder` predicts other GPU counts, with `--weak` keeping the batch per GPU ([planning.md](planning.md#step-time-calibration-and-the-scale-ladder)) |
| `planning.collectives` | RCCL and MORI time by message size; `--calibrate` turns `bench_collectives.sh` output into a table ([planning.md](planning.md#collectives-rccl-and-mori)) |
| `planning.remaining RUN` | Time left in a running job ([planning.md](planning.md#during-a-run-time-left-and-the-performance-database)) |
| `planning.model_index` | The models the planners know by name, with the Hugging Face repository and commit each config came from; `--add NAME REPO` pins another ([planning.md](planning.md#the-model-config-index)) |
| `observability.flops` | Model and executed FLOPs per op and per stage, and the layer split that balances the stages ([metrics.md](metrics.md#tools)) |

## Agent skills

Four skills hold the project's rules for coding agents, with identical
content in `.claude/skills/` (Claude Code) and `.cursor/skills/` (Cursor).
An agent working in this repository loads a skill when the task matches its
description; you can also name it in the request, for example "use
jax_titan_training to run the no-pipeline baseline".

| Skill | Use it when | What it holds |
|---|---|---|
| `jax_titan_coding` | Writing or reviewing code in this repository | Architecture rules (stages, meshes, shardings, operators, adapters, precision), testing rules (a new schedule, layout or model matches `jax.jit`), and code style |
| `jax_titan_training` | Launching, monitoring, comparing or cleaning up a training job | Checks before a run, launch commands, red flags while it runs, how to compare against a baseline, and a cleanup checklist; `reference.md` adds run names, what to record for each result and the notes file |
| `jax_titan_debug` | A stack trace or log points into jaxlib, XLA, Shardy or ROCm | Which component owns a failure, known failures and their fixes, and the workflow for an upstream patch |
| `jax_titan_slurm` | Running a job through Slurm sbatch | Requirements and cluster settings, the source trees, submitting with `slurm_submit.sh`, why a job is pending, what the job log shows stage by stage, expected step times, failures and their fixes |

Edit both copies of a skill: `tests/test_agent_skills.py` fails when they
differ. The SPUR runbook adds rules for agents on that cluster
([amd-jax-train-runbook/README.md](amd-jax-train-runbook/README.md#for-agents)).
