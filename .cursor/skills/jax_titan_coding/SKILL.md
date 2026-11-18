---
name: jax_titan_coding
description: >-
  Architecture rules and coding standards for JAX-Titan, the JAX-native MPMD
  training framework (Shardy MPMD pipelines run as IFRT IR programs on ROCm).
  Use when writing, reviewing or changing code in the JAX-Titan repo: pipeline
  schedules and stages, shardings and layouts, models, the trainer,
  checkpoints, tests and docs.
---

# JAX-Titan Coding Guide

## Core Principle

JAX-Titan is JAX-native: it builds on `jax.jit`, Shardy MPMD, IFRT IR, PJRT
and XLA, with no PyTorch runtime. A pipeline is one Shardy MPMD program, not
hand-written point-to-point code. Changes to jaxlib, XLA or Shardy are
upstreamable patches in `third_party/patches/`, never workarounds in
`src/jax_titan`. `third_party/jaxpp/` is the earlier JaxPP project, kept unchanged
for a later comparison: do not import from it or edit it.

## Architecture Rules

- **Stages and microbatches:** each stage is an `mpmd.named_computation`
  assigned to a mesh through `jax_titan.mpmd.make_config`; each microbatch is
  an `mpmd.call` with its own call counter (`jax_titan.mpmd.pipeline_loss`).
  Schedules come from Shardy's registry, or later from custom
  `FragmentScheduleRule` builders.
- **Meshes:** every stage mesh has `data` and `tensor` axes
  (`make_topology`). Sharding constraints inside one MPMD program must all use
  one stage mesh; Shardy MPMD only reads their axes.
- **Shardings:** models name the logical axes of each parameter
  (`models/llama.py: logical_axes`); `jax_titan.parallel.tree_specs` turns them
  into PartitionSpecs for DP, FSDP and TP. Do not write PartitionSpecs by hand
  in the trainer. Specs drop trailing Nones, because MPMD executables return
  normalized shardings and a train step's inputs must equal its last outputs.
- **Gradient reduction:** with `--reduce_grads=step`, each data-parallel rank
  trains its own parameter copy and stage functions must not add sharding
  constraints. With `microbatch`, stage functions constrain activations at
  stage boundaries.
- **Operators:** models call semantic ops from `jax_titan.ops` (`attention`,
  `rms_norm`, `swiglu`, `linear`), never a kernel library directly.
  Providers (`ops/providers/`: TE, JAX-AITER) are picked per op by the
  policy (`--ops`) while tracing, with `native` as the fallback. A provider
  op declines, through `supports()`, any call it cannot run correctly:
  wrong dtype or shape, under `jax.vmap` without a working batching rule,
  or JAX-AITER attention next to TE. It never fails or returns wrong values
  silently. A new provider op passes `python -m jax_titan.ops.conformance`
  in every mode before a policy uses it.
- **MaxText models:** `models/adapters/maxtext.py` builds stages from
  MaxText's own layers at the pinned release (`maxtext==0.2.5`, the
  `maxtext` extra). MaxText's layers constrain activations on the mesh they
  are built with, so build them on one stage mesh and reduce gradients per
  microbatch. Stage parameters keep MaxText's names and layer axis;
  `llama_params` maps Llama weights onto `models/llama.py`.
- **MaxDiffusion models:** `models/adapters/maxdiffusion.py` builds FLUX.1
  stages from MaxDiffusion's transformer at commit `821afc36` with the
  `third_party/patches/maxdiffusion-*.patch` patches. Stage boundaries carry
  a tuple (image and text tokens, conditioning, rotary embeddings), so these
  stages reduce gradients per microbatch.
- **Ideogram 4:** `models/ideogram.py` is JAX-Titan's own model, written after
  diffusers' `Ideogram4Transformer2DModel` and its training example. It
  trains through the layouts of `trainer/adapters.py` (`--model=ideogram4`),
  so with gradients reduced per microbatch. `models/adapters/ideogram.py`
  builds the same stages from MaxDiffusion's Ideogram branch
  (`maxdiffusion:ideogram4`), and its `native_params` maps those weights onto
  `models/ideogram.py`. Compute the rotary frequencies on the host: a compiled
  program folds the same expression to other last bits, and image positions
  start at 65536. The models around the transformer are JAX-Titan's own too:
  `models/qwen3.py` (the text encoder) on `jax_titan.ops`, and
  `models/flux2_vae.py` (the autoencoder) in plain JAX, on one device outside
  any mesh. Both are frozen: `data/ideogram.py` runs them once into a cache
  that the trainer reads (`--data`), and `models/ideogram_generate.py` runs
  them to draw pictures. Image tokens are scaled by `TOKEN_MEAN` and
  `TOKEN_STD` of `models/ideogram.py`, the constants of Ideogram's pipeline,
  never by the batch-norm statistics stored with the autoencoder. Published
  weights are read by `models/pretrained.py` without PyTorch; torch, diffusers
  and transformers appear in tests only, as references.
- **Precision:** parameters and optimizer state are fp32; compute casts to
  bf16 before the layer scan. Casting inside the rematerialized layer was
  measured slower; do not move it back without a new A/B.
- **Trainer:** builders return a `Layout` with `init()` and `abstract_state`,
  so checkpoints restore without initializing first. Entry points import jax
  inside `main()`, after `--platform=cpu` sets `JAX_PLATFORMS`.
- **Checkpoints** go through `jax_titan.trainer.checkpoint` (Orbax). Restoring
  takes target shardings from the current layout.

## Testing Rules

- `python -m pytest tests/` (CPU, 8 devices) must pass before every commit.
- CI runs the tests in `tests/` on the JAX built from source, with MaxText
  (the `maxtext` extra) and MaxDiffusion (at the `PINNED_COMMIT` of its
  adapter, with the patches) installed, so the adapter tests run there. It
  has neither `torch` nor MaxDiffusion's Ideogram branch, so Ideogram 4's
  comparisons with other implementations run only locally: run them after
  changing one of its models.
- Mark a test that needs a GPU `gpu`, or `gpu(n)` for n GPUs.
  `python -m pytest tests/` never runs it; `python -m pytest tests/ --gpus=N`
  runs only the `gpu` tests, with JAX on ROCm, and skips those that need more
  than N. Unless `XLA_FLAGS` is set, it turns XLA's GEMM autotuning off
  (hipBLASLt's first choice for every GEMM): with autotuning the tests take 20
  minutes instead of 90 seconds. CI's GPU job runs `--gpus=1` on one gfx950,
  so a test that needs more GPUs runs only by hand on a dev node.
- CI's JAX built from source is a release of wheels named in
  `.github/workflows/ci.yml`, with their checksums in
  `.github/ci-wheels.sha256`. After a change to a revision in
  `scripts/setup_sources.sh` or to a patch, build again and publish new
  wheels with `scripts/publish_ci_wheels.sh`, or CI tests the old stack.
- CI installs every other package at the version `.github/ci-constraints.txt`
  names, so a new release of a dependency cannot turn CI red. Update the file
  by hand, as its first lines say; a package it does not name is installed at
  its newest version.
- A new schedule, layout or model gets a case in `tests/test_mpmd_parity.py`:
  the pipeline step must match single-mesh `jax.jit` in fp32 with SGD
  (`rtol=1e-5` over 4 steps).
- Resume behavior is tested in `tests/test_checkpoint.py`: a resumed run must
  equal the uninterrupted one exactly on CPU.
- `tests/test_maxtext_adapter.py` checks MaxText's model, `models/llama.py`
  and the pipeline step against each other; run it after changing either
  model (it is skipped without the `maxtext` extra).
  `tests/test_maxdiffusion_adapter.py` does the same for FLUX.1 against
  MaxDiffusion's transformer (skipped without MaxDiffusion).
- `tests/test_ideogram.py` checks `models/ideogram.py` against diffusers'
  model (skipped without `torch` and `diffusers`), its stages against one
  stage, and its layouts against `--parallel=dp`.
  `tests/test_ideogram_adapter.py` checks it against the Ideogram branch's
  model and against one step of that branch's trainer (skipped without the
  branch). `tests/test_qwen3.py` and `tests/test_flux2_vae.py` check the text
  encoder and the autoencoder against transformers' and diffusers' models
  (skipped without `torch`). `tests/ideogram_checkpoint.py` writes a
  checkpoint in diffusers' layout at toy size, from which the cache of
  samples, training on it and generation run on CPU.
- A performance claim needs a GPU run on an idle node against the
  no-pipeline baseline from the same stack (see `jax_titan_training`), with
  its JSONL in `docs/results/<date>/`.

## Code Style

- Match the surrounding code: 2-space indentation, prose docstrings that say
  what a function returns and what its caller must know, type hints on
  public functions, `# pylint: disable=g-import-not-at-top` on deferred
  imports.
- Comments only state constraints the code cannot show, never what the next
  line does or why a change is correct.
- Spell the project `JAX-Titan`. Write docs in plain technical English with
  complete sentences, and keep measured numbers in `docs/pipelines.md`.
- Commit locally in logical pieces with a message that says what changed and
  why. Never push; the user pushes. Never edit the user's own `jax` or `xla`
  checkouts; use the worktrees in `${JAX_TITAN_ROOT}/src`.

## Keeping Skills in Sync

The skills exist twice, in `.cursor/skills/` and `.claude/skills/`, with
identical content. Edit both; `tests/test_agent_skills.py` fails when they
differ.

## Pairing

Pair with `jax_titan_training` to run and measure a change, and with
`jax_titan_debug` when the fault is in jaxlib, XLA, Shardy or ROCm.
