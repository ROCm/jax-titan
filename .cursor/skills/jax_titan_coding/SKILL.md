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
and XLA, and no training step runs PyTorch: the adapters of PyTorch models
execute PyTorch only while JAX traces a model. A pipeline is one Shardy MPMD program, not
hand-written point-to-point code. Changes to jaxlib, XLA or Shardy are
upstreamable patches in `third_party/patches/`, never workarounds in
`jax_titan`. `third_party/jaxpp/` is the earlier JaxPP project, kept unchanged
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
- **MaxText models:** `models/adapters/maxtext.py` runs MaxText's models
  from its own layers at the pinned release (`maxtext==0.2.5`, the `maxtext`
  extra). `WholeModel` calls MaxText's `Transformer` whole, as one stage for
  `--parallel=dp` and `fsdp`, with no code per model family, and refuses a
  config whose loss has more terms than the cross-entropy and the MoE
  load-balance term. Parameters keep MaxText's names and layer axis;
  `llama_config` and `llama_params` map a Llama model onto `models/llama.py`,
  whose reference the adapter is.
- **MaxDiffusion models:** `models/adapters/maxdiffusion.py` runs
  MaxDiffusion's denoisers at commit `821afc36` with the
  `third_party/patches/maxdiffusion-*.patch` patches. `WholeModel` calls a
  denoiser whole, as one stage for `--parallel=dp` and `fsdp`: FLUX.1 and
  Wan 2.1, each with a few lines (`whole_flux`, `whole_wan`).
- **A model of another repository is one stage.** The adapters of MaxText,
  MaxDiffusion, transformers and diffusers call the repository's model whole
  and do not cut it: a pipeline needs a model written for JAX-Titan, as
  `models/llama.py` is. The one exception is `maxdiffusion:ideogram4`, whose
  stages `models/adapters/ideogram.py` builds from the Ideogram branch.
- **PyTorch models:** `models/adapters/torch_bridge.py` turns a
  `torch.nn.Module` into one stage through torchax, and is the only module
  that imports torchax. PyTorch's autograd takes no part, so a model that
  leans on it (`torch.no_grad()`, `.data`, a custom `autograd.Function`)
  comes out wrong without an error: `models/adapters/transformers.py` and
  `diffusers.py` take only what was compared with PyTorch (`VERIFIED`,
  `FAMILIES`). Compare again when transformers, diffusers, torchax or torch
  moves (`docs/adapters/transformers.md`). These models take no `--ops`.
- **Diffusion objectives:** a denoiser's repository keeps the noising, the
  timestep and the target outside the model. JAX-Titan states them once per
  family in `data/diffusion.py`, for every adapter that runs the family.
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
- **FLUX.1:** `models/flux.py` is JAX-Titan's own model, written after
  diffusers' `FluxTransformer2DModel`, as one stage for `--parallel=dp` and
  `fsdp` on the batches of `data/diffusion.py` (`--model=flux-dev`,
  `flux-schnell`). Its queries and keys keep each head's features in split
  halves, the layout of `ops.rope`, where diffusers interleaves the pairs, so
  diffusers' weights map onto it with that permutation (`tests/test_flux.py`).
- **Command A+:** `models/command_a.py` is JAX-Titan's own model, written
  after transformers' `cohere2_moe` and `cohere2_vision`: the sparse
  mixture-of-experts language model, the SigLIP vision tower and the
  projector. Its parameters are the published checkpoint's, with kernels as
  [in, out], layers and experts stacked, and each head's query and key
  dimensions reordered for `ops.rope`; `params_from_tensors` and
  `tensors_from_params` map both ways, so keep them inverse. Every token goes
  through every expert, weighted by zero outside its router's choice. It
  trains through the layouts of `trainer/adapters.py`
  (`--model=command-a-plus`): as one stage, since its output head is its
  embedding matrix, or as a pipeline with the embedding frozen and a copy on
  the first stage and on the last. `data/command_a.py` applies the
  checkpoint's chat template, tokenizer and image settings and caches
  conversations to fine-tune on, and `models/command_a_generate.py` generates
  text. The published model is too large for one device:
  `load_stages` reads its checkpoint stage by stage onto several, a run of
  layers on each, with the transposing and casting done on the device, and
  generation runs over these stages.
- **Frozen parameters:** a staged model may train part of its parameters:
  `is_frozen(path)` names the rest (Command A+'s `freeze`). The layouts of
  `trainer/adapters.py` keep frozen parameters out of the trained state, in
  the model's compute dtype, with no gradient and no optimizer state; the
  step takes them after the state (`Layout.frozen`), and neither a
  checkpoint nor `--save_params` holds them: a run that resumes reads them
  again from `--init_from`, which the trainer checks against the checkpoint.
  Do not freeze by masking updates: the gradients of frozen parameters would
  still be computed, and every checkpoint would hold them.
  A model whose weights exceed the host's memory gives `place(path, targets)`,
  which reads published weights straight onto the stages' devices.
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
  adapter, with the patches) installed, so the adapter tests run there; the
  GPU job also installs the `transformers` and `diffusers` extras with
  PyTorch's CPU build. The CPU job has neither `torch` nor MaxDiffusion's
  Ideogram branch, so Ideogram 4's and Command A+'s comparisons with other
  implementations, which are CPU tests, run only locally: run them after
  changing one of these models.
- Mark a test that needs a GPU `gpu`, or `gpu(n)` for n GPUs.
  `python -m pytest tests/` never runs it; `python -m pytest tests/ --gpus=N`
  runs only the `gpu` tests, with JAX on ROCm, and skips those that need more
  than N. Unless `XLA_FLAGS` is set, it turns XLA's GEMM autotuning off
  (hipBLASLt's first choice for every GEMM): with autotuning the tests take
  several times as long. CI's GPU job runs `--gpus=1` on one gfx950, so a test
  that needs more GPUs runs only by hand on a dev node.
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
- `tests/test_maxtext_adapter.py` checks the whole model against MaxText's
  own loss function, FSDP against data parallelism, and `models/llama.py`
  against the whole model from the same weights, on GPUs (`--gpus=2`; skipped
  without the `maxtext` extra); run it after changing either Llama.
  `tests/test_maxdiffusion_adapter.py`
  checks FLUX.1 and Wan 2.1 against MaxDiffusion's own modules (skipped
  without MaxDiffusion). `tests/test_adapter_training.py` trains a model of
  each through the trainer on CPU, under dp and FSDP, and resumes it exactly.
  `tests/test_transformers_adapter.py` and `tests/test_diffusers_adapter.py`
  compare the traced PyTorch models with PyTorch's own loss and gradients, on
  GPUs (skipped without the `transformers` or `diffusers` extra).
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
- `tests/test_flux.py` checks `models/flux.py` against diffusers' model at
  the same weights in prediction, loss and gradients, on GPUs (dev and
  schnell; skipped without the diffusers extra), and its initialization, FSDP
  against data parallelism and resume on CPU.
  `tests/test_compare.py` checks `observability/compare.py` on runs it writes
  and on the trainer's own records.
- `tests/test_command_a.py` checks `models/command_a.py` against
  transformers' model in logits, loss and gradients, and the checkpoint
  layout both ways (skipped without `torch` and a `transformers` with
  `cohere2_moe`). It also holds the loss and gradient norm of the test model
  as recorded on CPU, which a `gpu` test repeats on ROCm: after a change that
  moves them, check against transformers again before recording new ones.
  It pins the checkpoint layout by a digest too, since CI runs none of the
  comparisons: record a new one only while transformers reads the layout.
  `tests/test_command_a_generate.py` checks preprocessing and
  generation, and `tests/test_command_a_training.py` its layouts against
  `--parallel=dp`, with and without frozen parameters, a resumed run, and
  fine-tuning from a cache of conversations to generated answers.
  `tests/command_a_checkpoint.py` writes a checkpoint in the published layout
  at toy size.
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
  complete sentences. A model of JAX-Titan's own has a recipe page in
  `docs/recipes/`, which holds what it is, how to run it and its measured
  numbers; `docs/pipelines.md` holds those of pipelines, layouts and
  operators, and describes no particular model.
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
