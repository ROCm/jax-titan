# MaxDiffusion adapter

`--model=maxdiffusion:<name>` trains a denoiser of MaxDiffusion, pinned at
commit `821afc36`, with JAX-Titan's trainer. The network is MaxDiffusion's and
runs as MaxDiffusion wrote it; the training is JAX-Titan's. `WholeModel`
(`models/adapters/maxdiffusion.py`) calls a denoiser whole, MaxDiffusion's own
`__call__`, under `--parallel=dp` or `fsdp`, with a few lines per family.

## Models

`<name>` is a denoiser of one of the two families this version has lines for,
FLUX.1 and Wan 2.1 text to video:

| `--model=` | Model | Parameters | Run on GPUs at real size |
|---|---|---|---|
| `maxdiffusion:flux-dev` | FLUX.1-dev | 11.90B | Yes |
| `maxdiffusion:flux-schnell` | FLUX.1-schnell | | No |
| `maxdiffusion:wan2.1-t2v-1.3b` | Wan 2.1 text to video, 1.3B | 1.42B | Yes |

MaxDiffusion's other families (Stable Diffusion, SDXL, Wan 2.2 and its
image-to-video and VACE models, LTX-Video, Z-Image) are not taken: each needs
its lines in the adapter and its objective. `maxdiffusion:ideogram4` is not
this adapter but the model of MaxDiffusion's Ideogram branch
([recipes/ideogram4.md](../recipes/ideogram4.md)).

## Running it

Install MaxDiffusion at the pinned commit with JAX-Titan's patches, and the
`maxdiffusion` extra ([getting_started.md](../getting_started.md)). FLUX.1
whole, on every GPU of the node:

```bash
python -m jax_titan.trainer.train --model=maxdiffusion:flux-dev --parallel=fsdp \
  --microbatches=1 --microbatch_size=8 --resolution=512 --steps=30 --lr=1e-5
```

Wan 2.1 1.3B on clips of 17 frames of 480 x 480 pixels:

```bash
python -m jax_titan.trainer.train --model=maxdiffusion:wan2.1-t2v-1.3b --parallel=fsdp \
  --microbatches=1 --microbatch_size=8 --resolution=480 --frames=17 --steps=30 --lr=1e-5
```

`--resolution` is the side of an image or a frame in pixels and `--frames` the
frames of a clip: Wan sees `(frames - 1) / 4 + 1` latent frames of
`(resolution / 16)^2` tokens each. `--text_tokens` defaults to 512, and to 256
for `flux-schnell`. `--model_overrides` takes the denoiser's constructor
arguments as JSON.

## What comes from MaxDiffusion and what from JAX-Titan

| Part | Comes from |
|---|---|
| Model: structure, layers, config, initialization | MaxDiffusion |
| Objective: noising, timestep, target, loss | JAX-Titan |
| Train step: gradient, clipping, optimizer, schedule | JAX-Titan |
| Layouts (`dp`, `fsdp`) | JAX-Titan |
| Checkpoints, validation, logs, metrics | JAX-Titan |

MaxDiffusion's own trainers are not used.

The objective is JAX-Titan's because a denoiser alone is not a training
objective: MaxDiffusion keeps the noising, the timestep and the target in its
trainers, not in the model. These three are stated once per family in
`jax_titan/data/diffusion.py` and shared with the
[diffusers adapter](diffusers.md); the loss is the adapter's. Both families
are rectified flows with an unweighted mean squared error: the denoiser sees
`(1 - s) latents + s noise` and predicts `noise - latents`.

| Family | Time `s` | What the denoiser is given | Against MaxDiffusion's trainer |
|---|---|---|---|
| FLUX.1 | Uniform in [0, 1) | `s` | The same noising and target; the trainer draws `s` from the grid of its sampler's steps |
| Wan 2.1 | `3u / (1 + 2u)`, `u` uniform: the 1.3B model's shift of 3 | `1000 s` | The same; the trainer also weights the samples of a batch against each other, which is left out |

## What is supported and what is not

| Feature | Supported | Note |
|---|---|---|
| `--parallel=dp` and `fsdp` | Yes | |
| Pipelines (`--parallel=mpmd`) | No | A model of another repository is called whole; a pipeline needs a model written for JAX-Titan |
| Random weights | Yes | |
| Published weights (`--init_from`) | No | MaxDiffusion's loaders need PyTorch |
| Real data | No | Synthetic batches only: no real images or videos, no text encoders, no VAEs |
| Checkpoints and resuming | Yes | |
| Validation | Yes | On held-out synthetic batches |
| Rematerialization | Yes | |
| `jax_titan.ops` (`--ops`) | FLUX.1 only | Its attention, which no test covers; Wan 2.1 refuses `--ops` |
| MFU | FLUX.1 only | Wan 2.1 has no FLOP count here |
| Writing the trained parameters (`--save_params`) | No | |
| Generation or sampling | No | |

## How it is verified

The unit tests are in `tests/test_maxdiffusion_adapter.py` and
`tests/test_adapter_training.py`.

| What is checked | By | Size | Result |
|---|---|---|---|
| FLUX.1 through the adapter gives the output of MaxDiffusion's `FluxTransformer2DModel` called directly | Unit test | Small | Loss within 1e-5 |
| Wan 2.1 through the adapter gives the output of MaxDiffusion's `WanModel` built and called directly | Unit test | Small | Loss within 1e-5 |
| FSDP shards every parameter of FLUX.1 and of Wan 2.1, and no stack of blocks along its block axis | Unit test | Small | Passes |
| FSDP on 2 GPUs trains as data parallelism does, for FLUX.1 and Wan 2.1 | Unit test | Small | Losses of 4 steps within 1e-4 |
| A small FLUX.1 and a small Wan 2.1 train through the trainer and resume exactly, under dp and under FSDP | Unit test | Small | Equal losses |
| A run resumes from its checkpoint and validates | [Experiment](#checkpoints) | Small | The uninterrupted run's losses exactly, and its validation loss |
| FLUX.1-dev and Wan 2.1 1.3B run, fit and their loss falls | [Experiment](#training-from-random-weights) | Real | The loss falls over 30 steps under `dp` and under `fsdp` |

Not checked:

- **Wan's objective against MaxDiffusion's trainer.** MaxDiffusion's trainers
  do not import in JAX-Titan's environment, so the objective was taken from
  their source and not compared in a run.
- **A reference at real size.** The experiments show that a model runs, fits
  and that its loss falls; the comparisons with MaxDiffusion's modules are at
  a small size.
- **`flux-schnell`**, in a unit test or in an experiment.
- **`--ops` for FLUX.1**, in a unit test or in an experiment.

## Experiments

These are runs of the trainer on the adapter's models, launched by hand on
GPUs. They are not part of the test suite, and the scripts that launched them
are not in the repository. They ran on 8x MI350X on 2026-11-19 and 20.

### Training from random weights

Each model trains from random weights on synthetic batches, one sample per
GPU, in bf16 with fp32 parameters: 30 steps with `--lr=1e-5` and
rematerialization. FLUX.1-dev (11.90B parameters) on 512 x 512 images, 1,024
image tokens and 512 text tokens a sample. Wan 2.1 1.3B (1.42B parameters) on
clips of 17 frames of 480 x 480 pixels, 4,500 video tokens and 512 text tokens
a sample. One run each; tokens/s per GPU is the median over steps 5-29. These
are single runs of 30 steps on random data, enough to see that a model runs,
fits and that its loss falls, not a measurement of throughput.

| Model | Layout | GPUs | Tokens per step | Tokens/s per GPU | Peak HBM per GPU | Loss, step 0 | Steps 25-29 |
|---|---|---|---|---|---|---|---|
| `flux-dev` | dp | 1 | 1,536 | 4,445 | 179 GiB | 3.52 | 3.34 |
| `flux-dev` | fsdp | 8 | 12,288 | 3,366 | 24 GiB | 3.54 | 2.84 |
| `wan2.1-t2v-1.3b` | dp | 1 | 5,012 | 23,468 | 23 GiB | 4.17 | 2.24 |
| `wan2.1-t2v-1.3b` | fsdp | 8 | 40,096 | 19,707 | 5 GiB | 4.21 | 2.11 |

### Checkpoints

A small Wan 2.1, under FSDP on 2 GPUs in fp32, restores the checkpoint of step
3 and gives the uninterrupted run's losses of steps 3 to 5 exactly, and its
validation loss after them.

## JAX-Titan's own FLUX.1 against the adapter

JAX-Titan's native FLUX.1 (`--model=flux-dev` and `flux-schnell`) is written
after diffusers' FLUX.1, from which this adapter's FLUX.1 differs in two places
([Known issues](#known-issues)), so the two are not compared from the same
weights. Trained alike from random weights, they are compared in the example
of [native_comparison.md](native_comparison.md#example-flux1-dev), where this
adapter's FLUX.1, whose layers MaxDiffusion initializes otherwise, ends 0.5
above JAX-Titan's. Wan 2.1 has no native model.

## Known issues

- **FLUX.1's timestep embedding is computed in bf16.** MaxDiffusion computes
  FLUX.1's timestep and guidance embeddings in bf16 whatever the model's
  dtype. Their angles of up to 1,000 lose their low bits there, so the
  embedding comes out differently in eager and in compiled execution, and on
  CPU and GPU, by as much as a sinusoid's whole range. A compiled step is
  consistent with itself, which is what the tests compare.
- **MaxDiffusion's FLUX.1 class defaults to a RoPE base of 1,000**, where the
  published FLUX.1 uses 10,000.
