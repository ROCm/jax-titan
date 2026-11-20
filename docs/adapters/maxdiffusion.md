# MaxDiffusion adapter

`--model=maxdiffusion:<name>` trains a denoiser of MaxDiffusion, pinned at
commit `821afc36`, with JAX-Titan's trainer. The network is MaxDiffusion's and
runs as MaxDiffusion wrote it; the training is JAX-Titan's. `WholeModel`
(`models/adapters/maxdiffusion.py`) calls a denoiser whole, MaxDiffusion's own
`__call__`, under `--parallel=dp` or `fsdp`, with a few lines per family. This
version has two: FLUX.1 (`flux-dev`, `flux-schnell`) and Wan 2.1 text to video
(`wan2.1-t2v-1.3b`).

A denoiser alone is not a training objective. MaxDiffusion keeps the noising,
the timestep and the target in its trainers, not in the model, so the
objective here is JAX-Titan's, stated once per family in
`jax_titan/data/diffusion.py` and shared with the
[diffusers adapter](diffusers.md). Both families are rectified flows with an
unweighted mean squared error: the denoiser sees `(1 - s) latents + s noise`
and predicts `noise - latents`.

| Family | Time `s` | What the denoiser is given | Against MaxDiffusion's trainer |
|---|---|---|---|
| FLUX.1 | Uniform in [0, 1) | `s` | The same noising and target; the trainer draws `s` from the grid of its sampler's steps |
| Wan 2.1 | `3u / (1 + 2u)`, `u` uniform: the 1.3B model's shift of 3 | `1000 s` | The same; the trainer also weights the samples of a batch against each other, which is left out |

## What comes from MaxDiffusion and what from JAX-Titan

| Part | Comes from |
|---|---|
| Model: structure, layers, config, initialization | MaxDiffusion |
| Objective: noising, timestep, target, loss | JAX-Titan |
| Train step: gradient, clipping, optimizer, schedule | JAX-Titan |
| Layouts (`dp`, `fsdp`) | JAX-Titan |
| Checkpoints, validation, logs, metrics | JAX-Titan |

MaxDiffusion's own trainers are not used.

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

## What is supported

- FLUX.1-dev and FLUX.1-schnell under `dp` and `fsdp`.
- Wan 2.1 text to video, 1.3B, under `dp` and `fsdp`.
- Random weights on synthetic batches, rematerialization and checkpoints.

## What is not supported

- **MaxDiffusion's other families** (Stable Diffusion, SDXL, Wan 2.2 and its
  image-to-video and VACE models, LTX-Video, Z-Image). Each needs its lines in
  the adapter and its objective.
- **Published weights**: MaxDiffusion's loaders need PyTorch. MaxDiffusion's
  FLUX.1 class also defaults to a RoPE base of 1,000, where the published
  FLUX.1 uses 10,000.
- **Real images and videos**, the text encoders and the VAEs, and sampling.
- **Pipelines** (`--parallel=mpmd`). A model of another repository is called
  whole; a pipeline needs a model written for JAX-Titan.
- **`--ops` for Wan 2.1.**
- **MFU for Wan**: it has no FLOP count here.
- **A FLUX.1 timestep embedding in full precision.** MaxDiffusion computes
  FLUX.1's timestep and guidance embeddings in bf16 whatever the model's
  dtype. Their angles of up to 1,000 lose their low bits there, so the
  embedding comes out differently in eager and in compiled execution, and on
  CPU and GPU, by as much as a sinusoid's whole range. A compiled step is
  consistent with itself, which is what the tests compare.
- **Wan's objective against MaxDiffusion's trainer.** MaxDiffusion's trainers
  do not import in JAX-Titan's environment, so the objective was taken from
  their source and not compared in a run.

## How it is verified

### Tests

`tests/test_maxdiffusion_adapter.py`, on GPUs, at a small size in fp32:

- **FLUX.1 through the adapter gives the output of MaxDiffusion's
  `FluxTransformer2DModel`** called directly.
- **FSDP shards every parameter of FLUX.1**, and no stack of blocks along its
  block axis.
- **Wan 2.1 through the adapter gives the output of MaxDiffusion's `WanModel`**
  built and called directly.
- **FSDP trains as data parallelism does**, on 2 GPUs, for FLUX.1 and Wan 2.1.

`tests/test_adapter_training.py`, on CPU: a small FLUX.1 trains through the
trainer and resumes exactly.

### Runs (2026-11-19 to 20, 8x MI350X)

From random weights on synthetic batches, one sample per GPU, in bf16 with
fp32 parameters: 30 steps with `--lr=1e-5` and rematerialization. FLUX.1-dev
(11.90B parameters) on 512 x 512 images, 1,024 image tokens and 512 text
tokens a sample. Wan 2.1 1.3B (1.42B parameters) on clips of 17 frames of 480 x 480
pixels, 4,500 video tokens and 512 text tokens a sample. One run each;
tokens/s per GPU is the median over steps 5-29. These are single runs of 30
steps on random data, enough to see that a model runs, fits and that its loss
falls, not a measurement of throughput.

| Model | Layout | GPUs | Tokens per step | Tokens/s per GPU | Peak HBM per GPU | Loss, step 0 | Steps 25-29 |
|---|---|---|---|---|---|---|---|
| `flux-dev` | dp | 1 | 1,536 | 4,445 | 179 GiB | 3.52 | 3.34 |
| `flux-dev` | fsdp | 8 | 12,288 | 3,366 | 24 GiB | 3.54 | 2.84 |
| `wan2.1-t2v-1.3b` | dp | 1 | 5,012 | 23,468 | 23 GiB | 4.17 | 2.24 |
| `wan2.1-t2v-1.3b` | fsdp | 8 | 40,096 | 19,707 | 5 GiB | 4.21 | 2.11 |

Checkpoints and validation: a small Wan 2.1, under FSDP on 2 GPUs in fp32,
restores the checkpoint of step 3 and gives the uninterrupted run's losses of
steps 3 to 5 exactly, and its validation loss after them.

The scripts that launched these runs are not in the repository.
