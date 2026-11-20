# MaxDiffusion adapter

`--model=maxdiffusion:<name>` trains a denoiser of MaxDiffusion, pinned at
commit `821afc36`, with JAX-Titan's trainer. The network is MaxDiffusion's and
runs as MaxDiffusion wrote it; the training is JAX-Titan's. `WholeModel`
(`models/adapters/maxdiffusion.py`) calls a denoiser whole, MaxDiffusion's own
`__call__`, as one stage under `--parallel=dp` or `fsdp`, with a few lines per
family. This version has two: FLUX.1 (`flux-dev`, `flux-schnell`) and Wan 2.1
text to video (`wan2.1-t2v-1.3b`). FLUX.1 can also be cut into pipeline stages
([pipelines.md](../pipelines.md)); it keeps those stages under
`--parallel=mpmd` and `dp`, and runs whole under `fsdp`.

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

| Part | From MaxDiffusion | From JAX-Titan |
|---|---|---|
| Denoiser structure and layers | `FluxTransformer2DModel`, `WanModel`, called as they are | |
| Sizes | The class's defaults | Wan 2.1 1.3B's sizes; `--model_overrides` |
| Parameter initialization | Its initializers | Placed on the trainer's shardings |
| Precision | Its `dtype` and `weights_dtype` | Their values: bf16 compute, fp32 weights |
| Block scan and rematerialization | Its own (`remat_policy`) | |
| Attention | Its dot-product attention | For FLUX.1, `--ops` routes it to `jax_titan.ops` |
| Objective: noising, timestep, target, loss | | `jax_titan/data/diffusion.py` |
| Batches | | Synthetic: standard normal latents and text features |
| Gradient, clipping, optimizer, schedule | | The train step |
| Layouts (`dp`, `fsdp`) | Its constraints on activations | The mesh; a parameter is sharded along its first dimension that the data axis divides, never along the block axis of a stack |
| Checkpoints, logs, metrics | | JAX-Titan's; a FLOP count for FLUX.1 only |
| Pipeline stages (FLUX.1) | The blocks inside a stage | The cut into stages and the MPMD schedule |

MaxDiffusion's own trainers take no part.

## Running it

Install MaxDiffusion at the pinned commit with JAX-Titan's patches, and the
`maxdiffusion` extra ([getting_started.md](../getting_started.md)). FLUX.1
whole, on every GPU of the node:

```bash
python -m jax_titan.trainer.train --model=maxdiffusion:flux-dev --parallel=fsdp --num_stages=1 \
  --microbatches=1 --microbatch_size=8 --resolution=512 --steps=30 --lr=1e-5
```

Wan 2.1 1.3B on clips of 17 frames of 480 x 480 pixels:

```bash
python -m jax_titan.trainer.train --model=maxdiffusion:wan2.1-t2v-1.3b --parallel=fsdp --num_stages=1 \
  --microbatches=1 --microbatch_size=8 --resolution=480 --frames=17 --steps=30 --lr=1e-5
```

`--resolution` is the side of an image or a frame in pixels and `--frames` the
frames of a clip: Wan sees `(frames - 1) / 4 + 1` latent frames of
`(resolution / 16)^2` tokens each. `--text_tokens` defaults to 512, and to 256
for `flux-schnell`.

## What is supported

- FLUX.1-dev and FLUX.1-schnell as one stage under `dp` and `fsdp`, and as a
  pipeline, as before.
- Wan 2.1 text to video, 1.3B, as one stage under `dp` and `fsdp`.
- Random weights on synthetic batches, rematerialization and checkpoints.

## What is not supported

- **MaxDiffusion's other families** (Stable Diffusion, SDXL, Wan 2.2 and its
  image-to-video and VACE models, LTX-Video, Z-Image). Each needs its lines in
  the adapter and its objective.
- **Published weights**: MaxDiffusion's loaders need PyTorch. MaxDiffusion's
  FLUX.1 class also defaults to a RoPE base of 1,000, where the published
  FLUX.1 uses 10,000.
- **Real images and videos**, the text encoders and the VAEs, and sampling.
- **Wan 2.1 as a pipeline**, and `--ops` for Wan.
- **MFU for Wan**: it has no FLOP count here.
- **Wan's objective against MaxDiffusion's trainer.** MaxDiffusion's trainers
  do not import in JAX-Titan's environment, so the objective was taken from
  their source and not compared in a run.

## How it is verified

### Tests

`tests/test_maxdiffusion_adapter.py`, on GPUs, at a small size in fp32:

- **FLUX.1 called whole gives the loss and gradients of its one stage**, which
  reimplements the call, within 1e-5 and 1e-3, from the same parameters.
- **FSDP shards every parameter of FLUX.1**, and no stack of blocks along its
  block axis.
- **Wan 2.1 through the adapter gives the output of MaxDiffusion's `WanModel`**
  built and called directly.
- **FSDP trains as data parallelism does**, on 2 GPUs, for FLUX.1 and Wan 2.1.

The pipeline stages of FLUX.1 have their own tests, on CPU
(`test_maxdiffusion_flux_parity`).

### Runs (2026-11-19 to 20, 8x MI350X)

From random weights on synthetic batches, one sample per GPU, in bf16 with
fp32 parameters: 30 steps with `--lr=1e-5` and rematerialization. FLUX.1-dev
(11.90B parameters) on 512 x 512 images, 1,024 image tokens and 512 text
tokens a sample: its one stage under `dp`, the transformer called whole under
`fsdp`. Wan 2.1 1.3B (1.42B parameters) on clips of 17 frames of 480 x 480
pixels, 4,500 video tokens and 512 text tokens a sample. One run each;
tokens/s per GPU is the median over steps 5-29. These are single runs of 30
steps on random data, enough to see that a model runs, fits and that its loss
falls, not a measurement of throughput.

| Model | Layout | GPUs | Tokens per step | Tokens/s per GPU | Peak HBM per GPU | Loss, step 0 | Steps 25-29 |
|---|---|---|---|---|---|---|---|
| `flux-dev` | dp | 1 | 1,536 | 4,472 | 179 GiB | 3.52 | 3.35 |
| `flux-dev` | fsdp | 8 | 12,288 | 3,366 | 24 GiB | 3.54 | 2.84 |
| `wan2.1-t2v-1.3b` | dp | 1 | 5,012 | 23,468 | 23 GiB | 4.17 | 2.24 |
| `wan2.1-t2v-1.3b` | fsdp | 8 | 40,096 | 19,707 | 5 GiB | 4.21 | 2.11 |

Checkpoints and validation: a small Wan 2.1, under FSDP on 2 GPUs in fp32,
restores the checkpoint of step 3 and gives the uninterrupted run's losses of
steps 3 to 5 exactly, and its validation loss after them.

The scripts that launched these runs are not in the repository.
