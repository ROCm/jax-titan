# diffusers adapter

`--model=diffusers:<name>` trains a denoiser of Hugging Face diffusers 0.43.0
with JAX-Titan's trainer, as one stage under `--parallel=dp` or `fsdp`. The
network is diffusers' and is written in PyTorch. It does not run as PyTorch:
torchax 0.0.13 executes the denoiser's `forward` on JAX arrays while JAX
traces it ([transformers.md](transformers.md) says what that means for
correctness). The adapter takes the denoiser classes it compared with
PyTorch, each with a few lines that name its inputs
(`models/adapters/diffusers.py`). This version registers one, FLUX.1's
`FluxTransformer2DModel`.

A denoiser alone is not a training objective. diffusers keeps the noising, the
timestep and the target in example scripts, not in the model, so the objective
here is JAX-Titan's, stated once per family in `jax_titan/data/diffusion.py`
and shared with the [MaxDiffusion adapter](maxdiffusion.md): FLUX.1 is trained
as a rectified flow, on `(1 - s) latents + s noise` with `s` uniform in
[0, 1), to predict `noise - latents`, with an unweighted mean squared error.

## What comes from diffusers and what from JAX-Titan

| Part | From diffusers | From JAX-Titan |
|---|---|---|
| Denoiser structure and modules | The model class, traced through torchax | |
| Config | The class's defaults, or a `config.json` | `--model_overrides` |
| Parameter initialization | The library's, under `--seed` | Placed on the trainer's shardings |
| Precision | | fp32 parameters, cast to bf16 inside the step (`"dtype"` in `--model_overrides`) |
| Rematerialization | Its gradient-checkpointing switch, which says where the blocks are | `jax.checkpoint` as the function that switch calls |
| Objective: noising, timestep, target, loss | | `jax_titan/data/diffusion.py` |
| Batches | | Synthetic: standard normal latents and text features |
| Gradient, clipping, optimizer, schedule | | The train step; AdamW without weight decay on the parameters of normalization modules |
| Layouts (`dp`, `fsdp`) | | The mesh; a parameter is sharded along its first dimension that the data axis divides |
| Checkpoints, logs, metrics | | JAX-Titan's; no FLOP count, so no MFU |

## Running it

Install the `diffusers` extra; PyTorch's CPU build is enough:

```bash
pip install -e ".[diffusers]" --extra-index-url https://download.pytorch.org/whl/cpu
```

`<name>` is a denoiser class of diffusers, built with the library's default
sizes, or a directory with a denoiser's `config.json`, or a pipeline's
directory, whose `transformer` is taken. `FluxTransformer2DModel`'s defaults
are FLUX.1's sizes; FLUX.1-dev also embeds the guidance scale:

```bash
python -m jax_titan.trainer.train --model=diffusers:FluxTransformer2DModel \
  --model_overrides='{"guidance_embeds": true}' --parallel=fsdp --num_stages=1 \
  --microbatches=1 --microbatch_size=8 --resolution=512 --steps=30 --lr=1e-5
```

`--resolution` is the image side in pixels, of which the denoiser sees
`(resolution / 16)^2` latent tokens, and `--text_tokens` the text tokens of a
sample (512 with the guidance embedding, 256 without).

## What is supported

- `FluxTransformer2DModel`, FLUX.1's denoiser, from random weights on
  synthetic batches. The tests and runs here embed the guidance scale, as
  FLUX.1-dev does.
- `--parallel=dp` and `fsdp`, rematerialization (`--remat`, the default) and
  checkpoints.

## What is not supported

- **Other denoiser classes** are refused until they are compared with
  PyTorch and given their family's batches.
- **Published weights** (`--init_from`), real images, the text encoders and
  the VAE: a step trains the denoiser on random latents and text features.
- **Sampling**: no picture is generated.
- **Pipelines** (`--parallel=mpmd`), `jax_titan.ops` (`--ops`), EMA, LoRA and
  ControlNet.
- **MFU.** No FLOP count is kept for a traced model.
- **A fast start from the compilation cache** was not checked for the
  denoiser: traced language models start slowly from it
  ([transformers.md](transformers.md#what-is-not-supported)).

## How it is verified

### Tests

`tests/test_diffusers_adapter.py`, on GPUs, at a small size in fp32:

- **The adapter's loss and gradients are PyTorch's**, over two traces: FLUX.1
  with the guidance embedding on a batch of `data/diffusion.py`, the loss
  within 1e-5 and the gradients within 1e-3.
- **Weight decay** skips the norms of the queries and keys and nothing else.
- **FSDP trains as data parallelism does**, on 2 GPUs.
- **Another denoiser class is refused.**

### Runs (2026-11-19, 8x MI350X)

`FluxTransformer2DModel` with the guidance embedding, FLUX.1-dev's size of
11.90B parameters, from random weights on synthetic batches of 512 x 512
images, 1,024 image tokens and 512 text tokens a sample, one sample per GPU,
in bf16 with fp32 parameters: 30 steps with `--lr=1e-5` and
rematerialization. One run each; tokens/s per GPU is the median over steps
5-29. These are single runs of 30 steps on random data, enough to see that
the model runs, fits and that its loss falls, not a measurement of throughput.

| Model | Layout | GPUs | Tokens per step | Tokens/s per GPU | Peak HBM per GPU | Loss, step 0 | Steps 25-29 |
|---|---|---|---|---|---|---|---|
| `FluxTransformer2DModel` | dp | 1 | 1,536 | 5,397 | 158 GiB | 2.32 | 2.05 |
| `FluxTransformer2DModel` | fsdp | 8 | 12,288 | 3,207 | 38 GiB | 2.33 | 1.88 |

Checkpoints and validation: a small FLUX.1, under FSDP on 2 GPUs in fp32,
restores the checkpoint of step 3 and gives the uninterrupted run's losses of
steps 3 to 5 exactly, and its validation loss after them.

The scripts that launched these runs are not in the repository.
