# diffusers adapter

`--model=diffusers:<name>` trains a denoiser of Hugging Face diffusers 0.43.0
with JAX-Titan's trainer, under `--parallel=dp` or `fsdp`. The network is
diffusers' and is written in PyTorch; the training is JAX-Titan's. The
network does not run as PyTorch: torchax 0.0.13 executes the denoiser's
`forward` on JAX arrays while JAX traces it (`models/adapters/diffusers.py`;
[transformers.md](transformers.md#models) says what that means for
correctness).

## Models

The adapter takes the denoiser classes it compared with PyTorch, each with a
few lines that name its inputs. This version registers one:

| `--model=` | Model | Parameters | Run on GPUs at real size |
|---|---|---|---|
| `diffusers:FluxTransformer2DModel` | FLUX.1's denoiser | 11.90B at FLUX.1-dev's size | Yes |

`<name>` is a denoiser class of diffusers, built with the library's default
sizes, or a directory with a denoiser's `config.json`, or a pipeline's
directory, whose `transformer` is taken. The tests and runs here embed the
guidance scale, as FLUX.1-dev does. Another denoiser class is refused.

## Running it

Install the `diffusers` extra; PyTorch's CPU build is enough:

```bash
pip install -e ".[diffusers]" --extra-index-url https://download.pytorch.org/whl/cpu
```

`FluxTransformer2DModel`'s defaults are FLUX.1's sizes; FLUX.1-dev also embeds
the guidance scale:

```bash
python -m jax_titan.trainer.train --model=diffusers:FluxTransformer2DModel \
  --model_overrides='{"guidance_embeds": true}' --parallel=fsdp \
  --microbatches=1 --microbatch_size=8 --resolution=512 --steps=30 --lr=1e-5
```

`--resolution` is the image side in pixels, of which the denoiser sees
`(resolution / 16)^2` latent tokens, and `--text_tokens` the text tokens of a
sample (512 with the guidance embedding, 256 without). `--model_overrides`
takes config fields as JSON, and `"dtype"` for the compute dtype.

## What comes from diffusers and what from JAX-Titan

| Part | Comes from |
|---|---|
| Model: structure, layers, config, initialization | diffusers |
| Objective: noising, timestep, target, loss | JAX-Titan |
| Train step: gradient, clipping, optimizer, schedule | JAX-Titan |
| Layouts (`dp`, `fsdp`) | JAX-Titan |
| Checkpoints, validation, logs, metrics | JAX-Titan |

diffusers' example training scripts are not used.

The objective is JAX-Titan's because a denoiser alone is not a training
objective: diffusers keeps the noising, the timestep and the target in example
scripts, not in the model. These three are stated once per family in
`jax_titan/data/diffusion.py` and shared with the
[MaxDiffusion adapter](maxdiffusion.md); the loss is the adapter's. FLUX.1 is
trained as a rectified flow, on `(1 - s) latents + s noise` with `s` uniform
in [0, 1), to predict `noise - latents`, with an unweighted mean squared
error.

## What is supported and what is not

| Feature | Supported | Note |
|---|---|---|
| `--parallel=dp` and `fsdp` | Yes | |
| Pipelines (`--parallel=mpmd`) | No | A model of another repository is called whole; a pipeline needs a model written for JAX-Titan |
| Random weights | Yes | |
| Published weights (`--init_from`) | No | |
| Real data | No | Synthetic batches only, random latents and text features: no real images, no text encoders, no VAE |
| Checkpoints and resuming | Yes | |
| Validation | Yes | On held-out synthetic batches |
| Rematerialization | Yes | `--remat`, the default |
| `jax_titan.ops` (`--ops`) | No | |
| MFU | No | No FLOP count is kept for a traced model |
| Writing the trained parameters (`--save_params`) | No | |
| Generation or sampling | No | No picture is generated |

EMA, LoRA and ControlNet are not supported either.

## How it is verified

The unit tests are in `tests/test_diffusers_adapter.py`.

| What is checked | By | Size | Result |
|---|---|---|---|
| The adapter's loss and gradients are PyTorch's, over two traces: FLUX.1 with the guidance embedding on a batch of `data/diffusion.py` | Unit test | Small | Loss within 1e-5, gradients within 1e-3 |
| Weight decay skips the norms of the queries and keys and nothing else | Unit test | Small | Passes |
| FSDP on 2 GPUs trains as data parallelism does | Unit test | Small | Losses of 4 steps within 1e-4 |
| Another denoiser class is refused | Unit test | Small | Passes |
| A run resumes from its checkpoint and validates | [Experiment](#checkpoints) | Small | The uninterrupted run's losses exactly, and its validation loss |
| The model runs, fits and its loss falls at FLUX.1-dev's size | [Experiment](#training-from-random-weights) | Real | The loss falls over 30 steps under `dp` and under `fsdp` |

Not checked:

- **A reference at real size.** The adapter loads no published weights: the
  experiments show that the model runs, fits and that its loss falls, and the
  comparison with PyTorch is at a small size.

## Experiments

These are runs of the trainer on the adapter's models, launched by hand on
GPUs. They are not part of the test suite, and the scripts that launched them
are not in the repository. They ran on 8x MI350X on 2026-11-19.

### Training from random weights

`FluxTransformer2DModel` with the guidance embedding, FLUX.1-dev's size of
11.90B parameters, trains from random weights on synthetic batches of
512 x 512 images, 1,024 image tokens and 512 text tokens a sample, one sample
per GPU, in bf16 with fp32 parameters: 30 steps with `--lr=1e-5` and
rematerialization. One run each; tokens/s per GPU is the median over steps
5-29. These are single runs of 30 steps on random data, enough to see that the
model runs, fits and that its loss falls, not a measurement of throughput.

| Model | Layout | GPUs | Tokens per step | Tokens/s per GPU | Peak HBM per GPU | Loss, step 0 | Steps 25-29 |
|---|---|---|---|---|---|---|---|
| `FluxTransformer2DModel` | dp | 1 | 1,536 | 5,397 | 158 GiB | 2.32 | 2.05 |
| `FluxTransformer2DModel` | fsdp | 8 | 12,288 | 3,207 | 38 GiB | 2.33 | 1.88 |

### Checkpoints

A small FLUX.1, under FSDP on 2 GPUs in fp32, restores the checkpoint of step
3 and gives the uninterrupted run's losses of steps 3 to 5 exactly, and its
validation loss after them.

## JAX-Titan's own FLUX.1 against the adapter

JAX-Titan's native FLUX.1 (`--model=flux-dev` and `flux-schnell`) is written
after this adapter's model, `FluxTransformer2DModel`: from the same weights
the two give the same prediction, loss and gradients in fp32
([recipes/flux.md](../recipes/flux.md#correctness)). Trained alike
from random weights, with the same arguments `$ARGS` for both, they are
compared as in the example of
[native_comparison.md](native_comparison.md#flux1-dev):

```bash
python -m jax_titan.trainer.train --model=flux-dev $ARGS --run_dir=runs/native
python -m jax_titan.trainer.train --model=diffusers:FluxTransformer2DModel \
  --model_overrides='{"guidance_embeds": true}' $ARGS --run_dir=runs/adapter
python -m jax_titan.observability.compare runs/native runs/adapter
```

## Known issues

- **A start from the compilation cache was not checked for the denoiser**:
  traced language models start slowly from it
  ([transformers.md](transformers.md#known-issues)).
