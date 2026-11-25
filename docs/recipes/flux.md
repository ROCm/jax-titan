# Recipe: FLUX.1 with FSDP on 8x MI350X

FLUX.1 is a text-to-image model of Black Forest Labs. The part trained here is
its denoiser, a rectified-flow transformer of 11.90B parameters at
FLUX.1-dev's size. JAX-Titan has it as its own model (`models/flux.py`),
written after diffusers' `FluxTransformer2DModel` 0.43.0, and trains it from
random weights on synthetic batches. It reads no published weights and draws
no pictures yet ([Known limits](#known-limits)).

## Running the recipe

### Training on synthetic batches

```bash
export JAX_TITAN_LAUNCH_TIME=$(date +%s.%N)
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" \
  python -m jax_titan.trainer.train --model=flux-dev --parallel=fsdp \
  --microbatches=1 --microbatch_size=8 --resolution=512 --lr=1e-5 --steps=300 \
  --eval_every=50 --run_dir=runs/flux-dev
python -m jax_titan.observability.summarize runs/flux-dev
```

- **Model:** `flux-dev` has the published FLUX.1-dev's sizes and guidance
  embedding, 11.90B parameters, and `flux-schnell` FLUX.1-schnell's, 11.89B.
  `--model_overrides` takes config fields, and `--dtype` sets the compute dtype.
- **Layout:** FSDP over the visible GPUs in one `jax.jit` program, no
  pipeline. `--parallel=dp` replicates the state instead.
- **Batch:** 8 samples of 512 x 512 pixels per step, each 1,024 image tokens
  and 512 text tokens: 12,288 tokens.
- **Precision:** fp32 parameters and AdamW state, bf16 compute.
- **Optimizer:** AdamW at 1e-5 after 10 warmup steps, then a cosine decay to
  1e-6 at the last step; weight decay 0.1 on all but the norm weights;
  gradient norm clipped at 1.0.
- **Data:** random latents, noise, text features and pooled features, no
  dataset: the synthetic batches the two FLUX.1 adapters take
  (`jax_titan.data.diffusion`). The model sees `(1 - s) latents + s noise`
  with `s` uniform in [0, 1) and predicts `noise - latents`.
- **Weights:** random, drawn as PyTorch draws those of diffusers' model:
  weights and biases uniform within 1/sqrt(fan-in) of zero, norm weights one.

## Correctness

### Loss convergence on synthetic batches

A run is checked by its loss curve. On these batches the loss has a floor: a
sample noised at `s` cannot score below `1 / ((1 - s)^2 + s^2)`, which
averages 1.54 to 1.59 over this recipe's windows of 50 training steps and is
1.6082 on its 4 validation batches. The floor needs only each noised latent
and its `s`: the text, the pooled features and the positions carry nothing
the target depends on. So the curve shows that the model trains, and not
whether its attention, rotary embedding or text stream are right; the
comparison with diffusers' model from the same weights, below, covers those.

- **The loss converges to the floor.** This recipe's run starts at 2.34, its
  mean over 50 steps falls from 1.9086 to 1.5812, and after 300 steps its
  validation loss is 1.6278, 0.02 above the floor. Three runs of the command
  give these numbers within 2e-4.
- **It trains like diffusers' FLUX.1.**
  [adapters/comparison.md](../adapters/comparison.md#flux1-dev)
  sets the same run beside the two adapters' FLUX.1: from step 150 on its
  means are within 2e-4 of diffusers', and MaxDiffusion's, which initializes
  its layers otherwise, ends 0.50 above.

The model against its reference, with the same weights on both sides:

| Reference | Where | Largest difference |
|---|---|---|
| diffusers' model, with and without the guidance embedding | one GPU, 2 double and 2 single blocks of width 32, fp32 | prediction and loss within the test's 1e-5, every gradient within its 1e-3 |
| diffusers' model, run by PyTorch 2.10.0 on CPU | one MI350X, full size, fp32, one sample of 512 x 512 pixels | prediction 7.7e-6, of values up to 2.62, and 1.4e-6 on average; loss 1.6e-8 of its value, 2.3219905 |

At full size the weights are those PyTorch initializes under
`torch.manual_seed(0)`, and the sample is drawn as the trainer draws a batch
of one sample (seed 0, step 0): 1,024 image tokens and 512 text tokens. The
gradients are compared in the test alone.

Training under FSDP on 8 GPUs against data parallelism on one, at full size
in fp32, from the same random weights and on the same 8 samples of 512 x 512
pixels a step, with `--lr=1e-5`: over 10 steps the losses differ by at most
1.2e-7 of their value and the gradient norms by 1.1e-7.

`tests/test_flux.py` also checks, at a small size, FSDP against data
parallelism for both configurations, a resumed run against the uninterrupted one and
the initialization, and at full size that FLUX.1-dev has the 11,901,408,320
parameters of diffusers' model and that FSDP shards every parameter, never
along the block axis.

- **JAX-Titan's FLUX.1 is diffusers' model in fp32.** In bf16 the two embed
  the timestep differently, beyond rounding: JAX-Titan's model scales the
  timestep and the guidance by 1000 in fp32, where diffusers' first rounds
  them to its compute dtype. They were not compared in bf16.

These runs are on MI350X on 2026-11-21 and 22; neither their scripts nor
their records are in the repository.

## Performance

With FSDP on 8 MI350X the recipe's command runs at 3,289 tokens/s per GPU,
the median of three runs that differ by 0.5%, and its GPUs peak at 23.0 GiB,
the first at 56.1. In fp32 the peaks are 25.5 GiB, 56.1 on the first, and
186.8 GiB under data parallelism on one GPU.
[adapters/comparison.md](../adapters/comparison.md#flux1-dev)
has the speed of the two adapters' FLUX.1 beside it. Other GPU counts and
`flux-schnell` were not measured.

## Known limits

- Only the denoiser trains, from random weights on synthetic batches. No
  published weights are read, and there is no autoencoder, no text encoder
  and no sampling, so the recipe has no pictures and no fine-tuning on
  images yet.
- No pipeline: the model trains with FSDP or data parallelism in one
  program, on one node, without tensor parallelism.
- The random weights are drawn on the first GPU and then sharded, as
  Ideogram 4's and Command A+'s are, which is that GPU's 56.1 GiB under FSDP.
- Not checked: FLUX.1-schnell's prediction and training at full size, and
  providers of `jax_titan.ops` other than `native`.
