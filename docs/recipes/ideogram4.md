# Recipe: Ideogram 4 with FSDP on 4x MI350X

Ideogram 4 is a text-to-image model. The part trained here is its transformer
of 9.28B parameters, which predicts the flow-matching velocity from noised
image latents and text features. JAX-Titan trains it on synthetic batches in
two modes.

## Running the recipe

JAX-Titan's own model:

```bash
export JAX_TITAN_LAUNCH_TIME=$(date +%s.%N)
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=ideogram4 --parallel=fsdp --num_stages=1 --microbatches=1 --microbatch_size=8 \
  --resolution=1024 --lr=1e-5 --steps=1000 --eval_every=50 --run_dir=runs/ideogram4
python -m jax_titan.observability.summarize runs/ideogram4
```

The adapter, which runs the model of MaxDiffusion's Ideogram branch and needs
that branch installed ([getting_started.md](../getting_started.md)):

```bash
export JAX_TITAN_LAUNCH_TIME=$(date +%s.%N)
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=maxdiffusion:ideogram4 --parallel=fsdp --num_stages=1 --microbatches=1 --microbatch_size=8 \
  --resolution=1024 --lr=1e-5 --steps=1000 --eval_every=50 --run_dir=runs/maxdiffusion-ideogram4
python -m jax_titan.observability.summarize runs/maxdiffusion-ideogram4
```

Both commands run the same configuration:

- **Layout:** FSDP over the 4 GPUs in one `jax.jit` program, no pipeline.
- **Batch:** 8 images of 1024x1024 per step, 34,816 tokens.
- **Precision:** fp32 parameters and AdamW state, bf16 compute.
- **Optimizer:** AdamW at 1e-5 after 10 warmup steps, gradient norm clipped
  at 1.0.
- **Data:** random latents, noise and text features, no dataset.

## Correctness

A run is checked by its loss curve. On random batches the loss has two known
levels: a model that predicts zero scores 2, and the best possible prediction
scores `1 / (t^2 + (1 - t)^2)` for a sample of timestep `t`. That floor
averages 1.57 over this recipe's training batches and is 1.5981 on its 4
validation batches. A run is right when its loss falls below 2 and settles at
the floor.

<p align="center">
  <img src="../images/ideogram4_loss.png" width="880" alt="Training loss over 1,000 steps of the adapter and of JAX-Titan's own Ideogram 4, and the difference of the two step by step">
</p>

- **The loss converges to the floor.** This recipe's run starts at 3.03, is
  below 2 from step 145 on, and after 1,000 steps its validation loss is
  1.6051, 0.0070 above the floor.
- **JAX-Titan's own model and the adapter draw one line.** From the adapter's
  weights, on its batches and with its timesteps, the own model on 4 GPUs
  follows the adapter on 8: the losses of a step differ by at most 0.0031
  from step 200 on, where the loss varies by 0.10 from batch to batch, and
  the validation losses end at 1.5964 and 1.5963.
- **The branch's own trainer is not in the figure,** because its run is
  another experiment: it starts from the pretrained weights, which JAX-Titan
  does not load yet, and at the commit the adapter pins its step returns the
  random key it was given, so every step draws the same noise and the same
  timesteps. One step of it matches the step of these runs in a test with
  small sizes.

The comparisons with diffusers' model and with the branch's model, in tests
with small sizes and at full size on one GPU, are in
[pipelines.md](../pipelines.md#ideogram-4-2026-11-17-4x-mi350x). The runs are
of 2026-11-17, on jax 0.12.1 with JAX-Titan's jaxlib and Shardy patches (JAX
`6b39d495e5`, XLA `d23c30982a`, Shardy `b6b99997`) in the TheRock ROCm 10.1
container, and their records are in
[results/2026-11-17](../results/2026-11-17).

## Performance

TODO: measure throughput and memory on an idle node, on 4 and on 8 GPUs. What
the runs above saw on a node shared with another job is in
[pipelines.md](../pipelines.md#ideogram-4-2026-11-17-4x-mi350x).

## Known limits

- Synthetic batches only: no dataset, no text encoder and no VAE.
- No pretrained weights. `--init_from` reads parameters in the model's own
  layout (`jax_titan.trainer.checkpoint.save_params`); a converter from the
  released checkpoint is not written.
- The model trains through the layouts of `jax_titan.trainer.adapters`:
  FSDP or data parallelism without a pipeline, or a pipeline with data
  parallelism inside its stages and gradients reduced per microbatch. It has
  no tensor parallelism and no FSDP inside pipeline stages.
- Attention runs on native JAX only: the model always passes a mask, which
  the TE and JAX-AITER attention decline.
- One node.
