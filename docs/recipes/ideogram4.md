# Recipe: Ideogram 4 with FSDP on 4x MI350X

JAX-Titan's own Ideogram 4 (`jax_titan.models.ideogram`, 9.28B parameters), the
flow-matching transformer of a text-to-image model, trained on synthetic
batches with FSDP over all GPUs and no pipeline. Nothing in this recipe comes
from MaxDiffusion.

```bash
export JAX_TITAN_LAUNCH_TIME=$(date +%s.%N)
XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" python -m jax_titan.trainer.train \
  --model=ideogram4 --parallel=fsdp --num_stages=1 --microbatches=1 --microbatch_size=8 \
  --resolution=1024 --lr=1e-5 --steps=1000 --eval_every=50 --run_dir=runs/ideogram4
python -m jax_titan.observability.summarize runs/ideogram4
```

**What runs.** One `jax.jit` program over the 4 GPUs. A step has 8 images of
1024x1024, 2 per GPU; an image is 256 text tokens followed by 4,096 latent
tokens (2x2 patches of a 128x128 latent with 32 channels), so a step has
34,816 tokens. The 34 blocks of width 4,608 with 18 heads run in a `lax.scan`
with full rematerialization. Parameters and AdamW state are fp32 and sharded
over the GPUs, compute is bf16, and every op is native JAX. AdamW (1e-5 after
10 warmup steps, cosine decay to 10%), one gradient norm clipped at 1.0.
Validation runs on 4 fixed held-out batches.

**The model.** The architecture follows diffusers'
`Ideogram4Transformer2DModel`, and the training step (noising, timesteps,
loss) its example `train_dreambooth_lora_ideogram4.py`. The text encoder and
the VAE are not part of the model, and no pretrained weights are loaded.

**The data.** Latents, noise and text features are random. A batch is noised
as in training: the input is `(1 - t) noise + t latents` for a timestep `t`
between 0 (noise) and 1 (data), and the target is `latents - noise`. Latents
and noise are independent, so the best prediction is a multiple of the input
and the loss has a floor, `1 / (t^2 + (1 - t)^2)` per sample, between 1 and 2.
A model that predicts zero scores 2. A run is right when its loss falls below
2 and approaches the floor of its batches; the text features carry nothing to
learn here.

**Stack.** JAX-Titan `ce7c5da`; jax 0.12.1 with JAX-Titan's jaxlib and Shardy
patches (JAX `6b39d495e5`, XLA `d23c30982a`, Shardy `b6b99997`); TheRock ROCm
10.1 container.

## Expected results (2026-11-17)

Another job held a GPU of the node during this run, so its step time is what
the run saw and not a throughput of record.

| Metric | Value |
|---|---|
| Loss | 3.03 at step 0; first below 2 at step 115 and below it from step 145 on; 0.0078 above the floor of its batches on average over the last 100 steps |
| Validation loss | 1.6051 after 1,000 steps, 0.0070 above the floor of its batches (1.5981) |
| Step time | 1.420 s median over the run, 24.5k tokens/s, 6.1k per GPU |
| Peak HBM | 47.6 GiB per GPU in use after the first step (60.6 on the first GPU), 70 GiB reserved (135 on the first) |
| Startup | init 2.1 s, compile 48 s, first step 7.4 s |
| Power | 828-836 W per GPU, 0.144 J/token at the GPUs; the power limit holds the clock down 75-76% of the time (1.60-1.62 GHz) |

Reference loss curve, for regression checks: means over 100 steps of the
training loss, of the floor of the same batches and of their difference, and
the validation loss after the last step of the window, whose floor is 1.5981.

| Steps | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 500-599 | 600-699 | 700-799 | 800-899 | 900-999 |
|---|---|---|---|---|---|---|---|---|---|---|
| Training loss | 2.8221 | 1.9075 | 1.6666 | 1.6110 | 1.5930 | 1.5672 | 1.5835 | 1.5774 | 1.5924 | 1.5803 |
| Floor | 1.5686 | 1.5599 | 1.5727 | 1.5572 | 1.5658 | 1.5478 | 1.5696 | 1.5664 | 1.5831 | 1.5726 |
| Loss above the floor | 1.2536 | 0.3476 | 0.0939 | 0.0538 | 0.0272 | 0.0195 | 0.0139 | 0.0110 | 0.0092 | 0.0078 |
| Validation loss | 2.4231 | 1.7492 | 1.6598 | 1.6323 | 1.6193 | 1.6159 | 1.6099 | 1.6072 | 1.6065 | 1.6051 |

The gradient norm falls from 11.6 to 0.3 and is clipped at every step before
step 360, 423 steps in all. No step is non-finite. The record is
[results/2026-11-17/ideogram4_fsdp4_1000-steps.jsonl](../results/2026-11-17/ideogram4_fsdp4_1000-steps.jsonl).

## Checks

The model is checked against diffusers' model and against the model of
MaxDiffusion's Ideogram branch, which `--model=maxdiffusion:ideogram4` runs as
an adapter: in tests with small sizes, at full size on one GPU, and step by
step over 1,000 training steps from the same weights
([pipelines.md](../pipelines.md#ideogram-4-2026-11-17-4x-mi350x)).

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
- Throughput has not been measured on an idle node or on 8 GPUs.
- One node.
