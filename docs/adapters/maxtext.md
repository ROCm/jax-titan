# MaxText adapter

`--model=maxtext:<name>` trains a model config of MaxText 0.2.5 with JAX-Titan's
trainer. The network is MaxText's and runs as MaxText wrote it; the training is
JAX-Titan's. A model family needs no code in JAX-Titan: `WholeModel`
(`models/adapters/maxtext.py`) calls MaxText's `Transformer` whole, as one
stage, under `--parallel=dp` or `fsdp`.

## What comes from MaxText and what from JAX-Titan

| Part | From MaxText | From JAX-Titan |
|---|---|---|
| Model structure and layers | `Transformer`, called as it is | |
| Model config | `base.yml` and `models/<name>.yml` | A few defaults (no dropout, the text tower alone), `--model_overrides`, `--seq_len` |
| Parameter initialization | Its initializers | Placed on the trainer's shardings |
| Precision | The config's default: bf16 compute, fp32 weights | |
| Layer scan and rematerialization | Its own | |
| Attention | Its dot-product attention | The choice of it; `--ops` routes it to `jax_titan.ops` |
| Expert layers of a mixture of experts | Its code, in three forms | The choice of the dense matmul |
| Loss | The load-balance term of a mixture of experts | The cross-entropy, and the sum of the two |
| Gradient, clipping, update | | The train step |
| Optimizer and schedule | | AdamW without weight decay on norms; warm-up and cosine decay |
| Layouts (`dp`, `fsdp`) | The logical axis names on its parameters; its constraints on activations | The mesh, how parameters and optimizer state are sharded, how a batch is spread |
| Data and tokenizer | | Token files and their sampling; the tokenizer is the model's published `tokenizer.json` |
| Published weights | Its converter, run once, and its reader | The shardings the weights are read onto |
| Checkpoints and resuming | | JAX-Titan's (Orbax) |
| Validation, logs, metrics | The FLOP count of a step | Everything else |
| What MaxText's training holds beyond this step | In its own trainer | Not implemented: such a config is refused |

MaxText's own trainer takes no part: not its training loop, its `loss_fn`, its
optimizer, its input pipeline or its checkpointing.

## Running it

Install the `maxtext` extra ([getting_started.md](../getting_started.md)). A
model that does not read Llama 3's tokens needs token files of its own
tokenizer:

```bash
python -m jax_titan.data.text --tokenizer=$MODEL_DIR --name=qwen3
```

From random weights, on every GPU of the node:

```bash
python -m jax_titan.trainer.train --model=maxtext:qwen3-8b --parallel=fsdp --num_stages=1 \
  --microbatches=1 --microbatch_size=32 --seq_len=2048 \
  --data=$DATA/wikitext103_train.qwen3.bin --eval_data=$DATA/wikitext103_validation.qwen3.bin
```

From published weights, add `--init_from` with a MaxText parameter checkpoint,
the output of MaxText's own converter
([getting_started.md](../getting_started.md) has the commands).
`--model_overrides` takes MaxText config fields as JSON.

## What is supported

- Every model config of MaxText 0.2.5 whose training is one gradient step of
  the cross-entropy and the load-balance term, as one stage under `dp` and
  `fsdp`. Six ran on GPUs at their real size: Qwen3-0.6B, Qwen3-8B, Llama 3.1
  8B, Gemma 3 4B (its text tower), Mistral 7B and Qwen3-30B-A3B.
- Random weights, or a MaxText parameter checkpoint with `--init_from`.
- Checkpoints and resuming, validation, and `--ops=attention=...` for a
  model's global causal attention.

## What is not supported

- **Pipelines**: `--parallel=mpmd`, more than one stage. A model of another
  repository is called whole; a pipeline needs a model written for JAX-Titan,
  as `--model=llama3.1-8b` is.
- **Configs under which MaxText's training holds more than this step are
  refused**: multi-token prediction, the z-loss, the indexer loss of DeepSeek
  V3.2 and V4, vocabulary tiling, updates of the routing bias, QK-clip,
  quantization and gradient accumulation. `--no-remat` is refused too:
  MaxText rematerializes its scanned layers under every policy.
- **A model that keeps weights outside its parameters**, as DeepSeek V3
  keeps its routing bias, does not take `--init_from`.
- **A config that MaxText's own validation rejects under its defaults**
  needs the override MaxText names, as `{"scan_layers": false}` for Gemma 4
  E2B. None was run.
- **MaxText's other model families** take the same path and were not run on
  GPUs here. The load-balance term ran in the tests at a small size only:
  none of these six configs weights it.
- **The dense expert matmul computes every expert.** It was measured against
  the other forms for Qwen3-30B-A3B's 8 of 128 experts only.
- **Attention through `jax_titan.ops`** was checked at a small size with the
  native provider, in the tests; no run here used TE or JAX-AITER.
- **Checkpoints**: one resume was checked. Qwen3-0.6B from its published
  weights under FSDP on 2 GPUs in fp32 restores the checkpoint of step 3,
  gives that step's loss exactly and the next two within 8e-8.
- **MaxText's `qwen3-0.6b` starts at a loss of 233 from random weights**: its
  config ties the output to an embedding of standard deviation 1 without
  normalizing the logits.

## How it is verified

### Tests

`tests/test_maxtext_adapter.py`, on GPUs, at a small size in fp32:

- **The adapter's loss and gradients are MaxText's own**: `WholeModel` against
  MaxText's `train.loss_fn` for six configs (Llama 3.1, Mistral, Gemma 3, Qwen3
  and two mixtures of experts, Qwen3-30B-A3B's and DeepSeek's with its summed
  load-balance loss), the loss within 1e-5 and the gradients within 1e-3.
- **The trainer starts at the model's loss.**
- **FSDP trains as data parallelism does**, on 2 GPUs, for four models.
- **JAX-Titan's own Llama gives the adapter's loss and gradients** from the
  same weights, with Llama 3.1's RoPE and with Llama 3's
  ([below](#jax-titans-own-llama-against-the-adapter)).

`tests/test_adapter_training.py`, on CPU: a small model trains through the
trainer, resumes exactly, and starts from a MaxText parameter checkpoint.

### Runs (2026-11-19, 8x MI350X)

These runs take the six models above. They ran on MI350X in bf16 with fp32
parameters and AdamW state, unless a row says fp32, on WikiText-103 in each
model's own tokens. Llama 3.1 8B's `dp` runs were made again on 2026-11-20:
until then `dp` ran Llama as the one stage of a pipeline the adapter no
longer has.

Published weights: each model's weights from the Hugging Face Hub, converted
with MaxText's own converter (`maxtext.checkpoint_conversion.to_maxtext`, and
`standalone_scripts/llama_or_mistral_ckpt` for Mistral) and loaded with
`--init_from`, against transformers 5.23.0's model in PyTorch on CPU. Both
compute the mean cross-entropy of the batch the trainer draws at step 0 from
WikiText-103's validation split, 8 sequences of 1,024 tokens. One MI350X;
Qwen3-30B-A3B with FSDP on 8. PyTorch's bf16 column holds bf16 weights;
JAX-Titan's holds fp32 weights and computes in bf16.

| `maxtext:` model | Published weights | Parameters | PyTorch, fp32 | JAX-Titan, fp32 | Difference, fp32 | PyTorch, bf16 | JAX-Titan, bf16 |
|---|---|---|---|---|---|---|---|
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 0.60B | 3.040727 | 3.040750 | 2.3e-05 | 3.040178 | 3.040360 |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 8.19B | 2.407396 | 2.407410 | 1.4e-05 | 2.409170 | 2.407982 |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B` | 8.03B | 1.959044 | 1.959045 | 1.1e-06 | 1.958628 | 1.958825 |
| `gemma3-4b` | `google/gemma-3-4b-it` | 3.88B | 3.170996 | 3.171343 | 3.5e-04 | 3.166130 | 3.179606 |
| `mistral-7b`, MaxText's config | `mistralai/Mistral-7B-v0.1` | 7.24B | 1.889283 | 2.196191 | 3.1e-01 | 1.889253 | 2.195423 |
| `mistral-7b`, RoPE base 10,000 | `mistralai/Mistral-7B-v0.1` | 7.24B | 1.889283 | 1.889286 | 3.0e-06 | 1.889253 | 1.889170 |
| `qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | 30.53B | 2.107349 | 2.107361 | 1.1e-05 | 2.106362 | 2.107455 |

- **The published weights give PyTorch's loss**: within 2.3e-5 in fp32 for
  Qwen3 and Llama, the mixture of experts included. In bf16 they are within
  6e-4 of PyTorch's fp32 loss, where PyTorch's own bf16 is up to 1.8e-3 from
  it.
- **MaxText's `mistral-7b` config is not the published Mistral 7B v0.1.** It
  sets `rope_max_timescale` to 1,000,000; the weights its converter names
  have a RoPE base of 10,000. With MaxText's config the loss is 2.196 against
  PyTorch's 1.889, and with
  `--model_overrides='{"rope_max_timescale": 10000}'` it is within 3.0e-6.
- **Gemma 3 is the least close**: 3.5e-4 in fp32, and 5.8e-4 when the
  converter saves fp32 instead of its default bf16, so the saved dtype is not
  the cause. In bf16 it is 8.6e-3 from PyTorch's fp32, where PyTorch's own
  bf16 is 4.9e-3. Its converter needs
  `--lazy_load_tensors=False --eager_load_method=transformers`: the published
  files name their tensors `language_model.model.*`, and the converter's lazy
  reader looks for `model.language_model.*`.

FSDP against data parallelism: random weights of `--seed=0`, the same 8
sequences per step, 10 steps, `--parallel=fsdp` on 2, 4 and 8 GPUs against
`--parallel=dp` on one GPU. The largest relative difference of the loss over the 10 steps, with sequences
of 2,048 tokens in bf16 and of 1,024 in fp32:

| `maxtext:` model | Compute | fsdp, 2 GPUs | fsdp, 4 GPUs | fsdp, 8 GPUs | Step 0, largest of the three |
|---|---|---|---|---|---|
| `qwen3-8b` | bf16 | 2.4e-03 | 8.3e-04 | 2.9e-03 | 2.8e-06 |
| `llama3.1-8b` | bf16 | 5.9e-03 | 1.7e-03 | 3.4e-03 | 1.9e-05 |
| `gemma3-4b` | bf16 | 1.7e-03 | 1.3e-03 | 2.7e-03 | 2.4e-05 |
| `mistral-7b` | bf16 | 1.4e-03 | 7.7e-04 | 1.7e-03 | 1.7e-05 |
| `qwen3-30b-a3b`, 8 of 48 layers | bf16 | 5.1e-05 | 7.2e-05 | 2.1e-05 | 9.9e-06 |
| `qwen3-8b` | fp32 | 3.0e-07 | 4.6e-07 | 7.6e-07 | 7.7e-08 |
| `llama3.1-8b` | fp32 | 2.4e-06 | 9.8e-06 | 1.6e-05 | 0.0e+00 |
| `gemma3-4b` | fp32 | 1.7e-06 | 2.2e-06 | 1.4e-06 | 7.4e-08 |
| `mistral-7b` | fp32 | 2.2e-06 | 6.2e-06 | 2.4e-06 | 8.8e-08 |
| `qwen3-30b-a3b`, 8 of 48 layers | fp32 | 2.1e-07 | 2.1e-07 | 2.1e-07 | 0.0e+00 |

- **In fp32 FSDP gives data parallelism's losses**, within 1.6e-5 over 10
  steps on 2, 4 and 8 GPUs for every model.
- **In bf16 the layouts part by rounding.** The difference is at most 2.4e-5
  at step 0 and reaches up to 5.9e-3 within 10 steps, as the warm-up's loss
  spikes amplify it. The `dp` run of Llama 3.1 8B itself, run twice on one
  GPU, differs by 8.7e-4, and by 2.0e-3 from a run on another GPU. Five of
  the twelve runs of the four 8B-class models exceed the 2e-3 the docs pass to
  `summarize --gate0`, which is what one `dp` run already differs from another
  on a second GPU: at these 16,384 tokens per step that gate is too tight for
  bf16, and fp32 is what tells one layout from another.

Training from random weights with 4 sequences of 2,048 tokens per GPU, AdamW
at the trainer's defaults (peak rate 3e-4 after 10 warm-up steps) and full
remat. Tokens/s per GPU is the median of three runs, one of 300 steps and two
of 40, each taken over its steps 5-39; the spread is their range over that
median. "Over 300 steps" is the 300-step run over its steps 5-299, and the
losses are that run's. The peak memory is the 40-step runs': the 300-step
runs predate the initialization of the weights on their shardings, and in
them one GPU holds more than the others. MFU is against MI350X's 2,306.9
TFLOP/s in bf16; the trainer's own `mfu` is against its default of 2,500.

| `maxtext:` model | Layout | GPUs | Tokens per step | Tokens/s per GPU | Spread | Over 300 steps | MFU | Peak HBM per GPU | Loss, step 0 | Steps 96-100 | 196-200 | 295-299 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3-8b` | dp | 1 | 8,192 | 10,074 | 0.2% | 9,823 | 20.6% | 126 GiB | 12.47 | 6.66 | 6.01 | 5.83 |
| `qwen3-8b` | fsdp | 2 | 16,384 | 6,672 | 0.3% | 6,553 | 13.7% | 80 GiB | 12.42 | 6.50 | 5.90 | 5.54 |
| `qwen3-8b` | fsdp | 4 | 32,768 | 8,230 | 0.9% | 8,012 | 16.8% | 44 GiB | 12.47 | 6.25 | 5.65 | 5.35 |
| `qwen3-8b` | fsdp | 8 | 65,536 | 9,417 | 1.6% | 9,036 | 19.3% | 28 GiB | 12.48 | 6.06 | 5.40 | 5.07 |
| `llama3.1-8b` | dp | 1 | 8,192 | 10,632 | 0.7% | 10,435 | 21.5% | 123 GiB | 12.33 | 7.22 | 6.67 | 6.30 |
| `llama3.1-8b` | fsdp | 2 | 16,384 | 6,913 | 0.4% | 6,873 | 14.0% | 78 GiB | 12.31 | 6.91 | 6.34 | 6.03 |
| `llama3.1-8b` | fsdp | 4 | 32,768 | 8,638 | 0.9% | 8,420 | 17.5% | 46 GiB | 12.29 | 6.57 | 6.00 | 5.73 |
| `llama3.1-8b` | fsdp | 8 | 65,536 | 9,955 | 1.8% | 9,609 | 20.1% | 28 GiB | 12.30 | 6.38 | 5.89 | 5.68 |
| `gemma3-4b` | dp | 1 | 8,192 | 20,057 | 0.5% | 19,708 | 20.8% | 77 GiB | 12.91 | 7.27 | 6.57 | 6.55 |
| `gemma3-4b` | fsdp | 2 | 16,384 | 13,816 | 0.4% | 13,722 | 14.3% | 49 GiB | 12.89 | 7.15 | 6.46 | 6.27 |
| `gemma3-4b` | fsdp | 4 | 32,768 | 16,317 | 0.3% | 16,113 | 16.9% | 36 GiB | 12.88 | 6.99 | 6.31 | 6.15 |
| `gemma3-4b` | fsdp | 8 | 65,536 | 18,077 | 0.6% | 17,728 | 18.8% | 29 GiB | 12.86 | 6.88 | 6.08 | 5.76 |
| `mistral-7b` | dp | 1 | 8,192 | 10,916 | 0.4% | 10,673 | 20.9% | 113 GiB | 10.81 | 6.56 | 5.74 | 5.87 |
| `mistral-7b` | fsdp | 2 | 16,384 | 7,146 | 1.1% | 7,009 | 13.7% | 73 GiB | 10.84 | 6.41 | 5.75 | 5.59 |
| `mistral-7b` | fsdp | 4 | 32,768 | 8,869 | 0.8% | 8,666 | 17.0% | 40 GiB | 10.85 | 6.22 | 5.58 | 5.40 |
| `mistral-7b` | fsdp | 8 | 65,536 | 10,230 | 1.6% | 9,893 | 19.6% | 23 GiB | 10.85 | 6.02 | 5.45 | 5.30 |
| `qwen3-30b-a3b` | fsdp | 8 | 65,536 | 2,158 | 0.1% | 2,114 | 1.9% | 82 GiB | 12.40 | 6.13 | 5.54 | 5.32 |

MaxText's own trainer (`maxtext.trainers.pre_train.train`, the same MaxText,
node, stack and day) with the same 8,192 tokens per GPU, full remat and
dot-product attention, on synthetic data. Tokens/s per GPU over steps 5-39,
the median of three runs of 40 steps on each side:

| `maxtext:` model | GPUs | MaxText's own trainer | JAX-Titan, one stage | Difference |
|---|---|---|---|---|
| `llama3.1-8b` | 1 | 10,189 | 10,632 | +4.3% |
| `llama3.1-8b` | 8, FSDP | 9,660 | 9,955 | +3.1% |
| `qwen3-8b` | 1 | 9,669 | 10,074 | +4.2% |
| `qwen3-8b` | 8, FSDP | 9,194 | 9,417 | +2.4% |

- **Every model trains**: the loss falls from 10.8-12.9 to 5.1-6.6 in 300
  steps, and lower where a step holds more tokens.
- **MaxText's models run 2.4% to 4.3% faster here than in MaxText's own
  trainer** over the same steps, on one GPU and under FSDP on 8.
- **Over 300 steps the throughput is 1% to 3% below that of the first 40.**
  The GPUs warm up and their clock drops under the power limit: for Llama
  3.1 8B on one GPU from 1,622 to 1,573 MHz, at 54 and 63 C and 1 kW.
- **Under FSDP a GPU's throughput rises with the number of GPUs** at the
  same tokens per GPU: 0.65-0.69 of one GPU's `dp` throughput on 2 GPUs,
  0.81-0.82 on 4 and 0.90-0.94 on 8. These runs do not show why.
- **Under FSDP every GPU holds its share and no more**: the weights are
  initialized and restored on their shardings. Qwen3-30B-A3B peaks at 65-68
  GiB on each of 8 GPUs with its published weights and at 82 GiB with random
  ones.

The mixture of experts: MaxText computes an expert layer in three forms. The
adapter's default is the dense matmul over every expert
(`sparse_matmul=false`), and `--model_overrides` selects the others.
Qwen3-30B-A3B, which routes a token to 8 of 128 experts, with 8 of its 48
layers (5.6B parameters) on one GPU: 4 sequences of 2,048 tokens, three runs
of 30 steps, each taken over its steps 5-29:

| Expert matmul | Tokens/s per GPU | Spread | Peak HBM | Loss, step 0 | Steps 25-29 |
|---|---|---|---|---|---|
| Dense matmul over every expert, the default | 14,792 | 0.8% | 103 GiB | 12.494 | 7.006 |
| `jax.lax.ragged_dot` (`sparse_matmul=true`, `megablox=false`) | 3,027 | 0.2% | 186 GiB | 12.494 | 7.007 |
| Megablox, MaxText's default (`sparse_matmul=true`, `megablox=true`) | 2,024 | 6.7% | 91 GiB | 12.494 | 7.006 |

- **The dense matmul is the fastest of the three forms for this model on
  this stack**: 4.9x `jax.lax.ragged_dot` and 7.3x megablox, with the same
  losses.
- **The whole model trains on 8 GPUs at 2,158 tokens/s per GPU** (the last
  row of the table before). Its MFU of 1.9% counts the 8 experts a token is
  routed to, as MaxText's FLOP count does; the dense matmul computes all 128.

The script of the PyTorch reference and the scripts that launched these runs
are not in the repository.

## JAX-Titan's own Llama against the adapter

The adapter is the reference for `jax_titan.models.llama`: MaxText's Llama
and JAX-Titan's are two implementations of one model. `llama_config` gives a
MaxText Llama's sizes as a config of JAX-Titan's model, and `llama_params`
maps its weights: the layer axis first, MaxText's `1/sqrt(head_dim)` taken
back out of the query projection, and, for Llama 3.1's RoPE, the query and
key features of each head reordered from adjacent pairs to split halves.

From the same weights the two give the same loss and the same gradients:

- **At a small size**, in the tests above: within 1e-5 for the loss and 1e-3
  for the gradients, in fp32 on a GPU, for `llama3.1-8b` and `llama3-8b`.
- **At real size**, from Meta's Llama 3.1 8B weights on one MI350X, on the
  batch of the published-weights table above (8 sequences of 1,024 tokens):
  in fp32 JAX-Titan's Llama gives a loss of 1.959045 and the adapter
  1.959046, 3.6e-7 apart, where PyTorch gives 1.959044. In bf16 they give
  1.958346 and 1.958825, 4.8e-4 apart, where PyTorch's own bf16 gives
  1.958628. At this size the loss was compared, not the gradients.

The script of the real-size comparison is not in the repository.
