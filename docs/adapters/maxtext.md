# MaxText adapter

`--model=maxtext:<name>` trains a model config of MaxText 0.2.5 with JAX-Titan's
trainer. The model is MaxText's and runs as MaxText wrote it; the training is
JAX-Titan's. A model family needs no code in JAX-Titan: `WholeModel`
(`models/adapters/maxtext.py`) calls MaxText's `Transformer` whole, under
`--parallel=dp` or `fsdp`.

## Models

`<name>` is a model config of MaxText, `configs/models/<name>.yml`. The adapter
takes every config whose training is one gradient step of the cross-entropy
and the load-balance term. Six ran on GPUs at their real size:

| `--model=` | Published weights | Parameters |
|---|---|---|
| `maxtext:qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 0.60B |
| `maxtext:qwen3-8b` | `Qwen/Qwen3-8B` | 8.19B |
| `maxtext:llama3.1-8b` | `meta-llama/Llama-3.1-8B` | 8.03B |
| `maxtext:gemma3-4b` | `google/gemma-3-4b-it`, its text tower | 3.88B |
| `maxtext:mistral-7b` | `mistralai/Mistral-7B-v0.1` | 7.24B |
| `maxtext:qwen3-30b-a3b` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | 30.53B |

MaxText's other model families take the same path and were not run on GPUs
here. A config that MaxText's own validation rejects under its defaults needs
the override MaxText names, as `{"scan_layers": false}` for Gemma 4 E2B; none
was run.

A config under which MaxText's training holds more than this step is refused:
multi-token prediction, the z-loss, the indexer loss of DeepSeek V3.2 and V4,
vocabulary tiling, updates of the routing bias, QK-clip, quantization and
gradient accumulation.

## Running it

Install the `maxtext` extra ([getting_started.md](../getting_started.md)). A
model that does not read Llama 3's tokens needs token files of its own
tokenizer:

```bash
python -m jax_titan.data.text --tokenizer=$MODEL_DIR --name=qwen3
```

From random weights, on every GPU of the node:

```bash
python -m jax_titan.trainer.train --model=maxtext:qwen3-8b --parallel=fsdp \
  --microbatches=1 --microbatch_size=32 --seq_len=2048 \
  --data=$DATA/wikitext103_train.qwen3.bin --eval_data=$DATA/wikitext103_validation.qwen3.bin
```

From published weights, add `--init_from` with a MaxText parameter checkpoint,
the output of MaxText's own converter
([getting_started.md](../getting_started.md) has the commands).
`--model_overrides` takes MaxText config fields as JSON.

## What comes from MaxText and what from JAX-Titan

| Part | Comes from |
|---|---|
| Model: structure, layers, config, initialization | MaxText |
| Published weights: converter and reader | MaxText |
| Loss | JAX-Titan |
| Train step: gradient, clipping, optimizer, schedule | JAX-Titan |
| Layouts (`dp`, `fsdp`) | JAX-Titan |
| Data | JAX-Titan |
| Checkpoints, validation, logs, metrics | JAX-Titan |

MaxText's own trainer is not used.

## What is supported and what is not

| Feature | Supported | Note |
|---|---|---|
| `--parallel=dp` and `fsdp` | Yes | |
| Pipelines (`--parallel=mpmd`) | No | A model of another repository is called whole; a pipeline needs a model written for JAX-Titan, as `--model=llama3.1-8b` is |
| Random weights | Yes | |
| Published weights (`--init_from`) | Yes | A MaxText parameter checkpoint; not for a model that keeps weights outside its parameters, as DeepSeek V3 keeps its routing bias |
| Real data | Yes | Token files of the model's own tokenizer |
| Checkpoints and resuming | Yes | |
| Validation | Yes | |
| Rematerialization | Yes | Always on: MaxText rematerializes its scanned layers under every policy, and `--no-remat` is refused |
| `jax_titan.ops` (`--ops`) | Attention only | `--ops=attention=...`, for a model's global causal attention |
| MFU | Yes | |
| Writing the trained parameters (`--save_params`) | No | |
| Generation or sampling | No | |

## How it is verified

The unit tests are in `tests/test_maxtext_adapter.py` and
`tests/test_adapter_training.py`.

| What is checked | By | Size | Result |
|---|---|---|---|
| The adapter's loss and gradients are MaxText's own: `WholeModel` against MaxText's `train.loss_fn` for six configs (Llama 3.1, Mistral, Gemma 3, Qwen3 and two mixtures of experts, Qwen3-30B-A3B's and DeepSeek's with its summed load-balance loss) | Unit test | Small | Loss within 1e-5, gradients within 1e-3 |
| The trainer starts at the model's loss | Unit test | Small | Within 1e-5 |
| FSDP on 2 GPUs trains as data parallelism does, for four models | Unit test | Small | Losses of 4 steps within 1e-4 |
| A small model trains through the trainer and resumes exactly, under dp and under FSDP | Unit test | Small | Equal losses |
| The trainer starts from a MaxText parameter checkpoint | Unit test | Small | The loss of the checkpoint's parameters, within 1e-5 |
| The published weights give PyTorch's loss, for the six models | [Experiment](#published-weights) | Real | In fp32 within 2.3e-5 for Qwen3 and Llama, the mixture of experts included, 3.0e-6 for Mistral with its RoPE base corrected and 3.5e-4 for Gemma 3 |
| FSDP gives data parallelism's losses on 2, 4 and 8 GPUs, for five models (Qwen3-30B-A3B with 8 of its 48 layers) | [Experiment](#fsdp-against-data-parallelism) | Real | In fp32 within 1.6e-5 over 10 steps |
| A run resumes from its checkpoint | [Experiment](#checkpoints) | Real | That step's loss exactly, the next two within 8e-8 |
| Every model trains | [Experiment](#training-from-random-weights) | Real | The loss falls from 10.8-12.9 to 5.1-6.6 in 300 steps |
| JAX-Titan's own Llama gives the adapter's loss and gradients from the same weights, with Llama 3.1's RoPE and with Llama 3's ([below](#jax-titans-own-llama-against-the-adapter)) | Unit test, experiment | Small; real | Loss within 1e-5, gradients within 1e-3; at real size the losses are 3.6e-7 apart in fp32 |

Not checked:

- **MaxText's other model families** on GPUs.
- **The load-balance term at real size.** It ran in the tests at a small size
  only: none of the six configs weights it.
- **Attention through `jax_titan.ops` with TE or JAX-AITER.** It was checked
  at a small size with the native provider, in the tests; no experiment here
  used another.
- **The gradients at real size.** The experiments compare the loss.

## Experiments

These are runs of the trainer on the adapter's models, launched by hand on
GPUs. They are not part of the test suite, and neither the scripts that
launched them nor the script of the PyTorch reference are in the repository.
They ran on 8x MI350X on 2026-11-19, Llama 3.1 8B's `dp` runs on 2026-11-20,
and take the six models of [Models](#models), in bf16 with fp32 parameters and
AdamW state unless a row says fp32, on WikiText-103 in each model's own
tokens.

### Published weights

Each model starts from its published weights, and its loss is compared with
PyTorch's. The weights come from the Hugging Face Hub, converted with
MaxText's own converter (`maxtext.checkpoint_conversion.to_maxtext`, and
`standalone_scripts/llama_or_mistral_ckpt` for Mistral) and loaded with
`--init_from`; the reference is transformers 5.23.0's model in PyTorch on CPU.
Both compute the mean cross-entropy of the batch the trainer draws at step 0
from WikiText-103's validation split, 8 sequences of 1,024 tokens. One MI350X;
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
- **The two `mistral-7b` rows differ by the RoPE base**: MaxText's config is
  not the published model's ([Known issues](#known-issues)).
- **Gemma 3 is the least close**: 3.5e-4 in fp32, and 5.8e-4 when the
  converter saves fp32 instead of its default bf16, so the saved dtype is not
  the cause. In bf16 it is 8.6e-3 from PyTorch's fp32, where PyTorch's own
  bf16 is 4.9e-3.

### FSDP against data parallelism

The same training under `--parallel=fsdp` on 2, 4 and 8 GPUs and under
`--parallel=dp` on one GPU: random weights of `--seed=0`, the same 8 sequences
per step, 10 steps. The table gives the largest relative difference of the
loss over the 10 steps, with sequences of 2,048 tokens in bf16 and of 1,024 in
fp32.

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

### Training from random weights

Each model trains from random weights with 4 sequences of 2,048 tokens per
GPU, AdamW at the trainer's defaults (peak rate 3e-4 after 10 warm-up steps)
and full remat. Tokens/s per GPU is the median of three runs, one of 300 steps
and two of 40, each taken over its steps 5-39; the spread is their range over
that median. "Over 300 steps" is the 300-step run over its steps 5-299, and
the losses are that run's. The peak memory is the 40-step runs': the 300-step
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

- **Every model trains**: the loss falls from 10.8-12.9 to 5.1-6.6 in 300
  steps, and lower where a step holds more tokens.
- **Over 300 steps the throughput is 1% to 4% below that of the first 40.**
  The GPUs warm up and their clock drops under the power limit: for Llama
  3.1 8B on one GPU from 1,622 to 1,573 MHz, at 54 and 63 C and 1 kW.
- **Under FSDP a GPU's throughput rises with the number of GPUs** at the
  same tokens per GPU: 0.65-0.69 of one GPU's `dp` throughput on 2 GPUs,
  0.81-0.82 on 4 and 0.90-0.94 on 8. These runs do not show why.
- **Under FSDP every GPU holds its share and no more**: the weights are
  initialized and restored on their shardings. Qwen3-30B-A3B peaks at 65-68
  GiB on each of 8 GPUs with its published weights and at 82 GiB with random
  ones.

### MaxText's own trainer

The throughput of the same models in MaxText's own trainer
(`maxtext.trainers.pre_train.train`, the same MaxText, node, stack and day),
with the same 8,192 tokens per GPU, full remat and dot-product attention, on
synthetic data. Tokens/s per GPU over steps 5-39, the median of three runs of
40 steps on each side:

| `maxtext:` model | GPUs | MaxText's own trainer | JAX-Titan | Difference |
|---|---|---|---|---|
| `llama3.1-8b` | 1 | 10,189 | 10,632 | +4.3% |
| `llama3.1-8b` | 8, FSDP | 9,660 | 9,955 | +3.1% |
| `qwen3-8b` | 1 | 9,669 | 10,074 | +4.2% |
| `qwen3-8b` | 8, FSDP | 9,194 | 9,417 | +2.4% |

- **MaxText's models run 2.4% to 4.3% faster here than in MaxText's own
  trainer** over the same steps, on one GPU and under FSDP on 8.

### Mixture of experts

MaxText computes an expert layer in three forms. The adapter's default is the
dense matmul over every expert (`sparse_matmul=false`), and
`--model_overrides` selects the others. Qwen3-30B-A3B, which routes a token to
8 of 128 experts, with 8 of its 48 layers (5.6B parameters) on one GPU: 4
sequences of 2,048 tokens, three runs of 30 steps, each taken over its steps
5-29:

| Expert matmul | Tokens/s per GPU | Spread | Peak HBM | Loss, step 0 | Steps 25-29 |
|---|---|---|---|---|---|
| Dense matmul over every expert, the default | 14,792 | 0.8% | 103 GiB | 12.494 | 7.006 |
| `jax.lax.ragged_dot` (`sparse_matmul=true`, `megablox=false`) | 3,027 | 0.2% | 186 GiB | 12.494 | 7.007 |
| Megablox, MaxText's default (`sparse_matmul=true`, `megablox=true`) | 2,024 | 6.7% | 91 GiB | 12.494 | 7.006 |

- **The dense matmul is the fastest of the three forms for this model on
  this stack**: 4.9x `jax.lax.ragged_dot` and 7.3x megablox, with the same
  losses.
- **The whole model trains on 8 GPUs at 2,158 tokens/s per GPU** (the last
  row of the table under
  [Training from random weights](#training-from-random-weights)). Its MFU of
  1.9% counts the 8 experts a token is routed to, as MaxText's FLOP count
  does; the dense matmul computes all 128.

### Checkpoints

Qwen3-0.6B from its published weights, under FSDP on 2 GPUs in fp32, restores
the checkpoint of step 3 and gives that step's loss exactly and the next two
within 8e-8.

## JAX-Titan's own Llama against the adapter

This section compares JAX-Titan's Llama (`--model=llama3.1-8b`) with the
adapter's Llama (`--model=maxtext:llama3.1-8b`): from the same weights the two
give the same loss and the same gradients, the adapter's being the reference.

MaxText's Llama and JAX-Titan's are two implementations of one model.
`llama_config` gives a MaxText Llama's sizes as a config of JAX-Titan's model,
and `llama_params` maps its weights: the layer axis first, MaxText's
`1/sqrt(head_dim)` taken back out of the query projection, and, for Llama
3.1's RoPE, the query and key features of each head reordered from adjacent
pairs to split halves.

At a small size, a unit test compares the two for `llama3.1-8b` and
`llama3-8b` in fp32: the loss is within 1e-5 and the gradients within 1e-3.

At real size, from Meta's Llama 3.1 8B weights on one MI350X, on the batch of
[Published weights](#published-weights) (8 sequences of 1,024 tokens), the
loss was compared, not the gradients:

| Compute | JAX-Titan's Llama | Adapter | Difference | PyTorch |
|---|---|---|---|---|
| fp32 | 1.959045 | 1.959046 | 3.6e-7 | 1.959044 |
| bf16 | 1.958346 | 1.958825 | 4.8e-4 | 1.958628 |

The script of the real-size comparison is not in the repository.

Trained alike from random weights, with the same arguments `$ARGS` for both,
the two are compared as in the example of
[comparison.md](comparison.md#llama-31-8b):

```bash
python -m jax_titan.trainer.train --model=llama3.1-8b $ARGS --run_dir=runs/jax-titan
python -m jax_titan.trainer.train --model=maxtext:llama3.1-8b $ARGS --run_dir=runs/maxtext
python -m jax_titan.observability.compare runs/jax-titan runs/maxtext
```

## Known issues

- **MaxText's `mistral-7b` config is not the published Mistral 7B v0.1.** It
  sets `rope_max_timescale` to 1,000,000; the weights its converter names
  have a RoPE base of 10,000. With MaxText's config the loss is 2.196 against
  PyTorch's 1.889, and with
  `--model_overrides='{"rope_max_timescale": 10000}'` it is within 3.0e-6.
- **MaxText's converter needs
  `--lazy_load_tensors=False --eager_load_method=transformers` for Gemma 3**:
  the published files name their tensors `language_model.model.*`, and the
  converter's lazy reader looks for `model.language_model.*`.
- **MaxText's `qwen3-0.6b` starts at a loss of 233 from random weights**: its
  config ties the output to an embedding of standard deviation 1 without
  normalizing the logits.
- **The dense expert matmul computes every expert.** It is the adapter's
  default, and it was measured against MaxText's other forms for
  Qwen3-30B-A3B's 8 of 128 experts only
  ([Mixture of experts](#mixture-of-experts)).
