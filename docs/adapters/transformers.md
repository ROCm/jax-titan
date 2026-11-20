# transformers adapter

`--model=transformers:<name>` trains a decoder-only causal language model of
Hugging Face transformers 5.23.0 with JAX-Titan's trainer, under
`--parallel=dp` or `fsdp`. The network is transformers' and is written in
PyTorch; the training is JAX-Titan's. The network does not run as PyTorch:
torchax 0.0.13 executes its `forward` on JAX arrays while JAX traces it, so
PyTorch runs on the host during that trace and never during a step
(`models/adapters/torch_bridge.py`, `models/adapters/transformers.py`).

## Models

The adapter supports 125 model types of transformers. To print them:

```bash
python -c "from jax_titan.models.adapters import transformers as t; print(t.VERIFIED)"
```

Among them are `llama`, `mistral`, `qwen2`, `qwen3`, `gemma`, `gemma2`,
`gemma3_text`, `phi3`, `gpt2`, `falcon`, `mamba`, and mixtures of experts such
as `qwen3_moe`, `mixtral`, `deepseek_v3` and `gpt_oss`. For each of the 125, a
small model gave PyTorch's loss and gradients.

Four of them were verified on GPUs at their real size:

| Published weights | Model type | Parameters |
|---|---|---|
| `meta-llama/Llama-3.1-8B` | `llama` | 8.03B |
| `Qwen/Qwen3-8B` | `qwen3` | 8.19B |
| `mistralai/Mistral-7B-v0.1` | `mistral` | 7.24B |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | `qwen3_moe` | 30.53B |

`<name>` is a directory with a model's `config.json`, such as a download from
the Hugging Face Hub, or a model type of transformers with the library's
default sizes (`llama`, `qwen3_moe`).

The list exists because the network is translated, operator by operator, into
a JAX function that `jax.grad` differentiates and XLA compiles, and a model
can come out wrong without an error: PyTorch's autograd takes no part, so
`torch.no_grad()`, `.data` and a custom `autograd.Function` in a model's
`forward` neither stop nor shape the gradient. The adapter therefore takes
only the model types whose loss and gradients it compared with PyTorch's own.
A model type outside the list is refused.

## Running it

Install the `transformers` extra; PyTorch's CPU build is enough:

```bash
pip install -e ".[transformers]" --extra-index-url https://download.pytorch.org/whl/cpu
```

From random weights, with the config of a downloaded Qwen3-8B and token files
of its own tokenizer ([maxtext.md](maxtext.md#running-it) has their command):

```bash
python -m jax_titan.trainer.train --model=transformers:$MODELS/qwen3-8b \
  --parallel=fsdp --microbatches=1 --microbatch_size=32 --seq_len=2048 \
  --data=$DATA/wikitext103_train.qwen3.bin --eval_data=$DATA/wikitext103_validation.qwen3.bin
```

From the published weights in that directory, add `--init_from=$MODELS/qwen3-8b`.
`--model_overrides` takes config fields as JSON, and `"dtype"` for the compute
dtype:

```bash
python -m jax_titan.trainer.train --model=transformers:llama --parallel=dp \
  --model_overrides='{"vocab_size": 1024, "hidden_size": 128, "intermediate_size": 256,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2}' \
  --microbatches=1 --microbatch_size=8 --seq_len=64 --data=synthetic
```

## What comes from transformers and what from JAX-Titan

| Part | Comes from |
|---|---|
| Model: structure, layers, config, initialization | transformers |
| Published weights: `from_pretrained` | transformers |
| Expert layer of a mixture of experts | JAX-Titan |
| Loss | JAX-Titan |
| Train step: gradient, clipping, optimizer, schedule | JAX-Titan |
| Layouts (`dp`, `fsdp`) | JAX-Titan |
| Data | JAX-Titan |
| Checkpoints, validation, logs, metrics | JAX-Titan |

transformers' own `Trainer` is not used.

## What is supported and what is not

| Feature | Supported | Note |
|---|---|---|
| `--parallel=dp` and `fsdp` | Yes | |
| Pipelines (`--parallel=mpmd`) | No | A model of another repository is called whole; a pipeline needs a model written for JAX-Titan |
| Random weights | Yes | |
| Published weights (`--init_from`) | Yes | The weights of a local directory |
| Real data | Yes | Token files of the model's own tokenizer |
| Checkpoints and resuming | Yes | |
| Validation | Yes | |
| Rematerialization | Yes | `--remat`, the default |
| `jax_titan.ops` (`--ops`) | No | The model's operators are what torchax lowers them to: no MX quantization and no provider kernels |
| MFU | No | No FLOP count exists for an arbitrary model |
| Writing the trained parameters (`--save_params`) | No | |
| Generation or sampling | No | |

Dropout is not supported either: every dropout probability of the config is
set to zero, and a forward that still draws random numbers is refused, since a
compiled step would draw the same numbers at every step.

## How it is verified

The unit tests are in `tests/test_transformers_adapter.py`.

| What is checked | By | Size | Result |
|---|---|---|---|
| Each model type of the list gives PyTorch's loss and gradients, over two traces | One comparison | Small | Loss within 1e-5, gradients within 1e-3 |
| Seven model types give PyTorch's loss and gradients, over two traces: `llama`, `mistral`, `qwen3`, `gemma3_text`, `phi3` with its long RoPE factors, `qwen3_moe` in the dense form and `gpt2` | Unit test | Small | Loss within 1e-5, gradients within 1e-3 |
| Norms take no weight decay, and nothing else is exempt | Unit test | Small | Passes |
| The trainer starts at the model's loss | Unit test | Small | Within 1e-5 |
| FSDP on 2 GPUs trains as data parallelism does | Unit test | Small | Losses of 4 steps within 1e-4 |
| A model type outside the list is refused, and so is a pipeline | Unit test | Small | Passes |
| The published weights give PyTorch's loss, for four models | [Experiment](#published-weights) | Real | Within 2.0e-5 in fp32, the mixture of experts included |
| A run resumes from its checkpoint and validates | [Experiment](#checkpoints) | Real | The uninterrupted run's losses exactly, its validation loss within 2e-8 |
| Every model trains | [Experiment](#training-from-random-weights) | Real | The loss falls from 11.2-12.8 to 7.2-8.0 in 40 steps |

`VERIFIED` is the outcome of one comparison with PyTorch, made on 2026-11-19
with transformers 5.23.0, torch 2.10.0 and torchax 0.0.13, on CPU in fp32. For
each of the 143 decoder-only text model types of `AutoModelForCausalLM` it
built a small model the adapter's way, from the configuration of the family's
own model tester in transformers' test suite, or from the default config with
its sizes shrunk. PyTorch computed the mean cross-entropy of the logits and
its gradients on a copy of the module. The adapter's loss function, with
rematerialization, was traced twice under `jax.jit(jax.value_and_grad)`, as
the trainer traces it for its train step and its validation step. A model type
passed when both traces gave PyTorch's loss within 1e-5 and every parameter's
gradient within 1e-3: 126 passed, 2 gave another gradient and 15 raised an
error. The list holds the 126 without `bert-generation`, which is not a
left-to-right model under its default config. Among those that passed, the
largest difference is 2.1e-7 for the loss and 3.4e-5 for a gradient. The
comparison needs transformers' test suite and is not in the repository; it has
to be run again when one of the three versions moves.

Not checked:

- **The gradients at real size.** The experiments compare the loss.

## Experiments

These are runs of the trainer on the adapter's models, launched by hand on
GPUs. They are not part of the test suite, and neither the scripts that
launched them nor the script of the comparison with PyTorch are in the
repository. They ran on 8x MI350X on 2026-11-19 and 20.

### Published weights

Each model starts from its published weights, and its loss is compared with
PyTorch's. The weights come from the Hugging Face Hub in a local directory,
named by `--model=transformers:` and `--init_from`, in fp32. The loss is the
trainer's at step 0, on the batch it draws from WikiText-103's validation
split in the model's own tokens, 8 sequences of 1,024 tokens. The reference is
the loss of transformers' model, run by PyTorch on CPU, on the same batch. One
MI350X; Qwen3-30B-A3B with FSDP on 8.

| Published weights | Parameters | GPUs | PyTorch | Through the adapter | Difference | Peak HBM per GPU |
|---|---|---|---|---|---|---|
| `meta-llama/Llama-3.1-8B` | 8.03B | 1 | 1.959044 | 1.959064 | 2.0e-05 | 123 GiB |
| `Qwen/Qwen3-8B` | 8.19B | 1 | 2.407396 | 2.407410 | 1.4e-05 | 125 GiB |
| `mistralai/Mistral-7B-v0.1` | 7.24B | 1 | 1.889283 | 1.889288 | 5.0e-06 | 111 GiB |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | 30.53B | 8 | 2.107349 | 2.107363 | 1.4e-05 | 114 GiB |

- **The published weights give PyTorch's loss** within 2.0e-5, the mixture
  of experts included.

### Training from random weights

Each model trains from random weights, with its config from the same
directory, in bf16 with fp32 parameters: 4 sequences of 2,048 tokens per GPU,
40 steps of AdamW at the trainer's defaults with rematerialization, on
WikiText-103 in the model's own tokens. One run each; tokens/s per GPU is the
median over steps 5-39.

| Config | Layout | GPUs | Tokens per step | Tokens/s per GPU | Peak HBM per GPU | Loss, step 0 | Steps 35-39 |
|---|---|---|---|---|---|---|---|
| `meta-llama/Llama-3.1-8B` | dp | 1 | 8,192 | 12,485 | 113 GiB | 12.57 | 7.99 |
| `meta-llama/Llama-3.1-8B` | fsdp | 8 | 65,536 | 11,585 | 39 GiB | 12.56 | 7.52 |
| `Qwen/Qwen3-8B` | dp | 1 | 8,192 | 11,867 | 117 GiB | 12.74 | 7.77 |
| `Qwen/Qwen3-8B` | fsdp | 8 | 65,536 | 10,811 | 35 GiB | 12.76 | 7.34 |
| `mistralai/Mistral-7B-v0.1` | dp | 1 | 8,192 | 12,931 | 99 GiB | 11.19 | 7.43 |
| `mistralai/Mistral-7B-v0.1` | fsdp | 8 | 65,536 | 11,971 | 35 GiB | 11.19 | 7.23 |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | fsdp | 8 | 65,536 | 3,607 | 131 GiB | 12.38 | 7.15 |

- **Every model trains**: the loss falls from 11.2-12.8 to 7.2-8.0 in 40
  steps.
- **Qwen3-30B-A3B runs through torchax on 8 GPUs** in the dense form of its
  experts, at 3,607 tokens/s per GPU.
- **These are single runs of 40 steps**, enough to see that a model runs,
  fits and learns, not a measurement of throughput.

### Checkpoints

Qwen3-0.6B from its published weights, under FSDP on 2 GPUs in fp32, restores
the checkpoint of step 3 and gives the uninterrupted run's losses of steps 3
to 5 exactly, and its validation loss after them within 2e-8.

## Known issues

- **A slow start from the compilation cache.** JAX-Titan's environment keeps
  JAX's persistent compilation cache (`scripts/env.sh`), and a traced model's
  step that comes from it starts slowly, with the GPUs idle: for Llama 3.1 8B
  under FSDP on 8 GPUs the first step takes 103 s, against 9 s after a fresh
  compile. For Qwen3-30B-A3B it runs past the 300 s the trainer allows a first
  step (`--step_timeout`) and had not finished after 10 minutes. MaxText's and
  MaxDiffusion's models do not show this, and the cause is not known. With
  `JAX_ENABLE_COMPILATION_CACHE=false` Llama 3.1 8B's first step is normal
  again; Qwen3-30B-A3B was run with an empty cache directory.
- **No fast mixture of experts.** The dense form computes every expert; this
  stack has no fast kernel for the routed forms
  ([maxtext.md](maxtext.md#mixture-of-experts)).
