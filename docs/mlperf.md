# MLPerf Training

JAX-Titan is being built to train MLPerf Training's Closed-division benchmarks.
This page holds the benchmark contracts, with the source of every value, and
what JAX-Titan implements of them.

> **Status.** The contract of Llama 3.1 8B and GPT-OSS 20B is in
> `jax_titan.mlperf.contract`; the trainer reads MLPerf's C4 in the
> reference's order and trains with the reference's schedule and AdamW. No
> compliant run exists yet.

## The benchmarks

| Benchmark | Quality target | Runs | Contract in code |
|---|---|---|---|
| `llama31_8b` | validation log perplexity ≤ 3.3 | 10 | yes |
| `gpt_oss_20b` | validation log perplexity ≤ 3.34 | 10 | yes |
| `flux1` | validation loss ≤ 0.586 | 10 | no |
| `deepseekv3_671b` | validation log perplexity ≤ 3.6 | 3 | no |
| `qwen35_397b_grpo` | pass@4 ≥ 0.69 | 3 | no |

A benchmark's score is the mean time to train of its runs without the fastest
and the slowest. The time to train runs from MLLOG's `run_start` to `run_stop`,
and a run may spend at most 30 minutes in initialization before `run_start`.

## The contract of Llama 3.1 8B

Sources: the [MLPerf Training rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc)
of 24 July 2026 (v6.1); the compliance checker's
`closed_llama31_8b.yaml` and `common.yaml` in
[mlcommons/logging](https://github.com/mlcommons/logging) at tag `6.1.0-rc2`
(`compliance_checker/training_6.1.0`); and the reference,
[`small_llm_pretraining/nemo`](https://github.com/mlcommons/training/tree/master/small_llm_pretraining/nemo),
which runs NeMo v2.1.0 on Megatron-core `core_r0.11.0` (its Dockerfiles).
GPT-OSS 20B has the same contract with a target of 3.34
(`closed_gpt_oss_20b.yaml`).

### Fixed

| MLLOG key | Value | Source |
|---|---|---|
| `max_sequence_length` | 8192 | rules, Hyperparameters; checker |
| `opt_name` | `adamw` | checker |
| `opt_adamw_beta_1` | 0.9 | rules; checker |
| `opt_adamw_beta_2` | 0.95 | rules; checker |
| `opt_adamw_epsilon` | 1e-5 | rules; checker |
| `opt_adamw_weight_decay` | 0.1 | rules; checker |
| `opt_gradient_clip_norm` | 1.0 | rules; checker |
| `opt_learning_rate_decay_schedule` | `cosine with linear warmup` | checker |
| `max_steps` | 1,200,000 | rules; checker |
| `eval_samples` | 1,024 sequences per validation | rules, Quality measure; checker |
| `eval_accuracy` | at least one ≤ 3.3, each with `samples_count` | checker |

Dropout is 0. Validation runs every 12,288 training sequences, rounded up to a
whole global batch: every `ceil(12288 / GBS)` steps
(`Contract.eval_every_steps`). Training samples are counted from 0
(`samples_count`).

### Tunable, and how the values tie together

| MLLOG key | Constraint |
|---|---|
| `global_batch_size` | any, in sequences; the reference convergence points (RCPs) cover 16, 32, 64, 96 and 128 |
| `gradient_accumulation_steps` | any positive integer |
| `opt_base_learning_rate` | any, subject to the RCP check |
| `opt_learning_rate_warmup_steps` | any |
| `opt_end_learning_rate` | 0.1 × the base learning rate |
| `opt_learning_rate_decay_steps` | 1,200,000 − the warm-up steps |

`Contract.check` refuses hyperparameters that depart from either table, or
that hold a key neither table has, and names every departure.

The RCPs (`rcp_checker/training_6.1.0/rcps_llama31_8b.json`) were made in BF16
with these values; a set of 10 runs that converges faster than the RCP mean
(without its fastest and slowest of 20) is scaled up to it, and one that is
significantly faster fails:

| GBS | Base learning rate | Warm-up | Mean samples to converge |
|---|---|---|---|
| 16 | 4e-4 | 256 samples (16 steps) | 161,109 |
| 32 | 8e-4 | 4,096 samples (128 steps) | 178,859 |
| 64 | 8e-4 | 6,144 samples (96 steps) | 226,645 |
| 96 | 1e-3 | 16,348 samples | 278,640 |
| 128 | 2e-3 | 32,768 samples (256 steps) | 344,747 |

### What the reference does beyond its README

Where the README is silent, or says otherwise than the code, JAX-Titan follows
what the code computes, except for the validation set:

- **Vocabulary: 128,256 rows.** The README says 128,000. NeMo's
  `AutoTokenizer.vocab_size` is `len(tokenizer)`, 128,256 for Llama 3.1's
  tokenizer, and Megatron pads it to a multiple of 128, which it already is.
  JAX-Titan's `llama3.1-8b` has 128,256 rows.
- **Training data: one file.** The README says the last 256 of C4's 1,024
  training shards. `utils/consolidate_data.sh` merges the shards into eight
  files of 128 each, and `pretrain_llama31.py` reads only
  `c4-train.en_6_text_document`, shards 768 to 895; the published dataset
  holds only that training file.
- **Validation set: the README.** The README uses the first 1,024 sequences of
  `c4-validation-91205-samples.en_text_document` and does not shuffle them. The
  reference code builds its validation set with Megatron's `GPTDataset`, which
  shuffles documents and samples of every split with the run's seed, over as
  many epochs as `(max_steps // val_check_interval + 1) × limit_val_batches × GBS`
  samples need, and evaluates the first 1,024 of that order. JAX-Titan follows
  the README: the first 1,024 sequences in file order. Both measure the loss
  on C4's validation documents; the difference goes to the review.
- **Learning rate.** NeMo's `CosineAnnealing`: `base × (s + 1) / (W + 1)` for
  `s ≤ W`, then a cosine from the base to the end learning rate over
  `1,200,000 − W` steps, where `s` counts the updates done (0 for the first)
  and `W` is the warm-up. NeMo's `MegatronStrategy` copies the trainer's
  `max_steps` into the scheduler.
- **Weight decay.** Megatron's optimizer decays every parameter but biases and
  1-D parameters: the norms' weights are not decayed (Llama has no biases);
  the embedding and the output head are.
- **Loss.** The mean cross-entropy over every token of the global batch, with
  inputs and targets one token apart: a sample is 8,193 tokens. Llama does not
  mask end-of-document tokens (`eod_mask_loss=False`); attention and positions
  run across documents.
- **Initialization.** Normal with standard deviation 0.02 (NeMo's
  `Llama31Config`), and 0.02 / sqrt(2 × 32) for the attention output and MLP
  down projections (Megatron's scaled output-layer initialization); norm
  weights 1. The embedding and the output head are separate matrices.
- **Adam.** Megatron's distributed fused Adam with decoupled weight decay:
  `p ← p − lr × (m̂ / (sqrt(v̂) + ε) + λ p)` with bias-corrected `m̂` and `v̂`.
- **Gradient clipping.** By the norm of all gradients after the gradient
  reduction. Megatron scales by `1.0 / (norm + 1e-6)` when that is below 1;
  JAX-Titan scales by `1.0 / norm` when the norm is 1.0 or more, a relative
  difference of `1e-6 / norm`.

## Data

`scripts/mlperf_download_c4.sh DIR` downloads the pretokenized C4 that
MLCommons publishes for Llama 3.1 8B and GPT-OSS 20B (85 GB), resuming a
transfer that breaks off, and checks every file against MLCommons' md5 sums.
The files are Megatron indexed datasets of Llama 3.1 token ids (int32):

| Prefix | Documents | Tokens | Samples of 8,193 tokens |
|---|---|---|---|
| `c4-train.en_6_text_document` | 45,608,611 | 20,994,193,570 | 2,562,767 per epoch |
| `c4-validation-91205-samples.en_text_document` | 91,205 | 41,642,091 | 5,083 |

The README counts 47,186,855 validation tokens, enough for 5,760 sequences;
the published file holds the 41,642,091 above, and validation needs 1,024.

The trainer reads a Megatron dataset when `--data` or `--eval_data` names its
prefix (`jax_titan.data.megatron`):

- **Training order: the reference's.** `GPTDataset` at `core_r0.11.0` cuts
  the documents into windows of 8,193 tokens that overlap by one, in a
  document order shuffled by `numpy.random.RandomState(seed)` over as many
  epochs as its sample count needs, and shuffles the windows with the same
  random state. The reference asks for `max_steps × GBS` samples. The trainer
  builds the same three indices for `--steps` × its global batch with the
  run's `--seed`, so with `--steps=1200000` and the same seed it reads the
  reference's samples in the reference's order. `tests/test_megatron_data.py`
  checks the indices against a line-by-line port of Megatron's code.
- **Cost.** At GBS 32, the indices span 38,441,516 samples over 15 epochs,
  all shuffled together, as in the reference: the 180,000 or so samples a run
  reads can hold a document twice. Building them reads the documents' lengths,
  not their tokens, which the rules allow before the clock starts; on the dev
  node it takes 55 s and 4.3 GB of memory. They are kept as .npy files (about
  3.2 GB per seed) in `${JAX_TITAN_ROOT}/cache/megatron_indices`, and
  `python -m jax_titan.data.megatron index` builds them before a run.
- **Validation: the first sequences in file order**, as the README says
  (above).

```bash
python -m jax_titan.trainer.train --model=llama3.1-8b --seq_len=8192 --steps=1200000 \
  --data=$DIR/c4-train.en_6_text_document \
  --eval_data=$DIR/c4-validation-91205-samples.en_text_document ...
```

## Model, optimizer and schedule

The trainer's flags that set the reference's training method, with the values
of the RCP at GBS 32:

```bash
--lr=8e-4 --warmup_steps=128 --lr_schedule=nemo_cosine --lr_decay_steps=1199872 \
--adam_b1=0.9 --adam_b2=0.95 --adam_eps=1e-5 --weight_decay=0.1 --max_grad_norm=1.0
```

- **Schedule.** `--lr_schedule=nemo_cosine` is NeMo's `CosineAnnealing`
  (`jax_titan.trainer.schedules`), with `--lr_decay_steps` = 1,200,000 − the
  warm-up. The default, `cosine`, is optax's, which starts the warm-up from 0.
- **AdamW.** `optax.adamw` computes the update of Apex's `FusedAdam` in
  `adam_w_mode`, which the rules list as compliant for Adam on large language
  models and which Megatron's distributed optimizer runs:
  `m ← β1 m + (1 − β1) g`, `v ← β2 v + (1 − β2) g²`,
  `p ← p − lr × ((m / (1 − β1^t)) / (sqrt(v / (1 − β2^t)) + ε) + λ p)`, with
  `λ = 0` for the norms' weights. `tests/test_mlperf_training.py` checks it
  over six steps against that equation and against `torch.optim.AdamW`.
- **Model.** `--model=llama3.1-8b` has Meta-Llama-3.1-8B's 8,030,261,248
  parameters. The tests check its loss and every gradient against
  transformers' `LlamaForCausalLM`, with Llama 3.1's RoPE scaling, from the
  same weights in fp32, and its initialization against the reference's.
- **Layouts.** Under this AdamW, `--parallel=dp`, `--parallel=fsdp` and the
  pipeline agree on CPU, and the pipeline's 1F1B, GPipe and zero-bubble
  schedules, also with FSDP and tensor parallelism in the stages, match one
  `jax.jit` step (`tests/test_mpmd_parity.py`). Each schedule takes one
  update per global batch from the gradients of all its microbatches, so no
  weight is stale.
