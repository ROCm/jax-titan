# MLPerf Training

JAX-Titan is being built to train MLPerf Training's Closed-division benchmarks
and to beat the times AMD published for MLPerf Training v6.1 with Primus. This
page holds the benchmark contracts, with the source of every value, and what
JAX-Titan implements of them.

> **Status.** The contract of Llama 3.1 8B and GPT-OSS 20B is in
> `jax_titan.mlperf.contract`. No compliant run exists yet.

## The benchmarks

| Benchmark | Quality target | Runs | AMD v6.1 on MI355X | Contract in code |
|---|---|---|---|---|
| `llama31_8b` | validation log perplexity ≤ 3.3 | 10 | 72.94 min, 8 GPUs | yes |
| `gpt_oss_20b` | validation log perplexity ≤ 3.34 | 10 | 56.18 min, 8 GPUs | yes |
| `flux1` | validation loss ≤ 0.586 | 10 | 47.11 min, 32 GPUs | no |
| `deepseekv3_671b` | validation log perplexity ≤ 3.6 | 3 | none | no |
| `qwen35_397b_grpo` | pass@4 ≥ 0.69 | 3 | none | no |

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
  `c4-train.en_6_text_document`, shards 768 to 895, as AMD's v6.1 configs do;
  the published dataset holds only that training file.
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
