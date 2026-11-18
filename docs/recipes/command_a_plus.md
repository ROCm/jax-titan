# Recipe: Command A+ with FSDP on 4x MI350X

Command A+ is Cohere's open-weight language model of 218.75B parameters. It is
a sparse mixture-of-experts transformer: in each of its 32 layers a router
sends a token to 8 of 128 experts, and a SigLIP vision tower lets it read
images. JAX-Titan has it as its own model, reads and writes the published
checkpoint's layout, trains it and generates text with it. So far it has run
at reduced sizes, up to 2.11B parameters; the published weights have not been
loaded.

## Running the recipe

### Training

A model of 2.11B parameters on WikiText-103, the text
`python -m jax_titan.data.text` prepares:

```bash
export JAX_TITAN_LAUNCH_TIME=$(date +%s.%N)
HIP_VISIBLE_DEVICES=0,1,2,3 XLA_FLAGS="$JAX_TITAN_ROCM_XLA_FLAGS" \
  python -m jax_titan.trainer.train --model=command-a-plus --parallel=fsdp --num_stages=1 \
  --microbatches=1 --microbatch_size=16 --seq_len=1024 --warmup_steps=100 --steps=3000 \
  --eval_every=250 --eval_batches=8 --run_dir=runs/command-a-plus \
  --model_overrides='{"vocab_size": 128256, "hidden_size": 1024, "num_hidden_layers": 16,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "intermediate_size": 1024, "num_experts": 32, "num_experts_per_tok": 8,
    "num_shared_experts": 4, "sliding_window": 256}'
python -m jax_titan.observability.summarize runs/command-a-plus
```

- **Size:** `--model_overrides` sets fields of the model's config; without
  it, the model is the published one of 218.75B parameters.
- **Layout:** FSDP over the visible devices in one `jax.jit` program, as one
  stage. `--parallel=dp` replicates the state instead.
- **Data:** windows of a uint32 token file, by default WikiText-103 as Llama
  3 tokens (`--data=FILE`), or random tokens (`--data=synthetic`).
- **Optimizer:** AdamW at 3e-4 after 100 warmup steps, then a cosine decay
  to 3e-5 at the last step; weight decay 0.1 on all but the norms; gradient
  norm clipped at 1.0.
- **Weights:** random, or with `--init_from` those of a checkpoint in the
  published layout or of parameters that `--save_params` wrote.
- **Images:** `"image_tiles": 2` in `--model_overrides` opens every sequence
  of a synthetic batch with two random image tiles, and then the vision tower
  and the projector train too. Otherwise the language model trains alone.
- **Precision:** fp32 parameters and AdamW state, bf16 compute; `"dtype":
  "float32"` in `--model_overrides` computes in fp32.

### Text from a checkpoint

`CKPT` is a checkpoint in the published layout, such as a snapshot of
`CohereLabs/command-a-plus-05-2026-bf16` (Apache-2.0). The command needs the
`command_a` extra.

```bash
python -m jax_titan.models.command_a_generate --checkpoint=$CKPT \
  --prompt="What is in this image?" --image=picture.png
```

- **Prompt:** the user's turn of a conversation that the checkpoint's chat
  template writes; `--system` adds a system message, `--image` a picture
  before the text.
- **Images:** a picture becomes a grid of up to 12 tiles of 512x512 and, from
  two tiles on, one more of the whole picture; a tile is 256 tokens.
- **Decoding:** the likeliest token at every step, or with `--temperature`
  drawn tokens, until the end-of-turn token or `--max_new_tokens`.
- **Weights:** `--params` replaces the checkpoint's by parameters that the
  trainer's `--save_params` wrote.

## Correctness

### Against transformers at a small size

The model is checked against transformers' PyTorch implementation, the one
Cohere published the checkpoint for (`cohere2_moe` and `cohere2_vision`,
transformers 5.23.0), at a small size in fp32. The small model keeps every
branch of the published config: sequences longer than the sliding window,
more query heads than key heads, fewer chosen experts than experts.

- **Logits, loss and gradients match, on CPU and on a GPU.** On text, on text
  padded on the left and on text with image tiles, logits differ by at most
  4.5e-6 on CPU and, for a checkpoint with logits up to 18.6, 1.4e-5 on one
  MI350X. The loss agrees to 2e-7 of its value, and the gradient of every
  parameter to 9.8e-7.
- **The checkpoint layout goes both ways.** transformers loads a checkpoint
  that JAX-Titan wrote, and JAX-Titan one that transformers wrote; the
  parameters come back bit for bit.
- **Generation matches.** The 8 tokens after a prompt are those of
  transformers' greedy decoding, with and without image tiles, on CPU and on
  a GPU.
- **bf16 moves the logits as it moves transformers'.** Against fp32, bf16
  logits differ by 0.08 to 0.09 on average and by up to 1.9 at single
  positions; transformers' own bf16 differs by 0.07 to 0.09 and up to 1.8. A
  router's choice between near-equal experts can flip.
- **Preprocessing matches the published processor.** With the published
  tokenizer, chat template and image settings, the rendered conversations and
  their token ids are equal; the tiles of a 1000x700 picture are within 2 of
  255 levels, and equal at more than 99 values in 100.
- **The trainer's layouts agree.** On 4 GPUs in fp32, FSDP follows the
  data-parallel run within 4.4e-7 of the loss at every step, on text and with
  image tiles, and both give the first loss of a run on CPU within 1.1e-7.
  On CPU a pipeline of one stage agrees too, the first loss from a checkpoint
  is that checkpoint's loss on the batch, and a resumed run repeats the
  uninterrupted one exactly.
- **The published checkpoint has the layout the model reads and writes:** the
  names and shapes of its 13,019 tensors, read from the headers of its files,
  follow the rule by which JAX-Titan names and shapes the tensors of a
  checkpoint.

### Loss convergence on text

The run of this recipe: 3,000 steps of 16 sequences of 1,024 tokens, 49M of
the text's 120M tokens, in bf16.

| Steps | 0-249 | 500-749 | 1000-1249 | 1500-1749 | 2000-2249 | 2500-2749 | 2750-2999 |
|---|---|---|---|---|---|---|---|
| Training loss | 6.6162 | 4.7200 | 4.2515 | 3.9265 | 3.7159 | 3.5953 | 3.5813 |
| Validation loss | 5.3735 | 4.5336 | 4.1323 | 3.8480 | 3.6912 | 3.6023 | 3.5828 |

- **The loss falls and held-out text follows.** It starts at 12.30, ends at
  3.58 on held-out text, and no step is non-finite. A uniform guess over the
  128,256 tokens scores 11.76, and predicting every token by its frequency
  in the training text 7.36: the model uses context.
- **8 GPUs draw the same curve.** On the same batches the means over 250
  steps differ by at most 0.0075 and the validation losses by at most 0.0096.
- **The trained parameters write text.** Saved by `--save_params` and loaded
  on one GPU, the model continues a title line of the text in its grammar.

There is no public trainer of Command A+ to compare a curve with. Not
checked: the published weights themselves.

The numbers of these runs are in
[pipelines.md](../pipelines.md#command-a-2026-11-18-4x-and-8x-mi350x).

## Performance

TODO: measure throughput and memory on an idle node. What the runs above saw
is in [pipelines.md](../pipelines.md#command-a-2026-11-18-4x-and-8x-mi350x).

## Known limits

- Run at reduced sizes only, up to 2.11B parameters: no run with the
  published weights.
- Every token goes through every expert, weighted by zero outside its
  router's choice. The result is the model's; the MLP work at the published
  size is 11 times that of the 8 chosen experts and the shared MLP.
- One stage only, no pipeline: the output head is the embedding matrix,
  which a pipeline would need on its first stage and on its last.
- Loading converts the whole checkpoint to fp32 in host memory before placing
  it.
- Training with images takes synthetic tiles only; there is no dataset of
  pictures and text.
- Generation recomputes the whole sequence for every token: there is no
  key-value cache.
- Tiles are resized by Pillow, the published processor's by torchvision.
- Attention runs on native JAX only: the model always passes a mask, which
  the TE and JAX-AITER attention decline.
- No tensor parallelism and no expert parallelism.
