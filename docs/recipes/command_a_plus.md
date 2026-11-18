# Recipe: Command A+

Command A+ is Cohere's open-weight language model of 218.75B parameters. It is
a sparse mixture-of-experts transformer: in each of its 32 layers a router
sends a token to 8 of 128 experts, and a SigLIP vision tower lets it read
images. JAX-Titan has it as its own model, reads and writes the published
checkpoint's layout, trains it and generates text with it. So far it has run
at a small size on CPU only.

## Running the recipe

### Training

A small model on token windows, on CPU:

```bash
python -m jax_titan.trainer.train --platform=cpu --model=command-a-plus --parallel=fsdp \
  --num_stages=1 --microbatches=2 --microbatch_size=8 --seq_len=16 --steps=4 --data=synthetic \
  --model_overrides='{"vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 4,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 32,
    "num_experts": 8, "num_experts_per_tok": 2, "num_shared_experts": 2, "sliding_window": 8}'
```

- **Size:** `--model_overrides` sets fields of the model's config; without
  it, the model is the published one.
- **Layout:** FSDP over the visible devices in one `jax.jit` program, as one
  stage. `--parallel=dp` replicates the state instead.
- **Data:** windows of a uint32 token file (`--data=FILE`), or random tokens
  (`--data=synthetic`).
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

The model is checked against transformers' PyTorch implementation, the one
Cohere published the checkpoint for (`cohere2_moe` and `cohere2_vision`,
transformers 5.23.0), at a small size in fp32 on CPU. The small model keeps
every branch of the published config: sequences longer than the sliding
window, more query heads than key heads, fewer chosen experts than experts.

- **Logits, loss and gradients match.** On text, on text padded on the left
  and on text with image tiles, logits of up to 4.5 differ by at most
  4.5e-6. The loss agrees to 2e-7 of its value, and the gradient of every
  parameter, up to 0.24, to 2.1e-7.
- **The checkpoint layout goes both ways.** transformers loads a checkpoint
  that JAX-Titan wrote, and JAX-Titan one that transformers wrote; the
  parameters come back bit for bit.
- **Generation matches.** The 8 tokens after a prompt are those of
  transformers' greedy decoding, with and without image tiles.
- **Preprocessing matches the published processor.** With the published
  tokenizer, chat template and image settings, the rendered conversations and
  their token ids are equal; the tiles of a 1000x700 picture are within 2 of
  255 levels, and equal at more than 99 values in 100.
- **The trainer's layouts agree.** FSDP, and a pipeline of one stage, follow
  the data-parallel run within 1e-5 of the loss at every step, on text and
  with image tiles; the first loss from a checkpoint is that checkpoint's
  loss on the batch, and a resumed run repeats the uninterrupted one exactly.
- **The published checkpoint has the layout the model reads and writes:** the
  names and shapes of its 13,019 tensors, read from the headers of its files,
  follow the rule by which JAX-Titan names and shapes the tensors of a
  checkpoint.

Not checked: the published weights themselves, any run on a GPU, bf16 against
a reference, and a loss curve.

## Performance

TODO: nothing is measured.

## Known limits

- Run at a small size on CPU only: no run with the published weights and none
  on a GPU.
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
