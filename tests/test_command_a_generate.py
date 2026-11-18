"""Command A+'s preprocessing gives the published processor's tokens and tiles,
and generation takes the likeliest token of a full pass."""

import dataclasses
import json
import os

import command_a_checkpoint
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_titan.models import command_a
from jax_titan.models import command_a_generate

CONFIG = command_a_checkpoint.CONFIG
# The published files without the weights, to compare with transformers' processor.
PUBLISHED = os.environ.get("COMMAND_A_CHECKPOINT", "")


def _picture(width=70, height=40):
  """Four quadrants of one color each: red, green, blue, white."""
  Image = pytest.importorskip("PIL.Image")  # pylint: disable=invalid-name
  pixels = np.zeros((height, width, 3), np.uint8)
  pixels[:height // 2, :width // 2] = (255, 0, 0)
  pixels[:height // 2, width // 2:] = (0, 255, 0)
  pixels[height // 2:, :width // 2] = (0, 0, 255)
  pixels[height // 2:, width // 2:] = 255
  return Image.fromarray(pixels)


def test_a_picture_gets_the_published_grid():
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top

  # (height, width) to (columns, rows), as transformers' image processor picks them.
  grids = {
      (512, 512): (1, 1), (700, 1000): (2, 2), (1000, 700): (2, 2), (300, 200): (1, 1),
      (200, 300): (1, 1), (1024, 1024): (2, 2), (768, 1024): (2, 2), (2000, 500): (1, 4),
      (500, 2000): (4, 1), (4000, 3000): (3, 4), (64, 64): (1, 1), (512, 1536): (3, 1),
      (1100, 1100): (3, 3), (513, 512): (1, 2), (1536, 2048): (4, 3), (600, 6200): (11, 1)}
  for (height, width), expected in grids.items():
    assert data.grid(height, width, 512, 12) == expected, (height, width)


def test_a_conversation_becomes_tokens_and_tiles(tmp_path):
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top

  command_a_checkpoint.write(tmp_path)
  preprocessor = data.Preprocessor.open(tmp_path)
  assert preprocessor.tile_tokens == CONFIG.tile_tokens == 4
  words = command_a_checkpoint.WORDS

  def ids(*names):
    return [words.index(n) for n in names]

  tokens, tiles = preprocessor.encode([{"role": "user", "content": "a b"}])
  assert tiles is None and tokens.dtype == np.int32
  expected = ids("<BOS_TOKEN>", "<|USER|>", "a", "b", "<|END_OF_TURN_TOKEN|>", "<|CHATBOT|>")
  np.testing.assert_array_equal(tokens, expected)
  assert preprocessor.decode(tokens[2:4]) == "a b"

  # 70x40 is shrunk least for a grid of 2x2 tiles of 32; the fifth tile is the whole picture.
  content = [{"type": "image"}, {"type": "text", "text": "c"}]
  tokens, tiles = preprocessor.encode([{"role": "user", "content": content}], [_picture()])
  assert tiles.shape == (5, 3, 32, 32) and tiles.dtype == np.float32
  tile = ids(*["<|IMG_PATCH|>"] * 4, "<|IMG_LINE_BREAK|>")
  expected = (ids("<BOS_TOKEN>", "<|USER|>", "<|START_OF_IMG|>") + tile * 5
              + ids("<|END_OF_IMG|>", "c", "<|END_OF_TURN_TOKEN|>", "<|CHATBOT|>"))
  np.testing.assert_array_equal(tokens, expected)
  colors = np.asarray([(1, -1, -1), (-1, 1, -1), (-1, -1, 1), (1, 1, 1)], np.float32)
  np.testing.assert_allclose(tiles[:4, :, 16, 16], colors, atol=0.02)
  np.testing.assert_allclose(tiles[4][:, [8, 8, 24, 24], [8, 24, 8, 24]].T, colors, atol=0.02)
  with pytest.raises(ValueError, match="1 image parts for 0 images"):
    preprocessor.encode([{"role": "user", "content": content}])


@pytest.mark.skipif(not (PUBLISHED and command_a_checkpoint.has_reference()),
                    reason="needs COMMAND_A_CHECKPOINT, torch and transformers with cohere2_moe")
def test_matches_the_published_processor():
  import transformers  # pylint: disable=g-import-not-at-top
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top

  processor = transformers.AutoProcessor.from_pretrained(PUBLISHED)
  preprocessor = data.Preprocessor.open(PUBLISHED)
  picture = _picture(1000, 700)
  chats = (
      [{"role": "user", "content": "What has keys but can't open locks?"}],
      [{"role": "system", "content": "Answer in one short sentence."},
       {"role": "user", "content": "What has keys but can't open locks?"},
       {"role": "assistant", "content": "A piano."},
       {"role": "user", "content": "And what has hands but can't clap?"}],
      [{"role": "user", "content": [{"type": "image", "image": picture},
                                    {"type": "text", "text": "Describe this image."}]}],
  )
  for messages in chats:
    theirs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")
    images = [picture] if "pixel_values" in theirs else []
    assert preprocessor.text(messages) == processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    tokens, tiles = preprocessor.encode(messages, images)
    np.testing.assert_array_equal(tokens, theirs["input_ids"][0])
    if images:
      # Pillow's bicubic against torchvision's: within 2 of 255 levels.
      assert tiles.shape == theirs["pixel_values"].shape == (5, 3, 512, 512)
      np.testing.assert_allclose(tiles, theirs["pixel_values"], atol=2.01 / 127.5)
      assert (np.abs(tiles - theirs["pixel_values"]) < 1e-6).mean() > 0.99


def _prompts():
  """Prompts of 24 tokens of the text batch and of the batch with image tiles."""
  batches = dict(command_a_checkpoint.batches())
  for name in ("text", "image"):
    tokens, pixels, _ = batches[name]
    yield name, tokens[:, :24], pixels


def test_generation_takes_the_likeliest_token_of_a_full_pass():
  # No token ends a sequence here, so every sequence runs to the last step.
  cfg, params = dataclasses.replace(CONFIG, eos_token_id=-1), command_a_checkpoint.params()
  for name, tokens, pixels in _prompts():
    new = command_a_generate.generate(cfg, params, tokens, pixels, new_tokens=4, dtype=jnp.float32)
    assert new.shape == (2, 4) and new.dtype == np.int32
    sequence = tokens
    for step in range(4):
      logits = command_a.forward(cfg, params, sequence, pixels, None, jnp.float32)
      np.testing.assert_array_equal(new[:, step], np.argmax(logits[:, -1], -1), err_msg=name)
      sequence = np.concatenate([sequence, new[:, step:step + 1]], axis=1)
    drawn = command_a_generate.generate(cfg, params, tokens, pixels, new_tokens=4,
                                        dtype=jnp.float32, temperature=5.0, seed=1)
    assert not np.array_equal(drawn, new)


def test_an_image_file_is_read_upright(tmp_path):
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top

  Image = pytest.importorskip("PIL.Image")  # pylint: disable=invalid-name
  upright = np.zeros((16, 32, 3), np.uint8)
  upright[:, :16] = 255
  # Stored on its side, with the tag that says to turn it a quarter clockwise.
  exif = Image.Exif()
  exif[0x0112] = 6
  Image.fromarray(np.rot90(upright)).save(tmp_path / "tagged.jpg", quality=100, exif=exif)
  image = np.asarray(data.open_image(tmp_path / "tagged.jpg"))
  assert image.shape == upright.shape
  assert image[:, :8].mean() > 200 and image[:, 24:].mean() < 50


def test_a_cache_takes_conversations_that_end_with_the_assistant(tmp_path):
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top

  jinja2 = pytest.importorskip("jinja2")
  command_a_checkpoint.write(tmp_path / "checkpoint")
  user, bot = {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}

  def prepare(*conversations):
    file = tmp_path / "conversations.jsonl"
    file.write_text("".join(json.dumps({"messages": c}) + "\n" for c in conversations),
                    encoding="utf-8")
    return data.prepare(tmp_path / "checkpoint", file, tmp_path / "cache")

  with pytest.raises(ValueError, match="line 2: a conversation ends with the assistant's turn"):
    prepare([user, bot], [user, bot, user])
  with pytest.raises(ValueError, match="line 1: a conversation ends with the assistant's turn"):
    prepare([bot])
  picture = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "a"}]}
  with pytest.raises(ValueError, match="line 1: the conversation has 1 image parts for 0 images"):
    prepare([picture, bot])
  with pytest.raises(ValueError, match="holds no conversation"):
    prepare()
  assert len(prepare([user, bot])) == 1
  with pytest.raises(FileExistsError):
    prepare([user, bot])

  # An assistant's reasoning under any of the names transformers takes it by.
  preprocessor = dataclasses.replace(
      data.Preprocessor.open(tmp_path / "checkpoint"),
      template=jinja2.Template("{% for m in messages %}[{{ m.thinking }}]{% endfor %}"))
  for name in ("thinking", "reasoning_content", "reasoning"):
    assert preprocessor.text([user, bot | {name: "why"}]) == "[][why]"


def test_stages_on_several_devices_generate_as_one_device(tmp_path):
  # No token ends a sequence here, so every sequence runs to the last step.
  cfg, params = dataclasses.replace(CONFIG, eos_token_id=-1), command_a_checkpoint.params()
  devices = jax.devices()[:3]
  stages = command_a.to_stages(cfg, params, devices, jnp.float32)
  assert [len(s["layers"]["norm"]) for s in stages] == [3, 3, 2]
  assert sorted(stages[0]) == ["embed", "layers", "projector", "vision"]
  assert sorted(stages[1]) == ["layers"] and sorted(stages[2]) == ["embed", "final_norm", "layers"]
  for stage, device in zip(stages, devices):
    assert {d for w in jax.tree.leaves(stage) for d in w.devices()} == {device}
  for name, tokens, pixels in _prompts():
    one = command_a_generate.generate(cfg, params, tokens, pixels, new_tokens=4, dtype=jnp.float32)
    several = command_a_generate.generate(cfg, stages, tokens, pixels, new_tokens=4,
                                          dtype=jnp.float32)
    np.testing.assert_array_equal(several, one, err_msg=name)
  # A checkpoint read stage by stage gives the same stages.
  command_a.save(tmp_path, CONFIG, params)
  _, read = command_a.load_stages(tmp_path, devices, jnp.float32)
  assert jax.tree.structure(read) == jax.tree.structure(stages)
  for a, b in zip(jax.tree.leaves(read), jax.tree.leaves(stages)):
    np.testing.assert_array_equal(a, b)
    assert a.devices() == b.devices()
  language = command_a.load_stages(tmp_path, devices[:1], jnp.float32, images=False)[1]
  assert sorted(language[0]) == ["embed", "final_norm", "layers"]


def test_generation_stops_at_the_end_of_sequence_token():
  params = command_a_checkpoint.params(images=False)
  tokens = next(_prompts())[1]
  free = command_a_generate.generate(
      dataclasses.replace(CONFIG, eos_token_id=-1), params, tokens, new_tokens=6, dtype=jnp.float32)
  # A token that the first sequence gives once and the second never ends the first alone.
  step = next(k for k in range(6) if free[0, k] not in free[0, :k] and free[0, k] not in free[1])
  ends = dataclasses.replace(CONFIG, eos_token_id=int(free[0, step]))
  new = command_a_generate.generate(ends, params, tokens, new_tokens=6, dtype=jnp.float32)
  np.testing.assert_array_equal(new[0, :step + 1], free[0, :step + 1])
  np.testing.assert_array_equal(new[0, step + 1:], CONFIG.pad_token_id)
  np.testing.assert_array_equal(new[1], free[1])
  both = command_a_generate.generate(ends, params, tokens[:1], new_tokens=6, dtype=jnp.float32)
  np.testing.assert_array_equal(both, free[:1, :step + 1])


@command_a_checkpoint.needs_reference
def test_generation_matches_transformers(tmp_path):
  import torch  # pylint: disable=g-import-not-at-top

  command_a.save(tmp_path, CONFIG, command_a_checkpoint.params())
  model = command_a_checkpoint.reference(tmp_path)
  cfg, params = command_a.load(tmp_path, jnp.float32)
  for name, tokens, pixels in _prompts():
    inputs = command_a_checkpoint.reference_inputs(tokens, pixels, None)
    with torch.no_grad():
      theirs = model.generate(**inputs, max_new_tokens=8, min_new_tokens=8, do_sample=False,
                              pad_token_id=CONFIG.pad_token_id)
    ours = command_a_generate.generate(dataclasses.replace(cfg, eos_token_id=-1), params, tokens,
                                       pixels, new_tokens=8, dtype=jnp.float32)
    np.testing.assert_array_equal(ours, theirs[:, 24:].numpy(), err_msg=name)


def test_command_line_answers_about_a_picture(tmp_path, capsys):
  command_a_checkpoint.write(tmp_path / "checkpoint")
  _picture().save(tmp_path / "picture.png")
  flags = [f"--checkpoint={tmp_path / 'checkpoint'}", "--max_new_tokens=3", "--platform=cpu"]
  picture = f"--image={tmp_path / 'picture.png'}"
  assert command_a_generate.main([*flags, "--prompt=a b", picture]) == 0
  out = capsys.readouterr().out
  assert "loaded on 8 devices" in out and "generated" in out
  assert "the prompt is 33 tokens with 5 image tiles" in out
  (tmp_path / "prompt.txt").write_text("a b c\n", encoding="utf-8")
  prompt = f"--prompt=@{tmp_path / 'prompt.txt'}"
  assert command_a_generate.main([*flags, prompt, "--system=d", "--dtype=float32"]) == 0
  assert "the prompt is 10 tokens\n" in capsys.readouterr().out
  # Several prompts are answered one after the other from one load.
  assert command_a_generate.main([*flags, "--prompt=a b", "--prompt=a b", "--prompt=d"]) == 0
  lines = capsys.readouterr().out.splitlines()
  assert len(lines) == 9 and lines[0].startswith("loaded on 8 devices")
  assert lines[0].endswith("; the prompt is 6 tokens") and lines[3] == "the prompt is 6 tokens"
  assert lines[6] == "the prompt is 5 tokens"
  assert all(lines[i].startswith("generated 3 tokens in ") for i in (1, 4, 7))
