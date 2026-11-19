"""Pretokenized text corpora and a fixed-length window sampler."""

import dataclasses
import pathlib
import urllib.request

import numpy as np

WIKITEXT_103_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-103-raw-v1/{split}-{shard:05d}-of-{num_shards:05d}.parquet"
)
WIKITEXT_103_SHARDS = {"train": 2, "validation": 1}


def prepare_wikitext103(data_dir: str, split: str = "train", tokenizer: str = "",
                        name: str = "") -> pathlib.Path:
  """Downloads WikiText-103 (raw) and writes its token ids as uint32.

  The ids are Llama 3's, in `wikitext103_<split>.llama3.bin`, or with
  `tokenizer`, the directory of a published model, that model's
  (`jax_titan.data.tokenizer.HubTokenizer`), in `wikitext103_<split>.<name>.bin`;
  `name` defaults to the directory's name.
  """
  import pyarrow.parquet as pq  # pylint: disable=g-import-not-at-top

  from jax_titan.data import tokenizer as tokenizers  # pylint: disable=g-import-not-at-top

  root = pathlib.Path(data_dir)
  root.mkdir(parents=True, exist_ok=True)
  name = name or (pathlib.Path(tokenizer).name if tokenizer else "llama3")
  out = root / f"wikitext103_{split}.{name}.bin"
  if out.exists():
    return out
  num_shards = WIKITEXT_103_SHARDS[split]
  lines = []
  for shard in range(num_shards):
    path = root / f"{split}-{shard:05d}.parquet"
    if not path.exists():
      url = WIKITEXT_103_URL.format(split=split, shard=shard, num_shards=num_shards)
      urllib.request.urlretrieve(url, path)
    lines.extend(pq.read_table(path).column("text").to_pylist())
  # WikiText marks article starts with a single " = Title = " line.
  docs, current = [], []
  for line in lines:
    if line.startswith(" = ") and not line.startswith(" = = ") and current:
      docs.append("".join(current))
      current = []
    current.append(line)
  if current:
    docs.append("".join(current))
  if tokenizer:
    tok = tokenizers.HubTokenizer(tokenizer)
    bos = []
  else:
    tok = tokenizers.Llama3Tokenizer()
    bos = [tok.bos_id]
  ids = []
  for start in range(0, len(docs), 2048):
    for doc_ids in tok.encode_batch(docs[start:start + 2048]):
      ids.append(np.asarray([*bos, *doc_ids, tok.eos_id], np.uint32))
  tokens = np.concatenate(ids)
  tmp = out.with_suffix(".tmp")
  tokens.tofile(tmp)
  tmp.rename(out)
  return out


@dataclasses.dataclass
class WindowSampler:
  """Samples [batch, seq_len + 1] windows; `state` restores the cursor."""

  path: str
  seq_len: int
  seed: int = 0
  step: int = 0

  def __post_init__(self):
    self._tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
    self.num_tokens = int(self._tokens.shape[0])

  def next(self, batch: int) -> np.ndarray:
    rng = np.random.default_rng((self.seed, self.step))
    starts = rng.integers(0, self.num_tokens - self.seq_len - 1, size=batch)
    self.step += 1
    return np.stack(
        [self._tokens[s:s + self.seq_len + 1] for s in starts]
    ).astype(np.int32)

  def state(self) -> dict:
    return {"seed": self.seed, "step": self.step}

  def max_token(self) -> int:
    return int(self._tokens.max())


if __name__ == "__main__":
  import argparse

  from jax_titan import paths  # pylint: disable=g-import-not-at-top

  _parser = argparse.ArgumentParser(description="Writes the WikiText-103 token files.")
  _parser.add_argument(
      "--tokenizer", default="",
      help="The directory of a published model, for that model's token ids instead of Llama 3's.")
  _parser.add_argument(
      "--name", default="", help="The token files' name; default: the directory's.")
  _args = _parser.parse_args()
  for _split in WIKITEXT_103_SHARDS:
    _path = prepare_wikitext103(
        str(paths.work_dir() / "data"), _split, _args.tokenizer, _args.name)
    _count = np.memmap(_path, dtype=np.uint32, mode="r").shape[0]
    print(f"{_split}: {_path} ({_count / 1e6:.1f}M tokens)")
