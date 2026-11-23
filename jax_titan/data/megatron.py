"""Megatron-LM indexed datasets (.bin and .idx), sampled as Megatron-core's GPTDataset samples them.

MLPerf's C4 for Llama 3.1 8B and GPT-OSS 20B is published pretokenized as
Megatron indexed datasets: `PREFIX.idx` holds each document's length and byte
offset, and `PREFIX.bin` the token ids. `IndexedDataset` maps both files into
memory with NumPy alone.

`build_indices` cuts a dataset into samples of `seq_len + 1` tokens as
Megatron-core's `GPTDataset` does at `core_r0.11.0`, the version the
reference runs:

- the document order: the documents repeated for as many epochs as
  `num_samples` samples need, shuffled by `numpy.random.RandomState(seed)`,
  the last epoch apart when it supplies few samples;
- the samples: windows over the documents in that order, each `seq_len + 1`
  tokens long and sharing its last token with the next window, so that a
  sample's inputs and targets are one token apart;
- the sample order: a shuffle of the samples by the same random state.

With the same seed, `seq_len` and `num_samples`, a run reads the same samples
in the same order as the reference. Without a seed, documents and samples stay
in file order, as for the validation set that the reference README reads
unshuffled (docs/mlperf.md). Building the indices reads the documents' lengths,
not their tokens; `cached_indices` keeps them as .npy files keyed by
everything they depend on.

  python -m jax_titan.data.megatron info PREFIX --seq_len 8192
  python -m jax_titan.data.megatron index PREFIX --seq_len 8192 --num_samples N --seed S
"""

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import struct
import time

import numpy as np

_HEADER = b"MMIDIDX\x00\x00"
# Megatron's DType codes in the .idx header.
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64,
           7: np.float32, 8: np.uint16}
_CODES = {np.dtype(t): code for code, t in _DTYPES.items()}
# Bump when the cached indices of the same inputs would change.
_INDEX_VERSION = 1
# A sample's tokens past `seq_len`: Megatron's add_extra_token_to_sequence, so that
# the targets are the inputs shifted by one.
EXTRA = 1


def prefix_of(path) -> str | None:
  """The dataset prefix of PREFIX, PREFIX.bin or PREFIX.idx when both files exist."""
  path = str(path)
  for suffix in (".bin", ".idx"):
    if path.endswith(suffix):
      path = path[:-len(suffix)]
  if os.path.isfile(path + ".bin") and os.path.isfile(path + ".idx"):
    return path
  return None


def write(prefix, documents, dtype=np.int32) -> None:
  """Writes `documents`, each a sequence of token ids, as PREFIX.bin and
  PREFIX.idx: one sequence per document, as Megatron's IndexedDatasetBuilder
  writes them."""
  dtype = np.dtype(dtype)
  lengths = np.asarray([len(d) for d in documents], np.int32)
  pointers = np.zeros(len(documents), np.int64)
  pointers[1:] = np.cumsum(lengths[:-1].astype(np.int64)) * dtype.itemsize
  with open(f"{prefix}.bin", "wb") as f:
    for doc in documents:
      f.write(np.asarray(doc, dtype).tobytes())
  with open(f"{prefix}.idx", "wb") as f:
    f.write(_HEADER)
    f.write(struct.pack("<Q", 1))
    f.write(struct.pack("<B", _CODES[dtype]))
    f.write(struct.pack("<Q", len(documents)))
    f.write(struct.pack("<Q", len(documents) + 1))
    f.write(lengths.tobytes())
    f.write(pointers.tobytes())
    f.write(np.arange(len(documents) + 1, dtype=np.int64).tobytes())


class IndexedDataset:
  """A Megatron indexed dataset: its index read, its tokens mapped."""

  def __init__(self, prefix):
    self.prefix = str(prefix)
    with open(self.prefix + ".idx", "rb") as f:
      if f.read(len(_HEADER)) != _HEADER:
        raise ValueError(f"{self.prefix}.idx is not a Megatron index")
      version, = struct.unpack("<Q", f.read(8))
      if version != 1:
        raise ValueError(f"{self.prefix}.idx has version {version}, not 1")
      code, = struct.unpack("<B", f.read(1))
      self.dtype = np.dtype(_DTYPES[code])
      count, = struct.unpack("<Q", f.read(8))
      document_count, = struct.unpack("<Q", f.read(8))
      offset = f.tell()
    index = np.memmap(self.prefix + ".idx", mode="r", dtype=np.uint8)
    self.sizes = np.frombuffer(index, np.int32, count, offset)
    self.pointers = np.frombuffer(index, np.int64, count, offset + 4 * count)
    self.document_indices = np.frombuffer(index, np.int64, document_count, offset + 12 * count)
    if self.document_indices[-1] != count:
      raise ValueError(f"{self.prefix}.idx: document indices end at "
                       f"{self.document_indices[-1]}, not at its {count} sequences")
    self.tokens = np.memmap(self.prefix + ".bin", mode="r", dtype=self.dtype)

  def __len__(self) -> int:
    return len(self.sizes)

  @property
  def num_tokens(self) -> int:
    return int(self.sizes.sum(dtype=np.int64))

  def get(self, sequence: int, offset: int = 0, length: int | None = None) -> np.ndarray:
    """Tokens of `sequence` from `offset`: `length` of them, or all to its end."""
    if length is None:
      length = int(self.sizes[sequence]) - offset
    start = int(self.pointers[sequence]) // self.dtype.itemsize + offset
    return self.tokens[start:start + length]


def num_epochs(tokens_per_epoch: int, num_samples: int | None, seq_len: int) -> int:
  """Epochs of the documents that `num_samples` samples need (1 for None)."""
  if num_samples is None:
    return 1
  requested = num_samples * seq_len + EXTRA
  return max(1, -(-requested // tokens_per_epoch))


def build_document_index(documents: np.ndarray, epochs: int, rng: np.random.RandomState,
                         separate_final_epoch: bool) -> np.ndarray:
  """Megatron's `_build_document_index`: the documents repeated `epochs`
  times and shuffled, the last epoch shuffled apart if `separate_final_epoch`."""
  if not separate_final_epoch or epochs == 1:
    index = np.tile(np.asarray(documents, np.int32), epochs)
    rng.shuffle(index)
    return index
  first = build_document_index(documents, epochs - 1, rng, False)
  last = build_document_index(documents, 1, rng, False)
  return np.concatenate((first, last))


def build_sample_index(sizes: np.ndarray, document_index: np.ndarray, seq_len: int,
                       epochs: int, tokens_per_epoch: int, drop_last: bool = True,
                       chunk: int = 1 << 24) -> np.ndarray:
  """Megatron's `helpers.build_sample_idx`: row k is the position in
  `document_index` and the token offset where sample k starts.

  Sample k covers the `seq_len + 1` tokens from position `k * seq_len` of the
  documents laid end to end in `document_index` order. Row k names the
  document holding that position (row 0 is (0, 0)). Megatron walks the
  documents one by one in C++; this finds each position by binary search
  over cumulative lengths, a chunk of documents at a time, with the same result.
  """
  total = epochs * tokens_per_epoch
  if drop_last:
    num_samples = (total - EXTRA) // seq_len
  else:
    # Megatron rounds up in single precision.
    num_samples = int(np.ceil(np.float32(total - EXTRA) / np.float32(seq_len)))
  index = np.empty((num_samples + 1, 2), np.int64)
  index[0] = 0
  k, before = 1, 0
  for start in range(0, len(document_index), chunk):
    if k > num_samples:
      break
    lengths = sizes[document_index[start:start + chunk]].astype(np.int64)
    ends = np.cumsum(lengths) + before
    last = min(num_samples, int(ends[-1] - 1) // seq_len)
    if last >= k:
      ks = np.arange(k, last + 1, dtype=np.int64)
      positions = ks * seq_len
      held_by = np.searchsorted(ends, positions, side="right")
      index[ks, 0] = start + held_by
      index[ks, 1] = positions - (ends[held_by] - lengths[held_by])
      k = last + 1
    before = int(ends[-1])
  if k <= num_samples:
    # Only without drop_last: the last sample runs past the last document.
    index[num_samples] = (len(document_index) - 1, sizes[document_index[-1]] - EXTRA)
  if index.max() < np.iinfo(np.int32).max:
    index = index.astype(np.int32)
  return index


def build_shuffle_index(num_samples: int, total: int, rng: np.random.RandomState) -> np.ndarray:
  """Megatron's `_build_shuffle_index`: [0, num_samples) shuffled, then
  [num_samples, total) shuffled apart."""
  dtype = np.uint32 if total < np.iinfo(np.uint32).max - 1 else np.int64
  first = np.arange(num_samples, dtype=dtype)
  rng.shuffle(first)
  if num_samples == total:
    return first
  last = np.arange(num_samples, total, dtype=dtype)
  rng.shuffle(last)
  return np.concatenate((first, last))


@dataclasses.dataclass(frozen=True)
class Indices:
  """`GPTDataset`'s three indices: sample i is row `shuffle_index[i]` of
  `sample_index`, which points into `document_index`."""

  document_index: np.ndarray
  sample_index: np.ndarray
  shuffle_index: np.ndarray


def build_indices(dataset: IndexedDataset, seq_len: int, num_samples: int | None = None,
                  seed: int | None = None, drop_last: bool = True) -> Indices:
  """The indices of `GPTDataset._build_document_sample_shuffle_indices` over
  every sequence of `dataset`, or file order without a seed."""
  documents = np.arange(len(dataset), dtype=np.int32)
  tokens_per_epoch = dataset.num_tokens
  epochs = num_epochs(tokens_per_epoch, num_samples, seq_len)
  separate_final_epoch = False
  if epochs > 1:
    before_final = ((epochs - 1) * tokens_per_epoch - EXTRA) // seq_len
    from_final = num_samples - before_final
    per_epoch = (tokens_per_epoch - EXTRA) // seq_len
    assert 0 <= from_final <= per_epoch + 1, (from_final, per_epoch)
    separate_final_epoch = from_final < int(0.80 * per_epoch)
  rng = None if seed is None else np.random.RandomState(seed)
  if rng is None:
    document_index = np.tile(documents, epochs)
  else:
    document_index = build_document_index(documents, epochs, rng, separate_final_epoch)
  sample_index = build_sample_index(dataset.sizes, document_index, seq_len, epochs,
                                    tokens_per_epoch, drop_last)
  count = len(sample_index) - 1
  if rng is None:
    shuffle_index = np.arange(count, dtype=np.int64)
  elif separate_final_epoch:
    shuffle_index = build_shuffle_index(before_final, count, rng)
  else:
    shuffle_index = build_shuffle_index(count, count, rng)
  return Indices(document_index, sample_index, shuffle_index)


def default_cache_dir() -> pathlib.Path:
  from jax_titan import paths  # pylint: disable=g-import-not-at-top

  return paths.work_dir() / "cache" / "megatron_indices"


def cached_indices(dataset: IndexedDataset, seq_len: int, num_samples: int | None = None,
                   seed: int | None = None, drop_last: bool = True,
                   cache_dir=None) -> Indices:
  """`build_indices`, kept as .npy files in `cache_dir` and memory-mapped
  from there on a later call with the same inputs."""
  key = {"version": _INDEX_VERSION, "dataset": os.path.basename(dataset.prefix),
         "sequences": len(dataset), "tokens": dataset.num_tokens, "seq_len": seq_len,
         "num_samples": num_samples, "seed": seed, "drop_last": drop_last}
  digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
  root = pathlib.Path(cache_dir or default_cache_dir())
  base = root / f"{key['dataset']}-{digest}"
  names = ("document_index", "sample_index", "shuffle_index")
  files = [base.with_name(f"{base.name}-{n}.npy") for n in names]
  if all(f.exists() for f in files):
    return Indices(*(np.load(f, mmap_mode="r") for f in files))
  indices = build_indices(dataset, seq_len, num_samples, seed, drop_last)
  root.mkdir(parents=True, exist_ok=True)
  for f, array in zip(files, (indices.document_index, indices.sample_index,
                              indices.shuffle_index)):
    tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as out:
      np.save(out, array)
    os.replace(tmp, f)
  base.with_name(f"{base.name}-description.json").write_text(json.dumps(key, indent=1) + "\n")
  return indices


class Samples:
  """The samples of `seq_len + 1` tokens that `indices` cut from `dataset`, in their order."""

  def __init__(self, dataset: IndexedDataset, indices: Indices, seq_len: int):
    self.dataset = dataset
    self.indices = indices
    self.seq_len = seq_len

  def __len__(self) -> int:
    return len(self.indices.shuffle_index)

  def __getitem__(self, i: int) -> np.ndarray:
    row = int(self.indices.shuffle_index[i])
    (first, offset), (last, end) = self.indices.sample_index[row:row + 2]
    docs = self.indices.document_index
    get = self.dataset.get
    if first == last:
      parts = [get(docs[first], int(offset), int(end - offset) + EXTRA)]
    else:
      parts = ([get(docs[first], int(offset))]
               + [get(docs[d]) for d in range(int(first) + 1, int(last))]
               + [get(docs[last], 0, int(end) + EXTRA)])
    tokens = np.concatenate(parts)
    if len(tokens) != self.seq_len + EXTRA:
      raise ValueError(f"sample {i} has {len(tokens)} tokens, not {self.seq_len + EXTRA}: "
                       "only the last sample of a dataset built without drop_last is short")
    return tokens

  def batch(self, start: int, count: int) -> np.ndarray:
    """Samples `start` to `start + count - 1` as int32 [count, seq_len + 1]."""
    return np.stack([self[i] for i in range(start, start + count)]).astype(np.int32)


class Sampler:
  """Global batches of a Megatron dataset in the reference's order: batch k is
  samples `k * batch` to `(k + 1) * batch - 1` of the indices that `seed` and
  `num_samples` build. `step` is the next batch; `state()` restores it."""

  def __init__(self, prefix, seq_len: int, num_samples: int, seed: int, step: int = 0,
               cache_dir=None):
    self.dataset = IndexedDataset(prefix)
    self.seed = seed
    self.samples = Samples(
        self.dataset, cached_indices(self.dataset, seq_len, num_samples, seed,
                                     cache_dir=cache_dir), seq_len)
    self.step = step
    self.num_tokens = self.dataset.num_tokens

  def next(self, batch: int) -> np.ndarray:
    start = self.step * batch
    if start + batch > len(self.samples):
      raise IndexError(f"batch {self.step} needs samples up to {start + batch}; the indices "
                       f"hold {len(self.samples)}")
    out = self.samples.batch(start, batch)
    self.step += 1
    return out

  def state(self) -> dict:
    return {"seed": self.seed, "step": self.step}


def validation_samples(prefix, seq_len: int, count: int) -> Samples:
  """The samples of `seq_len + 1` tokens in file order, unshuffled, checked to hold
  `count`; this reads the index alone."""
  dataset = IndexedDataset(prefix)
  samples = Samples(dataset, build_indices(dataset, seq_len), seq_len)
  if count > len(samples):
    raise ValueError(f"{prefix} holds {len(samples)} samples of {seq_len + EXTRA} tokens, "
                     f"not {count}")
  return samples


def validation(prefix, seq_len: int, count: int) -> np.ndarray:
  """The first `count` samples of `seq_len + 1` tokens in file order, unshuffled."""
  return validation_samples(prefix, seq_len, count).batch(0, count)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
  sub = parser.add_subparsers(dest="command", required=True)
  info = sub.add_parser("info", help="Documents, tokens and samples of a dataset.")
  info.add_argument("prefix")
  info.add_argument("--seq_len", type=int, default=8192)
  index = sub.add_parser("index", help="Builds and caches a training order's indices.")
  index.add_argument("prefix")
  index.add_argument("--seq_len", type=int, default=8192)
  index.add_argument("--num_samples", type=int, required=True,
                     help="Samples the run may read: its maximum steps times the global batch.")
  index.add_argument("--seed", type=int, required=True)
  index.add_argument("--cache_dir", default="")
  args = parser.parse_args(argv)
  dataset = IndexedDataset(prefix_of(args.prefix) or args.prefix)
  if args.command == "info":
    per_epoch = (dataset.num_tokens - EXTRA) // args.seq_len
    print(f"{dataset.prefix}: {len(dataset):,} documents, {dataset.num_tokens:,} tokens of "
          f"{dataset.dtype}, {per_epoch:,} samples of {args.seq_len + EXTRA} tokens per epoch")
    return 0
  t0 = time.time()
  indices = cached_indices(dataset, args.seq_len, args.num_samples, args.seed,
                           cache_dir=args.cache_dir or None)
  print(f"{dataset.prefix}: {len(indices.shuffle_index):,} samples over "
        f"{len(indices.document_index) // len(dataset)} epochs of {len(dataset):,} documents "
        f"in {time.time() - t0:.1f} s")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
