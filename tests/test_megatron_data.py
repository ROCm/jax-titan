"""Megatron indexed datasets and GPTDataset's sample order (jax_titan.data.megatron).

The reference for the indices is a line-by-line port of Megatron-core
core_r0.11.0: `helpers.cpp: build_sample_idx` and the index functions of
`gpt_dataset.py` (`_get_num_epochs`, `_build_document_sample_shuffle_indices`,
`_build_document_index`, `_build_shuffle_index`), kept here as plain loops.
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

from jax_titan.data import megatron
from jax_titan.observability import records

REPO = pathlib.Path(__file__).resolve().parents[1]


def _megatron_sample_idx(sizes, document_idx, seq_length, num_epochs, tokens_per_epoch,
                         drop_last=True, extra=1):
  """helpers.cpp build_sample_idx, line by line."""
  if drop_last:
    num_samples = (num_epochs * tokens_per_epoch - extra) // seq_length
  else:
    num_samples = int(np.ceil(np.float32(num_epochs * tokens_per_epoch - extra)
                              / np.float32(seq_length)))
  sample_idx = np.zeros((num_samples + 1, 2), np.int64)
  document_idx_index = 0
  doc_offset = 0
  sample_idx[0] = (document_idx_index, doc_offset)
  sample_idx_index = 1
  while sample_idx_index <= num_samples:
    remaining_seq_length = seq_length + extra
    while remaining_seq_length != 0:
      document_index = document_idx[document_idx_index]
      document_length = int(sizes[document_index]) - doc_offset
      remaining_seq_length -= document_length
      if remaining_seq_length <= 0:
        doc_offset += remaining_seq_length + document_length - extra
        remaining_seq_length = 0
      else:
        if document_idx_index == len(document_idx) - 1:
          assert sample_idx_index == num_samples
          doc_offset = int(sizes[document_idx[document_idx_index]]) - extra
          break
        document_idx_index += 1
        doc_offset = 0
    sample_idx[sample_idx_index] = (document_idx_index, doc_offset)
    sample_idx_index += 1
  return sample_idx


def _megatron_indices(sizes, seq_length, num_samples, seed, extra=1):
  """GPTDataset._build_document_sample_shuffle_indices over every sequence."""
  indices = np.arange(len(sizes), dtype=np.int32)
  num_tokens_per_epoch = int(np.sum(sizes[indices]))
  num_epochs = 1
  num_tokens = num_tokens_per_epoch
  if num_samples is not None:
    num_tokens_requested = num_samples * seq_length + extra
    while num_tokens < num_tokens_requested:
      num_epochs += 1
      num_tokens += num_tokens_per_epoch
  if num_epochs == 1:
    separate_final_epoch = False
  else:
    num_samples_sans_final_epoch = (
        (num_epochs - 1) * num_tokens_per_epoch - extra) // seq_length
    num_samples_from_final_epoch = num_samples - num_samples_sans_final_epoch
    num_samples_per_epoch = (num_tokens_per_epoch - extra) // seq_length
    separate_final_epoch = num_samples_from_final_epoch < int(0.80 * num_samples_per_epoch)
  numpy_random_state = np.random.RandomState(seed)

  def build_document_index(documents, num_epochs, separate_final_epoch):
    if not separate_final_epoch or num_epochs == 1:
      document_index = np.mgrid[0:num_epochs, 0:len(documents)][1]
      document_index[:] = documents
      document_index = document_index.reshape(-1)
      document_index = document_index.astype(np.int32)
      numpy_random_state.shuffle(document_index)
      return document_index
    doc_idx_first = build_document_index(documents, num_epochs - 1, False)
    doc_idx_last = build_document_index(documents, 1, False)
    return np.concatenate((doc_idx_first, doc_idx_last))

  def build_shuffle_index(num_samples, total_size):
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
      dtype_ = np.int64
    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
      return shuffle_idx_first
    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)
    return np.concatenate((shuffle_idx_first, shuffle_idx_last))

  document_index = build_document_index(indices, num_epochs, separate_final_epoch)
  sample_index = _megatron_sample_idx(sizes, document_index, seq_length, num_epochs,
                                      num_tokens_per_epoch)
  if separate_final_epoch:
    shuffle_index = build_shuffle_index(num_samples_sans_final_epoch, sample_index.shape[0] - 1)
  else:
    shuffle_index = build_shuffle_index(sample_index.shape[0] - 1, sample_index.shape[0] - 1)
  return document_index, sample_index, shuffle_index, separate_final_epoch


def _documents(seed=0, count=40, longest=30, vocab=1000):
  rng = np.random.default_rng(seed)
  lengths = rng.integers(0, longest, count)
  lengths[[3, 17]] = 0  # empty documents happen; the window walk skips them
  return [rng.integers(0, vocab, n) for n in lengths]


def _dataset(tmp_path, documents=None, dtype=np.int32, name="ds"):
  documents = _documents() if documents is None else documents
  prefix = str(tmp_path / name)
  megatron.write(prefix, documents, dtype)
  return megatron.IndexedDataset(prefix), documents


def test_write_and_read(tmp_path):
  for dtype in (np.int32, np.uint16):
    ds, docs = _dataset(tmp_path, dtype=dtype, name=f"ds_{np.dtype(dtype).name}")
    assert ds.dtype == dtype and len(ds) == len(docs)
    assert ds.num_tokens == sum(len(d) for d in docs)
    np.testing.assert_array_equal(ds.sizes, [len(d) for d in docs])
    for i in (0, 3, 5, len(docs) - 1):
      np.testing.assert_array_equal(ds.get(i), docs[i])
    longest = int(np.argmax(ds.sizes))
    np.testing.assert_array_equal(ds.get(longest, 2, 4), docs[longest][2:6])
  assert megatron.prefix_of(tmp_path / "ds_int32.idx") == str(tmp_path / "ds_int32")
  assert megatron.prefix_of(tmp_path / "missing") is None


@pytest.mark.parametrize("seq_len", [3, 7, 16])
@pytest.mark.parametrize("drop_last", [True, False])
def test_sample_index_is_megatrons(seq_len, drop_last):
  sizes = np.asarray([len(d) for d in _documents()], np.int32)
  rng = np.random.RandomState(1)
  for epochs in (1, 3):
    document_index = np.tile(np.arange(len(sizes), dtype=np.int32), epochs)
    rng.shuffle(document_index)
    expected = _megatron_sample_idx(sizes, document_index, seq_len, epochs, int(sizes.sum()),
                                    drop_last)
    for chunk in (3, 1 << 24):
      got = megatron.build_sample_index(sizes, document_index, seq_len, epochs,
                                        int(sizes.sum()), drop_last, chunk=chunk)
      np.testing.assert_array_equal(got, expected)


def test_indices_are_megatrons(tmp_path):
  ds, _ = _dataset(tmp_path)
  per_epoch = (ds.num_tokens - 1) // 16
  separated = set()
  # One epoch, one epoch with fewer samples, three epochs with the last one kept
  # in the global shuffle, and three with it shuffled apart.
  for num_samples in (None, per_epoch // 2, 2 * per_epoch + per_epoch * 9 // 10,
                      2 * per_epoch + 1):
    for seed in (0, 1234):
      doc, sample, shuffle, separate = _megatron_indices(ds.sizes, 16, num_samples, seed)
      got = megatron.build_indices(ds, 16, num_samples, seed)
      np.testing.assert_array_equal(got.document_index, doc)
      np.testing.assert_array_equal(got.sample_index, sample)
      np.testing.assert_array_equal(got.shuffle_index, shuffle)
      separated.add(separate)
  assert separated == {True, False}


def test_samples_are_windows_of_the_document_order(tmp_path):
  ds, docs = _dataset(tmp_path)
  indices = megatron.build_indices(ds, 7, num_samples=3 * ((ds.num_tokens - 1) // 7), seed=5)
  flat = np.concatenate([docs[d] for d in indices.document_index])
  samples = megatron.Samples(ds, indices, 7)
  assert len(samples) == len(indices.shuffle_index)
  for i in range(len(samples)):
    k = int(indices.shuffle_index[i])
    np.testing.assert_array_equal(samples[i], flat[7 * k:7 * k + 8])


def test_one_epoch_reads_every_sample_once(tmp_path):
  ds, docs = _dataset(tmp_path)
  for seed in (None, 3):
    indices = megatron.build_indices(ds, 5, seed=seed)
    count = (ds.num_tokens - 1) // 5
    assert sorted(indices.shuffle_index.tolist()) == list(range(count))
    if seed is None:
      np.testing.assert_array_equal(indices.document_index, np.arange(len(docs)))
      np.testing.assert_array_equal(indices.shuffle_index, np.arange(count))
  shuffled = megatron.build_indices(ds, 5, seed=4).shuffle_index
  assert shuffled.tolist() != megatron.build_indices(ds, 5, seed=3).shuffle_index.tolist()


def test_sampler_batches_resume_and_cache(tmp_path):
  ds, _ = _dataset(tmp_path)
  cache = tmp_path / "cache"
  sampler = megatron.Sampler(ds.prefix, 5, num_samples=40, seed=9, cache_dir=cache)
  assert len(list(cache.glob("*.npy"))) == 3
  samples = megatron.Samples(ds, megatron.build_indices(ds, 5, 40, 9), 5)
  batches = [sampler.next(4) for _ in range(3)]
  for k, batch in enumerate(batches):
    assert batch.dtype == np.int32 and batch.shape == (4, 6)
    np.testing.assert_array_equal(batch, samples.batch(4 * k, 4))
  resumed = megatron.Sampler(ds.prefix, 5, num_samples=40, seed=9, step=2, cache_dir=cache)
  np.testing.assert_array_equal(resumed.next(4), batches[2])
  assert resumed.state() == {"seed": 9, "step": 3}
  with pytest.raises(IndexError, match="the indices hold"):
    megatron.Sampler(ds.prefix, 5, num_samples=40, seed=9, step=10**6, cache_dir=cache).next(4)


def test_validation_is_file_order(tmp_path):
  ds, docs = _dataset(tmp_path)
  flat = np.concatenate(docs)
  held_out = megatron.validation(ds.prefix, 6, 10)
  assert held_out.shape == (10, 7)
  for k in range(10):
    np.testing.assert_array_equal(held_out[k], flat[6 * k:6 * k + 7])
  with pytest.raises(ValueError, match="samples of 7 tokens"):
    megatron.validation(ds.prefix, 6, 10**6)


def test_trainer_reads_megatron_data(tmp_path):
  rng = np.random.default_rng(0)
  docs = [rng.integers(0, 1000, n) for n in rng.integers(1, 300, 400)]
  megatron.write(tmp_path / "train", docs)
  megatron.write(tmp_path / "valid", docs[:60])
  log = tmp_path / "train.jsonl"
  result = subprocess.run(
      [sys.executable, "-m", "jax_titan.trainer.train", "--platform=cpu",
       "--model=llama3-test", "--parallel=dp", "--seq_len=32", "--microbatches=2",
       "--microbatch_size=8", "--steps=3", "--warmup_steps=1", "--eval_every=3",
       "--eval_batches=2", f"--data={tmp_path / 'train'}", f"--eval_data={tmp_path / 'valid'}",
       f"--log_file={log}", "--step_timeout=0", "--seed=7"],
      capture_output=True, text=True, check=False, cwd=REPO,
      env={**__import__("os").environ, "JAX_TITAN_ROOT": str(tmp_path / "root")})
  assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
  assert "Megatron indexed dataset" in result.stdout
  assert [e["step"] for e in records.events(records.read(log), "eval")] == [3]
  assert list((tmp_path / "root" / "cache" / "megatron_indices").glob("train-*.npy"))
