"""Safetensors files read back, and weight-only FP8 weights come out scaled."""

import ml_dtypes
import numpy as np
import pytest

from jax_titan.models import pretrained


def test_reads_what_it_wrote(tmp_path):
  rng = np.random.default_rng(0)
  tensors = {
      "a.weight": rng.standard_normal((3, 5)).astype(np.float32),
      "b": rng.standard_normal(7).astype(ml_dtypes.bfloat16),
      "c": np.arange(6, dtype=np.int64).reshape(2, 3),
      "d": np.float32(2.5),
  }
  pretrained.write(tmp_path / "x.safetensors", tensors)
  for path in (tmp_path / "x.safetensors", tmp_path):
    assert pretrained.is_safetensors(path)
    read = pretrained.read(path)
    assert set(read) == set(tensors)
    for name, value in tensors.items():
      assert read[name].dtype == value.dtype and read[name].shape == value.shape
      np.testing.assert_array_equal(read[name], value)
  assert not pretrained.is_safetensors(tmp_path / "missing")


def test_reads_a_directory_of_files_as_the_library_does(tmp_path):
  library = pytest.importorskip("safetensors.numpy")
  rng = np.random.default_rng(0)
  files = {
      "model-00001-of-00002.safetensors": {"a": rng.standard_normal((3, 5)).astype(np.float32)},
      "model-00002-of-00002.safetensors": {"b": np.arange(6, dtype=np.int64).reshape(2, 3),
                                           "c": rng.standard_normal(4).astype(np.float16)},
  }
  for name, tensors in files.items():
    library.save_file(tensors, tmp_path / name, metadata={"format": "pt"})
  read = pretrained.read(tmp_path)
  assert set(read) == {"a", "b", "c"}
  for tensors in files.values():
    for name, value in tensors.items():
      assert read[name].dtype == value.dtype
      np.testing.assert_array_equal(read[name], value)

  # A file that `write` wrote reads back in the library, too.
  pretrained.write(tmp_path / "written.safetensors", {"d": read["a"]})
  np.testing.assert_array_equal(
      library.load_file(tmp_path / "written.safetensors")["d"], read["a"])
  # One name in two files leaves no way to tell which tensor is meant.
  pretrained.write(tmp_path / "copy.safetensors", {"a": read["a"]})
  with pytest.raises(ValueError, match="more than one file"):
    pretrained.read(tmp_path)


def test_weight_applies_row_scales(tmp_path):
  rng = np.random.default_rng(1)
  w = rng.standard_normal((4, 6)).astype(np.float32)
  scale = (np.abs(w).max(axis=1) / 448).astype(np.float32)
  quantized = (w / scale[:, None]).astype(ml_dtypes.float8_e4m3fn)
  pretrained.write(tmp_path / "x.safetensors", {
      "l.weight": quantized, "l.weight_scale": scale, "l.bias": w[0].astype(ml_dtypes.bfloat16)})
  tensors = pretrained.read(tmp_path)
  got = pretrained.weight(tensors, "l.weight")
  np.testing.assert_array_equal(got, quantized.astype(np.float32) * scale[:, None])
  # e4m3 keeps three bits of mantissa.
  np.testing.assert_allclose(got, w, rtol=2**-4, atol=1e-3)
  np.testing.assert_array_equal(pretrained.weight(tensors, "l.bias"),
                                tensors["l.bias"].astype(np.float32))
