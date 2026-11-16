"""The FlyDSL provider's host side: B's block shuffle and the tile choice."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_titan.ops.providers import flydsl


def test_shuffle_matches_flydsl_shuffle_weight():
  n, k = 32, 64
  b = np.arange(n * k, dtype=np.float32).reshape(n, k)
  # FlyDSL's tests/utils.py shuffle_weight(layout=(16, 16)) for 2-byte elements.
  ref = b.reshape(-1, n // 16, 16, k // 32, 4, 8).transpose(0, 1, 3, 4, 2, 5).reshape(n, k)
  assert np.array_equal(np.asarray(flydsl.shuffle(jnp.asarray(b))), ref)


def test_tiles_are_the_widest_that_divide_n_and_k():
  assert flydsl.tiles(4096, 4096) == (64, 256, 128)
  assert flydsl.tiles(4160, 4096) == (64, 64, 128)
  assert flydsl.tiles(4096, 4160, tile_m=128) == (128, 256, 64)
  with pytest.raises(ValueError):
    flydsl.tiles(4100, 4096)


def test_kernels_need_a_built_library(monkeypatch):
  monkeypatch.delenv(flydsl.LIB_ENV, raising=False)
  flydsl.kernels.cache_clear()
  with pytest.raises(RuntimeError, match=flydsl.LIB_ENV):
    flydsl.kernels()
