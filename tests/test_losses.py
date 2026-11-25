import numpy as np
import pytest

from jax_titan.models import losses


def test_masked_cross_entropy_is_cross_entropy_when_no_target_is_negative():
  rng = np.random.default_rng(0)
  logits = rng.standard_normal((4, 6, 16)).astype(np.float32)
  targets = rng.integers(0, 16, (4, 6)).astype(np.int32)
  masked = float(losses.masked_cross_entropy(logits, targets))
  assert masked == pytest.approx(float(losses.cross_entropy(logits, targets)), rel=1e-6)
  # A negative target is left out of the mean.
  targets[0] = -1
  rest = float(losses.cross_entropy(logits[1:], targets[1:]))
  assert float(losses.masked_cross_entropy(logits, targets)) == pytest.approx(rest, rel=1e-6)
