"""The sampling schedule and the sequence layout of Ideogram 4's generation."""

import math
import statistics

import numpy as np

from jax_titan.models import ideogram_generate


def test_times_run_from_data_to_noise():
  steps, mean, std = 20, 0.0, 1.75
  ts = ideogram_generate.times(steps, 1024, 1024, mean, std)
  assert ts.shape == (steps + 1,) and ts.dtype == np.float32
  assert np.all(np.diff(ts) < 0)
  np.testing.assert_allclose(ts[0], 1 / (1 + math.exp(-7.5)), rtol=1e-6)
  np.testing.assert_allclose(ts[-1], 1 / (1 + math.exp(9.0)), rtol=1e-6)
  # Halfway, the noise level is the logistic of the mean, which is log 2 above
  # `mean` at four times the pixels of 512x512.
  assert statistics.NormalDist().inv_cdf(0.5) == 0
  halfway = 1 - 1 / (1 + math.exp(-(mean + math.log(2.0))))
  np.testing.assert_allclose(ts[steps // 2], halfway, rtol=1e-6)


def test_presets_weigh_every_step():
  for steps, guidance, _, _ in ideogram_generate.PRESETS.values():
    assert len(guidance) == steps
    assert guidance[0] == 3.0 and guidance[-1] == 7.0
