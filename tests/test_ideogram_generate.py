"""The sampling schedule of Ideogram 4's generation, and pictures from a toy checkpoint."""

import math

import ideogram_checkpoint
import numpy as np
import pytest
from scipy import special

from jax_titan.models.ideogram import generate as ideogram_generate


@pytest.mark.parametrize("preset", sorted(ideogram_generate.PRESETS))
@pytest.mark.parametrize("resolution", [512, 1024])
def test_times_follow_the_logit_normal_schedule(preset, resolution):
  steps, guidance, mean, std = ideogram_generate.PRESETS[preset]
  assert len(guidance) == steps
  ts = ideogram_generate.times(steps, resolution, resolution, mean, std)
  assert ts.shape == (steps + 1,) and ts.dtype == np.float32
  assert np.all(np.diff(ts) < 0)
  # The noise level is the logistic of a normal quantile, whose mean grows with the pixels;
  # the ends, where the quantile is infinite, are clamped.
  mu = mean + 0.5 * math.log(resolution * resolution / 512**2)
  quantiles = special.ndtri(np.linspace(0, 1, steps + 1))
  expected = np.clip(1 - special.expit(mu + std * quantiles),
                     1 / (1 + math.exp(9.0)), 1 / (1 + math.exp(-7.5)))
  np.testing.assert_allclose(ts, expected, rtol=1e-6)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
  path = tmp_path_factory.mktemp("toy") / "checkpoint"
  ideogram_checkpoint.write(path)
  return ideogram_generate.Pipeline(path, log=lambda _: None)


def test_pipeline_generates_images(pipeline):
  prompts = ["a", '{"high_level_description": "a b c"}']
  images = pipeline.generate(prompts, resolution=32, preset="turbo")
  assert images.shape == (2, 32, 32, 3) and images.dtype == np.uint8
  np.testing.assert_array_equal(pipeline.generate(prompts, resolution=32, preset="turbo"), images)
  for other in ({"seed": 1}, {"preset": "default"}):
    kwargs = {"resolution": 32, "preset": "turbo"} | other
    assert not np.array_equal(pipeline.generate(prompts, **kwargs), images), other
  # The prompt reaches the picture.
  assert not np.array_equal(
      pipeline.generate(["b", prompts[1]], resolution=32, preset="turbo")[0], images[0])


def test_pipeline_steps_from_noise_to_data(pipeline, monkeypatch):
  calls = []

  def step(conditional, unconditional, z, features, ids, t, s, weight):
    calls.append((t, s, weight))
    return z

  monkeypatch.setattr(pipeline, "_step", step)
  pipeline.generate(["a"], resolution=32, preset="default")
  steps, guidance, mean, std = ideogram_generate.PRESETS["default"]
  ts = ideogram_generate.times(steps, 32, 32, mean, std)
  t, s, weight = (np.asarray(x) for x in zip(*calls))
  np.testing.assert_array_equal(t, ts[:0:-1])
  np.testing.assert_array_equal(s, ts[-2::-1])
  # The preset counts its weights from the last step back.
  np.testing.assert_array_equal(weight, guidance[::-1])
