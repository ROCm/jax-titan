"""The trainer's learning-rate schedules (`--lr_schedule`).

`cosine` is optax's `warmup_cosine_decay_schedule`: from 0 at the first update
up to the peak over the warm-up, then a cosine down to the end value.

`nemo_cosine` is NeMo's `CosineAnnealing` (nemo/core/optim/lr_scheduler.py at
v2.1.0), which the MLPerf Llama 3.1 8B reference runs: update `s`, counted
from 0, takes `peak * (s + 1) / (W + 1)` while `s <= W`, so the first update
already moves the weights; then a cosine from the peak to the end value over
`decay_steps` updates, and the end value after them. Without a warm-up the
cosine starts at the first update.
"""

import jax.numpy as jnp
import optax


def cosine(peak: float, warmup_steps: int, decay_steps: int, end_value: float):
  return optax.warmup_cosine_decay_schedule(0.0, peak, warmup_steps, warmup_steps + decay_steps,
                                            end_value=end_value)


def nemo_cosine(peak: float, warmup_steps: int, decay_steps: int, end_value: float):
  def schedule(count):
    step = jnp.asarray(count, jnp.float32)
    progress = jnp.clip((step - warmup_steps) / decay_steps, 0.0, 1.0)
    decayed = end_value + (peak - end_value) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    if warmup_steps <= 0:
      return decayed
    return jnp.where(step <= warmup_steps, peak * (step + 1) / (warmup_steps + 1), decayed)

  return schedule


SCHEDULES = {"cosine": cosine, "nemo_cosine": nemo_cosine}
