"""The losses that the models' last stages return."""

import jax
import jax.numpy as jnp


def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
  """The mean loss of the next tokens `targets` over all of them."""
  logits = logits.astype(jnp.float32)
  logz = jax.nn.logsumexp(logits, axis=-1)
  gold = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
  return jnp.mean(logz - gold)


def masked_cross_entropy(
    logits: jax.Array, targets: jax.Array, microbatches: int = 1) -> jax.Array:
  """The mean loss over the targets that are not negative.

  With `microbatches`, the sequences are that many microbatches of equal size
  one after another, and the result is the mean of their mean losses: what a
  pipeline step gives for them, which differs from the mean over all targets
  when microbatches hold different numbers of targets.
  """
  logits = logits.astype(jnp.float32)
  keep = targets >= 0
  gold = jnp.take_along_axis(logits, jnp.maximum(targets, 0)[..., None], axis=-1)[..., 0]
  loss = (jax.nn.logsumexp(logits, axis=-1) - gold) * keep
  if microbatches == 1:
    return jnp.sum(loss) / jnp.maximum(jnp.sum(keep), 1)
  loss, keep = loss.reshape(microbatches, -1), keep.reshape(microbatches, -1)
  return jnp.mean(jnp.sum(loss, axis=-1) / jnp.maximum(jnp.sum(keep, axis=-1), 1))


def mean_squared_error(pred: jax.Array, target: jax.Array) -> jax.Array:
  """The mean squared error of `pred` to `target`, taken in fp32."""
  return jnp.mean((target.astype(jnp.float32) - pred.astype(jnp.float32)) ** 2)
