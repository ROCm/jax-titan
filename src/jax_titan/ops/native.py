"""Reference operator implementations in plain JAX."""

import jax
import jax.numpy as jnp


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
  x32 = x.astype(jnp.float32)
  y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
  return (y * weight.astype(jnp.float32)).astype(x.dtype)


def attention(q: jax.Array, k: jax.Array, v: jax.Array, *, causal: bool = True,
              scale: float | None = None, mask: jax.Array | None = None) -> jax.Array:
  """Attention over [batch, seq, heads, head_dim]; k/v may have fewer heads.

  `mask` is boolean, [batch or 1, heads or 1, seq, seq], true where a query
  position attends to a key position.
  """
  return jax.nn.dot_product_attention(q, k, v, mask=mask, scale=scale, is_causal=causal)


def swiglu(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array
) -> jax.Array:
  return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


def linear(x: jax.Array, w: jax.Array) -> jax.Array:
  """x [..., in] times w [in, out]."""
  return x @ w


def rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
  """Rotary embedding on [batch, seq, heads, head_dim], split-halves layout.

  `cos` and `sin` are [seq, head_dim // 2], or [batch, seq, head_dim // 2]
  when positions differ between samples.
  """
  c = jnp.expand_dims(cos.astype(jnp.float32), -2)
  s = jnp.expand_dims(sin.astype(jnp.float32), -2)
  x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
  return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)
