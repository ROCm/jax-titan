"""The inter-wave gfx950 Gluon BF16 GEMM, called with constant sizes and strides.

jax-triton passes scalar arguments at run time without divisibility hints, so
the GEMM's direct-to-LDS loads cannot prove 16-byte alignment and do not
compile. This kernel takes the sizes as constexprs and calls the GEMM with
row-major strides derived from them.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from jax_titan.ops.providers import _kernel_sources

_inter_wave = _kernel_sources.gluon_gemm("a16w16", "inter_wave").a16w16_kernel
_intra_wave = _kernel_sources.gluon_gemm(
    "a16w16", "intra_wave"
).v9_beyond_hotloop


@gluon.jit
def a16w16_inter(
    a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr, WARPS_N: gl.constexpr, GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
):
  """c [M, N] = a [M, K] times b_t [N, K] transposed; all three row-major."""
  _inter_wave(a_ptr, b_ptr, c_ptr, M, N, K, K, 1, 1, K, N, 1, BLOCK_M, BLOCK_N,
              BLOCK_K, WARPS_M, WARPS_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)


@gluon.jit
def a16w16_intra(
    a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
):
  """The intra-wave v9 form of the same row-major NT GEMM."""
  _intra_wave(a_ptr, b_ptr, c_ptr, M, N, K, K, 1, 1, K, N, 1, BLOCK_M, BLOCK_N,
              BLOCK_K, GRID_MN, NUM_XCDS, GROUP_SIZE_M)


# The original provider name remains the inter-wave route.
a16w16 = a16w16_inter
