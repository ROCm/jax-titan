"""AITER's Triton BF16 GEMM, called with constant sizes and strides.

jax-triton passes scalar arguments at run time without divisibility hints, so a
GEMM that takes its strides as arguments cannot prove its loads contiguous.
This kernel takes the sizes and strides as constexprs and calls AITER's GEMM
with them, so the backward GEMMs can read the forward's operands transposed.
"""

import triton
import triton.language as tl

from jax_titan.ops.providers import _kernel_sources

_a16w16_kernel = _kernel_sources.aiter_kernels("gemm.basic.gemm_a16w16")._gemm_a16_w16_kernel.fn


@triton.jit
def a16w16(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr,
           stride_bn: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
           BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr,
           EVEN_MN: tl.constexpr, cache_modifier: tl.constexpr):
  """c [M, N] row-major = a [M, K] times b [K, N], with a and b at the given strides."""
  # The GEMM assumes every stride positive; with one K split the split stride goes unused,
  # but a constant 0 there would make that assumption false.
  _a16w16_kernel(a_ptr, b_ptr, None, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                 M * N, N, 1, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, 1, K, EVEN_K,
                 EVEN_MN, cache_modifier, None, False, False, False)
