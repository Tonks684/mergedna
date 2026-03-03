"""
nanochat_infra.flash_attention

Unified FlashAttention interface with automatic FA3/SDPA switching.

Goal
----
Expose an API compatible with FlashAttention-3 (FA3) while allowing the codebase
to run on *any* device (non-Hopper CUDA GPUs, CPU, MPS) by falling back to
PyTorch's scaled_dot_product_attention (SDPA).

This module exports a `flash_attn` object (SimpleNamespace) with:
  - flash_attn.flash_attn_func(q, k, v, causal=..., window_size=...)
  - flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, ...)

Shape conventions
-----------------
FA3-style:
  q, k, v: (B, T, H, Dh)

PyTorch SDPA expects:
  q, k, v: (B, H, T, Dh)

We transpose as needed in the fallback path.

Sliding-window conventions
--------------------------
window_size is a tuple: (left, right)
  - left  = number of tokens allowed to attend to on the left of each query
  - right = number of tokens allowed to attend to on the right of each query
  - -1 means "unlimited"

Note: The SDPA helper below currently implements left window constraints only.
This matches common "sliding window causal" behaviour used for efficient decoding.

Note: The infra wrapper is causal-first; MergeDNA uses its own non-causal attention for reconstruction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs (sm90)
# =============================================================================
def _load_flash_attention_3():
    """
    Try to load Flash Attention 3 (FA3).

    Design choice:
      - FA3 kernels are typically compiled for Hopper (sm90).
      - On other GPU architectures, or non-CUDA devices, we use SDPA.

    Returns:
      - An object exposing FA3-compatible call signatures if available
      - None otherwise
    """
    if not torch.cuda.is_available():
        return None

    try:
        major, _ = torch.cuda.get_device_capability()

        # FA3 kernels are (commonly) compiled for Hopper (sm90) only.
        # If running on other architectures, default to SDPA fallback.
        if major != 9:
            return None

        # Minor quality-of-life: disable HF progress bars if `kernels.get_kernel`
        # pulls artifacts via HF Hub.
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        from kernels import get_kernel
        return get_kernel("varunneal/flash-attention-3").flash_attn_interface

    except Exception:
        # Any import/load failure => safe fallback.
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# For debugging/tests: force a specific implementation.
#   - 'fa3'  => require FA3
#   - 'sdpa' => force SDPA fallback
#   - None   => auto
_override_impl: str | None = None


def _use_fa3() -> bool:
    """
    Decide whether to use FA3.

    Design choice:
      - Allow explicit overrides for tests / debugging.
      - Default to FA3 if it loaded successfully.
    """
    if _override_impl == "fa3":
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == "sdpa":
        return False
    return HAS_FA3


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding-window support.

    Args:
      q, k, v: (B, H, T, Dh) tensors (PyTorch SDPA layout)
      window_size: (left, right) tuple
      enable_gqa: whether to pass enable_gqa to SDPA (PyTorch supports it)

    Returns:
      y: (B, H, Tq, Dh)

    Why a custom helper?
      - PyTorch SDPA supports `is_causal=True`, but when decoding with a KV-cache,
        query length (Tq) may differ from key length (Tk). In that case, `is_causal`
        no longer aligns with "absolute positions in the cache", so we build an
        explicit boolean mask in the general case.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    left = window_size[0]  # we currently use only the left window constraint

    # -------------------------------------------------------------------------
    # Fast path: full causal attention, same length
    # -------------------------------------------------------------------------
    # If window is "unlimited" and lengths match, we can use SDPA's is_causal flag.
    if (left < 0 or left >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,         
            enable_gqa=enable_gqa,
        )

    # -------------------------------------------------------------------------
    # Single-token query (typical incremental decoding step)
    # -------------------------------------------------------------------------
    # When Tq == 1, we can slice the KV-cache rather than construct a full mask.
    # This avoids allocating (1 x Tk) boolean masks repeatedly.
    if Tq == 1:
        if 0 <= left < Tk:
            # Keep last (left + 1) keys (the token itself + left context).
            start = max(0, Tk - (left + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]

        # For Tq == 1 after slicing, causal masking is unnecessary because the
        # KV-cache already represents "past" tokens only.
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
            enable_gqa=enable_gqa,
        )

    # -------------------------------------------------------------------------
    # General case: explicit boolean attention mask
    # -------------------------------------------------------------------------
    # This covers:
    #  - sliding-window training (if you choose to use it)
    #  - "chunk inference" where Tq != Tk (queries are aligned to the *end* of cache)
    #
    # We align query row indices to cache positions:
    #   row_idx[t] = (Tk - Tq) + t
    # so that the last query aligns with the last key.
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)  # (Tq, 1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)              # (1, Tk)

    # Base causal mask: allow attending to keys at positions <= the query's aligned position.
    mask = col_idx <= row_idx

    # Sliding window (left): restrict to keys within `left` positions behind.
    if 0 <= left < Tk:
        mask = mask & ((row_idx - col_idx) <= left)

    # Note: right-window is not implemented in this SDPA helper. If you ever need
    # bidirectional local attention (left+right), add:
    #   if 0 <= right: mask &= ((col_idx - row_idx) <= right)

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        enable_gqa=enable_gqa,
    )


# =============================================================================
# Public API: FA3-compatible functions
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Attention for training / reconstruction (no KV cache).

    Args:
      q, k, v: (B, T, H, Dh)
      causal: bool, whether attention should be causal
      window_size: (left, right) sliding window; -1 means unlimited.

    Returns:
      y: (B, T, H, Dh)

    Design choice:
      - Keep the signature identical to FA3 so MergeDNA / NanoChat-style callers
        can switch implementations transparently.
    """
    if _use_fa3():
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose to (B, H, T, Dh)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)

    # PyTorch SDPA supports grouped-query attention (GQA) via enable_gqa flag.
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)

    # NOTE: `_sdpa_attention` currently assumes causal semantics in its fast path.
    # If you rely on `causal=False` (MergeDNA reconstruction), make sure your call
    # path exercises the explicit-mask branch or adjust `_sdpa_attention`.
    y = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, Dh)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=None,
    causal=False,
    window_size=(-1, -1),
):
    """
    Attention for inference / decoding with a KV cache.

    Behaviour:
      - FA3 updates k_cache/v_cache in-place.
      - SDPA fallback updates k_cache/v_cache in-place as well to match FA3 semantics.

    Args:
      q: (B, T_new, H, Dh) queries for new tokens
      k_cache, v_cache: (B, T_max, H_kv, Dh) preallocated cache
      k, v: (B, T_new, H_kv, Dh) new KV to insert at cache_seqlens position
      cache_seqlens: (B,) current cache position (int32/long)
      causal: bool
      window_size: (left, right)

    Returns:
      y: (B, T_new, H, Dh)
    """
    if _use_fa3():
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size,
        )

    # -----------------------
    # SDPA fallback KV logic
    # -----------------------
    B, T_new, H, Dh = q.shape

    # This assumes all batch elements share the same cache position.
    # If you ever support per-example cache positions, you need a loop or scatter.
    pos = cache_seqlens[0].item()

    # Insert new KV into cache (in-place), matching FA3 behaviour.
    if k is not None and v is not None:
        k_cache[:, pos:pos + T_new, :, :] = k
        v_cache[:, pos:pos + T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]  # (B, Tk, H_kv, Dh)
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout
    q_sdpa = q.transpose(1, 2)          # (B, H, T_new, Dh)
    k_sdpa = k_full.transpose(1, 2)     # (B, H_kv, Tk, Dh)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)


# =============================================================================
# Export: `flash_attn` module interface (drop-in FA3 replacement)
# =============================================================================
from types import SimpleNamespace

flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)