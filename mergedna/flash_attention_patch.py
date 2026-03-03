"""
MergeDNA attention backend (Report §4.1 - §4.2).

What this module does
---------------------
MergeDNA needs *encoder-style* attention, not autoregressive decoding. Concretely:

  §4.1 Non-causal attention:
    - reconstruction-style passes (MTR, AMTM) require is_causal=False

  §4.2 Local-window attention:
    - Local Encoder / Local Decoder operate under a fixed sliding window (e.g. 16)

Rather than re-implementing attention kernels, I treat NanoChat as an execution layer:
  - If NanoChat's FlashAttention wrapper is available, I reuse its runtime dispatch
    (FA2/FA3/SDPA) while passing through `causal` and `window_size`.
  - Otherwise I fall back to PyTorch's scaled_dot_product_attention (SDPA), using:
      * is_causal for true causal masking
      * a boolean attention mask for sliding-window locality

Tensor layout conventions
-------------------------
NanoChat uses (B, T, H, Dh). PyTorch SDPA expects (B, H, T, Dh). I transpose
inside the fallback path only.

Design choices
--------------------
1) Use NanoChat when available:
   - preserves optimized kernels and whatever hardware dispatch NanoChat provides
   - keeps MergeDNA code small and focused on representation changes

2) Boolean SDPA mask fallback:
   - works on CPU/GPU without third-party attention packages
   - supports local windows even when FlashAttention isn't installed
   - makes behaviour testable against a reference implementation

3) "Right-aligned" row indices when Tq != Tk:
   - mirrors how attention should align when the query length differs from key length
     (e.g., incremental decoding or prefill-like shapes), and matches the earlier tests.
   - This keeps the mask consistent with the intent: query position t attends to the
     corresponding "current" key position near the end of the key sequence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional NanoChat infra import
#
# I attempt to reuse NanoChat's attention dispatch layer if present.
# This wrapper typically chooses the best available kernel at runtime:
#   - FlashAttention 2/3 if installed and supported
#   - otherwise torch SDPA
#
# Important:
#   - I import from nanochat_infra rather than nanochat to keep this repo minimal.
#   - If this import fails, I silently fall back to torch SDPA below.
# -----------------------------------------------------------------------------
try:
    from nanochat_infra.flash_attention import flash_attn as _nanochat_flash
except Exception:
    _nanochat_flash = None


def _sdpa_bool_mask(
    Tq: int,
    Tk: int,
    *,
    causal: bool,
    window_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Construct a boolean attention mask for torch.nn.functional.scaled_dot_product_attention.

    PyTorch SDPA supports `attn_mask` where:
      - boolean mask True  => position is allowed to attend
      - boolean mask False => position is masked out

    Args:
        Tq:
            Query length.
        Tk:
            Key/value length.
        causal:
            If True, disallow attending to "future" positions (col > row).
        window_size:
            (left, right) window constraints relative to the aligned diagonal.
            - (-1, -1) means "no window constraint" (full attention).
            - left >= 0 limits how far left I can attend: (row - col) <= left
            - right >= 0 limits how far right I can attend: (col - row) <= right
        device:
            Device for mask allocation.

    Alignment when Tq != Tk:
        I "right-align" the query positions to the key positions by defining:

            row = (Tk - Tq) + arange(Tq)

        This means query index 0 corresponds to key index (Tk - Tq), and query index
        Tq-1 corresponds to key index Tk-1. This matches typical alignment for
        prefill/continuation patterns and keeps locality constraints consistent.

    Returns:
        mask: (Tq, Tk) boolean, True for allowed attention positions.
    """
    left, right = window_size

    # row: (Tq,1) represents the key index aligned with each query position.
    row = (Tk - Tq) + torch.arange(Tq, device=device)[:, None]
    # col: (1,Tk) enumerates key indices.
    col = torch.arange(Tk, device=device)[None, :]

    # Start fully unmasked, then apply constraints.
    mask = torch.ones((Tq, Tk), dtype=torch.bool, device=device)

    # Causal constraint (if enabled): allow attending only to <= aligned position.
    if causal:
        mask &= (col <= row)

    # Sliding window constraints around the aligned "diagonal" position row.
    if left >= 0:
        mask &= ((row - col) <= left)   # limit how far into the past
    if right >= 0:
        mask &= ((col - row) <= right)  # limit how far into the future

    return mask


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
) -> torch.Tensor:
    """
    MergeDNA attention function, compatible with NanoChat attention conventions.

    Inputs:
        q, k, v: (B, T, H, Dh)  (NanoChat-style layout)
        causal:
            - True  => causal masking semantics (autoregressive)
            - False => non-causal semantics (encoder/reconstruction)  [MergeDNA §4.1]
        window_size:
            - (-1, -1) => full attention
            - (L, R)   => sliding-window local attention              [MergeDNA §4.2]

    Output:
        out: (B, T, H, Dh)

    Dispatch:
        - If NanoChat infra is available, delegate to its attention backend so that
          FlashAttention/SDPA selection happens there.
        - Otherwise, execute torch SDPA:
            * fast path for full attention with equal lengths
            * masked path for local windows and/or unequal lengths

    Note:
        This function is intentionally minimal: it is the glue between MergeDNA's
        architectural needs and the execution backend.
    """
    # Preferred path: reuse NanoChat's dispatch (FlashAttention/SDPA).
    if _nanochat_flash is not None:
        # NanoChat already supports `causal` and `window_size` arguments.
        # Keeping the same signature preserves drop-in compatibility with its blocks.
        return _nanochat_flash.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # -------------------------------------------------------------------------
    # Fallback path: torch SDPA
    #
    # torch SDPA expects (B,H,T,Dh), so I transpose.
    # -------------------------------------------------------------------------
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,T,Dh)
    _, _, Tq, _ = qh.shape
    Tk = kh.size(2)

    # Fast path: full attention and equal lengths -> use is_causal directly.
    # This avoids allocating a (Tq,Tk) mask.
    if window_size == (-1, -1) and Tq == Tk:
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
        return out.transpose(1, 2)  # back to (B,T,H,Dh)

    # Masked path:
    # - local window attention requires an explicit mask
    # - unequal lengths require explicit alignment logic
    mask = _sdpa_bool_mask(
        Tq,
        Tk,
        causal=causal,
        window_size=window_size,
        device=qh.device,
    )

    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    return out.transpose(1, 2)