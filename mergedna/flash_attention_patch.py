"""
Attention backend for MergeDNA.

Use's NanoChat's flash attention wrapper (FA2/FA3/SDPA dispatch).
Falls back to PyTorch scaled_dot_product_attention with optional causal and sliding-window masks.

q,k,v are expected in (B,T,H,Dh) layout to match NanoChat conventions.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

try:
    from nanochat.flash_attention import flash_attn as _nanochat_flash  # reuse FA3/SDPA dispatch
except Exception:
    _nanochat_flash = None

def _sdpa_bool_mask(Tq: int, Tk: int, *, causal: bool, window_size: tuple[int,int], device) -> torch.Tensor:
    """
    Returns a boolean mask (Tq, Tk) for PyTorch's scaled_dot_product_attention with the given causal and window_size settings.
    """
    left, right = window_size
    # When lengths differ, we want the mask to align the end of k with the end of q (like in sliding window attention).
    row = (Tk - Tq) + torch.arange(Tq, device=device)[:, None]
    col = torch.arange(Tk, device=device)[None, :]

    # Construct mask for allowed positions
    mask = torch.ones((Tq, Tk), dtype=torch.bool, device=device)
    # Causal: only allow attending to positions <= current position
    if causal:
        mask &= (col <= row)
    # Left window: only allow attending to positions within 'left' of the current position
    if left >= 0:
        mask &= ((row - col) <= left)
    # Right window: only allow attending to positions within 'right' of the current position
    if right >= 0:
        mask &= ((col - row) <= right)
    
    return mask

def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, window_size: tuple[int,int]) -> torch.Tensor:
    """
    q,k,v: (B,T,H,Dh) -> out (B,T,H,Dh)
    Supports:
      - causal=False full attention
      - window_size != (-1,-1) sliding window local attention
    """
    if _nanochat_flash is not None:
        # nanochat already passes through causal/window_size to FA3/SDPA
        return _nanochat_flash.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # fallback to torch SDPA
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,T,Dh)
    B, H, Tq, Dh = qh.shape
    Tk = kh.size(2)

    # fast path: no local mask and equal lengths
    if window_size == (-1, -1) and Tq == Tk:
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
        return out.transpose(1, 2)
    # otherwise, we need to construct a boolean mask for SDPA
    mask = _sdpa_bool_mask(Tq, Tk, causal=causal, window_size=window_size, device=qh.device)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    return out.transpose(1, 2)