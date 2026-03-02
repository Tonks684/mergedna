import torch
import torch.nn.functional as F
import pytest

import mergedna.flash_attention_patch as fa_patch
from mergedna.flash_attention_patch import flash_attn_func


def _sdpa_reference(q, k, v, *, causal: bool, window_size):
    """
    Reference implementation using torch SDPA directly.
    q,k,v input in (B,T,H,Dh) like our patch expects.
    """
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,T,Dh)
    _, _, Tq, _ = qh.shape
    Tk = kh.size(2)

    if window_size == (-1, -1) and Tq == Tk:
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
        return out.transpose(1, 2)

    left, right = window_size
    row = (Tk - Tq) + torch.arange(Tq, device=qh.device)[:, None]
    col = torch.arange(Tk, device=qh.device)[None, :]

    mask = torch.ones((Tq, Tk), dtype=torch.bool, device=qh.device)
    if causal:
        mask &= (col <= row)
    if left >= 0:
        mask &= ((row - col) <= left)
    if right >= 0:
        mask &= ((col - row) <= right)

    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    return out.transpose(1, 2)


def test_attention_fallback_forced_sdpa_full_matches_reference(monkeypatch):
    # Force the fallback path regardless of whether nanochat is importable.
    monkeypatch.setattr(fa_patch, "_nanochat_flash", None)

    torch.manual_seed(0)
    B, T, H, Dh = 2, 16, 4, 8
    q = torch.randn(B, T, H, Dh)
    k = torch.randn(B, T, H, Dh)
    v = torch.randn(B, T, H, Dh)

    out = flash_attn_func(q, k, v, causal=False, window_size=(-1, -1))
    ref = _sdpa_reference(q, k, v, causal=False, window_size=(-1, -1))

    assert out.shape == (B, T, H, Dh)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_attention_fallback_forced_sdpa_local_window_matches_reference(monkeypatch):
    monkeypatch.setattr(fa_patch, "_nanochat_flash", None)

    torch.manual_seed(0)
    B, T, H, Dh = 2, 32, 4, 8
    q = torch.randn(B, T, H, Dh)
    k = torch.randn(B, T, H, Dh)
    v = torch.randn(B, T, H, Dh)

    out = flash_attn_func(q, k, v, causal=False, window_size=(4, 4))
    ref = _sdpa_reference(q, k, v, causal=False, window_size=(4, 4))

    assert out.shape == (B, T, H, Dh)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_attention_fallback_forced_sdpa_causal_matches_reference(monkeypatch):
    monkeypatch.setattr(fa_patch, "_nanochat_flash", None)

    torch.manual_seed(0)
    B, T, H, Dh = 2, 16, 4, 8
    q = torch.randn(B, T, H, Dh)
    k = torch.randn(B, T, H, Dh)
    v = torch.randn(B, T, H, Dh)

    out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
    ref = _sdpa_reference(q, k, v, causal=True, window_size=(-1, -1))

    assert out.shape == (B, T, H, Dh)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_attention_fallback_backward_pass(monkeypatch):
    monkeypatch.setattr(fa_patch, "_nanochat_flash", None)

    torch.manual_seed(0)
    B, T, H, Dh = 2, 16, 4, 8
    q = torch.randn(B, T, H, Dh, requires_grad=True)
    k = torch.randn(B, T, H, Dh, requires_grad=True)
    v = torch.randn(B, T, H, Dh, requires_grad=True)

    out = flash_attn_func(q, k, v, causal=False, window_size=(-1, -1))
    loss = out.pow(2).mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None