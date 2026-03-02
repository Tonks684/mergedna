import math
import torch
import pytest

import mergedna.flash_attention_patch as attn_backend


def _naive_attention(q, k, v, mask=None):
    """
    Naive attention for correctness reference.
    q,k,v: (B,H,Tq,D), (B,H,Tk,D), (B,H,Tk,D)
    mask:  (Tq,Tk) bool where True=keep
    """
    B, H, Tq, D = q.shape
    Tk = k.size(2)
    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,Tq,Tk)
    if mask is not None:
        # mask False -> -inf
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)  # (B,H,Tq,D)
    return out


def _make_qkv(B=2, H=3, T=8, D=4, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    return q, k, v


def test_backend_noncausal_full_attention_matches_naive(monkeypatch):
    # force SDPA fallback
    monkeypatch.setattr(attn_backend, "_nanochat_flash", None)

    q, k, v = _make_qkv(B=2, H=2, T=7, D=8)
    # backend expects (B,T,H,Dh)
    out = attn_backend.flash_attn_func(q, k, v, causal=False, window_size=(-1, -1))

    # naive expects (B,H,T,D)
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    expected = _naive_attention(qh, kh, vh, mask=None).transpose(1, 2)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_backend_causal_full_attention_matches_naive(monkeypatch):
    monkeypatch.setattr(attn_backend, "_nanochat_flash", None)

    q, k, v = _make_qkv(B=1, H=2, T=9, D=8)
    out = attn_backend.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    T = qh.size(2)
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=qh.device))
    expected = _naive_attention(qh, kh, vh, mask=causal_mask).transpose(1, 2)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_backend_local_window_matches_naive_mask(monkeypatch):
    monkeypatch.setattr(attn_backend, "_nanochat_flash", None)

    q, k, v = _make_qkv(B=2, H=2, T=10, D=8)
    window = (2, 3)  # left=2, right=3
    out = attn_backend.flash_attn_func(q, k, v, causal=False, window_size=window)

    # build expected mask using the module's own mask generator for consistency
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    Tq = qh.size(2)
    Tk = kh.size(2)

    mask = attn_backend._sdpa_bool_mask(Tq, Tk, causal=False, window_size=window, device=qh.device)
    expected = _naive_attention(qh, kh, vh, mask=mask).transpose(1, 2)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_backend_local_window_plus_causal_matches_naive(monkeypatch):
    monkeypatch.setattr(attn_backend, "_nanochat_flash", None)

    q, k, v = _make_qkv(B=1, H=2, T=12, D=8)
    window = (4, 0)  # only attend to up to 4 left, no right lookahead
    out = attn_backend.flash_attn_func(q, k, v, causal=True, window_size=window)

    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    Tq = qh.size(2)
    Tk = kh.size(2)
    mask = attn_backend._sdpa_bool_mask(Tq, Tk, causal=True, window_size=window, device=qh.device)
    expected = _naive_attention(qh, kh, vh, mask=mask).transpose(1, 2)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_sdpa_mask_is_boolean_and_correct_shape():
    Tq, Tk = 5, 7
    mask = attn_backend._sdpa_bool_mask(Tq, Tk, causal=False, window_size=(-1, -1), device="cpu")
    assert mask.dtype == torch.bool
    assert mask.shape == (Tq, Tk)

    # with full attention and no causality/windowing, all should be True
    assert mask.all().item() is True