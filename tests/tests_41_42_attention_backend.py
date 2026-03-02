import torch
import torch.nn.functional as F

import mergedna.flash_attention_patch as fa_patch
from mergedna.flash_attention_patch import flash_attn_func


def _naive_attention_reference(q, k, v, *, causal: bool, window_size):
    """
    Ground-truth reference using explicit attention weights + masking.
    q,k,v: (B,T,H,Dh) -> out (B,T,H,Dh)
    """
    B, Tq, H, Dh = q.shape
    Tk = k.size(1)

    # (B,H,T,D)
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)

    # scores: (B,H,Tq,Tk)
    scores = torch.einsum("bhtd,bhsd->bhts", qh, kh) / (Dh**0.5)

    # build boolean allow-mask matching the patch semantics
    left, right = window_size
    row = (Tk - Tq) + torch.arange(Tq, device=q.device)[:, None]  # (Tq,1)
    col = torch.arange(Tk, device=q.device)[None, :]              # (1,Tk)

    allow = torch.ones((Tq, Tk), dtype=torch.bool, device=q.device)
    if causal:
        allow &= (col <= row)
    if left >= 0:
        allow &= ((row - col) <= left)
    if right >= 0:
        allow &= ((col - row) <= right)

    # apply allow-mask: disallow -> -inf
    scores = scores.masked_fill(~allow[None, None, :, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bhsd->bhtd", attn, vh)
    return out.transpose(1, 2)  # (B,Tq,H,Dh)


def test_flash_attn_patch_forced_sdpa_matches_naive_noncausal_full(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(fa_patch, "_nanochat_flash", None)

    B, T, H, Dh = 2, 32, 4, 8
    q = torch.randn(B, T, H, Dh)
    k = torch.randn(B, T, H, Dh)
    v = torch.randn(B, T, H, Dh)

    out = flash_attn_func(q, k, v, causal=False, window_size=(-1, -1))
    ref = _naive_attention_reference(q, k, v, causal=False, window_size=(-1, -1))

    assert out.shape == (B, T, H, Dh)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_flash_attn_patch_forced_sdpa_matches_naive_causal_full(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(fa_patch, "_nanochat_flash", None)

    B, T, H, Dh = 2, 32, 4, 8
    q = torch.randn(B, T, H, Dh)
    k = torch.randn(B, T, H, Dh)
    v = torch.randn(B, T, H, Dh)

    out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
    ref = _naive_attention_reference(q, k, v, causal=True, window_size=(-1, -1))

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_flash_attn_patch_forced_sdpa_matches_naive_local_window_noncausal(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(fa_patch, "_nanochat_flash", None)

    B, T, H, Dh = 2, 64, 4, 8
    q = torch.randn(B, T, H, Dh)
    k = torch.randn(B, T, H, Dh)
    v = torch.randn(B, T, H, Dh)

    out = flash_attn_func(q, k, v, causal=False, window_size=(8, 8))
    ref = _naive_attention_reference(q, k, v, causal=False, window_size=(8, 8))

    assert out.shape == (B, T, H, Dh)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_flash_attn_patch_backward_pass(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(fa_patch, "_nanochat_flash", None)

    B, T, H, Dh = 2, 32, 4, 8
    q = torch.randn(B, T, H, Dh, requires_grad=True)
    k = torch.randn(B, T, H, Dh, requires_grad=True)
    v = torch.randn(B, T, H, Dh, requires_grad=True)

    out = flash_attn_func(q, k, v, causal=False, window_size=(8, 8))
    loss = out.pow(2).mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()