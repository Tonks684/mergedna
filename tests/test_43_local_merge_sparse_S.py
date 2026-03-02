import torch

from mergedna.local_merge import (
    LocalTokenMerger,
    LocalMergeConfig,
    unmerge_tokens,
    project_mask_to_base,
)


def _dense_S_from_starts_lens(starts: torch.Tensor, lens: torch.Tensor, N: int) -> torch.Tensor:
    """
    Tiny-test-only dense S:
      S[b,l,n]=True iff n is in [start_l, start_l + len_l).
    """
    B, L = starts.shape
    n = torch.arange(N, device=starts.device)[None, None, :]  # (1,1,N)
    s = starts[:, :, None]
    e = (starts + lens)[:, :, None]
    return (n >= s) & (n < e)  # (B,L,N)


def _call_merger(merger, z, lens, target_len):
    """
    Support either:
      (z_L, lens_L) OR (z_L, lens_L, starts_L)
    """
    out = merger(z, lens, target_len)
    if isinstance(out, tuple) and len(out) == 2:
        z_L, lens_L = out
        starts_L = torch.cumsum(lens_L, dim=1) - lens_L
        return z_L, lens_L, starts_L
    if isinstance(out, tuple) and len(out) == 3:
        return out
    raise AssertionError(f"Unexpected merger return type: {type(out)} / {getattr(out, '__len__', None)}")


def test_project_mask_to_base_matches_dense_S():
    # Hand segmentation: spans cover N=7
    # token0 [0,2), token1 [2,5), token2 [5,7)
    B, L, N = 2, 3, 7
    lens = torch.tensor([[2, 3, 2],
                         [1, 4, 2]], dtype=torch.long)
    starts = torch.cumsum(lens, dim=1) - lens

    mask_L = torch.tensor([[True, False, True],
                           [False, True, False]], dtype=torch.bool)

    S = _dense_S_from_starts_lens(starts, lens, N)  # (B,L,N)
    expected = (S & mask_L[:, :, None]).any(dim=1)  # (B,N)

    got = project_mask_to_base(mask_L, lens, N)
    assert torch.equal(got, expected)


def test_unmerge_tokens_matches_dense_S_linear_operator():
    torch.manual_seed(0)

    B, L, D, N = 1, 4, 5, 10
    lens = torch.tensor([[2, 3, 1, 4]], dtype=torch.long)  # sums to 10
    starts = torch.cumsum(lens, dim=1) - lens
    z_L = torch.randn(B, L, D)

    S = _dense_S_from_starts_lens(starts, lens, N).to(z_L.dtype)  # (B,L,N) float
    expected = torch.matmul(S.transpose(1, 2), z_L)  # (B,N,D)

    got = unmerge_tokens(z_L, lens, N)
    assert torch.allclose(got, expected, atol=0, rtol=0)


def test_merger_outputs_valid_contiguous_segmentation_invariants():
    torch.manual_seed(0)

    B, N, D = 2, 32, 16
    target_len = 20
    cfg = LocalMergeConfig(window_size=8, group_dim=8)
    merger = LocalTokenMerger(d_model=D, cfg=cfg)

    z = torch.randn(B, N, D)
    token_lens = torch.ones(B, N, dtype=torch.long)

    z_L, lens_L, starts_L = _call_merger(merger, z, token_lens, target_len)

    assert z_L.shape == (B, target_len, D)
    assert lens_L.shape == (B, target_len)
    assert starts_L.shape == (B, target_len)

    assert lens_L.dtype == torch.long
    assert torch.all(lens_L > 0)

    # starts must be cumsum(lens)-lens
    expected_starts = torch.cumsum(lens_L, dim=1) - lens_L
    assert torch.equal(starts_L, expected_starts)

    # monotone
    assert torch.all(starts_L[:, 1:] >= starts_L[:, :-1])

    # coverage: merged spans should cover at least N bases (or exactly N depending on your implementation)
    assert torch.all(lens_L.sum(dim=1) >= N)


def test_merge_then_unmerge_piecewise_constant_over_spans():
    """
    Unmerge repeats token embeddings across their span.
    After unmerge, each span should be constant and equal to its token embedding.
    """
    torch.manual_seed(0)

    B, N, D = 1, 24, 12
    target_len = 16
    cfg = LocalMergeConfig(window_size=8, group_dim=8)
    merger = LocalTokenMerger(d_model=D, cfg=cfg)

    z = torch.randn(B, N, D)
    lens0 = torch.ones(B, N, dtype=torch.long)

    z_L, lens_L, starts_L = _call_merger(merger, z, lens0, target_len)
    z_N = unmerge_tokens(z_L, lens_L, N)

    # check a few spans (not all, to keep test fast)
    for l in [0, 3, 7, target_len - 1]:
        s = int(starts_L[0, l].item())
        e = min(s + int(lens_L[0, l].item()), N)
        if s >= N or e <= s:
            continue
        # span should equal z_L[:, l]
        span = z_N[0, s:e, :]
        tok = z_L[0, l, :].unsqueeze(0).expand_as(span)
        assert torch.allclose(span, tok, atol=0, rtol=0)