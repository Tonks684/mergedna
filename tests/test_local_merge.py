import torch
import pytest

from mergedna.local_merge import (
    LocalTokenMerger,
    LocalMergeConfig,
    unmerge_tokens,
    project_mask_to_base,
)

def _dense_S_from_starts_lens(starts: torch.Tensor, lens: torch.Tensor, N: int) -> torch.Tensor:
    """
    Build dense S for tiny tests only.
    starts, lens: (B,L) int64
    returns S: (B,L,N) bool where S[b,l,n]=True if n in span of token l
    """
    B, L = starts.shape
    n = torch.arange(N, device=starts.device)[None, None, :]  # (1,1,N)
    s = starts[:, :, None]  # (B,L,1)
    e = (starts + lens)[:, :, None]  # (B,L,1)
    return (n >= s) & (n < e)

def test_project_mask_to_base_matches_dense_S():
    # Hand-constructed segmentation: L=3 spans cover N=7
    # token 0 covers [0,2), token 1 covers [2,5), token 2 covers [5,7)
    B, L, N = 2, 3, 7
    lens = torch.tensor([[2, 3, 2],
                         [1, 4, 2]], dtype=torch.int64)
    starts = torch.cumsum(lens, dim=1) - lens

    # mask merged tokens (B,L)
    mask_L = torch.tensor([[True, False, True],
                           [False, True, False]], dtype=torch.bool)

    # expected base mask via dense S
    S = _dense_S_from_starts_lens(starts, lens, N)  # (B,L,N)
    expected = (S & mask_L[:, :, None]).any(dim=1)  # (B,N)

    got = project_mask_to_base(mask_L, lens, N)
    assert torch.equal(got, expected)

def test_unmerge_tokens_matches_dense_S_linear_operator():
    # If S is dense membership, unmerge is equivalent to multiplying S^T with token embeddings
    # under the "repeat token embedding across its span" rule.
    torch.manual_seed(0)

    B, L, D, N = 1, 4, 5, 10
    lens = torch.tensor([[2, 3, 1, 4]], dtype=torch.int64)
    starts = torch.cumsum(lens, dim=1) - lens
    z_L = torch.randn(B, L, D)

    # Dense S: (B,L,N)
    S = _dense_S_from_starts_lens(starts, lens, N).to(z_L.dtype)  # float
    # Expected z_N = for each n, pick the l whose span contains n and copy z_L[l]
    # This equals S^T @ z_L in a one-hot membership setting:
    # (B,N,L) @ (B,L,D) -> (B,N,D)
    expected = torch.matmul(S.transpose(1, 2), z_L)  # (B,N,D)

    got = unmerge_tokens(z_L, lens, N)
    assert torch.allclose(got, expected, atol=0, rtol=0)

def test_local_token_merger_returns_consistent_starts_and_lens_and_shapes():
    torch.manual_seed(0)

    B, T, D = 2, 32, 16
    target_len = 20
    cfg = LocalMergeConfig(window_size=8, group_dim=8)
    merger = LocalTokenMerger(d_model=D, cfg=cfg)

    z = torch.randn(B, T, D)
    token_lens = torch.ones(B, T, dtype=torch.int64)

    z_new, lens_new, starts_new = merger(z, token_lens, target_len)

    # shape checks
    assert z_new.shape == (B, target_len, D)
    assert lens_new.shape == (B, target_len)
    assert starts_new.shape == (B, target_len)

    # starts should be cumulative sum - lens (monotonic non-decreasing)
    for b in range(B):
        assert torch.equal(starts_new[b], torch.cumsum(lens_new[b], dim=0) - lens_new[b])

    # starts should be monotone increasing (strictly if all lens>0)
    assert torch.all(starts_new[:, 1:] >= starts_new[:, :-1])

    # Lens are positive integers
    assert torch.all(lens_new > 0)

def test_local_merge_unmerge_and_mask_projection_end_to_end_smoke():
    """
    End-to-end smoke:
      - merge -> (z_L, lens_L, starts_L)
      - unmerge -> z_N
      - project mask -> base mask
    Validates consistency and shapes.
    """
    torch.manual_seed(0)

    B, T, D = 1, 24, 12
    target_len = 16
    cfg = LocalMergeConfig(window_size=8, group_dim=8)
    merger = LocalTokenMerger(d_model=D, cfg=cfg)

    z = torch.randn(B, T, D)
    lens = torch.ones(B, T, dtype=torch.int64)

    z_L, lens_L, starts_L = merger(z, lens, target_len)

    # unmerge to base length T (treat T as N here)
    z_T = unmerge_tokens(z_L, lens_L, T)
    assert z_T.shape == (B, T, D)

    # mask projection preserves shape
    mask_L = torch.zeros(B, target_len, dtype=torch.bool)
    mask_L[:, ::2] = True
    mask_T = project_mask_to_base(mask_L, lens_L, T)
    assert mask_T.shape == (B, T)