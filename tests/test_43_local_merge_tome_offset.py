# tests/test_local_merge_tome_offset.py

import torch
import torch.nn as nn
import pytest

from mergedna.local_merge import LocalTokenMerger, LocalMergeConfig


def _make_merger_identity_group(d_model: int, window_size: int = 64) -> LocalTokenMerger:
    """
    Create a LocalTokenMerger but patch `group` to Identity so similarity is computed
    directly from z (deterministic + controllable).
    """
    cfg = LocalMergeConfig(window_size=window_size, group_dim=d_model)
    merger = LocalTokenMerger(d_model=d_model, cfg=cfg)
    merger.group = nn.Identity()
    return merger


def test_tome_offset0_prefers_even_left_edges_and_is_collision_free():
    """
    With offset=0, allowed adjacent pairs are (0,1), (2,3), (4,5), ...
    We set similarities so (0,1) and (2,3) are highest, and (4,5) is low,
    then require 2 merges (T=6 -> target=4). Expected merges: (0,1) and (2,3).
    """
    torch.manual_seed(0)

    # Tokens: 6, d_model=1 so we can easily inspect values after weighted averaging
    # We'll still control similarity via 2D embeddings then project to 1D values for inspection.
    # But since we patched group=Identity and normalize, we should keep d_model=2.
    d_model = 2
    merger = _make_merger_identity_group(d_model=d_model, window_size=64)

    # Construct embeddings so:
    # sim(0,1) high, sim(2,3) high, sim(4,5) low.
    # Use nearly identical unit vectors for high sim; orthogonal for low sim.
    z = torch.tensor(
        [[[1.0, 0.0],   # 0
          [0.99, 0.01], # 1  ~ similar to 0
          [0.0, 1.0],   # 2
          [0.01, 0.99], # 3  ~ similar to 2
          [1.0, 0.0],   # 4
          [0.0, 1.0]]], # 5  ~ dissimilar to 4
        dtype=torch.float32
    )  # (B=1, T=6, D=2)

    token_lens = torch.ones((1, 6), dtype=torch.long)

    # We need exactly 2 merges to go from 6 -> 4
    target_len = 4

    # offset=0 should pick even-left edges; with budget=2 it should pick (0,1) and (2,3)
    z_new, lens_new, starts_new = merger(z, token_lens, target_len=target_len, offset=0)

    assert z_new.shape == (1, 4, d_model)
    assert lens_new.shape == (1, 4)
    assert starts_new.shape == (1, 4)

    # If merges are (0,1) and (2,3), surviving token indices are [0,2,4,5]
    # with 0 updated ~ average of 0&1, 2 updated ~ average of 2&3.
    # Because token_lens are all ones, it's simple average.
    # We can verify by checking lens pattern and approximate direction of vectors.
    # lens should be [2,2,1,1]
    assert torch.equal(lens_new[0], torch.tensor([2, 2, 1, 1], dtype=torch.long))

    # starts should be [0,2,4,5] in base space given these lens
    assert torch.equal(starts_new[0], torch.tensor([0, 2, 4, 5], dtype=torch.long))

    # Check merged vectors point roughly toward the expected originals:
    # token 0 merged should be close to [1,0], token 1 merged close to [0,1]
    # (we use cosine similarity checks)
    def cos(a, b, eps=1e-8):
        a = a / (a.norm(dim=-1, keepdim=True) + eps)
        b = b / (b.norm(dim=-1, keepdim=True) + eps)
        return (a * b).sum(dim=-1)

    v0 = z_new[0, 0]
    v1 = z_new[0, 1]
    assert cos(v0, torch.tensor([1.0, 0.0])) > 0.99
    assert cos(v1, torch.tensor([0.0, 1.0])) > 0.99


def test_tome_offset1_prefers_odd_left_edges():
    """
    With offset=1, allowed adjacent pairs are (1,2), (3,4), (5,6), ...
    For T=6 and target=4, we need 2 merges; offset=1 should merge (1,2) and (3,4).
    """
    torch.manual_seed(0)

    d_model = 2
    merger = _make_merger_identity_group(d_model=d_model, window_size=64)

    z = torch.tensor(
        [[[1.0, 0.0],   # 0
          [0.0, 1.0],   # 1
          [0.01, 0.99], # 2  ~ similar to 1
          [1.0, 0.0],   # 3
          [0.99, 0.01], # 4  ~ similar to 3
          [0.0, 1.0]]], # 5
        dtype=torch.float32
    )

    token_lens = torch.ones((1, 6), dtype=torch.long)
    target_len = 4

    z_new, lens_new, starts_new = merger(z, token_lens, target_len=target_len, offset=1)

    # Expected survivors: [0,1,3,5], with 1 merged with 2, and 3 merged with 4
    assert torch.equal(lens_new[0], torch.tensor([1, 2, 2, 1], dtype=torch.long))
    assert torch.equal(starts_new[0], torch.tensor([0, 1, 3, 5], dtype=torch.long))

    def cos(a, b, eps=1e-8):
        a = a / (a.norm(dim=-1, keepdim=True) + eps)
        b = b / (b.norm(dim=-1, keepdim=True) + eps)
        return (a * b).sum(dim=-1)

    v1 = z_new[0, 1]
    v2 = z_new[0, 2]
    assert cos(v1, torch.tensor([0.0, 1.0])) > 0.99  # merged (1,2) stays ~ y-axis
    assert cos(v2, torch.tensor([1.0, 0.0])) > 0.99  # merged (3,4) stays ~ x-axis