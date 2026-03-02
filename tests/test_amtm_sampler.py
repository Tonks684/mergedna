import torch
import pytest

from mergedna.amtm import compute_amtm_probs_from_groups, sample_exact_k_tokens, AMTMMaskSampler
from mergedna.local_merge import project_mask_to_base


def test_probs_favor_smaller_groups():
    # One batch: L=6, K_groups=3
    # group sizes: g0=1, g1=2, g2=3
    gid = torch.tensor([[0, 1, 1, 2, 2, 2]], dtype=torch.long)
    probs = compute_amtm_probs_from_groups(gid, K=3, power=2.0)

    # token in group 0 should have highest probability
    p0 = probs[0, 0].item()
    p1 = probs[0, 1].item()  # group 1
    p2 = probs[0, 3].item()  # group 2

    assert p0 > p1 > p2
    assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-6)


def test_sample_exact_k_tokens():
    torch.manual_seed(0)
    probs = torch.tensor([[0.10, 0.20, 0.30, 0.40]], dtype=torch.float32)  # (1,4)
    mask = sample_exact_k_tokens(probs, K_samples=2)
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 4)
    assert int(mask.sum().item()) == 2


def test_mask_projection_matches_lengths():
    # lengths define spans: [0..2), [2..5), [5..7) => N=7
    lengths = torch.tensor([[2, 3, 2]], dtype=torch.long)  # (1,3)
    N = 7
    mask_L = torch.tensor([[True, False, True]], dtype=torch.bool)  # mask token 0 and 2

    mask_N = project_mask_to_base(mask_L, lengths, N)
    # Expect base mask True at [0,1] and [5,6]
    expected = torch.tensor([[True, True, False, False, False, True, True]], dtype=torch.bool)
    assert torch.equal(mask_N, expected)


def test_amtm_sampler_end_to_end():
    torch.manual_seed(0)
    B, L, N = 2, 8, 16
    K_groups = 4
    K_samples = 4

    # Make some group ids in range [0..3]
    gid = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0, 1, 1, 1, 2, 3, 3, 3],
    ], dtype=torch.long)

    # lengths sum to N
    lengths = torch.tensor([
        [2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 2, 2, 2, 2, 3, 3],
    ], dtype=torch.long)

    sampler = AMTMMaskSampler()
    mask_L, mask_N = sampler(gid, lengths, N, K_groups=K_groups, K_samples=K_samples)

    assert mask_L.shape == (B, L)
    assert mask_N.shape == (B, N)
    assert int(mask_L[0].sum().item()) == K_samples
    assert int(mask_L[1].sum().item()) == K_samples