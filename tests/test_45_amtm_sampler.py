import torch

from mergedna.amtm import compute_amtm_probs_from_groups, sample_exact_k_tokens, AMTMMaskSampler
from mergedna.local_merge import project_mask_to_base


def test_probs_favor_smaller_groups():
    # L=6, groups K=3; sizes: g0=1, g1=2, g2=3
    gid = torch.tensor([[0, 1, 1, 2, 2, 2]], dtype=torch.long)
    probs = compute_amtm_probs_from_groups(gid, K=3, power=2.0)

    p0 = probs[0, 0].item()  # group 0
    p1 = probs[0, 1].item()  # group 1
    p2 = probs[0, 3].item()  # group 2

    assert p0 > p1 > p2
    assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-6)


def test_sample_exact_k_tokens_returns_exact_K():
    torch.manual_seed(0)
    probs = torch.tensor([[0.10, 0.20, 0.30, 0.40]], dtype=torch.float32)
    mask = sample_exact_k_tokens(probs, K_samples=2)

    assert mask.dtype == torch.bool
    assert mask.shape == (1, 4)
    assert int(mask.sum().item()) == 2


def test_amtm_sampler_projects_mask_to_base_shape_ok():
    torch.manual_seed(0)

    B, L, N = 2, 8, 16
    K_groups = 4
    K_samples = 4

    gid = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0, 1, 1, 1, 2, 3, 3, 3],
    ], dtype=torch.long)

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

    # projection should agree with direct project_mask_to_base for the same inputs
    mask_N2 = project_mask_to_base(mask_L, lengths, N)
    assert torch.equal(mask_N, mask_N2)