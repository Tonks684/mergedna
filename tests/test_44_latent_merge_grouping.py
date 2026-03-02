import torch

from mergedna.model import LatentEncoder
from mergedna.config import MergeDNAConfig


def _tiny_cfg():
    cfg = MergeDNAConfig()
    cfg.N = 128
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.ffn_mult = 2
    cfg.n_latent_enc = 2
    return cfg


def test_global_merge_to_K_shapes_and_ranges():
    torch.manual_seed(0)

    cfg = _tiny_cfg()
    enc = LatentEncoder(cfg)

    B, L, D = 2, 32, cfg.d_model
    K = 8
    z = torch.randn(B, L, D)

    z_ctx = enc(z)
    z_K, gid = enc.global_merge_to_K(z_ctx, K=K)

    assert z_K.shape == (B, K, D)
    assert gid.shape == (B, L)
    assert gid.dtype in (torch.int64, torch.long)

    assert int(gid.min().item()) >= 0
    assert int(gid.max().item()) < K


def test_global_merge_unmerge_gather_contract():
    torch.manual_seed(0)

    cfg = _tiny_cfg()
    enc = LatentEncoder(cfg)

    B, L, D = 1, 16, cfg.d_model
    K = 4
    z = torch.randn(B, L, D)

    z_ctx = enc(z)
    z_K, gid = enc.global_merge_to_K(z_ctx, K=K)

    # Unmerge K->L via gather (as used in latent reconstruction pass)
    z_bar_L = torch.gather(z_K, 1, gid.unsqueeze(-1).expand(B, L, D))
    assert z_bar_L.shape == (B, L, D)

    # spot-check: position j uses group gid[j]
    j = 7
    g = int(gid[0, j].item())
    assert torch.allclose(z_bar_L[0, j], z_K[0, g])