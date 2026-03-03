import torch
import pytest

from mergedna.config import MergeDNAConfig
from mergedna.model import LocalEncoder


def test_local_encoder_layerwise_merging_calls_merger_multiple_times(monkeypatch):
    """
    Confirms:
      - LocalEncoder performs >1 merge call (layer-wise schedule) rather than single-shot.
      - The merge targets match the schedule produced by linear_merge_schedule.
      - Final output has length target_L and token lengths sum to N.
    """
    cfg = MergeDNAConfig()
    cfg.N = 64
    cfg.d_model = 32
    cfg.n_heads = 4
    cfg.ffn_mult = 4
    cfg.n_local_enc = 4
    cfg.local_window_size = 8

    enc = LocalEncoder(cfg)

    # Capture merger calls
    called_target_lens = []
    orig_forward = enc.merger.forward

    def wrapped_forward(z, token_lens, target_len: int):
        called_target_lens.append(int(target_len))
        return orig_forward(z, token_lens, target_len=target_len)

    monkeypatch.setattr(enc.merger, "forward", wrapped_forward)

    B, N, D = 2, cfg.N, cfg.d_model
    x_emb = torch.randn(B, N, D)

    target_L = 32
    z_L, lengths_L = enc(x_emb, target_L=target_L)

    # Expect layer-wise merge schedule with n_merges = min(2, n_layers-1) = 2 (given n_local_enc=4)
    expected_schedule = enc.linear_merge_schedule(N, target_L, n_merges=2)

    # We should have at least those schedule merges.
    # (There may be an extra final safety merge if something went off-by-1; in most cases this will match exactly.)
    assert called_target_lens[: len(expected_schedule)] == expected_schedule

    # If your encoder sometimes triggers a final "ensure exact length" merge, allow that:
    assert called_target_lens[-1] == target_L

    # Shape and segmentation invariants
    assert z_L.shape == (B, target_L, D)
    assert lengths_L.shape == (B, target_L)
    assert lengths_L.dtype == torch.long
    assert torch.all(lengths_L > 0)
    assert torch.all(lengths_L.sum(dim=1) == N)


def test_local_encoder_no_merge_when_target_ge_N():
    """
    If target_L >= N, local merger should not compress.
    (Your merger already returns identity in that case.)
    """
    cfg = MergeDNAConfig()
    cfg.N = 32
    cfg.d_model = 16
    cfg.n_heads = 4
    cfg.n_local_enc = 2

    enc = LocalEncoder(cfg)

    x_emb = torch.randn(1, cfg.N, cfg.d_model)
    z, lengths = enc(x_emb, target_L=cfg.N)  # no compression

    assert z.shape[1] == cfg.N
    assert lengths.shape[1] == cfg.N
    assert torch.all(lengths == 1)