from __future__ import annotations

from pathlib import Path

import torch

from mergedna.infra.checkpoint import save_checkpoint, load_checkpoint
from mergedna.model import MergeDNA
from mergedna.config import MergeDNAConfig


def _clone_state_dict(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def test_checkpoint_roundtrip_model_params(tmp_path: Path):
    torch.manual_seed(0)
    cfg = MergeDNAConfig(N=32, d_model=32, n_heads=4, n_local_enc=1, n_latent_enc=1, n_latent_dec=1, n_local_dec=1, L_avg=16, K_avg=8)
    model = MergeDNA(cfg)

    # initialize params deterministically
    before = _clone_state_dict(model.state_dict())

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model=model, step=7, cfg=cfg, extra={"note": "unit-test"})

    # modify model params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    # load back
    meta = load_checkpoint(ckpt_path, model=model, map_location="cpu", strict=True)
    after = _clone_state_dict(model.state_dict())

    assert meta["step"] == 7
    assert meta["extra"]["note"] == "unit-test"

    # params restored exactly
    for k in before.keys():
        assert torch.equal(before[k], after[k])


def test_checkpoint_roundtrip_optimizer_state(tmp_path: Path):
    torch.manual_seed(0)
    cfg = MergeDNAConfig(N=32, d_model=32, n_heads=4, n_local_enc=1, n_latent_enc=1, n_latent_dec=1, n_local_dec=1, L_avg=16, K_avg=8)
    model = MergeDNA(cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Do one tiny step to populate optimizer state
    x = torch.randint(0, 6, (2, cfg.N))
    logits, _ = model.forward_reconstruct(x, target_L=cfg.L_avg)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 6), x.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()

    ckpt_path = tmp_path / "ckpt_opt.pt"
    save_checkpoint(ckpt_path, model=model, optimizer=opt, step=3, cfg=cfg)

    # create fresh model/opt and load
    model2 = MergeDNA(cfg)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

    meta = load_checkpoint(ckpt_path, model=model2, optimizer=opt2, map_location="cpu", strict=True)
    assert meta["step"] == 3

    # optimizer state dict should now be non-empty and match keys
    s1 = opt.state_dict()
    s2 = opt2.state_dict()
    assert s1.keys() == s2.keys()
    assert len(s2["state"]) > 0