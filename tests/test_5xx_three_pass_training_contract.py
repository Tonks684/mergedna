import torch
import torch.nn.functional as F

from mergedna.model import MergeDNA
from mergedna.config import MergeDNAConfig
from mergedna.dna_vocab import VOCAB


def _tiny_cfg() -> MergeDNAConfig:
    cfg = MergeDNAConfig()
    cfg.N = 128
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.ffn_mult = 2

    cfg.n_local_enc = 1
    cfg.n_latent_enc = 2
    cfg.n_latent_dec = 1
    cfg.n_local_dec = 1

    cfg.local_window_size = 8
    cfg.L_avg = 64
    cfg.K_avg = 32
    cfg.lambda_latent_mtr = 0.25
    return cfg


def _any_grad(params):
    return any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in params)


def _all_no_grad(params):
    return all(p.grad is None or p.grad.abs().sum() == 0 for p in params)


def test_three_pass_losses_run_and_local_encoder_frozen_for_latent_mtr():
    torch.manual_seed(0)
    device = "cpu"

    cfg = _tiny_cfg()
    model = MergeDNA(cfg).to(device)
    model.train()

    B, N = 2, cfg.N
    x = torch.randint(0, 5, (B, N), device=device)  # exclude MASK

    # -------------------------
    # 1) MTR: local encoder SHOULD get grads
    # -------------------------
    model.zero_grad(set_to_none=True)
    logits_mtr, _ = model.forward_reconstruct(x, target_L=cfg.L_avg)
    loss_mtr = F.cross_entropy(logits_mtr.view(-1, VOCAB.size), x.view(-1))
    assert torch.isfinite(loss_mtr).item() is True
    loss_mtr.backward()

    assert _any_grad(model.local_encoder.parameters()), "Expected local_encoder grads in MTR pass."
    assert _any_grad(model.latent_encoder.parameters()), "Expected latent_encoder grads in MTR pass."

    # -------------------------
    # 2) Latent-MTR: local encoder MUST be frozen (ϕ)
    # -------------------------
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        x_emb = model.embed(x)
        z_L, lengths_L = model.local_encoder(x_emb, target_L=cfg.L_avg)

    z_L_ctx = model.latent_encoder(z_L)
    z_K, gid = model.latent_encoder.global_merge_to_K(z_L_ctx, K=cfg.K_avg)
    z_bar_L = torch.gather(z_K, 1, gid.unsqueeze(-1).expand(z_K.size(0), gid.size(1), z_K.size(-1)))

    z_L_hat = model.latent_decoder(z_bar_L)
    logits_lat = model.local_decoder(z_L_hat, lengths_L, N=x.size(1))
    loss_lat = F.cross_entropy(logits_lat.view(-1, VOCAB.size), x.view(-1))
    assert torch.isfinite(loss_lat).item() is True
    loss_lat.backward()

    assert _all_no_grad(model.local_encoder.parameters()), "Expected NO local_encoder grads in latent-MTR pass."
    assert _any_grad(model.latent_encoder.parameters()), "Expected latent_encoder grads in latent-MTR pass."

    # -------------------------
    # 3) AMTM: should run end-to-end and backprop through full net (no latent merge in prediction)
    # -------------------------
    model.zero_grad(set_to_none=True)

    out = model.forward_amtm(
        x,
        target_L=cfg.L_avg,
        target_K=cfg.K_avg,
        mask_token_id=VOCAB.MASK,
    )
    loss_amtm = out[0]  # robust to 4 vs 5 returns
    assert torch.isfinite(loss_amtm).item() is True
    loss_amtm.backward()

    assert _any_grad(model.local_encoder.parameters()), "Expected local_encoder grads in AMTM pass."