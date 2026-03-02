# scripts/pretrain_smoke.py
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from mergedna.model import MergeDNA
from mergedna.config import MergeDNAConfig
from mergedna.dna_vocab import VOCAB

print(f'[pretrain_smoke] import ok', flush=True)

# ---------------------------------------------------------
# Synthetic dataset for smoke run
# ---------------------------------------------------------

def make_synthetic_batch(B, N, device):
    # Random A,C,G,T,N tokens only
    return torch.randint(0, 5, (B, N), device=device)


# ---------------------------------------------------------
# MergeDNA three-pass training step
# ---------------------------------------------------------

def training_step(model, x, cfg, device):

    target_L = cfg.L_avg
    target_K = cfg.K_avg

    # -----------------------------------------------------
    # 1. MTR pass (full)
    # -----------------------------------------------------
    logits_mtr, _ = model.forward_reconstruct(
        x,
        target_L=target_L,
    )
    loss_mtr = F.cross_entropy(
        logits_mtr.view(-1, VOCAB.size),
        x.view(-1),
    )

    # -----------------------------------------------------
    # 2. Latent-MTR pass (freeze Local Encoder ϕ)
    # -----------------------------------------------------
    with torch.no_grad():
        x_emb = model.embed(x)
        z_L, lengths_L = model.local_encoder(x_emb, target_L=target_L)

    z_L_ctx = model.latent_encoder(z_L)
    z_K, group_of_token = model.latent_encoder.global_merge_to_K(
        z_L_ctx,
        K=target_K,
    )

    z_bar_L = torch.gather(
        z_K,
        1,
        group_of_token.unsqueeze(-1).expand(
            z_K.size(0),
            group_of_token.size(1),
            z_K.size(-1),
        ),
    )

    z_L_hat = model.latent_decoder(z_bar_L)
    logits_latent = model.local_decoder(
        z_L_hat,
        lengths_L,
        N=x.size(1),
    )

    loss_latent = F.cross_entropy(
        logits_latent.view(-1, VOCAB.size),
        x.view(-1),
    )

    # -----------------------------------------------------
    # 3. AMTM pass
    # -----------------------------------------------------
    loss_amtm, _, _, _ = model.forward_amtm(
        x,
        target_L=target_L,
        target_K=target_K,
        mask_token_id=VOCAB.MASK,
    )

    # -----------------------------------------------------
    # Combined objective
    # -----------------------------------------------------
    loss = (
        loss_mtr
        + cfg.lambda_latent_mtr * loss_latent
        + loss_amtm
    )

    return loss, loss_mtr, loss_latent, loss_amtm


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    print(f'[pretrain_smoke] args={args}', flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MergeDNAConfig()
    if args.tiny:
        cfg.N = 256
        cfg.d_model = 128
        cfg.n_heads = 4
        cfg.n_local_enc = 1
        cfg.n_latent_enc = 1
        cfg.n_local_dec = 1
        cfg.L_avg = 128
        cfg.K_avg = 64

    model = MergeDNA(cfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
    )

    for step in range(args.steps):

        x = make_synthetic_batch(
            args.batch,
            cfg.N,
            device,
        )

        opt.zero_grad()

        loss, lmtr, llat, lamtm = training_step(
            model,
            x,
            cfg,
            device,
        )

        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(
                f"step={step} "
                f"loss_mtr={lmtr.item():.4f} "
                f"loss_latent={llat.item():.4f} "
                f"loss_amtm={lamtm.item():.4f}",
                flush=True,
            )
    print(f'[pretrain_smoke] training complete', flush=True)


if __name__ == "__main__":
    main()