# scripts/pretrain_smoke.py
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

from mergedna.model import MergeDNA
from mergedna.config import MergeDNAConfig
from mergedna.dna_vocab import VOCAB
from mergedna.infra.logging import make_run_logger, ThroughputMeter
from mergedna.infra.checkpoint import save_checkpoint, load_checkpoint

from mergedna.infra.wandb_logger import WandbLogger, WandbConfig
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

from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--tiny", action="store_true")

    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/pretrain_smoke")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")

    parser.add_argument("--wandb", type=str, default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb-project", type=str, default="mergedna-smoke")
    parser.add_argument("--wandb-name", type=str, default=None)

    args = parser.parse_args()

    logger = make_run_logger(run_id="pretrain_smoke", enable_file=True, enable_print=True)
    tput = ThroughputMeter()

    print(f"[pretrain_smoke] args={args}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MergeDNAConfig()
    if args.tiny:
        cfg.N = 256
        cfg.d_model = 128
        cfg.n_heads = 4
        cfg.n_local_enc = 1
        cfg.n_latent_enc = 1
        cfg.n_latent_dec = 1
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

    # W&B mirror (optional)
    wb = WandbLogger(WandbConfig(
        mode=args.wandb,
        project=args.wandb_project,
        name=args.wandb_name,
        tags=["smoke"],
        run_dir=str(Path("runs") / "wandb"),
    ))

    start_step = 0
    if args.resume is not None:
        meta = load_checkpoint(args.resume, model=model, optimizer=opt, map_location=device)
        start_step = int(meta.get("step") or 0) + 1

    for step in range(start_step, args.steps):
        x = make_synthetic_batch(args.batch, cfg.N, device)

        opt.zero_grad()
        loss, lmtr, llat, lamtm = training_step(model, x, cfg, device)
        loss.backward()
        opt.step()

        if step % 10 == 0:
            tokens = int(x.numel())
            tput_metrics = tput.update(tokens)

            payload = logger.log(
                step,
                loss=float(loss.item()),
                loss_mtr=float(lmtr.item()),
                loss_latent=float(llat.item()),
                loss_amtm=float(lamtm.item()),
                device=device,
                batch=int(args.batch),
                seq_len=int(cfg.N),
                **tput_metrics,
            )
            wb.log(payload, step=step)

        if args.ckpt_every > 0 and step % args.ckpt_every == 0 and step > 0:
            save_checkpoint(
                Path(args.ckpt_dir) / f"ckpt_step{step}.pt",
                model=model,
                optimizer=opt,
                step=step,
                cfg=cfg,
            )

    wb.finish()
    print("[pretrain_smoke] training complete", flush=True)
    
if __name__ == "__main__":
    main()