from __future__ import annotations

import os
import math
import time
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn

from mergedna.common import compute_init, compute_cleanup, print0, autodetect_device_type
from mergedna.config import MergeDNAConfig
from mergedna.model import MergeDNA
from mergedna.losses import (
    cross_entropy_over_bases,
    sample_L_gaussian,
    derive_am_tm_mask,
    token_mask_to_base_mask,
)
from mergedna.dna_vocab import VOCAB
from mergedna.data_hf import make_hf_dna_dataloader

def cosine_lr(step: int, base_lr: float, warmup: int, total_steps: int) -> float:
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total_steps - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="mergedna-dev")
    parser.add_argument("--num-iterations", type=int, default=100_000)
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Data
    parser.add_argument("--hf-dataset", type=str, default="InstaDeepAI/multi_species_genomes")
    parser.add_argument("--seq-len", type=int, default=4096)
    args = parser.parse_args()

    # init distributed, set device, etc.
    compute_init()
    device_type = args.device_type or autodetect_device_type()
    device = torch.device("cuda" if device_type == "cuda" else "cpu")

    cfg = MergeDNAConfig(N=args.seq_len)
    print0(f"MergeDNA config: {asdict(cfg)}")

    model = MergeDNA(cfg).to(device)
    model.train()

    # AdamW per paper
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # data loader yields (B,N) int64 base tokens
    loader = make_hf_dna_dataloader(
        dataset_name=args.hf_dataset,
        seq_len=args.seq_len,
        global_batch_size=args.batch_size,
        device_batch_size=args.device_batch_size,
    )

    step = 0
    t0 = time.time()

    while step < args.num_iterations:
        optim.zero_grad(set_to_none=True)

        # gradient accumulation to reach global batch size
        total_loss = 0.0

        for micro in range(args.grad_accum):
            x = next(loader).to(device)  # (b,N)

            # ===== Pass 1: MTR (updates all params) =====
            L = sample_L_gaussian(N=cfg.N, device=device)  # centre N/2, clamp [0.4N, 0.6N]
            logits, _lengths = model.forward_reconstruct(x, target_L=L)
            loss_mtr = cross_entropy_over_bases(logits, x, denom=cfg.N)

            # ===== Pass 2: latent MTR (freeze Local Encoder params) =====
            K = max(1, L // 2)
            with torch.no_grad():
                # Detach Local Encoder by running it under no_grad inside model call.
                # Easiest is to inline its call, but we keep the API and rely on no_grad context.
                pass
            # Re-run pass but prevent grads into local_encoder by disabling requires_grad temporarily
            # (This avoids duplicating model code.)
            local_params = list(model.local_encoder.parameters())
            prev_flags = [p.requires_grad for p in local_params]
            for p in local_params:
                p.requires_grad = False

            logits_lat, lengths_L, group_of_token = model.forward_latent_reconstruct(x, target_L=L, target_K=K)
            loss_lat_mtr = cross_entropy_over_bases(logits_lat, x, denom=cfg.N)

            # restore local encoder grads
            for p, f in zip(local_params, prev_flags):
                p.requires_grad = f

            # ===== Pass 3: AMTM (mask informative tokens; no latent token merging) =====
            # Build token mask from latent grouping
            M_L = derive_am_tm_mask(group_of_token, K=K)              # (B,L), exactly K masked tokens
            M_N = token_mask_to_base_mask(M_L, lengths_L, N=cfg.N)    # (B,N)

            x_masked = x.clone()
            x_masked[M_N] = VOCAB.MASK

            logits_mlm, _ = model.forward_reconstruct(x_masked, target_L=L)
            # Paper denominator is K (masked tokens), even though base masked positions can exceed K
            loss_amtm = cross_entropy_over_bases(logits_mlm[M_N].unsqueeze(0), x[M_N].unsqueeze(0), denom=K)

            # Eq.8 combine (λ=0.25)
            loss = loss_mtr + cfg.lambda_latent_mtr * loss_lat_mtr + loss_amtm
            loss = loss / float(args.grad_accum)

            loss.backward()
            total_loss += float(loss.item())

        # grad clip and step
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # LR schedule
        lr = cosine_lr(step, args.lr, args.warmup, args.num_iterations)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.step()

        if step % 50 == 0:
            dt = time.time() - t0
            print0(f"step {step:6d} | loss {total_loss:.5f} | lr {lr:.2e} | {dt:.1f}s")
            t0 = time.time()

        step += 1

    compute_cleanup()

if __name__ == "__main__":
    main()