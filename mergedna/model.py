from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mergedna.amtm import AMTMMaskSampler, AMTMCfg
from mergedna.config import MergeDNAConfig
from mergedna.dna_vocab import VOCAB
from mergedna.transformer import EncoderConfig, TransformerEncoder, rmsnorm
from mergedna.local_merge import LocalMergeConfig, LocalTokenMerger, unmerge_tokens

class LocalEncoder(nn.Module):
    def __init__(self, cfg: MergeDNAConfig):
        super().__init__()
        enc_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_local_enc,
            ffn_mult=cfg.ffn_mult,
            max_seq_len=cfg.N,
        )
        self.enc = TransformerEncoder(enc_cfg)
        self.merger = LocalTokenMerger(cfg.d_model, LocalMergeConfig(window_size=cfg.local_window_size))

    def forward(self, x_emb: torch.Tensor, target_L: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_emb: (B,N,D) embedded nucleotides
        Returns:
          z_L: (B,L,D)
          lengths: (B,L) token lengths summing to N
        """
        B, N, D = x_emb.shape
        lengths = torch.ones((B, N), device=x_emb.device, dtype=torch.long)  # start as single-base tokens

        z = self.enc(x_emb)
        # token counts are still N here. We merge down to target_L.
        z_L, lengths_L, starts_new = self.merger(z, lengths, target_len=target_L)
        return z_L, lengths_L

class LatentEncoder(nn.Module):
    def __init__(self, cfg: MergeDNAConfig):
        super().__init__()
        enc_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_latent_enc,
            ffn_mult=cfg.ffn_mult,
            max_seq_len=cfg.N,  # covers L too
        )
        self.enc = TransformerEncoder(enc_cfg)
        # A lightweight projection for global token merging similarity
        self.group = nn.Linear(cfg.d_model, 64, bias=False)

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        return self.enc(z_L)

    def global_merge_to_K(self, z_L_ctx: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        z_K: (B,K,D)
        group_of_token: (B,L) int64 in [0..K-1]
        """
        B, L, D = z_L_ctx.shape
        if K >= L:
            group = torch.arange(L, device=z_L_ctx.device).unsqueeze(0).expand(B, L)
            return z_L_ctx, group

        # projection used for similarity / grouping
        g_raw = self.group(z_L_ctx)                    # (B,L,G)
        scores = g_raw.norm(dim=-1)                    # (B,L)  (meaningful!)
        g = g_raw / (scores.unsqueeze(-1) + 1e-8)      # (B,L,G) normalized for cosine

        # choose K anchors per batch by highest pre-normalization norm
        topk_idx = torch.topk(scores, k=K, dim=1).indices  # (B,K)
        anchors = torch.gather(
            g, 1, topk_idx.unsqueeze(-1).expand(B, K, g.size(-1))
        )  # (B,K,G)

        # assign each token to nearest anchor (cosine sim)
        sim = torch.einsum("blg,bkg->blk", g, anchors)      # (B,L,K)
        with torch.no_grad():
            group_of_token = sim.argmax(dim=-1)             # (B,L)

        # compute z_K as mean of members (vectorized)
        z_K = torch.zeros((B, K, D), device=z_L_ctx.device, dtype=z_L_ctx.dtype)
        counts = torch.zeros((B, K), device=z_L_ctx.device, dtype=z_L_ctx.dtype)

        # scatter-add embeddings
        z_K.scatter_add_(
            1,
            group_of_token.unsqueeze(-1).expand(B, L, D),
            z_L_ctx
        )
        counts.scatter_add_(
            1,
            group_of_token,
            torch.ones((B, L), device=z_L_ctx.device, dtype=z_L_ctx.dtype)
        )

        z_K = z_K / counts.clamp_min(1.0).unsqueeze(-1)
        return z_K, group_of_token

class LatentDecoder(nn.Module):
    def __init__(self, cfg: MergeDNAConfig):
        super().__init__()
        dec_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_latent_dec,
            ffn_mult=cfg.ffn_mult,
            max_seq_len=cfg.N,
        )
        self.dec = TransformerEncoder(dec_cfg)

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        return self.dec(z_L)

class LocalDecoder(nn.Module):
    def __init__(self, cfg: MergeDNAConfig):
        super().__init__()
        dec_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_local_dec,
            ffn_mult=cfg.ffn_mult,
            max_seq_len=cfg.N,
        )
        self.dec = TransformerEncoder(dec_cfg)
        self.head = nn.Linear(cfg.d_model, VOCAB.size, bias=False)

    def forward(self, z_L_hat: torch.Tensor, lengths_L: torch.Tensor, N: int) -> torch.Tensor:
        """
        z_L_hat: (B,L,D)
        lengths_L: (B,L) sum to N
        Returns logits over base vocab: (B,N,V)
        """
        # Unmerge merged-token embeddings back to base resolution
        zN = unmerge_tokens(z_L_hat, lengths_L, N)  # (B,N,D)
        zN = self.dec(zN)
        return self.head(rmsnorm(zN))

class MergeDNA(nn.Module):
    def __init__(self, cfg: MergeDNAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(VOCAB.size, cfg.d_model)

        self.local_encoder = LocalEncoder(cfg)
        self.latent_encoder = LatentEncoder(cfg)
        self.latent_decoder = LatentDecoder(cfg)
        self.local_decoder = LocalDecoder(cfg)

    def forward_reconstruct(self, x: torch.Tensor, target_L: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full reconstruction pass (MTR):
          x -> local enc (merge to L) -> latent enc (full) -> latent dec -> local dec -> logits
        """
        x_emb = self.embed(x)
        z_L, lengths_L = self.local_encoder(x_emb, target_L=target_L)
        z_L_ctx = self.latent_encoder(z_L)
        z_L_hat = self.latent_decoder(z_L_ctx)
        logits = self.local_decoder(z_L_hat, lengths_L, N=x.size(1))
        return logits, lengths_L

    def forward_latent_reconstruct(self, x: torch.Tensor, target_L: int, target_K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Latent reconstruction pass:
          x -> local enc (merge to L) [frozen outside] -> latent enc -> global merge to K -> unmerge to L -> latent dec -> local dec
        Returns logits and group assignments.
        """
        x_emb = self.embed(x)

        # Local encoder is expected to be run with no_grad in the training loop for this pass
        z_L, lengths_L = self.local_encoder(x_emb, target_L=target_L)

        z_L_ctx = self.latent_encoder(z_L)
        z_K, group_of_token = self.latent_encoder.global_merge_to_K(z_L_ctx, K=target_K)

        # Unmerge K -> L by copying group embedding back to each token position
        z_bar_L = torch.gather(
            z_K, 1, group_of_token.unsqueeze(-1).expand(z_K.size(0), group_of_token.size(1), z_K.size(-1))
        )

        z_L_hat = self.latent_decoder(z_bar_L)
        logits = self.local_decoder(z_L_hat, lengths_L, N=x.size(1))
        return logits, lengths_L, group_of_token
    
    def forward_amtm(
        self,
        x: torch.Tensor,
        *,
        target_L: int,
        target_K: int,
        sampler_cfg: AMTMCfg | None = None,
        mask_token_id: int | None = None,
        ignore_ambiguous_targets: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adaptive Masked Token Modelling (AMTM) forward pass.

        Paper semantics:
          - Use latent grouping (S') to DEFINE which local tokens are important.
          - Sample exactly K local tokens to mask according to P_L(j) ∝ 1 / g(group(j))^2.
          - Project mask to base resolution via S (here: lengths_L sparse mapping).
          - Run the full network WITHOUT latent merging (i.e. use forward_reconstruct on masked input).
          - Compute loss on masked base positions only.

        Implementation notes:
          - Mask token defaults to VOCAB.MASK unless overridden.
          - Optionally exclude ambiguous targets (VOCAB.N) from the loss mask.

        Returns:
          loss: scalar
          logits: (B,N,V)
          mask_L: (B,L) bool
          mask_N: (B,N) bool
        """
        B, N = x.shape
        device = x.device

        # Default to explicit [MASK] token in our DNA vocab
        if mask_token_id is None:
            mask_token_id = VOCAB.MASK

        # 1) Local tokens + span mapping
        x_emb = self.embed(x)
        z_L, lengths_L = self.local_encoder(x_emb, target_L=target_L)

        # 2) Latent grouping S' (only for defining mask)
        z_L_ctx = self.latent_encoder(z_L)
        _, group_of_token = self.latent_encoder.global_merge_to_K(z_L_ctx, K=target_K)

        # 3) Sample masks and project to base
        sampler = AMTMMaskSampler(sampler_cfg)
        mask_L, mask_N = sampler(group_of_token, lengths_L, N, K_groups=target_K, K_samples=target_K)

        # 4) Mask the base tokens
        x_masked = x.clone()
        x_masked[mask_N] = mask_token_id

        # 5) Prediction pass WITHOUT latent merge (per paper)
        logits, _ = self.forward_reconstruct(x_masked, target_L=target_L)

        # 6) Loss on masked positions only (optionally excluding ambiguous targets)
        loss_mask = mask_N
        if ignore_ambiguous_targets:
            loss_mask = loss_mask & (x != VOCAB.N)

        if loss_mask.any():
            loss = F.cross_entropy(
                logits[loss_mask],   # (num_loss, V)
                x[loss_mask],        # (num_loss,)
                reduction="mean",
            )
        else:
            loss = torch.zeros((), device=device, dtype=logits.dtype)

        return loss, logits, mask_L, mask_N