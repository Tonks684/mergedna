"""
MergeDNA model components.

This module implements:
  - LocalEncoder (Section 4.3)
  - LatentEncoder + global_merge_to_K (Section 4.4)
  - LatentDecoder (reconstruction-only, paper pretraining)
  - LocalDecoder + unmerge (Section 4.3)
  - MergeDNA forward passes: MTR, Latent-MTR, AMTM (Sections 4.5 / 5.0)

See report/MergeDNA_Implementation_Report.md for design rationale.
"""

from __future__ import annotations

import math
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
            attn_window_size=(cfg.local_window_size, cfg.local_window_size),
        )
        self.enc = TransformerEncoder(enc_cfg)
        self.merger = LocalTokenMerger(cfg.d_model, LocalMergeConfig(window_size=cfg.local_window_size))

    def forward(self, x_emb: torch.Tensor, target_L: int) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x_emb.shape
        lengths = torch.ones((B, N), device=x_emb.device, dtype=torch.long)
        z = x_emb

        n_layers = self.enc.cfg.n_layers
        # Minimal default: 2 merges if we have enough layers, otherwise fewer.
        n_merges = min(2, max(0, n_layers - 1))  # ensures we always have at least 1 block after final merge

        # If no merges requested, just run full local encoder and return (no compression)
        if n_merges == 0:
            z = self.enc(z)
            return z, lengths

        schedule_L = self.linear_merge_schedule(N, target_L, n_merges)  # len == n_merges

        # Partition layers into (n_merges + 1) segments with guaranteed non-empty segments.
        n_segments = n_merges + 1
        seg_sizes = [n_layers // n_segments] * n_segments
        for i in range(n_layers % n_segments):
            seg_sizes[i] += 1

        boundaries = [0]
        for sz in seg_sizes:
            boundaries.append(boundaries[-1] + sz)
        # boundaries length = n_segments+1, ends at n_layers

        for seg_idx in range(n_segments):
            s, e = boundaries[seg_idx], boundaries[seg_idx + 1]
            z = self.enc.forward_range(z, s, e)

            # Merge after this segment if we still have merges remaining
            offset = seg_idx % 2  # alternate merge partitions: even, then odd, then even...
            if seg_idx < n_merges:
                z, lengths, _starts = self.merger(z, lengths, target_len=schedule_L[seg_idx],offset=offset)

        # Ensure exact final length
        offset = n_merges % 2  # If we already hit target_L, this will be a no-op.
        if z.size(1) != target_L:
            z, lengths, _starts = self.merger(z, lengths, target_len=target_L, offset=offset)

        return z, lengths

    @staticmethod
    def linear_merge_schedule(N: int, target_L: int, n_merges: int) -> list[int]:
        """
        Monotone schedule of intermediate lengths ending at target_L.

        Example:
          N=4096, target_L=2048, n_merges=2 -> [3072, 2048]
        """
        if n_merges <= 0:
            return []
        steps = []
        for i in range(1, n_merges + 1):
            Li = int(round(N - (N - target_L) * (i / n_merges)))
            steps.append(max(target_L, min(N, Li)))
        steps[-1] = target_L
        return steps
        
class LatentEncoder(nn.Module):
    """
    MergeDNA Latent Encoder E_psi (Section 4.4) + global merge operator (L -> K).

    Responsibilities:
      1) Contextualize the locally-merged token sequence Z_L using full attention
         (TransformerEncoder).
      2) Produce a global bottleneck sequence Z_K of length K < L, and a sparse
         grouping structure S' represented as group_of_token ∈ [0..K-1]^(B×L).

    Paper intent:
      - The latent stage compresses the locally-merged sequence into fewer "salient"
        tokens, enabling global context modelling and (later) AMTM importance sampling.
      - The paper describes a ToMe-style global merge. Here we keep the interface and
        training semantics but implement a simpler anchor-based hard clustering
        approximation, which is easy to reason about and swap out later.

    Output contracts:
      - z_K: (B, K, D)  latent / bottleneck token embeddings
      - group_of_token: (B, L)  mapping from each local token index to its latent group

    Design choice (approximation):
      - We select K anchors using a learned grouping projection and token "importance"
        score, then assign each token to the nearest anchor (cosine similarity).
      - Assignments are discrete (argmax) and run under no_grad. The aggregation
        into z_K is still differentiable w.r.t z_L_ctx.
    """

    def __init__(self, cfg: MergeDNAConfig):
        super().__init__()

        enc_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_latent_enc,
            ffn_mult=cfg.ffn_mult,
            max_seq_len=cfg.N,  # enough for L (since L <= N)
        )
        self.enc = TransformerEncoder(enc_cfg)

        # Lightweight projection to a smaller "grouping space" used only for selecting
        # anchors + computing similarities. Using a small dimension reduces overhead.
        #
        # Note: This is not the NanoChat tokenizer; it is an internal projection
        # used to define merge/group structure.
        self.group = nn.Linear(cfg.d_model, 64, bias=False)

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        """
        Full-attention contextualization of local tokens.

        Args:
          z_L: (B, L, D) locally merged token embeddings

        Returns:
          z_L_ctx: (B, L, D) contextualized embeddings
        """
        return self.enc(z_L)

    def global_merge_to_K(self, z_L_ctx: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a contextualized local sequence (length L) into a latent sequence (length K).

        This returns a sparse representation of the paper's S' grouping structure:
          - group_of_token[b, l] = k   means local token l belongs to latent group k

        Args:
          z_L_ctx: (B, L, D) contextualized local embeddings
          K: target latent length (K < L)

        Returns:
          z_K: (B, K, D) latent tokens (group means)
          group_of_token: (B, L) long in [0..K-1]
        """
        B, L, D = z_L_ctx.shape

        # Edge case: if K >= L, there is no meaningful compression.
        # We return the identity mapping (each token is its own "group").
        if K >= L:
            group = torch.arange(L, device=z_L_ctx.device).unsqueeze(0).expand(B, L)
            return z_L_ctx, group

        # ---------------------------------------------------------------------
        # 1) Project tokens into a grouping space and compute "anchor scores"
        # This is a learned projection into a low-dimensional grouping space.
        # ---------------------------------------------------------------------
        # g_raw: (B, L, G)
        g_raw = self.group(z_L_ctx)

        # scores: (B, L) magnitude in grouping space; used as an "importance" proxy.
        # We select anchors by top-K scores per batch item.
        scores = g_raw.norm(dim=-1)

        # Normalize for cosine similarity.
        # g: (B, L, G), anchors will also be normalized -> dot = cosine sim
        g = g_raw / (scores.unsqueeze(-1) + 1e-8)

        # Choose K anchors per batch item. These are indices into the L tokens.
        # Tokens with larger projection magnitude are more salient.
        # topk_idx: (B, K)
        topk_idx = torch.topk(scores, k=K, dim=1).indices

        # Gather normalized anchor vectors:
        # anchors: (B, K, G)
        anchors = torch.gather(
            g, 1, topk_idx.unsqueeze(-1).expand(B, K, g.size(-1))
        )

        # ---------------------------------------------------------------------
        # 2) Assign each token to the nearest anchor (hard clustering)
        # ---------------------------------------------------------------------
        # sim: (B, L, K) cosine similarity between each local token and each anchor
        sim = torch.einsum("blg,bkg->blk", g, anchors)

        # Discrete assignment. We explicitly do this under no_grad because:
        #   - argmax is non-differentiable
        #   - we treat the grouping as a structural routing decision, similar in spirit
        #     to hard ToMe merges / routing.
        with torch.no_grad():
            group_of_token = sim.argmax(dim=-1)  # (B, L), values in [0..K-1]

        # ---------------------------------------------------------------------
        # 3) Aggregate local embeddings into latent tokens via group means
        # ---------------------------------------------------------------------
        # z_K is computed as the mean of all z_L_ctx assigned to each group.
        # This keeps gradients flowing from z_K back into z_L_ctx (aggregation is differentiable),
        # even though the assignment itself is discrete.
        z_K = torch.zeros((B, K, D), device=z_L_ctx.device, dtype=z_L_ctx.dtype)
        counts = torch.zeros((B, K), device=z_L_ctx.device, dtype=z_L_ctx.dtype)

        # Accumulate embeddings per group:
        # scatter_add_ over dim=1 with index shape (B, L, D)
        z_K.scatter_add_(
            1,
            group_of_token.unsqueeze(-1).expand(B, L, D),
            z_L_ctx
        )

        # Count members per group (for mean):
        counts.scatter_add_(
            1,
            group_of_token,
            torch.ones((B, L), device=z_L_ctx.device, dtype=z_L_ctx.dtype)
        )

        # Avoid division by zero in pathological cases (should be rare but safe):
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
        """
        Latent Decoder E_omega (paper: reconstruction-only module).

        Args:
        z_L: (B, L, D) latent-decoder input at local-token resolution.
            In latent-MTR this is typically z_bar_L (K->L unmerge),
            in MTR this is typically z_L_ctx.

        Returns:
        z_L_hat: (B, L, D) decoded local-token representations
        """
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
            attn_window_size=(cfg.local_window_size, cfg.local_window_size),
        )
        self.dec = TransformerEncoder(dec_cfg)
        self.head = nn.Linear(cfg.d_model, VOCAB.size, bias=False)

    def forward(self, z_L_hat: torch.Tensor, lengths_L: torch.Tensor, N: int) -> torch.Tensor:
        """
        Local Decoder E_zeta (Section 4.3).

        Steps:
        1) Unmerge merged-token embeddings back to base resolution using lengths_L.
        2) Run decoder transformer blocks at base resolution.
        3) Produce logits over DNA vocab.

        Args:
        z_L_hat: (B, L, D) local-token representations to detokenize
        lengths_L: (B, L) span lengths that define the sparse segmentation mapping S
        N: base sequence length

        Returns:
        logits: (B, N, V) base-resolution logits over VOCAB
        """
        zN = unmerge_tokens(z_L_hat, lengths_L, N)  # (B, N, D)
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
        MTR forward pass (Section 5.0, uses modules 4.1–4.4 + 4.3 unmerge).

        Pipeline:
        x -> embed -> LocalEncoder (N->L) -> LatentEncoder (full attention) ->
        LatentDecoder -> LocalDecoder (L->N) -> logits

        Args:
        x: (B, N) token ids
        target_L: merged sequence length L

        Returns:
        logits: (B, N, V)
        lengths_L: (B, L) sparse segmentation lengths for unmerge/mask projection
        """
        # x -> embed
        x_emb = self.embed(x)
        # LocalEncoder (N->L) -> LatentEncoder (full attention)
        z_L, lengths_L = self.local_encoder(x_emb, target_L=target_L)
        # Contextualize with full attention before global merge.
        z_L_ctx = self.latent_encoder(z_L)
        # LatentDecoder (reconstruction-only) -> LocalDecoder (L->N)
        z_L_hat = self.latent_decoder(z_L_ctx)
        # LocalDecoder (L->N) -> logits
        logits = self.local_decoder(z_L_hat, lengths_L, N=x.size(1))
        return logits, lengths_L

    def forward_latent_reconstruct(self, x: torch.Tensor, target_L: int, target_K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Latent-MTR forward pass (Section 5.0; latent bottleneck is active).

        IMPORTANT:
        This method does not itself freeze LocalEncoder parameters.
        The training loop should wrap the LocalEncoder call in torch.no_grad()
        (or otherwise prevent gradient updates) to match the paper's objective.

        Pipeline:
        x -> embed -> LocalEncoder (N->L) -> LatentEncoder ->
        global_merge_to_K (L->K) -> gather unmerge (K->L) ->
        LatentDecoder -> LocalDecoder (L->N) -> logits

        Returns:
        logits: (B, N, V)
        lengths_L: (B, L)
        group_of_token: (B, L) latent group assignment used for AMTM probabilities
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
        sampler = AMTMMaskSampler(sampler_cfg)  # sampler_cfg=None => default AMTM settings
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