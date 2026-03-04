"""
-----------------------------------------------------------------------------
Local token merging (MergeDNA §4.3)

Goal:
  - Start with base-resolution token sequence z of length T (typically T == N).
  - Merge adjacent tokens inside local windows until I reach a target length L.
  - Track the resulting segmentation sparsely via token_lens (and derived starts).
Representation / invariants:
  - token_lens[b, l] is the number of base tokens covered by merged token l.
  - Merges are adjacent only, so every merged token corresponds to a contiguous span.
  - Starts are derivable by cumulative sum:
        starts = cumsum(lens) - lens
  - This is a run-length encoding of the paper's dense S ∈ {0,1}^{L×N}.
Design choices:
  - Avoids materialising dense S (O(LN) memory).
  - Still supports:
      (1) "unmerge" embeddings back to base resolution (repeat by span length)
      (2) "project" masks from merged tokens back to base resolution (repeat mask bits)
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class LocalMergeConfig:
    # Window size W used for local merge scheduling and (optionally) local attention.
    # Merges are selected independently per window to preserve locality.
    window_size: int = 16

    # Dimensionality of the learned grouping space used to score adjacency similarity.
    group_dim: int = 64


class LocalTokenMerger(nn.Module):
    """
    Differentiable local token merger producing a shorter sequence and sparse segmentation.

    This module implements a practical approximation to the local merge mechanism:
      - project tokens into a grouping space
      - compute adjacent similarity
      - select non-overlapping adjacent pairs to merge
      - merge by span-length-weighted averaging
      - accumulate span lengths

    Outputs:
      z_new:      (B, target_len, D)
      lens_new:   (B, target_len)  int64 span lengths, summing (approximately) to N
      starts_new: (B, target_len)  start positions in base index space (derived)
    """

    def __init__(self, d_model: int, cfg: LocalMergeConfig):
        super().__init__()
        self.cfg = cfg

        # Lightweight MLP to embed tokens into a "grouping space" where adjacency
        # similarity is computed. This is not meant to be a full encoder: just
        # enough capacity to learn local merge structure.
        self.group = nn.Sequential(
            nn.Linear(d_model, cfg.group_dim, bias=False),
            nn.ReLU(),
            nn.Linear(cfg.group_dim, cfg.group_dim, bias=False),
        )

    def _choose_nonoverlap_pairs_tome_adjacent(
        self, sim_adj: torch.Tensor, merge_budget: int, offset: int = 0
    ) -> torch.Tensor:
        """
        ToMe-style bipartite selection specialized to ADJACENT edges only.

        Taken from https://arxiv.org/pdf/2210.09461 

        sim_adj: (W-1,) similarity for edges (i,i+1) within the window.
        Returns merge_right: (W,) bool where merge_right[j]=True means merge j into j-1.
        """
        W = sim_adj.numel() + 1
        device = sim_adj.device

        merge_right = torch.zeros(W, dtype=torch.bool, device=device)

        # left indices for edges (i, i+1)
        edge_left = torch.arange(0, W - 1, device=device)

        # ToMe-style partition: choose edges where left index parity matches offset
        # offset = 0 → edges (0,1), (2,3), ...
        # offset = 1 → edges (1,2), (3,4), ...
        allowed = (edge_left % 2 == offset)

        candidate_edges = edge_left[allowed]
        candidate_scores = sim_adj[candidate_edges]

        if candidate_edges.numel() == 0:
            return merge_right

        k = min(merge_budget, candidate_edges.numel())

        # pick top-k edges by similarity
        topk = torch.topk(candidate_scores, k=k).indices
        chosen_left = candidate_edges[topk]
        chosen_right = chosen_left + 1

        merge_right[chosen_right] = True

        return merge_right

    def forward(
        self,
        z: torch.Tensor,
        token_lens: torch.Tensor,
        target_len: int,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge base-resolution tokens down to `target_len`.

        Args:
            z:
                (B, T, D) token embeddings at current resolution.
            token_lens:
                (B, T) int64 span lengths (usually all ones at base resolution).
            target_len:
                desired output length L.
            offset: enables alternate partition switching even and odd.

        Returns:
            z_new:
                (B, L, D) merged token embeddings.
            lens_new:
                (B, L) int64 span lengths of merged tokens (run-length encoding).
            starts_new:
                (B, L) int64 start offsets in base space (derived from lens_new).

        Behaviour / invariants:
            - If target_len >= T, no merging is done (identity), but I still return starts.
            - If merges happen, I enforce exact L by truncation (deterministic):
                z_new = z_new[:target_len]
                lens_new = lens_new[:target_len]
              This keeps downstream shapes stable for training/tests.
        """
        B, T, D = z.shape

        # Fast path: no compression needed.
        if target_len >= T:
            starts = torch.cumsum(token_lens, dim=1) - token_lens
            return z, token_lens, starts

        merges_needed = T - target_len
        W = self.cfg.window_size

        # Compute normalized grouping vectors g so dot-product ≈ cosine similarity.
        # g: (B, T, G)
        # Project tokens in grouping space
        g = self.group(z)
        # Normalise via cosine sim: this makes similarity scores more stable and comparable across different inputs and training steps.

        g = g / (g.norm(dim=-1, keepdim=True) + 1e-8)

        # Then compute adjacent similarity (not full window) per batch:
        # sim_all[b, t] = cos_sim(g[b,t], g[b,t+1]) for t in [0..T-2]
        # shape: (B, T-1)
        sim_all = (g[:, :-1] * g[:, 1:]).sum(dim=-1)

        z_out, lens_out = [], []

        # I do merging per-sample to avoid complicated ragged ops.
        # (B is typically small enough; correctness > micro-optimizations for the take-home.)
        for b in range(B):
            zb = z[b]             # (T, D)
            lb = token_lens[b]    # (T,)

            # keep[j] indicates token j survives as a token in the compressed sequence.
            # When I merge token j into j-1, I mark keep[j]=False.
            keep = torch.ones(T, dtype=torch.bool, device=z.device)

            # merge_right[j]=True means merge token j into token j-1.
            merge_right = torch.zeros(T, dtype=torch.bool, device=z.device)

            # Divide total merge budget across windows to keep locality roughly uniform.
            # nwin windows of size W (last window may be smaller).
            ## local-window constraints
            nwin = (T + W - 1) // W
            base = merges_needed // nwin
            extra = merges_needed % nwin

            for widx in range(nwin):
                s = widx * W
                e = min((widx + 1) * W, T)

                # Need at least 2 tokens to form an edge.
                if e - s <= 1:
                    continue

                # Allocate budget for this window (base + 1 for first 'extra' windows).
                budget = base + (1 if widx < extra else 0)

                # In a window of size M, max non-overlapping merges is floor(M/2).
                budget = min(budget, (e - s) // 2)
                if budget <= 0:
                    continue

                # Adjacent similarities restricted to this window.
                # local has shape: (e-s-1,)
                local = sim_all[b, s : e - 1]

                # Choose which right-tokens inside the window get merged.
                mr = self._choose_nonoverlap_pairs_tome_adjacent(local, budget,offset=offset)

                # Convert local window indices to global indices in [0..T-1].
                idx = torch.nonzero(mr, as_tuple=False).flatten() + s

                # Mark merges globally.
                merge_right[idx] = True
                keep[idx] = False

            # -----------------------------------------------------------------
            # Apply merges: "right token -> left token"
            #
            # I do this in a second pass after selecting all merges so that:
            #   - selection is stable
            #   - merging doesn't change similarity scores mid-pass
            #
            # Merge rule (span-weighted average):
            #   z_i := (z_i * len_i + z_j * len_j) / (len_i + len_j)
            #   len_i := len_i + len_j
            #
            # This is the key to preserving a faithful reconstruction weighting:
            # tokens representing more base positions carry more "mass".
            # -----------------------------------------------------------------
            zb2 = zb.clone()
            lb2 = lb.clone()

            idxs = torch.nonzero(merge_right, as_tuple=False).flatten()
            for j in idxs.tolist():
                i = j - 1
                if i < 0:
                    continue

                w_i = lb2[i].to(zb2.dtype)
                w_j = lb2[j].to(zb2.dtype)

                zb2[i] = (zb2[i] * w_i + zb2[j] * w_j) / (w_i + w_j)
                lb2[i] = lb2[i] + lb2[j]

            # Keep only surviving tokens (compressed sequence).
            zb_new = zb2[keep]
            lb_new = lb2[keep]

            # Enforce exact L deterministically.
            zb_new = zb_new[:target_len]
            lb_new = lb_new[:target_len]

            z_out.append(zb_new)
            lens_out.append(lb_new)

        z_new = torch.stack(z_out, 0)       # (B, L, D)
        lens_new = torch.stack(lens_out, 0) # (B, L)

        # Derived sparse segmentation metadata: start offsets per token.
        starts_new = torch.cumsum(lens_new, dim=1) - lens_new
        return z_new, lens_new, starts_new


def unmerge_tokens(z_L: torch.Tensor, token_lens: torch.Tensor, N: int) -> torch.Tensor:
    """
    "Unmerge" merged token embeddings back to base resolution (MergeDNA §4.3).

    Conceptually:
        If dense S[b,l,n] indicates membership of base position n in token l,
        then unmerge is equivalent to (S^T @ z_L) under a one-hot contiguous segmentation.

    Practical implementation:
        For each batch element b, repeat each z_L[b, l] exactly token_lens[b, l] times.

    Args:
        z_L:
            (B, L, D) merged token embeddings.
        token_lens:
            (B, L) int64 span lengths.
        N:
            desired base sequence length.

    Returns:
        z_N:
            (B, N, D) where each base position receives the embedding of the merged token
            that covers it. If sum(lens) != N, I truncate/pad deterministically.

    Notes:
        - This is the sparse equivalent of materialising S and multiplying by S^T.
        - It assumes contiguous spans (true for adjacent-pair merges).
    """
    if token_lens.dtype != torch.long:
        token_lens = token_lens.long()

    B, L, D = z_L.shape
    out = torch.zeros((B, N, D), device=z_L.device, dtype=z_L.dtype)

    for b in range(B):
        # repeat_interleave expects 1D repeats; token_lens[b] is (L,)
        zb = torch.repeat_interleave(z_L[b], token_lens[b], dim=0)  # (sum(lens_b), D)

        # Deterministic pad/truncate to exactly N.
        if zb.size(0) < N:
            out[b, : zb.size(0)] = zb
        else:
            out[b] = zb[:N]

    return out


def project_mask_to_base(mask_L: torch.Tensor, lengths_L: torch.Tensor, N: int) -> torch.Tensor:
    """
    Project a boolean mask from merged-token space (length L) to base-token space (length N).

    This is used in AMTM (§4.5):
        - sample a mask over merged/local tokens (mask_L)
        - project it back to base resolution to know which base tokens are masked

    Mechanism:
        For each token l, repeat mask_L[b,l] for lengths_L[b,l] base positions.

    Args:
        mask_L:
            (B, L) bool mask over merged tokens.
        lengths_L:
            (B, L) int64 span lengths (run-length encoding).
        N:
            output base length.

    Returns:
        mask_N:
            (B, N) bool mask over base positions, padded/truncated to exactly N.

    Assumptions:
        - contiguous spans (valid for adjacent merges)
        - lengths_L corresponds to the same segmentation used to produce z_L
    """
    if mask_L.dtype != torch.bool:
        raise TypeError(f"mask_L must be bool, got {mask_L.dtype}")
    if lengths_L.dtype != torch.long:
        lengths_L = lengths_L.long()

    B, L = mask_L.shape
    out = torch.zeros((B, N), dtype=torch.bool, device=mask_L.device)

    for b in range(B):
        # repeat_interleave needs 1D repeats; lengths_L[b] is (L,)
        mb = torch.repeat_interleave(mask_L[b], lengths_L[b], dim=0)  # (sum(lengths_b),)

        # Deterministic pad/truncate to exactly N.
        if mb.numel() < N:
            out[b, : mb.numel()] = mb
        else:
            out[b] = mb[:N]

    return out