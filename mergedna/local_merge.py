
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class LocalMergeConfig:
    window_size: int = 16
    group_dim: int = 64

class LocalTokenMerger(nn.Module):
    def __init__(self, d_model: int, cfg: LocalMergeConfig):
        super().__init__()
        self.cfg = cfg
        self.group = nn.Sequential(
            nn.Linear(d_model, cfg.group_dim, bias=False),
            nn.ReLU(),
            nn.Linear(cfg.group_dim, cfg.group_dim, bias=False),
        )

    @torch.no_grad()
    def _choose_nonoverlap_pairs(self, sim_adj: torch.Tensor, merge_budget: int) -> torch.Tensor:
        """
        sim_adj: (W-1,) similarity for edges (i,i+1) inside a window
        returns merge_right: (W,) bool where True at j means merge token j into j-1
        """
        W = sim_adj.numel() + 1
        merge_right = torch.zeros(W, dtype=torch.bool, device=sim_adj.device)
        used = torch.zeros(W, dtype=torch.bool, device=sim_adj.device)

        order = torch.argsort(sim_adj, descending=True)
        picked = 0
        for e in order.tolist():
            if picked >= merge_budget:
                break
            i, j = e, e + 1
            if used[i] or used[j]:
                continue
            merge_right[j] = True
            used[i] = True
            used[j] = True
            picked += 1
        return merge_right

    def forward(self, z: torch.Tensor, token_lens: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z: (B,T,D), token_lens: (B,T) int64
        returns: z_new (B,target_len,D), lens_new (B,target_len)
        """
        B, T, D = z.shape
        if target_len >= T:
            starts = torch.cumsum(token_lens, dim=1) - token_lens
            return z, token_lens, starts
        merges_needed = T - target_len
        W = self.cfg.window_size

        g = self.group(z)  # (B,T,G)
        g = g / (g.norm(dim=-1, keepdim=True) + 1e-8)
        sim_all = (g[:, :-1] * g[:, 1:]).sum(dim=-1)  # (B,T-1)

        z_out, lens_out = [], []
        for b in range(B):
            zb, lb = z[b], token_lens[b]
            keep = torch.ones(T, dtype=torch.bool, device=z.device)
            merge_right = torch.zeros(T, dtype=torch.bool, device=z.device)

            nwin = (T + W - 1) // W
            base = merges_needed // nwin
            extra = merges_needed % nwin

            for widx in range(nwin):
                s = widx * W
                e = min((widx + 1) * W, T)
                if e - s <= 1:
                    continue
                budget = base + (1 if widx < extra else 0)
                budget = min(budget, (e - s) // 2)

                if budget <= 0:
                    continue

                local = sim_all[b, s : e - 1]
                mr = self._choose_nonoverlap_pairs(local, budget)  # (e-s,)
                idx = torch.nonzero(mr, as_tuple=False).flatten() + s
                merge_right[idx] = True
                keep[idx] = False

            # merge: right token -> left token
            zb2 = zb.clone()
            lb2 = lb.clone()
            idxs = torch.nonzero(merge_right, as_tuple=False).flatten()
            for j in idxs.tolist():
                i = j - 1
                if i < 0:
                    continue
                # Weighted average based on token length (more faithful reconstruction)
                w_i = lb2[i].to(zb2.dtype)
                w_j = lb2[j].to(zb2.dtype)
                zb2[i] = (zb2[i] * w_i + zb2[j] * w_j) / (w_i + w_j)
                lb2[i] = lb2[i] + lb2[j]

            zb_new = zb2[keep]
            lb_new = lb2[keep]

            # enforce exact length deterministically
            zb_new = zb_new[:target_len]
            lb_new = lb_new[:target_len]
            z_out.append(zb_new)
            lens_out.append(lb_new)

        z_new = torch.stack(z_out, 0)
        lens_new = torch.stack(lens_out, 0)
        starts_new = torch.cumsum(lens_new, dim=1) - lens_new
        return z_new, lens_new, starts_new

def unmerge_tokens(z_L: torch.Tensor, token_lens: torch.Tensor, N: int) -> torch.Tensor:
    """
    z_L: (B,L,D), token_lens: (B,L) int64
    returns z_N: (B,N,D) by repeating each token embedding over its base span.
    """
    if token_lens.dtype != torch.long:
        token_lens = token_lens.long()

    B, L, D = z_L.shape
    out = torch.zeros((B, N, D), device=z_L.device, dtype=z_L.dtype)

    for b in range(B):
        zb = torch.repeat_interleave(z_L[b], token_lens[b], dim=0)  # (sum(lens_b), D)
        if zb.size(0) < N:
            out[b, : zb.size(0)] = zb
        else:
            out[b] = zb[:N]
    return out

def project_mask_to_base(mask_L: torch.Tensor, lengths_L: torch.Tensor, N: int) -> torch.Tensor:
    """
    Project a merged-token mask (B,L) to a base-resolution mask (B,N) using token lengths.

    mask_L:    (B,L) bool
    lengths_L: (B,L) int64
    N:         base sequence length

    Returns:
      mask_N: (B,N) bool
    """
    if mask_L.dtype != torch.bool:
        raise TypeError(f"mask_L must be bool, got {mask_L.dtype}")
    if lengths_L.dtype != torch.long:
        lengths_L = lengths_L.long()

    B, L = mask_L.shape
    out = torch.zeros((B, N), dtype=torch.bool, device=mask_L.device)

    for b in range(B):
        # repeat_interleave needs 1D repeats
        mb = torch.repeat_interleave(mask_L[b], lengths_L[b], dim=0)  # (sum(lengths_b),)
        if mb.numel() < N:
            out[b, : mb.numel()] = mb
        else:
            out[b] = mb[:N]
    return out