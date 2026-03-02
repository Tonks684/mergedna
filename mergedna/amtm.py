from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from mergedna.local_merge import project_mask_to_base


@dataclass
class AMTMCfg:
    """
    AMTM configuration.
    - power=2 corresponds to P_L(j) ∝ 1 / g_i^2 as described in the paper.
    """
    power: float = 2.0
    eps: float = 1e-12


def compute_amtm_probs_from_groups(group_of_token: torch.Tensor, K: int, *, power: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute AMTM sampling probabilities over L tokens given group assignments.

    group_of_token: (B,L) int64, values in [0..K-1]
    returns probs: (B,L) float, rows sum to 1
      P(j) ∝ 1 / g(group(j))^power
    """
    if group_of_token.dtype != torch.long:
        group_of_token = group_of_token.long()

    B, L = group_of_token.shape
    device = group_of_token.device

    probs = torch.empty((B, L), device=device, dtype=torch.float32)
    for b in range(B):
        gid = group_of_token[b]  # (L,)
        counts = torch.bincount(gid, minlength=K).to(torch.float32)  # (K,)
        g_tok = counts[gid].clamp_min(1.0)  # (L,)
        p = 1.0 / (g_tok ** power)
        p = p / (p.sum() + eps)
        probs[b] = p
    return probs


def sample_exact_k_tokens(probs: torch.Tensor, K_samples: int) -> torch.Tensor:
    """
    Sample exactly K_samples token indices without replacement for each batch row.

    probs: (B,L) nonnegative, rows sum to 1 (or close)
    returns mask_L: (B,L) bool with exactly K_samples True per row (unless K_samples > L).
    """
    B, L = probs.shape
    K_eff = min(K_samples, L)

    mask_L = torch.zeros((B, L), dtype=torch.bool, device=probs.device)
    for b in range(B):
        p = probs[b].clamp_min(0.0)
        # If numerical issues lead to all zeros, fall back to uniform.
        if float(p.sum().item()) <= 0.0:
            p = torch.ones_like(p) / float(L)
        else:
            p = p / p.sum()

        idx = torch.multinomial(p, num_samples=K_eff, replacement=False)  # (K_eff,)
        mask_L[b, idx] = True
    return mask_L


class AMTMMaskSampler:
    """
    End-to-end AMTM mask sampler:
      group_of_token (B,L), lengths_L (B,L), base length N -> mask_L (B,L), mask_N (B,N)
    """
    def __init__(self, cfg: Optional[AMTMCfg] = None):
        self.cfg = cfg or AMTMCfg()

    @torch.no_grad()
    def __call__(self, group_of_token: torch.Tensor, lengths_L: torch.Tensor, N: int, *, K_groups: int, K_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        group_of_token: (B,L) int64 group ids
        lengths_L: (B,L) int64 base-span lengths for each merged token
        N: base sequence length
        K_groups: number of groups (target_K)
        K_samples: number of local tokens to mask (paper uses K)
        returns:
          mask_L: (B,L) bool
          mask_N: (B,N) bool projected to base resolution
        """
        probs = compute_amtm_probs_from_groups(
            group_of_token,
            K_groups,
            power=self.cfg.power,
            eps=self.cfg.eps
        )
        mask_L = sample_exact_k_tokens(probs, K_samples)

        # Project merged-token mask to base tokens via sparse segmentation representation.
        mask_N = project_mask_to_base(mask_L, lengths_L, N)
        return mask_L, mask_N