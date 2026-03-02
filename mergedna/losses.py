from __future__ import annotations
import torch
import torch.nn.functional as F

from mergedna.dna_vocab import VOCAB

def cross_entropy_over_bases(logits: torch.Tensor, targets: torch.Tensor, denom: int) -> torch.Tensor:
    """
    logits: (B,N,V), targets: (B,N)
    returns mean CE with explicit denominator control (paper uses 1/N or 1/K in places).
    """
    B, N, V = logits.shape
    loss = F.cross_entropy(logits.view(B * N, V), targets.view(B * N), reduction="sum")
    return loss / float(denom)

def sample_L_gaussian(N: int, device: torch.device) -> int:
    # Paper example: L ~ Gaussian centred at N/2 with L in [0.4N, 0.6N]. (Use clamp.)
    mean = 0.5 * N
    std = 0.05 * N
    L = torch.normal(mean=torch.tensor(mean, device=device), std=torch.tensor(std, device=device)).round().long()
    L = int(torch.clamp(L, int(0.4 * N), int(0.6 * N)).item())
    return max(L, 1)

def derive_am_tm_mask(group_of_token: torch.Tensor, K: int) -> torch.Tensor:
    """
    group_of_token: (B,L) values in [0..K-1]
    Returns token-level mask M_L: (B,L) with exactly K ones (tokens) sampled without replacement.
    """
    B, L = group_of_token.shape
    M_L = torch.zeros((B, L), device=group_of_token.device, dtype=torch.bool)

    for b in range(B):
        g = group_of_token[b]                           # (L,)
        counts = torch.bincount(g, minlength=K).float()  # (K,)
        # weight per token j is 1 / (g_i^2)
        w = 1.0 / counts.clamp_min(1.0)                 # (K,)
        token_w = (w[g] / counts[g].clamp_min(1.0))      # (L,) == 1/count^2
        p = token_w / token_w.sum().clamp_min(1e-12)

        idx = torch.multinomial(p, num_samples=min(K, L), replacement=False)
        M_L[b, idx] = True

    return M_L

def token_mask_to_base_mask(M_L: torch.Tensor, lengths_L: torch.Tensor, N: int) -> torch.Tensor:
    """
    Map token-level mask to base-level mask by repeating across token lengths:
      M_N = U(M_L, S)
    """
    M_N = torch.repeat_interleave(M_L.to(torch.long), lengths_L, dim=1).bool()
    return M_N[:, :N]