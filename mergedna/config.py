from __future__ import annotations
from dataclasses import dataclass

@dataclass
class MergeDNAConfig:
    """
    MergeDNA 380M-ish configuration (paper Section 4; 30 Transformer blocks total)
    plus a small number of training hyperparameters.

    Block counts:
      n_local_enc (4) + n_latent_enc (20) + n_latent_dec (4) + n_local_dec (2) = 30 blocks

    Why this is ~380M parameters (back-of-the-envelope):
      We assume a standard Transformer encoder block with:
        - Multi-head self-attention implemented as 4 dense projections:
            Wq, Wk, Wv, Wo ∈ R^{d_model x d_model}
        - A 2-layer FFN (MLP) with hidden size d_ff = ffn_mult * d_model

      With d_model = 1024 and ffn_mult = 4 => d_ff = 4096:

      Attention params per block (ignoring biases):
        4 * (1024 * 1024) = 4,194,304  (~4.2M)

      FFN params per block (ignoring biases):
        (1024 * 4096) + (4096 * 1024) = 8,388,608  (~8.4M)

      Total per block:
        ~12.6M

      Total across 30 blocks:
        30 * 12.6M ≈ 378M

      Remaining parameters (LayerNorm/RMSNorm, embeddings, small merge/grouping projections)
      bring the total close to the paper's “~380M” setting.

    Assumptions / notes:
      - This estimate assumes a standard 2-layer FFN (no gated MLP like SwiGLU).
      - n_heads=16 is chosen because d_model is divisible by n_heads and yields head_dim=64,
        which is common and efficient (FlashAttention-friendly).
      - No weight sharing between stacks (local/latent encoder/decoder are independent).
      - Exact parameter count depends on implementation details (biases, norm type, vocab size).
    """

    # Base sequence length
    N: int = 4096

    # Model width
    d_model: int = 1024  # paper setting
    n_heads: int = 16    # assumption -> head_dim = 1024/16 = 64
    ffn_mult: int = 4    # assumption -> d_ff = 4096

    # Block counts (paper’s ~380M setting)
    n_local_enc: int = 4
    n_latent_enc: int = 20
    n_latent_dec: int = 4
    n_local_dec: int = 2

    # Local merge / attention settings
    local_window_size: int = 16  # paper setting
    # Compression defaults (paper reports L≈N/2, K≈L/2)
    L_avg: int = 2048
    K_avg: int = 1024

    # Loss weighting (paper: λ = 0.25)
    lambda_latent_mtr: float = 0.25


def count_params(model) -> int:
    """Total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)