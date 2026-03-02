from dataclasses import dataclass

@dataclass
class MergeDNAConfig:
    # Base sequence length
    N: int = 4096

    # Model width
    d_model: int = 1024
    n_heads: int = 16
    ffn_mult: int = 4

    # Blocks (paper’s 380M config)
    n_local_enc: int = 4
    n_latent_enc: int = 20
    n_latent_dec: int = 4
    n_local_dec: int = 2

    # Local merge / attention settings
    local_window_size: int = 16
    # Compression defaults (paper’s reported L=N/2, K=L/2)
    L_avg: int = 2048
    K_avg: int = 1024

    # Loss weighting
    lambda_latent_mtr: float = 0.25