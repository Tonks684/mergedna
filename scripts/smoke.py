import torch
from mergedna.config import MergeDNAConfig
from mergedna.model import MergeDNA
from mergedna.dna_vocab import VOCAB

def main():
    cfg = MergeDNAConfig(
        d_model=32, n_heads=4,
        n_local_enc=1, n_latent_enc=1, n_latent_dec=1, n_local_dec=1,
        ffn_mult=4, N=64, local_window_size=16,
    )
    model = MergeDNA(cfg)
    x = torch.randint(0, VOCAB.size, (2, cfg.N))
    logits, _ = model.forward_reconstruct(x, target_L=cfg.N // 2)
    print("OK logits", logits.shape)

if __name__ == "__main__":
    main()