import torch
import torch.nn.functional as F

from mergedna.amtm import AMTMMaskSampler
from mergedna.dna_vocab import VOCAB


class FakeLocalEncoder:
    def __call__(self, x_emb, target_L):
        B, N, D = x_emb.shape
        L = target_L
        z_L = torch.randn(B, L, D, device=x_emb.device, dtype=x_emb.dtype)
        lengths = torch.full((B, L), N // L, device=x_emb.device, dtype=torch.long)
        lengths[:, -1] += (N - lengths.sum(dim=1))
        return z_L, lengths


class FakeLatentEncoder:
    def __init__(self):
        self.merge_called = 0

    def __call__(self, z_L):
        return z_L

    def global_merge_to_K(self, z_L_ctx, K):
        self.merge_called += 1
        B, L, D = z_L_ctx.shape
        gid = torch.arange(L, device=z_L_ctx.device).unsqueeze(0).expand(B, L) % K
        z_K = torch.randn(B, K, D, device=z_L_ctx.device, dtype=z_L_ctx.dtype)
        return z_K, gid


class FakeModel:
    """
    Contract test for forward_amtm semantics:
      - uses latent grouping to DEFINE masks
      - runs prediction path WITHOUT latent merge (calls forward_reconstruct on masked input)
    """
    def __init__(self, vocab_size=6, d_model=8):
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.local_encoder = FakeLocalEncoder()
        self.latent_encoder = FakeLatentEncoder()
        self.reconstruct_called = 0
        self.last_reconstruct_x = None

    def forward_reconstruct(self, x_masked, target_L):
        self.reconstruct_called += 1
        self.last_reconstruct_x = x_masked.detach().clone()
        B, N = x_masked.shape
        V = self.embed.num_embeddings
        logits = torch.randn(B, N, V, device=x_masked.device)
        return logits, None

    def forward_amtm(self, x, *, target_L, target_K, mask_token_id, sampler_cfg=None):
        B, N = x.shape

        x_emb = self.embed(x)
        z_L, lengths_L = self.local_encoder(x_emb, target_L=target_L)
        z_L_ctx = self.latent_encoder(z_L)
        _, gid = self.latent_encoder.global_merge_to_K(z_L_ctx, K=target_K)

        sampler = AMTMMaskSampler(sampler_cfg)
        mask_L, mask_N = sampler(gid, lengths_L, N, K_groups=target_K, K_samples=target_K)

        x_masked = x.clone()
        x_masked[mask_N] = mask_token_id

        logits, _ = self.forward_reconstruct(x_masked, target_L=target_L)

        if mask_N.any():
            loss = F.cross_entropy(logits[mask_N], x[mask_N], reduction="mean")
        else:
            loss = torch.zeros((), device=x.device, dtype=logits.dtype)

        return loss, logits, mask_L, mask_N


def test_forward_amtm_calls_global_merge_once_and_reconstruct_once():
    torch.manual_seed(0)
    model = FakeModel(vocab_size=VOCAB.size, d_model=8)

    B, N = 2, 16
    x = torch.randint(0, 5, (B, N))  # exclude MASK id
    mask_id = VOCAB.MASK

    loss, logits, mask_L, mask_N = model.forward_amtm(x, target_L=8, target_K=4, mask_token_id=mask_id)

    assert model.latent_encoder.merge_called == 1
    assert model.reconstruct_called == 1
    assert logits.shape == (B, N, VOCAB.size)
    assert mask_N.shape == (B, N)
    assert mask_N.any().item() is True
    assert (model.last_reconstruct_x[mask_N] == mask_id).all().item() is True


def test_loss_mask_excludes_N_targets_logic_example():
    x = torch.tensor([[VOCAB.A, VOCAB.N, VOCAB.C, VOCAB.N]])
    mask_N = torch.tensor([[True, True, True, False]])
    loss_mask = mask_N & (x != VOCAB.N)
    assert torch.equal(loss_mask, torch.tensor([[True, False, True, False]]))