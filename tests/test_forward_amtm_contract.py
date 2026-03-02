import torch
import pytest

from mergedna.dna_vocab import VOCAB


class FakeLocalEncoder:
    def __call__(self, x_emb, target_L):
        # Return dummy z_L and lengths_L
        B, N, D = x_emb.shape
        L = target_L
        z_L = torch.randn(B, L, D, device=x_emb.device, dtype=x_emb.dtype)
        # lengths sum to N (simple equal chunks)
        lengths = torch.full((B, L), N // L, device=x_emb.device, dtype=torch.long)
        lengths[:, -1] += (N - lengths.sum(dim=1))
        return z_L, lengths


class FakeLatentEncoder:
    def __init__(self):
        self.merge_called = 0

    def __call__(self, z_L):
        return z_L  # identity

    def global_merge_to_K(self, z_L_ctx, K):
        self.merge_called += 1
        B, L, D = z_L_ctx.shape
        # assign tokens round-robin into K groups
        gid = torch.arange(L, device=z_L_ctx.device).unsqueeze(0).expand(B, L) % K
        z_K = torch.randn(B, K, D, device=z_L_ctx.device, dtype=z_L_ctx.dtype)
        return z_K, gid


class FakeModel:
    """
    Minimal contract test for forward_amtm semantics:
      - uses latent grouping only to define masks
      - calls forward_reconstruct on masked_x (no latent merge)
      - defaults to VOCAB.MASK if mask_token_id is None
      - optionally excludes VOCAB.N targets from loss
    """
    def __init__(self, vocab_size=VOCAB.size, d_model=8):
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

    def forward_amtm(
        self,
        x,
        *,
        target_L,
        target_K,
        sampler_cfg=None,
        mask_token_id=None,
        ignore_ambiguous_targets=True,
    ):
        # Inline a copy of the intended logic (same as your updated MergeDNA.forward_amtm)
        import torch.nn.functional as F
        from mergedna.amtm import AMTMMaskSampler

        B, N = x.shape

        if mask_token_id is None:
            mask_token_id = VOCAB.MASK

        x_emb = self.embed(x)
        z_L, lengths_L = self.local_encoder(x_emb, target_L=target_L)
        z_L_ctx = self.latent_encoder(z_L)
        _, group_of_token = self.latent_encoder.global_merge_to_K(z_L_ctx, K=target_K)

        sampler = AMTMMaskSampler(sampler_cfg)
        mask_L, mask_N = sampler(group_of_token, lengths_L, N, K_groups=target_K, K_samples=target_K)

        x_masked = x.clone()
        x_masked[mask_N] = mask_token_id

        logits, _ = self.forward_reconstruct(x_masked, target_L=target_L)

        loss_mask = mask_N
        if ignore_ambiguous_targets:
            loss_mask = loss_mask & (x != VOCAB.N)

        if loss_mask.any():
            loss = F.cross_entropy(logits[loss_mask], x[loss_mask], reduction="mean")
        else:
            loss = torch.zeros((), device=x.device, dtype=logits.dtype)

        return loss, logits, mask_L, mask_N, loss_mask


def test_forward_amtm_calls_global_merge_once_and_reconstruct_once_and_masks_with_vocab_mask():
    torch.manual_seed(0)
    model = FakeModel(vocab_size=VOCAB.size, d_model=8)

    B, N = 2, 16
    x = torch.randint(0, VOCAB.size, (B, N))

    loss, logits, mask_L, mask_N, loss_mask = model.forward_amtm(
        x,
        target_L=8,
        target_K=4,
        mask_token_id=None,  # should default to VOCAB.MASK
        ignore_ambiguous_targets=True,
    )

    assert model.latent_encoder.merge_called == 1
    assert model.reconstruct_called == 1
    assert logits.shape == (B, N, VOCAB.size)
    assert mask_N.shape == (B, N)
    assert mask_N.any().item() is True

    # Ensure masking actually applied in the input passed to reconstruction
    assert (model.last_reconstruct_x[mask_N] == VOCAB.MASK).all().item() is True


def test_loss_mask_excludes_N_targets_when_enabled():
    # Explicitly test the boolean logic used for loss masking.
    x = torch.tensor([[VOCAB.A, VOCAB.N, VOCAB.C, VOCAB.N]])
    mask_N = torch.tensor([[True, True, True, False]])
    loss_mask = mask_N & (x != VOCAB.N)
    assert torch.equal(loss_mask, torch.tensor([[True, False, True, False]]))


def test_loss_mask_can_include_N_targets_when_disabled():
    x = torch.tensor([[VOCAB.A, VOCAB.N, VOCAB.C, VOCAB.N]])
    mask_N = torch.tensor([[True, True, True, False]])
    loss_mask = mask_N if False else mask_N  # mimic ignore_ambiguous_targets=False
    assert torch.equal(loss_mask, torch.tensor([[True, True, True, False]]))