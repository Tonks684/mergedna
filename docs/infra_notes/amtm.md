# AMTM (Adaptive Masked Token Modelling) — Infra Notes

## Goal

AMTM is a third pretraining pass in MergeDNA. It uses the latent grouping structure S′ only to define *where to mask*, then predicts masked base tokens using the full model **without latent merging**.

We implement AMTM using:
- group assignments `group_of_token` from the global merge (4.4)
- sparse local segmentation representation `lengths_L` from local merge (4.3)
- explicit `[MASK]` token in the base vocabulary (`VOCAB.MASK = 5`)

## Data structures

Inputs needed for AMTM pass:
- `x`: (B, N) base tokens in {A,C,G,T,N,MASK}
- `lengths_L`: (B, L) run-lengths mapping merged tokens to base spans (sum ≈ N)
- `group_of_token`: (B, L) maps each local token to group id in [0..K-1]

Outputs:
- `mask_L`: (B, L) merged-token mask (exactly K positions True per row)
- `mask_N`: (B, N) base mask after projection

## Probability definition

Per sample b:
- group sizes: g_i = bincount(group_of_token[b], minlength=K)
- token-level group sizes: g_tok[j] = g_{ group_of_token[b,j] }

Sampling distribution:
- P_L(j) ∝ 1 / (g_tok[j]^2)

We sample exactly K tokens without replacement using `torch.multinomial`.

## Mask projection (merged → base)

We avoid dense S by using run-length encoding:
- `mask_N = repeat_interleave(mask_L, lengths_L, dim=1)[:, :N]`

This masks all base positions spanned by selected merged tokens.

## Input corruption & targets

We replace masked base positions with:
- `VOCAB.MASK` (explicit mask token)

Loss:
- cross entropy computed only on masked base positions.
- Recommended: exclude ambiguous targets `VOCAB.N` because N denotes unknown bases; predicting it provides weak learning signal.

Loss mask:
- `loss_mask = mask_N & (x != VOCAB.N)`  (optional but recommended)

## AMTM forward semantics (important)

AMTM MUST run the prediction pass **without** latent merging. The global merge / grouping is used only to define the mask.
Therefore:
- Mask selection uses `global_merge_to_K(...)` to get `group_of_token`
- Prediction uses `forward_reconstruct(x_masked, target_L)` (no global merge)

## Testing considerations

Unit tests should verify:

### 1) Distribution correctness
- Tokens in smaller groups have higher probability:
  - if g0 < g1 then P(token in group0) > P(token in group1)

### 2) Exact-K sampling
- `mask_L[b].sum() == K` for each b (unless K > L)

### 3) Projection correctness
- `mask_N` aligns with spans defined by `lengths_L`
- For a hand-constructed lengths example, compare to expected base mask

### 4) Forward semantics
- The AMTM prediction pass does not call the latent merge path; it calls the normal reconstruction pass.
- This can be tested with a spy/mock model that counts calls.

### 5) MASK token application
- In the tensor passed to reconstruction, positions where `mask_N` is True must equal `VOCAB.MASK`.

### 6) Optional: excluding VOCAB.N targets
- If excluding N targets, verify that loss is computed on `mask_N & (x != N)` only.