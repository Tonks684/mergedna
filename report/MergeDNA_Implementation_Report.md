# MergeDNA Implementation Report

---

## 1. Implementation Overview

This implementation treats the [NanoChat](https://github.com/karpathy/nanochat) codebase as an execution-layer infrastructure for sequence modelling, rather than as a fixed language modelling architecture.

MergeDNA introduces modelling innovations primarily at the level of sequence representation, including genomic patch embeddings and latent bottleneck tokens, without fundamentally altering the underlying attention mechanism. Re-implementing a transformer stack from scratch would therefore introduce unnecessary variance into the reproduction and divert engineering effort toward already-solved systems problems (e.g. optimized attention execution, distributed training orchestration, and checkpointing logic).

Instead, NanoChat was adopted as a modular infrastructure layer providing:

- A compact and efficient Transformer core with integrated unified SDPA attention (enabling runtime FlashAttention-2/FA3 dispatch)
- Rotary positional embeddings
- Distributed training patterns (`torchrun`-compatible)
- Standardized checkpointing and logging conventions

This allowed the implementation effort to focus on faithfully reproducing the architectural contributions of MergeDNA while preserving hardware-efficient execution for long genomic sequences.

---

## 2. Architectural Requirements of MergeDNA

MergeDNA proposes a hierarchical, autoencoder-style Transformer for genome modelling that:

- Learns a *dynamic tokenizer* via differentiable token merging under local-window constraints
- Learns a global context model over merged tokens using a latent Transformer

At a high level, the architecture consists of four modules:

| Module | Function |
|--------|----------|
| Local Encoder *(Eϕ)* | Local-window attention + token merging → produces merged-token sequence Z_L |
| Latent Encoder *(Eψ)* | Full-attention global Transformer |
| Latent Decoder *(Eω)* | Reconstruction (pre-training only) |
| Local Decoder *(Eζ)* | Detokenisation back to base resolution |

---

### 2.1 Representation Requirements

MergeDNA introduces modelling requirements not natively supported by NanoChat:

- Patch-like genomic representations (implemented via differentiable local token merging)
- Segmentation-tracking structure (S)
- Hierarchical compression of sequence length

---

### 2.2 Attention Requirements

- Non-causal Transformer encoder blocks  
- Local-window self-attention  
- Global full-attention latent encoder  

---

### 2.3 Training Requirements

Pre-training is driven by three forward passes:

- Merged Token Reconstruction (MTR)
- Latent MTR (with frozen Local Encoder parameters)
- Adaptive Masked Token Modelling (AMTM)

Combined objective:
```
L_total = L_MTR(θ) + λ L_MTR(θ \ {ϕ}) + L_AMTM(θ)
```
Where: λ = 0.25

---

## 3. Infrastructure Adaptation Strategy

The implementation separates modelling infrastructure from domain-specific representation:

### NanoChat Responsibilities

- Transformer execution
- Unified SDPA attention
- Distributed training
- Checkpointing / logging

### MergeDNA Responsibilities


- Token merging
- Latent compression
- Hierarchical reconstruction
- Adaptive masking


All architectural changes were introduced at the representation and input-interface level while preserving the underlying transformer execution path.

---

## 4. MergeDNA Module Implementations


### Overview of Implementation
The following subsections (4.1-4.5) describe each architectural component and its corresponding implementation details within the `mergedna/`. Each component is validated via section-aligned unit tests (see README.md Test ↔ Report Section Mapping)

```
Base Tokens (N)
   ↓ (4.1 / 4.2)
Local Encoder + Token Merge (4.3)
   ↓  N → L
Latent Encoder + Global Merge (4.4)
   ↓  L → K
Latent Decoder
   ↓
Local Decoder + Unmerge (4.3)
   ↓
Base Logits (N)

AMTM (4.5): Mask sampling via 1/g_i^2 from latent grouping.
```

Each of the following subsections describes an architectural modification required to support MergeDNA within the NanoChat execution stack.

---

### 4.1 Non-Causal Transformer Execution

MergeDNA pre-training objectives (MTR, AMTM) require reconstruction-style attention rather than autoregressive next-token prediction.

This was implemented by introducing a non-causal attention execution path via the MergeDNA attention backend.

Implementation location:

- `mergedna/flash_attention_patch.py::flash_attn_func`

NanoChat’s FlashAttention-backed attention path is reused where available, with causal masking disabled:

```
F.scaled_dot_product_attention(
    q, k, v,
    is_causal=False
)
```

This preserves:

- FlashAttention-2/FA3 runtime dispatch
- Rotary embedding compatibility
- Distributed execution support

while removing autoregressive masking semantics.

---
### 4.2 Local-Window Attention Module

MergeDNA’s Local Encoder and Local Decoder operate under fixed-window locality constraints (window size = 16 in the reported configuration).

Sliding-window attention support was introduced via the same attention backend by passing a local attention window.

Implementation locations:

- `mergedna/flash_attention_patch.py::flash_attn_func`
- `mergedna/flash_attention_patch.py::_sdpa_bool_mask`

Where NanoChat’s FlashAttention backend is unavailable, a boolean SDPA mask is constructed to enforce locality:
```
mask = _sdpa_bool_mask(
    Tq, Tk,
    causal=causal,
    window_size=window_size,
    device=qh.device
)
```
Trade-off:

- Enables token-merging locality constraints
- Reduces global receptive field within Local Encoder blocks
- Optimized FlashAttention execution is preserved when available via NanoChat’s runtime dispatch.
---
#### 4.3 Token Merging + Segmentation Mapping

MergeDNA’s Local Encoder performs local token merging and produces both a compressed token sequence $(Z_L)$ and a segmentation-tracking structure $(S)$ that supports reconstruction and mask projection back to base resolution.

Rather than materialising the paper’s dense binary source matrix $(S \in \{0,1\}^{L \times N})$, this implementation uses a sparse run-length encoding of the segmentation, which is sufficient to support:

1) **Unmerge** of embeddings back to base resolution  
2) **Projection** of merged-token masks back to base masks (required by AMTM)

#### Implementation locations

- `mergedna/local_merge.py::LocalTokenMerger`  
- `mergedna/local_merge.py::unmerge_tokens`  
- `mergedna/local_merge.py::project_mask_to_base`

#### Representation of segmentation (sparse “S”)

Segmentation is represented by two per-token vectors:

- `token_lens` $(\in \mathbb{Z}^{B \times L})$: base span length covered by each merged token  
- `token_starts` $(\in \mathbb{Z}^{B \times L})$: base start index for each merged token span  

This representation is memory-efficient $(O(L)$ rather than $O(LN))$ and fully defines a contiguous segmentation under adjacent-pair merges.


#### Merge behaviour

Within each local window, the merger computes adjacency similarity in a learned grouping space and selects a set of **non-overlapping adjacent pairs** to merge, subject to a target length \(L\) (merge budget). Merges are applied as a span-weighted average:

- right token is merged into the left token embedding
- lengths are accumulated to preserve span accounting (`token_lens`)

#### Unmerge and mask projection

- **Unmerge:** `unmerge_tokens(z_L, token_lens, N)` repeats each merged embedding over its base span  
- **Mask projection:** `project_mask_to_base(mask_L, token_lens, N)` repeats merged-token boolean masks over their base spans

#### Trade-offs

- **Pros:** avoids dense \(S\); deterministic and efficient unmerge/mask projection; clean integration into training loops.  
- **Cons:** assumes merged groups are contiguous spans (valid for adjacent merges); if later variants introduce non-contiguous grouping, additional metadata would be required.

---

### 4.4 Latent Bottleneck Pipeline (Global Merge L → K + Unmerge K → L)

MergeDNA’s latent reconstruction and AMTM objectives require a global compression stage over the locally-merged sequence $(Z_L)$, producing a shorter sequence $(Z_K)$ with $(K < L))$, together with a grouping structure $(S')$ that enables computation of group sizes for importance-based masking.

This implementation introduces a global merge operator that compresses $(L \rightarrow K)$ and returns a per-token group assignment vector that acts as a sparse representation of $(S')$.

#### Implementation locations

- `mergedna/model.py::LatentEncoder.global_merge_to_K`
- `mergedna/model.py::MergeDNA.forward_latent_reconstruct`

#### Interface and outputs

Given contextualised local tokens $(z_{L,\mathrm{ctx}} \in \mathbb{R}^{B \times L \times D})$:

- `global_merge_to_K(z_L_ctx, K)` returns:
  - `z_K` of shape `(B, K, D)` — global latent “salient” tokens
  - `group_of_token` of shape `(B, L)` — maps each local token index to a group id in `[0, K-1]`

This `group_of_token` provides the required grouping structure to compute per-group sizes $(g_i)$ for AMTM without materialising a dense $(S')$.

#### Merge algorithm (current approximation)

The paper describes a ToMe-style global merge. The current implementation provides a faithful *interface* and training semantics using an anchor-based hard clustering approximation:

1. Project each token into a grouping space using a lightweight linear map:
   - `g_raw = Linear(z_L_ctx)`
2. Select $(K)$ anchors per batch as tokens with highest grouping-space norm:
   - `topk(scores)`
3. Assign each token to its nearest anchor by cosine similarity:
   - `group_of_token = argmax(sim)`
4. Compute each latent token embedding as the mean of its assigned members via `scatter_add_`.

The assignment step is performed under `no_grad` (discrete), while the computation of `z_K` remains differentiable w.r.t. `z_L_ctx`.

#### Unmerge K → L for latent reconstruction

For the latent reconstruction pass, MergeDNA requires unmerging $(Z_K)$ back to length $(L)$ before decoding. This is implemented as a hard unmerge using the group assignments:

- `z_bar_L[l] = z_K[group_of_token[l]]`

implemented via `torch.gather`, producing `z_bar_L` of shape `(B, L, D)`.

#### Trade-offs

- **Pros:** provides the required $(L \rightarrow K)$ bottleneck, returns an explicit grouping structure for AMTM, keeps the rest of the architecture modular and swappable (ToMe can later replace this merge operator without changing downstream interfaces).
- **Cons:** anchor-based grouping is an approximation of ToMe-style merging; group assignments are discrete and not differentiable. This is acceptable for the current scaffold, and the merge operator can be swapped to a ToMe implementation later while preserving the same `z_K` + `group_of_token` interface.
---

#### 4.5 Adaptive Masked Token Modelling (AMTM)

AMTM defines a masking objective over base-resolution tokens, where the mask is adaptively sampled based on the latent merge grouping structure $(S')$. The key idea is to preferentially mask tokens that are less redundantly represented under the latent grouping.

#### Implementation locations

- `mergedna/amtm.py`
    - `compute_amtm_probs_from_groups`
    - `sample_exact_k_tokens`
    - `AMTMMaskSampler`
- `mergedna/model.py::MergeDNA.forward_amtm`

#### Group-driven mask distribution

Given latent grouping assignments `group_of_token ∈ [0..K-1]^(B×L)` produced by the global merge (Section 4.4), group sizes are computed per sample:

- $(g_i = \text{count}(\text{group\_of\_token} = i))$

AMTM assigns per-token sampling probabilities over the L local tokens:

- $(w_i = 1 / g_i)$
- $(P_L(j) ∝ w_i / g_i = 1/g_i^2)$ for token $(j)$ in group $(i)$

Exactly $(K)$ local tokens are sampled *without replacement* according to $(P_L)$, yielding a merged-token mask `mask_L`.

#### Projection to base resolution

The merged-token mask `mask_L` is projected to base resolution `mask_N` using the sparse segmentation representation from Section 4.3 (run-length encoding via `lengths_L`), avoiding dense materialisation of the source matrix $(S)$.

#### AMTM forward pass semantics (no latent merge)

Per the paper, AMTM uses the latent grouping only to define the mask, and the prediction forward pass is run **without latent token merging**:

- `x_masked → forward_reconstruct(x_masked, target_L)`  

Masked base tokens are replaced with the explicit `[MASK]` token (`VOCAB.MASK`). The AMTM loss is computed as cross-entropy **only at masked base positions** (optionally excluding ambiguous targets `VOCAB.N`).

#### Trade-offs

- **Pros:** importance sampling focuses supervision on informative (less-merged) regions; grouping-derived masks are consistent with the model’s hierarchical structure.
- **Cons:** introduces additional sampling and mask-projection complexity; objective becomes dependent on the quality of latent grouping $(S')$.


The correctness of Sections 4.1-4.5 is validated via unit tests aligned explicitly to each architectural component (see repository Test ↔ Report Section Mapping)
---

## 5. Training Pipeline Adaptation

Training loop modified to support:

- Three forward-pass objective computation  
- Frozen Local Encoder parameters during latent-MTR  
- λ-weighted latent reconstruction loss  

Compression-ratio sampling implemented by dynamically sampling L per iteration: $L ∼ 𝒩(N/2, σ²)$

---

## 6. Execution Guarantees Preserved

The following NanoChat execution guarantees were preserved:

- Unified SDPA attention dispatch (FA2/FA3)
- torchrun distributed compatibility
- Checkpoint format
- Training loop structure
- Rotary positional embeddings

---

## 7. Design Trade-offs

| Design Choice | Benefit | Cost |
|--------------|--------|------|
| Sparse S mapping | Memory efficiency | Explicit unmerge required |
| Local window attention | Merge locality | Reduced global receptive field |
| Latent token compression | Context modelling | Reconstruction fidelity |

---

## 8. Extensibility

This modular separation enables:

- Alternate token merging strategies  
- Different latent compression ratios  
- Downstream task-specific decoders  
- Integration with alternate tokenisation pipelines  

Architectural experiments on latent sequence representations can therefore be conducted without re-engineering the underlying transformer execution stack.

---
## 9. Verification & Reproducibility
Verification is provided via unit tests aligned to Sections 4.1–4.5 and the training contract in Section 5.
See `README.md` (Test ↔ Report Section Mapping) and `tests/`.

Optional training infrastructure (logging/checkpointing/W&B) is described in `README.md` and does not modify the MergeDNA modelling pipeline.
