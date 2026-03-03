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

- Genomic patch embeddings (implemented via differentiable local token merging)
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
The following subsections (4.1-4.5) describe each architectural component and its corresponding implementation details within the `mergedna/` module. Each component is validated via section-aligned unit tests (see README.md Test ↔ Report Section Mapping)

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

### 4.3 Token Merging + Segmentation Mapping (Layer-wise)

MergeDNA’s Local Encoder performs **progressive, layer-wise token merging**, interleaved with local self-attention blocks.

Rather than applying a single merge operation after all local layers, this implementation interleaves:

Local attention blocks → merge → local attention → merge → ... → final $Z_L$

At each merge stage, the sequence length is reduced according to a deterministic merge schedule that ensures the final length equals the requested target $L$. This preserves the hierarchical tokenization dynamics described in the paper.

---

#### Sparse Segmentation Representation

The paper defines a dense source matrix:

$$
S \in \{0,1\}^{L \times N}
$$

mapping base tokens to merged tokens.

Materializing this matrix is impractical for long sequences (e.g., $N = 4096$).  
Instead, this implementation uses a **sparse run-length encoding** that fully determines the same mapping:

- `token_lens ∈ ℤ^{B × L}` — span length of each merged token  
- `token_starts ∈ ℤ^{B × L}` — starting base index of each span  

This representation is:

- Memory-efficient: $O(L)$ rather than $O(LN)$  
- Exact for contiguous merges  
- Composable across multiple merge stages  

Unit tests validate equivalence to a dense $S$ reference for small sequences.


---

#### Merge Behaviour (Per Stage)

Within each merge stage:

1. Tokens are projected into a learned grouping space.
2. Adjacent token similarities are computed within fixed local windows.
3. A set of **non-overlapping adjacent pairs** is selected (merge budget).
4. Each selected right token is merged into its left neighbor using span-weighted averaging:
   - embeddings combined proportionally to `token_lens`
   - lengths accumulated

Non-overlapping merges ensure:

- Each base token belongs to exactly one merged token.
- Segmentation structure remains well-defined.
- Progressive merging remains stable across layers.

Although the MergeDNA paper does not explicitly state “non-overlapping” constraints, this is inherent in pairwise merging semantics and aligns with the ToMe merging strategy.

---

#### Unmerge and Mask Projection

The sparse representation supports:

**Unmerge**

`unmerge_tokens(z_L, token_lens, N)`

Repeats each merged embedding over its base span.

**Mask projection**

`project_mask_to_base(mask_L, token_lens, N)`

Repeats merged-token masks over their spans.

Both operations are exact under contiguous span merges.

---

#### Trade-offs

| Design choice | Benefit | Cost |
|---------------|---------|------|
| Sparse span encoding instead of dense $S$ | Memory efficient and simple | Assumes contiguous merges |
| Layer-wise progressive merging | Closer to paper, hierarchical dynamics | Slightly more complex control flow |
| Deterministic merge schedule | Stable and reproducible | Fixed schedule rather than adaptive per-layer $r_\ell$ |

## 4.4 Latent Bottleneck Pipeline (Global Merge L → K + Unmerge K → L)

MergeDNA’s latent reconstruction and AMTM objectives require a global compression stage over the locally merged sequence $Z_L$, producing a shorter latent sequence $Z_K$ with $K < L$, together with a grouping structure $S'$ used for reconstruction and importance-based masking.

This implementation provides a faithful interface and training semantics for the latent bottleneck while approximating the paper’s ToMe-style merge with a deterministic anchor-based clustering procedure.

### Implementation Locations

- `mergedna/model.py::LatentEncoder.global_merge_to_K`
- `mergedna/model.py::MergeDNA.forward_latent_reconstruct`

---

### Interface and Outputs

Given contextualized local tokens:

$$
z_{L,\mathrm{ctx}} \in \mathbb{R}^{B \times L \times D}
$$

`global_merge_to_K(z_L_ctx, K)` returns:

- `z_K` of shape `(B, K, D)` — compressed latent tokens  
- `group_of_token` of shape `(B, L)` — integer group assignments in `[0, K-1]`

`group_of_token` is a sparse representation of the paper’s grouping structure $S'$:  
each local token belongs to exactly one latent group.

This structure is later used to compute group sizes $g_i$ for AMTM.

---

### Merge Algorithm (Anchor-Based Hard Clustering)

The paper describes a ToMe-style global merging process. Here, I implement a simpler but semantically equivalent bottleneck with the following steps:

#### 1. Learned Grouping Projection

Each local token is projected into a lower-dimensional grouping space:

$$
g_{\text{raw}} = W_g z_{L,\mathrm{ctx}}
$$

The magnitude of this projection

$$
\|g_{\text{raw}}\|
$$

is interpreted as a learned saliency score. Tokens with larger magnitude are treated as structurally important candidates for latent anchors.
Each column of $W_g$ can be viewed as detecting some structural pattern in the contextualised embedding eg. long-range interaction strength or motif boundary signals. If a token strongly activates many learned grouping directions $g_{raw}$ will have large magnitude thus norm measures how strongly the token activies learned grouping subspaces.

#### 2. Anchor Selection

For each batch element, the top-$K$ tokens by grouping-space magnitude are selected as anchors.

This replaces ToMe’s pairwise merge scheduling with a deterministic selection of $K$ representative tokens. Selecting high-norm tokens ensures each latent token corresponds to a token the model considers salient in the learned grouping space.

#### 3. Hard Assignment via Cosine Similarity

All local tokens are assigned to their nearest anchor by cosine similarity:

$$
\text{group\_of\_token}[l] = \arg\max_k \langle g_l, g_{\text{anchor},k} \rangle
$$

This maps a continuous similarity vector to a discrete clusteing index.

#### 4. Group Aggregation

Latent tokens are computed as the mean of assigned local embeddings:

$$
z_K[k] = \frac{1}{g_k} \sum_{l \in \text{group}(k)} z_{L,\mathrm{ctx}}[l]
$$

implemented via `scatter_add_`.
For a fixed grouping assignment, this operation is: additon, division, tensor indexing. All of these are differentiable.

---

### Gradient Semantics

The group assignment step uses `argmax` and is therefore non-differentiable.  
However, aggregation into $z_K$ remains differentiable with respect to $z_{L,\mathrm{ctx}}$.

Thus:

- Gradients flow through the latent bottleneck via group means.
- Grouping decisions are treated as structural routing rather than continuously optimized soft clustering.

This design mirrors many hard-routing architectures and preserves stable training dynamics while maintaining a true compression bottleneck.

---

### Unmerge (K → L)

For latent reconstruction, $Z_K$ must be expanded back to local-token resolution prior to decoding.

This is implemented as:

$$
\bar{z}_L[l] = z_K[\text{group\_of\_token}[l]]
$$

using `torch.gather`.

This deterministic routing provides a clean reconstruction pathway without materializing dense grouping matrices.

---

### Trade-offs

| Aspect | Benefit | Limitation |
|--------|----------|------------|
| Anchor-based grouping | Deterministic, simple, preserves L→K interface | Not identical to ToMe pairwise merging |
| Hard assignment | Clear $S'$ structure, stable training | Non-differentiable grouping |
| Mean aggregation | Fully differentiable latent representation | May reduce fine-grained merge flexibility |

The merge operator is modular and can be replaced with a ToMe-style similarity merge or soft clustering variant without modifying downstream interfaces.

---

## 4.5 Adaptive Masked Token Modelling (AMTM)

AMTM defines a masking objective over base-resolution tokens where mask probabilities are derived from the latent grouping structure $S'$.

The key intuition is that tokens belonging to smaller latent groups represent more unique structural information and should be masked with higher probability.

### Implementation Locations

- `mergedna/amtm.py`
  - `compute_amtm_probs_from_groups`
  - `sample_exact_k_tokens`
  - `AMTMMaskSampler`
- `mergedna/model.py::MergeDNA.forward_amtm`

---

### Group-Driven Mask Distribution

Given latent grouping assignments:

$$
\text{group\_of\_token} \in [0..K-1]^{B \times L}
$$

Group sizes are computed per sample:

$$
g_i = \text{count}(\text{group\_of\_token} = i)
$$

Per-token sampling weights are defined as:

$$
w_i = \frac{1}{g_i}
$$

and token probabilities follow:

$$
P_L(j) \propto \frac{1}{g_i^2}
$$

for token $j$ in group $i$.

Thus, tokens in smaller groups (low redundancy) receive higher masking probability.

---

### Exact-K Sampling

Exactly $K$ local tokens are sampled without replacement according to $P_L$.

This ensures:

- Stable training signal magnitude per batch
- Deterministic mask count
- No duplicate sampling

The resulting boolean mask over local tokens is `mask_L ∈ {0,1}^{B×L}`.

---

### Projection to Base Resolution

The merged-token mask `mask_L` is projected to base resolution using the sparse segmentation representation (run-length encoding via `lengths_L`), avoiding dense materialization of the source matrix $S$.

This yields:

$$
\text{mask}_N \in \{0,1\}^{B \times N}
$$

---

### AMTM Forward Semantics

Per the paper, AMTM uses latent grouping only to define which tokens to mask.

The prediction forward pass is executed **without latent merging**:

$$
x_{\text{masked}} \rightarrow \text{forward\_reconstruct}(x_{\text{masked}}, L)
$$

Masked base tokens are replaced with the explicit `[MASK]` token from the DNA vocabulary.

The loss is computed as cross-entropy **only at masked base positions**, optionally excluding ambiguous targets.

---

### Design Rationale

- Importance-based masking focuses supervision on structurally unique regions.
- Masking depends only on grouping structure, not on prediction logits.
- The forward pass excludes latent compression, aligning exactly with the paper’s objective formulation.

---

### Trade-offs

| Aspect | Benefit | Limitation |
|--------|----------|------------|
| Importance sampling | Targets informative regions | Dependent on grouping quality |
| Exact-K masking | Stable optimization | Adds sampling complexity |
| Sparse mask projection | Memory efficient | Assumes contiguous span merges |

AMTM therefore integrates naturally with the latent bottleneck while preserving architectural modularity.


The correctness of Sections 4.1-4.5 is validated via unit tests aligned explicitly to each architectural component (see repository Test ↔ Report Section Mapping)

---

## 5. Training Pipeline Adaptation

Training loop modified to support:

- Three forward-pass objective computation  
- Frozen Local Encoder parameters during latent-MTR  
- λ-weighted latent reconstruction loss  

Compression-ratio sampling implemented by dynamically sampling L per iteration: $L ∼ 𝒩(N/2, σ²)$

---
## 6. Parameter Accounting (~380M parameters)
The configuration used in this implementation corresponds to the ~380M parameter setting described in Section 4 of the paper. The dominant contribution to parameter count arises from the Transformer blocks.

The architecture comprises:
- 4 Local Encoder blocks
- 20 Latent Encoder blocks
- 4 Latent Decoder blocks
- 2 Local Decoder blocks

forming a total of 30 Transformer blocks.

Each block uses:

- Model width: 𝑑𝑚𝑜𝑑𝑒𝑙 = 1024
- Multi-head attention with 16 heads (head dimension = 64)
- Feedforward hidden size: 𝑑𝑓𝑓 = 4 × 𝑑𝑚𝑜𝑑𝑒𝑙 = 4096

Assuming a standard Transformer block with separate $𝑊𝑞,𝑊𝑘,𝑊𝑣,𝑊𝑜$ projections and a two-layer FFN:

Attention parameters per block:
4 × 1024 × 1024 ≈ 4.2𝑀

FFN parameters per block:
1024 × 4096 + 4096 × 1024 ≈ 8.4𝑀

Total per block:
≈ 12.6𝑀

Across 30 blocks:
30 × 12.6𝑀 ≈ 378𝑀

Layer normalization parameters, token embeddings, and small projection layers used for local and global merging account for the remaining parameters, bringing the total close to the paper’s reported ~380M scale.

This estimate assumes:
- Standard two-layer FFNs (no gated MLP variants),
- Independent parameter sets for local and latent stacks (no weight sharing),
- Standard QKV attention projections.

---
## 7. Execution Guarantees Preserved

The following NanoChat execution guarantees were preserved:

- Unified SDPA attention dispatch (FA2/FA3)
- torchrun distributed compatibility
- Checkpoint format
- Training loop structure
- Rotary positional embeddings

---

## 8. Design Trade-offs

| Design Choice | Benefit | Cost |
|--------------|--------|------|
| Sparse S mapping | Memory efficiency | Explicit unmerge required |
| Local window attention | Merge locality | Reduced global receptive field |
| Latent token compression | Context modelling | Reconstruction fidelity |

---

## 9. Extensibility

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
