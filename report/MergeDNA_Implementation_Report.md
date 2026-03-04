# MergeDNA Implementation Report

## 1. Implementation Overview

This implementation treats the [NanoChat](https://github.com/karpathy/nanochat) codebase as an execution-layer infrastructure for sequence modelling, rather than as a fixed language modelling architecture.

MergeDNA introduces modelling innovations primarily at the level of sequence representation, including genomic patch embeddings and latent bottleneck tokens, without fundamentally altering the underlying attention mechanism. 

Re-implementing a transformer stack from scratch would introduce unnecessary variance into the reproduction and divert engineering effort toward already-solved systems problems (e.g. optimized attention execution, distributed training orchestration, and checkpointing logic).

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

### Execution Guarantees Preserved

Using NanoChat as the execution backend preserves the following properties:

- Unified SDPA attention dispatch (FlashAttention-2/FA3)
- `torchrun` distributed training compatibility
- NanoChat checkpoint format
- Rotary positional embeddings
- Training loop structure
---

## 4. MergeDNA Module Implementations


### Overview of Implementation
The following subsections (4.1-4.5) describe each architectural component and its corresponding implementation details within the `mergedna/` module. Each component is validated via section-aligned unit tests (see README.md Test ↔ Report Section Mapping)

**Notation**  
$N$: base token length. \
$L$: locally merged length. \
$K$: latent bottleneck length. \
$Z_L \in \mathbb{R}^{B\times L\times D}$: locally merged tokens. \
$Z_K \in \mathbb{R}^{B\times K\times D}$: latent tokens.  
$S$: base→local segmentation mapping (represented sparsely via `token_lens`/`token_starts`).  
$S'$: local→latent grouping (represented sparsely via `group_of_token`).

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

**Overview**
MergeDNA pre-training objectives (MTR, AMTM) require reconstruction-style attention rather than autoregressive next-token prediction. This means attention must run with causal masking disabled.

**Implementation locations**

- `mergedna/transformer.py::SelfAttention.forward`
- `mergedna/flash_attention_patch.py::flash_attn_func`

**Implementation details**

NanoChat’s FlashAttention-backed execution path is reused where available. When falling back to Pytorch SDPA, I pass `is_causal=False`; for MergeDNA reconstruction passes I invoke `flash_attn_func(..., causal=False)` so SDPA runs with causal masking disabled.


In `mergedna/transformer.py:: SelfAttention.forward` non-causal attention guranteed regardless of backend.
```python
y = flash_attn.flash_attn_func(
            q, k, v,
            causal=False,
            window_size=self.cfg.attn_window_size,
        )
```
In `mergedna/flash_attention_patch.py::flash_attn_func` even without Nanochat I still gave the same non-causal attention.
```python
out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=False)
```
where `causal=False`.

**Design choices**
**From paper**
- Reconstruction-style (non-autoregressive) attention is required for the reconstruction objectives used in MergeDNA pre-training.

**Assumptions / engineering choices**
- Reuse Nanochat FlashAttention-2/FA3 runtime dispatch when present otherwise implement SDPA fallback with Pytorch.
- Reuse Nanochat rotary embedding, RMSnorm and distributed execution support.

**Verification**
- `tests/test_41_42_attention_backend.py`

**Trade-offs**
- **Benefit**: Preserves high-performance attention dispatch (FA/SDPA) while matching reconstruction semantics
- **Cost**: Requires careful masking logic 
---

### 4.2 Local-Window Attention Module

**Overview**
MergeDNA’s local stages(Local Encoder and Local Decoder) operate under **fixed-window locality constraints**. This requires **sliding-window attention masks** rather than full global attention.

**Implementation locations**

- `mergedna/transformer.py:: EncoderConfig`
- `mergedna/transformer.py:: SelfAttention.forward`
- `mergedna/flash_attention_patch.py::_sdpa_bool_mask`
- `mergedna/flash_attention_patch.py:: flash_attn_func`


**Implementation details**

Local-window attention is enabled by passing a `window_size=(left,right)` arguement through the attention backend. Where FA is unavailable, `_sdpa_bool_mask` constructs an explicit boolean allow-mask for Pytorch SDPA that enforces the optional causal constraint and local window constraints (left/right distance bounds)

In `mergedna/transformer.py:: EncoderConfig` the attention window is defined
```python
# Local encoder/decoder: sliding window attention
enc_cfg = EncoderConfig(..., attn_window_size=(cfg.local_window_size, cfg.local_window_size))

# Latent encoder/decoder: full attention
latent_cfg = EncoderConfig(..., attn_window_size=(-1, -1))
```

In `mergedna/transformer.py:: SelfAttention.forward` configure the window size in:
```python
y = flash_attn.flash_attn_func(
    q, k, v,
    causal=False,
    window_size=self.cfg.attn_window_size
    )
```

Where NanoChat’s FlashAttention backend is unavailable, a boolean SDPA mask is constructed to enforce locality in `mergedna/flash_attention_patch.py`:
```python
# mergedna/flash_attention_patch.py :: _sdpa_bool_mask
mask &= ((row - col) <= left)
mask &= ((col - row) <= right)
# mergedna/flash_attention_patch.py :: flash_attn_func (SDPA fallback)
mask = _sdpa_bool_mask(
    Tq, Tk,
    causal=causal,
    window_size=window_size,
    device=qh.device
)
```
**Design choices**

**From paper**
- Local stages are locality-constrained (eg. window size 16 in the reported configuration)
**Assumptions / engineering choices**
- Implement local attention via the same unified backend interface so the rest of the model does not depend on which attention kernel is used.

**Verification**
- `tests/test_41_42_attention_backend.py`

**Trade-off**:
- **Benefit**: Correct locality semantics with a single attention interface, compatible with FA dispatch where available.
- **Cost**: SDPA fallback requires mask construction, which can be slower than kernel native sliding window attention.
---

### 4.3 Token Merging + Segmentation Mapping (Layer-wise)

**Overview**
The core merging logic is implemented in `mergedna/local_merge.py`, while the merge scheduling and interleaving with the transformer encoder are implemented in `mergedna/model.py::LocalEncoder`.

MergeDNA’s Local Encoder performs **progressive, layer-wise token merging**, interleaved with local self-attention blocks.

Rather than applying a single merge operation after all local layers, this implementation interleaves:

`Local attention blocks → merge → local attention → merge → ... → final Z_L` as identfied in the paper.

The Local Encoder stack is partitioned into merge segments, and a merge operation is applied after each segment. The merge targets are determined by a deterministic schedule ensuring the final sequence length equals the requested target length `𝐿`.

This preserves the hierarchical tokenization dynamics described in the paper.

**Implementation locations**

- Layerwise merge scheduling & encoder interleaving: `mergedna/model.py::LocalEncoder.forward`
- Deterministic merge length schedule (N -> L): `mergedna/model.py::LocalEncoder.linear_merge_schedule`
- Collision-free ToMe-style adjacency selection:
    - `mergedna/local_merge.py::_choose_nonoverlap_pairs_tome_adjacent`
- Span-weighted merge update and token span tracking: `mergedna/local_merge.py::LocalTokenMerger.forward`
- Sparse segmentation utilities:
    - `mergedna/local_merge.py::unmerge_tokens`
    - `mergedna/local_merge.py::project_mask_to_base`

**Implementation details**

**Layerwise merge scheduling & encoder interleaving**
The local encoder partitions transfomer layers into `n_segments = n_merges +1` segments, forwards each segment, then merges to the next scheduled length before continuing. The schedule is deterministic and ends at `target_L`.
Below is a snippet from `mergedna/model.py::LocalEncoder.forward` showing the layer-wise compression, merge schedule and the use of `offset` for ToMe-style alternating between even and odd.
```python
for seg_idx in range(n_segments):
    s, e = boundaries[seg_idx], boundaries[seg_idx + 1]
    z = self.enc.forward_range(z, s, e)
    # merge after encoder segment
    if seg_idx < n_merges:
        offset = seg_idx % 2 # ToMe style alternating
        z, lengths, _ = self.merger(
            z, lengths,
            target_len=schedule_L[seg_idx],
            offset=offset
        )
```
This implements the layer-wise progressive compression described in the paper.

**Deterministic merge length schedule (N -> L)**
The paper describes progressive merging but does not specify the exact intermediate lengths or budgeting strategy per layer segment. This reproduction uses a simple linear schedule to produce intermediate targets `L_1...L_m` ending at `target_L`.

```python
def linear_merge_schedule(N: int, target_L: int, n_merges: int) -> list[int]:
    """
    Monotone schedule of intermediate lengths ending at target_L.

    Example:
        N=4096, target_L=2048, n_merges=2 -> [3072, 2048]
    """
    if n_merges <= 0:
        return []
    steps = []
    for i in range(1, n_merges + 1):
        Li = int(round(N - (N - target_L) * (i / n_merges)))
        steps.append(max(target_L, min(N, Li)))
    steps[-1] = target_L
    return steps
```

**Collision-free merge pair selection**
To prevent overlap (a token being merged twice in the same stage), the candidate adjacent edges are restricted using a bipartite parity split ("offset") as mentioned above. This alternates between stages:
| Merge stage | Allowed pairs |
|--------------|---------------|
| Stage 0 | (0,1), (2,3), (4,5), … |
| Stage 1 | (1,2), (3,4), (5,6), … |
| Stage 2 | (0,1), (2,3), … |
...

Offset is also threaded into `LocalTokenMerger.forward(..., offset=,...)` and into `_choose_nonoverlap_pairs_tome_adjacent(..., offset=...)`.
Specifically, this alternating bipartite schedule gurantees:

- No token participates in more than one merge per stage
- Adjacent span structure is preserved
- Merge decisions remain deterministic and collision-free

This design mirrors the non-overlapping pair selection used in ToMe while maintaining the **contiguous-span invariant required for efficient segmentation tracking**.

I represent the chosen merges as a boolean vector `merge_right` of length `W`, where `merge_right[j]=True` means 'merge token j into token j-1'.

Below is a snippet from `mergedna/local_merge.py::_choose_nonoverlap_pairs_tome_adjacent` showing non-overlapping merges

- Here the edge indices are computed
- Select parity-allowed edges
- Select top-k by similarity within the budget

```python
# left indices for edges (i, i+1)
edge_left = torch.arange(0, W - 1, device=device)

allowed = (edge_left % 2 == offset)

candidate_edges = edge_left[allowed]
candidate_scores = sim_adj[candidate_edges]

if candidate_edges.numel() == 0:
    return merge_right

k = min(merge_budget, candidate_edges.numel())

# pick top-k edges by similarity
topk = torch.topk(candidate_scores, k=k).indices
chosen_left = candidate_edges[topk]
chosen_right = chosen_left + 1
merge_right[chosen_right] = True
```
---

**Span-weighted merge update and token span tracking**
The paper defines a dense source matrix $S$ mapping base tokens to merged tokens. Materialising dense `S` is impractical for long sequences (e.g., $N = 4096$).  
Instead, this implementation uses a **sparse run-length encoding** that fully determines the same base -> local mapping under contiguous-span merges:

- `token_lens ∈ ℤ^{B × L}`: span length of each merged token  
- `token_starts ∈ ℤ^{B × L}`: starting base index of each span  

This representation is:

- Memory-efficient: $O(L)$ rather than $O(LN)$  
- Exact for contiguous merges  
- Composable across multiple merge stages  

Within each merge stage:

1. Tokens are projected into a learned grouping space.
2. Adjacent token similarities are computed within fixed local windows.
3. A set of **adjacent merge pairs** is selected using a merge budget.
4. Each selected right token is merged into its left neighbor using span-weighted averaging.
    - embeddings are averaged weighted by current span lengths (length of merged tokens).
    - span lengths are accumalated to preserve accounting.
5. Keep or drop merged-right tokens
6. `token_starts` is derived deterministically from `token_lens`.

Primary snippets from `mergedna/local_merge.py::LocalTokenMerger.forward` link to the numbered steps above.

```python
g = self.group(z)  # 1.
# ....
sim_all = (g[:, :-1] * g[:, 1:]).sum(dim=-1) # 2.
# ...
mr = self._choose_nonoverlap_pairs_tome_adjacent(local, budget,offset=offset) # 3.
# ...
merge_right[idx] = True # 5. 
keep[idx] = False # 5.
# ...
for j in idxs.tolist(): # 4.
    i = j - 1

    w_i = lb2[i].to(zb2.dtype)
    w_j = lb2[j].to(zb2.dtype)

    zb2[i] = (zb2[i] * w_i + zb2[j] * w_j) / (w_i + w_j) # 4a.
    lb2[i] = lb2[i] + lb2[j]
# ...
zb_new = zb2[keep] # 5.
lb_new = lb2[keep] # 5.
# ...
lens_new = torch.stack(lens_out, 0) # (B, L) # 4b.
#...
starts_new = torch.cumsum(lens_new, dim=1) - lens_new # 6.
```
This span-weighted merge (4a.) preserves proportional contribution from larger spans and maintains consistent reconstruction behaviour.
`token_starts` is derived from `token_lens` and is not required for unmerge/mask projection in the current implementation, but it is kept for completeness and potential future span logic.

**Sparse segmentation utilities**
Primary snippets for implementing the sparse `S` from:
`mergedna/local_merge.py::unmerge_tokens`
```python
zb = torch.repeat_interleave(z_L[b], token_lens[b], dim=0)[:N]
```

`mergedna/local_merge.py::project_mask_to_base`
```python
mb = torch.repeat_interleave(mask_L[b], lengths_L[b], dim=0)[:N]
```
These operations are equivalent to multiplying by `S^T` (unmerge) and `S`-projection for masks, validated against dense reference for small N.

---

#**Design choices**
**From paper**

- Layer-wise progressive merging

**Assumptions / engineering choices**
- Linear merge schedule (paper doesn’t specify exact intermediate lengths)
- Alternating bipartite merge selection
- Sparse span encoding instead of dense $(S)$

---

**Trade-offs**

| Design choice | Benefit | Cost |
|---------------|--------|------|
| Sparse span encoding instead of dense $(S)$ | Memory efficient and simple | Requires explicit unmerge operations |
| Layer-wise progressive merging | Closer to MergeDNA architecture | Slightly more complex control flow |
| Alternating bipartite merge selection | Collision-free merges | Restricts candidate edges per stage |
| Linear merge schedule | Stable and reproducible | Less adaptive than learned merge budgets |
---

**Verification**
- `tests/test_43_layerwise_merge_schedule.py`
- `tests/test_43_local_merge_sparse_S.py`
- `tests/test_43_local_merge_tome_offset.py`


### 4.4 Latent Bottleneck Pipeline (Global Merge L → K + Unmerge K → L)

**Overview**

MergeDNA’s latent reconstruction and AMTM objectives require a global compression stage over the locally merged sequence $Z_L$, producing a shorter latent sequence $Z_K$ with $K < L$, together with a grouping structure $S'$ used for reconstruction and importance-based masking.

This implementation provides a faithful interface and training semantics for the latent bottleneck while approximating the paper’s ToMe-style merge with a deterministic anchor-based clustering procedure.

**Implementation locations**

- Latent grouping and global merge (L → K): `mergedna/model.py::LatentEncoder.global_merge_to_K`
- Latent reconstruction routing (K → L): `mergedna/model.py::MergeDNA.forward_latent_reconstruct`

**Implementation details**

Given contextualized local tokens: $z_{L,\mathrm{ctx}} \in \mathbb{R}^{B \times L \times D}$ `global_merge_to_K(z_L_ctx, K)` returns:

- `z_K` of shape `(B, K, D)` - compressed latent tokens  
- `group_of_token` of shape `(B, L)` - integer group assignments in `[0, K-1]`

`group_of_token` is a sparse representation of the paper’s grouping structure $S'$: each local token belongs to exactly one latent group. This structure is later used to compute group sizes $g_i$ for AMTM.

---

### Merge Algorithm (Anchor-Based Hard Clustering)

The paper describes a ToMe-style global merging process. Here, I implement a simpler but semantically equivalent bottleneck with the following steps:

**1. Learned Grouping Projection**

Each local token is projected into a lower-dimensional grouping space: $g_{\text{raw}} = W_g z_{L,\mathrm{ctx}}$

Primary snippet from `mergedna/model.py::LatentEncoder.global_merge_to_K`:

```python
g_raw = self.group(z_L_ctx)           # grouping projection
scores = g_raw.norm(dim=-1)           # token saliency
```
The projection `self.group` learns a grouping space used for both anchor selection and clustering similarity.

The magnitude of this projection $\|g_{\text{raw}}\|$ is interpreted as a learned saliency score. Tokens with larger magnitude are treated as structurally important candidates for latent anchors.
Each column of $W_g$ can be viewed as detecting some structural pattern in the contextualised embedding eg. long-range interaction strength or motif boundary signals. If a token strongly activates many learned grouping directions $g_{raw}$ will have large magnitude thus norm measures how strongly the token activities learned grouping subspaces.


**2. Anchor Selection**

For each batch element, the top-$K$ tokens by grouping-space magnitude are selected as anchors.

This replaces ToMe’s pairwise merge scheduling with a deterministic selection of $K$ representative tokens. Selecting high-norm tokens ensures each latent token corresponds to a token the model considers salient in the learned grouping space.

Primary snippet from `mergedna/model.py::LatentEncoder.global_merge_to_K`:

```python
topk_idx = torch.topk(scores, k=K, dim=1).indices

anchors = torch.gather(
    g_raw,
    1,
    topk_idx.unsqueeze(-1).expand(B, K, g_raw.size(-1))
)
```
This selects the top-k tokens in grouping space as cluster anchors.

**3. Hard Assignment via Cosine Similarity**

All local tokens are assigned to their nearest anchor by cosine similarity: $\text{group\_of\_token}[l] = \arg\max_k \langle g_l, g_{\text{anchor},k} \rangle$

```python
sim = torch.einsum("blg,bkg->blk", g, anchors)

with torh.no_grad():
    group_of_token = sim.argmax(dim=-1) # (B,L)
```
`group_of_token` stores the sparse grouping structure $S'$ mapping each local token to a latent group.

**4. Group Aggregation**

Latent tokens are computed as the mean of assigned local embeddings:

$z_K[k] = \frac{1}{g_k} \sum_{l \in \text{group}(k)} z_{L,\mathrm{ctx}}[l]$ implemented via `scatter_add_`. 

```python
z_K.scatter_add_(
    1,
    group_of_token.unsqueeze(-1).expand(B, L, D),
    z_L_ctx
)

counts.scatter_add_(
    1,
    group_of_token,
    torch.ones((B, L), device=z_L_ctx.device)
)

z_K = z_K / counts.clamp_min(1.0).unsqueeze(-1)
```
`scatter_add_` accumulates embeddings into latent groups before normalizing by group size.

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

$\bar{z}_L[l] = z_K[\text{group\_of\_token}[l]]$ using `torch.gather`.

```python
z_bar_L = torch.gather(
    z_K,
    1,
    group_of_token.unsqueeze(-1).expand(
        z_K.size(0),
        group_of_token.size(1),
        z_K.size(-1)
    )
)
```
Each local token retrieves the embedding of its assigned latent group.

---

**Design choices**

**From paper**

- A global latent bottleneck compressing $Z_L$ into $Z_K$
- Hard grouping structure $S'$ used for AMTM importance sampling

**Assumptions / engineering choices**

- Anchor-based clustering rather than ToMe pairwise merging
- Hard assignment via argmax rather than soft clustering
- Mean aggregation for latent token construction

**Trade-offs**

| Aspect | Benefit | Limitation |
|--------|----------|------------|
| Anchor-based grouping | Deterministic, simple, preserves L→K interface | Not identical to ToMe pairwise merging |
| Hard assignment | Clear $S'$ structure, stable training | Non-differentiable grouping |
| Mean aggregation | Fully differentiable latent representation | May reduce fine-grained merge flexibility |

The merge operator is modular and can be replaced with a ToMe-style similarity merge or soft clustering variant without modifying downstream interfaces.

**Verification**

- `tests/test_44_latent_grouping_shapes.py`
- `tests/test_44_latent_unmerge_consistency.py`

---

### 4.5 Adaptive Masked Token Modelling (AMTM)

AMTM defines a masking objective over base-resolution tokens where mask probabilities are derived from the latent grouping structure $S'$.

The key intuition is that tokens belonging to smaller latent groups represent more unique structural information and should be masked with higher probability.

**Implementation Locations**

- `mergedna/amtm.py`
  - `compute_amtm_probs_from_groups`
  - `sample_exact_k_tokens`
  - `AMTMMaskSampler`
- `mergedna/model.py::MergeDNA.forward_amtm`

---

**Group-Driven Mask Distribution**

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

**Exact-K Sampling**

Exactly $K$ local tokens are sampled without replacement according to $P_L$.

This ensures:

- Stable training signal magnitude per batch
- Deterministic mask count
- No duplicate sampling

The resulting boolean mask over local tokens is `mask_L ∈ {0,1}^{B×L}`.

---

**Projection to Base Resolution**

The merged-token mask `mask_L` is projected to base resolution using the sparse segmentation representation (run-length encoding via `lengths_L`), avoiding dense materialization of the source matrix $S$.

This yields:

$$
\text{mask}_N \in \{0,1\}^{B \times N}
$$

---

**AMTM Forward Semantics**

Per the paper, AMTM uses latent grouping only to define which tokens to mask.

The prediction forward pass is executed **without latent merging**:

$$
x_{\text{masked}} \rightarrow \text{forward\_reconstruct}(x_{\text{masked}}, L)
$$

Masked base tokens are replaced with the explicit `[MASK]` token from the DNA vocabulary.

The loss is computed as cross-entropy **only at masked base positions**, optionally excluding ambiguous targets.

---

**Design Choices**

- Importance-based masking focuses supervision on structurally unique regions.
- Masking depends only on grouping structure, not on prediction logits.
- The forward pass excludes latent compression, aligning exactly with the paper’s objective formulation.

---

**Trade-offs**

| Aspect | Benefit | Limitation |
|--------|----------|------------|
| Importance sampling | Targets informative regions | Dependent on grouping quality |
| Exact-K masking | Stable optimization | Adds sampling complexity |
| Sparse mask projection | Memory efficient | Assumes contiguous span merges |

AMTM therefore integrates naturally with the latent bottleneck while preserving architectural modularity.


The correctness of Sections 4.1-4.5 is validated via unit tests aligned explicitly to each architectural component (see repository Test ↔ Report Section Mapping)

---

## 5. Parameter Accounting (~380M parameters)
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

## 6. Training Pipeline Adaptation

**Overview**
The training loop modified to support:

- Three forward-pass objective computation  
- Frozen Local Encoder parameters during latent-MTR  
- λ-weighted latent reconstruction loss  

Compression-ratio sampling implemented by dynamically sampling L per iteration: $L ∼ 𝒩(N/2, σ²)$

Each training step performs three forward passes:

1. **Merged Token Reconstruction (MTR)**: `forward_reconstruct(x, L)`
2. **Latent MTR (Local Encoder frozen)**: `forward_reconstruct(x, L)` with gradients disabled for `Eϕ`
3. **Adaptive Masked Token Modelling (AMTM)**:`forward_amtm(x)`

The total loss is:

L_total = L_MTR(θ) + λ L_MTR(θ \ {ϕ}) + L_AMTM(θ)

**Implementation locations**

- `mergedna/model.py::MergeDNA.forward_reconstruct`
- `mergedna/model.py::MergeDNA.forward_amtm`
- `train.py::training_step`

---

## 8. Verification & Reproducibility
Verification is provided via unit tests aligned to Sections 4.1–4.5 and the training contract described in Section 6.

See:

- `tests/`
- `README.md` (Test ↔ Report Section Mapping)

These tests validate:

- attention backend behaviour
- local token merging invariants
- sparse segmentation mapping
- latent grouping consistency
- AMTM masking semantics

---

## 9. Extensibility
The modular separation between:

- token merging
- latent grouping
- masking objectives
- transformer execution

allows architectural experimentation without modifying the underlying execution backend.

---
