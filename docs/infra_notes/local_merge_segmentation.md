# Local Merge: Segmentation Representation Notes (LocalTokenMerger 4.3)

## Goal

MergeDNA (paper) defines a local token merging process that produces:
- a merged token sequence `Z_L` (length L),
- a segmentation-tracking matrix `S` mapping merged tokens back to base tokens (length N).

`S` is conceptually a binary matrix of shape (L, N) such that:
- `S[l, n] = 1` if base position `n` belongs to merged token `l`,
- each base position belongs to exactly one merged token.

Storing `S` densely is O(LN) and infeasible at scale.

## Our choice: sparse “S” via run-length encoding

We represent segmentation as two per-token vectors:

- `token_lens[b, l]`: number of base tokens covered by merged token `l`
- `token_starts[b, l]`: base start index of merged token `l`

This implicitly defines `S` as contiguous spans:
- merged token `l` covers base indices:
  `[token_starts[l], token_starts[l] + token_lens[l])`

### Why this works

Our local merge implementation only merges adjacent pairs (j merges into j-1).
Therefore merged groups remain contiguous spans, so run-length encoding is lossless.

If we later allow non-adjacent or non-contiguous grouping, run-length encoding becomes insufficient,
and we would need either:
- explicit membership lists, or
- a sparse COO representation of S, or
- a parent-pointer merge tree.

## Operations we need from S

### 1) Unmerge embeddings (Z_L -> Z_N)
We must expand merged embeddings back to base resolution for reconstruction / token-level tasks.

Implementation:
- `unmerge_tokens(z_L, token_lens, N)` uses `repeat_interleave` along sequence dim.
- Produces `z_N` of shape (B, N, D), truncated to N if needed.

### 2) Project masks from merged to base (mask_L -> mask_N)
AMTM and related objectives define masking at merged resolution and require masking base tokens.

Implementation:
- `project_mask_to_base(mask_L, token_lens, N)` uses `repeat_interleave`.
- Convention: `mask_L=True` means "selected/masked" merged token; repetition applies to all bases in its span.

## Merge budgeting and determinism

Merge uses a target compressed length `target_len` (L), implying merges_needed = T - L.

Within each window, we allocate a per-window merge budget and select non-overlapping adjacent pairs by
descending similarity. This ensures:
- merges do not overlap (a token participates in at most one merge per pass),
- segmentation remains well-defined.

### Note on "differentiability"
Pair selection is discrete (arg-sort and greedy selection), similar in spirit to ToMe-style merging.
Gradients flow through the merged embeddings (length-weighted average), but selection itself is not
differentiable. If the paper requires truly differentiable merging, this is an approximation.

## Validation checklist

- Shape checks: lengths and starts have shape (B, L), unmerge gives (B, N, D)
- Conservation: sum(token_lens[b]) >= N typically; exactness depends on truncation/target enforcement
- Mask projection equivalence: base mask should align with span coverage
- Compare against dense S on tiny toy examples:
  - build dense S from lens/stats
  - verify unmerge and mask projection match dense operations