# Attention Backend Notes (MergeDNA 4.1 / 4.2)

## Context

MergeDNA requires two attention behaviours that differ from NanoChat’s default autoregressive setup:

- **4.1 Non-causal attention**: reconstruction/MLM-style objectives require `is_causal=False`
- **4.2 Local-window attention**: Local Encoder/Decoder use fixed sliding windows (e.g. size 16)

We implement both via a single attention backend wrapper:
- Prefer NanoChat’s FlashAttention wrapper when available (FA2/FA3/SDPA dispatch)
- Fallback to PyTorch `scaled_dot_product_attention` (SDPA) when NanoChat backend is unavailable

## Why use a backend wrapper?

- Keeps NanoChat code read-only (provenance)
- Centralizes attention semantics (causal/non-causal/windowed)
- Preserves high-performance kernels when available
- Provides a correctness-first fallback

## Tensor layout

Backend accepts and returns:
- `q, k, v`: (B, T, H, Dh)
- output: (Batch size, Tokens (sequence length), Heads, Dimension of heads)

PyTorch SDPA expects (B, H, T, Dh), so fallback transposes:
- `qh = q.transpose(1,2)` etc.
- and transposes back afterward.

## Non-causal vs causal

- **Non-causal**: all query positions may attend to all key positions (subject to optional local window constraints)
- **Causal**: query position t may only attend to keys <= t (classic autoregressive)

In PyTorch SDPA:
- causal behaviour is enabled via `is_causal=True`
- for non-causal objectives, use `is_causal=False`

## Sliding-window local attention mask

For local attention, we restrict attention to a band around the "aligned" diagonal.
We support unequal query/key lengths by defining an alignment row index:

- `row = (Tk - Tq) + arange(Tq)` so that the last query aligns to the last key.

Then for each (q_idx, k_idx):
- optionally enforce causal: `k_idx <= row`
- enforce left window:  `row - k_idx <= left`
- enforce right window: `k_idx - row <= right`

Mask is boolean:
- `True` means "allowed / keep"
- `False` means "masked out"

This mask is broadcast across batch and head dimensions in SDPA.

## Performance notes

- Constructing boolean masks has overhead; this is acceptable for fallback correctness paths.
- The preferred fast path is NanoChat’s FlashAttention wrapper, which can handle causal + windowing efficiently
  and may dispatch to FA2/FA3 depending on hardware.

## Validation strategy

Unit tests verify:
- non-causal backend matches a naive full-attention implementation
- causal backend matches naive causal attention
- local-window backend matches attention computed with an explicit boolean mask
- output shapes and dtype are consistent