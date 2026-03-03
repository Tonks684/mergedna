"""
Transformer execution backbone used by MergeDNA.

This module is intentionally minimal and borrows "NanoChat-style" choices:
  - Parameter-free RMSNorm (via torch.rms_norm)
  - Rotary positional embeddings (RoPE)
  - ReLU^2 MLP nonlinearity (NanoChat choice; keeps plain 2-layer FFN parameter count)
  - FlashAttention wrapper dispatch (FA2/FA3/SDPA) via nanochat_infra.flash_attention

MergeDNA’s novelty is in hierarchical sequence representation (merging / latent bottleneck),
so this file stays close to a standard encoder stack to reduce implementation variance.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# This wrapper is expected to expose flash_attn_func(q,k,v, causal=..., window_size=...)
# and to handle dispatch to FA3/SDPA where available.
from nanochat_infra.flash_attention import flash_attn


# ---------------------------------------------------------------------------
# Normalization + Rotary embeddings
# ---------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Parameter-free RMSNorm (NanoChat-style).

    Design choice:
      - Keeps the transformer block simple and stable.
      - Parameter count differences from learned LayerNorm/RMSNorm are negligible at ~380M scale.
      - Avoids additional state (gamma/beta), which is convenient for reproduction scaffolding.
    """
    return F.rms_norm(x, (x.size(-1),), eps=eps)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to a (B, T, H, Dh) tensor.

    Args:
      x:   (B, T, H, Dh) where Dh is even
      cos: (1, T, 1, Dh/2)
      sin: (1, T, 1, Dh/2)

    Returns:
      x_rot: (B, T, H, Dh) with rotary position encoding applied.

    Note:
      RoPE rotates pairs of features (even/odd) in the head dimension. We implement this by
      splitting Dh into two halves (Dh/2 + Dh/2) corresponding to those paired dimensions.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """
    Configuration for a plain Transformer encoder stack.

    MergeDNA uses multiple encoders/decoders with different depths, but a shared width.
    max_seq_len must cover:
      - base sequence length N (e.g. 4096)
      - merged length L (<= N)
    """
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 20
    ffn_mult: int = 4

    rope_base: int = 10_000
    max_seq_len: int = 4096
    attn_window_size: tuple[int, int] = (-1, -1)  # default to full attention; override for local attention if needed


# ---------------------------------------------------------------------------
# Core blocks
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Multi-head self-attention in (B, T, C) input layout.

    Internal convention:
      - q,k,v are reshaped to (B, T, H, Dh) to match NanoChat attention wrapper conventions.
      - RoPE is applied to q and k only.
      - Optional q/k RMSNorm is used (NanoChat stabilizer).

    Attention backend:
      Uses flash_attn.flash_attn_func(...) which may dispatch to FA3 or SDPA.
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        # Standard dense projections (no bias): matches the parameter-count arithmetic used in your report.
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:   (B, T, C)
          cos: (1, T, 1, Dh/2)
          sin: (1, T, 1, Dh/2)

        Returns:
          y: (B, T, C)
        """
        B, T, C = x.shape
        H = self.cfg.n_heads
        Dh = self.head_dim

        # Project and reshape into multi-head format expected by attention backend.
        q = self.q(x).view(B, T, H, Dh)
        k = self.k(x).view(B, T, H, Dh)
        v = self.v(x).view(B, T, H, Dh)

        # Rotary embeddings on q/k.
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Optional q/k normalization (NanoChat stability choice).
        q = rmsnorm(q)
        k = rmsnorm(k)

        # MergeDNA reconstruction uses non-causal attention.
        # Local-window behaviour (Section 4.2) is supported in the backend, but this generic
        # encoder defaults to full attention. Locality is controlled at the backend call site
        # or via a specialized encoder wrapper if you choose to add one later.
        y = flash_attn.flash_attn_func(
            q, k, v,
            causal=False,
            window_size=self.cfg.attn_window_size,
        )  # (B, T, H, Dh)

        y = y.contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """
    2-layer feed-forward network (FFN).

    Design choice:
      - hidden = ffn_mult * d_model (standard GPT-style multiplier; matches report param math).
      - Activation uses ReLU^2 (NanoChat-style):
            ReLU(x)^2
        This keeps the FFN "plain" (no gating) while being stable and fast.
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.fc(x)).square())


class EncoderBlock(nn.Module):
    """
    Pre-norm Transformer block:
      x = x + Attn(RMSNorm(x))
      x = x + MLP(RMSNorm(x))

    This is a standard and stable structure used in many modern transformer variants.
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.attn = SelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x), cos, sin)
        x = x + self.mlp(rmsnorm(x))
        return x


# ---------------------------------------------------------------------------
# Rotary cache
# ---------------------------------------------------------------------------

class RotaryCache(nn.Module):
    """
    Precomputed RoPE cache up to cfg.max_seq_len.

    Why cache?
      - RoPE cos/sin tensors are deterministic given (max_seq_len, rope_base, head_dim).
      - Precomputing avoids recomputing outer products every forward.
      - registered as buffers (persistent=False) so they move with the module but do not
        bloat checkpoints.

    Shape convention:
      cos/sin are stored as (1, T, 1, Dh/2) so they broadcast across batch and heads.
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        head_dim = cfg.d_model // cfg.n_heads
        assert head_dim % 2 == 0, "RoPE requires even head_dim"

        # Precompute frequencies for t = 0..max_seq_len-1
        t = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(t, inv_freq)

        cos = freqs.cos()[None, :, None, :]  # (1, T, 1, Dh/2)
        sin = freqs.sin()[None, :, None, :]

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def get(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Slice to current sequence length and cast to match x dtype/device.

        Args:
          seqlen: current token length T
          device: device of the activations
          dtype: dtype of the activations

        Returns:
          cos, sin: (1, T, 1, Dh/2)
        """
        cos = self.cos[:, :seqlen].to(device=device, dtype=dtype)
        sin = self.sin[:, :seqlen].to(device=device, dtype=dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Encoder stack
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Plain transformer encoder stack used across MergeDNA modules:
      - Local encoder/decoder stacks
      - Latent encoder/decoder stacks

    Note on causality:
      MergeDNA’s pretraining is reconstruction-style, so these blocks are non-causal
      by default (causal=False in SelfAttention).
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.rope = RotaryCache(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, T, D)

        Returns:
          x: (B, T, D) after n_layers encoder blocks
        """
        cos, sin = self.rope.get(x.size(1), x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x

    def forward_range(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
      """
      Forward a contiguous subset of encoder blocks [start, end).

      This is used by MergeDNA LocalEncoder to interleave:
        (some local blocks) -> merge -> (some local blocks) -> merge -> ...

      Args:
        x: (B, T, D)
        start: inclusive block index
        end: exclusive block index

      Returns:
        x: (B, T, D) after applying blocks[start:end]
      """
      assert 0 <= start <= end <= len(self.blocks), (start, end, len(self.blocks))

      cos, sin = self.rope.get(x.size(1), x.device, x.dtype)
      for blk in self.blocks[start:end]:
          x = blk(x, cos, sin)
      return x
