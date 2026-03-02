from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn

def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Parameter-free RMSNorm (nanochat style). Small param differences are negligible vs 380M scale.
    return F.rms_norm(x, (x.size(-1),), eps=eps)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B,T,H,D), cos/sin: (1,T,1,D/2)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)

@dataclass
class EncoderConfig:
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 20
    ffn_mult: int = 4
    rope_base: int = 10_000
    max_seq_len: int = 4096  # must cover base length N and token length L

class SelfAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        B, T, C = x.shape
        H = self.cfg.n_heads
        D = self.head_dim

        q = self.q(x).view(B, T, H, D)
        k = self.k(x).view(B, T, H, D)
        v = self.v(x).view(B, T, H, D)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # optional QK norm (nanochat does this; keep for stability)
        q = rmsnorm(q)
        k = rmsnorm(k)

        y = flash_attn.flash_attn_func(q, k, v, causal=False, window_size=(-1, -1))  # full attention
        y = y.contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nanochat uses ReLU^2; it’s simple and matches the “plain 4x FFN” parameter structure
        return self.proj(F.relu(self.fc(x)).square())

class EncoderBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.attn = SelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x), cos, sin)
        x = x + self.mlp(rmsnorm(x))
        return x

class RotaryCache(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.d_model // cfg.n_heads
        assert head_dim % 2 == 0, "rotary needs even head_dim"

        # Precompute to cfg.max_seq_len; MergeDNA uses N=4096, L=N/2, K=L/2. Keep margin if desired.
        t = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :]  # (1,T,1,D/2)
        sin = freqs.sin()[None, :, None, :]

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def get(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[:, :seqlen].to(device=device, dtype=dtype)
        sin = self.sin[:, :seqlen].to(device=device, dtype=dtype)
        return cos, sin

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.rope = RotaryCache(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rope.get(x.size(1), x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x