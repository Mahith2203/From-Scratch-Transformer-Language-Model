"""
SprintLM model definition.

A decoder-only transformer whose architectural components are individually
switchable via ModelConfig, so the *same* file produces every point in the
ablation table:

    baseline            : learned pos-emb + LayerNorm + GELU-MLP
    +RoPE               : RoPE          + LayerNorm + GELU-MLP
    +RoPE +RMSNorm      : RoPE          + RMSNorm   + GELU-MLP
    full (SprintLM)     : RoPE          + RMSNorm   + SwiGLU-MLP

Everything else (attention mechanism, residual structure, param count
target) is held fixed across variants, which is what makes the comparison
meaningful.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprintlm.config import ModelConfig


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm (no mean-centering, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def make_norm(dim: int, use_rmsnorm: bool) -> nn.Module:
    return RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)


# --------------------------------------------------------------------------
# Positional encoding: RoPE vs learned absolute embeddings
# --------------------------------------------------------------------------

def build_rope_cache(context_length: int, head_dim: int, theta: float, device, dtype):
    """Precompute cos/sin tables for rotary embeddings. Shape: [T, head_dim/2]."""
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(context_length, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [T, head_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [B, n_heads, T, head_dim]
    cos, sin: [T, head_dim/2] (sliced to the current position range by the caller)
    Rotates each consecutive pair of dims by position-dependent angle.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    out = torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)
    return out.to(x.dtype)


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention. Supports an optional incremental
    KV-cache (past_kv) for autoregressive decoding — see
    sprintlm.inference.kv_cache for how this is driven at generation time.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.use_rope = cfg.use_rope

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, rope_cos=None, rope_sin=None, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Dh]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # is_causal only valid when there's no cached prefix (single full-sequence pass);
        # when decoding with a cache, the new query attends to all cached + new keys, so
        # no additional causal mask is needed (there's nothing "future" to hide).
        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y, new_kv


# --------------------------------------------------------------------------
# Feed-forward: SwiGLU vs vanilla GELU-MLP
# --------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU gated MLP (as in LLaMA/PaLM). d_ff is the gated hidden width."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GELUMLP(nn.Module):
    """Vanilla GPT-2 style 2-layer MLP with GELU. Used in the baseline ablation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def make_mlp(d_model: int, d_ff: int, use_swiglu: bool) -> nn.Module:
    return SwiGLU(d_model, d_ff) if use_swiglu else GELUMLP(d_model, d_ff)


# --------------------------------------------------------------------------
# Transformer block
# --------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = make_norm(cfg.d_model, cfg.use_rmsnorm)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg.d_model, cfg.use_rmsnorm)
        self.mlp = make_mlp(cfg.d_model, cfg.d_ff, cfg.use_swiglu)

    def forward(self, x, rope_cos=None, rope_sin=None, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.norm1(x), rope_cos, rope_sin, past_kv, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------

class SprintLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if not cfg.use_rope:
            self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        else:
            self.pos_emb = None

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = make_norm(cfg.d_model, cfg.use_rmsnorm)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._rope_cache = None  # lazily built on first forward, on the right device/dtype
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def _get_rope_cache(self, device, dtype):
        if self._rope_cache is None or self._rope_cache[0].device != device:
            head_dim = self.cfg.d_model // self.cfg.n_heads
            self._rope_cache = build_rope_cache(
                self.cfg.context_length, head_dim, self.cfg.rope_theta, device, dtype
            )
        return self._rope_cache

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                past_kv_list=None, use_cache: bool = False, start_pos: int = 0):
        """
        idx: [B, T] token ids for this forward pass (full prompt on first call,
             single new token per step when decoding with a KV-cache).
        past_kv_list: list of (k, v) tuples per layer, or None.
        start_pos: absolute position of idx[:, 0] in the sequence — needed to
             index into the RoPE table correctly when decoding incrementally.
        """
        B, T = idx.shape
        x = self.tok_emb(idx)

        if self.cfg.use_rope:
            cos_full, sin_full = self._get_rope_cache(idx.device, x.dtype)
            rope_cos = cos_full[start_pos:start_pos + T]
            rope_sin = sin_full[start_pos:start_pos + T]
        else:
            pos = torch.arange(start_pos, start_pos + T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
            rope_cos = rope_sin = None

        new_kv_list = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, new_kv = block(x, rope_cos, rope_sin, past_kv, use_cache)
            if use_cache:
                new_kv_list.append(new_kv)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss, new_kv_list

    def configure_optimizer(self, weight_decay, lr, betas, eps):
        """Standard practice: no weight decay on norms/biases/embeddings."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
