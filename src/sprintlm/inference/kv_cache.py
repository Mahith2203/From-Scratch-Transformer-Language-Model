"""
Autoregressive generation, with and without a KV-cache.

Why this file matters for the project write-up: the *readable* training
path deliberately doesn't use a cache (see model.py forward — it always
recomputes attention over the full context). Naive decoding therefore
recomputes K/V for the entire prefix at every new token, which is O(T^2)
work across a generation instead of O(T). generate_no_cache and
generate_with_cache implement exactly the same sampling procedure so that
tools/benchmark_inference.py can produce an apples-to-apples latency/
throughput comparison.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sprintlm.model import SprintLM


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@dataclass
class GenerationResult:
    tokens: list
    wall_time_s: float
    new_tokens: int

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / max(self.wall_time_s, 1e-9)


@torch.no_grad()
def generate_no_cache(model: SprintLM, idx: torch.Tensor, max_new_tokens: int,
                       temperature: float = 0.8, top_k: int = 40) -> GenerationResult:
    """Baseline decoding: re-run the full forward pass over the whole
    sequence-so-far at every step. Simple, correct, and O(T^2) in total work."""
    model.eval()
    device = idx.device
    ctx = model.cfg.context_length
    t0 = time.time()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -ctx:]
        logits, _, _ = model(idx_cond, start_pos=0)
        next_tok = sample_next_token(logits, temperature, top_k)
        idx = torch.cat([idx, next_tok], dim=1)
    torch.cuda.synchronize() if device.type == "cuda" else None
    wall = time.time() - t0
    return GenerationResult(tokens=idx[0].tolist(), wall_time_s=wall, new_tokens=max_new_tokens)


@torch.no_grad()
def generate_with_cache(model: SprintLM, idx: torch.Tensor, max_new_tokens: int,
                         temperature: float = 0.8, top_k: int = 40) -> GenerationResult:
    """KV-cached decoding: the prompt is processed once, then each new token
    only computes attention against its own query + the cached K/V of every
    prior position — O(T) total work instead of O(T^2)."""
    model.eval()
    device = idx.device
    ctx = model.cfg.context_length

    t0 = time.time()
    idx_cond = idx[:, -ctx:]
    logits, _, past_kv = model(idx_cond, start_pos=0, use_cache=True)
    next_tok = sample_next_token(logits, temperature, top_k)
    idx = torch.cat([idx, next_tok], dim=1)
    pos = idx_cond.shape[1]

    for _ in range(max_new_tokens - 1):
        if pos >= ctx:
            # context window full: for this minimal implementation we fall
            # back to a fresh (uncached) pass over the trailing window,
            # documented in README as a known limitation rather than
            # implementing sliding-window cache eviction.
            idx_cond = idx[:, -ctx:]
            logits, _, past_kv = model(idx_cond, start_pos=0, use_cache=True)
            pos = idx_cond.shape[1]
        else:
            logits, _, past_kv = model(
                next_tok, start_pos=pos, past_kv_list=past_kv, use_cache=True
            )
            pos += 1
        next_tok = sample_next_token(logits, temperature, top_k)
        idx = torch.cat([idx, next_tok], dim=1)

    torch.cuda.synchronize() if device.type == "cuda" else None
    wall = time.time() - t0
    return GenerationResult(tokens=idx[0].tolist(), wall_time_s=wall, new_tokens=max_new_tokens)
