import torch

from sprintlm.config import ModelConfig
from sprintlm.model import GELUMLP, RMSNorm, SwiGLU, SprintLM
from sprintlm.inference.kv_cache import generate_no_cache, generate_with_cache


def tiny_cfg(**overrides) -> ModelConfig:
    base = dict(vocab_size=128, context_length=16, d_model=32, n_layers=2,
                n_heads=4, d_ff=64)
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_shape():
    cfg = tiny_cfg()
    model = SprintLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))
    logits, loss, _ = model(idx, targets=idx)
    assert logits.shape == (2, cfg.context_length, cfg.vocab_size)
    assert loss.item() > 0


def test_ablation_flags_swap_modules():
    full = SprintLM(tiny_cfg(use_rope=True, use_rmsnorm=True, use_swiglu=True))
    baseline = SprintLM(tiny_cfg(use_rope=False, use_rmsnorm=False, use_swiglu=False))

    assert isinstance(full.blocks[0].norm1, RMSNorm)
    assert isinstance(baseline.blocks[0].norm1, torch.nn.LayerNorm)

    assert isinstance(full.blocks[0].mlp, SwiGLU)
    assert isinstance(baseline.blocks[0].mlp, GELUMLP)

    assert full.pos_emb is None
    assert baseline.pos_emb is not None


def test_param_count_reasonably_matched_across_ablations():
    """Ablations should differ in *kind* of component, not wildly in size,
    or the comparison isn't controlled. SwiGLU has 3 matrices vs GELU-MLP's
    2, so we only assert they're within the same order of magnitude."""
    full = SprintLM(tiny_cfg(use_rope=True, use_rmsnorm=True, use_swiglu=True))
    baseline = SprintLM(tiny_cfg(use_rope=False, use_rmsnorm=False, use_swiglu=False))
    ratio = full.num_params() / baseline.num_params()
    assert 0.5 < ratio < 2.0


def test_kv_cache_matches_no_cache_greedy():
    """With temperature -> 0 (effectively greedy) and top_k=1, cached and
    uncached decoding must produce identical token sequences — this is the
    correctness check for the KV-cache implementation."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = SprintLM(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))

    no_cache = generate_no_cache(model, idx.clone(), max_new_tokens=6, temperature=1.0, top_k=1)
    with_cache = generate_with_cache(model, idx.clone(), max_new_tokens=6, temperature=1.0, top_k=1)

    assert no_cache.tokens == with_cache.tokens
