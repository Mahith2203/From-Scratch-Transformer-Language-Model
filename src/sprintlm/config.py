"""
Config dataclasses for SprintLM.

Design note: the ablation study is driven entirely by three booleans on
ModelConfig (use_rope, use_rmsnorm, use_swiglu). Flipping them swaps in the
"vanilla GPT-2 style" component (learned pos-emb / LayerNorm / GELU-MLP)
instead of the modern one, while everything else (param count target,
training schedule, data) stays fixed. This is what makes the ablation table
a controlled comparison rather than four unrelated runs.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 50257          # GPT-2 BPE vocab (tiktoken "gpt2")
    context_length: int = 192
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    d_ff: int = 1536                 # for SwiGLU this is the *inner* dim before the gate split
    dropout: float = 0.0
    rope_theta: float = 10000.0

    # --- ablation switches ---
    use_rope: bool = True            # False -> learned absolute positional embeddings
    use_rmsnorm: bool = True         # False -> LayerNorm
    use_swiglu: bool = True          # False -> GELU MLP (2-layer, d_ff hidden)

    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class RuntimeDatasetConfig:
    """Paths + context length used at train/eval time (as opposed to
    DatasetPrepConfig below, which controls the offline tokenization step)."""
    train_bin: str = "data/Prepared/TinyStoriesHF/train.bin"
    val_bin: str = "data/Prepared/TinyStoriesHF/val.bin"
    context_length: int = 192


@dataclass
class TrainConfig:
    batch_size: int = 24
    max_steps: int = 5000
    log_every: int = 50
    eval_every: int = 250
    eval_iters: int = 50
    save_every: int = 250

    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip: float = 1.0

    seed: int = 7
    device: str = "cuda"             # falls back to mps/cpu automatically if unavailable
    compile: bool = False
    dtype: str = "bfloat16"          # "float32" | "float16" | "bfloat16"

    out_dir: str = "outputs"
    run_name: Optional[str] = None   # if None, derived from ablation flags + seed


@dataclass
class Config:
    """Top-level config combining the three groups the CLI overrides target:
    `model.*`, `train.*`, `dataset.*` (e.g. `train.batch_size=24`)."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: RuntimeDatasetConfig = field(default_factory=RuntimeDatasetConfig)


@dataclass
class DatasetPrepConfig:
    dataset_name: str = "roneneldan/TinyStories"
    text_field: str = "text"
    out_dir: str = "data/Prepared/TinyStoriesHF"
    tokenizer_name: str = "gpt2"
    append_eot: bool = True
    dtype: str = "uint16"
    val_fraction: float = 0.005
