"""
SprintLM training loop.

Usage (matches the original repo's dotted-override CLI style):

    uv run python -m sprintlm.train.train \
        dataset.train_bin=data/Prepared/TinyStoriesHF/train.bin \
        dataset.val_bin=data/Prepared/TinyStoriesHF/val.bin \
        train.batch_size=24 dataset.context_length=192 \
        train.max_steps=5000 train.log_every=50 \
        train.eval_every=250 train.save_every=250

Ablation runs just add model.* overrides, e.g.:

    uv run python -m sprintlm.train.train \
        model.use_rope=false model.use_rmsnorm=false model.use_swiglu=false \
        train.run_name=baseline_seed7 train.seed=7 ...

Every run writes:
    outputs/<run_name>/metrics.jsonl   (one JSON line per logged step)
    outputs/<run_name>/ckpts/step*.pt
    outputs/<run_name>/config.yaml     (fully resolved config, for reproducibility)
"""

import json
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sprintlm.config import Config
from sprintlm.data import TokenDataset
from sprintlm.model import SprintLM


def pick_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def derive_run_name(cfg: Config) -> str:
    m = cfg.model
    tag = "full" if (m.use_rope and m.use_rmsnorm and m.use_swiglu) else (
        "baseline" if not (m.use_rope or m.use_rmsnorm or m.use_swiglu) else
        f"rope{int(m.use_rope)}_rms{int(m.use_rmsnorm)}_swiglu{int(m.use_swiglu)}"
    )
    return f"{tag}_seed{cfg.train.seed}"


def lr_at_step(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model, dataset: TokenDataset, batch_size: int, iters: int, device: str) -> float:
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = dataset.get_batch(batch_size, device)
        _, loss, _ = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    base = OmegaConf.structured(Config)
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base, cli)
    cfg = OmegaConf.to_object(cfg)  # back to plain dataclasses, cli-validated by structured merge

    torch.manual_seed(cfg.train.seed)

    device = pick_device(cfg.train.device)
    run_name = cfg.train.run_name or derive_run_name(cfg)
    run_dir = Path(cfg.train.out_dir) / run_name
    ckpt_dir = run_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(OmegaConf.structured(cfg), run_dir / "config.yaml")
    metrics_path = run_dir / "metrics.jsonl"

    print(f"[sprintlm] run_name={run_name} device={device}")
    print(f"[sprintlm] ablation flags: rope={cfg.model.use_rope} "
          f"rmsnorm={cfg.model.use_rmsnorm} swiglu={cfg.model.use_swiglu}")

    cfg.model.context_length = cfg.dataset.context_length
    train_ds = TokenDataset(cfg.dataset.train_bin, cfg.dataset.context_length)
    val_ds = TokenDataset(cfg.dataset.val_bin, cfg.dataset.context_length)

    model = SprintLM(cfg.model).to(device)
    n_params = model.num_params()
    print(f"[sprintlm] non-embedding params: {n_params/1e6:.2f}M")

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[cfg.train.dtype]
    use_amp = device == "cuda" and dtype != torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and dtype == torch.float16))

    optimizer = model.configure_optimizer(
        cfg.train.weight_decay, cfg.train.lr, (cfg.train.beta1, cfg.train.beta2), cfg.train.eps
    )

    if cfg.train.compile:
        model = torch.compile(model)

    t0 = time.time()
    tokens_seen = 0
    metrics_file = open(metrics_path, "a")

    for step in range(cfg.train.max_steps + 1):
        lr = lr_at_step(step, cfg.train)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = train_ds.get_batch(cfg.train.batch_size, device)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            _, loss, _ = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        tokens_seen += cfg.train.batch_size * cfg.dataset.context_length

        if step % cfg.train.log_every == 0:
            elapsed = time.time() - t0
            tok_per_s = tokens_seen / max(elapsed, 1e-6)
            rec = {
                "step": step, "split": "train", "loss": loss.item(),
                "lr": lr, "grad_norm": float(grad_norm), "tok_per_s": tok_per_s,
                "tokens_seen": tokens_seen, "elapsed_s": elapsed,
            }
            print(f"step {step:5d} | train loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | grad_norm {float(grad_norm):.3f} | "
                  f"{tok_per_s/1000:.2f}k tok/s")
            metrics_file.write(json.dumps(rec) + "\n")
            metrics_file.flush()

        if step % cfg.train.eval_every == 0 and step > 0:
            val_loss = evaluate(model, val_ds, cfg.train.batch_size, cfg.train.eval_iters, device)
            ppl = math.exp(val_loss)
            rec = {"step": step, "split": "val", "loss": val_loss, "ppl": ppl}
            print(f"step {step:5d} | val loss {val_loss:.4f} | ppl {ppl:.2f}")
            metrics_file.write(json.dumps(rec) + "\n")
            metrics_file.flush()

        if step % cfg.train.save_every == 0 and step > 0:
            ckpt = {
                "model": model.state_dict(),
                "model_cfg": cfg.model,
                "step": step,
            }
            torch.save(ckpt, ckpt_dir / f"step{step}.pt")

    metrics_file.close()
    final_val = evaluate(model, val_ds, cfg.train.batch_size, cfg.train.eval_iters, device)
    print(f"[sprintlm] FINAL val loss {final_val:.4f} | ppl {math.exp(final_val):.2f}")
    return final_val


if __name__ == "__main__":
    main()
