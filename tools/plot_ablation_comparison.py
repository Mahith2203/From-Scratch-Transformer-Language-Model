"""
Overlays validation-loss curves from multiple runs on a single chart —
built specifically for comparing the ablation ladder (baseline / rope_only
/ rope_rmsnorm / full_sprintlm) at a glance, rather than four separate
plots that have to be compared side by side.

Usage (defaults to the standard ablation run names):
    uv run python tools/plot_ablation_comparison.py

Or point it at custom run directories:
    uv run python tools/plot_ablation_comparison.py \
        --runs outputs/baseline_seed7 outputs/rope_only_seed7 \
               outputs/rope_rmsnorm_seed7 outputs/full_sprintlm_seed7 \
        --labels baseline +RoPE +RoPE+RMSNorm full

Writes outputs/ablation_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_val_curve(run_dir: Path):
    steps, losses = [], []
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"[plot_ablation_comparison] WARNING: {metrics_path} not found, skipping")
        return steps, losses
    with open(metrics_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("split") == "val":
                steps.append(rec["step"])
                losses.append(rec["loss"])
    return steps, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[
        "outputs/baseline_seed7",
        "outputs/rope_only_seed7",
        "outputs/rope_rmsnorm_seed7",
        "outputs/full_sprintlm_seed7",
    ])
    p.add_argument("--labels", nargs="+", default=[
        "baseline (no RoPE/RMSNorm/SwiGLU)",
        "+RoPE",
        "+RoPE +RMSNorm",
        "full SprintLM (+SwiGLU)",
    ])
    p.add_argument("--out", default="outputs/ablation_comparison.png")
    args = p.parse_args()

    assert len(args.runs) == len(args.labels), "runs and labels must be the same length"

    plt.figure(figsize=(8, 5))
    for run_dir, label in zip(args.runs, args.labels):
        steps, losses = load_val_curve(Path(run_dir))
        if steps:
            plt.plot(steps, losses, marker="o", label=label)
            # annotate the final point with its value for a quick read
            plt.annotate(f"{losses[-1]:.3f}", (steps[-1], losses[-1]),
                         textcoords="offset points", xytext=(5, 0), fontsize=8)

    plt.xlabel("step")
    plt.ylabel("validation loss")
    plt.title("Ablation study: validation loss by architecture variant")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_ablation_comparison] wrote {out_path}")


if __name__ == "__main__":
    main()
