"""
Usage:
    uv run python tools/plot_metrics.py outputs/full_seed7

Reads metrics.jsonl from the given run directory and writes
loss_train.png, loss_val.png, tokps.png into <run_dir>/assets/.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(run_dir: Path):
    train_steps, train_loss = [], []
    val_steps, val_loss = [], []
    tok_steps, tok_per_s = [], []
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("split") == "train":
                train_steps.append(rec["step"])
                train_loss.append(rec["loss"])
                if "tok_per_s" in rec:
                    tok_steps.append(rec["step"])
                    tok_per_s.append(rec["tok_per_s"])
            elif rec.get("split") == "val":
                val_steps.append(rec["step"])
                val_loss.append(rec["loss"])
    return train_steps, train_loss, val_steps, val_loss, tok_steps, tok_per_s


def main():
    run_dir = Path(sys.argv[1])
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    train_steps, train_loss, val_steps, val_loss, tok_steps, tok_per_s = load_metrics(run_dir)

    plt.figure()
    plt.plot(train_steps, train_loss)
    plt.xlabel("step"); plt.ylabel("train loss"); plt.title("Training loss")
    plt.savefig(assets_dir / "loss_train.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(val_steps, val_loss, marker="o")
    plt.xlabel("step"); plt.ylabel("val loss"); plt.title("Validation loss")
    if val_loss:
        plt.gca().annotate(f"final: {val_loss[-1]:.3f}",
                            (val_steps[-1], val_loss[-1]))
    plt.savefig(assets_dir / "loss_val.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(tok_steps, [t / 1000 for t in tok_per_s])
    plt.xlabel("step"); plt.ylabel("k tokens/sec"); plt.title("Throughput")
    plt.savefig(assets_dir / "tokps.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[plot_metrics] wrote plots to {assets_dir}")


if __name__ == "__main__":
    main()
