"""
Trains the full SprintLM config across several seeds to report the
val-loss spread — protects the headline number from "was that just a
lucky run?" in interviews.

Usage:
    uv run python tools/seed_variance.py \
        dataset.train_bin=... dataset.val_bin=... \
        train.max_steps=5000 --seeds 7 13 42
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 42])
    args, passthrough = parser.parse_known_args()

    cli = OmegaConf.from_cli(passthrough)
    out_dir = cli.get("train", {}).get("out_dir", "outputs")

    val_losses = []
    for seed in args.seeds:
        run_name = f"full_seedvar_{seed}"
        cmd = ([sys.executable, "-m", "sprintlm.train.train"] + passthrough +
               [f"train.seed={seed}", f"train.run_name={run_name}"])
        print(f"\n[seed_variance] === seed {seed} ===\n{' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)

        metrics_path = Path(out_dir) / run_name / "metrics.jsonl"
        final_val = None
        with open(metrics_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("split") == "val":
                    final_val = rec["loss"]
        val_losses.append(final_val)
        print(f"[seed_variance] seed={seed} final val loss={final_val:.4f}")

    mean = statistics.mean(val_losses)
    stdev = statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0
    print(f"\n[seed_variance] val loss = {mean:.4f} ± {stdev:.4f} "
          f"across seeds {args.seeds}")

    out_path = Path(out_dir) / "seed_variance.md"
    out_path.write_text(
        f"Val loss across seeds {args.seeds}: **{mean:.4f} ± {stdev:.4f}**\n\n"
        f"Raw values: {val_losses}\n"
    )
    print(f"[seed_variance] wrote {out_path}")


if __name__ == "__main__":
    main()
