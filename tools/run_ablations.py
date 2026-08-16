"""
Runs the full ablation ladder under a fixed compute budget and writes the
markdown table used in the README / docs site.

    baseline            : learned pos-emb + LayerNorm + GELU-MLP
    +RoPE               : RoPE          + LayerNorm + GELU-MLP
    +RoPE +RMSNorm      : RoPE          + RMSNorm   + GELU-MLP
    full (SprintLM)     : RoPE          + RMSNorm   + SwiGLU-MLP

Everything else (data, steps, batch size, seed, lr schedule) is held fixed,
which is what makes the comparison a controlled ablation rather than four
unrelated runs.

Usage:
    uv run python tools/run_ablations.py \
        dataset.train_bin=data/Prepared/TinyStoriesHF/train.bin \
        dataset.val_bin=data/Prepared/TinyStoriesHF/val.bin \
        train.max_steps=5000 train.seed=7
"""

import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ABLATIONS = [
    ("baseline",        dict(use_rope=False, use_rmsnorm=False, use_swiglu=False)),
    ("rope_only",       dict(use_rope=True,  use_rmsnorm=False, use_swiglu=False)),
    ("rope_rmsnorm",    dict(use_rope=True,  use_rmsnorm=True,  use_swiglu=False)),
    ("full_sprintlm",   dict(use_rope=True,  use_rmsnorm=True,  use_swiglu=True)),
]


def main():
    # Everything after the script name that isn't consumed here is forwarded
    # verbatim to each training run (dataset paths, max_steps, seed, etc).
    passthrough = sys.argv[1:]
    cli = OmegaConf.from_cli(passthrough)
    out_dir = cli.get("train", {}).get("out_dir", "outputs")

    results = []
    for tag, flags in ABLATIONS:
        run_name = f"{tag}_seed{cli.get('train', {}).get('seed', 7)}"
        overrides = [f"model.{k}={str(v).lower()}" for k, v in flags.items()]
        overrides += [f"train.run_name={run_name}"]
        cmd = [sys.executable, "-m", "sprintlm.train.train"] + passthrough + overrides
        print(f"\n[run_ablations] === {tag} ===\n{' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)

        metrics_path = Path(out_dir) / run_name / "metrics.jsonl"
        final_val, final_ppl = None, None
        with open(metrics_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("split") == "val":
                    final_val, final_ppl = rec["loss"], rec["ppl"]
        results.append((tag, run_name, final_val, final_ppl))

    print("\n\n| variant | RoPE | RMSNorm | SwiGLU | val loss | val ppl |")
    print("|---|---|---|---|---|---|")
    lines = ["| variant | RoPE | RMSNorm | SwiGLU | val loss | val ppl |", "|---|---|---|---|---|---|"]
    for tag, run_name, val, ppl in results:
        flags = dict(ABLATIONS)[tag]
        row = (f"| {tag} | {flags['use_rope']} | {flags['use_rmsnorm']} | "
               f"{flags['use_swiglu']} | {val:.4f} | {ppl:.2f} |")
        print(row)
        lines.append(row)

    out_path = Path(out_dir) / "ablation_results.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n[run_ablations] wrote {out_path}")


if __name__ == "__main__":
    main()
