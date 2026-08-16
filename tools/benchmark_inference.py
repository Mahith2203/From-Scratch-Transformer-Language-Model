"""
Produces the KV-cache before/after latency table for the write-up.

Usage:
    uv run python tools/benchmark_inference.py \
        --ckpt outputs/full_seed7/ckpts/step5000.pt \
        --lengths 32 64 128 256 --repeats 3

Runs both generate_no_cache and generate_with_cache for each generation
length, several repeats each (to smooth out warm-up / scheduling noise),
and writes a markdown table to outputs/<run>/inference_benchmark.md.
"""

import argparse
import statistics
from pathlib import Path

import tiktoken
import torch

from sprintlm.inference.kv_cache import generate_no_cache, generate_with_cache
from sprintlm.model import SprintLM


def bench(gen_fn, model, idx, n_tokens, temperature, top_k, repeats):
    times, tps = [], []
    for _ in range(repeats):
        result = gen_fn(model, idx, n_tokens, temperature, top_k)
        times.append(result.wall_time_s)
        tps.append(result.tokens_per_s)
    return statistics.mean(times), statistics.mean(tps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prompt", default="Once upon a time, ")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = SprintLM(ckpt["model_cfg"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=args.device)

    # warm-up (first CUDA call pays kernel-compile/allocator cost)
    generate_no_cache(model, idx, 8, args.temperature, args.top_k)
    generate_with_cache(model, idx, 8, args.temperature, args.top_k)

    rows = []
    for n in args.lengths:
        t_nc, tps_nc = bench(generate_no_cache, model, idx, n, args.temperature, args.top_k, args.repeats)
        t_c, tps_c = bench(generate_with_cache, model, idx, n, args.temperature, args.top_k, args.repeats)
        speedup = t_nc / max(t_c, 1e-9)
        rows.append((n, t_nc, tps_nc, t_c, tps_c, speedup))
        print(f"n={n:4d} | no-cache {t_nc:6.3f}s ({tps_nc:6.1f} tok/s) | "
              f"kv-cache {t_c:6.3f}s ({tps_c:6.1f} tok/s) | speedup {speedup:.2f}x")

    out_dir = Path(args.ckpt).parent.parent
    md_path = out_dir / "inference_benchmark.md"
    with open(md_path, "w") as f:
        f.write("| new_tokens | no-cache (s) | no-cache tok/s | kv-cache (s) | kv-cache tok/s | speedup |\n")
        f.write("|---|---|---|---|---|---|\n")
        for n, t_nc, tps_nc, t_c, tps_c, speedup in rows:
            f.write(f"| {n} | {t_nc:.3f} | {tps_nc:.1f} | {t_c:.3f} | {tps_c:.1f} | {speedup:.2f}x |\n")
    print(f"\n[benchmark_inference] wrote {md_path}")


if __name__ == "__main__":
    main()
