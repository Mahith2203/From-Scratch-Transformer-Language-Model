"""
Sample text from a trained checkpoint.

Usage:
    uv run python tools/decode_cli.py \
        --ckpt outputs/full_seed7/ckpts/step5000.pt \
        --prompt "Once upon a time, " --context_length 192 \
        --temperature 0.8 --top_k 40 --max_new_tokens 120 --use_cache
"""

import argparse

import tiktoken
import torch

from sprintlm.inference.kv_cache import generate_no_cache, generate_with_cache
from sprintlm.model import SprintLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Once upon a time, ")
    p.add_argument("--context_length", type=int, default=None,
                    help="defaults to the value stored in the checkpoint's model config")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    if args.context_length:
        model_cfg.context_length = args.context_length

    model = SprintLM(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=args.device)

    gen_fn = generate_with_cache if args.use_cache else generate_no_cache
    result = gen_fn(model, idx, args.max_new_tokens, args.temperature, args.top_k)

    text = enc.decode(result.tokens)
    print(text)
    print(f"\n--- {'KV-cache' if args.use_cache else 'no-cache'}: "
          f"{result.new_tokens} tokens in {result.wall_time_s:.2f}s "
          f"({result.tokens_per_s:.1f} tok/s) ---")


if __name__ == "__main__":
    main()
