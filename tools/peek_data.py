"""
Peek at decoded text from the tokenized train/val .bin files.

Usage:
    python tools/peek_data.py
    python tools/peek_data.py --split val --n 5
    python tools/peek_data.py --bin data/Prepared/TinyStoriesHF/train.bin --n 3
"""

import argparse

import numpy as np
import tiktoken


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/Prepared/TinyStoriesHF")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--bin", default=None, help="override: exact path to a .bin file")
    p.add_argument("--n", type=int, default=5, help="number of stories to print")
    p.add_argument("--tokenizer", default="gpt2")
    args = p.parse_args()

    bin_path = args.bin or f"{args.out_dir}/{args.split}.bin"
    enc = tiktoken.get_encoding(args.tokenizer)
    eot = enc.eot_token

    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    print(f"[peek_data] {bin_path} — {len(data):,} tokens total\n")

    story_ids = []
    n_shown = 0
    for tok in data:
        if n_shown >= args.n:
            break
        if tok == eot:
            if story_ids:
                text = enc.decode(story_ids)
                print(f"--- story {n_shown + 1} ({len(story_ids)} tokens) ---")
                print(text.strip())
                print()
                n_shown += 1
            story_ids = []
        else:
            story_ids.append(int(tok))


if __name__ == "__main__":
    main()
