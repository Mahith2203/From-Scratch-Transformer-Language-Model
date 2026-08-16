"""
Tokenize a Hugging Face text dataset into flat uint16 .bin files for
memmap-based training.

Usage:
    uv run python tools/prepare_hf.py \
        dataset_name=roneneldan/TinyStories text_field=text \
        out_dir=data/Prepared/TinyStoriesHF append_eot=true \
        dtype=uint16 tokenizer_name=gpt2

Packs documents back-to-back separated by an end-of-text token (rather than
padding each document to context_length), so no compute is wasted on pad
tokens during training — this is why context_length doesn't need to match
any particular document's length.
"""

from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from sprintlm.config import DatasetPrepConfig


def main():
    base = OmegaConf.structured(DatasetPrepConfig)
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.to_object(OmegaConf.merge(base, cli))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(cfg.tokenizer_name)
    eot = enc.eot_token if hasattr(enc, "eot_token") else enc.encode(
        "<|endoftext|>", allowed_special="all"
    )[0]

    print(f"[prepare_hf] loading {cfg.dataset_name} ...")
    ds = load_dataset(cfg.dataset_name)

    # TinyStories ships with its own train/validation split; fall back to a
    # manual split for datasets that only provide "train".
    if "validation" in ds:
        train_split, val_split = ds["train"], ds["validation"]
    else:
        split = ds["train"].train_test_split(test_size=cfg.val_fraction, seed=7)
        train_split, val_split = split["train"], split["test"]

    dtype = np.dtype(cfg.dtype)

    def tokenize_split(split, name):
        all_ids = []
        for row in tqdm(split, desc=f"tokenizing {name}"):
            text = row[cfg.text_field]
            ids = enc.encode_ordinary(text)
            if cfg.append_eot:
                ids.append(eot)
            all_ids.extend(ids)
        arr = np.array(all_ids, dtype=dtype)
        path = out_dir / f"{name}.bin"
        arr.tofile(path)
        print(f"[prepare_hf] wrote {path} ({len(arr):,} tokens, {arr.nbytes/1e6:.1f} MB)")
        return len(arr)

    n_train = tokenize_split(train_split, "train")
    n_val = tokenize_split(val_split, "val")

    print(f"[prepare_hf] done. train={n_train:,} tokens, val={n_val:,} tokens, "
          f"vocab_size={enc.n_vocab}")


if __name__ == "__main__":
    main()
