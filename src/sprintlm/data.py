"""
Lightweight memmap-backed dataset for token streams.

Data is stored as a flat uint16 binary file (token ids concatenated across
documents, with an EOT token inserted between documents — see
tools/prepare_hf.py). We memmap rather than load into RAM so this scales
past what fits in memory, and re-open the memmap per __getitem__ call to
stay safe with multi-worker DataLoaders (each worker process needs its own
mmap handle).
"""

import numpy as np
import torch


class TokenDataset:
    """Not a torch.utils.data.Dataset on purpose: for a tiny LM, random-batch
    sampling directly from the memmap (as in nanoGPT-style training loops) is
    simpler and faster than going through a DataLoader for this scale."""

    def __init__(self, bin_path: str, context_length: int, dtype=np.uint16):
        self.bin_path = bin_path
        self.context_length = context_length
        self.dtype = dtype
        # touch once to fail fast if the path is wrong / file is empty
        arr = np.memmap(bin_path, dtype=dtype, mode="r")
        self.n_tokens = len(arr)
        if self.n_tokens <= context_length + 1:
            raise ValueError(
                f"{bin_path} has only {self.n_tokens} tokens, "
                f"need > context_length+1 = {context_length + 1}"
            )
        del arr

    def get_batch(self, batch_size: int, device: str):
        # Re-open memmap each call (cheap, OS page-caches it) to avoid a known
        # issue where a long-lived memmap handle slowly leaks memory across a
        # multi-hour training run.
        data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        ix = np.random.randint(0, self.n_tokens - self.context_length - 1, size=(batch_size,))
        x = torch.from_numpy(
            np.stack([data[i: i + self.context_length].astype(np.int64) for i in ix])
        )
        y = torch.from_numpy(
            np.stack([data[i + 1: i + 1 + self.context_length].astype(np.int64) for i in ix])
        )
        if "cuda" in device:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
