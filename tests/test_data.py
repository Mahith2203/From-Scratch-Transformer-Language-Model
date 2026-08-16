import numpy as np
import pytest
import torch

from sprintlm.data import TokenDataset


def test_get_batch_shapes(tmp_path):
    n_tokens = 1000
    arr = np.arange(n_tokens, dtype=np.uint16)
    bin_path = tmp_path / "toy.bin"
    arr.tofile(bin_path)

    ds = TokenDataset(str(bin_path), context_length=16)
    x, y = ds.get_batch(batch_size=4, device="cpu")
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # since the underlying data is just arange(), y should equal x + 1
    # everywhere, which also confirms next-token alignment is correct.
    assert torch.equal(y, x + 1)


def test_raises_on_too_short_file(tmp_path):
    arr = np.arange(4, dtype=np.uint16)
    bin_path = tmp_path / "tiny.bin"
    arr.tofile(bin_path)
    with pytest.raises(ValueError):
        TokenDataset(str(bin_path), context_length=16)
