"""Text loading and random batch sampling from tokenised ``.bin`` files."""

import os
from typing import Optional, Tuple

import numpy as np
import requests
import torch


def sourcing_data(txt_folder: Optional[str] = None) -> str:
    """Load raw text.

    Parameters
    ----------
    txt_folder : str, optional
        Folder whose ``*.txt`` files are concatenated. If None, the tiny
        Shakespeare file is downloaded to ``input.txt`` next to this module
        (once) and read.

    Returns
    -------
    str
        The text.
    """
    if txt_folder is None:
        # Download the tiny shakespeare dataset
        input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
        if not os.path.exists(input_file_path):
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(input_file_path, "w") as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, "r") as f:
            data = f.read()
    else:
        # Load data from text files in the given folder
        data = ""
        for file_name in os.listdir(txt_folder):
            file_path = os.path.join(txt_folder, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                with open(file_path, "r") as f:
                    data += f.read()

    return data


def get_batch(
    train_data: np.ndarray,
    val_data: np.ndarray,
    split: str,
    batch_size: int,
    block_size: int,
    device_type: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of next-token-prediction pairs.

    Parameters
    ----------
    train_data, val_data : numpy.ndarray
        1-D ``uint16`` token arrays (typically ``np.memmap``).
    split : str
        ``"train"`` selects ``train_data``; anything else selects ``val_data``.
    batch_size, block_size : int
        Number of sequences and tokens per sequence.
    device_type : str
        ``"cuda"`` pins memory and copies asynchronously.
    device : str
        Target device.

    Returns
    -------
    x, y : torch.Tensor
        ``torch.int64`` tensors of shape ``(batch_size, block_size)``; ``y`` is
        ``x`` shifted left by one token.
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y
