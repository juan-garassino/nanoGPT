"""Training helpers: loss estimation and the learning-rate schedule."""


from contextlib import AbstractContextManager
from typing import Dict

import numpy as np
import torch
from nanoGPT.sourcing.sourcing import get_batch
import math

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, train_data: np.ndarray, val_data: np.ndarray, eval_iters: int,
                  batch_size: int, block_size: int, device_type: str, device: str,
                  ctx: AbstractContextManager) -> Dict[str, torch.Tensor]:
    """Average the loss over ``eval_iters`` random batches for each split.

    Puts ``model`` in eval mode for the measurement and back in train mode after.

    Parameters
    ----------
    model : torch.nn.Module
        Called as ``model(X, Y)`` and returning ``(logits, loss)``.
    train_data, val_data : numpy.ndarray
        1-D ``uint16`` token arrays (typically ``np.memmap``).
    eval_iters : int
        Number of batches per split.
    batch_size, block_size : int
        Batch shape, ``(batch_size, block_size)``.
    device_type : str
        ``"cuda"`` or ``"cpu"``.
    device : str
        Device the batches are moved to.
    ctx : contextlib.AbstractContextManager
        Context (e.g. autocast) wrapped around the forward pass.

    Returns
    -------
    dict of {str: torch.Tensor}
        Scalar mean loss under the keys ``"train"`` and ``"val"``.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, batch_size, block_size, device_type, device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it: int, warmup_iters: int, learning_rate: float, lr_decay_iters: int, min_lr: float) -> float:
    """Learning rate at iteration ``it``: linear warmup, then cosine decay to ``min_lr``.

    Parameters
    ----------
    it : int
        Current iteration.
    warmup_iters : int
        Iterations of linear warmup from 0 to ``learning_rate``.
    learning_rate : float
        Peak learning rate.
    lr_decay_iters : int
        Iteration at which the cosine decay reaches ``min_lr``; ``min_lr`` after.
    min_lr : float
        Floor of the schedule.

    Returns
    -------
    float
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
