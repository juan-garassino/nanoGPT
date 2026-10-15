"""Character-level tokenisation of a text corpus into ``train.bin`` / ``val.bin`` / ``meta.pkl``."""


import os
import numpy as np
import pickle

from nanoGPT.sourcing.sourcing import sourcing_data

def character_tokenizer(data: str) -> None:
    """Encode text at character level and write the splits next to this module.

    The first 90% of ``data`` is the train split, the rest validation. Writes
    ``train.bin`` and ``val.bin`` (``uint16`` ids) and ``meta.pkl`` (a dict with
    ``vocab_size``, ``itos`` and ``stoi``) into the directory of this file.

    Parameters
    ----------
    data : str
        Raw text.
    """
    # Get all the unique characters that occur in the text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # Create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        """Map a string to a list of character ids."""
        return [stoi[c] for c in s]

    def decode(l):
        """Map a list of character ids back to a string."""
        return "".join([itos[i] for i in l])

    # 90/10 train/validation split
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # Save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    # Sourcing the data
    txt_folder_path = "/path/to/folder"

    data = sourcing_data(txt_folder_path)

    character_tokenizer(data)
