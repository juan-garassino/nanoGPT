import torch
import numpy as np
import os
import requests

def sourcing_data(txt_folder=None):
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


def get_batch(train_data, val_data, split, batch_size, block_size, device_type, device):
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
