from nanoGPT.model.model import GPT, GPTConfig
import torch
import os

def initialize_model(init_from, out_dir, device, model_args, meta_vocab_size=None):
    if init_from == "scratch":
        # Initialize a new model from scratch
        print("Initializing a new model from scratch")
        # Determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # Resume training from a checkpoint
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # Force these config attributes to be equal; otherwise, we can't even resume training
        # The rest of the attributes (e.g., dropout) can stay as desired from the command line
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        # Create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # Fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # Initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=model_args["dropout"])
        model = GPT.from_pretrained(init_from, override_args)
        # Read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)

    return model, model_args
