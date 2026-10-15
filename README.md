# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

A personal learning fork of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT): a small GPT-2 style language model in PyTorch, reorganised from the original single-file scripts into an importable `nanoGPT` package with a single entry point, `main.py`.

The model code (`nanoGPT/model/model.py`) follows the upstream design: a decoder-only Transformer with causal self-attention, weight tying between the token embedding and the output head, and an optional loader for the OpenAI GPT-2 checkpoints through Hugging Face `transformers`. The fork adds a weight-heatmap plotting helper, a character-level tokenizer and a GPT-2 BPE tokenizer for custom text folders, and a small Spanish-language corpus (Cortazar, Bioy Casares) under `data/Self LLM/`.

**Status:** work in progress. The training script runs end to end on CPU, but it has known limitations (see [Known limitations](#known-limitations)). Nothing in this repository has been trained to convergence, and no trained checkpoints or sample outputs are included.

## Contents

- [Repository layout](#repository-layout)
- [Install](#install)
- [Quick start](#quick-start)
- [Training options](#training-options)
- [Results and artifacts](#results-and-artifacts)
- [Notebooks](#notebooks)
- [Known limitations](#known-limitations)
- [Credits](#credits)

## Repository layout

| Path | What it contains |
| --- | --- |
| `main.py` | Training entry point (argparse options, optional DDP, AdamW, gradient accumulation, checkpointing, heatmap plotting). |
| `nanoGPT/model/model.py` | `GPT`, `GPTConfig`, and the building blocks (`LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`). |
| `nanoGPT/model/init_from.py` | `initialize_model`: build a model from scratch, resume from `ckpt.pt`, or load GPT-2 weights. |
| `nanoGPT/model/tools.py` | `estimate_loss` and `get_lr` (cosine schedule with linear warmup). |
| `nanoGPT/sourcing/sourcing.py` | `sourcing_data` (load text) and `get_batch` (random batches from a `.bin` memmap). |
| `nanoGPT/tokenizer/` | `character_tokenizer` and `subword_tokenizer`: turn text into `train.bin` / `val.bin`. |
| `nanoGPT/heatmaps/heatmaps.py` | `plot_heatmaps`: save heatmaps of the model's weight matrices. |
| `nanoGPT/data/*/prepare.py` | Dataset preparation scripts (tiny Shakespeare, OpenWebText). |
| `nanoGPT/configurator/configurator.py` | Upstream "poor man's configurator", executed by `main.py`. |
| `nanoGPT/trainer/trainer.py` | Copy of the upstream `train.py`, kept for reference. It is not importable as is. |
| `config/` | Upstream-style config files. `main.py` does not read these yet (see limitations). |
| `samples/` | Upstream `sample.py` and `bench.py`, kept for reference. They are not wired to the package yet. |
| `plots/output_folder/` | Weight heatmaps written by `plot_heatmaps`. |
| `notebooks/` | Analysis notebooks (see [Notebooks](#notebooks)). |
| `data/Self LLM/` | Plain-text corpus (Spanish short stories). |

## Install

Python 3.12 or newer is required (`pyproject.toml`). With [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Or with pip:

```
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt` and `pyproject.toml`: torch, numpy, transformers, datasets, tiktoken, wandb, tqdm, black, matplotlib. The prepare scripts also import `requests`, which is not listed there but is normally present as a dependency of `transformers` and `datasets`.

## Quick start

These steps were run on a CPU-only Linux machine with Python 3.12 and PyTorch 2.14.

**1. Prepare the character-level Shakespeare data** (downloads about 1 MB):

```
python nanoGPT/data/shakespeare_char/prepare.py
```

This writes `train.bin`, `val.bin` and `meta.pkl` next to the script. Printed output: vocab size 65, 1,003,854 training tokens, 111,540 validation tokens (also recorded in `nanoGPT/data/shakespeare_char/readme.md`).

**2. Make the hard-coded paths resolve.** `main.py` and `plot_heatmaps` currently look for the repository at `$HOME/Code/juan-garassino/mySandbox/nanoGPT`. If your clone lives elsewhere, link it there:

```
mkdir -p ~/Code/juan-garassino/mySandbox
ln -s "$PWD" ~/Code/juan-garassino/mySandbox/nanoGPT
```

**3. Train.** `main.py` must currently be run without command-line options (see [Known limitations](#known-limitations)); the defaults are a tiny model (2 layers, 2 heads, 16 embedding channels, block size 8) on CPU:

```
MPLBACKEND=Agg python main.py
```

Expected start of the output (from a verified run):

```
tokens per iteration will be: 1,280
Initializing a new model from scratch
number of parameters: 0.81M
step 0: train loss 10.8225, val loss 10.8227
iter 0: loss 10.8459, time 2638.74ms, mfu -100.00%
```

The initial loss of about 10.8 is close to ln(50304) because the from-scratch path uses a 50,304 token vocabulary, not the 65-character vocabulary of the prepared data (see limitations). After every iteration `main.py` also writes a weight-heatmap PNG into `plots/output_folder/`, which takes several seconds per iteration. Stop the run with Ctrl-C when you have seen enough; the default is 3,000 iterations.

**Use the model directly.** The model can be exercised without the training script:

```python
import torch
from nanoGPT.model.model import GPT, GPTConfig

model = GPT(GPTConfig(block_size=8, vocab_size=65, n_layer=2, n_head=2, n_embd=16))
x = torch.randint(0, 65, (4, 8))          # (batch, time) token ids
logits, loss = model(x, x)                # logits (4, 8, 65), scalar loss
out = model.generate(x[:1, :3], max_new_tokens=5)  # shape (1, 8)
```

## Training options

`python main.py --help` lists every option. The main ones and their defaults:

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `shakespeare_char` | Folder name under `nanoGPT/data/` that holds `train.bin` and `val.bin`. |
| `--n_layer`, `--n_head`, `--n_embd` | 2, 2, 16 | Model size. |
| `--block_size`, `--batch_size` | 8, 4 | Context length and micro-batch size. |
| `--gradient_accumulation_steps` | 40 | Micro-batches per optimiser step. |
| `--max_iters` | 3000 | Training iterations. |
| `--learning_rate`, `--decay_lr` | 6e-4, off | Peak learning rate and optional cosine decay. |
| `--eval_interval`, `--eval_iters` | 2000, 200 | How often and how many batches to use for loss estimation. |
| `--init_from` | `scratch` | `scratch` or `resume` (reads `<out_dir>/ckpt.pt`). |
| `--device`, `--dtype` | `cpu`, `float16` | Device and autocast dtype (autocast is only applied on CUDA). |
| `--wandb_log` | off | Log to Weights & Biases. |

Multi-GPU training uses DistributedDataParallel when launched with `torchrun` (the code reads `RANK`, `LOCAL_RANK` and `WORLD_SIZE`). This path has not been exercised in this repository.

## Results and artifacts

This repository does not contain measured training results of its own. What is checked in:

| Artifact | Source |
| --- | --- |
| `plots/output_folder/heatmaps[2023-07-03-*].png` | Ten weight-matrix heatmaps in the `plot_heatmaps` output format, timestamped 2023-07-03 in the file names. The run that produced them is not documented in the repository. |
| `notebooks/attention_plot.png` | Image written by `notebooks/diffusionmaps.ipynb` (the notebook references this file name). |
| `assets/gpt2_124M_loss.png` | Loss curve inherited from the upstream nanoGPT README (GPT-2 124M reproduction on OpenWebText). It was not produced by code in this repository. |

For reference numbers on GPT-2 reproduction, character-level Shakespeare and the OpenAI checkpoint baselines, see the upstream [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) README. They are deliberately not repeated here because they cannot be reproduced from files in this repository.

## Notebooks

- `notebooks/transformer_sizing.ipynb`: parameter, FLOP and memory estimates for a Transformer (from upstream).
- `notebooks/scaling_laws.ipynb`: parameter and compute calculations following the Chinchilla paper (from upstream; the first cell notes it does not match the paper's numbers exactly).
- `notebooks/diffusionmaps.ipynb`: loads the Hugging Face `gpt2` model and plots weight heatmaps.

## Known limitations

These are documented rather than fixed in this cleanup, which made no behaviour changes:

- **Command-line options do not work with `main.py`.** It parses options with `argparse` and then also executes `nanoGPT/configurator/configurator.py`, which rejects them (`--max_iters 2` triggers an `AssertionError`; `--max_iters=2` raises `ValueError: Unknown config key`). Run it with no options, or edit the defaults in `parse_args`.
- **Hard-coded paths.** `main.py` and `plot_heatmaps` use `$HOME/Code/juan-garassino/mySandbox/nanoGPT`; the tokenizer scripts use the placeholder `/path/to/folder`.
- **Vocabulary size.** The from-scratch path always uses 50,304 tokens (`initialize_model`), even for the 65-character Shakespeare data.
- **`--decay_lr`** calls `get_lr(iter_num)` but `nanoGPT/model/tools.py:get_lr` takes five arguments.
- **`--init_from resume`** unpacks `iter_num` and `best_val_loss` in the opposite order to what `initialize_model` returns, and `--init_from gpt2*` is not handled in `main.py`.
- **Heatmaps every iteration** are slow and write a new PNG each time.
- `nanoGPT/trainer/trainer.py`, `samples/sample.py`, `samples/bench.py` and `config/*.py` are upstream scripts that expect a top-level `model/` package and `train.py`; they do not run in this layout. `nanoGPT/generator/generator.py` and `nanoGPT/data/cortazar/prepare.py` are empty files.
- There is no automated test suite.

## Credits

Based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT licence upstream) and [minGPT](https://github.com/karpathy/minGPT). Background: Andrej Karpathy's [Zero To Hero](https://karpathy.ai/zero-to-hero.html) series.
