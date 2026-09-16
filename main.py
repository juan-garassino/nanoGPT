import argparse
import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import time
from contextlib import nullcontext
import numpy as np

from nanoGPT.sourcing.sourcing import sourcing_data, get_batch
from nanoGPT.model.model import GPT, GPTConfig
from nanoGPT.model.init_from import initialize_model
from nanoGPT.model.tools import get_lr, estimate_loss
from nanoGPT.heatmaps.heatmaps import plot_heatmaps

def parse_args():
    parser = argparse.ArgumentParser(description="GPT2 Training Configuration")

    # I/O
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Evaluation interval")
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--eval_iters", type=int, default=200, help="Number of evaluation iterations")
    parser.add_argument("--eval_only", action="store_true", help="If True, script exits after the first evaluation")
    parser.add_argument("--always_save_checkpoint", action="store_true", help="If True, always save a checkpoint after each evaluation")
    parser.add_argument("--init_from", type=str, default="scratch", help="'scratch', 'resume', or 'gpt2*'")

    # Wandb logging
    parser.add_argument("--wandb_log", action="store_true", help="Enable Wandb logging")
    parser.add_argument("--wandb_project", type=str, default="owt", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="gpt2", help="Wandb run name")

    # Data
    parser.add_argument("--dataset", type=str, default="shakespeare_char", help="Dataset name")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5 * 8, help="Number of gradient accumulation steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size") # 12
    parser.add_argument("--block_size", type=int, default=8, help="Block size") # 1024

    # Model
    parser.add_argument("--n_layer", type=int, default=2, help="Number of layers") # 12
    parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads") # 12
    parser.add_argument("--n_embd", type=int, default=16, help="Embedding dimension") # 768
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--bias", action="store_true", help="Use bias inside LayerNorm and Linear layers")

    # AdamW optimizer
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--max_iters", type=int, default=3000, help="Total number of training iterations")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip value")

    # Learning rate decay settings
    parser.add_argument("--decay_lr", action="store_true", help="Whether to decay the learning rate")
    parser.add_argument("--warmup_iters", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--lr_decay_iters", type=int, default=600000, help="Learning rate decay iterations")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")

    # DDP settings
    parser.add_argument("--backend", type=str, default="nccl", help="DDP backend")

    # System
    parser.add_argument("--device", type=str, default="cpu", help="Device") # CUDA

    parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    #parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float16", help="Data type")
    parser.add_argument("--compile", action="store_false", help="Compile the model using PyTorch 2.0") # TRUE

    return parser.parse_args()

args = parse_args()

print(args)

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]

exec(open(f"{os.environ.get('HOME')}/Code/juan-garassino/mySandbox/nanoGPT/nanoGPT/configurator/configurator.py").read())  # overrides from command line or config file

config = {k: globals()[k] for k in config_keys}  # will be useful for logging

ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?

if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert args.gradient_accumulation_steps % ddp_world_size == 0
    args.gradient_accumulation_steps //= ddp_world_size

else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size

print(f"tokens per iteration will be: {tokens_per_iter:,}")

iter_num = 0

best_val_loss = 1e9

device_type = "cuda" if "cuda" in args.device else "cpu"  # for later use in torch.autocast

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[args.dtype]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

print('model_args')

# model init
model_args = dict(
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    block_size=args.block_size,
    bias=args.bias,
    vocab_size=None,
    dropout=args.dropout,
)  # start with model_args from command line

if args.init_from == 'scratch':
    model, model_args = initialize_model(args.init_from, args.out_dir, args.device, model_args)

if args.init_from == 'resume':
    model, model_args, best_val_loss, iter_num = initialize_model(args.init_from, args.out_dir, args.device, model_args)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

local_iter_num = 0  # number of iterations in the lifetime of this process

raw_model = model.module if ddp else model  # unwrap DDP container if needed

# logging
if args.wandb_log and master_process:
    import wandb

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

data_dir = os.path.join(f"{os.environ.get('HOME')}/Code/juan-garassino/mySandbox/nanoGPT/nanoGPT/data", args.dataset)

train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")

val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

X, Y = get_batch(train_data, val_data, 'train', args.batch_size, args.block_size, device_type, args.device)

running_mfu = -1.0

t0 = time.time()

#### TRAINING LOOP

print('loop')

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if args.decay_lr else args.learning_rate

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % args.eval_interval == 0 and master_process:
        losses = estimate_loss(model, train_data, val_data, args.eval_iters, args.batch_size, args.block_size, device_type, args.device, ctx)

        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        if args.wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )

        if losses["val"] < best_val_loss or args.always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {args.out_dir}")
                torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))

    if iter_num == 0 and args.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(args.gradient_accumulation_steps):

        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == args.gradient_accumulation_steps - 1
            )

        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / args.gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU

        X, Y = get_batch(train_data, val_data, 'train', args.batch_size, args.block_size, device_type, args.device)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % args.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * args.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )

    iter_num += 1

    local_iter_num += 1

    # Access the weights from the model

    weights = model.state_dict()

    # Plot the heatmaps
    plot_heatmaps(weights, "output_folder")

    # termination conditions
    if iter_num > args.max_iters:
        break

if ddp:
    destroy_process_group()
