import argparse
import torch

from nanoGPT.sourcing.sourcing import sourcing_data, get_batch

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
    parser.add_argument("--dataset", type=str, default="openwebtext", help="Dataset name")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5 * 8, help="Number of gradient accumulation steps")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size")

    # Model
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
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
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float16", help="Data type")
    parser.add_argument("--compile", action="store_true", help="Compile the model using PyTorch 2.0")

    return parser.parse_args()

args = parse_args()

# Access the argument values
print(args.out_dir)
print(args.eval_interval)
# and so on...




#### TRAINING LOOP

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
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
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
