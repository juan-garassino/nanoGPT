"""Regression tests added by the guardian agent (CPU-only, no network)."""
import ast
import os
import pickle

import pytest

torch = pytest.importorskip("torch")

from nanoGPT.model.init_from import initialize_model
from nanoGPT.model.model import GPT, GPTConfig
from nanoGPT.model.tools import get_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = dict(n_layer=1, n_head=2, n_embd=8, block_size=8, bias=False, vocab_size=None, dropout=0.0)


def test_get_lr_schedule():
    kw = dict(warmup_iters=10, learning_rate=1.0, lr_decay_iters=100, min_lr=0.1)
    assert get_lr(0, **kw) == 0.0
    assert get_lr(5, **kw) == pytest.approx(0.5)
    assert get_lr(10, **kw) == pytest.approx(1.0)
    assert get_lr(1000, **kw) == pytest.approx(0.1)


def test_tiny_model_forward_and_generate():
    model, _ = initialize_model("scratch", "unused", "cpu", dict(TINY), meta_vocab_size=11)
    x = torch.randint(0, 11, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 11) and torch.isfinite(loss)
    out = model.generate(x[:, :2], max_new_tokens=3)
    assert out.shape == (2, 5)


def _save_ckpt(tmp_path, iter_num=7, best_val_loss=1.25):
    args = dict(TINY, vocab_size=11)
    model = GPT(GPTConfig(**args))
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "model_args": args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": {"out_dir": str(tmp_path), "batch_size": 4, "decay_lr": False},
    }
    torch.save(ckpt, os.path.join(tmp_path, "ckpt.pt"))


def test_resume_returns_iter_then_best_loss(tmp_path):
    _save_ckpt(tmp_path)
    model, args, iter_num, best_val_loss = initialize_model("resume", str(tmp_path), "cpu", dict(TINY))
    assert iter_num == 7 and best_val_loss == pytest.approx(1.25)
    assert args["vocab_size"] == 11


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_resume_rejects_pickled_code(tmp_path):
    # torch.load must use weights_only=True so a crafted ckpt.pt cannot run code
    torch.save({"model_args": _Evil()}, os.path.join(tmp_path, "ckpt.pt"))
    with pytest.raises(pickle.UnpicklingError):
        initialize_model("resume", str(tmp_path), "cpu", dict(TINY))


def _main_tree():
    with open(os.path.join(ROOT, "main.py")) as f:
        return ast.parse(f.read())


def test_main_has_no_exec_and_calls_get_lr_with_schedule_args():
    calls = [n for n in ast.walk(_main_tree()) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert not [c for c in calls if c.func.id == "exec"]
    lr_calls = [c for c in calls if c.func.id == "get_lr"]
    assert lr_calls and all(len(c.args) == 5 for c in lr_calls)


def test_main_unpacks_resume_in_return_order():
    for node in ast.walk(_main_tree()):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, "id", None) == "initialize_model" \
                and isinstance(node.targets[0], ast.Tuple) and len(node.targets[0].elts) == 4:
            names = [e.id for e in node.targets[0].elts]
            assert names == ["model", "model_args", "iter_num", "best_val_loss"]
            return
    pytest.fail("resume branch of initialize_model not found in main.py")
