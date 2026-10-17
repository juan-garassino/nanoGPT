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


def test_restore_optimizer_loads_checkpoint_state(tmp_path):
    # run #2: resume used to drop the AdamW moments saved in ckpt.pt
    from nanoGPT.model.init_from import restore_optimizer

    args = dict(TINY, vocab_size=11)
    model = GPT(GPTConfig(**args))
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    x = torch.randint(0, 11, (2, 8))
    model(x, x)[1].backward()
    opt.step()
    torch.save({"optimizer": opt.state_dict()}, os.path.join(tmp_path, "ckpt.pt"))

    fresh = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    assert not fresh.state_dict()["state"]
    restore_optimizer(fresh, str(tmp_path), "cpu")
    assert fresh.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_main_restores_optimizer_and_wraps_ddp():
    calls = [n for n in ast.walk(_main_tree()) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    names = {c.func.id for c in calls}
    assert "restore_optimizer" in names  # resume keeps optimizer state
    assert "DDP" in names  # raw_model = model.module needs a DDP wrapper


def test_sourcing_folder_order_is_deterministic(tmp_path, monkeypatch):
    # os.listdir order is filesystem-dependent; the 90/10 split must not depend on it
    from nanoGPT.sourcing import sourcing

    for name, text in [("a.txt", "AAA"), ("b.txt", "ñBB"), ("c.txt", "CCC")]:
        (tmp_path / name).write_text(text, encoding="utf-8")
    real = os.listdir
    monkeypatch.setattr(sourcing.os, "listdir", lambda p: list(reversed(sorted(real(p)))))
    assert sourcing.sourcing_data(str(tmp_path)) == "AAAñBBCCC"


def test_plot_heatmaps_handles_real_state_dict(tmp_path):
    # run #2: the grid math indexed past num_heads columns (IndexError on any real model)
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from nanoGPT.heatmaps.heatmaps import plot_heatmaps

    for n_layer in (1, 2, 6):
        model = GPT(GPTConfig(**dict(TINY, n_layer=n_layer, vocab_size=11)))
        out = tmp_path / f"L{n_layer}"
        plot_heatmaps(model.state_dict(), str(out), figsize=(4, 4), dpi=20)
        assert len(list(out.glob("heatmaps*.png"))) == 1


def test_sample_script_runs_from_resume_checkpoint(tmp_path):
    # run #2: samples/sample.py had a broken import, a cwd-relative configurator path and
    # ignored the checkpoint's data_dir when looking for meta.pkl
    import subprocess
    import sys

    _save_ckpt(tmp_path)
    ckpt_path = os.path.join(tmp_path, "ckpt.pt")
    ckpt = torch.load(ckpt_path, weights_only=True)
    ckpt["config"].update(dataset="tiny", data_dir=str(tmp_path))
    torch.save(ckpt, ckpt_path)
    os.makedirs(tmp_path / "tiny")
    chars = "abcdefghijk"
    with open(tmp_path / "tiny" / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 11, "stoi": {c: i for i, c in enumerate(chars)},
                     "itos": dict(enumerate(chars))}, f)
    res = subprocess.run(
        [sys.executable, os.path.join(ROOT, "samples", "sample.py"), "--device=cpu", "--dtype=float32",
         f"--out_dir={tmp_path}", "--num_samples=1", "--max_new_tokens=3", "--start=ab"],
        cwd=str(tmp_path), env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    assert "Loading meta" in res.stdout


@pytest.mark.parametrize("rel", ["samples/bench.py", "samples/sample.py", "nanoGPT/trainer/trainer.py"])
def test_scripts_import_package_and_resolve_configurator(rel):
    # `from model.model import ...` and exec(open("configurator.py")) only worked from one cwd
    with open(os.path.join(ROOT, rel)) as f:
        src = f.read()
    assert "from model.model import" not in src
    assert 'open("configurator.py")' not in src


def test_read_meta_vocab_size(tmp_path):
    # G-017: char-level datasets must train with the meta.pkl vocab, not 50304
    from nanoGPT.model.init_from import read_meta_vocab_size

    assert read_meta_vocab_size(str(tmp_path)) is None
    with open(tmp_path / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 65, "stoi": {}, "itos": {}}, f)
    assert read_meta_vocab_size(str(tmp_path)) == 65


def test_main_uses_meta_vocab_and_compile_flag():
    tree = _main_tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    init_calls = [c for c in calls if getattr(c.func, "id", None) == "initialize_model"]
    assert any(len(c.args) == 5 or any(k.arg == "meta_vocab_size" for k in c.keywords) for c in init_calls)
    assert any(getattr(c.func, "attr", None) == "compile" and getattr(c.func.value, "id", None) == "torch"
               for c in calls)
    compile_opt = [c for c in calls if c.args and getattr(c.args[0], "value", None) == "--compile"]
    assert compile_opt and any(k.arg == "action" and k.value.value == "store_true" for k in compile_opt[0].keywords)
