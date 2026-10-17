# GUARDIAN

Automated security & bug guardian ledger for this repo. Branch: `claude/guardian`.

## Checks

| Area | Command |
|---|---|
| Dependencies (unpinned) | `pip-audit -r requirements.txt` |
| Dependencies (lockfile) | `uv.lock` exported to `name==version` list, then `pip-audit --no-deps --disable-pip -r <list>` |
| Static analysis | `bandit -r nanoGPT samples main.py config -q` |
| Pattern grep | `grep -rnE 'torch\.load\|pickle\.load\|yaml\.load\|eval\(\|exec\(\|shell=True\|os\.system\|verify=False\|requests\.' --include=*.py .` |
| Secrets (history) | `gitleaks detect --source . --redact -v` |
| Secrets (worktree) | `gitleaks detect --no-git --source . --redact` |
| Tracked env/log files | `git ls-files \| grep -iE 'env\|log\|egg-info'` |
| CI / containers | none present (no `.github/workflows`, no Dockerfile) |
| Tests | `python -m pytest -q tests` (CPU torch venv) |
| Train smoke | tiny random `train.bin`/`val.bin`, `python main.py --data_dir <tmp> --dataset tiny --n_layer 1 ... --max_iters 3`, then `--init_from resume` |

## Findings ledger

| ID | Severity | Type | File | Status | Date |
|---|---|---|---|---|---|
| G-001 | High | Bug: resume unpacked `best_val_loss`/`iter_num` swapped | main.py | fixed | 2026-10-15 |
| G-002 | High | Bug: `--decay_lr` called `get_lr(iter_num)` with 1 of 5 args (TypeError) | main.py | fixed | 2026-10-15 |
| G-003 | Medium | `exec()` of a hardcoded `$HOME/...` configurator path (crash for every other user; arbitrary code exec surface) | main.py | fixed | 2026-10-15 |
| G-004 | Medium | Bug: hardcoded `$HOME/...` data dir; `--init_from gpt2*` left `model` undefined; model never moved to `--device`; `out_dir` not created before save | main.py | fixed | 2026-10-15 |
| G-005 | Medium | `torch.load` without `weights_only=True` on checkpoints (pickle RCE on older torch) | init_from.py, trainer.py, samples/sample.py | fixed | 2026-10-15 |
| G-006 | Medium | `requests.get` without timeout (3 sites) | data/*/prepare.py, sourcing.py | fixed | 2026-10-15 |
| G-007 | Medium | Vulnerable transitive dep multidict 6.8.0 (PYSEC-2026-4244) in uv.lock | uv.lock | fixed (6.9.1) | 2026-10-15 |
| G-008 | Low | Local env file, logs and egg-info build metadata tracked in git (no credentials found) | repo root | fixed (untracked + .gitignore) | 2026-10-15 |
| G-009 | Medium | `exec()` config-file idiom (upstream nanoGPT design; local CLI only) | configurator.py, trainer.py, samples/*.py | wontfix (by design) | 2026-10-15 |
| G-010 | Medium | `load_dataset` without revision pin (B615) | data/openwebtext/prepare.py | open | 2026-10-15 |
| G-011 | Low | `pickle.load` of local `meta.pkl`; `assert` for validation (B101) | trainer.py, sample.py, model.py | open (accepted) | 2026-10-15 |
| G-012 | Low | `plot_heatmaps` writes to a hardcoded `$HOME/...` path every iteration (slow, outside repo) | heatmaps.py, main.py | fixed (repo `plots/`, only at eval_interval on master) | 2026-10-17 |
| G-013 | High | Bug: `plot_heatmaps` grid indexed past `num_heads` columns: IndexError on the first training iteration for any model | heatmaps.py | fixed | 2026-10-17 |
| G-014 | Medium | Bug: resume never restored optimizer state (AdamW moments) from ckpt.pt | main.py, init_from.py | fixed | 2026-10-17 |
| G-015 | High | Bug: DDP runs never wrapped the model in `DDP` (`model.module` AttributeError) and ignored the per-rank `cuda:N` device | main.py | fixed | 2026-10-17 |
| G-016 | Low | Bug: `sourcing_data(folder)` concatenated files in `os.listdir` order (filesystem-dependent train/val split) with locale-default encoding | sourcing.py | fixed (sorted, utf-8) | 2026-10-17 |
| G-017 | Low | Bug: `main.py` never reads `meta.pkl`, so char-level datasets train with vocab 50304; `--compile` flag is parsed but unused | main.py | open | 2026-10-17 |
| G-018 | Medium | `from_pretrained` without revision pin (B615) | model.py | open | 2026-10-17 |

## Metrics

| Metric | Before | After |
|---|---|---|
| Open vulns crit/high/med/low (bandit + pip-audit) | run #1: 0/0/16/21 -> 0/0/8/21; run #2: 0/0/8/21 -> 0/0/8/21 (pip-audit requirements + uv.lock: 0) |
| Known bugs (high) | run #1: 2 -> 0; run #2: 2 -> 0 |
| Failing tests | run #2: 0 (new repro tests failed before the fixes) | 0 |
| Test count | run #1: 0 -> 6; run #2: 6 | 10 |
| Train smoke (3 iters + resume, CPU) | crashed at iter 0 (heatmaps IndexError) | passes |
| Lint (black) | not run in CI | not changed |
| Secrets found (gitleaks history + worktree) | 0 | 0 |

## Backlog

- Pin `datasets.load_dataset` revision in openwebtext prepare (G-010).
- Read `meta_vocab_size` from `<data_dir>/<dataset>/meta.pkl` in main.py and either wire up or drop `--compile` (G-017).
- Pin `from_pretrained` revisions (G-018).
- Replace `exec()` configurator with argparse/config module (G-009) if upstream parity is not required.
- Pin dependency lower bounds in requirements.txt (e.g. `torch>=2.6`).

## Learnings

- torch >= 2.6 already defaults `weights_only=True`; explicit flag protects users on older torch since requirements are unpinned.
- `download.pytorch.org` is blocked by the sandbox proxy; CPU tests used the PyPI torch wheel.
- `main.py` is a script; a tiny random-token `train.bin` + `--max_iters 3` run is a fast (~10s CPU) end-to-end smoke check that caught a crash unit tests missed.
- `uv lock --upgrade-package` with a newer uv rewrites the whole lockfile (revision 1 -> 3); splice the single package entry to keep diffs small.

## Ideas

1. Add a tiny CI workflow (pinned action SHAs, `permissions: contents: read`) running `pytest` on CPU.
2. Turn `main.py` into an importable `train(args)` function so it can be smoke-tested end to end.
3. Add a char-level smoke dataset fixture for a 2-iteration training test.
4. Make `plot_heatmaps` opt-in via a `--heatmaps` flag and skip 1-D (LayerNorm) weights.
5. Save checkpoints atomically (write `ckpt.pt.tmp` then `os.replace`) so an interrupted save cannot corrupt resume.

## Log

- 2026-10-15 run #1: first guardian run. Fixed 2 high bugs in main.py (resume ordering, get_lr call), removed hardcoded exec/paths, added weights_only + request timeouts, bumped multidict in uv.lock, untracked env/log/egg-info files, added 6 tests. No secrets found.
- 2026-10-17 run #2: no PR review feedback, no CI configured. Scans clean (pip-audit 0 on requirements + uv.lock, no secrets, bandit unchanged). Fixed 4 bugs with failing-first tests: heatmap grid IndexError that crashed training at iter 0 (G-013), missing DDP wrap/device (G-015), optimizer state not restored on resume (G-014), nondeterministic folder sourcing (G-016); heatmaps now repo-relative and only at eval_interval (G-012). Tests 6 -> 10; CPU train+resume smoke passes.
