# RunPod Setup — `autoresearch-sft-grpo` on `main`

[RunPod](https://www.runpod.io/) is an on-demand GPU cloud where you rent a "pod" (a containerised GPU instance) by the hour. This doc walks through bootstrapping `autoresearch-sft-grpo` on a fresh A100 80GB pod, but the same steps work on any cloud that gives you root SSH on a CUDA-12.1+ machine with ≥ 80 GB VRAM (Lambda Labs, Vast.ai, Modal, etc.).

The goal is six steps from a fresh pod to a printed `METRIC: 0.XXXX`. For autonomous Claude Code handoff (running unattended), see [`docs/autoresearch-handoff.md`](docs/autoresearch-handoff.md). For known failure modes, see [`FRICTION.md`](FRICTION.md).

## Pod requirements (pick before launching)

| Requirement | Value | Why |
|---|---|---|
| GPU generation | **A100 80GB** (or any BF16-capable accelerator with ≥80 GB) | Full BF16 30B base + LoRA + activations + optimizer state |
| VRAM | ≥ 80 GB | Peak ~78 GB observed during eval (`BRANCH_NOTES.md`) |
| CUDA | ≥ 12.1 | Standard `transformers==4.51.3` / `peft` / `trl` support |
| Disk | ≥ 100 GB | ~60 GB BF16 weights + pip cache + adapter output |

If the pod doesn't meet all four, stop and pick a different template — none of the workarounds in `train.py` rescue a hardware mismatch. The NVFP4 path lives on `nvfp4-blackwell`, not here.

## 1. Get code + data on the pod

```bash
git clone https://github.com/priyanlc/autoresearch-sft-grpo.git
cd autoresearch-sft-grpo
# main is the default branch; explicit checkout shown for autoresearch-agent setups
git checkout main

# Confirm data ships with the repo (CC BY 4.0; see data/README.md)
ls -la data/train.csv data/test.csv
```

> If you forked the repo, replace the URL with your fork. The repo is published standalone on GitHub — it is not nested inside another tree, so the post-clone working directory is flat `autoresearch-sft-grpo/`.

The `train.csv` (9,500 puzzles) and `test.csv` (3-row preview) are tracked in the repo and arrive with the clone — no separate download needed. License and provenance are documented in [`data/README.md`](data/README.md). If those files are missing for any reason (unusual fork state, sparse checkout), pull them from the [Kaggle competition page](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) and copy into `data/`.

## 2. Create venv and install Python deps

This repo uses [`uv`](https://docs.astral.sh/uv/) — a fast Rust-based Python package manager and venv tool from Astral. Single binary, `pip`-compatible interface, ~10–100× faster installs.

```bash
# Install uv (one-time, ~5 seconds)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or open a new shell

# Create + activate venv
uv venv
source .venv/bin/activate

# Install dependencies
bash bootstrap.sh                    # CUDA-built packages (torch, mamba_ssm, build helpers)
uv pip install -r requirements.txt   # pure-Python deps (transformers, peft, trl, etc.)
```

> Always work inside the venv — no `uv pip install` (or `pip install`) into the system Python.

`bootstrap.sh` handles the CUDA source-build dance (torch from CDN → ninja → `mamba_ssm --no-build-isolation`) that can't safely run from `requirements.txt` under PEP 517 build isolation. A bare `uv pip install -r requirements.txt` would re-download torch into a throwaway temp venv just to compile mamba_ssm — very slow on RunPod EU pods, often fails. See `bootstrap.sh`'s header for the per-step rationale.

`causal_conv1d` is required at install time (per T1.14, restoring the dep that T1.9 had removed). It must be installed even though `train.py:398` force-disables the Mamba fast path: transformers' dynamic-module loader does AST-level static import checking on `modeling_nemotron_h.py` and rejects the conditional `from causal_conv1d import ...` even though that import sits inside an `if is_causal_conv1d_available():` guard at runtime. T1.9's premise that the conditional import would handle absence was runtime-correct but static-analysis-incorrect; FRICTION F-009 documents the rediscovery. At runtime the dep is harmless — `causal_conv1d_fn` / `causal_conv1d_update` are loaded but never called because the fast path is still force-disabled. See [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) for the full mechanical treatment.

For a richer worked example of the staged-install pattern (including failure modes captured live), see the `nvfp4-blackwell` branch's setup doc: `git show nvfp4-blackwell:runpod-setup.md` § Part 1 § 2, and `git show nvfp4-blackwell:build.log` for timing data.

> **build.log:** On the first fresh-pod bootstrap of `main`, populate `./build.log` with install attempts (including failures), throughput samples, and lessons learned. The methodology in [`docs/methodology.md`](docs/methodology.md) § build.log explains the pattern. The file is intentionally absent from the repo until then — write it when first needed, not pre-emptively.

## 3. Authenticate to Hugging Face and W&B

```bash
# Export your tokens (paste the real values in place of <YOUR_KEY>)
export HF_TOKEN=<YOUR_KEY>
export WANDB_API_KEY=<YOUR_KEY>

hf auth login --token "$HF_TOKEN"   # token saved to ~/.cache/huggingface/token
wandb login "$WANDB_API_KEY"        # key saved to ~/.netrc
```

`hf auth login` without `--token` also supports interactive paste if you'd rather not put tokens in your shell history.

If you don't want W&B for the first run:
```bash
export WANDB_MODE=disabled
```

## 4. Pre-flight verification

```bash
python check_install.py
```

Should print `All checks passed`. If anything fails, fix it before Step 5 — re-read the failure line; the script tells you which dep / file / GPU is missing. For the deeper pre-flight that the autoresearch agent runs before each Tier 2 sweep (transformers pin, modeling-cache state, BF16 support), see [`program.md`](program.md) § Pre-flight verification.

## 5. Prepare data and validation split

```bash
python prepare.py   # downloads the model + creates data/val_split.json
```

This downloads ~60 GB of BF16 weights to `~/.cache/huggingface/`. First time only.

## 6. Run training

```bash
python train.py
```

Watch for `METRIC: 0.XXXX` at the end. The locked floor is **0.5333** (commit `c1bb0a6`); current best on `main` is **0.6000** (T2.8, `c4a9d1c`) — see [`BRANCH_NOTES.md`](BRANCH_NOTES.md) for the full configuration and [`STATUS.md`](STATUS.md) / [`results.tsv`](results.tsv) for the latest run. GRPO is wrapped in a `try/except` that falls back to SFT-only on the known Mamba/MoE+TRL tensor mismatch — that's expected, not a regression (see [`FRICTION.md`](FRICTION.md) F-002).

The `program.md` Validation Contract requires an adapter-on-fresh-BF16-base sanity check at every Tier transition — see [`program.md`](program.md) § Validation Contract, point 5, for the snippet.

## Next steps

- **Iterate on the loop manually:** read [`program.md`](program.md), edit `train.py`, run, commit, repeat. The methodology is in [`docs/methodology.md`](docs/methodology.md).
- **Hand off to autonomous Claude Code:** see [`docs/autoresearch-handoff.md`](docs/autoresearch-handoff.md) for non-root user setup, the kickoff prompt, and the `--dangerously-skip-permissions` constraints.
- **Hit a known failure mode:** check [`FRICTION.md`](FRICTION.md) first — the cache poisoning, GRPO crashes, 4-bit/USE_COT pitfalls, and other recurring issues are documented as F-001..F-006 with their workarounds.
