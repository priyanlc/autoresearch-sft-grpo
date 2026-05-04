# RunPod Setup — `autoresearch-sft-grpo` on `main`

This doc is split into three parts. Most users only need **Part 1** (minimal first-run). Parts 2 and 3 are for handing the pod over to autonomous Claude Code or for resolving a setup issue.

| Part | When to use |
|---|---|
| 1. Minimal first-run | Always — gets you from a fresh pod to the first `METRIC:` line |
| 2. Autoresearch operator setup | Only when you'll leave Claude Code running unattended |
| 3. Troubleshooting & appendix | When something fails (cache poisoning, etc.) |

---

## Pod requirements (pick before launching)

| Requirement | Value | Why |
|---|---|---|
| GPU generation | **A100 80GB** (or any BF16-capable accelerator with ≥80 GB) | Full BF16 30B base + LoRA + activations + optimizer state |
| VRAM | ≥ 80 GB | Peak ~78 GB observed during eval (`BRANCH_NOTES.md`) |
| CUDA | ≥ 12.1 | Standard `transformers==4.51.3` / `peft` / `trl` support |
| Disk | ≥ 100 GB | ~60 GB BF16 weights + pip cache + adapter output |

If the pod doesn't meet all four, stop and pick a different template — none of the workarounds in `train.py` rescue a hardware mismatch. The NVFP4 path lives on `nvfp4-blackwell`, not here.

---

# Part 1 — Minimal first-run

Goal: get from a fresh pod to a printed `METRIC: 0.XXXX` line. Seven steps.

## 1. Get code + data on the pod

```bash
git clone <your-fork-url> nemotron
cd nemotron/notebooks/05-autoresearch/autoresearch-sft-grpo
git checkout main

# Confirm data ships with the repo (CC BY 4.0; see data/README.md)
ls -la data/train.csv data/test.csv
```

The `train.csv` (9,500 puzzles) and `test.csv` (3-row preview) are tracked in the repo and arrive with the clone — no separate download needed. License and provenance are documented in [`data/README.md`](data/README.md). If those files are missing for any reason (unusual fork state, sparse checkout), pull them from the [Kaggle competition page](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) and copy into `data/`.

## 2. Create venv and install Python deps

> Always use a venv — no `pip install` directly into the system Python.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

`main`'s `requirements.txt` contains one source-built CUDA package (`mamba_ssm`). It `import torch`s in its `setup.py`, so under PEP 517 build isolation pip re-downloads torch into a throwaway temp venv just to compile — the install becomes both very slow (PyPI throughput on RunPod EU pods is ~50 KB/s vs ~36 MB/s on the PyTorch CDN) and very fragile ("ninja not found" or similar). Use this staged sequence instead of a bare `pip install -r requirements.txt`:

```bash
# Step 1 — torch first, from the PyTorch CDN
pip install 'torch>=2.2.0' --index-url https://download.pytorch.org/whl/cu121

# Step 2 — build helpers (ninja in particular — without it the source
# build in step 3 falls back to slow serial nvcc compilation or fails outright)
pip install ninja packaging wheel setuptools

# Step 3 — source-built CUDA package, --no-build-isolation
pip install mamba_ssm --no-build-isolation

# Step 4 — the rest of requirements.txt (transformers, peft, trl, etc.)
pip install -r requirements.txt
```

`causal_conv1d` is intentionally absent from this list — it is commented out in `requirements.txt` per T1.9 because the F-001 workaround in `train.py:386` force-disables the Mamba fast path that would otherwise consume it. With `causal_conv1d` not installed, `is_fast_path_available` evaluates to `False` automatically (same outcome as the train.py toggle, no runtime difference). When F-001 resolves upstream and the fast path is reactivated, `causal_conv1d` should be added back to `requirements.txt` and to the Step 3 install. See [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) for the full mechanical treatment.

For a richer worked example of the staged-install pattern (including failure modes captured live), see the `nvfp4-blackwell` branch's setup doc: `git show nvfp4-blackwell:runpod-setup.md` § Part 1 § 2, and `git show nvfp4-blackwell:build.log` for timing data.

> **build.log:** On the first fresh-pod bootstrap of `main`, populate `./build.log` with install attempts (including failures), throughput samples, and lessons learned. The methodology in [`docs/methodology.md`](docs/methodology.md) § build.log explains the pattern. The file is intentionally absent from the repo until then — write it when first needed, not pre-emptively.

## 3. Authenticate to Hugging Face and W&B

```bash
hf auth login --token "$HF_TOKEN"   # token saved to ~/.cache/huggingface/token
wandb login "$WANDB_API_KEY"        # key saved to ~/.netrc
```

If you don't have `$HF_TOKEN` / `$WANDB_API_KEY` exported in the pod's env yet, paste the actual token in place of the variable. `hf auth login` (no `--token`) also supports interactive paste.

If you don't want W&B for the first run:
```bash
export WANDB_MODE=disabled
```

## 4. Pre-flight verification

Four checks; abort if any flags an issue. C-1 is a smoke check (trivially passes on A100/H100); C-2/C-3/C-4 are load-bearing. This mirrors `program.md` § Pre-flight verification — the autoresearch agent runs the same C-1/C-2/C-3 set before any Tier 2 sweep, so passing here means passing there.

```bash
# (C-1) CUDA version + BF16 support (smoke check)
nvidia-smi | grep "CUDA Version"
python -c "import torch; print('OK' if torch.cuda.is_bf16_supported() else 'BF16 NOT SUPPORTED')"

# (C-2) transformers pin matches check_install.py expectation (4.51.3)
python -c "import transformers; print(transformers.__version__)"

# (C-3) Nemotron transformers_modules cache (touches F-001 surface)
ls ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/ 2>/dev/null && echo "cache present" || echo "cache missing"

# (C-4) Full dependency + GPU + data check
python check_install.py
```

**Decision tree:**

| Check | Output | Action |
|---|---|---|
| C-1 | CUDA ≥ 12.1 + `OK` | Proceed |
| C-1 | `BF16 NOT SUPPORTED` | The GPU lacks BF16 (rare on A100/H100). STOP, add a FRICTION entry — switch hardware or branch off; do not switch dtype on `main` |
| C-1 | anything else (CUDA init error, etc.) | STOP and add a FRICTION entry rather than guessing |
| C-2 | `4.51.3` | Matches the pin and `check_install.py:83`. Proceed |
| C-2 | any other version | The pin is being violated. Reinstall via `pip install -r requirements.txt`; do **not** rename or relax the kwarg in `train.py` |
| C-3 | `cache present` followed by directory listing | OK — model has been downloaded before. F-001's in-place `modeling_nemotron_h.py` edits (if applied this pod-life) should still be in place |
| C-3 | `cache missing` | First run on this pod — Step 5 (`prepare.py`) will download. Not an error |
| C-4 | `All checks passed` | Proceed |
| C-4 | any failure | Re-read the failure line; fix and re-run before Step 5 |

If any check hits a "neither expected outcome" branch, **STOP and add a FRICTION entry** rather than guessing — the verification command itself can become stale (this is the lesson behind the original F-001 pattern on `nvfp4-blackwell`).

## 5. Prepare data and validation split

```bash
python prepare.py   # downloads the model + creates data/val_split.json
```

This downloads ~60 GB of BF16 weights to `~/.cache/huggingface/`. First time only.

## 6. First smoke run (SFT-only with graceful GRPO fallback)

The active mode on `main` is **SFT-only**. `train.py` always attempts GRPO inside a `try/except` block (around `train.py:511`) — when GRPO crashes with the known Mamba/MoE+TRL tensor mismatch (FRICTION F-002), the SFT adapter from Phase 1 is preserved and eval proceeds. This is by design, not a regression.

```bash
python train.py
```

Watch for `METRIC: 0.XXXX` at the end. The pre-T1 baseline is **0.5333** at commit `c1bb0a6`. Capture your run's METRIC in `STATUS.md` before applying any Tier 2 changes (per `program.md`).

## 7. Adapter-on-BF16 sanity check (required at every Tier transition)

The submitted adapter gets loaded onto a BF16 base at scoring time, which is the same base used during training on `main`. The methodology's Validation Contract (`program.md` § Validation Contract, point 5) requires this check at every Tier transition — it is the most likely silent-break point in the deployment path. Run it at least once per fresh pod bootstrap, plus whenever a Tier 1 → Tier 2 transition lands:

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prepare import MODEL_ID
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(base, './adapter')
out = model.generate(**tok('Test prompt: 2+2 =', return_tensors='pt').to('cuda'), max_new_tokens=20)
print(tok.decode(out[0]))
"
```

If this fails, the adapter is unusable for submission regardless of training METRIC. Log to `FRICTION.md`.

---

# Part 2 — Autoresearch operator setup

**Required when you'll use `claude --dangerously-skip-permissions` for autonomous iteration.** Claude Code refuses to start with `--dangerously-skip-permissions` if invoked as root or under sudo (verified 2026-05-02; error: `--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons`). Since RunPod pods boot as root, the non-root user isn't optional for autonomous mode — it's a precondition. (Also limits blast radius if Claude issues a destructive command, but that's the secondary benefit.)

## Create a non-root user

RunPod pods run as root. Create a dedicated user before handing over to autonomous Claude Code.

```bash
# As root:
useradd -m -s /bin/bash claude-runner
usermod -aG video claude-runner
usermod -aG render claude-runner

# Move ownership of the project + venv + HF cache
chown -R claude-runner:claude-runner /path/to/nemotron
chown -R claude-runner:claude-runner /home/claude-runner/.cache

# Switch
su - claude-runner
```

> Do all `pip install`, `apt-get`, and `chown` commands as root **before** switching to `claude-runner`. The non-root user does not get sudo.

## Install Claude Code

```bash
# Node.js (skip if already system-wide)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
sudo apt-get install -y nodejs

npm install -g @anthropic-ai/claude-code
claude login
```

## Confirm git state for experiment tracking

After `git clone` in Part 1 § 1, the repo already has full history — no `git init` needed. Confirm the working tree is on `main` and clean before kicking off:

```bash
cd /path/to/autoresearch-sft-grpo
git rev-parse --abbrev-ref HEAD   # expect: main
git status --porcelain            # expect: empty
git log --oneline -1              # capture the starting commit hash for the first results.tsv row
```

The autoresearch loop uses `git rev-parse --short HEAD` for the `commit` column in `results.tsv` — this is why the working tree must be clean before each `python train.py` invocation, and why each experiment lands as its own commit before the next run.

## Run Claude Code

Run **as `claude-runner`** (not root). The skip-permissions mode requires this — see the constraint note below.

```bash
# Standard — prompts for permission on each tool call
claude

# Skip prompts — only on disposable pods, only as a non-root user
claude --dangerously-skip-permissions
```

Then prime Claude:

> "Read `program.md`, `BRANCH_NOTES.md`, and `FRICTION.md` first. Then run `python train.py` once *unmodified* to land **T1.8b** — that captures the post-T1 regression baseline against the pre-T1 0.5333 number, appends a row to `results.tsv`, and prepends a heartbeat to `STATUS.md`. After T1.8b lands, begin Tier 2 sweeps: pick one axis from `program.md` § 'Tier 2 sweep targets', edit `train.py`, commit with a `T2.x:` prefix, run, append `results.tsv`, repeat. Honour the Tier 1 → Tier 2 sequencing — do not start Tier 2 until T1.8b is committed."

> **Warning:** `--dangerously-skip-permissions` lets Claude run any command without confirmation. Use only on disposable RunPod instances with no sensitive data.

> **Hard constraint (verified 2026-05-02):** Claude Code refuses to start with `--dangerously-skip-permissions` when invoked as root or under sudo. Error: `--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons` ([GitHub issue #9184](https://github.com/anthropics/claude-code/issues/9184)). The non-root user above is a precondition for this mode, not just defense-in-depth. See "Advanced: skipping the non-root user" below for two unsupported workarounds on throwaway pods.

## Advanced: skipping the non-root user (throwaway pods only)

Two ways to bypass the user-creation step on a pod you intend to terminate after the run. Both are non-standard. Only use on a pod you'll throw away after the experiment — not on long-lived pods, not on anything sharing data with production.

### Approach 1: `IS_SANDBOX=1` env var (the hack)

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Undocumented bypass surfaced by [Levels.io](https://x.com/levelsio/status/1959012607270236619). Works in current versions but isn't in Anthropic's docs, `--help`, or release notes. Could be removed in any release without warning.

**Risk profile**: doesn't make Claude *more* destructive than running as root normally would; it just removes the safety rail that was added after a model executed `rm -rf /`. On a throwaway pod, worst case is the pod itself (~$0.50–2 + ~10 min restart).

### Approach 2: One-liner non-root user (the supported path)

If you don't want to depend on an undocumented env var:

```bash
# As root, all in one command (~30 seconds total):
useradd -m -s /bin/bash cr && \
  echo "export HF_TOKEN=$HF_TOKEN" >> /home/cr/.bashrc && \
  echo "export WANDB_API_KEY=$WANDB_API_KEY" >> /home/cr/.bashrc && \
  echo "export GIT_REPO_URL=$GIT_REPO_URL" >> /home/cr/.bashrc && \
  su - cr
```

Then continue with `npm install -g @anthropic-ai/claude-code`, `claude login`, the clone/venv/scp steps, and the kickoff. Survives Claude Code updates regardless of what they do to the `IS_SANDBOX` path.

### Which to use

| Scenario | Recommended |
|---|---|
| First kickoff on a new branch, or any unfamiliar workflow | **Approach 2** — 30 seconds isn't worth the dependency on an undocumented mechanism while you're still validating the workflow |
| Repeated launches of a known-good kickoff on disposable pods | **Approach 1** is reasonable — you've calibrated trust in the prompt and model behaviour on this branch, and the time saved compounds across launches |
| Any pod that isn't immediately disposable | Use the full Part 2 flow above (proper non-root user) — neither approach belongs on a long-lived pod |

---

# Part 3 — Troubleshooting & appendix

## Stale `transformers_modules` cache after a version bump

If you upgrade `transformers` and start seeing weird "module has no attribute" errors:

```bash
# First inspect what's actually in the cache so you don't run a no-op rm:
ls ~/.cache/huggingface/modules/transformers_modules/nvidia/

# Then delete the Nemotron entry. On modern transformers the directory uses
# literal hyphens; older versions sometimes used a "_hyphen_" encoding. Use
# whichever pattern matches your `ls` output:
rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
# Or if your install uses the older encoding:
# rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA_hyphen_Nemotron*
```

This is also the path to clear when re-applying the in-place edits to `modeling_nemotron_h.py` for the F-001 cache fix — see `FRICTION.md` F-001 and [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md). Note: clearing the cache wipes those in-place edits; you'll need to re-apply them or re-derive them after the next `from_pretrained()`.

## Reset between experiments

```bash
rm -rf ./adapter            # delete the saved LoRA adapter
rm -rf ~/.cache/huggingface/datasets   # if datasets caching causes issues
```

## Useful commands

```bash
# Check which user owns what
ls -la /path/to/autoresearch-sft-grpo

# List real users (not system accounts)
cat /etc/passwd | grep -E '/bin/(bash|sh|zsh)'

# Watch GPU usage during training
watch -n 2 nvidia-smi

# Tail the metric line from a long-running train
python train.py 2>&1 | tee run.log | grep -E "METRIC:|loss|step"
```

## Pod recycling

RunPod pods are disposable. If anything goes irrecoverably wrong, the fastest path is usually `terminate the pod → spin up fresh → repeat Part 1`. The HF cache and pip wheels rebuild quickly; saving the broken state rarely pays off.
