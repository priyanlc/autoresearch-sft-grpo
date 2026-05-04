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

mkdir -p data
# Copy train.csv from wherever you keep it (Kaggle download, S3, scp, etc.)
cp /workspace/train.csv data/
```

## 2. Create venv and install Python deps

> Always use a venv — no `pip install` directly into the system Python.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

`main`'s `requirements.txt` contains two source-built CUDA packages (`mamba_ssm` and `causal_conv1d`). They `import torch` in their `setup.py`, so under PEP 517 build isolation pip re-downloads torch into a throwaway temp venv just to compile — the install becomes both very slow (PyPI throughput on RunPod EU pods is ~50 KB/s vs ~36 MB/s on the PyTorch CDN) and very fragile ("ninja not found" or similar). Use this staged sequence instead of a bare `pip install -r requirements.txt`:

```bash
# Step 1 — torch first, from the PyTorch CDN
pip install 'torch>=2.2.0' --index-url https://download.pytorch.org/whl/cu121

# Step 2 — build helpers (ninja in particular — without it the source
# builds in step 3 fall back to slow serial nvcc compilation or fail outright)
pip install ninja packaging wheel setuptools

# Step 3 — source-built CUDA packages, in one no-isolation call
pip install mamba_ssm causal_conv1d --no-build-isolation

# Step 4 — the rest of requirements.txt (transformers, peft, trl, etc.)
pip install -r requirements.txt
```

For a richer worked example of the staged-install pattern (including failure modes captured live), see the `nvfp4-blackwell` branch's setup doc: `git show nvfp4-blackwell:runpod-setup.md` § Part 1 § 2, and `git show nvfp4-blackwell:build.log` for timing data.

> **build.log:** On the first fresh-pod bootstrap of `main`, capture install attempts (including failures), throughput samples, and lessons in a new `build.log` per the methodology in [`nemotron-vault/wiki/04-autoresearch-methodology.md`](../../../nemotron-vault/wiki/04-autoresearch-methodology.md) § build.log. Skipped pre-emptively per the assimilation plan; written when actually needed.

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

Run all three checks; abort if any flags an issue.

```bash
# CUDA version (must be >= 12.1)
nvidia-smi | grep "CUDA Version"

# BF16 support
python -c "import torch; print('BF16 supported' if torch.cuda.is_bf16_supported() else 'BF16 NOT SUPPORTED — investigate')"

# Full dependency + GPU + data check
python check_install.py
```

Decision tree for the BF16 check:

| Output | Action |
|---|---|
| `BF16 supported` | OK — proceed |
| `BF16 NOT SUPPORTED — investigate` | The GPU doesn't support BF16 (rare on A100/H100; fail fast). STOP, add a FRICTION entry, and either pick a different pod or a different dtype |
| `NEITHER` (anything else, e.g. CUDA not initialised) | The driver/CUDA install is broken. STOP and add a FRICTION entry rather than guessing |

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

## 7. (Optional) Adapter-on-BF16 sanity check

The submitted adapter gets loaded onto a BF16 base at scoring time, which is the same base used during training on `main`. Verify the load path works at least once:

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

## Initialise git for experiment tracking

```bash
cd /path/to/autoresearch-sft-grpo
git init
git add -A
git commit -m "initial baseline (post-T0 setup)"
```

The autoresearch loop uses `git rev-parse --short HEAD` for the `commit` column in `results.tsv`.

## Run Claude Code

Run **as `claude-runner`** (not root). The skip-permissions mode requires this — see the constraint note below.

```bash
# Standard — prompts for permission on each tool call
claude

# Skip prompts — only on disposable pods, only as a non-root user
claude --dangerously-skip-permissions
```

Then prime Claude:

> "Read program.md and start optimizing. Run `python train.py`, check the METRIC, modify `train.py` to improve it, and repeat. Honour the Tier 1 → Tier 2 sequencing."

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
rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA_hyphen_Nemotron*
```

This is also the path to clear when re-applying the in-place edits to `modeling_nemotron_h.py` for the F-001 cache fix — see `FRICTION.md` F-001.

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
