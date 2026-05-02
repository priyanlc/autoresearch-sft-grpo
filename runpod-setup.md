# RunPod Setup — `autoresearch-sft-grpo` on `nvfp4-blackwell`

This doc is split into three parts. Most users only need **Part 1** (minimal first-run). Parts 2 and 3 are for handing the pod over to autonomous Claude Code or for resolving a setup issue.

| Part | When to use |
|---|---|
| 1. Minimal first-run | Always — gets you from a fresh pod to the first `METRIC:` line |
| 2. Autoresearch operator setup | Only when you'll leave Claude Code running unattended |
| 3. Troubleshooting & appendix | When something fails or you hit the known `transformers` pin conflict |

---

## Pod requirements (pick before launching)

| Requirement | Value | Why |
|---|---|---|
| GPU generation | **Blackwell** (RTX PRO 6000, B200, GB10) | NVFP4 path silently falls back to BF16 on Hopper/Ampere; the 30B model won't fit |
| VRAM | ≥ 48 GB | NVFP4 base ≈ 17 GB + activations + LoRA + optimizer state |
| CUDA | ≥ 12.8 | `fp_quant` and `qutlass` kernels require this |
| Disk | ≥ 100 GB | Model weights ≈ 60 GB before quantization, plus pip cache |

If the pod doesn't meet all four, stop and pick a different template — none of the patches in `train.py` work around hardware mismatch.

---

# Part 1 — Minimal first-run

Goal: get from a fresh pod to a printed `METRIC: 0.XXXX` line. Seven steps, ~20–30 minutes the first time.

## 1. Get code + data on the pod

```bash
git clone <your-fork-url> nemotron
cd nemotron/notebooks/05-autoresearch/autoresearch-sft-grpo
git checkout nvfp4-blackwell

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

The remaining install must be **staged** — a single `pip install -r requirements.txt` will not succeed on a fresh Blackwell pod. Two reasons, both verified 2026-05-02:

1. `mamba_ssm` and `causal_conv1d` are source builds that `import torch` in their `setup.py`. Under PEP 517 build isolation (pip's default) they re-download torch into a throwaway temp venv just to compile, so the install is both very slow and very fragile. They need `--no-build-isolation` exactly like `fp_quant`/`qutlass`.
2. From plain PyPI on RunPod EU pods, the torch wheels download at ~50 KB/s (sustained ~30 min for 25 % of a single 530 MB wheel — see `build.log`). The PyTorch CDN is roughly 700× faster on the same network.

Use this sequence:

```bash
# Step 1 — torch first, from the PyTorch CDN's cu128 channel
#   The cu128 channel pins CUDA 12.8 user-space libs, which is what
#   fp_quant / qutlass kernels expect. Drivers ≥ 580 also support cu130
#   wheels, but cu128 is the stable channel as of 2026-05-02.
pip install 'torch>=2.7.0' --index-url https://download.pytorch.org/whl/cu128

# Step 2 — build helpers (ninja in particular; without it the source
#   builds in step 3 either fail with "ninja: command not found" or
#   fall back to slow serial nvcc compilation)
pip install ninja packaging wheel setuptools

# Step 3 — source-built CUDA packages, in one no-isolation call
#   TORCH_CUDA_ARCH_LIST forces the kernels to target sm_120 (Blackwell)
#   plus PTX for forward-compat. mamba_ssm and causal_conv1d compile
#   from source (~5–10 min each); fp_quant comes as a pre-built wheel.
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
pip install mamba_ssm causal_conv1d fp_quant qutlass --no-build-isolation

# Step 3b — replace the qutlass stub with the real QuTLASS
#   `qutlass` on PyPI is a placeholder package (version 0.0.0, summary
#   "Temp", empty __init__.py). The real CUTLASS-based NVFP4 kernels
#   are at github.com/IST-DASLab/QuTLASS. The PyPI stub satisfies
#   `import qutlass` but raises `ValueError("QuTLASS is not installed")`
#   from `fp_quant/module/linear.py` at the first FPQuantLinear init —
#   i.e., during model load in `train.py`, not during pip install.
pip uninstall qutlass -y
pip install 'git+https://github.com/IST-DASLab/QuTLASS.git' --no-build-isolation

# Step 4 — the rest of requirements.txt
#   Most of `transformers`, `huggingface_hub`, `tokenizers`, `scipy`,
#   `numpy`, `safetensors`, etc. are already satisfied as transitive deps
#   of mamba_ssm. Step 4 only newly installs accelerate, peft, trl,
#   datasets, sentencepiece, bitsandbytes, wandb, polars and their deps.
pip install -r requirements.txt
```

Total wall time on a fresh RunPod EU Blackwell pod: ~50 minutes. Step 3 is the longest phase (~25 min — mostly nvcc kernel compilation, not network). See `build.log` for a step-by-step timing breakdown if anything regresses.

## 3. Authenticate to Hugging Face and W&B

```bash
# transformers 5.x deprecates `huggingface-cli`; use `hf` instead
hf auth login --token "$HF_TOKEN"   # token saved to /workspace/.cache/huggingface/token
wandb login "$WANDB_API_KEY"        # key saved to /root/.netrc
```

If you don't have `$HF_TOKEN` / `$WANDB_API_KEY` exported in the pod's env yet, paste the actual token in place of the variable. `hf auth login` (no `--token`) also supports interactive paste.

If you don't want W&B for the first run:
```bash
export WANDB_MODE=disabled
```

## 4. Pre-flight verification

This catches the two most common silent-failure modes (wrong CUDA, wrong `transformers` API). Run all three checks; abort if any flags an issue.

```bash
# CUDA version (must be >= 12.8)
nvidia-smi | grep "CUDA Version"

# Transformers API check — confirms the dtype kwarg matches what train.py uses
python -c "import transformers, inspect; sig = inspect.signature(transformers.AutoModelForCausalLM.from_pretrained); params = list(sig.parameters.keys()); print(f'transformers={transformers.__version__}'); print('uses dtype' if 'dtype' in params else 'uses torch_dtype' if 'torch_dtype' in params else 'NEITHER — investigate')"

# Full dependency + GPU + data check
python check_install.py
```

Decision tree for the transformers check:

| Output | Action |
|---|---|
| `transformers=5.x.y` + `uses dtype` | OK — current `train.py:679` matches |
| `transformers=4.x.y` + `uses torch_dtype` | The version pin is being violated. Reinstall via the pinned `requirements.txt`, do **not** rename the kwarg in `train.py` |
| `NEITHER` | On `transformers=5.7.0+` this is **expected** — the signature is now `(model_args, **kwargs)` and `dtype` is consumed via `kwargs.pop("dtype")` inside `PreTrainedModel.from_pretrained` (~line 252 of `modeling_utils.py`). Confirm by running: `python -c "import transformers, inspect; print('OK' if 'kwargs.pop(\"dtype\"' in inspect.getsource(transformers.PreTrainedModel.from_pretrained) else 'BROKEN')"` — if it prints `OK`, the install is fine and `train.py:679` works as written. If it prints `BROKEN`, the install really is broken; stop and add a FRICTION entry. The original signature-based check above does not survive the transformers 5.x refactor; see F-001 in `FRICTION.md` for context. |

> If `check_install.py` warns about `transformers != 4.51.3`, that warning is **stale** — see Part 3.

## 5. Prepare data and validation split

```bash
python prepare.py   # downloads the model + creates data/val_split.json
```

This downloads ~60 GB of weights to `~/.cache/huggingface/`. First time only.

## 6. First smoke run (SFT-only)

The active mode on this branch is **SFT-only** (`SKIP_GRPO=True` in `train.py`). Confirm that's still the default before running:

```bash
grep "^SKIP_GRPO" train.py
# Expected: SKIP_GRPO = False        # ...
# Wait — actually you want SKIP_GRPO = True per program.md. If it's False, edit before running.
```

Then:

```bash
python train.py
```

Watch for `METRIC: 0.XXXX` at the end. That's your post-T0 baseline. Capture it in `STATUS.md` before applying any Tier 1 changes (per `program.md`).

## 7. (Optional) Adapter-on-BF16 sanity check

The submitted adapter gets loaded onto a **BF16** base at scoring time. Verify this path works at least once before doing any sweep work:

```bash
# Quick verification — load adapter trained on NVFP4 onto a fresh BF16 base and run one inference
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prepare import MODEL_ID
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={'':0}, trust_remote_code=True)
model = PeftModel.from_pretrained(base, './adapter')
out = model.generate(**tok('Test prompt: 2+2 =', return_tensors='pt').to('cuda'), max_new_tokens=20)
print(tok.decode(out[0]))
"
```

If this fails, the adapter is unusable for submission regardless of the training METRIC. Log to `FRICTION.md`.

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

## Known issue: `transformers` pin conflict

`requirements.txt` pins `transformers>=5.0.0` but `check_install.py:83` warns if `transformers != 4.51.3`. These are inconsistent and one of them is stale. **Resolve this before T1.3 lands**:

1. On a working pod, capture the actually-working version: `pip show transformers | grep Version`
2. Update `requirements.txt` to pin that exact version (`transformers==X.Y.Z`)
3. Either update the hardcoded version in `check_install.py:83` to match, or delete the hardcoded check and have it just print whatever version is installed

This conflict is what bit the Codex review (P1 #1) — without resolving it, the dtype-vs-torch_dtype debate keeps recurring.

## Stale `transformers_modules` cache after a version bump

If you upgrade `transformers` and start seeing weird "module has no attribute" errors:

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA_hyphen_Nemotron*
```

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

# Tail W&B-style metric line from a long-running train
python train.py 2>&1 | tee run.log | grep -E "METRIC:|loss|step"
```

## Pod recycling

RunPod pods are disposable. If anything goes irrecoverably wrong, the fastest path is usually `terminate the pod → spin up fresh → repeat Part 1`. The HF cache and pip wheels rebuild in ~10 minutes; saving the broken state rarely pays off.
