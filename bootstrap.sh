#!/usr/bin/env bash
# Bootstrap CUDA-built dependencies that can't be safely installed from
# requirements.txt under PEP 517 build isolation.
#
# Why this script exists (do not collapse into requirements.txt):
#   - torch must come from the PyTorch CDN (cu121), not PyPI, for both
#     speed (~36 MB/s vs ~50 KB/s on RunPod EU pods) and CUDA-tag accuracy.
#   - mamba_ssm has `import torch` in setup.py. Under PEP 517 build
#     isolation (pip's default), the installer creates a throwaway venv
#     and re-downloads torch into it just to compile — slow and fragile.
#     --no-build-isolation reuses the real venv's torch.
#   - ninja must be present before mamba_ssm builds, or the CUDA compile
#     falls back to slow serial nvcc or fails outright.
#
# Run inside an active uv venv (`.venv/bin/activate` must already be sourced).

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: no active venv. Run 'uv venv && source .venv/bin/activate' first." >&2
  exit 1
fi

# Step 1 — torch from PyTorch CDN
uv pip install 'torch>=2.2.0' --index-url https://download.pytorch.org/whl/cu121

# Step 2 — build helpers (ninja in particular — see header).
# einops is a RUNTIME dep of mamba_ssm; we install it explicitly here because
# Step 3 uses --no-deps (see F-013) and would otherwise skip it.
uv pip install ninja packaging wheel setuptools einops

# Step 3 — source-built CUDA packages, --no-build-isolation.
# Both must be installed because transformers' dynamic-module loader does
# AST-level static import checking on modeling_nemotron_h.py and refuses
# to load it if any imported module is missing — including causal_conv1d,
# whose import sits inside an `if is_causal_conv1d_available():` guard at
# runtime but is detected unconditionally by AST scan. At runtime the
# fast-path kernels run during SFT teacher-forced forward (post-T2.9) and
# are disabled before eval at train.py:582 (paired with use_cache=False
# at train.py:579 as the F-001 redundant defense pair).
# See FRICTION.md F-001 and F-009.
#
# --no-deps is LOAD-BEARING (F-013): --no-build-isolation only governs the
# *build* env, not runtime dependency resolution. Without --no-deps, uv
# re-resolves mamba_ssm's `torch` requirement and silently upgrades the
# Step-1 torch (2.5.1+cu121) to the newest PyPI torch (a cu13 build), which
# (a) leaves mamba_ssm's already-compiled .so ABI-incompatible and (b) makes
# causal_conv1d's source build fail against the system CUDA 12.x toolkit
# (cu130 vs nvcc 12.8 -> hard "CUDA version mismatch").
#
# The VERSION PINS are equally load-bearing (F-013): mamba_ssm 2.3.2.post1
# hard-requires torch==2.12 (cu13); 2.3.1 is the last release compatible with
# torch 2.5.x/cu121. Leaving mamba_ssm unpinned installs 2.3.2.post1 here but
# requirements.txt's full-graph resolve (with transformers==4.51.3) pins 2.3.1
# -> version mismatch -> uv rebuilds 2.3.1 under build isolation downstream,
# re-triggering the torch drift. Pin both stages to the same compatible
# versions so requirements.txt sees them satisfied and never rebuilds.
uv pip install 'mamba_ssm==2.3.1' --no-build-isolation --no-deps
uv pip install 'causal_conv1d==1.6.2.post1' --no-build-isolation --no-deps
