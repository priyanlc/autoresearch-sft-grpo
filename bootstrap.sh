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

# Step 2 — build helpers (ninja in particular — see header)
uv pip install ninja packaging wheel setuptools

# Step 3 — source-built CUDA packages, --no-build-isolation.
# Both must be installed even though train.py:398 force-disables the Mamba
# fast path: transformers' dynamic-module loader does AST-level static
# import checking on modeling_nemotron_h.py and refuses to load it if any
# imported module is missing — including causal_conv1d, whose import sits
# inside an `if is_causal_conv1d_available():` guard at runtime but is
# detected unconditionally by AST scan. See FRICTION.md F-001 and F-009.
uv pip install mamba_ssm --no-build-isolation
uv pip install causal_conv1d --no-build-isolation
