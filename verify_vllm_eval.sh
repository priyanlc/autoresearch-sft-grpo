#!/usr/bin/env bash
# verify_vllm_eval.sh — One-command smoke test for the T2.14 vLLM eval path.
#
# Closes the T2.14 verification gate from STATUS.md: vLLM METRIC must match
# the HF-eval METRIC for the same adapter before flipping USE_VLLM_EVAL=True
# in train.py (07-train-py-gap-analysis.md § Gap A acceptance criterion).
#
# What this does, in order:
#   1. Pre-flight: venv active, ./adapter exists, val_split.json exists,
#      vllm_eval.py + train.py + prepare.py in cwd.
#   2. Capture F-013 pin baseline (torch / mamba_ssm / causal_conv1d versions).
#   3. Install vllm (`--no-deps` so F-013 pins survive), plus ray on demand.
#   4. F-013 drift detection (re-check pins, fail loud if anything moved).
#   5. Run `python vllm_eval.py --adapter ./adapter`, tee output to a log.
#   6. Parse METRIC, compare to expected. Exact match → PASS.
#      1-sample diff → MISMATCH (variance vs real divergence — re-run).
#      >1-sample diff → MISMATCH (likely real bug — do NOT promote).
#
# Usage:
#   ./verify_vllm_eval.sh                      # default: expected METRIC = 0.6667 (T2.10)
#   ./verify_vllm_eval.sh --expected 0.7333    # if you've moved past T2.10
#   ./verify_vllm_eval.sh --adapter /path/...  # non-default adapter dir
#   ./verify_vllm_eval.sh --skip-install       # vllm already installed, just run
#
# Exit codes:
#   0 — PASS (measured METRIC == expected)
#   1 — install failure (vllm, ray) or F-013 drift
#   2 — METRIC mismatch (the actual T2.14 verification failure)
#   3 — pre-flight failure (missing venv, files, args)
#
# Time budget: ~15 min first run (install + vLLM init + 30-sample eval),
#              ~5 min re-runs (--skip-install).

set -euo pipefail

# Defaults — override via CLI or by editing here.
EXPECTED_METRIC="0.6667"   # T2.10 baseline as of 2026-05-30; override if you've moved past it.
ADAPTER_DIR="./adapter"
SKIP_INSTALL=0
LOG_FILE="vllm_eval_$(date +%Y%m%d_%H%M%S).log"

# --- ANSI colors (only when stdout is a tty) ---
if [[ -t 1 ]]; then
    GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YEL=$'\033[1;33m'; RST=$'\033[0m'
else
    GREEN=''; RED=''; YEL=''; RST=''
fi

step() { printf '\n%s=== %s ===%s\n' "$YEL" "$1" "$RST"; }
pass() { printf '%s [PASS] %s%s\n' "$GREEN" "$1" "$RST"; }
fail() { printf '%s [FAIL] %s%s\n' "$RED" "$1" "$RST"; }
info() { printf '       %s\n' "$1"; }

usage() {
    sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \?//;/^set -euo pipefail/d'
    exit 0
}

# --- Arg parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --expected)     EXPECTED_METRIC="$2"; shift 2 ;;
        --adapter)      ADAPTER_DIR="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --help|-h)      usage ;;
        *) printf 'unknown arg: %s\n' "$1" >&2; exit 3 ;;
    esac
done

# --- 1. Pre-flight ---
step "1. Pre-flight checks"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    fail "no active venv. Run 'source .venv/bin/activate' first."
    exit 3
fi
pass "venv active ($VIRTUAL_ENV)"

for f in vllm_eval.py train.py prepare.py; do
    if [[ ! -f "$f" ]]; then
        fail "$f not found in cwd. Run from the autoresearch-sft-grpo repo root."
        exit 3
    fi
done
pass "vllm_eval.py / train.py / prepare.py present in cwd"

if [[ ! -d "$ADAPTER_DIR" ]]; then
    fail "adapter dir $ADAPTER_DIR not found. Run 'python train.py' first to save an adapter."
    exit 3
fi
pass "adapter dir present: $ADAPTER_DIR"

if [[ ! -f "data/val_split.json" ]]; then
    fail "data/val_split.json not found. Run 'python prepare.py' first."
    exit 3
fi
pass "val_split.json present"

# --- 2. F-013 pin baseline (capture BEFORE install for drift detection) ---
step "2. F-013 pin baseline (pre-install)"

BEFORE_PINS="$(python - <<'PY'
import torch, mamba_ssm, causal_conv1d
print(f"torch={torch.__version__}")
print(f"mamba_ssm={mamba_ssm.__version__}")
print(f"causal_conv1d={causal_conv1d.__version__}")
PY
)"
printf '%s\n' "$BEFORE_PINS"

# --- 3. Install vllm (unless --skip-install) ---
if [[ "$SKIP_INSTALL" == "0" ]]; then
    step "3. Install vllm (--no-deps to protect F-013 pins)"

    if python -c "import vllm" 2>/dev/null; then
        existing="$(python -c 'import vllm; print(vllm.__version__)')"
        pass "vllm already installed (version $existing) — skipping install"
    else
        info "Installing vllm>=0.6.6,<0.7 with --no-deps (~5 min)..."
        if ! uv pip install 'vllm>=0.6.6,<0.7' --no-deps; then
            fail "vllm install failed"
            exit 1
        fi
        pass "vllm installed"
    fi

    # ray: common runtime dep of vllm that --no-deps skips. Install only if missing.
    if ! python -c "import ray" 2>/dev/null; then
        info "Installing ray (runtime dep of vllm, not pulled by --no-deps)..."
        if ! uv pip install ray; then
            fail "ray install failed"
            exit 1
        fi
        pass "ray installed"
    else
        pass "ray already importable"
    fi

    # --- 4. F-013 drift detection (post-install) ---
    step "4. F-013 drift detection (post-install)"

    AFTER_PINS="$(python - <<'PY'
import torch, mamba_ssm, causal_conv1d
print(f"torch={torch.__version__}")
print(f"mamba_ssm={mamba_ssm.__version__}")
print(f"causal_conv1d={causal_conv1d.__version__}")
PY
)"
    printf '%s\n' "$AFTER_PINS"

    if [[ "$BEFORE_PINS" != "$AFTER_PINS" ]]; then
        fail "F-013 pins drifted during vllm install — do NOT proceed."
        printf '\n--- diff ---\n'
        diff <(printf '%s\n' "$BEFORE_PINS") <(printf '%s\n' "$AFTER_PINS") || true
        printf '\n'
        info "Recovery: uv pip install --force-reinstall 'torch>=2.2.0' --index-url https://download.pytorch.org/whl/cu121"
        info "          uv pip install --force-reinstall mamba_ssm==2.3.1 --no-build-isolation --no-deps"
        info "          uv pip install --force-reinstall causal_conv1d==1.6.2.post1 --no-build-isolation --no-deps"
        info "Then re-run with --skip-install to verify pins hold and just exercise vllm_eval.py."
        exit 1
    fi
    pass "F-013 pins intact"
else
    step "3. Install (skipped per --skip-install)"
    if ! python -c "import vllm" 2>/dev/null; then
        fail "--skip-install passed but vllm not importable. Remove --skip-install for first run."
        exit 3
    fi
    pass "vllm importable: $(python -c 'import vllm; print(vllm.__version__)')"
fi

# --- 5. Run vllm_eval.py ---
step "5. Run vllm_eval.py --adapter $ADAPTER_DIR --max-new-tokens 512"
info "Logging to: $LOG_FILE"
printf '\n'

if ! python -u vllm_eval.py --adapter "$ADAPTER_DIR" --max-new-tokens 512 2>&1 | tee "$LOG_FILE"; then
    fail "vllm_eval.py crashed. See $LOG_FILE for trace."
    exit 1
fi

# --- 6. Parse METRIC and compare ---
step "6. METRIC comparison"

MEASURED="$(grep -E '^METRIC: ' "$LOG_FILE" | tail -1 | awk '{print $2}')"
if [[ -z "$MEASURED" ]]; then
    fail "no 'METRIC: ' line found in vllm_eval.py output. See $LOG_FILE."
    exit 1
fi

printf '       Expected (HF baseline): %s\n' "$EXPECTED_METRIC"
printf '       Measured (vLLM):       %s\n' "$MEASURED"

# Bash can't do float math; defer to python for the comparison.
DELTA="$(python -c "print(abs(float('$MEASURED') - float('$EXPECTED_METRIC')))")"
SAMPLES_DELTA="$(python -c "import math; print(round(abs(float('$MEASURED') - float('$EXPECTED_METRIC')) / (1/30)))")"
printf '       Delta: %s (~%s samples on a 30-sample val)\n\n' "$DELTA" "$SAMPLES_DELTA"

if python -c "import sys; sys.exit(0 if float('$MEASURED') == float('$EXPECTED_METRIC') else 1)"; then
    pass "vLLM METRIC matches HF METRIC exactly."
    info "Safe to flip USE_VLLM_EVAL=True in train.py and re-run a full sweep."
    info "Suggested commit after the flip: T1.34: USE_VLLM_EVAL=True default after T2.14 verified"
    exit 0
elif [[ "$SAMPLES_DELTA" == "1" ]]; then
    fail "1-sample diff ($DELTA). At the variance floor — could be tokenization noise or a real divergence."
    info "Recovery: re-run this script (no --skip-install needed) to check stability."
    info "          Inspect per-category in $LOG_FILE vs the T2.10 entry in STATUS.md."
    info "          Do NOT flip USE_VLLM_EVAL=True until the result is stable."
    exit 2
else
    fail "${SAMPLES_DELTA}-sample diff ($DELTA). This is above the variance floor — likely a real bug."
    info "Recovery: inspect $LOG_FILE per-category, compare with T2.10 entry in STATUS.md."
    info "          Common causes: chat-template drift, sampling-params mismatch, LoRA-not-applied at inference."
    info "          Do NOT flip USE_VLLM_EVAL=True until resolved."
    exit 2
fi
