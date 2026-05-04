# Autoresearch SFT+GRPO — `main` branch (BF16 SFT-only anchor)

**Branch:** `main`
**Parent:** none — this is the parent-of-branches reference.
**Reference baseline:** METRIC 0.5333 at commit `c1bb0a6` ("SFT-only optimized: METRIC 0.5333 vs 0.5000 baseline").
**Strategic plan:** [`nemotron-vault/wiki/bf16-sft-only-plan.md`](../../../nemotron-vault/wiki/bf16-sft-only-plan.md)

## Purpose

`main` is the canonical BF16 SFT-only configuration for the `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` model on the NVIDIA Nemotron Reasoning Challenge. Future research branches (e.g. `nvfp4-blackwell`) diff from this baseline and capture their delta in their own `BRANCH_NOTES.md`.

## Configuration — the locked-in identity of `main`

| Parameter | Value | Why locked |
|---|---|---|
| Base dtype | `bfloat16` | 4-bit degrades METRIC 0.5333 → 0.1333; see FRICTION F-003 |
| Quantization | none | Same as above |
| GRPO | skipped via try/except | Mamba/MoE + TRL incompat; see FRICTION F-002 |
| `USE_COT` | `True` | `False` yields no `\boxed{}` (METRIC 0.1667); see FRICTION F-004 |
| `SFT_SAMPLES_PER_TYPE` | 200 (1200 total) | Matches Kaggle baseline |
| `SFT_EPOCHS` | 1 | Empirically sufficient for current data ratio |
| `LORA_RANK` | 32 | Competition cap |
| `target_modules` | `all-linear` | Full coverage on BF16 base |
| `LORA_DROPOUT` | 0.05 (default) | No FPQuant constraint on this branch |
| `EVAL_MAX_NEW_TOKENS` | 512 | 128 truncates; see FRICTION F-005 |
| `EVAL_BATCH_SIZE` | 1 | OOM avoidance on A100 80GB |
| `MAX_GRAD_NORM` | 1.0 | Default (`nvfp4-blackwell` overrides to 0.1) |
| `BATCH_SIZE` | 1 | A100 80GB with full BF16 30B model |

## Hardware

- Tested on: A100 80GB (peak ~78 GB observed during eval)
- Cold model load: ~5 min (13 shards from local cache)
- SFT (1200 samples, 1 epoch): ~1 hour
- Eval (30 samples, no KV cache): ~3 hours
- Total per run: ~4–5 hours

## Patches in `train.py`

`main` carries no FPQuant/PEFT patches (no NVFP4 stack). Two non-trivial workarounds live in `train.py`, both defending FRICTION F-001 as a redundant pair:

1. **Mamba fast-path disable** (around `train.py:386`) — loops `sys.modules` for `modeling_nemotron_h` and sets `is_fast_path_available = False`. Forces pure-PyTorch math even where the fused CUDA kernels would otherwise run. **Defends F-001 in tandem with patch 2** — without this, an accidental `use_cache=True` downstream would re-trigger the cache bugs from a different angle.
2. **`model.config.use_cache = False`** at `train.py:536` before eval — prevents generation from touching the broken `HybridMambaAttentionDynamicCache`. Defends F-001 directly.

These are redundant defenses, not duplicates. See `nemotron-vault/wiki/nemotron-fast-path-and-cache.md` for the full mechanical treatment of why both are needed. Cross-references are added inline as `# See FRICTION.md F-NNN` comments in T1.7.

## Tier 1 chronology

- **T1.1 .. T1.8a (2026-05-03)** — Methodology assimilation. Converted `main` from a working snapshot into the methodology-compliant 8-artefact set documented in [`04-autoresearch-methodology.md`](../../../nemotron-vault/wiki/04-autoresearch-methodology.md). No `train.py` logic changes during T1; T1.7 added inline F-id comments only. T1.8 (regression run on a real pod) deferred to T1.8b.

## Fallback

If a future change ever regresses METRIC below 0.5333, revert to commit `c1bb0a6` (the pre-T1 baseline) and identify the offending T-id before adding more variables.
