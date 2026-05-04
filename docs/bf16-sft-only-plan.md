---
title: BF16 SFT-Only Plan — main branch baseline
type: plan
status: active
updated: 2026-05-04
target_branch: main
sources: [
  methodology.md,
  fast-path-and-cache.md,
  ../program.md,
  ../BRANCH_NOTES.md,
]
---

> **Origin note:** Originally drafted in a separate documentation vault as `wiki/bf16-sft-only-plan.md`. This in-repo copy at `docs/bf16-sft-only-plan.md` is the canonical version going forward.

# BF16 SFT-Only Plan — `main` branch baseline

Strategic anchor for `autoresearch-sft-grpo:main`. A companion strategic plan for the `nvfp4-blackwell` branch (the NVFP4 worked example) lives in the documentation vault outside this repo. Referenced from `main`'s [`program.md`](../program.md) *Plan reference* line per the 8-artefact methodology in [`methodology.md`](methodology.md).

## Why this plan exists

`main` holds the proven 0.5333 baseline at commit `c1bb0a6`, produced by:

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` in plain BF16 (no quantization)
- SFT-only training (GRPO punted; see `FRICTION.md` F-002)
- 1200 SFT samples (200 per category), 1 epoch
- `LORA_RANK=32`, `target_modules='all-linear'`
- `USE_COT=True` with static category-specific CoT templates
- `EVAL_MAX_NEW_TOKENS=512`, `EVAL_BATCH_SIZE=1` on A100 80GB

This plan exists to (a) lock that configuration as the parent-of-branches reference, (b) document why each component is locked vs parameterizable, and (c) state the punt boundary for GRPO.

## What is locked

| Component | Locked value | Lock rationale |
|---|---|---|
| Base dtype | bfloat16 | 4-bit degrades 0.5333 → 0.1333 (FRICTION F-003) |
| Training mode | SFT-only | GRPO infeasible on Mamba/MoE+TRL (FRICTION F-002) |
| `USE_COT` | `True` | `False` yields METRIC 0.1667 (FRICTION F-004) |
| `EVAL_MAX_NEW_TOKENS` | 512 | 128 truncates answers before `\boxed{}` closes (FRICTION F-005) |
| KV cache during eval | Disabled | Nemotron `HybridMambaAttentionDynamicCache` bugs (FRICTION F-001) |

Reverting any of these locks is a research bet that needs its own branch and its own `BRANCH_NOTES.md` delta — not a `main` modification.

## What is parameterizable for future Tier 2 sweeps

- `SFT_SAMPLES_PER_TYPE` (currently 200; sweep 100–400)
- `SFT_EPOCHS` (currently 1; sweep 1–3)
- `SFT_LR` (sweep 1e-5 .. 5e-4)
- `LORA_ALPHA` (rank=32 fixed by competition; alpha sweep 32, 64)
- `LORA_DROPOUT` (currently 0.05; sweep 0.0, 0.05, 0.1)
- `target_modules` subset (currently `all-linear`; sweep skip-last-N-layers)
- CoT template variations (static → dynamic → brief hints; the in-progress `bdxqkblt5` task in the 2026-04-06 STATUS.md snapshot was such an attempt)
- Data strategy: oversample hard categories (cipher 0%, gravity 20%, symbol 40% are the largest gaps)

## What is explicitly punted

- **GRPO of any flavour** — deferred to a dedicated branch when TRL has stable Mamba/MoE support; do not re-enable on `main`. The standing `try/except` around `GRPOTrainer.train()` (`train.py:506–523`) is the smoke harness, not a regression.
- **4-bit, FP8, NVFP4** — each gets its own branch. See `nvfp4-blackwell` for the NVFP4 worked example.

## Acceptance bar for changes on `main`

Any commit that touches `train.py` logic must produce METRIC ≥ 0.5333 on the validation set, recorded in `results.tsv` with a non-empty description. Regressions get reverted, not patched on top — see `program.md` § Branch Hygiene.

## Relationship to other branches

`main` is the parent. Branches diff from `main`; their `BRANCH_NOTES.md` captures the delta. As of 2026-05-03 the only active sibling is `nvfp4-blackwell`.

## Changelog

- **2026-05-03** — Initial plan, written against `main` HEAD `c1bb0a6` to anchor the methodology assimilation. The narrative assimilation plan that drove T1.1..T1.12 lives in the separate documentation vault.
- **2026-05-04** — Copied into the autoresearch repo at `docs/bf16-sft-only-plan.md` as part of T1.13 so the repo is self-contained when published standalone.
