# Autoresearch SFT+GRPO — `main` branch (BF16 SFT-only anchor)

**Branch:** `main`
**Parent:** none — this is the parent-of-branches reference.
**Reference baseline:** METRIC 0.5333 at commit `c1bb0a6` ("SFT-only optimized: METRIC 0.5333 vs 0.5000 baseline").
**Strategic plan:** [`docs/bf16-sft-only-plan.md`](docs/bf16-sft-only-plan.md)

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

1. **Mamba fast-path disable** (around `train.py:398`) — loops `sys.modules` for `modeling_nemotron_h` and sets `is_fast_path_available = False`. Forces pure-PyTorch math even where the fused CUDA kernels would otherwise run. **Defends F-001 in tandem with patch 2** — without this, an accidental `use_cache=True` downstream would re-trigger the cache bugs from a different angle.
2. **`model.config.use_cache = False`** at `train.py:545` before eval — prevents generation from touching the broken `HybridMambaAttentionDynamicCache`. Defends F-001 directly.

These are redundant defenses, not duplicates. See [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) for the full mechanical treatment of why both are needed. Cross-references are added inline as `# See FRICTION.md F-NNN` comments in T1.7.

## Tier 1 chronology

- **T1.1 .. T1.8a (2026-05-03)** — Methodology assimilation. Converted `main` from a working snapshot into the methodology-compliant 8-artefact set documented in [`docs/methodology.md`](docs/methodology.md). No `train.py` logic changes during T1; T1.7 added inline F-id comments only. T1.8 (regression run on a real pod) deferred to T1.8b.
- **T1.3a / T1.7a / T1.9 / T1.10 / T1.11 / T1.12 (2026-05-04)** — Doc-only correctness sweep across `runpod-setup.md`, `program.md`, `BRANCH_NOTES.md`, `FRICTION.md`, `requirements.txt`, `data/README.md`, `.gitignore`. Triggered by the post-T1.7 line-number drift, the discovery that `causal_conv1d` is (was thought to be) inert, the data-attribution gap, and the staged-install over-engineering. **T1.9's claim that `causal_conv1d` could be safely uninstalled was later disproved — see T1.14.**
- **T1.13 (2026-05-04)** — Inlined three documentation files (`docs/methodology.md`, `docs/bf16-sft-only-plan.md`, `docs/fast-path-and-cache.md`) into the repo so cross-references resolve when the repo is published standalone on GitHub. Vault and repo will drift; in-repo `docs/` is canonical going forward.
- **T1.14 (2026-05-06)** — Restored `causal_conv1d` as a hard install-time dep, reverting T1.9. Closes FRICTION F-009. The previous "conditional import handles absence" mental model was runtime-correct but missed that transformers' dynamic-module loader (`dynamic_module_utils.py:check_imports`) does AST-level static import checking on `modeling_nemotron_h.py` and rejects the file before any guard runs. Fast-path is still force-disabled at runtime by `train.py:398`, so the package adds no behavioural change — it just exists to satisfy the static check.
- **T1.15 / T1.16 / T1.17 (2026-05-07)** — First end-to-end autonomous run on a fresh A100 80GB pod after the T1.14 fix. T1.15 logs FRICTION F-010 (hf_xet worker→main deadlock during weights download; workaround = `unset HF_XET_HIGH_PERFORMANCE` and rely on `hf_transfer` alone). T1.16 captures the resulting METRIC 0.5667 baseline (+0.0334 vs c1bb0a6, single-cipher-sample shift) into STATUS.md + results.tsv and adds `adapter_sanity_check.py` as a reusable helper for the program.md § Validation Contract item 5. T1.17 adds `hf_transfer` to `requirements.txt` (closes F-008, which had been worked around by ad-hoc install during the run) and logs the quantified cost of the `use_cache=False` workaround in F-001 (eval ~3 h vs ~10 min with cache, ≈18× slowdown; tie-in to F-002 GRPO economics).
- **T1.20 (2026-05-09)** — README.md doc-only refresh: headline METRIC updated 0.5333 → 0.6000 (T2.8); regression-bar wording in agent-handover section clarified as "floor 0.5333 / current 0.6000"; note added that some Tier 2 has landed on `main` (T2.7 reverted, T2.8 kept); sanity-check references repointed from `sanity_check.py` to the canonical `adapter_sanity_check.py`; repo-layout block extended to include `adapter_sanity_check.py`, `check_install.py`, and `eval_only.py`. No `train.py` logic change. Open follow-up: the dual `sanity_check.py` / `adapter_sanity_check.py` pair should be deduplicated under a separate T-id (out of scope for T1.20).
- **T1.21 (2026-05-09)** — Deduplicate sanity-check helpers (closes the T1.20 follow-up). Deleted older `sanity_check.py` (May 6, predates T1.16); kept canonical `adapter_sanity_check.py` per program.md § Validation Contract item 5 and the T1.16 chronology. Removed the back-compat line from README.md repo layout. No external references existed beyond the deleted file's own docstring. No `train.py` logic change.
- **T1.22 (2026-05-09)** — Sync `prompt.md` regression-bar wording with the post-T2.8 README (closes the second T1.20 follow-up). Lines 23–24 now reference 0.5333 as the "locked floor" rather than the "locked baseline" and call out current best 0.6000 at `c4a9d1c`. Step 8 reworded to keep the 0.5333 STOP threshold but adds a forward note that some Tier 2 has already landed, so beating 0.5333 still requires explicit go-ahead before new sweeps. **Threshold unchanged at 0.5333; agent behaviour preserved.** No `train.py` logic change.
- **T1.23 (2026-05-09)** — `train.py` line-number citation sweep across docs (T1.7a-style). T2.8's `_build_dynamic_cot` additions shifted the model-load block, fast-path disable, GRPO try/except, and `use_cache=False` patch by ~9 lines. Updated citations: `:386 → :398` (fast-path disable), `:536 → :545` (`use_cache=False`), `:511 → :520` (GRPO try line), `:383 → :392` (`torch_dtype=bfloat16`), `:380-385 → :388-394` (model load), `:511-528 → :520-540` (GRPO block), `:506-523 → :520-540` (alt GRPO block), `:386-389 → :395-398` (fast-path block range), `line 156 → line 165` (USE_COT usage in `build_sft_text`). Files swept: `BRANCH_NOTES.md`, `program.md`, `prompt.md`, `FRICTION.md`, `runpod-setup.md`, `check_install.py`, `adapter_sanity_check.py`, `docs/fast-path-and-cache.md`, `docs/bf16-sft-only-plan.md`. STATUS.md historical entries left untouched per append-only convention. No `train.py` logic change.
- **T1.24 (2026-05-09)** — Stale METRIC wording sweep across remaining docs (extends T1.20/T1.22 pattern). Updated `runpod-setup.md` (locked-baseline → locked floor + current 0.6000 pointer), `docs/autoresearch-handoff.md` (added current-best context + Tier 2 caveat to the regression-bar sentence), `docs/bf16-sft-only-plan.md` § "Why this plan exists" + § "Acceptance bar" (reframed locked-floor / current-best with `STATUS.md` pointer; clarified that floor 0.5333 is the revert target while current 0.6000 is the de-facto bar for new Tier 2), and `README.md:75` "Where to read next" bullet for `prompt.md`. Historical references in `docs/fast-path-and-cache.md` ("0.5333-baseline session 2026-04-06") left as-is — those are timestamped events, not current-state claims. **Threshold unchanged at 0.5333**; floor → revert target framing preserved everywhere.

## Fallback

If a future change ever regresses METRIC below 0.5333, revert to commit `c1bb0a6` (the pre-T1 baseline) and identify the offending T-id before adding more variables.
