# Autoresearch SFT+GRPO — Session Status

**Last updated:** 2026-04-06
**Working directory:** `/workspace/9-kaggle/autoresearch-sft-grpo/`

## Current best METRIC: **0.5333** (vs baseline 0.5000)

| Category  | Score  | Notes |
|-----------|--------|-------|
| bit_ops   | 60%    | 3/5   |
| cipher    | 0%     | 0/5 — biggest opportunity |
| gravity   | 20%    | 1/5   |
| numeral   | 100%   | 5/5 ✅ |
| symbol    | 40%    | 2/5   |
| unit_conv | 100%   | 5/5 ✅ |

## Winning configuration (committed in train.py as of 0.5333 run)

- **Model loading:** bf16, no 4-bit quantization (4-bit degraded perf: 0.1333)
- **SFT_SAMPLES_PER_TYPE = 200** (1200 total — matching Kaggle baseline)
- **SFT_EPOCHS = 1**
- **USE_COT = True** — SFT teaches thinking pattern matching eval's `enable_thinking=True`
- **EVAL_MAX_NEW_TOKENS = 512** — need room for thinking + `\boxed{}`
- **EVAL_BATCH_SIZE = 1** — avoid OOM on A100 80GB
- **GRPO:** skipped/fails gracefully (Nemotron cache bugs make it impractical)
- **LORA_RANK = 32**, `target_modules='all-linear'`
- **Peak VRAM:** ~78 GB / 80 GB

## Timing (per run, bf16 + 1200 samples)

| Stage              | Time        |
|--------------------|-------------|
| Model load (13 shards) | ~5 min  |
| SFT training (1200 samples, 1 epoch) | ~1 hour |
| GRPO fails quickly | ~1 min      |
| Eval (30 samples, no KV cache) | ~3 hours |
| **Total**          | **~4-5 hours** |

## Key blockers & lessons

1. **Nemotron cache bugs** — `HybridMambaAttentionDynamicCache` has multiple bugs (missing `conv_kernel_size`, list vs tensor `.device`, wrong `conv_dim` 4096 vs 6144, SSM state shape mismatches). Cache is **disabled** — generation works but is slow.
2. **GRPO impossible in practice** — Without KV cache each GRPO step takes ~30 min (generation is O(n²)). TRL's GRPOTrainer also had tensor mismatches with the Mamba/MoE model. **Accepted: SFT-only.**
3. **4-bit quantization degrades quality** — METRIC 0.1333 with 4-bit vs 0.5333 with bf16. Use bf16.
4. **USE_COT=True is crucial** — With USE_COT=False, model generates long thinking via `enable_thinking=True` but never produces `\boxed{}` answer → 0.1667. With USE_COT=True, SFT teaches the think→boxed pattern → 0.5333.
5. **EVAL_MAX_NEW_TOKENS matters** — 128 tokens cuts off before the answer. 256 barely works. 512 works reliably.

## In-progress run (at time of pause)

**Task ID:** `bdxqkblt5`
**Change:** Added `_build_dynamic_cot()` in `build_sft_text` — constructs per-example reasoning for cipher (substitution map), gravity (g = 2d/t²), unit_conv (factor = out/in), bit_ops (hint). Falls back to static CoT otherwise.
**Hypothesis:** Dynamic CoT showing the actual derivation should unlock cipher (0% → ?) and improve gravity (20% → ?).
**Status when paused:** SFT training active, 67 GB VRAM used.

Output log: `/tmp/claude-1001/-home-claude-user/84fb044d-df67-4a34-a4dc-a869f261630d/tasks/bdxqkblt5.output`

## To resume

1. Check if the run (`bdxqkblt5`) ever finished — grep the output file for `METRIC:`.
2. If completed and METRIC > 0.5333: commit and continue to next optimization.
3. If completed and METRIC ≤ 0.5333: `git checkout train.py` to revert dynamic CoT, then try next idea.
4. If not completed (pod restarted): re-run `python -u train.py` — it should pick up the current `train.py` with dynamic CoT changes.

## Next optimization ideas (in priority order)

1. **Improve cipher CoT** — the current regex for extracting pairs may not match all prompt formats; verify it actually fires on training data.
2. **SFT_EPOCHS = 2** — give the model more time to learn the dynamic patterns (doubles SFT time to ~2 hours).
3. **Oversample hard categories** (cipher, gravity, symbol) via a custom sampler instead of stratified.
4. **Add a cipher-specific format reward** during eval — check the answer is lowercase words.
5. **Verify the prompt format** — read a few raw examples from `data/train.csv` to confirm regex assumptions.

## Files modified

- `train.py` — dynamic CoT added, USE_COT=True, 1200 samples, EVAL_MAX_NEW_TOKENS=512, EVAL_BATCH_SIZE=1, `model.config.use_cache = False` before eval
- `train.py.bak` — earlier backup
- `eval_only.py` — standalone eval script (loads saved adapter) — optional utility
- `/home/claude-user/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/.../modeling_nemotron_h.py` — **modified in place** with cache bug fixes (conv_dim, conv_kernel_size, `.device` access). These don't affect SFT but matter if anyone tries GRPO with cache again. Pyc cache should be cleared with `rm -rf .../__pycache__` before any run.

## Git state

Commits are inconsistent due to a .git/index lock issue earlier in the session. The 0.5333 config is in the working tree (train.py). `git log` shows only the original commits; my attempted commits did not take. **Important:** save a copy of `train.py` before `git checkout`.
