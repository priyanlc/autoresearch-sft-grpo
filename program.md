# Autoresearch: SFT + GRPO for Nemotron Reasoning Challenge

## Goal
Maximize **validation accuracy** on 6 types of "Alice's Wonderland" reasoning puzzles by optimizing a two-phase training pipeline:
1. **SFT warmup** — teaches the model answer format and basic reasoning patterns
2. **GRPO** — reinforcement learning that rewards correct answers, good formatting, and reasoning

## Metric
The single metric to optimize is printed at the end of `train.py`:
```
METRIC: 0.XXXX
```
This is the proportion of correctly answered puzzles on a held-out validation set (30 samples, 5 per category). Higher is better. Maximum is 1.0.

## What you can modify
You may ONLY edit `train.py`. Everything in `prepare.py` is read-only.

### Things worth trying (in rough priority order):

**Reward function design (highest impact):**
- Reward weights: `W_CORRECTNESS`, `W_FORMAT`, `W_REASONING`, `W_CATEGORY_BONUS`
- Reward function logic: modify how correctness, format, reasoning, and category rewards are computed
- Add new reward functions (e.g., answer-length penalty, confidence calibration)
- Reward shaping: partial credit for near-correct answers

**SFT prompt format (high impact):**
- The chain-of-thought in `_COT_BY_TYPE` — make it more specific or problem-aware
- Whether SFT includes reasoning at all vs just `\boxed{answer}`
- Category-specific formatting strategies

**GRPO hyperparameters (high impact):**
- `GRPO_NUM_GENERATIONS`: more = better advantage estimates but slower (2-8)
- `GRPO_TEMPERATURE`: controls exploration during training (0.5-1.0)
- `GRPO_BETA`: KL penalty strength (0.0-0.1, 0 = no constraint)
- `GRPO_MAX_COMPLETION`: max tokens for generation (256-1024)
- `GRPO_LR`: learning rate (1e-6 to 5e-5)

**Data strategy (high impact):**
- `SFT_SAMPLES_PER_TYPE` and `GRPO_SAMPLES_PER_TYPE`: sample counts
- Using different data for SFT vs GRPO (currently non-overlapping)
- Weighted sampling toward harder categories
- Curriculum ordering

**LoRA configuration (medium impact):**
- `LORA_RANK`: adapter rank (1-32, competition max is 32)
- `LORA_ALPHA`: scaling factor (typically 1x or 2x the rank)
- `LORA_DROPOUT`: regularization (0.0 to 0.1)
- `TARGET_MODULES`: which layers to adapt

**SFT hyperparameters (medium impact):**
- `SFT_LR`: learning rate (1e-5 to 5e-4)
- `SFT_EPOCHS`: number of passes (1-3)
- `SFT_MAX_SEQ_LEN`: max sequence length

## Constraints
- LoRA rank must be <= 32 (competition rule)
- The model must output answers in `\boxed{}` format
- Training must complete within the time budget (~25 min target)
- Do not modify `prepare.py`
- SFT and GRPO data should not overlap with validation (handled automatically)

## The 6 puzzle types
1. **bit_ops** — deduce bit transformation from input/output examples -> 8-char binary string
2. **cipher** — crack substitution cipher -> lowercase words
3. **gravity** — infer gravitational constant from d=0.5gt^2 -> decimal number
4. **numeral** — convert number to Roman numerals -> Roman numeral string
5. **symbol** — figure out symbol substitution/arithmetic rules -> symbol string
6. **unit_conv** — find hidden linear conversion factor -> decimal number

Each type has 5 validation samples. Check per-category accuracy to find weak spots.

## Tips
- The reward functions are the most powerful lever — GRPO learns whatever the rewards incentivize
- Correctness reward dominates; other rewards are auxiliary signals
- SFT warmup matters: it sets the starting point for GRPO exploration
- If GRPO degrades performance vs SFT-only, try reducing `GRPO_LR` or increasing `GRPO_BETA`
- Check per-category accuracy to find which puzzle types need attention
- The model uses `enable_thinking=True` at inference — ensure SFT and GRPO both use it
- If a change crashes, revert and try something smaller
