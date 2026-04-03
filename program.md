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

## Known Baselines

| Run | SFT samples | GRPO | USE_COT | Overall | bit_ops | cipher | gravity | numeral | symbol | unit_conv |
|---|---|---|---|---|---|---|---|---|---|---|
| Kaggle (4-bit) | 1,200 | failed | False | 0.5000 | 80% | 0% | 0% | 100% | 20% | 100% |
| A100 (bf16) | 300 | failed | True | 0.4000 | 20% | 0% | 0% | 100% | 20% | 100% |

### Observations
- numeral and unit_conv are solved — focus elsewhere
- cipher and gravity score 0% — debug output shows correct format but wrong answers
- With `USE_COT=True`, the model parroted static templates verbatim instead of reasoning. Inconclusive if harmful (also fewer samples) — worth testing both settings.
- GRPO crashes on A100 with tensor mismatch (Nemotron Mamba/MoE + TRL incompatibility). SFT fallback is used.

## What you can modify
You may ONLY edit `train.py`. Everything in `prepare.py` is read-only.

### Things worth trying (in rough priority order):

**Reward function design (highest impact):**
- Reward weights: `W_CORRECTNESS`, `W_FORMAT`, `W_REASONING`, `W_CATEGORY_BONUS`
- Reward function logic: modify how correctness, format, reasoning, and category rewards are computed
- Add new reward functions (e.g., answer-length penalty, confidence calibration)
- Reward shaping: partial credit for near-correct answers

**SFT prompt format (high impact):**
- `USE_COT` flag: toggle static CoT templates on/off (default: False)
- With `USE_COT=True`, model got template parroting — try with more data or varied templates
- With `USE_COT=False`, model uses native reasoning via `enable_thinking=True`
- Category-specific formatting strategies, brief hints (not full reasoning templates)

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

9,500 puzzles total, evenly balanced (~1,555-1,602 each). Every puzzle is few-shot rule induction: the model sees examples of a hidden rule and must generalize.

| Category | Count | Few-shot examples | Answer format | Difficulty |
|---|---|---|---|---|
| bit_ops | 1,602 | 7-10 binary pairs | 8-char binary string (e.g., `10010111`) | Hard |
| cipher | 1,576 | 2-4 text pairs | 3-5 lowercase words (e.g., `cat imagines book`) | Medium |
| gravity | 1,597 | 3-5 time/distance points | Decimal, mostly 2dp (e.g., `154.62`) | Easy-Medium |
| numeral | 1,576 | 2-4 number/Roman pairs | Roman numerals, numbers 1-100 (e.g., `XXXVIII`) | Easy |
| symbol | 1,555 | 2-4 symbol equations | 1-4 chars from 36 unique symbols (e.g., `@&`) | Hard |
| unit_conv | 1,594 | 3-5 measurement pairs | Decimal, always 2dp (e.g., `16.65`) | Easy |

### Category details

- **bit_ops**: Must reverse-engineer arbitrary bit operations (shifts, rotations, XOR, AND, OR, NOT). Gets the most examples (7-10) because the rules are complex. Bit distribution in answers is roughly normal, centered around 4 ones.
- **cipher**: Pure lowercase alphabetic substitution cipher. Spaces preserved, full a-z alphabet used in both cipher and plain text. Most answers are 4 words.
- **gravity**: Formula is `d = 0.5 * g * t²` with a secret `g` per puzzle (range ~5.8 to 17.4). Solve for `g` from any data point, then apply. 90% of answers have exactly 2 decimal places.
- **numeral**: Standard Roman numerals (I, V, X, L, C, D, M). Numbers range 1-100, mean 49. Deterministic conversion — the easiest category.
- **symbol**: The most opaque category. Uses 36 unique characters including punctuation, digits, brackets. Some prompts mix numeric and symbolic operations. Rules are arbitrary and hard to generalize.
- **unit_conv**: Linear scaling with a hidden factor (range 0.5-1.8). Divide any output by input to get the factor. All answers have exactly 2 decimal places.

Each type has 5 validation samples. Check per-category accuracy to find weak spots.

## Strategic recommendations

1. **Focus on hard categories**: bit_ops and symbol are where the model struggles most and where gains matter most. Consider oversampling these in training data.
2. **Be cautious with static CoT templates**: With `USE_COT=True`, the model may parrot templates verbatim instead of reasoning. If experimenting with CoT, try dynamic/varied templates or brief hints rather than full static reasoning traces.
3. **Answer format enforcement**: Each category has a strict format (8-bit binary, 2dp decimal, Roman, etc.). The category_specific_reward should penalize format violations heavily.
4. **Easy wins first**: numeral, unit_conv, and gravity should be near-perfect with good prompting. If these aren't scoring high, fix prompting before optimizing harder categories.
5. **Cipher needs character-level reasoning**: The model must build a substitution map character by character. CoT that explicitly constructs the mapping table will help.
6. **Symbol is the long tail**: With arbitrary rules and 36 unique characters, symbol may have a natural accuracy ceiling. Don't over-invest here at the expense of other categories.

## Tips
- The reward functions are the most powerful lever — GRPO learns whatever the rewards incentivize
- Correctness reward dominates; other rewards are auxiliary signals
- SFT warmup matters: it sets the starting point for GRPO exploration
- If GRPO degrades performance vs SFT-only, try reducing `GRPO_LR` or increasing `GRPO_BETA`
- Check per-category accuracy to find which puzzle types need attention
- The model uses `enable_thinking=True` at inference — with `USE_COT=False`, the model fills `<think>` with its own reasoning
- If GRPO crashes with tensor mismatch, this is a known Mamba/MoE + TRL issue — SFT-only is still a valid approach
- If a change crashes, revert and try something smaller
