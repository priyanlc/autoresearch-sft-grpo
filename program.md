# Autoresearch: SFT + GRPO for Nemotron Reasoning Challenge

> **Plan reference:** the strategic plan for this branch is `nemotron-vault/wiki/sft-only-nvfp4-plan.md`. When in doubt about *why* a change is being made, that file is the source of truth. This file (`program.md`) is the operational instruction sheet for the autoresearch agent.

## Active Mode (as of 2026-05-02): SFT-Only on `nvfp4-blackwell`

This branch is being run in **SFT-only mode** (`SKIP_GRPO=True` in `train.py`). GRPO is a known crash on Mamba/MoE + TRL and the SFT path is now the primary research track. Do not flip `SKIP_GRPO=False` unless explicitly testing GRPO loading; the SFT-only mode is the default and the comparison baseline.

### Preconditions (the human handles these before you start)

When the autoresearch agent is invoked, the following are already true. **Confirm them with the commands below; abort and ask if any check fails.** Do not try to fix these yourself — they're the human's responsibility per `wiki/01-runpod-bootstrap.md`.

| Precondition | Verify command | Expected |
|---|---|---|
| cwd is the autoresearch directory | `pwd` | path ends in `/autoresearch-sft-grpo` |
| Branch is `nvfp4-blackwell` | `git rev-parse --abbrev-ref HEAD` | `nvfp4-blackwell` |
| venv is active | `which python` | path contains `.venv/bin/python` |
| `train.csv` is present | `ls data/train.csv` | file exists |
| HF + W&B authenticated | `env \| grep -E '^(HF_TOKEN\|WANDB_API_KEY)='` | both set |

### Pre-flight verification — run before applying Tier 1

Codex review (2026-05-02) flagged the `dtype=torch.bfloat16` kwarg at `train.py:679` as potentially wrong (suggesting `torch_dtype` instead). This is **likely a false positive** — `requirements.txt` pins `transformers>=5.0.0`, where `dtype` is the canonical name and `torch_dtype` is the deprecated alias. **But verify on the actual environment before any T1 work**, because if the env somehow ends up on transformers <5 the kwarg is silently ignored and the model loads at the wrong dtype, blowing the memory budget on the very first run.

**Verification command** — run on the RunPod pod that will execute the training:

```bash
python -c "import transformers, inspect; sig = inspect.signature(transformers.AutoModelForCausalLM.from_pretrained); params = list(sig.parameters.keys()); print(f'transformers={transformers.__version__}'); print('uses dtype' if 'dtype' in params else 'uses torch_dtype' if 'torch_dtype' in params else 'NEITHER — investigate')"
```

**Decision tree:**

| Output | Action |
|---|---|
| `transformers=5.x.y` + `uses dtype` | Current code is correct. No change. Record the verification output in `STATUS.md`. |
| `transformers=4.x.y` + `uses torch_dtype` | The version pin in `requirements.txt` is being violated. **Fix the pin first** (this is part of T1.3) — don't rename the kwarg in `train.py`. |
| `transformers=5.7.0+` + `NEITHER` | **Expected on 5.7.0+** — the signature is now `(model_args, **kwargs)` and both `dtype` / `torch_dtype` are extracted via `kwargs.pop()` inside `PreTrainedModel.from_pretrained` (~line 252 of `modeling_utils.py`). The static-signature check is too narrow for this version. To confirm the kwarg is still wired: `python -c "import transformers, inspect; print('OK' if 'kwargs.pop(\"dtype\"' in inspect.getsource(transformers.PreTrainedModel.from_pretrained) else 'BROKEN')"` — if `OK`, current `train.py:679` works as written; proceed. If `BROKEN`, stop and add a FRICTION entry. (See F-001.) |
| `NEITHER` on a non-5.7.0+ version | Investigate — possibly the install is broken. Halt before T1 and add a `FRICTION.md` entry. |

If verification confirms the env matches the pin, proceed to Tier 1. If it diverges, fix the env via T1.3 first, then re-verify.

### Tier 1 changes — apply before any sweeping

These are one-time correctness/cleanup improvements derived from the latest NVFP4 research. Each lands as its own commit with a `T1.x:` prefix in the commit message.

| ID | Change | Status |
|---|---|---|
| T1.1 | Add `forward_method='quest'` to `FPQuantConfig(...)` (`abs_max` is PTQ-tuned; `quest` is QAT-tuned per HF FP-Quant docs) | TODO — agent applies |
| T1.2 | Replace `for name, mod in sys.modules.items():` Mamba fast-path patch with explicit `import` of the `modeling_nemotron_h` module, then set `is_fast_path_available = False` on it directly | TODO — agent applies |
| T1.3 | Pin all package versions in `requirements.txt` to current working set (`pip freeze` snapshot) | TODO — agent applies |
| T1.4 | Remove duplicate `model.gradient_checkpointing_enable()` call (`SFTConfig(gradient_checkpointing=True)` already covers it). Keep `enable_input_require_grads()` — that's not redundant. | TODO — agent applies |

After T1.1–T1.4 each pass, run end-to-end with `SKIP_GRPO=True` and confirm METRIC ≥ pre-T1 baseline. T1.1 should be applied and validated *first*, alone, before stacking T1.2–T1.4 — it's the change with the most uncertainty about whether it helps the LoRA-on-frozen-NVFP4 case.

### Goal
Maximize **validation accuracy** on 6 types of "Alice's Wonderland" reasoning puzzles via SFT-only training:
1. **SFT** — teaches the model answer format, reasoning patterns, and category-specific solution strategies
2. **GRPO** — *deprioritized*; kept in code path but `SKIP_GRPO=True` is the default

## Metric
The single metric to optimize is printed at the end of `train.py`:
```
METRIC: 0.XXXX
```
This is the proportion of correctly answered puzzles on a held-out validation set (30 samples, 5 per category). Higher is better. Maximum is 1.0.

## Known Baselines

These pre-T1 baselines are **legacy** — the new SFT-only baseline gets captured in `results.tsv` and `STATUS.md` once T1.1–T1.4 land. Tier 2 experiments must beat the post-T1 SFT-only baseline, not these legacy rows.

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

**Tier 2 sweep targets (highest impact — start here after Tier 1 lands):**

These are the pre-identified, research-backed sweep axes. Each variation = one row in `results.tsv` with `description` capturing what was changed. Keep `SKIP_GRPO=True` locked through this phase — mixing GRPO experiments contaminates the comparison.

| ID | Sweep | Values | Why |
|---|---|---|---|
| T2.1 | `MAX_GRAD_NORM` | {0.1 baseline, 0.5, 1.0} | Current 0.1 may over-clip NVFP4 gradients (backward_dtype="bf16" already adds noise) |
| T2.2 | Synthetic ratio | {30%, 50%, 71% baseline} | 71% synth may hurt diversity per `paper-synthetic-data-diversity.md` |
| T2.3 | LoRA targets | {all FPQuantLinear, skip-last-15%-of-layers} | Matches NVIDIA Super recipe — last 15% kept high-precision |
| T2.4 | `MODE` flag | {`direct_nvfp4` baseline, `bf16_then_qat`} | NVIDIA's gpt-oss recipe upcasts before SFT, then QAT. Adds ~10 min startup but is the published path. |
| T2.5 | Loader | {FPQuantConfig baseline, Unsloth} | NVIDIA Unsloth blog claims 2× speed, 70% less VRAM on Blackwell |

**Reward function design (deprioritized — only relevant when GRPO re-enabled):**
- Reward weights, reward function logic, new reward functions, reward shaping
- These live in `train.py` already but only fire when `SKIP_GRPO=False`

**SFT prompt format (medium impact):**
- `USE_COT` flag: toggle static CoT templates on/off (default: False)
- With `USE_COT=True`, model got template parroting — try with more data or varied templates
- With `USE_COT=False`, model uses native reasoning via `enable_thinking=True`
- Category-specific formatting strategies, brief hints (not full reasoning templates)

**Data strategy (medium impact):**
- `SFT_SAMPLES_PER_TYPE`: sample count
- Weighted sampling toward harder categories (bit_ops, symbol)
- Curriculum ordering — easy categories (numeral, unit_conv, gravity) first, then hard

**LoRA configuration (medium impact):**
- `LORA_RANK`: adapter rank (1-32, competition max is 32)
- `LORA_ALPHA`: scaling factor (typically 1x or 2x the rank)
- `LORA_DROPOUT`: must stay 0.0 (FPQuant constraint — see Branch Notes)
- `TARGET_MODULES`: see T2.3 for the principled sweep

**SFT hyperparameters (medium impact):**
- `SFT_LR`: learning rate (1e-5 to 5e-4)
- `SFT_EPOCHS`: number of passes (1-3)
- `SFT_MAX_SEQ_LEN`: max sequence length

**Tier 3 — DO NOT IMPLEMENT YET (scaffolding only):**

These are research bets requiring multi-run setup. The agent may only add the *flag definitions* (default `False`, no-op when off) so future work can enable them without restructuring. Do not implement bodies. Document each new flag in `BRANCH_NOTES.md`.

- `USE_DISTILLATION` — load BF16 teacher (cached logits path)
- `USE_TORCHAO_QAT` — switch loader to TorchAO `NVFP4DynamicActivationNVFP4WeightConfig`
- `USE_GPTQ_PREPROCESS` — pre-process weights before NVFP4 quantization
- `USE_FOUR_OVER_SIX` — adaptive block scaling (arxiv 2512.02010)
- `USE_PEFT_CUSTOM_DISPATCH` — replace `__bases__` hack with PEFT `custom_module_mapping` API (T3.5; verified 2026-05-02 that no upstream `dispatch_fp_quant` exists in PEFT ≤ 0.19.1)

## Constraints
- LoRA rank must be <= 32 (competition rule)
- The model must output answers in `\boxed{}` format
- No per-experiment wall-clock cap — let runs complete on their own terms
- Do not modify `prepare.py`
- SFT and GRPO data should not overlap with validation (handled automatically)
- Keep `SKIP_GRPO=True` for all Tier 1 / Tier 2 work; do not toggle to test GRPO unless explicitly running the co-existence smoke test (see Branch Hygiene)
- `LORA_DROPOUT` must stay `0.0` (FPQuantLinear + PEFT constraint)
- `device_map={'': 0}` is required (FPQuantConfig has no CPU offload)

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

## Logging & Reporting

> **Why this section is non-negotiable:** the human author of this loop is writing a follow-up blog post on the autoresearch pattern. The blog post depends on being able to reconstruct what happened during this run — what was tried, what broke, what stuck, and why — *without* re-deriving it from raw stdout/stderr after the fact. Four artifacts must be kept current. Treat them as part of the contract, not optional decoration.

### 1. `results.tsv` — one row per experiment, including failures

Every `python train.py` invocation appends one row, even if it crashed before reaching the METRIC line. Schema is fixed (header already in the file):

```
commit  metric  bit_ops  cipher  gravity  numeral  symbol  unit_conv  status  description
```

- `commit`: short hash of the `train.py` state that ran (`git rev-parse --short HEAD`)
- `metric`: overall accuracy as `0.XXXX`, or `FAILED` if the run did not produce a METRIC line
- per-category columns: percentage as integer (e.g., `60` for 60%), or empty string for failed runs
- `status`: one of `success` / `failed` / `aborted`
- `description`: ≤120 chars, one-line summary of what was tried this experiment (e.g., `cosine reward, GRPO temp=0.5, +500 synth/cat`)

If a run fails before printing METRIC, still append the row — `status=failed`, `description=` includes the failure signature (e.g., `qutlass build failed: missing CUDA 12.8 headers`).

### 2. `STATUS.md` — append-only progress log

Append a status block every ~40 minutes *and* after any experiment that changed the trajectory. **Do not overwrite prior entries** — STATUS.md is a ledger. Each block:

- **Timestamp** (UTC)
- **Current best METRIC** and per-category breakdown
- **Experiments since last status:** count + 1-line summary of the most informative one
- **What was tried** and whether it helped / hurt / was neutral
- **What you plan to try next**, and the reason
- **Any errors or blockers** — link to the `FRICTION.md` entry id (e.g., "see F-007")

### 3. `FRICTION.md` — structured failure log

When a non-trivial failure occurs (anything that requires more than a one-line config tweak to resolve), add a numbered entry to `FRICTION.md` using the template at the top of that file. Each entry must include:

- **id** (sequential, e.g., `F-007`)
- **timestamp** (UTC)
- **phase**: one of `env_install`, `model_load`, `sft`, `grpo`, `eval`, `cleanup`, `other`
- **signature**: the actual error message or observed symptom (truncate stacktraces to the most informative ~10 lines)
- **hypothesized root cause** — be specific; it's OK to say "uncertain"
- **attempts**: bulleted list of what was tried, each with outcome (`worked` / `no effect` / `made it worse` / `crashed differently`)
- **final state**: `resolved` / `worked-around` / `punted` / `open`

Friction entries are **the most valuable artifact for the follow-up blog post**. Be specific about what was actually tried, even when nothing worked — *especially* when nothing worked. "I tried X, Y, Z and none of them moved it" is publishable; "GRPO didn't work" is not.

### 4. End-of-session summary

Before the loop terminates (out of time, out of disk, or human-stopped), prepend a final block to `STATUS.md` titled `### Session Summary YYYY-MM-DD`:

- Best METRIC achieved + per-category breakdown
- Number of experiments run, with success/failure split
- Top 3 friction items (with `FRICTION.md` ids)
- Open problems / what to try next session
- One-paragraph reflection: what surprised you about this run

This is the section the human reads first when returning to the loop.

## Branch Hygiene

This branch is the source of truth for the SFT-only research track and is referenced by the LinkedIn tutorial draft. Keep `git log` parseable as an experiment record.

- **One commit per change ID.** T1.1 = one commit, T2.3 = one commit. Don't squash.
- **Commit message prefix.** Format: `T1.1: <one-line summary>` so `git log --oneline` matches the tier table.
- **Revert, don't fix-forward, on regressions.** If a Tier 2 experiment hurts METRIC, `git revert` it rather than patching on top — keeps the experiment record honest.
- **`BRANCH_NOTES.md` gets a per-tier section** so anyone reading the branch later sees the chronology and which T-IDs landed.
- **Co-existence smoke test.** After Phase 3 (Tier 2 sweeps complete), run *one* `SKIP_GRPO=False` invocation just to confirm the GRPO loading path still imports and reaches the trainer (it's expected to crash downstream — that's a known issue, not a regression). Catches breakage early in case future GRPO work resumes.

## Validation Contract (every Tier transition)

Each run produces (already wired in `train.py`):
1. **METRIC** — primary number, parsed by autoresearch loop
2. **Per-category accuracy** — diagnoses where gains came from
3. **Peak VRAM + tokens/sec** — for the LinkedIn article's benchmarks section
4. **W&B run URL** — recorded in `STATUS.md`

Plus, **once per Tier transition** (not every run, too expensive):
5. **Adapter-on-BF16 sanity check** — train adapter on NVFP4 base, then load it onto a BF16 base and verify a sample inference works. This is the actual scoring deployment path; most likely silent-break point. Log result in `FRICTION.md` if it fails, even if the run otherwise produced a METRIC.

**Regression bar:** post-T1 SFT-only METRIC ≥ pre-T1 SFT-only METRIC before Tier 2 starts. Tier 1 should be neutral or positive; if negative, identify which of T1.1–T1.4 caused it before adding more variables.

## Tips
- The reward functions are the most powerful lever — GRPO learns whatever the rewards incentivize
- Correctness reward dominates; other rewards are auxiliary signals
- SFT warmup matters: it sets the starting point for GRPO exploration
- If GRPO degrades performance vs SFT-only, try reducing `GRPO_LR` or increasing `GRPO_BETA`
- Check per-category accuracy to find which puzzle types need attention
- The model uses `enable_thinking=True` at inference — with `USE_COT=False`, the model fills `<think>` with its own reasoning
- If GRPO crashes with tensor mismatch, this is a known Mamba/MoE + TRL issue — SFT-only is still a valid approach
- If a change crashes, revert and try something smaller

## NVFP4 Branch Notes

This branch uses NVFP4 quantization on Blackwell GPUs (RTX PRO 6000):

- **Base model ~17GB** instead of ~60GB in bf16 → more VRAM for training
- **FPQuantLinear `__bases__` hack** is required — PEFT doesn't support FPQuantLinear natively. The hack is applied AFTER model loading in main(). Do NOT remove it.
  - *Verified 2026-05-02:* no `dispatch_fp_quant` exists in PEFT ≤ 0.19.1; the hack works because PEFT falls through to its generic `nn.Linear` handler. T3.5 plans a clean `custom_module_mapping` replacement (not yet implemented). See `nemotron-vault/wiki/nvfp4-fine-tuning.md § Upstream status`.
- **`forward_method='quest'`** should be set in `FPQuantConfig` (T1.1) — `abs_max` is PTQ-tuned; `quest` is QAT-tuned per HF FP-Quant docs.
- **`LORA_DROPOUT = 0.0`** is required — FPQuantLinear with PEFT doesn't support dropout
- **`device_map={'': 0}`** is required — FPQuantConfig doesn't support CPU offload
- **Mamba fast-path patch** must be a robust import-then-disable (T1.2) — the `sys.modules.items()` loop is fragile and can silently no-op
- **Gradient checkpointing** is enabled via `SFTConfig(gradient_checkpointing=True)` only (T1.4) — do not also call `model.gradient_checkpointing_enable()`. Keep `model.enable_input_require_grads()` — that's separate and required.
- **Synthetic data** (currently 3000 samples = 71% of mix) provides perfect CoT for all 6 categories. T2.2 plans to sweep this ratio down to ~50% based on synthetic-data-diversity research.
- **Cosine reward** replaces binary correctness — prevents zero-gradient when all completions are correct (relevant only when `SKIP_GRPO=False`)
- **`ground_truth`** column name in GRPO dataset (not `answer`) to avoid TRL conflicts (kept for compatibility even though SFT-only mode doesn't use it)
- If NVFP4 fails, set `USE_NVFP4 = False` to fall back to bf16
- **`requirements.txt` must be version-pinned** (T1.3) — patches above break on minor version bumps of `transformers` / `peft` / `fp_quant` / `qutlass`
- **`qutlass` cannot come from PyPI.** The PyPI name is squatted by an empty stub (version 0.0.0, summary "Temp"). The real CUTLASS-based NVFP4 kernels live at `github.com/IST-DASLab/QuTLASS` and must be installed via `pip install 'git+https://github.com/IST-DASLab/QuTLASS.git' --no-build-isolation`. T1.3 should pin `qutlass @ git+https://github.com/IST-DASLab/QuTLASS.git@<sha>` rather than just `qutlass`. See F-002 for the failure signature when the stub is in place.
