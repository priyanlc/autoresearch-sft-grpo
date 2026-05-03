# Autoresearch: SFT-only on `main` (BF16 baseline)

> **Plan reference:** the strategic plan for this branch is [`nemotron-vault/wiki/bf16-sft-only-plan.md`](../../../nemotron-vault/wiki/bf16-sft-only-plan.md). When in doubt about *why* a change is being made, that file is the source of truth. This file (`program.md`) is the operational instruction sheet for the autoresearch agent.

## Active Mode (as of 2026-05-03): BF16 SFT-only on `main`

`main` is the parent-of-branches reference. The active mode is **SFT-only**, with GRPO wrapped in a `try/except` block (`train.py:506`) that gracefully falls back to the SFT adapter when the known Mamba/MoE+TRL tensor-mismatch crash fires (`FRICTION.md` F-002). Do not flip the GRPO path to mandatory; do not enable 4-bit/FP8/NVFP4 quantization on `main` — those are research bets that need their own branches (see `nvfp4-blackwell` for the NVFP4 worked example).

### Preconditions (the human handles these before you start)

When the autoresearch agent is invoked, the following are already true. **Confirm them with the commands below; abort and ask if any check fails.** Do not try to fix these yourself — they're the human's responsibility per `runpod-setup.md` Part 1.

| Precondition | Verify command | Expected |
|---|---|---|
| cwd is the autoresearch directory | `pwd` | path ends in `/autoresearch-sft-grpo` |
| Branch is `main` | `git rev-parse --abbrev-ref HEAD` | `main` |
| venv is active | `which python` | path contains `.venv/bin/python` |
| `train.csv` is present | `ls data/train.csv` | file exists |
| HF + W&B authenticated | `env \| grep -E '^(HF_TOKEN\|WANDB_API_KEY)='` | both set |

### Pre-flight verification — run before any Tier 2 sweep

Two checks. Run on the actual pod that will execute training; abort if either fails.

```bash
# (C-1) BF16 supported
python -c "import torch; print('OK' if torch.cuda.is_bf16_supported() else 'BF16 NOT SUPPORTED')"

# (C-2) transformers pin matches check_install.py expectation
python -c "import transformers; print(transformers.__version__)"
```

**Decision tree:**

| Check | Output | Action |
|---|---|---|
| C-1 | `OK` | Proceed |
| C-1 | `BF16 NOT SUPPORTED` | The GPU lacks BF16 (rare on A100/H100). STOP and add a `FRICTION.md` entry — do not switch dtype on `main`; switch hardware or branch off |
| C-1 | anything else (CUDA init error, etc.) | STOP and add a `FRICTION.md` entry |
| C-2 | `4.51.3` | Matches the requirements pin and `check_install.py:83`. Proceed |
| C-2 | any other version | The pin is being violated. Reinstall via `pip install -r requirements.txt`; do **not** rename or relax the kwarg in `train.py` |

If C-1 or C-2 hits a "neither expected outcome" branch, **STOP and add a `FRICTION.md` entry** rather than guessing — the verification command can itself become stale (this is the lesson behind the original F-001 pattern on `nvfp4-blackwell`).

### Tier 1 changes — chronology

| ID | Change | Status |
|---|---|---|
| T1.1 | Add `FRICTION.md` template + conventions | Landed 2026-05-03 |
| T1.2 | Add `BRANCH_NOTES.md` describing main as BF16 SFT-only anchor | Landed 2026-05-03 |
| T1.3 | Add `runpod-setup.md` three-part scaffold for BF16 main | Landed 2026-05-03 |
| T1.4 | Restructure `program.md` to 14-section methodology template | Landed 2026-05-03 (this file) |
| T1.5 | Convert `STATUS.md` to append-only ledger | Landed 2026-05-03 |
| T1.6 | Seed `FRICTION.md` with F-001..F-006 from main history | Landed 2026-05-03 |
| T1.7 | Cross-reference FRICTION ids in `train.py` | Landed 2026-05-03 |
| T1.8a | Mark legacy baselines + heartbeat noting T1.8b deferred | Landed 2026-05-03 |
| T1.8b | Pod regression run + first heartbeat under new format | **PENDING** — requires A100 80GB; defer until next RunPod session |

T1.1..T1.7 made no `train.py` logic changes (T1.7 is comment-only). The regression risk for T1.8b is therefore near-zero, but full methodology compliance still requires it.

## Goal

Maximize **validation accuracy** on 6 types of "Alice's Wonderland" reasoning puzzles via SFT-only training:

1. **SFT** — teaches the model answer format, reasoning patterns, and category-specific solution strategies.
2. **GRPO** — *deprioritized*; kept in code path but expected to crash gracefully (FRICTION F-002).

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
| **A100 (bf16, locked)** | **1,200** | **failed gracefully** | **True** | **0.5333** | **60%** | **0%** | **20%** | **100%** | **40%** | **100%** |

### Observations

- `numeral` and `unit_conv` are solved — focus elsewhere.
- `cipher` and `gravity` score 0% / 20% — debug output shows correct format but wrong answers.
- With `USE_COT=False`, the model parroted thinking but never closed with `\boxed{}` (METRIC 0.1667) — see FRICTION F-004.
- GRPO crashes on Mamba/MoE + TRL tensor mismatch — see FRICTION F-002. SFT-only fallback via try/except.

## What you can modify

You may ONLY edit `train.py`. Everything in `prepare.py` is read-only.

### Tier 2 sweep targets — research bets, sweep one axis at a time

**Reward function design (highest impact for any future GRPO work; lower priority while GRPO is punted):**

- Reward weights: `W_CORRECTNESS`, `W_FORMAT`, `W_REASONING`, `W_CATEGORY_BONUS`
- Reward function logic: modify how correctness, format, reasoning, and category rewards are computed
- Add new reward functions (e.g., answer-length penalty, confidence calibration)
- Reward shaping: partial credit for near-correct answers

**SFT prompt format (high impact):**

- `USE_COT` flag is **locked** on `main` — see FRICTION F-004 / `BRANCH_NOTES.md`. Do not flip without a Tier 3 promotion.
- Category-specific formatting strategies, brief hints (not full reasoning templates)
- Dynamic CoT (per-example derivation injected into `<think>`) — the in-progress 2026-04-06 task `bdxqkblt5` was such an attempt; consider promoting from Tier 3 to Tier 2 if hard categories remain stuck.

**Data strategy (high impact):**

- `SFT_SAMPLES_PER_TYPE` (currently 200; sweep 100–400) and `GRPO_SAMPLES_PER_TYPE`
- Weighted sampling toward harder categories (cipher, gravity, symbol)
- Curriculum ordering

**LoRA configuration (medium impact):**

- `LORA_RANK` ≤ 32 (competition cap)
- `LORA_ALPHA` (typically 1× or 2× the rank)
- `LORA_DROPOUT` (currently 0.05; sweep 0.0, 0.05, 0.1)
- `TARGET_MODULES` subset (currently `all-linear`)

**SFT hyperparameters (medium impact):**

- `SFT_LR` (1e-5 to 5e-4)
- `SFT_EPOCHS` (1 to 3)
- `SFT_MAX_SEQ_LEN`

### Tier 3 — DO NOT IMPLEMENT YET (scaffolding only)

- `USE_DYNAMIC_COT` flag (default `False`) — promote to Tier 2 when hard categories warrant it.
- `USE_OVERSAMPLE_HARD` flag (default `False`) — promote when stratified sampling is shown insufficient.
- `USE_CURRICULUM` flag (default `False`) — promote when sample-count sweeps plateau.

When promoting Tier 3 → Tier 2, write the implementation, do *not* expand scope. The scaffolding flag stays as the on/off switch.

## Constraints

- LoRA rank must be ≤ 32 (competition rule).
- The model must output answers in `\boxed{}` format.
- Training must complete within the time budget (~5 hr total per run on A100 80GB).
- Do not modify `prepare.py`.
- SFT and GRPO data should not overlap with validation (handled automatically).
- **Do NOT enable GRPO as mandatory.** It's punted (FRICTION F-002); the try/except is the smoke harness, not a regression.
- **Do NOT switch dtype away from `bfloat16`.** 4-bit degrades 0.5333 → 0.1333 (FRICTION F-003).
- **VRAM ceiling:** peak ~78 GB / 80 GB on A100. Don't push past with bigger batch / longer seq without explicit Tier 2 work.

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
- `description`: ≤120 chars, one-line summary of what was tried this experiment (e.g., `+200 cipher samples, USE_COT=True, SFT_LR=1e-4`)

If a run fails before printing METRIC, still append the row — `status=failed`, `description=` includes the failure signature (e.g., `OOM at SFT step 12; reduced BATCH_SIZE`).

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

**FRICTION conventions:**

- **Read FRICTION.md before applying patches.** F-001 through F-006 already document the failure modes that the current `train.py` patches defend against (see "Patches in `train.py` — DO NOT REMOVE" below). If you find yourself reaching for a workaround that "feels familiar," check whether it's already in FRICTION first.
- **Sequential IDs only** — next entry is `F-007`, then `F-008`, etc. Don't re-use IDs even if a prior issue was later resolved.
- **Cross-reference F-IDs everywhere.** Commit messages: `fix: F-007 PEFT crash on adapter merge`. STATUS.md blocks: `Blockers: F-007, F-008`. `train.py` inline comments: `# See FRICTION.md F-006: ...`. Three-way linkage (code ↔ commits ↔ STATUS) is what makes the run-log reconstructible months later.
- **Open ≠ stuck.** A FRICTION entry can be `final state: open` while you proceed past it with a workaround. The point is to capture what you learned, not to gate progress on a clean resolution.

### 4. End-of-session summary

Before the loop terminates (out of time, out of disk, or human-stopped), prepend a final block to `STATUS.md` titled `### Session Summary YYYY-MM-DD`:

- Best METRIC achieved + per-category breakdown
- Number of experiments run, with success/failure split
- Top 3 friction items (with `FRICTION.md` ids)
- Open problems / what to try next session
- One-paragraph reflection: what surprised you about this run

This is the section the human reads first when returning to the loop.

## Branch Hygiene

`main` is the source of truth for the BF16 SFT-only baseline and is referenced by other branches' `BRANCH_NOTES.md`. Keep `git log` parseable as an experiment record.

- **One commit per change ID.** T1.1 = one commit, T2.3 = one commit. Don't squash.
- **Commit message prefix.** Format: `T1.1: <one-line summary>` so `git log --oneline` matches the tier table.
- **Revert, don't fix-forward, on regressions.** If a Tier 2 experiment hurts METRIC, `git revert` it rather than patching on top — keeps the experiment record honest.
- **`BRANCH_NOTES.md` gets a per-tier section** so anyone reading the branch later sees the chronology and which T-IDs landed.
- **Co-existence smoke test.** GRPO smoke run is expected to crash; that's F-002, not a regression. The standing `try/except` block at `train.py:506` is the smoke harness — every `python train.py` exercises it. No additional `SKIP_GRPO=False` invocation needed.

## Validation Contract (every Tier transition)

Each run produces (already wired in `train.py`):

1. **METRIC** — primary number, parsed by autoresearch loop
2. **Per-category accuracy** — diagnoses where gains came from
3. **Peak VRAM + tokens/sec** — for the LinkedIn article's benchmarks section
4. **W&B run URL** — recorded in `STATUS.md`

Plus, **once per Tier transition** (not every run, too expensive):

5. **Adapter-on-fresh-BF16-base sanity check** — load adapter onto a fresh BF16 base from a *separate Python process* (not the same process that just trained) and verify a sample inference works. This is the actual scoring deployment path; most likely silent-break point. Log result in `FRICTION.md` if it fails, even if the run otherwise produced a METRIC.

**Regression bar:** post-T1 SFT-only METRIC ≥ pre-T1 SFT-only METRIC (0.5333) before Tier 2 starts. Tier 1 should be neutral or positive; if negative, identify which T1.x caused it before adding more variables.

## Tips

- The reward functions are the most powerful lever for any future GRPO work — but GRPO is punted, so reward changes don't help on `main` in current state.
- Correctness reward dominates; other rewards are auxiliary signals.
- SFT warmup matters: it sets the starting point for any future GRPO exploration.
- Check per-category accuracy to find which puzzle types need attention.
- The model uses `enable_thinking=True` at inference — with `USE_COT=True` (locked), SFT teaches the think→`\boxed{}` closing pattern.
- If a change crashes, revert and try something smaller.
- If anything novel breaks, **FRICTION.md is the first place to look and the first place to write.**

## Branch Notes

See [`BRANCH_NOTES.md`](BRANCH_NOTES.md) for the locked configuration table, hardware profile, and Tier 1 chronology.

## Patches in `train.py` — DO NOT REMOVE

`main` carries no FPQuant/PEFT patches (no NVFP4 stack on this branch). The two non-trivial workarounds in `train.py` are:

| Location | Patch | Defends against |
|---|---|---|
| `train.py` ~L384 | Mamba fast-path disable (predates assimilation) | Mamba kernel selection issues on the BF16 base; harmless. |
| `train.py:530` | `model.config.use_cache = False` before eval | FRICTION F-001 (`HybridMambaAttentionDynamicCache` bugs) |

Inline `# See FRICTION.md F-NNN` comments are added in T1.7. The Patches table grows when (and only when) a future fix on `main` defends a specific FRICTION entry.

## Operational realities

Pod-specific quirks captured here so the next operator (human or agent) doesn't repeat the wasted hours.

- Cold model load on A100 80GB: ~5 min (13 shards from local cache). Faster than the `nvfp4-blackwell` MooseFS pod (which paid ~22 min on first load) because `main` doesn't typically run on RunPod EU.
- SFT (1200 samples, 1 epoch): ~1 hour.
- Eval (30 samples, no KV cache per F-001): ~3 hours. **This is the wall-clock dominant phase.** KV cache rehabilitation would cut eval to ~10 min, but is out of scope on `main`.
- Total per run: ~4–5 hours.
- Peak VRAM: ~78 GB / 80 GB. Headroom is thin; raising `BATCH_SIZE` or `SFT_MAX_SEQ_LEN` requires explicit Tier 2 work.
- `.git/index.lock` from concurrent shell sessions cost a session in the past (FRICTION F-006). Pre-flight `ls .git/index.lock` before any commit.
