# Autoresearch: SFT-only on `main` (BF16 baseline)

> **Plan reference:** the strategic plan for this branch is [`docs/bf16-sft-only-plan.md`](docs/bf16-sft-only-plan.md). When in doubt about *why* a change is being made, that file is the source of truth. This file (`program.md`) is the operational instruction sheet for the autoresearch agent. The 8-artefact methodology that frames everything else is [`docs/methodology.md`](docs/methodology.md).

## Active Mode (as of 2026-05-04): BF16 SFT-only on `main`

`main` is the parent-of-branches reference. The active mode is **SFT-only**, with GRPO wrapped in a `try/except` block (`train.py:520`) that gracefully falls back to the SFT adapter when the known Mamba/MoE+TRL tensor-mismatch crash fires (`FRICTION.md` F-002). Do not flip the GRPO path to mandatory; do not enable 4-bit/FP8/NVFP4 quantization on `main` — those are research bets that need their own branches (see `nvfp4-blackwell` for the NVFP4 worked example).

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

Three checks. Run on the actual pod that will execute training; abort if any fails. C-1 is a smoke check (trivially passes on A100/H100 — present so a misconfigured pod fails fast); C-2 and C-3 are the load-bearing ones.

```bash
# (C-1) BF16 supported (smoke check)
python -c "import torch; print('OK' if torch.cuda.is_bf16_supported() else 'BF16 NOT SUPPORTED')"

# (C-2) transformers pin matches check_install.py expectation
python -c "import transformers; print(transformers.__version__)"

# (C-3) Nemotron transformers_modules cache (touches F-001 surface)
ls ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/ 2>/dev/null && echo "cache present" || echo "cache missing"
```

**Decision tree:**

| Check | Output | Action |
|---|---|---|
| C-1 | `OK` | Proceed |
| C-1 | `BF16 NOT SUPPORTED` | The GPU lacks BF16 (rare on A100/H100). STOP and add a `FRICTION.md` entry — do not switch dtype on `main`; switch hardware or branch off |
| C-1 | anything else (CUDA init error, etc.) | STOP and add a `FRICTION.md` entry |
| C-2 | `4.51.3` | Matches the requirements pin and `check_install.py:83`. Proceed |
| C-2 | any other version | The pin is being violated. Reinstall via `pip install -r requirements.txt`; do **not** rename or relax the kwarg in `train.py` |
| C-3 | `cache present` followed by directory listing | OK — the model has been downloaded before. F-001's in-place `modeling_nemotron_h.py` edits (if applied this pod-life) should still be in place; verify via `grep "conv_kernel_size" ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/*/modeling_nemotron_h.py` if the run will use `use_cache=True` (it shouldn't on `main`) |
| C-3 | `cache missing` | First run on this pod — `python prepare.py` must run before `python train.py` (will download ~60 GB; ~5 min cold load). Not an error |

If any check hits a "neither expected outcome" branch, **STOP and add a `FRICTION.md` entry** rather than guessing — the verification command can itself become stale (this is the lesson behind the original F-001 pattern on `nvfp4-blackwell`).

### Tier 1 changes — chronology

| ID | Change | Status |
|---|---|---|
| T1.1 | Add `FRICTION.md` template + conventions | Landed 2026-05-03 |
| T1.2 | Add `BRANCH_NOTES.md` describing main as BF16 SFT-only anchor | Landed 2026-05-03 |
| T1.3 | Add `runpod-setup.md` three-part scaffold for BF16 main | Landed 2026-05-03 |
| T1.3a | Fix critical bugs in `runpod-setup.md` Part 1 (source-built packages claim, stale line number) | Landed 2026-05-04 |
| T1.4 | Restructure `program.md` to 14-section methodology template | Landed 2026-05-03 (this file) |
| T1.5 | Convert `STATUS.md` to append-only ledger | Landed 2026-05-03 |
| T1.6 | Seed `FRICTION.md` with F-001..F-006 from main history | Landed 2026-05-03 |
| T1.7 | Cross-reference FRICTION ids in `train.py` | Landed 2026-05-03 |
| T1.7a | Fix stale `train.py` line numbers across docs (12 citations) | Landed 2026-05-04 |
| T1.8a | Mark legacy baselines + heartbeat noting T1.8b deferred | Landed 2026-05-03 |
| T1.8b | Pod regression run + first heartbeat under new format | **PENDING** — requires A100 80GB; defer until next RunPod session |
| T1.9 | Comment out `causal_conv1d` in `requirements.txt` (currently inert per F-001 workaround) | Landed 2026-05-04 — **superseded by T1.14** (mental model was wrong: transformers' static AST check_imports rejects the conditional import even when the runtime fast path is disabled) |
| T1.10 | Correctness sweep on `program.md` + `BRANCH_NOTES.md` (Patches table rationale, Tier 1 chronology, Validation Contract holdover, pre-flight expansion) | Landed 2026-05-04 |
| T1.14 | Restore `causal_conv1d` as a hard install-time dep (revert T1.9) + sweep all referencing docs (`requirements.txt`, `bootstrap.sh`, `check_install.py`, `prompt.md`, `program.md`, `BRANCH_NOTES.md`, `runpod-setup.md`, `docs/fast-path-and-cache.md`). Closes FRICTION F-009. | Landed 2026-05-06 |
| T1.15 | Log F-010 (hf_xet worker→main thread deadlock during weights download). Workaround: `unset HF_XET_HIGH_PERFORMANCE` before training, keep `HF_HUB_ENABLE_HF_TRANSFER=1`. Not yet promoted into setup docs (per F-010 § notes — wait for second reproduction). | Landed 2026-05-07 |
| T1.16 | Post-T1.15 baseline-restoration run: METRIC 0.5667 on fresh A100 80GB pod (+0.0334 vs c1bb0a6). Adapter sanity check on fresh BF16 base passed. Heartbeat block + Session Summary 2026-05-07 prepended to STATUS.md; first row appended to results.tsv; new `adapter_sanity_check.py` helper script added. | Landed 2026-05-07 |
| T1.17 | Add `hf_transfer` to `requirements.txt` (closes F-008) + log cost/impact of the `use_cache=False` workaround in F-001 (eval ~10 min → ~3 h ≈ 18×, GRPO rollout ~1 min → ~30 min, run ~1.5 h → ~4.5 h ≈ 3×; quantifies what F-001 actually costs and how it ties into F-002 GRPO economics). | Landed 2026-05-07 |
| T1.18 | Capture T2.7 run results — METRIC 0.5333 (regression vs T1.16 0.5667). STATUS.md heartbeat + results.tsv row appended. T2.7 reverted per branch hygiene "revert on regression". | Landed 2026-05-07 |
| T1.19 | Capture T2.8 run results — METRIC 0.6000 (≥ T1.16 0.5667; the F-011 dynamic-CoT regex fix produced gravity 1/5 → 5/5). STATUS.md heartbeat + Session Summary + results.tsv row + new F-012 (adapter_sanity_check.py hung). | Landed 2026-05-07 |
| T1.20 | README.md doc-only refresh: headline METRIC 0.5333 → 0.6000 with locked-floor pointer; sanity-check refs repointed to canonical `adapter_sanity_check.py`; agent-handover regression-bar reworded; Tier 2 progress note added; repo-layout block extended. | Landed 2026-05-09 |
| T1.21 | Deduplicate sanity-check helpers — drop older `sanity_check.py` (predates T1.16); keep canonical `adapter_sanity_check.py`. | Landed 2026-05-09 |
| T1.22 | Sync `prompt.md` regression-bar wording with the post-T2.8 README. Threshold unchanged at 0.5333; agent behaviour preserved. | Landed 2026-05-09 |
| T1.23 | `train.py` line-number citation sweep across docs (T1.7a-style). T2.8's `_build_dynamic_cot` additions shifted ~9 lines in the canonical patch locations; updated `:386 → :398`, `:536 → :545`, `:511 → :520`, `:383 → :392`, plus block ranges. | Landed 2026-05-09 |
| T1.24 | Stale METRIC wording sweep across remaining docs (`runpod-setup.md`, `docs/autoresearch-handoff.md`, `docs/bf16-sft-only-plan.md`, `README.md` "Where to read next"). Floor 0.5333 / current 0.6000 framing applied uniformly. | Landed 2026-05-09 |
| T1.25 | program.md Tier 1 chronology fill-in (T1.18..T1.24 rows) + regression-bar phrasing update ("before Tier 2 starts" → floor=revert target / current=de-facto bar for new Tier 2). T1.8b status reconciled — effectively satisfied by T1.16 baseline-restoration run on a fresh A100 80GB pod. | Landed 2026-05-09 |
| T1.26 | `README.md` § "Handover" simplification: collapsed the two-paragraph agent-handover block into a single tighter paragraph (~50% shorter); load-bearing facts preserved (locked floor 0.5333, current best 0.6000, Tier 2 already landed → `STATUS.md`). | Landed 2026-05-09 |
| T1.27 | `README.md` § "Tiers" section added between Quickstart and the agent-handover section so "Tier 2 sweep" is defined before it is used. Mirrors `docs/methodology.md` § "Tiered work — T1 / T2 / T3"; cites T2.7/T2.8 outcomes as concrete example. | Landed 2026-05-09 |
| T1.28 | Surface `bootstrap.sh` in `README.md` Quickstart + finish T1.23's missed citations (one in `bootstrap.sh:31`, three in `requirements.txt` lines 20/23/30). Reframed README install commentary from "if mamba_ssm fails" (fallback) to canonical pair. | Landed 2026-05-09 |
| T1.29 | Doc citation + content sweep after T2.9 moved the fast-path disable from pre-SFT to pre-eval. Bumped `train.py:398 → :582` and `train.py:545 → :579` across nine files (`bootstrap.sh`, `requirements.txt`, `check_install.py`, `adapter_sanity_check.py`, `runpod-setup.md`, `prompt.md`, `BRANCH_NOTES.md`, `FRICTION.md`, `docs/fast-path-and-cache.md`) plus the Patches tables in `BRANCH_NOTES.md`/`program.md`. Reworded the "loaded but never called" / "harmless at runtime" claims; new framing: kernels run during SFT, disabled only before eval. Also backfilled this table with T1.26..T1.29. | Landed 2026-05-30 |
| T1.30 | Promote F-007 (MooseFS wedges wheel extraction) into `runpod-setup.md` § 2 as a `df -T /workspace` pre-flight + standard-path / MooseFS-path decision tree. Second reproduction triggered the promotion per F-007 § notes. Existing `bootstrap.sh` / `causal_conv1d` prose preserved. | Landed 2026-05-30 |
| T1.31 | Promote F-010 (`hf_xet` worker→main deadlock) into `runpod-setup.md` § 5 as a one-line `unset HF_XET_HIGH_PERFORMANCE` guard before `python prepare.py`. Preemptive (no second reproduction observed this session) at user request — cost/benefit logged in F-010 § notes. Autonomous-flow note added: unset must precede `claude` launch inside tmux. | Landed 2026-05-30 |

T1.1..T1.7 made no `train.py` logic changes (T1.7 is comment-only). T1.3a/T1.7a/T1.9/T1.10/T1.20/T1.21/T1.22/T1.23/T1.24/T1.25 are all doc/config corrections that touch no `train.py` logic either. T1.8b's "post-T1 regression run" was effectively satisfied by T1.16, which produced METRIC 0.5667 on a fresh A100 80GB pod after the T1.14 dep restoration — the original "prove the methodology assimilation didn't break anything" intent is met. The PENDING marker on T1.8b is preserved as a methodological artefact: the run wasn't framed *as* T1.8b at the time.

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

## Known Baselines (legacy — pre-T1)

The table below captures the pre-T1.1 state of `main` (commit `c1bb0a6`). T1.1..T1.7 made no `train.py` logic changes (T1.7 was comment-only), so the 0.5333 row is expected to reproduce post-T1; the post-T1 regression run lands as T1.8b on the next pod session.

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
- **Co-existence smoke test.** GRPO smoke run is expected to crash; that's F-002, not a regression. `main` has no `SKIP_GRPO` flag — the standing `try/except` block at `train.py:520` IS the smoke harness, and every `python train.py` exercises it for free.

## Validation Contract (every Tier transition)

Each run produces (already wired in `train.py`):

1. **METRIC** — primary number, parsed by autoresearch loop
2. **Per-category accuracy** — diagnoses where gains came from
3. **Peak VRAM + tokens/sec** — hardware reality for the eventual write-up's benchmarks section
4. **W&B run URL** — recorded in `STATUS.md`

Plus, **once per Tier transition** (not every run, too expensive):

5. **Adapter-on-fresh-BF16-base sanity check** — load adapter onto a fresh BF16 base from a *separate Python process* (not the same process that just trained) and verify a sample inference works. This is the actual scoring deployment path; most likely silent-break point. Log result in `FRICTION.md` if it fails, even if the run otherwise produced a METRIC.

**Regression bar:** SFT-only METRIC ≥ locked floor 0.5333 (commit `c1bb0a6`) on every run that touches `train.py` logic. The locked floor is the revert target — `git revert` returns to it. The current best on `main` is the de-facto bar for *new* Tier 2 sweeps (currently 0.6000 at T2.8, `c4a9d1c` — see `STATUS.md` and `results.tsv`); a sweep that lands above the floor but below the current best is an inconclusive run, not a regression, but it does not raise the de-facto bar. If a run regresses below the floor, identify which T-id caused it before adding more variables.

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

`main` carries no FPQuant/PEFT patches (no NVFP4 stack on this branch). Two non-trivial workarounds in `train.py` together defend FRICTION F-001:

| Location | Patch | Defends against |
|---|---|---|
| `train.py:582` | Mamba fast-path disable (before eval): loops `sys.modules` for `modeling_nemotron_h` and sets `is_fast_path_available = False` | F-001. Forces pure-PyTorch math even where the fused CUDA kernels would otherwise run. Pairs with the `use_cache=False` patch below. |
| `train.py:579` | `model.config.use_cache = False` before eval | F-001. Prevents generation from touching the broken `HybridMambaAttentionDynamicCache`. |

These are **redundant defenses, not duplicates** — both are needed. The fast path won't even be selected if `use_cache=False`, but if `use_cache=True` is accidentally re-enabled somewhere downstream and the fast-path disable is removed, you'd hit F-001 from a different angle. Post-T2.9 the pair is applied as a single block right before eval; the fast path stays ON during SFT (teacher-forced forward never touches the cache). See [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) for the full mechanical treatment.

Inline `# See FRICTION.md F-NNN` comments are added in T1.7. The Patches table grows when (and only when) a future fix on `main` defends a specific FRICTION entry.

## Operational realities

Pod-specific quirks captured here so the next operator (human or agent) doesn't repeat the wasted hours.

- Cold model load on A100 80GB: ~5 min (13 shards from local cache). Faster than the `nvfp4-blackwell` MooseFS pod (which paid ~22 min on first load) because `main` doesn't typically run on RunPod EU.
- SFT (1200 samples, 1 epoch): ~1 hour.
- Eval (30 samples, no KV cache per F-001): ~3 hours. **This is the wall-clock dominant phase.** KV cache rehabilitation would cut eval to ~10 min, but is out of scope on `main`.
- Total per run: ~4–5 hours.
- Peak VRAM: ~78 GB / 80 GB. Headroom is thin; raising `BATCH_SIZE` or `SFT_MAX_SEQ_LEN` requires explicit Tier 2 work.
- `.git/index.lock` from concurrent shell sessions cost a session in the past (FRICTION F-006). Pre-flight `ls .git/index.lock` before any commit.
