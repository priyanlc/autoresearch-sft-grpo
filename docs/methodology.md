---
title: Autoresearch Run Methodology — Reusable Patterns
type: ops
status: active
related: [
  ../program.md,
  ../runpod-setup.md,
  ../STATUS.md,
  ../FRICTION.md,
  ../BRANCH_NOTES.md,
  ../build.log,
  ../results.tsv,
  ../check_install.py,
  bf16-sft-only-plan.md,
  fast-path-and-cache.md,
]
---

> **Origin note:** Originally drafted in a separate documentation vault as `wiki/04-autoresearch-methodology.md`. This in-repo copy at `docs/methodology.md` is the canonical version going forward — vault and repo will drift, repo wins.

# Autoresearch Run Methodology — Reusable Patterns

This page captures the **reusable conventions** that accumulated on the `nvfp4-blackwell` branch of `autoresearch-sft-grpo` between 2026-04-30 and 2026-05-02. The intent is to make it cheap to lift the *process* onto another branch (or another repo) without re-deriving it. The branch-specific content (NVFP4 patches, FPQuant friction, the Nemotron Reasoning Challenge) stays where it is; this page extracts only the methodology.

If you're starting a new autoresearch loop on a different problem, read the **Transferring to a new branch — checklist** section at the end first. The rest of the page explains *why* each piece exists.

## Why this methodology exists

Karpathy's autoresearch pattern is small in code (`read program.md → edit train.py → run → parse METRIC → git keep/revert → repeat`) but the surrounding *operational discipline* is what makes it produce article-grade artefacts and not a pile of unreproducible logs. Across one branch on one problem, eight artefacts ended up doing distinct, complementary jobs — and a handful of cross-cutting conventions kept them in sync. They are the reusable parts.

## TL;DR — the 8-artefact set

| # | Artefact | Role | Reusability class |
|---|---|---|---|
| 1 | `program.md` | Contract for the autoresearch agent: goal, metric, knobs, constraints, logging requirements | Skeleton-reusable; content branch-specific |
| 2 | `runpod-setup.md` | Three-part human onboarding: minimal first-run / autoresearch operator / troubleshooting | Skeleton-reusable; some bash verbatim-reusable |
| 3 | `BRANCH_NOTES.md` | Per-branch config delta from the parent branch + branch-only requirements | Pattern reusable; content always branch-specific |
| 4 | `STATUS.md` | Append-only heartbeat log + end-of-session summaries | Scaffold reusable; entries always run-specific |
| 5 | `FRICTION.md` | Structured failure log with template + numbered entries (`F-001`, `F-002`, …) | Template fully reusable; entries run-specific |
| 6 | `results.tsv` | Per-experiment metric ledger including failed runs | Schema fully reusable; rows run-specific |
| 7 | `build.log` | Step-by-step install/build provenance for one fresh-pod bootstrap | Pattern reusable; written once per environment |
| 8 | `check_install.py` | Automated pre-flight: dependencies + GPU + data + version checks | Skeleton reusable; specific imports branch-specific |

A single rule binds them: **if the agent is doing it, it must be reconstructible later from disk alone**. No information should live only in stdout, in transcripts, or in the human's head.

## Artefact details

### 1. `program.md` — the agent contract

The single most important file. Treat it as the system prompt for the autoresearch agent. The reusable structure is:

```
## Plan reference (link to wiki/<plan>.md — the why-doc)
## Active Mode (current branch's primary mode, e.g., "SFT-only on <branch>")

### Preconditions (the human handles these — agent confirms, doesn't fix)
  - cwd
  - branch
  - venv active
  - data files present
  - tokens / API keys set

### Pre-flight verification (decision-tree, see "Pre-flight verification pattern" below)

### Tier 1 changes — apply before any sweeping
  - One-time correctness/cleanup commits (`T1.1`, `T1.2`, …) with status

## Goal
## Metric (the single scalar autoresearch parses from stdout)
## Known Baselines (legacy table; mark as legacy after T1 lands)

## What you can modify (file scope) + Things worth trying (priority-ordered)
  - **Tier 2 sweep targets** (the actual research bets, table form)
  - **Tier 3 — DO NOT IMPLEMENT YET** (scaffolding-only flags for future work)

## Constraints (LoRA rank cap, output format, VRAM ceiling, dropout=0, etc.)

## Logging & Reporting (the 4-artifact contract — see "Logging contract" section below)
## Branch Hygiene (commit prefix convention, revert-don't-fix-forward)
## Validation Contract (per Tier transition)
## Tips
## NVFP4 Branch Notes (or whatever the active branch's quirks are)
## Patches in `train.py` — DO NOT REMOVE (table mapping each patch → FRICTION id)
## Operational realities (pod-specific quirks, e.g., cold model load time)
```

The Tiered work, Pre-flight verification, Logging contract, and Validation contract sub-sections are reusable verbatim — see their dedicated sections below. Everything else is branch-specific in content but reusable in shape.

**Source of truth:** `notebooks/05-autoresearch/autoresearch-sft-grpo/program.md` on `nvfp4-blackwell`.

### 2. `runpod-setup.md` — three-part human onboarding

Splits the pod-side setup into three distinct audiences:

| Part | Audience | Goal |
|---|---|---|
| 1. Minimal first-run | Anyone who wants a `METRIC:` line out of a fresh pod | Single-page, ~7-step recipe |
| 2. Autoresearch operator setup | Someone who'll run `claude --dangerously-skip-permissions` overnight | Non-root user, Claude Code install, kickoff prompt |
| 3. Troubleshooting & appendix | Someone hitting a known issue (version pin conflict, cache poisoning, …) | Symptoms → fixes |

**Pod requirements table** at the top (GPU generation, VRAM floor, CUDA version, disk floor) — non-negotiable preconditions for the branch's tech stack. If the pod doesn't match, no patch downstream rescues it.

**Reusable verbatim:**
- The non-root-user creation + Claude Code launch flow (Part 2)
- The "skipping the non-root user" advanced section (`IS_SANDBOX=1` and one-liner shortcut, with their risk notes)
- The "stale `transformers_modules` cache" recovery `rm -rf` recipe
- The pod-recycling note (often cheaper to spin a fresh pod than to fix a broken one)

**Branch-specific:**
- The exact pip-install staging order (depends on the branch's source-built packages)
- Pre-flight Python one-liners (depend on which APIs the train script uses)

**Source of truth:** `notebooks/05-autoresearch/autoresearch-sft-grpo/runpod-setup.md` on `nvfp4-blackwell`. See also `wiki/01-runpod-bootstrap.md` for the higher-level handoff doc.

### 3. `BRANCH_NOTES.md` — per-branch delta

Lives at the repo root on every active research branch. Captures **what makes this branch different from main** — a configuration table is the typical core. Examples of what belongs here:

- Quantization mode (NVFP4 / FP8 / BF16)
- Which patches `train.py` carries
- Required version constraints (e.g., `LORA_DROPOUT=0.0`)
- Required hardware (Blackwell-only, etc.)
- Synthetic data ratio
- Reward shape (cosine vs binary)

The pattern is reusable; the contents are always branch-specific by definition.

**Convention:** when a Tier 1 change lands, add a sub-section below the config table noting the change ID, what it did, and which `FRICTION.md` entry (if any) it resolved. This makes `BRANCH_NOTES.md` a chronology readers can follow long after the branch ships.

**Source of truth:** `notebooks/05-autoresearch/autoresearch-sft-grpo/BRANCH_NOTES.md` on `nvfp4-blackwell`.

### 4. `STATUS.md` — append-only run log

Two-mode use:

1. **Heartbeat blocks** — every ~40 minutes during the loop, *and* after any experiment that changed the trajectory. Newest at the top. Each block: timestamp, current best METRIC + per-category, experiments since last status, what was tried + helped/hurt/neutral, what's next + why, blockers (with FRICTION ids).
2. **Session summary** — prepended before the loop terminates. Heading: `### Session Summary YYYY-MM-DD`. Contains: best METRIC, experiment count + success/failure split, top 3 friction items by id, open problems, one-paragraph reflection.

**Critical constraint:** STATUS.md is a **ledger, never a snapshot**. Don't rewrite prior entries. The agent appends; the human reads top-down.

**Reusable verbatim:** the scaffold file with header, conventions, and an HTML-comment block holding the per-block template.

**Source of truth:** `notebooks/05-autoresearch/autoresearch-sft-grpo/STATUS.md` on `nvfp4-blackwell`.

### 5. `FRICTION.md` — structured failure log

The **highest-leverage artefact** for any follow-up writing about the run. The template at the top of the file is reusable verbatim across branches.

A friction entry exists when something took **more than a one-line config tweak** to resolve. Each entry has:

- **id** — sequential `F-001`, `F-002`, … never re-used
- **timestamp** (UTC)
- **phase** — one of `env_install`, `model_load`, `sft`, `grpo`, `eval`, `cleanup`, `other`
- **signature** — the actual error or symptom (truncate stacktraces to ~10 most informative lines)
- **hypothesized root cause** — be specific; "uncertain" is allowed
- **attempts** — bulleted, each with outcome (`worked` / `no effect` / `made it worse` / `crashed differently`)
- **final state** — `resolved` / `worked-around` / `punted` / `open`
- **notes** (optional) — context that doesn't fit above

**Conventions worth carrying forward:**

- **Read FRICTION.md before applying patches.** Half the value is reuse — don't re-derive a workaround that already exists.
- **Sequential IDs only.** Never re-use even if the issue was later resolved.
- **Cross-reference F-ids in three places**: commit messages (`fix: F-007 PEFT crash on adapter merge`), STATUS.md blocks (`Blockers: F-007`), and `train.py` inline comments (`# See FRICTION.md F-006: ...`). Three-way linkage is what makes the run reconstructible months later.
- **Open ≠ stuck.** An entry can be `final state: open` while you proceed past it with a workaround. The point is to capture what was learned, not to gate progress.

**Source of truth:** `notebooks/05-autoresearch/autoresearch-sft-grpo/FRICTION.md` on `nvfp4-blackwell` (entries `F-001` through `F-006` are worked examples).

### 6. `results.tsv` — per-experiment ledger

Single TSV file. Header is fixed; one row per experiment, **including failures**. Schema:

```
commit  metric  bit_ops  cipher  gravity  numeral  symbol  unit_conv  status  description
```

For a non-puzzle problem, replace the per-category columns with whatever diagnostic axes the problem has. The other four columns (`commit`, `metric`, `status`, `description`) are universal.

- `commit` — short hash from `git rev-parse --short HEAD`
- `metric` — `0.XXXX` on success, or literal `FAILED` on failure
- `status` — `success` / `failed` / `aborted`
- `description` — ≤120 chars, what was tried this run (e.g., `cosine reward, GRPO temp=0.5, +500 synth/cat`)

**The failed-run row is non-negotiable.** Capturing crashes alongside successes is what makes the ledger usable as both a metric history *and* a failure-rate signal.

### 7. `build.log` — install/build provenance

Per-environment, written once per fresh bootstrap. The 2026-05-02 example captures:

- Pod profile (GPU, driver, CUDA, Python, working dir, branch, storage)
- Pre-step cleanup (what was deleted from a stale `.venv`)
- Step-by-step install attempts including failed ones (the build-isolation trap, the PyPI-vs-CDN throughput gap, the qutlass squat)
- Throughput samples (KB/s timing tables) when network was the suspect
- Lessons that flow back into `runpod-setup.md`

Pattern: when a fresh bootstrap surfaces non-obvious behaviour (slow downloads, dependency-resolution surprises, package squats), capture it in `build.log` first, then *if it's reproducible* lift the lesson into `runpod-setup.md`. `build.log` is the rough draft; `runpod-setup.md` is the redacted recipe.

### 8. `check_install.py` — automated pre-flight

Single Python script that prints version + presence of every dependency, the GPU, expected data files, and any version pins. Returns non-zero on failure. The structure is reusable as a skeleton; the import list is branch-specific.

Run after `pip install` and before `prepare.py`. Re-run after any dependency bump. If it warns about a hardcoded version that's been intentionally bumped, fix the warning *in the script*, not by ignoring it — a stale check that everyone learns to ignore is worse than no check.

## Cross-cutting patterns

### The Logging contract (in program.md)

The `## Logging & Reporting` section in `program.md` is **reusable verbatim**. It tells the agent that the four artefacts (results.tsv / STATUS.md / FRICTION.md / end-of-session summary) are part of the contract, with the explicit framing that *the data is for a follow-up blog post*. That framing matters — agents log differently when they know the audience.

### Tiered work — T1 / T2 / T3

Three tiers, each with a different change cadence and risk profile.

| Tier | Purpose | Cadence | Commit prefix | What goes here |
|---|---|---|---|---|
| **Tier 1** | One-time correctness / cleanup before sweeping | Once per branch refresh | `T1.x:` | Pin versions, remove dead code, fix bugs surfaced by review |
| **Tier 2** | Research sweeps with comparable baselines | Many runs | `T2.x:` | Hyperparameter sweeps, reward shape variations, data ratios |
| **Tier 3** | Research bets requiring multi-run setup | Scaffolding only | `T3.x:` | New flags defaulting to off; *do not implement bodies* until promoted |

**Discipline:** Tier 1 must land (and pass the regression bar — see Validation Contract) before Tier 2 starts. Tier 3 stays as no-op flags until explicitly promoted to Tier 2. Mixing tiers contaminates the experiment record.

### Pre-flight verification pattern

When a code review (Codex, the human, or anyone) flags a possible issue, the response in `program.md` follows a fixed shape:

1. State the flagged issue and assess (likely real / likely false positive).
2. Provide a verification command — usually a one-liner Python check or a `grep` against installed source.
3. Provide a **decision tree table**: column 1 is the verification output, column 2 is the action.
4. Default failure path: **STOP and add a FRICTION.md entry** rather than guessing.

The pattern surfaced through F-001 (Codex flagged `dtype` as wrong; turned out to be a transformers 5.x signature change, *and* the original signature-introspection check itself didn't survive the refactor — verifying via source-string search did). The lesson generalised: a verification command can itself become stale, so a decision tree should always include a "neither expected outcome happened" branch.

### Cross-referencing convention

Three-way linkage between code, commits, and run-log:

- **Commits** include T-ids (`T1.3: pin requirements.txt to verified working set`) and F-ids (`fix: F-007 PEFT crash on adapter merge`).
- **STATUS.md** heartbeat blocks list active blockers as F-ids only.
- **`train.py` inline comments** reference F-ids next to the patch they implement (`# See FRICTION.md F-006: PEFT/_replace_module crashes on qweight=None`).

Apply this discipline consistently and `git log --oneline` becomes the experiment table; the FRICTION ledger stays grep-able from any direction.

### Branch hygiene

- **One commit per change ID.** T1.1 = one commit. Don't squash.
- **Revert, don't fix-forward, on regressions.** If a Tier 2 experiment hurts METRIC, `git revert` rather than patching on top. Keeps the experiment record honest.
- **`BRANCH_NOTES.md` gets a per-tier section** so anyone reading the branch later sees the chronology.
- **Co-existence smoke tests at phase boundaries.** When a major path is deprecated (e.g., `SKIP_GRPO=True` becomes the default), still run *one* sanity invocation of the deprecated path at phase-end to catch regressions early.

### Validation contract per tier transition

Every run produces (already wired in `train.py`):

1. **METRIC** — primary scalar, parsed by autoresearch
2. **Per-category accuracy** — diagnoses where gains came from
3. **Peak VRAM + tokens/sec** — hardware reality
4. **W&B run URL** — recorded in STATUS.md

Plus, **once per Tier transition** (not every run, too expensive):

5. **Adapter-on-deployment-base sanity check** — train adapter on whatever the training base is, then load it on whatever the *scoring* base is, and verify a sample inference works. This is the actual deployment path; most likely silent-break point. Log to FRICTION.md if it fails, even if the run otherwise produced a METRIC.

**Regression bar:** post-T1 METRIC ≥ pre-T1 METRIC before Tier 2 starts. Tier 1 should be neutral or positive; if it regresses, identify which T1.x caused it before adding more variables.

### Operational realities — the per-environment quirks

Every pod environment has surprises that don't appear in the docs. Capture them in `program.md`'s `## Operational realities` section so the next operator (human or agent) doesn't repeat the wasted hours. Examples from the 2026-05-02 nvfp4 install:

- Cold model load is ~22 min on RunPod's MooseFS; warm is ~3 min. Plan experiments to reuse page cache within one pod session.
- PyPI from EU pods runs at ~50 KB/s; the PyTorch CDN at ~36 MB/s. Use the index-url override.
- `qutlass` on PyPI is a v0.0.0 squat; the real package is on GitHub.
- Page cache invalidates on pod stop/restart even though MooseFS persists.
- Fresh-pod end-to-end install is ~50 min — one-off cost, not per-experiment.

These are mostly RunPod-specific. A different cloud vendor will surface a different set; the pattern of recording them is what transfers.

## Transferring this methodology to a new branch — checklist

When you start a new autoresearch branch (or a new repo entirely), this is the order of operations.

### Files to copy verbatim (no edits needed)

- [ ] `FRICTION.md` — keep the header + template + conventions block; clear the `## Entries` section.
- [ ] `STATUS.md` — keep the scaffold; clear any prior session blocks.
- [ ] `results.tsv` — keep the header line only; truncate rows.

### Files to copy and adapt

- [ ] `program.md` — keep the section structure (Plan reference, Active Mode, Preconditions, Pre-flight, Tier 1, Goal, Metric, Known Baselines, What you can modify, Constraints, Logging & Reporting, Branch Hygiene, Validation Contract, Tips, Branch Notes, Patches table, Operational Realities). Replace branch-specific content. **The Logging & Reporting section is reusable verbatim.**
- [ ] `runpod-setup.md` — keep the three-part split. Adapt Part 1's pip-install staging to the new branch's source-built packages. Keep Part 2 (non-root user + Claude Code launch) and Part 3's pin-conflict / cache-poisoning recipes verbatim.
- [ ] `check_install.py` — keep the structure; swap in the new branch's import list and version pins.

### Files to start fresh

- [ ] `BRANCH_NOTES.md` — write from scratch for the new branch. Copy the *table layout* (Parameter | parent | this branch) but not the rows.
- [ ] `build.log` — written when the new branch's first fresh pod is bootstrapped. Don't carry over the old one.

### Documentation updates

- [ ] Create the strategic plan file under `docs/<plan-name>.md` in the autoresearch repo (the in-repo `docs/bf16-sft-only-plan.md` is the worked example for `main`) and reference it from the new `program.md`'s top-of-file Plan Reference line.
- [ ] If the new branch has a runpod handoff worth a stand-alone doc, add it as `docs/<topic>-bootstrap.md`.
- [ ] If you maintain a separate documentation vault for cross-project pages, mirror new files into the vault index — but treat the in-repo `docs/` versions as canonical when they drift.

### Process — once the files are in place

- [ ] **First commit** sets the baseline (METRIC pre-T1).
- [ ] **Tier 1 commits** land one-by-one with `T1.x:` prefixes. After each one, re-run `python train.py` and confirm METRIC ≥ baseline. Append a STATUS.md block.
- [ ] **Mark the legacy baselines** in `program.md` as legacy after T1 completes.
- [ ] **Begin Tier 2 sweeps.** Each sweep variation = one `T2.x:` commit + one `results.tsv` row + (where novel) one `FRICTION.md` entry.
- [ ] **Run one adapter-on-deployment-base check** at every Tier transition.
- [ ] **End-of-session summary** prepended to STATUS.md before the loop terminates.

## What I deliberately did *not* extract

Some things from the nvfp4-blackwell branch are *not* reusable, even though they sit alongside the methodology. Listing them so the boundary is explicit:

- **The five `train.py` patches** (FPQuantConfig store_master_weights, Mamba fast-path disable, gradient_checkpointing + input_require_grads, FPQuantLinear `__bases__` hack, qweight=None strip block, `_hf_peft_config_loaded=True` flag). Each defends a specific failure mode in the transformers/peft/fp_quant/qutlass version stack. Branch-specific.
- **F-001 through F-006** as content. Their *form* is reusable (the FRICTION template); their content is FPQuant-specific.
- **The Tier 2 sweep table** (MAX_GRAD_NORM / synthetic ratio / LoRA targets / MODE flag / Loader). Branch-specific research bets.
- **NVFP4-specific BRANCH_NOTES** (memory math, master-weights cost, qutlass GitHub source).
- **The Reasoning Challenge category metadata** (6 puzzle types, 5 val samples each). Problem-specific.
- **The `sft-only-nvfp4-plan.md`** strategic plan. Plan-specific.

## Worked source — files in this repo to study

The methodology was originally extracted from the `nvfp4-blackwell` branch of `autoresearch-sft-grpo` between 2026-04-30 and 2026-05-02. The same artefacts now exist on `main` (assimilated as T1.1..T1.12 on 2026-05-03 — 2026-05-04). If a concrete example is more useful than this abstract page, read these in order from the repo root:

1. [`../program.md`](../program.md) — full agent contract with all conventions instantiated
2. [`../runpod-setup.md`](../runpod-setup.md) — three-part split with all the bash recipes
3. [`../FRICTION.md`](../FRICTION.md) — F-001 through F-006 as worked examples
4. [`../STATUS.md`](../STATUS.md) — backdated session summary + heartbeat as worked examples
5. [`../build.log`](../build.log) — written on first fresh-pod bootstrap (intentionally absent until then)
6. [`../results.tsv`](../results.tsv) — schema only; rows populated by autoresearch runs
7. [`../BRANCH_NOTES.md`](../BRANCH_NOTES.md) — branch identity and Tier 1 chronology as a worked example

## Related docs in this repo

- [`bf16-sft-only-plan.md`](bf16-sft-only-plan.md) — Strategic plan that anchors `program.md`'s top-of-file *Plan reference* line.
- [`fast-path-and-cache.md`](fast-path-and-cache.md) — Deep dive on F-001: why `train.py` carries two redundant defenses, what changes when F-001 resolves upstream.

A handful of related notes (NVFP4 LoRA tutorial, runpod-bootstrap walkthrough, tutorial-vs-impl audit, NVFP4 strategic plan) live in a separate documentation vault outside this repo — they're branch-specific to `nvfp4-blackwell` and not directly relevant to operating `main`.

## Changelog

- **2026-05-03** — Initial extraction from `nvfp4-blackwell` branch state at commit `6785b57` plus the 2026-05-02 working-tree (Logging & Reporting contract, runpod-setup three-part split, FRICTION template, build.log convention, T1/T2/T3 tiering).
