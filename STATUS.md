# Autoresearch SFT+GRPO — Run Log

This is the append-only heartbeat log for autoresearch runs on this repo. The contract for what to log and when is in `program.md` § "Logging & Reporting".

For the configuration delta between branches (e.g., `main` vs `nvfp4-blackwell`), see `BRANCH_NOTES.md`.

For structured failure entries, see `FRICTION.md`.

For per-experiment metric history, see `results.tsv`.

---

<!--
Append new status blocks below this divider, **newest at the top**.

Block template (per program.md):

### YYYY-MM-DD HH:MM UTC — <one-line summary>

- **Current best METRIC:** 0.XXXX (per-category: bit_ops X%, cipher X%, gravity X%, numeral X%, symbol X%, unit_conv X%)
- **Experiments since last status:** N (most informative: <one line>)
- **What was tried:** <bullets>; net effect: helped / hurt / neutral
- **Next:** <plan + reason>
- **Blockers:** <FRICTION.md ids, e.g., F-003, F-007>

End-of-session summary blocks go at the very top under the heading `### Session Summary YYYY-MM-DD`.
-->

### 2026-05-03 ~12:00 UTC — Methodology assimilation (T1.1..T1.8a) landed; T1.8b regression run pending pod

- **Current best METRIC:** 0.5333 (per-category: bit_ops 60%, cipher 0%, gravity 20%, numeral 100%, symbol 40%, unit_conv 100%) — **carried over from c1bb0a6**, no new training run yet under the new methodology format.
- **Experiments since last status:** 0 (this is a documentation-only landing; no `train.py` execution).
- **What was tried:**
  - T1.1: added `FRICTION.md` template + conventions (commit `c6350b9`).
  - T1.2: added `BRANCH_NOTES.md` describing main as BF16 SFT-only anchor (`d097311`).
  - T1.3: added `runpod-setup.md` three-part scaffold for BF16 main (`11d869a`).
  - T1.4: restructured `program.md` to the 14-section methodology template (`0bc24e8`).
  - T1.5: converted this STATUS.md to append-only ledger format (`c7c23f3`).
  - T1.6: seeded `FRICTION.md` with F-001..F-006 from main history (`49d9c38`).
  - T1.7: cross-referenced FRICTION ids in `train.py` — 5 comment-only edits (`1a44857`).
  - T1.8a: marked Known Baselines as legacy (pre-T1) and prepended this heartbeat (this commit).
  - Net effect: **neutral** on METRIC by construction (no logic change). All eight commits land in one session.
- **Next:** T1.8b — run `python train.py` on a fresh A100 80GB pod, confirm METRIC ≥ 0.5333 against the post-T1.7 working tree, append a row to `results.tsv` with `description="post-T1 regression sanity (no train.py logic change)"`. Until then, the assimilation status is **documentation-complete, regression-pending**.
- **Blockers:** none for T1.1..T1.8a. T1.8b is gated on RunPod availability, not on any technical issue.
- **Notes:** All five `# See FRICTION.md F-NNN` cross-references in `train.py` resolve cleanly against the F-001..F-006 entries seeded in T1.6. The methodology spec is at `docs/methodology.md` (inlined as part of T1.13); the narrative assimilation plan that drove T1.1..T1.12 lives in the separate documentation vault outside this repo.

### Session Summary 2026-04-06 — BF16 SFT-only baseline at METRIC 0.5333

- **Best METRIC achieved:** 0.5333 (per-category: bit_ops 60%, cipher 0%, gravity 20%, numeral 100%, symbol 40%, unit_conv 100%).

- **Experiments run:** several SFT-only iterations across the session. The winning configuration was 1200 SFT samples (200/category), 1 epoch, `USE_COT=True`, BF16, `LORA_RANK=32`, `target_modules='all-linear'`, `EVAL_MAX_NEW_TOKENS=512`, `EVAL_BATCH_SIZE=1` on A100 80GB. Peak VRAM ~78 GB / 80 GB.

- **What was tried (across the session):**
  - 4-bit quantization → METRIC 0.1333 (degraded). Reverted.
  - `USE_COT=False` (50 samples/type) → METRIC 0.1667 (model emitted long thinking but no `\boxed{}` answer). Reverted.
  - `EVAL_MAX_NEW_TOKENS=128` → many outputs truncated mid-think. Bumped to 256 → still some truncation. 512 reliable.
  - In-place edits to `~/.cache/.../modeling_nemotron_h.py` (cache class bug fixes) → some paths unblocked, GRPO still tensor-mismatched.
  - Try/except around `GRPOTrainer.train()` → SFT adapter preserved on GRPO crash; eval proceeds.

- **Top friction items (seeded as F-001..F-006 in T1.6):**
  - **F-001** Nemotron `HybridMambaAttentionDynamicCache` bugs — worked-around (cache disabled at eval).
  - **F-002** GRPOTrainer tensor mismatch on Mamba/MoE — punted (SFT-only adopted).
  - **F-003** 4-bit quantization degrades 0.5333 → 0.1333 — resolved (BF16 retained).
  - **F-004** `USE_COT=False` yields no `\boxed{}` (METRIC 0.1667) — resolved (`USE_COT=True`).
  - **F-005** `EVAL_MAX_NEW_TOKENS=128` truncates — resolved at 512.
  - **F-006** Prior-session `.git/index` lock prevented commits — resolved (saved `train.py`, removed lock, recommitted).

- **Open problems / what to try next session:**
  1. `cipher` 0% and `gravity` 20% are the largest gaps; **dynamic CoT** was the in-progress hypothesis when the session paused (task `bdxqkblt5` — added `_build_dynamic_cot()` in `build_sft_text` to construct per-example reasoning for cipher/gravity/unit_conv/bit_ops with fallback to static CoT).
  2. 3-hour eval is the wall-clock dominant phase; KV-cache rehabilitation would unlock GRPO and faster eval but is out of scope on `main`.
  3. Improve cipher CoT — verify the regex for extracting pairs actually fires on training data.
  4. `SFT_EPOCHS = 2` to give the model more time on dynamic patterns (doubles SFT to ~2 hours).
  5. Oversample hard categories (cipher, gravity, symbol) via custom sampler instead of stratified.

- **Reflection (one paragraph):** the session converged on a defensible 0.5333 by ruling out 4-bit (F-003), ruling out `USE_COT=False` (F-004), and accepting GRPO as out-of-reach for this model + TRL combination (F-002). The 5× improvement from F-005 alone (128 → 512 token cap) was the single biggest unlock; the rest was disciplined locking-in of choices that had clear empirical support. The `.git/index` lock issue (F-006) cost meaningful time and is one of the explicit motivations for the methodology assimilation that lands as T1.1..T1.8a on 2026-05-03.

(Working-tree archival: at session pause, modified-in-place `~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/.../modeling_nemotron_h.py` carried cache bug fixes — `conv_dim` 4096→6144, `conv_kernel_size` attribute, `.device` access. Pyc cache must be cleared with `rm -rf .../__pycache__` before any run on a fresh pod. In-progress task `bdxqkblt5` had SFT training active at 67 GB VRAM with dynamic CoT applied; output log preserved under `/tmp/claude-1001/.../tasks/bdxqkblt5.output`.)
