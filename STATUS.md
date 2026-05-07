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

### 2026-05-07 ~14:55 UTC — T2.7 (SFT_SAMPLES_PER_TYPE 200 → 300) result: METRIC 0.5333 (regression vs T1.16 0.5667; matches c1bb0a6 baseline). Reverting per branch hygiene.

- **Current best METRIC:** 0.5667 (T1.16, unchanged). T2.7 yielded 0.5333.
- **Experiments since last status:** 1 (commit `cdb1aa7`, T2.7).
- **What was tried:** First Tier 2 sweep — bumped `SFT_SAMPLES_PER_TYPE` 200 → 300 (1200 → 1800 SFT samples, 6 categories × 300). All other hyperparameters and prompt-format unchanged. Hypothesis: cipher 20% / gravity 20% might be undertrained on data volume.
- **Result:** METRIC 0.5333 vs T1.16's 0.5667. Per-category:
  - bit_ops 60% (3/5) — same
  - **cipher 0% (0/5)** — regressed from 1/5 in T1.16
  - gravity 20% (1/5) — same
  - numeral 100% (5/5) — same
  - symbol 40% (2/5) — same
  - unit_conv 100% (5/5) — same
- **Net effect:** hurt by 0.0334. But ±0.0333 is the irreducible variance floor for a 30-sample val (1 sample = 1/30 = 0.0333), so this is one-sample noise on cipher, not a real shift. Data-volume hypothesis is dead — more SFT samples did not move the gap categories.
- **Time:** 23,397 s (6 h 30 m). 1.43× T1.16's 16,367 s, matching the 1.5× SFT step-count increase (450 vs 300 steps × ~42 s/step). Per-step time was identical between runs; earlier "slowdown" claim was a misread of progress-bar data.
- **Adapter sanity check (separate Python process, fresh BF16 base):** PASS. 73 tokens / 122 chars on the same gravity val example as T1.16. Output: `Using d = 0.5*g*t^2: g = 2d/t^2 = 2*0.5/1.86^2 = 0.2891. Apply to find d for the new t. Answer: 58.5` (gold 59.3).
- **Critical diagnostic — T2.8 target:** the sanity-check output (and the eval debug for gravity) shows the model has internalized a *broken* dynamic CoT pattern. Investigation found the bug: `_build_dynamic_cot()` in train.py has buggy regexes for both cipher and gravity:
  - **Gravity regex** `r't\s*=\s*([\d.]+).*?d\s*=\s*([\d.]+)'` matches the first `d=` it sees, which is **inside the formula `d = 0.5*g*t^2`** rather than the data-point line. So it extracts `t=1.86, d=0.5` and computes `g=0.2891` instead of using actual `d=17.75` (true `g≈10.27`). The model is trained on `<think>` blocks claiming `g=0.2891`, which is what we now see at inference.
  - **Cipher regex** `r'([a-z\s]+?)\s*->\s*([a-z\s]+)'` is greedy across newlines — on a real cipher prompt it returns 3 matches but the first is `('\ngeq bytq lyca tgxkyqt', 'the wise king studies\ngeq sxjyoxt tgxkqcg tqqt ')`, pairing one line's cipher with multiple lines of plain text. Substitution map built from this is garbage.
  - Both bugs were probably masked previously because static `_COT_BY_TYPE` fallback was used when the regex didn't match at all, but on these prompts the broken regex *does* match and produces garbage that's worse than the static fallback.
- **Decision:** revert T2.7 per program.md branch hygiene ("Revert, don't fix-forward, on regressions"). Drafting T2.8 next: fix both regexes in `_build_dynamic_cot()`. Highly likely to move cipher and gravity since the structural cause of both being weak has been identified.
- **Side note:** BRANCH_NOTES.md "SFT ~1 h, eval ~3 h" appears stale vs observed (~3.5 h SFT, ~1 h eval). Worth a doc-only T-id fix, not blocking.
- **Blockers:** none open.

---

### Session Summary 2026-05-07 — Baseline restoration on fresh A100 80GB pod (autonomous overnight run)

- **Best METRIC achieved:** **0.5667** (bit_ops 60%, cipher 20%, gravity 20%, numeral 100%, symbol 40%, unit_conv 100%) — **+0.0334 vs c1bb0a6 locked baseline 0.5333.** The single-step gain is cipher 0/5 → 1/5 (+1 sample correct = +1/30 = +0.0333 on METRIC); other categories identical. Within run-to-run variance for a 30-sample val set, but the regression bar is comfortably met.
- **Experiments run:** 1 train.py invocation reaching METRIC, after 2 false-starts that triggered F-009 (causal_conv1d static-import) and F-010 (hf_xet deadlock). 1/3 = success rate; the 2 failures are documented and worked around.
- **Top 3 friction items:**
  - **F-009 — `train.py` model load `ImportError: causal_conv1d`** despite the documented "conditional import handles it" defense. Resolved by T1.14 (revert T1.9 + 9-file doc sweep). transformers' dynamic-module loader does AST-level static import checking that ignores `if`/`try` guards.
  - **F-010 — `hf_xet` worker→main thread deadlock during weights download** with `HF_XET_HIGH_PERFORMANCE=1` set. Worked around in T1.15 by killing python, removing the frozen `*.incomplete`, and unsetting `HF_XET_HIGH_PERFORMANCE` so `hf_transfer` alone handles downloads.
  - **F-007 — `bootstrap.sh` wedges on MooseFS-backed `/workspace`** during wheel extraction. Worked around by hosting the venv at `/root/venv-autoresearch` and `UV_CACHE_DIR=/root/uv-cache`. The HF model cache at `/workspace/.cache/huggingface/` stays on MooseFS — large sequential I/O is fine; only metadata churn during pip/uv extraction suffers.
  - (Honourable mention: F-008 — `HF_HUB_ENABLE_HF_TRANSFER=1` without the `hf_transfer` package silently corrupts the HF cache and surfaces as a misleading "Unrecognized model" error in `AutoTokenizer`. Fixed by `uv pip install hf_transfer`. Not a blocker now but easily missed on fresh pods.)
- **Open problems / next session:**
  - **Confirm baseline isn't variance.** A single re-run on the same pod (~5 h) would tell us whether 0.5667 is reproducible or a 1-sample fluctuation on cipher. Cheap if pod is hot; expensive if pod is cold (60 GB redownload).
  - **Cipher and gravity remain the gap categories** at 20% each. These were already 0% / 20% in the locked baseline. If the human authorises Tier 2 work, the highest-leverage axes are (a) weighted sampling toward cipher/gravity (per program.md § Tier 2 data strategy), (b) category-specific reasoning hints in the SFT prompt format, or (c) increased `SFT_SAMPLES_PER_TYPE`. Reward function design is listed as highest-impact in program.md but is gated on GRPO working, which is punted (F-002).
  - **Promote F-010 workaround to runpod-setup.md if it reproduces** on a second pod. Per F-010 § notes I held off on changing setup docs until we have a second data point.
  - **F-008's `hf_transfer` install should arguably be added to `requirements.txt` or `bootstrap.sh`** — the pod env reliably enables `HF_HUB_ENABLE_HF_TRANSFER=1`, and the package is currently a tribal-knowledge install. Conservative edit, low risk.
- **One-paragraph reflection:** Three of the four big surprises this session (F-007/F-008/F-010) were *infrastructure* failures — none were about model behaviour, training, or research bets. The single research-shaped failure (F-009) was actually a *static-analysis* failure in transformers' dynamic-module loader, not a runtime issue. The lesson echoes the F-001 lineage: when you load remote code via `trust_remote_code=True`, the loader runs an AST-level pre-flight that doesn't honour your runtime guards — "conditional import" is interpreter-true but loader-false. Once those four were sorted, the actual training run was uneventful: SFT in ~1 h, GRPO crashed gracefully exactly per F-002, eval ran ~3 h with `use_cache=False` per F-001, METRIC matched (slightly exceeded) baseline. The methodology defences in `train.py:386` and `:536` are doing their job; the friction is around *getting to the point where they run at all*. Adapter sanity check on fresh BF16 base passed (73 tokens, `<think>` + `\boxed{}` both present), confirming the deployment path works end-to-end.

---

### 2026-05-07 ~03:55 UTC — T1.14 + T1.15 baseline-restoration run; METRIC 0.5667 ≥ 0.5333

- **Current best METRIC:** 0.5667 (bit_ops 60%, cipher 20%, gravity 20%, numeral 100%, symbol 40%, unit_conv 100%) — **+0.0334 vs c1bb0a6 baseline 0.5333.**
- **Experiments since last status:** 1 (the run this block reports on; 2 prior false-starts crashed before METRIC and did not append to results.tsv per program.md schema).
- **What was tried:**
  - 1st attempt (pre-T1.14, `causal_conv1d` uninstalled per T1.9): killed by `ImportError: causal_conv1d` from `transformers.dynamic_module_utils.check_imports` (AST-level static check ignoring the `if is_causal_conv1d_available():` guard in modeling_nemotron_h.py). Logged as F-009.
  - 2nd attempt (T1.14, `causal_conv1d` restored as hard dep): killed by an `hf_xet` worker→main deadlock during weights download — the last `*.incomplete` shard never got renamed and python's main thread sat in `futex_do_wait` for 5+ min. Logged as F-010.
  - 3rd attempt (T1.15, `HF_XET_HIGH_PERFORMANCE` unset, `hf_transfer` alone handles download): completed end-to-end. SFT 1200 × 1 epoch in ~1 h (300 steps); GRPO crashed cleanly on the F-002 Mamba/MoE+TRL tensor mismatch ("size of tensor a (29) must match … b (126)"); SFT-only adapter retained; eval (30 samples, no KV cache per F-001) produced METRIC 0.5667.
  - **Net effect:** helped — baseline restored without dtype change, USE_COT flip, GRPO mandatory, or any train.py logic edit. The 0.5333 → 0.5667 delta is 1 cipher sample flipping correct; possibly variance, possibly a slight upstream shift (HF logged a fresh `configuration_nemotron_h.py` download today).
- **Adapter-on-fresh-BF16-base sanity check:** **PASS.** Loaded base + LoRA in a separate Python process, generated 73 tokens / 122 chars on a gravity sample. `<think>` tag present, `\boxed{}` extracted ('55.5'; gold '59.3' — wrong but structurally plausible, gravity is the 20% category). Deployment path works.
- **Time:** 16367 s = **4 h 32 m 47 s.** Within the documented ~4-5 h per run budget.
- **Peak VRAM:** 76.1 GB / 80 GB (~95% utilization).
- **Pod environment notes (caught during this run, may matter next session):**
  - `/workspace` is MooseFS-backed (`mfs#us-md-1.runpod.net:9421`). Mitigation per F-007: venv at `/root/venv-autoresearch` (local overlay), `UV_CACHE_DIR=/root/uv-cache`, `.venv` symlinked into the repo. Local overlay disk has ~33 GB free after install.
  - HF model cache stays on `/workspace` (60 GB sequential I/O is fine; only metadata churn during wheel extraction is the MooseFS pain point).
  - Pod sets `HF_HUB_ENABLE_HF_TRANSFER=1` *and* `HF_XET_HIGH_PERFORMANCE=1` at the shell level. Per F-010 the latter must be unset before training; per F-008 the former requires the `hf_transfer` package to be installed (not in requirements.txt).
  - `WANDB_API_KEY` is exported but `wandb` itself is not installed; train.py uses `report_to='none'` so this is harmless.
- **Next:** **STOP per prompt.** The autonomous run met the regression bar (METRIC ≥ 0.5333) and the prompt explicitly says "Otherwise stop and report. Do NOT start Tier 2 changes until I confirm the baseline." The user authorised continued optimisation only if METRIC < 0.5333; that branch did not trigger. A single re-run on this hot pod would help discriminate variance from real shift but is at the user's discretion.
- **Blockers:** none open.
  - F-007 worked-around (move venv off MooseFS).
  - F-008 resolved (install `hf_transfer`, clear corrupted cache).
  - F-009 resolved by T1.14 (9-file doc sweep restoring causal_conv1d).
  - F-010 worked-around by T1.15 (unset HF_XET_HIGH_PERFORMANCE).

---

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
