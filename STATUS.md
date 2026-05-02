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

### Session Summary 2026-05-02 — NVFP4 + SFT-only baseline blocked at PEFT/fp_quant seam (no METRIC produced)

- **Best METRIC achieved:** none. No `train.py` invocation reached the validation loop; no row appended to `results.tsv`. Three sequential failures, each unblocking the next, none reaching training.
- **Experiments run:** 3 attempted, all failed.
  1. `train.py` baseline as-shipped on the branch → `transformers.validate_quantization_for_training` raised "FPQUANT do not support training" before the SFT loop. Logged as **F-003**.
  2. With `model._hf_peft_config_loaded = True` patch (F-003 fix) → reached step 0 of SFT, then `fp_quant.linear_fns.FPQuant4x16NoMasterFn.backward` raised `NotImplementedError`. Logged as **F-005**.
  3. With `store_master_weights=True` patch (F-005 fix) → `peft._replace_module` crashed at `module.to(weight.device)` with `AttributeError: 'NoneType' object has no attribute 'device'` because fp_quant sets `qweight = None` in master mode and PEFT's `hasattr` short-circuit picks it up. Logged as **F-006**.
- **What was tried:**
  - Three single-line patches in `train.py` (still in working tree, uncommitted):
    - `SKIP_GRPO = True` (operational, not a Tier 1 change)
    - `model._hf_peft_config_loaded = True` after `get_peft_model(...)`
    - `store_master_weights=True` in `FPQuantConfig`
  - Patches 1 and 2 individually unblock the run; patch 3 trades the F-005 unimplemented-backward problem for the F-006 PEFT-incompat problem. Net: each patch is correct in isolation, but together they expose the underlying gap that the LoRA-on-FPQuant-Master training path is not integration-tested across this exact (transformers 5.7.0 + peft 0.19.1 + fp_quant 0.3.2 + qutlass 0.2.0) version set.
- **Top 3 friction items:**
  - **F-006** — PEFT crashes on `qweight=None` from fp_quant master mode. Concrete fix candidate: `del module.qweight` for any FPQuantLinear where `qweight is None`, between the `__bases__` patch and `get_peft_model`. Untested.
  - **F-005** — fp_quant 0.3.2 has no backward for `FPQuant4x16NoMasterFn`. Real upstream gap; can be worked around only by toggling master weights or pseudoquant.
  - **F-004** — RunPod `/workspace` is MooseFS-over-network at ~33 MB/s; 60 GB cold load takes ~22 min. Page-cache effect mitigates within a single session (warm load was ~3 min on the second run). For T2 sweeps this dominates wall-clock cost.
- **Other friction also logged:** F-001 (stale transformers signature check, false-alarm STOP), F-002 (qutlass PyPI stub trap — resolved by installing real QuTLASS from GitHub).
- **Open problems / what to try next session:**
  1. **Apply F-006 candidate fix** — `del` the `None`-valued `qweight`/`scales`/`dqweight` attributes before `get_peft_model`. Smallest, most localised next step. ~5 min to write, ~5 min to validate (page cache warm).
  2. **If that succeeds:** verify the run reaches "Phase 1: SFT Warmup" loss line and observe peak VRAM. F-005's note projected ~87–97 GB steady-state with master weights + LoRA + activations on a 95 GB GPU; load-peak we observed was 96.6 GB so margins are slim. May still OOM during a long-sequence training step.
  3. **If F-006 fix succeeds and training fits:** proceed to the originally planned baseline capture and STOP per the kickoff instructions ("Do NOT start Tier 1 changes until I confirm the baseline").
  4. **If F-006 fix succeeds but training OOMs:** options narrow to (a) shorter `SFT_MAX_SEQ_LEN`, (b) layer-subset LoRA targeting (T2.3 last-15%-skip), (c) Unsloth NVFP4 loader (T2.5), (d) bigger GPU (H200/B200). User explicitly does not want BF16 fallback.
  5. **If F-006 has no easy fix** (e.g. fp_quant calls qweight before get_peft_model finishes): consider patching PEFT's `_replace_module` at runtime via monkey-patch, *or* opening upstream issues against PEFT (suggest `getattr(..., None) is not None`) and fp_quant (suggest `del` instead of `= None`).
- **Reflection — what surprised me:** the install phase was the *easy* part. PyTorch CDN (~36 MB/s) instead of PyPI (~50 KB/s) made the dependency download tractable, and `mamba_ssm`/`causal_conv1d`/`qutlass` all built fine for sm_120 once `--no-build-isolation` was applied uniformly. What killed the day was not infrastructure but **integration**: four upstream libraries each work in isolation, but the LoRA-on-FPQuant-master training path threads through all four and trips on a seam that no one has integration-tested. F-002 (qutlass squat), F-005 (no-master backward), F-006 (qweight=None) are all small, individually fixable, and yet collectively make the documented `nvfp4-blackwell` baseline unreachable on the current pinned versions. The branch's "Branch Notes" already calls out the FPQuantLinear `__bases__` hack; the picture that's emerging is that this hack is the *first* of several similar workarounds that need to live in `train.py` to make the integration close. The follow-up blog post likely has more material than expected.

- **Working-tree state at end of session:**
  - `train.py` has 3 uncommitted edits (SKIP_GRPO, F-003 patch, F-005 patch). Revert with `git checkout train.py` if a clean restart is desired.
  - `runpod-setup.md`, `program.md` have committed-quality docs updates from this session (staged install order, `hf auth login` deprecation, qutlass GitHub source, transformers 5.x signature-check rewrite). Currently uncommitted in working tree.
  - `build.log` (new file, 361 lines) and `FRICTION.md` (entries F-001..F-006) are uncommitted in working tree.
  - `results.tsv` not modified — no successful run to record.

