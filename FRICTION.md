# Friction Log

This file records non-trivial failures encountered during autoresearch runs. The intent is to make it easy for the human author to reconstruct *what broke, what was tried, and what stuck* without re-deriving it from raw logs — both for the next session and for the follow-up blog post on the autoresearch pattern.

A "non-trivial failure" is anything that took more than a one-line config tweak to resolve (or is still unresolved). Examples that qualify:

- GRPO crashed with a TRL tensor-mismatch error on Mamba/MoE
- The model's KV cache class crashed during generation (Nemotron `HybridMambaAttentionDynamicCache` bugs)
- The model loaded but generation produced empty output
- A quantization mode degraded METRIC vs the BF16 baseline
- `transformers_modules` cache poisoning after a version bump

Examples that do *not* need an entry: a typo in `train.py`, an obvious off-by-one in a reward function, a missing import.

---

## Template

Copy this block for each new entry. **Newer entries go at the top of the Entries section below.**

```markdown
### F-NNN — <one-line summary>

- **timestamp:** <YYYY-MM-DD HH:MM UTC>
- **phase:** <env_install | model_load | sft | grpo | eval | cleanup | other>
- **signature:**

  ```
  <actual error message or observed symptom; truncate stacktrace to ~10 most informative lines>
  ```

- **hypothesized root cause:** <best guess; say "uncertain" if uncertain>
- **attempts:**
  - <attempt 1> → <worked | no effect | made it worse | crashed differently>
  - <attempt 2> → <outcome>
  - ...
- **final state:** <resolved | worked-around | punted | open>
- **notes:** <optional — context that doesn't fit above; e.g., "happens only with `BATCH_SIZE > 1`">
```

---

## Entries

<!-- Append new entries below this line, newest first. Use sequential ids: F-001, F-002, ... -->

### F-006 — `.git/index` lock from concurrent shell prevented commits mid-session

- **timestamp:** 2026-04-06 17:00 UTC
- **phase:** cleanup
- **signature:**

  ```
  fatal: Unable to create '.git/index.lock': File exists.
  Another git process seems to be running in this repository, e.g. an editor
  opened by 'git commit'. Please make sure all processes are terminated then try again.
  ```

  Commits during the 0.5333-tuning session did not take. The 0.5333 config landed in the working tree but not in any commit until `c1bb0a6` was finally made by a different shell.

- **hypothesized root cause:** A long-running git command (likely `git status` from a parallel Claude Code shell) crashed without releasing the lock; subsequent commits failed silently from automation that did not check exit codes.
- **attempts:**
  - Identified the lock and the working-tree drift after the session paused → diagnosed.
  - Saved `train.py` to a backup, removed `.git/index.lock`, re-staged and committed → worked. Commit `c1bb0a6` captured the 0.5333 config from the working tree.
- **final state:** resolved.
- **notes:** Generalisable lesson: in long-running automation, `git status` and `git commit` should not run concurrently from sibling shells against the same working tree. Pre-flight check before any T1 commit: `ls .git/index.lock` (verified empty during 2026-05-03 assimilation; safe to proceed).

### F-005 — `EVAL_MAX_NEW_TOKENS=128` truncates answers before `\boxed{}` closes

- **timestamp:** 2026-04-06 16:00 UTC
- **phase:** eval
- **signature:**

  ```
  Outputs visibly cut off mid-reasoning. Many ended with `<think>...` and
  never emitted `\boxed{<answer>}`. METRIC artificially low (~0.20-0.30 on
  the same trained adapter that scored 0.5333 with a higher cap).
  ```

- **hypothesized root cause:** Native thinking traces frequently exceed 128 tokens; the model needs headroom to think *and* close with a boxed answer when `enable_thinking=True` is in effect.
- **attempts:**
  - `EVAL_MAX_NEW_TOKENS=128` → many outputs truncated.
  - `EVAL_MAX_NEW_TOKENS=256` → barely works; some still cut off.
  - `EVAL_MAX_NEW_TOKENS=512` (now at `train.py:75`) → reliably emits `\boxed{}`.
- **final state:** resolved at 512.
- **notes:** Cost is wall-clock-bound on the 3-hour eval phase, not memory. Pairs with `EVAL_BATCH_SIZE=1` (`train.py:76`) to stay under 80 GB on A100.

### F-004 — `USE_COT=False` produces long thinking but no `\boxed{}` answer; METRIC 0.1667

- **timestamp:** 2026-04-06 15:00 UTC
- **phase:** eval
- **signature:**

  ```
  METRIC dropped to 0.1667. Inspecting model output:
  <think>...substantial reasoning content...</think>
  <response ends without emitting \boxed{}>
  ```

- **hypothesized root cause:** Eval uses `enable_thinking=True` so the model fills `<think>` with native reasoning. Without SFT teaching the think→`\boxed{}` pattern, the model exhausts its budget on thinking and never closes with a boxed answer. SFT with `USE_COT=True` (`train.py:79`, used at line 154) injects static CoT templates inside `<think>` followed by `\boxed{}`, teaching the closing pattern.
- **attempts:**
  - `USE_COT=False` on 50 samples/type → METRIC 0.1667 (no boxed answers extracted).
  - `USE_COT=True` on 200 samples/type → METRIC 0.5333.
- **final state:** resolved (USE_COT=True locked on main).
- **notes:** The CoT templates in `_COT_BY_TYPE` (`train.py:91` onwards) are static-per-category. Earlier observation noted "model parroted templates verbatim instead of reasoning" — at this sample count that parroting is *not* a regression: it is what teaches the closing `\boxed{}` pattern. A future branch may sweep dynamic CoT (the in-progress 2026-04-06 task `bdxqkblt5` was such an attempt).

### F-003 — 4-bit quantization degrades METRIC from 0.5333 to 0.1333; BF16 retained

- **timestamp:** 2026-04-06 14:00 UTC
- **phase:** sft
- **signature:** empirical regression. Two runs at the same git revision, identical except for `load_in_4bit=True` (BitsAndBytes) vs `torch_dtype=torch.bfloat16`, produced 0.1333 vs 0.5333 respectively.
- **hypothesized root cause:** The Reasoning Challenge requires precise arithmetic and string manipulation (gravity decimals, bit_ops binary, cipher characters). 4-bit weight noise corrupts the rule-induction step. Master-weight quantization paths that work for general SFT do not preserve task-relevant precision here.
- **attempts:**
  - `load_in_4bit=True` with BitsAndBytes quant config → METRIC 0.1333.
  - Reverted to plain `torch_dtype=torch.bfloat16` (`train.py:380`) → METRIC 0.5333.
- **final state:** resolved (BF16 is the locked configuration; see `BRANCH_NOTES.md`).
- **notes:** BF16 is the parent-of-branches dtype for `main`. The `nvfp4-blackwell` branch demonstrates a different quantization-during-training story (master weights kept) where this F-003 lesson does not directly apply; that's its own research bet.

### F-002 — `GRPOTrainer` raises `size of tensor` mismatch on Mamba/MoE; SFT-only adopted

- **timestamp:** 2026-04-06 13:00 UTC
- **phase:** grpo
- **signature:**

  ```
  RuntimeError: The size of tensor a (...) must match the size of tensor b (...)
  (raised inside trl.GRPOTrainer.train(), wrapped in train.py's try/except at line 506)
  ```

- **hypothesized root cause:** Known TRL incompatibility with Nemotron's hybrid Mamba/MoE architecture. References:
  - https://github.com/huggingface/trl/issues/3681
  - https://github.com/unslothai/unsloth/issues/3387

  Compounded by F-001: even if the tensor mismatch were fixed, each GRPO step would take ~30 min without a working KV cache (generation is O(n²)), making GRPO infeasible on `main` regardless.

- **attempts:**
  - try/except around `grpo_trainer.train()` so SFT-only completes (`train.py:506-523`) → worked: SFT adapter is preserved; GRPO failure is non-fatal.
  - Pursuing TRL upgrade or Unsloth GRPOTrainer → not attempted on `main`; deferred to a future branch.
- **final state:** punted — SFT-only is the documented active mode for `main`.
- **notes:** `train.py` prints a diagnosis block on this exception (lines 513-523) listing four candidate fixes. Reading the diagnosis is the first step for anyone re-attempting GRPO. Do not enable GRPO as mandatory on `main`; the standing try/except is the smoke harness, not a regression.

### F-001 — Nemotron `HybridMambaAttentionDynamicCache` has multiple bugs; KV cache disabled at eval

- **timestamp:** 2026-04-06 12:00 UTC
- **phase:** model_load
- **signature:**

  ```
  AttributeError: 'HybridMambaAttentionDynamicCache' object has no attribute 'conv_kernel_size'
  AttributeError: 'list' object has no attribute 'device'
  RuntimeError: shape mismatch in conv state: expected 6144, got 4096
  RuntimeError: SSM state shape mismatch
  ```

- **hypothesized root cause:** `~/.cache/huggingface/modules/.../modeling_nemotron_h.py` ships with several latent bugs in the hybrid cache class. Hits any code path that touches the cache (generation with `use_cache=True`, GRPO rollout).
- **attempts:**
  - Patch `modeling_nemotron_h.py` in place (`conv_dim` 4096→6144, `conv_kernel_size` attribute, `.device` access fixes) → worked for some paths but did not unblock GRPO.
  - Set `model.config.use_cache = False` before `evaluate_model(...)` (`train.py:530`) → worked for SFT eval; eval is slow (~3 hours for 30 samples) without cache but produces a correct METRIC.
- **final state:** worked-around (cache disabled at eval; GRPO still blocked, see F-002).
- **notes:** The in-place edit to the HF cache module is per-pod; clearing the `transformers_modules` cache wipes it. Re-run the same edits or re-derive on each fresh pod (`runpod-setup.md` Part 3 has the cache-clear path). Long term, an upstream PR to the model card is the real fix.
