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

### F-009 — `train.py` model load fails with `ImportError: causal_conv1d` despite the documented "conditional import handles it" defense

- **timestamp:** 2026-05-06 22:18 UTC
- **phase:** model_load
- **signature:**

  ```
  Encountered exception while importing causal_conv1d: No module named 'causal_conv1d'
  Traceback (most recent call last):
    ...
    File "/.../transformers/dynamic_module_utils.py", line 215, in check_imports
      raise ImportError(
  ImportError: This modeling file requires the following packages that were not found
  in your environment: causal_conv1d. Run `pip install causal_conv1d`
  ```

  The cached `modeling_nemotron_h.py` (`@cbd3fa9f`) wraps the import correctly:

  ```python
  if is_causal_conv1d_available():
      from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
  else:
      causal_conv1d_update, causal_conv1d_fn = None, None
  ```

  But transformers' `dynamic_module_utils.get_imports()` walks the AST top-down and collects every `Import`/`ImportFrom` node including children of `If`/`Try` nodes (it calls `recursive_look_for_imports(child)` on every AST child). `check_imports()` then `importlib.import_module()`s each one and raises if any is missing — bypassing the runtime conditional guard entirely.

- **hypothesized root cause:** The repo's policy on `main` (T1.9 + `requirements.txt` comment + `prompt.md`'s explicit "Do NOT install causal_conv1d") was based on a runtime-correct but static-analysis-incorrect mental model. The locked baseline `c1bb0a6` (METRIC 0.5333) must have been achieved either (a) with `causal_conv1d` actually installed, (b) on an older transformers version whose `check_imports` handled conditional imports, or (c) with an older `modeling_nemotron_h.py` upstream that didn't have the static `from causal_conv1d import ...`. The HF-side log confirms a recent upstream change today: "A new version of the following files was downloaded ... configuration_nemotron_h.py", strong hint the model code has been updated since the locked baseline.
- **attempts:**
  - Read transformers 4.51.3 `dynamic_module_utils.py:170-220` directly → confirmed AST walk is unconditional; `check_imports` cannot honour `if is_*_available()` guards.
  - Inspected cached `modeling_nemotron_h.py` (snapshot `cbd3fa9f`) → confirmed conditional import is present but ineffective against `check_imports`.
- **final state:** resolved by T1.14 (Option 1).
  - **Resolution:** Restored `causal_conv1d` as a hard install-time dep across `requirements.txt`, `bootstrap.sh`, `check_install.py`, `prompt.md`, `program.md` (T1.9 row marked superseded; new T1.14 row added), `BRANCH_NOTES.md`, `runpod-setup.md`, and `docs/fast-path-and-cache.md`. The `train.py:386` runtime fast-path disable still applies, so the package is dead code at execution time — it just needs to be present so transformers' AST scanner accepts the modeling file. No behavioural change vs the locked baseline; F-001's defensive posture is unchanged.
  - **Lesson:** "Conditional import handles absence" is true for the Python interpreter at runtime but false for transformers' dynamic-module loader, which does AST-level static import checking *before* the conditional guard ever evaluates. Future audits should treat any "this dep is conditional" claim as suspect when the file in question is loaded via `trust_remote_code=True`.

### F-008 — `prepare.py` AutoTokenizer fails with "Unrecognized model" when `HF_HUB_ENABLE_HF_TRANSFER=1` is set without the `hf_transfer` package

- **timestamp:** 2026-05-06 22:13 UTC
- **phase:** env_install
- **signature:**

  ```
  ValueError: Unrecognized model in nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
  Should have a `model_type` key in its config.json, or contain one of the
  following strings in its name: albert, align, ... nemotron, ...

  (raised from transformers/models/auto/configuration_auto.py:1151)
  ```

  But the model ID *does* contain "nemotron". Direct `huggingface_hub.hf_hub_download` raises a clearer error:

  ```
  ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1)
  but 'hf_transfer' package is not available in your environment.
  ```

- **hypothesized root cause:** The RunPod base image sets `HF_HUB_ENABLE_HF_TRANSFER=1` (and `HF_XET_HIGH_PERFORMANCE=1`) at the shell level by default to accelerate HF downloads. `huggingface_hub` only raises the clear "fast download enabled but not installed" error on direct `hf_hub_download`. Inside `transformers.AutoTokenizer.from_pretrained`, an earlier code path silently fell back to a no-op / empty download, producing a `config.json` with no `model_type` field. `AutoConfig` then matched on filename rather than model_type, mis-routed, and raised the misleading "Unrecognized model" error. The `nemotron_h` model_type for Nemotron-Nano-30B-BF16 *does* require `trust_remote_code=True` (which prepare.py passes), so the auto_map path would have worked if the config had been read correctly. The real failure is the silent download corruption upstream.
- **attempts:**
  - Verified model exists, public, not gated via `HfApi.model_info` → confirmed model is reachable. The error is not auth/visibility.
  - Direct `huggingface_hub.hf_hub_download('config.json')` → reproduced as a clean `hf_transfer` ImportError. Smoking gun.
  - `uv pip install hf_transfer` (`hf-transfer==0.1.9`), `rm -rf /workspace/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` to drop the corrupt config, re-ran `python prepare.py` → **worked**. Tokenizer loaded, vocab 131072, val_split.json written.
- **final state:** resolved.
- **notes:** Pre-flight check before any HF download: `env | grep -E '^HF_HUB_ENABLE_HF_TRANSFER='`; if set, `python -c "import hf_transfer"` must succeed or installs/downloads will silently corrupt the cache. Either install `hf_transfer` or `unset HF_HUB_ENABLE_HF_TRANSFER`. Generalisable: any time an HF env-var optimization is enabled at the pod level, the matching package must be installed in the venv. Easy to miss because the package is *not* in `requirements.txt` (the env var is a pod-level optimization, not a project dep).

### F-007 — `bootstrap.sh` wedges on MooseFS-backed `/workspace` during wheel extraction

- **timestamp:** 2026-05-06 21:58 UTC
- **phase:** env_install
- **signature:**

  ```
  bash bootstrap.sh hangs after the uv "Downloading ..." block. No further stdout
  for 9+ minutes. uv process alive (PID, futex_wait, 0% CPU). uv cache flat at
  ~187 MB; .venv remains 4.9 MB; new ./workspace/.cache/uv/.tmpXXXX extraction
  dirs keep being created (18 → 58) but their contents never finalize.
  TCP download connections went idle ~6 min before. Thread wchan reads
  `request_wait_answer` (FUSE wait state).
  ```

- **hypothesized root cause:** RunPod pods on `mfs#us-md-1.runpod.net:9421` mount `/workspace` via FUSE/MooseFS. uv wheel extraction is metadata-heavy (thousands of small writes + renames + hardlinks per wheel); each op is a network round-trip on MooseFS. The extraction phase effectively never finishes within a usable timeframe, even though TCP downloads complete normally. Same family of pain referenced in `bootstrap.sh`'s header (`~50 KB/s vs ~36 MB/s on RunPod EU pods`) and `BRANCH_NOTES.md` (the `nvfp4-blackwell` MooseFS pod's 22-min cold model load).
- **attempts:**
  - Waited 9 minutes for extraction to complete on `/workspace` MooseFS → no effect; new tmp dirs accumulating but nothing finalizes.
  - Killed `bash bootstrap.sh` + `uv pip install`, removed `/workspace/.cache/uv` and the empty MooseFS `.venv`, created a fresh venv at `/root/venv-autoresearch` (local overlay disk), set `UV_CACHE_DIR=/root/uv-cache`, symlinked `/workspace/autoresearch-sft-grpo/.venv → /root/venv-autoresearch` for compatibility with scripts that source `.venv/bin/activate`, then re-ran `bash bootstrap.sh` → **worked**. uv installed torch + 22 CUDA packages in **2.03 s** vs 9 minutes wedged.
- **final state:** worked-around.
- **notes:** Pre-flight check before bootstrap: `df -T /workspace | grep -q fuse && echo "MooseFS — move venv + UV_CACHE_DIR to /root before installing"`. Local overlay (`/`) is 40 GB on this pod — comfortable for the ~6 GB extracted CUDA libs. The HF model cache (~60 GB) stays on `/workspace` because large sequential reads/writes are exactly where MooseFS doesn't suffer. Generalisable: any FUSE-mounted shared filesystem (MooseFS, Ceph, Lustre, NFS) will exhibit this; the failure mode is metadata churn during pip/uv extraction, not bandwidth.

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
  - `EVAL_MAX_NEW_TOKENS=512` (now at `train.py:76`) → reliably emits `\boxed{}`.
- **final state:** resolved at 512.
- **notes:** Cost is wall-clock-bound on the 3-hour eval phase, not memory. Pairs with `EVAL_BATCH_SIZE=1` (`train.py:77`) to stay under 80 GB on A100.

### F-004 — `USE_COT=False` produces long thinking but no `\boxed{}` answer; METRIC 0.1667

- **timestamp:** 2026-04-06 15:00 UTC
- **phase:** eval
- **signature:**

  ```
  METRIC dropped to 0.1667. Inspecting model output:
  <think>...substantial reasoning content...</think>
  <response ends without emitting \boxed{}>
  ```

- **hypothesized root cause:** Eval uses `enable_thinking=True` so the model fills `<think>` with native reasoning. Without SFT teaching the think→`\boxed{}` pattern, the model exhausts its budget on thinking and never closes with a boxed answer. SFT with `USE_COT=True` (`train.py:81`, used at line 156) injects static CoT templates inside `<think>` followed by `\boxed{}`, teaching the closing pattern.
- **attempts:**
  - `USE_COT=False` on 50 samples/type → METRIC 0.1667 (no boxed answers extracted).
  - `USE_COT=True` on 200 samples/type → METRIC 0.5333.
- **final state:** resolved (USE_COT=True locked on main).
- **notes:** The CoT templates in `_COT_BY_TYPE` (`train.py:94` onwards) are static-per-category. Earlier observation noted "model parroted templates verbatim instead of reasoning" — at this sample count that parroting is *not* a regression: it is what teaches the closing `\boxed{}` pattern. A future branch may sweep dynamic CoT (the in-progress 2026-04-06 task `bdxqkblt5` was such an attempt).

### F-003 — 4-bit quantization degrades METRIC from 0.5333 to 0.1333; BF16 retained

- **timestamp:** 2026-04-06 14:00 UTC
- **phase:** sft
- **signature:** empirical regression. Two runs at the same git revision, identical except for `load_in_4bit=True` (BitsAndBytes) vs `torch_dtype=torch.bfloat16`, produced 0.1333 vs 0.5333 respectively.
- **hypothesized root cause:** The Reasoning Challenge requires precise arithmetic and string manipulation (gravity decimals, bit_ops binary, cipher characters). 4-bit weight noise corrupts the rule-induction step. Master-weight quantization paths that work for general SFT do not preserve task-relevant precision here.
- **attempts:**
  - `load_in_4bit=True` with BitsAndBytes quant config → METRIC 0.1333.
  - Reverted to plain `torch_dtype=torch.bfloat16` (`train.py:383`) → METRIC 0.5333.
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
  - try/except around `grpo_trainer.train()` so SFT-only completes (`train.py:511-528`) → worked: SFT adapter is preserved; GRPO failure is non-fatal.
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
  - Set `model.config.use_cache = False` before `evaluate_model(...)` (`train.py:536`) → worked for SFT eval; eval is slow (~3 hours for 30 samples) without cache but produces a correct METRIC.
- **final state:** worked-around (cache disabled at eval; GRPO still blocked, see F-002).
- **notes:** The in-place edit to the HF cache module is per-pod; clearing the `transformers_modules` cache wipes it. Re-run the same edits or re-derive on each fresh pod. Long term, an upstream PR to the model card is the real fix.

  **Clearing the cache** (after a `transformers` upgrade, or to force a re-derive of the in-place edits):

  ```bash
  # First inspect what's actually in the cache so you don't run a no-op rm:
  ls ~/.cache/huggingface/modules/transformers_modules/nvidia/

  # Then delete the Nemotron entry. On modern transformers the directory uses
  # literal hyphens; older versions sometimes used a "_hyphen_" encoding.
  rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  # Older encoding (use whichever matches your `ls` output above):
  # rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA_hyphen_Nemotron*
  ```

  Clearing wipes any in-place edits to `modeling_nemotron_h.py`; you'll need to re-apply or re-derive them after the next `from_pretrained()`. See [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) for what the edits do and why.
