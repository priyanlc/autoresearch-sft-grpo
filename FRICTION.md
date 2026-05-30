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

### F-012 — `adapter_sanity_check.py` hangs between `PeftModel.from_pretrained` and the first `model.generate` call

- **timestamp:** 2026-05-07 20:30 UTC (first attempt) / 20:48 UTC (retry, killed during graceful shutdown)
- **phase:** eval (post-train sanity check)
- **signature:**

  ```
  Loading checkpoint shards: 100%|...| 13/13 [02:26<00:00, 11.25s/it]
  Loading LoRA adapter from ./adapter
  <process hangs here for 17+ minutes>
  ```

  Process state evolves: starts in `Rl` (running multi-thread, ~1140% CPU across ~10 worker threads), then transitions to `Sl` (all threads in `futex_do_wait`). GPU stays at 0% util / ~62 GB used (model loaded into VRAM but no kernels firing). No new stdout. The "Generating (max_new_tokens=512)..." print line never fires, meaning `model.generate(...)` is never reached — the hang is inside the adapter-application path between `PeftModel.from_pretrained(base, ADAPTER_DIR)` and the next Python statement.

- **hypothesized root cause:** Uncertain. The same script + same code path completed in ~5-7 minutes for the T1.16 and T2.7 sanity checks on the same pod. T2.8's adapter is structurally identical (same `LORA_RANK=32`, same `target_modules='all-linear'`, same 880M trainable params, 3.5 GB safetensors file). Three plausible causes:

  1. **MooseFS read-latency variability on the 3.5 GB `adapter_model.safetensors`** — the file lives at `/workspace/autoresearch-sft-grpo/adapter/` (MooseFS-backed). After the file fd was closed (verified via `/proc/<pid>/fd`), python continued at very high CPU for ~7 min then went to sleep. A first-touch FUSE metadata operation could be the trigger, similar in flavour to F-007 (MooseFS wedges wheel extraction) but at a different layer.
  2. **PEFT silent merge / dtype-conversion path** — `PeftModel.from_pretrained` shouldn't merge unless `merge_and_unload()` is called, but it does some tensor placement work that depends on the base model's dtype/device-map. With BF16 base and BF16 adapter, this should be a no-op. Maybe a deserialization edge case.
  3. **Eventfd / CUDA stream wait** — three eventfds were open on the python process (`/proc/<pid>/fd/4`, `17`, `24`); thread states going from Running to futex-wait suggests workers completed, but the main thread never woke up. Could be a missed signal.

- **attempts:**
  - Verified python process alive, multi-threaded, no open files (file reads complete) → not a download/IO stall.
  - Killed PID 24023 cleanly via `SIGTERM` after 18 minutes; GPU memory freed → process was responsive to signals, just stuck waiting.
  - Retried sanity check fresh (`python adapter_sanity_check.py`) → reached the same `Loading LoRA adapter from ./adapter` print and hung again (killed during graceful shutdown).
- **final state:** open. The hang prevented running the methodology-required adapter-on-fresh-base sanity check for T2.8. T2.8's METRIC, per-category breakdown, time, and peak VRAM all printed cleanly to stdout and are recoverable, but the deployment-path validation step (program.md § Validation Contract item 5) was not completed this session.
- **notes:** Next-session investigation:
  - Try the sanity check on a fresh pod (cold MooseFS cache) — if it works, the bug is something about the warm-cache state on this pod after a long session.
  - Try copying the adapter to local overlay disk first (`cp -r adapter /root/adapter-T2.8 && python -c "ADAPTER_DIR='/root/adapter-T2.8'; ..."`) to rule out MooseFS in the adapter-application phase.
  - Try `torch.cuda.synchronize()` calls around the PEFT load to flush any pending CUDA work before adapter application.
  - If reproducible, file an issue with the `peft` repo with the trace + minimal repro.

### F-011 — `_build_dynamic_cot()` regexes for cipher and gravity silently produce garbage training CoT

- **timestamp:** 2026-05-07 14:55 UTC (diagnosis) / fixed by T2.8
- **phase:** sft (training data preparation)
- **signature:**

  Cipher: `_build_dynamic_cot('cipher', prompt, answer)` was supposed to extract `(ciphertext, plaintext)` pairs from prompts like

  ```
  geq bytq lyca tgxkyqt -> the wise king studies
  geq sxjyoxt tgxkqcg tqqt -> the curious student sees
  ...
  ```

  and build a substitution map. The old regex `r'([a-z\s]+?)\s*->\s*([a-z\s]+)'` was greedy across newlines: on a real prompt with 5 cipher/plaintext pairs, it returned 3 matches with pairings like `('\ngeq bytq lyca tgxkyqt', 'the wise king studies\ngeq sxjyoxt tgxkqcg tqqt ')` — pairing one cipher line with the first plaintext line *plus* the start of the next sample. The substitution map built from this is nonsense.

  Gravity: `_build_dynamic_cot('gravity', prompt, answer)` was supposed to compute `g` from one `(t, d)` data point in prompts like

  ```
  In Alice's Wonderland, the gravitational constant has been secretly changed.
  For t = 1.86s, distance = 17.75 m
  For t = 2.98s, distance = 45.55 m
  ...
  Now, determine the falling distance for t = 3.4s given d = 0.5*g*t^2.
  ```

  The old regex `r't\s*=\s*([\d.]+).*?d\s*=\s*([\d.]+)'` (with `re.DOTALL`) greedily walked past the data lines and matched the FIRST `d=` it found, which is **inside the formula `d = 0.5*g*t^2`** at the end of the prompt. So it extracted `t=1.86, d=0.5` and computed `g = 2*0.5/1.86² ≈ 0.2891` instead of using `d=17.75` (true `g ≈ 10.27`). The model trained on `<think>` blocks claiming `g=0.2891`, then reproduced exactly that pattern at inference (verified in T1.16, T2.7 eval debug, and T2.7 sanity check output: `g = 2*0.5/1.86^2 = 0.2891. Apply to find d for the new t. Answer: 58.5`).

- **hypothesized root cause:** The dynamic-CoT path was written when the prompt format was simpler / different. As the prompts evolved, the regexes silently started misfiring — but the call site in `build_sft_text()` falls back to the static `_COT_BY_TYPE` template *only when `_build_dynamic_cot()` returns `None`*. The broken regexes returned successful (non-None) garbage, not None, so the fallback never engaged. The model was trained on bad CoT for cipher and gravity in every run on `main` since this regex shipped.
- **attempts:**
  - Tested both regexes against the actual val split prompts → confirmed cipher returns 3 garbled pairs (instead of 5 clean), gravity returns `('1.86', '0.5')` (instead of `('1.86', '17.75')`).
  - T2.8: replace cipher regex with `r'^([^\n]+?)\s*->\s*([^\n]+?)\s*$'` + `re.MULTILINE` (line-anchored, no greedy newline crossing) → 5 clean pairs. Replace gravity regex with `r't\s*=\s*([\d.]+)s?\s*,?\s*distance\s*=\s*([\d.]+)\s*m'` → matches data lines, ignores the formula. Produces correct `g ≈ 10.26`.
  - Did not change `bit_ops` or `unit_conv` dynamic-CoT branches — both currently 100% on val and their regexes don't show the same failure mode.
- **final state:** resolved by T2.8 (code change in `train.py:_build_dynamic_cot`).
- **notes:** Root-cause for cipher/gravity remaining at 0-20% across the c1bb0a6 baseline, T1.16, and T2.7 runs. Data-volume sweeps couldn't fix this because more training data on broken CoT just reinforces the broken pattern. Strong prior that T2.8 will move both categories — by how much depends on whether the model can actually *use* a clean substitution map / correct g-value at inference, or whether it just memorises the training-time computation. Either way, removing the structural defect is necessary before any other Tier 2 prompt-format experiment can be evaluated meaningfully.

### F-010 — `hf_xet` worker→main thread deadlock during model weights download leaves a permanent `.incomplete` shard

- **timestamp:** 2026-05-06 22:53 UTC (deadlock onset) / 22:58 UTC (diagnosis)
- **phase:** model_load
- **signature:**

  ```
  train.py stdout last line: "Loading model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  HF cache: 19 finalized shards + 1 frozen .incomplete file at 4,995,693,272 bytes,
  mtime 22:53:12, never grew or got renamed.
  python (PID 6515) alive, ~26% CPU, all 119 threads in futex_do_wait, one thread
  in FUSE wait (request_wait_answer). Zero active TCP connections to HF endpoints.
  xet log (/workspace/.cache/huggingface/xet/logs/...) ends at 22:53:11 with a
  successful `Received CAS response status_code=206` — no error, no further
  log lines for 4+ minutes.
  ```

- **hypothesized root cause:** With the pod-level env var `HF_XET_HIGH_PERFORMANCE=1` enabled, `huggingface_hub` routes large downloads through the Rust `hf_xet` client (parallel S3-range chunked downloads). One worker apparently terminated (cleanly, judging by the absence of an error in the xet log) without notifying the main download thread; the main thread is stuck at `futex_do_wait` waiting for a queue event that never arrives. Compounding factor: `/workspace` is MooseFS-backed (FUSE), and the `request_wait_answer` thread suggests xet's final write/rename was waiting on FUSE flush at the time the worker exited. Whether this is xet's bug or a MooseFS-induced manifestation isn't determined; the practical effect is the same — download cannot complete.
- **attempts:**
  - Watched for 4+ min after the freeze → no spontaneous recovery; cache size constant, file mtime constant, write_bytes constant.
  - Inspected open fds (`.incomplete` file open w/o, `.lock` file open) and thread states → confirmed deadlock signature, no active workers.
  - Killed train.py (PID 6515), removed the stuck `*.incomplete` and stale `*.lock` files, **unset `HF_XET_HIGH_PERFORMANCE`** for the retry (kept `HF_HUB_ENABLE_HF_TRANSFER=1` — that's the older `hf_transfer` package, a separate code path that worked fine for the first 12 shards), re-ran `python train.py` → **download progressing again** at last check.
- **final state:** worked-around.
- **notes:** Pre-flight check before any HF model download on RunPod: if both `HF_XET_HIGH_PERFORMANCE=1` and `HF_HUB_ENABLE_HF_TRANSFER=1` are set at the pod level (as on this image), prefer to `unset HF_XET_HIGH_PERFORMANCE` and rely on `hf_transfer` alone — fewer moving parts, no FUSE-tickling parallel S3 ranges. Generalisable: any time multiple competing "fast download" stacks are enabled simultaneously, hangs become a possibility — pick one.
- **Promotion (2026-05-30):** landed in `runpod-setup.md` § 5 as a `[[ -n ... ]] && unset HF_XET_HIGH_PERFORMANCE` guard before `prepare.py`. Done preemptively at user request as part of T1.31 (rather than waiting for the formal second-reproduction threshold from these notes) — the cost of the one-line guard is zero, the cost of hitting F-010 is a wedged ~5 GB partial download + a recovery cycle. The original "second reproduction" gate stands as a methodology default for cases where the workaround is non-trivial; this one is one line.

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
  - **Resolution:** Restored `causal_conv1d` as a hard install-time dep across `requirements.txt`, `bootstrap.sh`, `check_install.py`, `prompt.md`, `program.md` (T1.9 row marked superseded; new T1.14 row added), `BRANCH_NOTES.md`, `runpod-setup.md`, and `docs/fast-path-and-cache.md`. The `train.py:398` runtime fast-path disable still applied at the time of the T1.14 resolution, so the package was dead code at execution time — it just needed to be present so transformers' AST scanner accepts the modeling file. No behavioural change vs the locked baseline at T1.14; F-001's defensive posture unchanged at the time. (Citation update: T2.9 later moved the fast-path disable to `train.py:582` paired with `use_cache=False` at `train.py:579`, applied only before eval — the kernels now run during SFT but the install-time requirement and the F-001 defensive posture are unchanged. See T1.29 chronology.)
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
- **final state:** resolved by T1.17 (added `hf_transfer` to `requirements.txt`).
- **notes:** Pre-flight check before any HF download: `env | grep -E '^HF_HUB_ENABLE_HF_TRANSFER='`; if set, `python -c "import hf_transfer"` must succeed or installs/downloads will silently corrupt the cache. Post-T1.17 `requirements.txt` covers this for the standard install path; the check is still worth running on any pod where the venv was created before T1.17 landed, or if `--no-deps` was used. Generalisable: any time an HF env-var optimization is enabled at the pod level, the matching package must be installed in the venv.

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
- **Second reproduction (2026-05-30):** hit on a fresh RunPod pod (`mfs#us-md-1.runpod.net:9421`, 75% used on a ~333 TB share) during `bash bootstrap.sh`. Failure signature this time: `Failed to extract archive: nvidia_nvjitlink_cu12-12.9.86-...whl` + `I/O operation failed during extraction` + uv suggesting `UV_HTTP_TIMEOUT`. The UV_HTTP_TIMEOUT suggestion is misleading — the bottleneck is extraction, not download. Recovery: `rm -rf /workspace/.venv /workspace/.cache/uv /workspace/autoresearch-sft-grpo/.venv && uv venv /root/venv-autoresearch && export UV_CACHE_DIR=/root/uv-cache && ln -s /root/venv-autoresearch /workspace/autoresearch-sft-grpo/.venv && source .venv/bin/activate && bash bootstrap.sh` — completed in ~2s after the move. This second reproduction met the "promote workaround if reproduces" threshold from these notes; landed as **T1.30** (pre-flight + decision tree in `runpod-setup.md` § 2).

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

- **hypothesized root cause:** Eval uses `enable_thinking=True` so the model fills `<think>` with native reasoning. Without SFT teaching the think→`\boxed{}` pattern, the model exhausts its budget on thinking and never closes with a boxed answer. SFT with `USE_COT=True` (`train.py:81`, used at line 165) injects static CoT templates inside `<think>` followed by `\boxed{}`, teaching the closing pattern.
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
  - Reverted to plain `torch_dtype=torch.bfloat16` (`train.py:392`) → METRIC 0.5333.
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
  - try/except around `grpo_trainer.train()` so SFT-only completes (`train.py:520-540`) → worked: SFT adapter is preserved; GRPO failure is non-fatal.
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
  - Set `model.config.use_cache = False` before `evaluate_model(...)` (`train.py:579` post-T2.9) → worked for SFT eval; eval is slow (~3 hours for 30 samples) without cache but produces a correct METRIC.
- **final state:** worked-around (cache disabled at eval; GRPO still blocked, see F-002).
- **notes:** The in-place edit to the HF cache module is per-pod; clearing the `transformers_modules` cache wipes it. Re-run the same edits or re-derive on each fresh pod. Long term, an upstream PR to the model card is the real fix.

  **Impact / cost of the `use_cache=False` workaround** (logged 2026-05-07):

  Running eval with no KV cache means every decoded token re-runs the full forward over all prior tokens — roughly O(N²) instead of O(N) per generation. For a 512-token completion budget this is a 100×–500× slowdown on the per-token cost.

  Concrete numbers from this branch's hardware profile (A100 80GB, BF16):

  | Phase | With cache + fast path | Without cache (current) | Multiplier |
  |---|---|---|---|
  | Eval (30 samples, 512 tokens each) | ~10 min (model-load dominated) | ~3 h (generation dominated) | ~18× |
  | GRPO rollout (one round-trip) | ~1 min | ~30 min | ~30× |
  | Total per `train.py` run | ~1.5 h | ~4.5 h | ~3× |

  Eval is the wall-clock dominant phase of every run on `main`; SFT itself is only ~1 h. The 3-hour eval is the price paid for correctness — the cache bugs would otherwise produce silent generation errors or crashes, neither of which a `\boxed{}`-extraction METRIC can recover from.

  Knock-on effect: F-002 (GRPO punted on Mamba/MoE+TRL tensor mismatch) is *partly* a TRL compatibility problem and *partly* a cache-cost problem — even if TRL were fixed, ~30-min rollouts make GRPO uneconomical at this branch's iteration cadence. Restoring the cache changes that calculus entirely. See [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) § "Speed gains worth quantifying" for the full sequence of unwinds when an upstream fix lands.

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
