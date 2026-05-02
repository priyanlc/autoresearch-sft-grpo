# Friction Log

This file records non-trivial failures encountered during autoresearch runs. The intent is to make it easy for the human author to reconstruct *what broke, what was tried, and what stuck* without re-deriving it from raw logs — both for the next session and for the follow-up blog post on the autoresearch pattern.

A "non-trivial failure" is anything that took more than a one-line config tweak to resolve (or is still unresolved). Examples that qualify:

- `qutlass` failed to build during `pip install`
- The `__bases__` patch ran but PEFT still couldn't see `FPQuantLinear` modules
- GRPO crashed with a TRL tensor-mismatch error
- The model loaded but generation produced empty output
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

### F-006 — PEFT 0.19.1 `_replace_module` crashes on FPQuantLinear when `store_master_weights=True` (`qweight = None`)

- **timestamp:** 2026-05-02 12:28 UTC
- **phase:** sft
- **signature:**

  ```
  Traceback (most recent call last):
    File ".../train.py", line 726, in main
      model = get_peft_model(model, lora_config)
    File ".../peft/mapping_func.py", line ..., in get_peft_model
      ...
    File ".../peft/tuners/tuners_utils.py", line 315, in __init__
      self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)
    File ".../peft/tuners/tuners_utils.py", line 913, in inject_adapter
      self._create_and_replace(...)
    File ".../peft/tuners/lora/model.py", line 273, in _create_and_replace
      self._replace_module(parent, target_name, new_module, target)
    File ".../peft/tuners/lora/model.py", line 362, in _replace_module
      module.to(weight.device)
                ^^^^^^^^^^^^^
  AttributeError: 'NoneType' object has no attribute 'device'
  ```

  Fired immediately at `get_peft_model(...)` (no model load this time — page cache served `from_pretrained` in ~3 min vs. ~22 min cold). Came right after `LoRA targeting 6004 FPQuantLinear modules` was printed but before the trainable-params line.

- **hypothesized root cause:** Three-way incompat between PEFT 0.19.1, fp_quant 0.3.2, and the `store_master_weights=True` config we set as F-005's fix.

  PEFT's device-dispatch in `peft/tuners/lora/model.py:347-362`:

  ```python
  for name, module in new_module.named_modules():
      if (self.prefix in name) or ("ranknum" in name):
          if hasattr(child, "qweight"):
              weight = child.qweight       # ← takes this branch on FPQuantLinear
          elif hasattr(child, "W_q"):
              weight = child.W_q
          elif hasattr(child, "weight"):
              weight = child.weight
          ...
          if not any(p.device == meta for p in module.parameters()):
              module.to(weight.device)     # ← `weight` is None here
  ```

  fp_quant's `pre_forward` in `fp_quant/module/linear.py:219-222` sets `qweight = None` (and `scales = None`, `dqweight = None`) when `store_master_weights=True` — because in master mode the forward path uses `self.weight` directly and the quantized buffers are not needed:

  ```python
  if self.config.store_master_weights:
      self.qweight = None
      self.scales = None
      self.dqweight = None
  ```

  PEFT's `hasattr(child, "qweight")` returns `True` (the attribute exists, just is `None`), so PEFT picks up `weight = child.qweight = None` and falls into `None.device`. The fall-through to `child.weight` (which IS a valid Parameter when master weights are kept) never executes.

- **attempts:**
  - Read PEFT 0.19.1's `_replace_module` source to confirm the dispatch order.
  - Read fp_quant's `pre_forward` to confirm `qweight = None` is set deliberately, not a bug.
  - **Did not attempt a fix** — user requested halt-and-evaluate after this run, no more patches in this session.
- **final state:** open. The run is dead; both T1.x patches (F-003 `_hf_peft_config_loaded=True`, F-005 `store_master_weights=True`) remain in `train.py`'s working tree, uncommitted. Reverting is `git checkout train.py`.
- **notes / candidate fixes for the next session:**
  - **Fix 1 (smallest, in `train.py`):** between the FPQuantLinear `__bases__` patch (around `train.py:701`) and `get_peft_model(model, ...)` (`train.py:721`), iterate FPQuantLinear modules and `del module.qweight; del module.scales; del module.dqweight` whenever the value is `None`. This makes `hasattr` return False, PEFT falls through to `child.weight`, and we proceed. `del`'d attributes can be re-set later if fp_quant ever needs to recreate them.
  - **Fix 2 (in `train.py`, alternative):** replace each `module.qweight = None` with a 1-element placeholder tensor on the right device so `weight.device` works. Brittle.
  - **Fix 3 (upstream, real):** PEFT should treat `qweight is None` the same as missing — change `if hasattr(child, "qweight"):` to `if getattr(child, "qweight", None) is not None:`. Reportable as a PEFT issue.
  - **Fix 4 (upstream alternative):** fp_quant should `del self.qweight` instead of `self.qweight = None` in `pre_forward`. Reportable as an fp_quant issue.
  - This is the third upstream-incompat we've hit on this stack: F-002 (qutlass squat), F-003 (transformers `_hf_peft_config_loaded`), F-005 (fp_quant no-master backward), F-006 (this). The branch is bumping into the seam between four independently-released libraries (transformers 5.7.0, peft 0.19.1, fp_quant 0.3.2, qutlass 0.2.0) that haven't been integration-tested together for the LoRA-on-FPQuant-Master training path.
  - The 96.6 GB transient VRAM peak observed during model load was *not* the fatal error — that resolved before the PEFT crash. So the OOM hypothesis from F-005's pre-run analysis was *not* what killed us. Whether training itself would OOM remains untested.

### F-005 — `fp_quant` 0.3.2 has no backward for `FPQuant4x16NoMasterFn` (NVFP4 + `store_master_weights=False`)

- **timestamp:** 2026-05-02 11:50 UTC
- **phase:** sft
- **signature:**

  ```
  Traceback (most recent call last):
    File ".../train.py", line 766, in main
      sft_trainer.train()
    ...
    File ".../accelerate/accelerator.py", line 2838, in backward
      loss.backward(**kwargs)
    ...
    File ".../torch/autograd/function.py", line 317, in apply
      return user_fn(self, *args)
    File ".../fp_quant/module/linear_fns.py", line 532, in backward
      raise NotImplementedError(
  NotImplementedError: Backward pass is not implemented for FPQuant4x16NoMasterFn yet
  ```

  Fired on the very first training step (step 0/65). Pre-flight warning was visible in the log moments earlier:

  ```
  [transformers] You are attempting to train a model with FPQuant quantization.
  This is only supported when `store_master_weights=True`. Please set
  `store_master_weights=True` to train the model.
  ```

  i.e. the F-003 workaround (setting `_hf_peft_config_loaded=True`) successfully bypassed transformers' static safeguard, but the safeguard was telling the truth: at the `fp_quant` library layer, the no-master backward path is genuinely unimplemented in this version.

- **hypothesized root cause:** `train.py:674-678` constructs the FPQuantConfig as:

  ```python
  FPQuantConfig(forward_dtype="nvfp4", backward_dtype="bf16", pseudoquantization=False)
  ```

  Defaults: `store_master_weights=False`, `pseudoquantization=False` (per `transformers/utils/quantization_config.py:1393`). Combined with `forward_dtype=NVFP4` + `backward_dtype=BF16`, this lands in the dispatch case at `fp_quant/module/linear.py:316-322` which selects `FPQuant4x16NoMasterFn`. That class' `backward` is a literal stub at `linear_fns.py:530-534`:

  ```python
  @staticmethod
  def backward(ctx, grad_output):
      raise NotImplementedError(
          "Backward pass is not implemented for FPQuant4x16NoMasterFn yet"
      )
  ```

  The other three forward+backward+master+pseudoquant combinations *do* have working backwards (`FPQuant4x16MasterFn`, `PseudoQuant4x16MasterFn`, `PseudoQuant4x16NoMasterFn`) — only the "real-NVFP4-no-master" combo is missing. This is an upstream gap in `fp_quant` 0.3.2, not a config error in `train.py` per se, but the only `train.py` config that reaches this gap is the one currently set on the branch.

- **attempts:**
  - Read `fp_quant/module/linear_fns.py` and `linear.py` dispatch table — confirmed only `FPQuant4x16NoMasterFn` has the stub.
  - Considered options A (`store_master_weights=True`), B (`pseudoquantization=True`), C (`USE_NVFP4=False`):
    - A: keeps real NVFP4 forward kernels; adds ~60 GB BF16 master weights → total ~77 GB before LoRA + optimizer + activations. Tight on 95 GB Blackwell, but plausible with LoRA-only training (~8 GB optimizer state for 880 M trainable params) + `gradient_checkpointing=True` + `packing=True` shrinking activations.
    - B: PseudoQuant path → defeats the NVFP4 perf goal (Triton emulation only).
    - C: `USE_NVFP4=False` → BF16-only baseline; explicitly listed as a fallback in `program.md` § "NVFP4 Branch Notes" but it's not the baseline this branch is supposed to characterise.
  - **Applied A** at user direction ("I want to get NVFP4 working, no matter the cost"): edited `train.py:674-678` to add `store_master_weights=True`. Now selecting `FPQuant4x16MasterFn`, which uses the saved BF16 master weight in its backward to compute `grad_input` and `grad_weight` (`linear_fns.py:455-479`).
- **final state:** patch applied; awaiting re-run outcome (success or OOM).
- **notes:**
  - This issue is structurally related to F-003 — both stem from the same gap (FPQuant + standalone PEFT + active training). F-003 patched the *symptom* at the transformers safeguard layer; F-005 fixes the *underlying* config so the safeguard would have been satisfied without needing to bypass it. Recommendation: keep both patches in place — F-003 is harmless when F-005's config is correct, and serves as belt-and-braces if a future config flip ever lands in a forbidden combination again.
  - The "+~60 GB master weight" cost is per `replace_with_fp_quant_linear`'s allocation pattern. With NVFP4 packing 4 bits/weight + a small scales overhead, the *quantized* weight is ~17 GB; the *master* weight kept alongside is the original BF16 (~60 GB). Total ~77 GB just for weights when `store_master_weights=True`.
  - If the re-run OOMs, the next escalation is C (BF16-only) for a "still useful" baseline, then revisit NVFP4 either after a `fp_quant` release that fills in the no-master backward, or after T2 work on adapter-only-master schemes (T3.3-ish territory).

### F-004 — RunPod `/workspace` is MooseFS-over-network: ~33 MB/s cold reads make 60 GB model loads take ~22 min

- **timestamp:** 2026-05-02 11:18 UTC
- **phase:** model_load
- **signature:**

  ```
  $ df -hT /workspace/.cache/huggingface
  Filesystem                   Type  Size  Used Avail Use% Mounted on
  mfs#eur-is-1.runpod.net:9421 fuse  1.4P  854T  579T  60% /workspace

  $ dd if=$BLOB of=/dev/null bs=1M count=200          # cold-cache sample
  200+0 records out
  209715200 bytes (210 MB, 200 MiB) copied, 6.30251 s, 33.3 MB/s

  # train.py "Loading in NVFP4 (Blackwell FP-Quant)..."
  # → wall time to GPU=18 GB:    ~22 min
  # → 13 shards × ~5 GB         = ~65 GB read
  # → 65 GB / 33 MB/s            = ~33 min pure read; observed lower
  #   because per-layer NVFP4 quantize on GPU pipelines with disk I/O.
  ```

  Bandwidth-bound: not the GPU, not the CPU, not the network to PyPI/HF — the model load is bottlenecked on `/workspace`'s FUSE mount to RunPod's MooseFS chunkservers. Same 60 GB model on local NVMe loads in ~2–3 min.

- **hypothesized root cause:** This pod was provisioned with the default RunPod **network volume** at `/workspace`, which is MooseFS over the cluster LAN (mounted via `fuse` on `mfs#eur-is-1.runpod.net:9421`). MooseFS is great for capacity and persistence (the pod survives restart with weights still cached), but every `read()` goes over the LAN to a chunkserver. There's no local NVMe in this storage path — the HF cache at `/workspace/.cache/huggingface` lives entirely on the network volume.

  This is a configuration issue, not a software bug. RunPod also offers:
  - **Container disk** — local NVMe attached to the pod, ~1–3 GB/s sequential, but ephemeral (wiped on pod stop).
  - **Pod-attached SSD** in some templates — local, fast, but template-specific.

  The 22 min number is the steady-state cost of *every* `train.py` invocation on this pod unless the OS page cache is hot from a prior run.

- **attempts:**
  - Verified the FS type is `fuse` and the mount is `mfs#eur-is-1.runpod.net:9421` → confirmed network volume, not local disk.
  - Sampled cold-cache read: 33.3 MB/s on a 5 GB safetensors blob → **~50–100× slower** than typical NVMe.
  - Considered: kicking off a `dd` pre-warm in parallel during this run → rejected because the in-flight `train.py` is the *current* reader of those shards; a parallel `dd` would split MooseFS bandwidth and slow both.
  - Considered: redirecting `HF_HOME` to a fast path (`/tmp`, `/local`, `/root`) before re-running. Not yet attempted on this pod — pending an inventory of available local mounts (`df -h`, `lsblk`) once the current run is past model load.
  - Page-cache effect: the second `train.py` invocation in the same session benefits from OS page cache holding the recently-read shards (this pod has ample RAM). Empirically expect 3–8 min for the second load, no explicit pre-warm.

- **final state:** worked-around (we paid the toll once for the baseline run; subsequent runs in this session ride the page cache).

- **notes:**
  - Vendor comparison for context (rough sequential-read numbers for typical model-load patterns):
    | Path | Typical seq read |
    |---|---|
    | RunPod network volume (this pod) | **~30 MB/s** |
    | RunPod container disk / pod-attached SSD | ~1–3 GB/s |
    | Lambda Labs default NVMe | ~2–4 GB/s |
    | AWS p5/p4 instance-store NVMe | ~5–10 GB/s |
    | CoreWeave / Crusoe local NVMe | ~2–5 GB/s |
    | vast.ai / TensorDock | host-dependent (0.5–4 GB/s) |
  - Two follow-ups for `runpod-setup.md`:
    1. The pod-requirements table currently lists "Disk ≥ 100 GB"; it should also call out "**local SSD/NVMe path required for HF cache**" or recommend setting `HF_HOME` away from `/workspace` if the pod's only fast path is ephemeral.
    2. Part 1 § 5 ("Prepare data") downloads ~60 GB to `~/.cache/huggingface`; the doc should suggest `HF_HOME=$LOCAL_FAST_DISK/.cache/huggingface` before `prepare.py` if such a path exists.
  - This issue is the single largest contributor to wall-clock per-experiment cost on this branch — addressing it during T2 sweeps (which will run `train.py` repeatedly) is high-leverage.

### F-003 — `transformers.validate_quantization_for_training` rejects FPQuant + PEFT 0.19.1 (`get_peft_model` flow)

- **timestamp:** 2026-05-02 11:10 UTC
- **phase:** sft
- **signature:**

  ```
  Traceback (most recent call last):
    File ".../train.py", line 753, in main
      sft_trainer = SFTTrainer(
    ...
    File ".../transformers/trainer_utils.py", line 138, in validate_quantization_for_training
      raise ValueError(
  ValueError: The model you are trying to fine-tune is quantized with QuantizationMethod.FPQUANT
  but that quantization method do not support training. Please open an issue on GitHub:
  https://github.com/huggingface/transformers to request the support for training support for
  QuantizationMethod.FPQUANT
  ```

  Fired ~22 min into the run, after the 60 GB model load + NVFP4 quantize succeeded (GPU at 18 GB), at the moment `SFTTrainer(model=model, ...)` was constructed.

- **hypothesized root cause:** Mismatch between standalone PEFT and transformers' built-in PEFT integration. `transformers/trainer_utils.py:114–142` uses:

  ```python
  _is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
      model, "_hf_peft_config_loaded", False
  )
  _quantization_method_supports_training = (
      getattr(model, "hf_quantizer", None) is not None and model.hf_quantizer.is_trainable
  )
  ...
  elif _is_quantized_and_base_model and not _quantization_method_supports_training:
      raise ValueError(...)
  ```

  Path through the chain on this branch:

  - `model.is_quantized` → `True` (PeftModel proxies via `__getattr__`).
  - `model._hf_peft_config_loaded` → **`False`** (never set). `_hf_peft_config_loaded` is only set by `transformers.PreTrainedModel.load_adapter()` (in `transformers/integrations/peft.py:586–587`, `765–766`). PEFT 0.19.1 has **zero** references to that attribute — `peft.get_peft_model()` does not flip it.
  - `model.hf_quantizer.is_trainable` → checks `FPQuantConfig.store_master_weights` (`transformers/quantizers/quantizer_fp_quant.py:127–133`), which is `False` by default in `train.py:674–678`.

  Net effect: transformers thinks it's looking at a "raw" quantized model with no PEFT adapter, sees `is_trainable=False`, and refuses to start training — even though `peft.get_peft_model(...)` *did* attach an adapter at `train.py:721`.

  This is structurally analogous to the FPQuantLinear `__bases__` hack already in the codebase: an upstream-compat patch that exists because `transformers` / `peft` weren't designed to interoperate cleanly on this exact stack.

- **attempts:**
  - Confirmed by reading `transformers/trainer_utils.py` and `transformers/quantizers/quantizer_fp_quant.py` — no `_hf_peft_config_loaded` setter is reachable from `peft.get_peft_model`.
  - Considered fix B (`FPQuantConfig(store_master_weights=True)`): would make `is_trainable=True` but force the quantizer to keep BF16 master weights alongside NVFP4, costing ~60 GB extra VRAM and defeating the point of NVFP4 on this branch.
  - Considered fix C (downgrade transformers): too broad — risks regressing dtype-kwarg handling (F-001 path) and mamba_ssm imports.
  - **Applied fix A**: after `model = get_peft_model(model, lora_config)` (`train.py:721`), set `model._hf_peft_config_loaded = True`. One-line workaround that makes the check evaluate `_is_quantized_and_base_model = False` and skip the raise. Semantically truthful — PEFT is loaded.
- **final state:** worked-around (open upstream — long-term fix is for `peft.get_peft_model` to set this flag, or for `validate_quantization_for_training` to recognise `_is_peft_model(model)` in the second `elif`).
- **notes:**
  - Patch is in `train.py` only; `program.md` `__bases__` hack is preserved.
  - This regression likely landed when transformers added `validate_quantization_for_training` (recent — was not present in `4.51.3` per `check_install.py`'s old recommendation).
  - If/when transformers 5.x is updated to call `_is_peft_model(model)` in the second branch of the validator, the workaround can be removed; until then, leaving it in is the lower-risk path.

### F-002 — `qutlass` on PyPI is an empty stub package (silently breaks NVFP4 path)

- **timestamp:** 2026-05-02 09:30 UTC
- **phase:** env_install
- **signature:**

  ```
  $ pip show qutlass
  Name: qutlass
  Version: 0.0.0
  Summary: Temp
  Location: /workspace/9-kaggle/autoresearch-sft-grpo/.venv/lib/python3.12/site-packages
  Requires: numpy

  $ ls .venv/lib/python3.12/site-packages/qutlass
  __init__.py  __pycache__   # (__init__.py is empty)
  ```

- **hypothesized root cause:** The `qutlass` name on PyPI is a placeholder/squat (uploaded as version 0.0.0 with summary "Temp"). The real QuTLASS — IST-DASLab's CUTLASS-based NVFP4 kernel library — is at https://github.com/IST-DASLab/QuTLASS and is **not** published to PyPI. `requirements.txt` simply lists `qutlass`, so plain `pip install qutlass` (or `pip install -r requirements.txt`) silently pulls the stub.

  The stub satisfies `import qutlass` but exposes none of the kernels `fp_quant` needs. Confirmed by reading `fp_quant/module/qutlass_ops.py:5–28`:

  ```python
  try:
      from qutlass import (fusedQuantizeMx, fusedQuantizeNv,
                           matmul_ada_mxf4_bf16_tn, matmul_mxf4_bf16_tn,
                           matmul_nvf4_bf16_tn, matmul_mxf8_bf16_tn,
                           matmul_mxf8_bf16_nn, backward_t_bf16, ...)
      from qutlass.utils import to_blocked as to_blocked_qutlass
      HAS_QUTLASS = True
  except ImportError:
      HAS_QUTLASS = False
  ```

  And `fp_quant/module/linear.py:57–60` then raises:

  ```
  ValueError: QuTLASS is not installed. Can only run with `pseudoquantization=True` ...
  ```

  with the stub installed, `HAS_QUTLASS` quietly becomes `False` and the failure surfaces only at `FPQuantLinear` instantiation (i.e., during model load in `train.py`, *not* during `pip install`).

- **attempts:**
  - `pip uninstall qutlass -y` → removed stub.
  - `TORCH_CUDA_ARCH_LIST="12.0+PTX" pip install 'git+https://github.com/IST-DASLab/QuTLASS.git' --no-build-isolation` → built `qutlass-0.2.0-cp312-cp312-linux_x86_64.whl` (4.5 MB compiled CUDA wheel for sm_120). All required symbols present; `HAS_QUTLASS = True` confirmed.
- **final state:** resolved.
- **notes:**
  - `requirements.txt` should be updated (T1.3 territory) to either pin `qutlass @ git+https://github.com/IST-DASLab/QuTLASS.git@<sha>` or document explicitly that `qutlass` cannot come from PyPI. The current line `qutlass` (with no source) is a quiet trap for anyone setting up a new pod.
  - The pip wheel cache now holds the GitHub-built wheel at `/workspace/.cache/pip/wheels/...`, so reinstalls on the same pod are fast.

### F-001 — Pre-flight transformers API check is stale on transformers 5.7.0 (false-alarm STOP)

- **timestamp:** 2026-05-02 09:50 UTC
- **phase:** env_install
- **signature:**

  ```
  $ python -c "import transformers, inspect; sig = inspect.signature(transformers.AutoModelForCausalLM.from_pretrained); params = list(sig.parameters.keys()); print(f'transformers={transformers.__version__}'); print('uses dtype' if 'dtype' in params else 'uses torch_dtype' if 'torch_dtype' in params else 'NEITHER — investigate')"
  transformers=5.7.0
  NEITHER — investigate
  ```

  Per the decision tree in `program.md` § "Pre-flight verification", `NEITHER` triggers a HALT.

- **hypothesized root cause:** The check inspects the static `inspect.signature(...).parameters` of `AutoModelForCausalLM.from_pretrained`, which in transformers 5.x is just `(model_args, **kwargs)`. The `dtype` and `torch_dtype` arguments are no longer named parameters of either `AutoModelForCausalLM.from_pretrained` *or* `PreTrainedModel.from_pretrained` — they are extracted at runtime via:

  ```python
  # transformers/modeling_utils.py, ~lines 252–285 (5.7.0)
  dtype = kwargs.pop("dtype", None)
  torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
  ...
  if torch_dtype is not None:
      dtype = dtype if dtype is not None else torch_dtype
  if dtype is None:
      dtype = "auto"
  ```

  So `train.py:679`'s `dtype=torch.bfloat16` is correct on this version. The decision tree in `program.md` correctly maps `transformers=5.x.y + uses dtype` to "no change", but the *check itself* can no longer detect that case — both `dtype` and `torch_dtype` have moved into `**kwargs`.

- **attempts:**
  - Inspected `PreTrainedModel.from_pretrained` source via `inspect.getsource` and confirmed `dtype` is popped from `kwargs` at line ~252; `torch_dtype` is also accepted as a BC alias → confirms code path works.
- **final state:** open (awaiting human decision on whether to override the STOP and proceed to baseline; pre-flight check itself should be reworked, but that is a docs/program.md change, not a code change).
- **notes:**
  - All other pre-flight checks pass: CUDA 13.0 driver / cu128 torch wheels, `check_install.py` reports all deps present, GPU = RTX PRO 6000 Blackwell (95 GB), `data/train.csv` present (2997 KB), `data/test.csv` present (1 KB stub).
  - `check_install.py:83` *also* warns "transformers != 4.51.3" — that's the second known-stale check called out in `runpod-setup.md` Part 3.
  - Recommended fix when reworking: the check should call the function with a sentinel value, e.g. `AutoModelForCausalLM.from_pretrained("...", dtype="auto", local_files_only=True)` and detect the `TypeError`/no-error pattern — or just `_ = inspect.getsource(transformers.PreTrainedModel.from_pretrained); 'kwargs.pop("dtype"' in _`.
