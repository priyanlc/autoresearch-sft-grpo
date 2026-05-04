---
title: Nemotron-3-Nano Fast Path & HybridMambaAttentionDynamicCache — Why F-001 Is Load-Bearing
type: note
status: active
updated: 2026-05-04
related: [
  methodology.md,
  bf16-sft-only-plan.md,
  ../FRICTION.md,
  ../program.md,
  ../BRANCH_NOTES.md,
]
---

> **Origin note:** Originally drafted in a separate documentation vault as `wiki/nemotron-fast-path-and-cache.md`. This in-repo copy at `docs/fast-path-and-cache.md` is the canonical version going forward.

# Nemotron-3-Nano Fast Path & `HybridMambaAttentionDynamicCache`

A deep dive on the relationship between the Mamba/Conv1d CUDA fast-path kernels, Nemotron's hybrid cache class, and the workarounds carried in `autoresearch-sft-grpo:main`. Anchored to FRICTION entry **F-001**.

If you only need the operational summary, read [`../BRANCH_NOTES.md`](../BRANCH_NOTES.md) Patches table and [`../FRICTION.md`](../FRICTION.md) F-001. This page exists for the technical context behind those entries — what's actually broken, why both `train.py:386` and `train.py:536` are needed (not redundant), and what concretely changes when F-001 resolves upstream.

## What's broken (F-001, in detail)

The Nemotron-3-Nano model card ships a custom `modeling_nemotron_h.py` (loaded via `trust_remote_code=True`) that includes a `HybridMambaAttentionDynamicCache` class. This class is the per-request state container during generation — it holds **three different things in parallel**, one for each layer type in the 52-layer hybrid stack:

| Layer type | Count | What's cached | Shape concern |
|---|---|---|---|
| Mamba-2 | 23 | SSM hidden state + a small ring buffer of the last few conv inputs | `conv_dim` must match the model's actual conv channel count |
| MoE | 23 | Nothing (stateless feed-forward) | — |
| GQA attention | 6 | Standard K and V per token | Standard transformer KV-cache shape |

The 0.5333-baseline session (`autoresearch-sft-grpo:main`, commit `c1bb0a6`, 2026-04-06) surfaced these specific bugs in that class:

1. **Missing `conv_kernel_size` attribute** — code paths read `cache.conv_kernel_size` but the class never set it. `AttributeError`.
2. **`.device` on a list** — when looking up the device of cached tensors, the code did something like `cache.tensors.device` but `tensors` was a Python list, not a tensor stack.
3. **`conv_dim` shape mismatch** — the cache allocated `conv_dim=4096` but the actual Mamba blocks expected `6144` (some `hidden_size × expand` calculation that the class hardcoded wrong).
4. **SSM state shape mismatches** — analogous to (3) for the recurrent state.

These all manifest as crashes when *anything* tries to read or write the cache. Generation (autoregressive decode) does this on every token. KV cache during eval does this. GRPO rollout does this.

## Why the fast path can't work without a working cache

The "fast path" in HF's Mamba implementations isn't just "faster math on the same tensors." It's a fundamentally different decode strategy.

**Slow path (full forward each step):**

```
for each new token:
    run the entire 52-layer forward pass on [input, *all_previous_tokens]
    output the next token
```

At every decode step, every layer recomputes everything from scratch. Quadratic in sequence length. Doesn't need a cache because nothing is remembered between steps.

**Fast path (incremental update):**

```
allocate a per-layer cache (conv buffer + SSM state + KV)
for each new token:
    layer.update(token, cache[layer])    # reads and mutates the cache
    output the next token
```

At every decode step, each layer reads its tiny cached state, applies a single-token update (this is where `causal_conv1d_update` and `selective_state_update` live), writes the new state back, and moves on. Linear in sequence length.

The fast path is *structurally built around the cache*. There's no way to turn it on without the cache being correct, because the kernels' inputs and outputs flow through cache slots. So when `HybridMambaAttentionDynamicCache` has shape bugs, the fast path either crashes (shape mismatch) or silently corrupts (wrong device, wrong layout).

## Why `train.py` carries two workarounds, not one

| Line | Workaround | What it bypasses |
|---|---|---|
| `train.py:386` | `mod.is_fast_path_available = False` (loops `sys.modules` for `modeling_nemotron_h`) | Forces pure-PyTorch math even where the fused CUDA kernels would otherwise run |
| `train.py:536` | `model.config.use_cache = False` before eval | Makes generation re-do the full forward each step instead of touching the cache |

Both are needed. The fast path won't even be *selected* if `use_cache=False` (one code path), but if `use_cache=True` is accidentally re-enabled somewhere downstream and the fast-path disable were removed, you'd hit the same bugs from a different angle. The pair are redundant defenses that together guarantee no code path touches the broken cache.

This is the operational reality reflected in `autoresearch-sft-grpo:main`'s `BRANCH_NOTES.md` Patches table and `program.md` § "Patches in `train.py` — DO NOT REMOVE."

## What an upstream fix looks like

Two realistic shapes:

**Shape A — model card patch.** NVIDIA pushes a corrected `modeling_nemotron_h.py` to `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` on Hugging Face. The class definition gets:

- `self.conv_kernel_size = config.conv_kernel` set in `__init__`
- `.device` resolved against the actual underlying tensor, not the wrapping list
- `conv_dim` derived from `config.hidden_size * config.expand` (or whatever the correct formula is) instead of hardcoded
- SSM state shapes recomputed from `config.state_size` consistently

A `git pull` of the HF cache picks this up next time `from_pretrained()` runs.

**Shape B — `transformers`-side fix.** If the bugs are in `transformers`'s shared cache plumbing rather than the model card's custom code, a `transformers` release pins the fix. Less likely here because the cache class is defined in the model card's custom file (it's prefixed `Hybrid…` because it's specific to Nemotron's hybrid architecture), but possible if the bug ever gets traced to `transformers.cache_utils`.

In practice the fix lands wherever the file lives. Today it lives in the model card; tomorrow it could be promoted into `transformers.models.nemotron_h` if the architecture goes mainline.

## What `main` would change when F-001 resolves

Concretely, this set of edits:

1. **Drop the fast-path disable** (`train.py:386–389`).
2. **Drop `model.config.use_cache = False`** before eval (`train.py:536`).
3. **Drop the in-place edits to `modeling_nemotron_h.py`** that the prior session applied to the HF cache to keep the slow path from crashing on tangentially related issues. Those workarounds become obsolete and should be removed so they don't shadow the upstream fix.
4. **Update FRICTION F-001** to `final state: resolved`, with a note pointing at the upstream fix commit.
5. **Update `BRANCH_NOTES.md` Patches table** to drop the row for `use_cache = False`.
6. **Re-add `causal_conv1d`** to `requirements.txt` if it was previously dropped — at this point it stops being inert and starts mattering.
7. **Re-evaluate `EVAL_BATCH_SIZE=1` → larger** since memory pressure was driven partly by no-cache eval re-running the full forward each step.

Each lands as its own `T1.x` commit; the regression bar (METRIC ≥ 0.5333) still applies, but the *expected* outcome is METRIC ≈ 0.5333 on accuracy and roughly a 10–20× speedup on eval.

## Speed gains worth quantifying

Today's eval on `main`: 30 samples × ~3 minutes per sample (no cache, regenerating from scratch token-by-token) = ~90 minutes the methodology rounds up to "3 hours" because shard reload and model load add overhead. The wall-clock dominant phase per run.

With cache + fast path enabled:

- Each token's decode drops from "rerun 52 layers on N tokens" to "update 52 layers' cache slots + one token of forward." For a 512-token generation budget, this is a 100×–500× speedup on the per-token cost.
- 30-sample eval would land somewhere between 5 and 15 minutes, dominated by the model load itself rather than generation.
- **Knock-on effect: GRPO becomes feasible again.** F-002 was punted *partly* because of the TRL tensor mismatch on Mamba/MoE but also because each GRPO rollout took ~30 minutes without cache (one rollout = generate completions, score, repeat). With cache restored, the 30-min figure drops to ~1 minute and the cost calculus for re-attempting GRPO changes entirely.

## Why the two CUDA packages are not symmetric

Worth calling out so readers don't conflate them:

| Package | Status on `main` | Why |
|---|---|---|
| `mamba_ssm` | **Required** to load the model at all | The modeling file does an unconditional `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn` wrapped in `try/except ImportError: raise`. This is the layer's RMSNorm, used regardless of fast-path setting. Different code path, different requirement profile. |
| `causal_conv1d` | **Optional and currently inert** | All `causal_conv1d` imports are conditional with graceful `None` fallback. When absent, `is_fast_path_available` is automatically `False` — same as what `train.py:386` was forcing. Becomes load-bearing again only when F-001 resolves and the fast path is re-enabled. |

The two packages historically ship together (both are part of the Mamba ecosystem), which makes them feel symmetric. They are not. Inside the Nemotron modeling file, `mamba_ssm` does double duty: it provides the fast-path SSM kernels *and* the RMSNorm primitive. Only the kernel side is gated by `is_fast_path_available`; the RMSNorm import is unconditional.

## When this might realistically happen

Three plausible triggers:

1. **NVIDIA pushes a fix proactively.** The bugs are pretty obvious once anyone exercises the cache path — likely surfaced on their internal regression tests for the Omni or VL variants of the same family. A fix on one model card often back-ports to siblings.
2. **A community PR.** The Hugging Face "Community" tab on the model card lets external contributors propose fixes; NVIDIA reviewers merge them. Someone hitting the same bugs and motivated to fix upstream is a low-cost path.
3. **Kaggle competitor pressure.** This is a Kaggle competition — if enough teams hit F-001 and complain, the model card gets attention. The competition forum + HF Community pages tend to resolve this kind of issue within days once it's loud enough.

Practically, "F-001 gets resolved" is most likely either weeks-out (community-pressure timeline) or never on this exact model card (if NVIDIA decides Nemotron-3-Nano is locked and ships fixes only to the next generation). The defensive posture in `main` — fast-path disabled, no-cache eval, BF16-only — is built to work either way.

## See also

- [`bf16-sft-only-plan.md`](bf16-sft-only-plan.md) — The strategic plan that anchors `main`'s configuration; references F-001 in its "What is locked" table.
- [`methodology.md`](methodology.md) — The 8-artefact autoresearch methodology that frames `program.md`, `FRICTION.md`, and the rest of the operational ledger.
- [`../FRICTION.md`](../FRICTION.md) **F-001** — The operational ledger entry this note expands on.
- [`../program.md`](../program.md) § "Patches in `train.py` — DO NOT REMOVE" — Where the two workarounds are tabulated for the autoresearch agent.

A handful of higher-level architecture notes (Mamba state-space-model background, Nemotron's three-layer-type hybrid stack) live in a separate documentation vault outside this repo.

## Changelog

- **2026-05-04** — Initial note, written against `autoresearch-sft-grpo:main` HEAD `e5edfab` (post-T1.7a). Captures the F-001 / fast-path / causal_conv1d relationship that came up during the requirements.txt audit.
- **2026-05-04** — Copied into the autoresearch repo at `docs/fast-path-and-cache.md` as part of T1.13 so the repo is self-contained when published standalone.
