# Autoresearch SFT+GRPO — Nemotron Reasoning Challenge

Karpathy-style [autoresearch](https://github.com/karpathy/autoresearch) loop applied to the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle. An autoresearch agent (e.g. Claude Code) reads `program.md`, edits `train.py`, runs `python train.py`, parses `METRIC: 0.XXXX` from stdout, commits or reverts, and repeats.

`main` is the **BF16 SFT-only** baseline at commit `c1bb0a6` (METRIC **0.5333**). GRPO is wrapped in a `try/except` that falls back to the SFT adapter when the known Mamba/MoE+TRL tensor mismatch fires — see `FRICTION.md` F-002. NVFP4 / FP8 / 4-bit experiments live on dedicated branches (e.g. `nvfp4-blackwell`), not here.

## Quickstart

Detailed instructions are in [`runpod-setup.md`](runpod-setup.md). For the impatient on a fresh A100 80GB pod:

```bash
git clone https://github.com/priyanlc/autoresearch-sft-grpo.git
cd autoresearch-sft-grpo

# Venv + staged install (mamba_ssm needs --no-build-isolation)
python -m venv .venv && source .venv/bin/activate && pip install -U pip
pip install 'torch>=2.2.0' --index-url https://download.pytorch.org/whl/cu121
pip install ninja packaging wheel setuptools
pip install mamba_ssm --no-build-isolation
pip install -r requirements.txt

# Auth + pre-flight
hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_API_KEY"   # or: export WANDB_MODE=disabled
python check_install.py

# Run
python prepare.py    # downloads ~60 GB BF16 weights, builds val_split.json (first run only)
python train.py      # ~4–5 hours; emits "METRIC: 0.XXXX" at the end
```

The 9,500-puzzle `train.csv` and the 3-row `test.csv` preview ship with the repo under CC BY 4.0. See [`data/README.md`](data/README.md) for license and attribution details.

## File structure

```
autoresearch-sft-grpo/
├── README.md           # This file
├── program.md          # Autoresearch agent contract (the operational instruction sheet)
├── BRANCH_NOTES.md     # main's locked configuration + Tier 1 chronology
├── FRICTION.md         # Structured failure log (F-001 through F-006 seeded)
├── STATUS.md           # Append-only heartbeat log + session summaries
├── runpod-setup.md     # Three-part pod onboarding (first-run / autoresearch / troubleshooting)
├── results.tsv         # Per-experiment metric ledger (one row per train.py invocation)
├── requirements.txt    # Pip dependencies (mamba_ssm; causal_conv1d commented out per F-001)
├── check_install.py    # Pre-flight: deps + GPU + data + version checks
├── prepare.py          # One-time setup + evaluation harness (READ-ONLY)
├── train.py            # Training script (autoresearch agent modifies this)
├── eval_only.py        # Standalone eval of a saved adapter (utility)
├── .gitignore
├── data/
│   ├── README.md       # Provenance + CC BY 4.0 attribution
│   ├── train.csv       # 9,500 puzzles with ground-truth answers
│   └── test.csv        # 3-row preview (real test set is hidden at scoring)
├── docs/
│   ├── methodology.md          # 8-artefact autoresearch methodology
│   ├── bf16-sft-only-plan.md   # Strategic plan that anchors main's configuration
│   └── fast-path-and-cache.md  # F-001 deep dive: why train.py carries two redundant defenses
└── adapter/            # Output LoRA adapter (created by train.py; gitignored)
```

## Documentation map

Read these in order, depending on what you need:

- [`program.md`](program.md) — start here if you're operating the autoresearch loop. The agent contract: goal, metric, what to modify, Tier 1/2/3 sequencing, logging contract.
- [`runpod-setup.md`](runpod-setup.md) — start here if you're bootstrapping a fresh pod. Includes the autonomous Claude Code handoff in Part 2.
- [`BRANCH_NOTES.md`](BRANCH_NOTES.md) — the locked configuration table. What `main` is and why. Read before any training-config change.
- [`FRICTION.md`](FRICTION.md) — read **before** applying any patch that "feels familiar." F-001..F-006 capture the cache, GRPO, 4-bit, USE_COT, EVAL_MAX_NEW_TOKENS, and `.git/index` lock failures already worked through.
- [`STATUS.md`](STATUS.md) — append-only run log. Session summaries at the top; heartbeats below.
- [`results.tsv`](results.tsv) — every experiment, including failures. Schema in `program.md` § Logging & Reporting.
- [`docs/methodology.md`](docs/methodology.md) — the 8-artefact methodology that frames the loop.
- [`docs/bf16-sft-only-plan.md`](docs/bf16-sft-only-plan.md) — the strategic plan that anchors `main` (what's locked, what's parameterizable, what's punted).
- [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) — technical deep-dive on F-001 (why `train.py:386` and `train.py:536` are both needed, what changes when F-001 resolves upstream).
- [`data/README.md`](data/README.md) — CC BY 4.0 source, attribution, schema.

## Setting up autonomous Claude Code

`runpod-setup.md` Part 2 has the full procedure (non-root user creation, `--dangerously-skip-permissions` constraint, `IS_SANDBOX=1` advanced workaround for throwaway pods). The kickoff prompt for a fresh operator:

> "Read `program.md`, `BRANCH_NOTES.md`, and `FRICTION.md` first. Then run `python train.py` once *unmodified* to land **T1.8b** — that captures the post-T1 regression baseline against the pre-T1 0.5333 number, appends a row to `results.tsv`, and prepends a heartbeat to `STATUS.md`. After T1.8b lands, begin Tier 2 sweeps: pick one axis from `program.md` § 'Tier 2 sweep targets', edit `train.py`, commit with a `T2.x:` prefix, run, append `results.tsv`, repeat. Honour the Tier 1 → Tier 2 sequencing — do not start Tier 2 until T1.8b is committed."

Manual iteration without an autoresearch agent works the same way — edit `train.py`, run, capture METRIC, commit (or `git revert` on regression). The methodology's discipline (one commit per change ID, `T1.x:`/`T2.x:` prefixes, `results.tsv` per run, `FRICTION.md` for non-trivial failures) applies either way.

## Time per experiment

On A100 80GB with the locked configuration (1,200 SFT samples, GRPO graceful fallback, BF16, `EVAL_MAX_NEW_TOKENS=512`, no KV cache per F-001):

| Stage | Time |
|---|---|
| Cold model load (13 shards) | ~5 min |
| SFT (1,200 samples, 1 epoch) | ~1 hour |
| GRPO attempt → fast crash → SFT fallback | ~1 min |
| Eval (30 samples, no KV cache) | **~3 hours** |
| **Total per run** | **~4–5 hours** |

Eval is the wall-clock-dominant phase because cache-disabled generation re-runs the full forward each step. KV cache rehabilitation (gated on F-001 resolving upstream) would cut eval to ~10 minutes — see `docs/fast-path-and-cache.md`.

## License

- **Code:** [Apache License 2.0](LICENSE). Copyright 2026 Priyan Chandrapala. Permits commercial use, modification, distribution, patent grant; requires preserving the license/notice in derivative works.
- **Data** (`data/train.csv`, `data/test.csv`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) per the [Kaggle competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) data terms. Attribution requirements are documented in [`data/README.md`](data/README.md).

The two licenses are compatible: Apache-2.0 code can consume CC BY 4.0 data and produce LoRA adapters without conflict, as long as downstream redistribution attributes both per their respective terms.

## Attribution

Built on the [autoresearch](https://github.com/karpathy/autoresearch) pattern by Andrej Karpathy. The 8-artefact methodology in [`docs/methodology.md`](docs/methodology.md) extends the bare loop with operational discipline (FRICTION ledger, append-only STATUS, `results.tsv` per run, T1/T2/T3 tiering, three-way commit↔code↔ledger cross-referencing) so a finished run produces article-grade artefacts and not a pile of unreproducible logs.
