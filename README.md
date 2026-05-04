# Autoresearch SFT+GRPO — Nemotron Reasoning Challenge

A Karpathy-style [autoresearch](https://github.com/karpathy/autoresearch) loop applied to the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle. An AI coding agent (e.g. Claude Code) reads `program.md`, edits `train.py`, runs it, parses `METRIC: 0.XXXX` from stdout, commits or reverts, and repeats.

`main` runs **BF16 SFT-only** with graceful GRPO fallback. Current best METRIC: **0.5333**.

## Quickstart

Bootstrap a fresh A100 80GB pod by following [`runpod-setup.md`](runpod-setup.md). The short version:

```bash
git clone https://github.com/priyanlc/autoresearch-sft-grpo.git
cd autoresearch-sft-grpo

# Install uv (one-time): https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt   # see runpod-setup.md for the staged install if mamba_ssm fails
python prepare.py                    # one-time: downloads the model, builds val_split.json
python train.py                      # ~4–5 hours; emits "METRIC: 0.XXXX" at the end
```

`train.csv` (9,500 puzzles) and `test.csv` (3-row preview) ship with the repo.

## Where to read next

- [`program.md`](program.md) — the autoresearch agent contract: what to optimise, what's locked, what to log.
- [`runpod-setup.md`](runpod-setup.md) — full pod onboarding and the autonomous Claude Code handoff.
- [`BRANCH_NOTES.md`](BRANCH_NOTES.md) — `main`'s locked configuration and the rationale for each lock.
- [`FRICTION.md`](FRICTION.md) — failure log. Read before applying any patch that "feels familiar."
- [`STATUS.md`](STATUS.md) — append-only run log; session summaries at the top.
- [`docs/methodology.md`](docs/methodology.md) — the 8-artefact autoresearch methodology that frames the loop.
- [`docs/bf16-sft-only-plan.md`](docs/bf16-sft-only-plan.md) — strategic plan for `main` (what's locked vs parameterizable).
- [`docs/autoresearch-handoff.md`](docs/autoresearch-handoff.md) — handing the pod over to autonomous Claude Code (non-root user, kickoff prompt, IS_SANDBOX workaround).
- [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) — technical deep-dive on the Mamba cache workaround.
- [`data/README.md`](data/README.md) — data provenance and CC BY 4.0 attribution.

## Repo layout

```
program.md          agent contract
runpod-setup.md     pod bootstrap
BRANCH_NOTES.md     locked config
FRICTION.md         failure log
STATUS.md           run log
results.tsv         per-experiment metric ledger
train.py            training script (agent edits this)
prepare.py          one-time setup + eval harness (read-only)
requirements.txt
data/               train.csv, test.csv, README.md
docs/               methodology, strategic plan, autoresearch handoff, deep-dives
```

## License

- **Code:** [Apache License 2.0](LICENSE). Copyright 2026 Priyan Chandrapala.
- **Data:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) per the [Kaggle competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge); see [`data/README.md`](data/README.md) for attribution requirements.

## Attribution

Built on the [autoresearch](https://github.com/karpathy/autoresearch) pattern by Andrej Karpathy.
