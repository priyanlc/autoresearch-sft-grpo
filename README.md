# Autoresearch SFT+GRPO — Nemotron Reasoning Challenge

A Karpathy-style [autoresearch](https://github.com/karpathy/autoresearch) loop applied to the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle. An AI coding agent (e.g. Claude Code) reads `program.md`, edits `train.py`, runs it, parses `METRIC: 0.XXXX` from stdout, commits or reverts, and repeats.

`main` runs **BF16 SFT-only** with graceful GRPO fallback. Current best METRIC: **0.6000** (T2.8, `c4a9d1c`); locked floor: 0.5333 (`c1bb0a6`). See [`STATUS.md`](STATUS.md) and [`results.tsv`](results.tsv) for the latest run.

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

For an **unattended agent run** instead of the manual flow above, see [Claude Code agent — install and handover](#claude-code-agent--install-and-handover) below.

## Claude Code agent — install and handover

Hand the pod over to Claude Code (or any agent in unattended mode) to reproduce the baseline and iterate on Tier 2 sweeps without you in the loop.

### Install the agent

After the Quickstart through `source .venv/bin/activate` (the agent handles dep install and training itself):

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
sudo apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code
claude login
```

### Handover

**1. Set up a non-root user.** RunPod pods boot as root, and `claude --dangerously-skip-permissions` refuses to run as root for safety. See [`docs/autoresearch-handoff.md`](docs/autoresearch-handoff.md) § "Create a non-root user" for the proper flow, or the `IS_SANDBOX=1` shortcut for throwaway pods.

**2. Export your tokens** — `prompt.md` step 1 verifies these are set before authenticating:

```bash
export HF_TOKEN=<YOUR_KEY>
export WANDB_API_KEY=<YOUR_KEY>
```

**3. Kick off the run.** As the non-root user, in the activated venv with the repo cloned:

```bash
claude --dangerously-skip-permissions
```

Then paste the full contents of [`prompt.md`](prompt.md) into the session. It runs the full setup → train → eval → sanity-check loop end-to-end and **stops for your confirmation before any Tier 2 sweep**. Verify METRIC ≥ 0.5333 (the locked floor) before authorising new sweeps; current best is 0.6000 (T2.8). Some Tier 2 has already landed — check [`STATUS.md`](STATUS.md) first. Full handover details (non-root setup, `IS_SANDBOX` bypass, branch hygiene) are in [`docs/autoresearch-handoff.md`](docs/autoresearch-handoff.md).

## Where to read next

- [`program.md`](program.md) — the autoresearch agent contract: what to optimise, what's locked, what to log.
- [`runpod-setup.md`](runpod-setup.md) — full manual pod onboarding (the agent path lives in `docs/autoresearch-handoff.md` + `prompt.md`).
- [`BRANCH_NOTES.md`](BRANCH_NOTES.md) — `main`'s locked configuration and the rationale for each lock.
- [`FRICTION.md`](FRICTION.md) — failure log. Read before applying any patch that "feels familiar."
- [`STATUS.md`](STATUS.md) — append-only run log; session summaries at the top.
- [`docs/methodology.md`](docs/methodology.md) — the 8-artefact autoresearch methodology that frames the loop.
- [`docs/bf16-sft-only-plan.md`](docs/bf16-sft-only-plan.md) — strategic plan for `main` (what's locked vs parameterizable).
- [`docs/autoresearch-handoff.md`](docs/autoresearch-handoff.md) — handing the pod over to autonomous Claude Code (non-root user, IS_SANDBOX workaround).
- [`prompt.md`](prompt.md) — the autonomous-run kickoff prompt itself; paste into Claude Code on a freshly-bootstrapped pod to reproduce the locked 0.5333 floor end-to-end before starting new Tier 2 sweeps. Some Tier 2 has already landed on `main` (T2.7 reverted, T2.8 kept; current best 0.6000) — see [`STATUS.md`](STATUS.md).
- [`docs/fast-path-and-cache.md`](docs/fast-path-and-cache.md) — technical deep-dive on the Mamba cache workaround.
- [`data/README.md`](data/README.md) — data provenance and CC BY 4.0 attribution.

## Repo layout

```
program.md               agent contract
runpod-setup.md          pod bootstrap (manual)
prompt.md                autonomous-run kickoff prompt
BRANCH_NOTES.md          locked config
FRICTION.md              failure log
STATUS.md                run log
results.tsv              per-experiment metric ledger
train.py                 training script (agent edits this)
prepare.py               one-time setup + eval harness (read-only)
adapter_sanity_check.py  adapter-on-fresh-base check (Validation Contract point 5; canonical helper from T1.16)
eval_only.py             quick eval of saved SFT adapter (standalone)
check_install.py         dependency + GPU verification (run before train.py)
bootstrap.sh             CUDA-built deps (torch, mamba_ssm) — see runpod-setup.md
requirements.txt
data/                    train.csv, test.csv, README.md
docs/                    methodology, strategic plan, autoresearch handoff, deep-dives
```

## License

- **Code:** [Apache License 2.0](LICENSE). Copyright 2026 Priyan Chandrapala.
- **Data:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) per the [Kaggle competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge); see [`data/README.md`](data/README.md) for attribution requirements.

## Attribution

Built on the [autoresearch](https://github.com/karpathy/autoresearch) pattern by Andrej Karpathy.
