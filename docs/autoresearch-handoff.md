# Autonomous Claude Code Handoff

This doc covers handing the pod over to autonomous Claude Code (or any AI coding agent in unattended mode). Read it after [`../runpod-setup.md`](../runpod-setup.md) — the pod must already have the repo cloned, deps installed, and `python train.py` produce a `METRIC:` line before the autonomous loop is meaningful.

For interactive iteration (you in the loop, approving each step), this doc isn't needed — just `cd` into the repo and run `claude` from your existing user.

## Why a non-root user is mandatory

Claude Code refuses to start with `--dangerously-skip-permissions` when invoked as root or under sudo:

```
--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons
```

(verified 2026-05-02; tracked in [Claude Code issue #9184](https://github.com/anthropics/claude-code/issues/9184).) RunPod pods boot as root, so creating a dedicated non-root user is a precondition for autonomous mode — not optional. As a secondary benefit it limits blast radius if the agent issues a destructive command.

## Create a non-root user

```bash
# As root:
useradd -m -s /bin/bash claude-runner
usermod -aG video claude-runner
usermod -aG render claude-runner

# Move ownership of the project + venv + HF cache
chown -R claude-runner:claude-runner /path/to/autoresearch-sft-grpo
chown -R claude-runner:claude-runner /home/claude-runner/.cache

# Switch
su - claude-runner
```

> Do all `pip install`, `apt-get`, and `chown` commands as root **before** switching to `claude-runner`. The non-root user does not get sudo.

## Install Claude Code

```bash
# Node.js (skip if already system-wide)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
sudo apt-get install -y nodejs

npm install -g @anthropic-ai/claude-code
claude login
```

## Confirm git state for experiment tracking

After `git clone` in [`../runpod-setup.md`](../runpod-setup.md) § 1, the repo already has full history — no `git init` needed. Confirm the working tree is on `main` and clean before kicking off:

```bash
cd /path/to/autoresearch-sft-grpo
git rev-parse --abbrev-ref HEAD   # expect: main
git status --porcelain            # expect: empty
git log --oneline -1              # capture the starting commit hash for the first results.tsv row
```

The autoresearch loop uses `git rev-parse --short HEAD` for the `commit` column in `results.tsv` — this is why the working tree must be clean before each `python train.py` invocation, and why each experiment lands as its own commit before the next run. See [`../program.md`](../program.md) § Branch Hygiene for the full discipline.

## Run Claude Code

Run **as `claude-runner`** (not root):

```bash
# Standard — prompts for permission on each tool call
claude

# Skip prompts — only on disposable pods, only as a non-root user
claude --dangerously-skip-permissions
```

Then prime Claude with the kickoff prompt:

> "Read `program.md`, `BRANCH_NOTES.md`, and `FRICTION.md` first. Then run `python train.py` once *unmodified* to capture the current regression baseline — append a row to `results.tsv` and prepend a heartbeat to `STATUS.md`. After that baseline lands, begin Tier 2 sweeps per `program.md` § 'Tier 2 sweep targets': pick one axis, edit `train.py`, commit, run, append `results.tsv`, repeat. Honour the Tier 1 → Tier 2 sequencing."

> **Warning:** `--dangerously-skip-permissions` lets Claude run any command without confirmation. Use only on disposable pods with no sensitive data.

## Advanced: skipping the non-root user (throwaway pods only)

Two ways to bypass the user-creation step on a pod you intend to terminate after the run. Both are non-standard. Only use on a pod you'll throw away after the experiment — not on long-lived pods, not on anything sharing data with production.

### Approach 1: `IS_SANDBOX=1` env var (the hack)

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Undocumented bypass surfaced by [Levels.io](https://x.com/levelsio/status/1959012607270236619). Works in current versions but isn't in Anthropic's docs, `--help`, or release notes. Could be removed in any release without warning.

**Risk profile:** doesn't make Claude *more* destructive than running as root normally would; it just removes the safety rail that was added after a model executed `rm -rf /`. On a throwaway pod, worst case is the pod itself (~$0.50–2 + ~10 min restart).

### Approach 2: One-liner non-root user (the supported path)

If you don't want to depend on an undocumented env var:

```bash
# As root, all in one command (~30 seconds total):
useradd -m -s /bin/bash cr && \
  echo "export HF_TOKEN=$HF_TOKEN" >> /home/cr/.bashrc && \
  echo "export WANDB_API_KEY=$WANDB_API_KEY" >> /home/cr/.bashrc && \
  echo "export GIT_REPO_URL=$GIT_REPO_URL" >> /home/cr/.bashrc && \
  su - cr
```

Then continue with `npm install -g @anthropic-ai/claude-code`, `claude login`, the clone/venv steps, and the kickoff. Survives Claude Code updates regardless of what they do to the `IS_SANDBOX` path.

### Which to use

| Scenario | Recommended |
|---|---|
| First kickoff on a new branch, or any unfamiliar workflow | **Approach 2** — 30 seconds isn't worth the dependency on an undocumented mechanism while you're still validating the workflow |
| Repeated launches of a known-good kickoff on disposable pods | **Approach 1** is reasonable — you've calibrated trust in the prompt and model behaviour on this branch, and the time saved compounds across launches |
| Any pod that isn't immediately disposable | Use the full non-root flow above (proper user, no shortcuts) — neither approach belongs on a long-lived pod |

## See also

- [`../runpod-setup.md`](../runpod-setup.md) — the pod-bootstrap doc this is a follow-up to.
- [`../program.md`](../program.md) — the autoresearch agent contract Claude reads as its system prompt.
- [`../FRICTION.md`](../FRICTION.md) — failure log; the agent reads this before applying any patch that "feels familiar."
- [`methodology.md`](methodology.md) — the 8-artefact methodology that frames the loop.
