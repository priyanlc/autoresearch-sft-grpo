# Autoresearch SFT+GRPO — Nemotron Reasoning Challenge

Autonomous SFT+GRPO optimization using the [autoresearch](https://github.com/karpathy/autoresearch) pattern.

Two-phase pipeline: SFT warmup teaches format/reasoning basics, then GRPO uses reward signals to improve accuracy.

## Setup (RunPod A100)

```bash
# 1. Clone this directory to your RunPod instance
# 2. Copy train.csv into ./data/
mkdir -p data
cp /path/to/train.csv data/

# 3. Install dependencies
pip install torch transformers accelerate peft trl datasets polars mamba_ssm causal_conv1d sentencepiece

# 4. Run one-time preparation
python prepare.py

# 5. Verify baseline works
python train.py

# 6. Initialize git (autoresearch uses git to track experiments)
git init
git add -A
git commit -m "initial baseline"
```

## Setting up Claude Code on RunPod

```bash
# 1. Install Node.js (required for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

# 2. Install Claude Code
npm install -g @anthropic-ai/claude-code

# 3. Authenticate (this will open a browser or give you a link)
claude login

# 4. Navigate to this directory and start Claude Code
cd /path/to/autoresearch-sft-grpo
claude

# 5. Inside Claude Code, tell it to iterate:
#    "Read program.md and start optimizing. Run python train.py,
#     check the METRIC, modify train.py to improve it, and repeat."
```

Alternatively, use any AI coding agent (Cursor, Aider, etc.) — the pattern is agent-agnostic.

## Running with autoresearch

```bash
# Clone autoresearch
git clone https://github.com/karpathy/autoresearch
cd autoresearch

# Point it at our directory (or copy our files into autoresearch structure)
# The AI agent will:
#   1. Read program.md for instructions
#   2. Modify train.py
#   3. Run: python train.py
#   4. Parse METRIC: X.XXXX from output
#   5. Keep or discard the change
#   6. Repeat
```

## File structure

```
autoresearch-sft-grpo/
├── program.md      # Instructions for the AI agent (what to optimize)
├── prepare.py      # One-time setup + evaluation harness (READ-ONLY)
├── train.py        # Training script (AI agent modifies this)
├── data/
│   ├── train.csv   # Competition training data (you provide)
│   └── val_split.json  # Fixed validation split (created by prepare.py)
└── adapter/        # Output LoRA adapter (created by train.py)
```

## Manual iteration

You can also iterate manually without autoresearch:

```bash
# Edit train.py (change config, rewards, prompt format, etc.)
python train.py
# Check METRIC at the end of output
# If improved, git commit. If not, git checkout train.py
```

## Time per experiment

On A100 80GB with default settings (300 SFT + 300 GRPO samples):
- Model loading: ~2 min
- SFT warmup: ~5 min
- GRPO training: ~12 min
- Evaluation: ~5 min
- **Total: ~24 min per experiment**

For faster iteration, reduce `SFT_SAMPLES_PER_TYPE` and `GRPO_SAMPLES_PER_TYPE`, or reduce `GRPO_NUM_GENERATIONS` to 2.
