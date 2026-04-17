# Autoresearch SFT+GRPO — NVFP4 Blackwell Branch

**Branch:** `nvfp4-blackwell`
**Created:** 2026-04-12
**Base:** main branch (METRIC 0.5333, bf16 on A100)

## Purpose

This branch adapts the autoresearch pipeline for Kaggle's RTX PRO 6000 (Blackwell) using
NVFP4 quantization via FPQuantConfig. Key changes from main:

1. **NVFP4 model loading** — base model ~17GB instead of ~60GB in bf16
2. **FPQuantLinear `__bases__` hack** — patches PEFT compatibility after model load
3. **Synthetic data** — 3000 additional samples with perfect CoT for all 6 categories
4. **Cosine-scaled reward** — replaces binary correctness for GRPO
5. **Updated training config** — adamw_8bit, packing, aggressive gradient clipping

## Configuration delta from main

| Parameter | main (bf16) | nvfp4-blackwell |
|-----------|-------------|-----------------|
| USE_NVFP4 | N/A | True |
| SFT_LR | 2e-4 | 5e-5 |
| BATCH_SIZE | 1 | 2 |
| MAX_GRAD_NORM | 1.0 | 0.1 |
| WEIGHT_DECAY | 0.0 | 0.1 |
| optim | adamw_torch | adamw_8bit |
| packing | False | True |
| SFT_MAX_SEQ_LEN | 1024 | 2048 |
| LORA_DROPOUT | 0.05 | 0.0 (FPQuantLinear req) |
| GRPO_BETA | 0.01 | 0.0 (no ref model) |
| Synthetic data | None | 500/type = 3000 |
| Reward | Binary + weighted | Cosine-scaled |

## Known requirements

- **Blackwell GPU** (RTX PRO 6000 or similar) for real NVFP4
- `pip install fp_quant qutlass --no-build-isolation` (requires CUDA 12.8+)
- `pip install mamba_ssm causal_conv1d` (CUDA extensions)
- PEFT does NOT natively support FPQuantLinear — the `__bases__` hack is required

## Fallback

If the code fails, still stick with NVFP4, fix the code in train.py config and continute until successful. 
