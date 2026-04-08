"""
train.py — SFT warmup + GRPO training script for autoresearch iteration.

This is the ONLY file the AI agent should modify.
Each run: loads model, SFT warmup, GRPO training, evaluates, prints METRIC.

Usage:
    python train.py
"""

import os
import sys
import gc
import re
import json
import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TRITON_JIT_DISABLE_OPT'] = '1'  # Mitigate Triton kernel issues with Mamba layers

import polars as pl
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

# Import evaluation harness from prepare.py (read-only)
from prepare import (
    MODEL_ID, TRAIN_CSV, classify_type, extract_boxed_answer, answers_match,
    evaluate_model, load_val_data, stratified_sample, METRIC_SUFFIX,
    VAL_SAMPLES_PER_TYPE, _tokenize_prompt,
)

# ============================================================
# === CONFIGURATION — Modify this section to improve METRIC ===
# ============================================================

# Data
SFT_SAMPLES_PER_TYPE = 200        # 6 * 200 = 1200 SFT samples (matching Kaggle baseline)
GRPO_SAMPLES_PER_TYPE = 20        # 6 * 20 = 120 GRPO prompts

# SFT Phase (warmup)
SFT_LR = 2e-4
SFT_EPOCHS = 1
SFT_MAX_SEQ_LEN = 1024

# GRPO Phase
GRPO_LR = 5e-6
GRPO_EPOCHS = 1
GRPO_NUM_GENERATIONS = 4          # Completions per prompt for advantage estimation
GRPO_MAX_COMPLETION = 512         # Max tokens per generation
GRPO_TEMPERATURE = 0.7            # Sampling temperature for generation
GRPO_BETA = 0.01                  # KL penalty (0 = no constraint, higher = more conservative)

# Reward weights
W_CORRECTNESS = 1.0
W_FORMAT = 0.3
W_REASONING = 0.15
W_CATEGORY_BONUS = 0.2

# LoRA
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = 'all-linear'

# Training
BATCH_SIZE = 1
GRAD_ACCUM = 4
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

# Evaluation
EVAL_MAX_NEW_TOKENS = 512         # Need enough for thinking + boxed answer
EVAL_BATCH_SIZE = 1               # Reduced to avoid OOM on A100 80GB

# SFT CoT (chain-of-thought)
USE_COT = True                    # Teach thinking pattern that eval uses
                                  # Note: with USE_COT=True (50 samples/type), model parroted templates verbatim
                                  # instead of reasoning. Inconclusive if harmful — may improve with more data.

# Output
OUTPUT_DIR = './adapter'


# ============================================================
# === SFT PROMPT FORMAT — Controls what model learns in warmup
# ============================================================

# Category-specific reasoning traces (used when USE_COT=True)
_COT_BY_TYPE = {
    'bit_ops': "Let me analyze each input-output pair to identify the bit transformation rule. I'll compare the binary strings to find the pattern, then apply it to the new input.",
    'gravity': "I need to determine the gravitational constant g from the given time-distance pairs using d = 0.5*g*t^2. I'll solve for g using a data point, then compute the distance for the new time.",
    'unit_conv': "I need to find the conversion factor from the given examples. I'll divide the output by the input to get the ratio, then apply it to the new measurement.",
    'cipher': "I need to build the substitution mapping from the ciphertext-plaintext pairs. I'll map each character, then use the mapping to decrypt the new ciphertext.",
    'numeral': "I need to convert the given number to the appropriate numeral system by breaking it down into components.",
    'symbol': "I need to identify the transformation rules from the examples and apply them to the new equation.",
}
_COT_DEFAULT = "Let me analyze the pattern in the given examples and apply it to solve the problem."


def _build_dynamic_cot(qtype, prompt, answer):
    """Build a category-specific reasoning trace using example data + answer."""
    if qtype == 'cipher':
        # Extract cipher/plain pairs from the prompt to build a substitution map
        import re as _re
        pairs = _re.findall(r'([a-z\s]+?)\s*->\s*([a-z\s]+)', prompt.lower())
        mapping = {}
        for cipher, plain in pairs:
            cipher = cipher.strip()
            plain = plain.strip()
            if len(cipher) == len(plain):
                for c, p in zip(cipher, plain):
                    if c.isalpha() and p.isalpha():
                        mapping[c] = p
        if mapping:
            map_str = ', '.join(f'{k}->{v}' for k, v in sorted(mapping.items()))
            return (f"I'll build the substitution map from the examples. Mapping: {map_str}. "
                    f"Applying this to the cipher gives: {answer}")
    elif qtype == 'gravity':
        # Try to extract a (t, d) pair to compute g
        import re as _re
        m = _re.search(r't\s*=\s*([\d.]+).*?d\s*=\s*([\d.]+)', prompt, _re.DOTALL | _re.IGNORECASE)
        if m:
            t, d = float(m.group(1)), float(m.group(2))
            g = 2 * d / (t * t) if t > 0 else 0
            return (f"Using d = 0.5*g*t^2: g = 2d/t^2 = 2*{d}/{t}^2 = {g:.4f}. "
                    f"Apply to find d for the new t. Answer: {answer}")
    elif qtype == 'unit_conv':
        import re as _re
        m = _re.search(r'([\d.]+)\s*[a-zA-Z]+\s*->\s*([\d.]+)', prompt)
        if m:
            inp, out = float(m.group(1)), float(m.group(2))
            factor = out / inp if inp > 0 else 1
            return (f"Conversion factor = output/input = {out}/{inp} = {factor:.4f}. "
                    f"Apply factor to new input. Answer: {answer}")
    elif qtype == 'bit_ops':
        return (f"Comparing input/output bit patterns to identify the operation "
                f"(shift, rotate, XOR, AND, OR, NOT, or combinations). Result: {answer}")
    return None


def build_sft_text(example, tokenizer):
    """Format a training example for SFT warmup.

    Modify this function to change what the model learns during SFT.
    When USE_COT=False, the model learns answer format only; native reasoning
    via enable_thinking=True handles the <think> block at inference.
    When USE_COT=True, static CoT templates are included in the <think> block.
    """
    user_msg = example['prompt'] + METRIC_SUFFIX

    if USE_COT:
        qtype = classify_type(example['prompt'])
        # Try dynamic CoT first; fall back to static template
        dynamic_cot = _build_dynamic_cot(qtype, example['prompt'], example['answer'])
        cot = dynamic_cot if dynamic_cot else _COT_BY_TYPE.get(qtype, _COT_DEFAULT)
        assistant_msg = f'<think>\n{cot}\n</think>\n\\boxed{{{example["answer"]}}}'
    else:
        assistant_msg = f'\\boxed{{{example["answer"]}}}'

    messages = [
        {'role': 'user', 'content': user_msg},
        {'role': 'assistant', 'content': assistant_msg},
    ]
    for kwargs in [{'enable_thinking': True}, {}]:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs
            )
            return {'text': text}
        except Exception:
            continue
    return {'text': f'<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>'}


# ============================================================
# === REWARD FUNCTIONS — Core of GRPO optimization
# ============================================================

# Thinking tag patterns the model might use
THINK_PATTERNS = [
    re.compile(r'<think>(.*?)</think>', re.DOTALL),
    re.compile(r'<\|think_start\|>(.*?)<\|think_end\|>', re.DOTALL),
    re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL),
]


def extract_thinking(text):
    """Extract thinking/reasoning content from model output."""
    for pattern in THINK_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return None


def _to_str(x):
    """Coerce a completion or prompt to a plain string.
    TRL may pass strings, lists of message dicts, or other types."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        # List of message dicts like [{'role': 'user', 'content': '...'}]
        if x and isinstance(x[0], dict) and 'content' in x[0]:
            return x[0]['content']
        # List of strings — join them
        return ' '.join(str(item) for item in x)
    return str(x)


def correctness_reward(completions, answer, **kwargs):
    """Binary correctness: correct=1.0, wrong=0.0."""
    rewards = []
    for comp, gold in zip(completions, answer):
        comp = _to_str(comp)
        pred = extract_boxed_answer(comp)
        if pred is not None and answers_match(pred, str(gold)):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    """Reward for proper \\boxed{} format.
    Single boxed=1.0, multiple=0.5, none=-1.0."""
    rewards = []
    for comp in completions:
        comp = _to_str(comp)
        boxed_count = len(re.findall(r'\\boxed\{', comp))
        if boxed_count == 1:
            rewards.append(1.0)
        elif boxed_count > 1:
            rewards.append(0.5)
        else:
            rewards.append(-1.0)
    return rewards


def reasoning_reward(completions, **kwargs):
    """Reward for showing reasoning in thinking tags.
    Substantive=1.0, brief=0.5, trivial=0.0, missing=-0.3."""
    rewards = []
    for comp in completions:
        comp = _to_str(comp)
        thinking = extract_thinking(comp)
        if thinking is not None:
            word_count = len(thinking.split())
            if word_count > 20:
                rewards.append(1.0)
            elif word_count > 5:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        else:
            boxed_idx = comp.find('\\boxed{')
            if boxed_idx > 100:
                rewards.append(0.3)
            else:
                rewards.append(-0.3)
    return rewards


def category_specific_reward(completions, prompts, answer, **kwargs):
    """Category-aware answer format validation."""
    rewards = []
    for comp, p in zip(completions, prompts):
        comp = _to_str(comp)
        prompt_text = _to_str(p)
        qtype = classify_type(prompt_text)
        pred = extract_boxed_answer(comp)

        if pred is None:
            rewards.append(0.0)
            continue

        bonus = 0.0
        if qtype == 'bit_ops':
            if re.match(r'^[01]{8}$', pred):
                bonus = 0.5
        elif qtype in ('gravity', 'unit_conv'):
            try:
                float(pred)
                bonus = 0.5
            except ValueError:
                bonus = -0.5
        elif qtype == 'cipher':
            if re.match(r'^[a-z\s]+$', pred):
                bonus = 0.5
        elif qtype == 'numeral':
            if re.match(r'^[IVXLCDM]+$', pred):
                bonus = 0.5
        elif qtype == 'symbol':
            if pred and len(pred) <= 4:
                bonus = 0.3

        rewards.append(bonus)
    return rewards


def make_weighted_reward(fn, weight):
    """Wrap a reward function to scale its output by a weight."""
    def weighted(completions, **kwargs):
        raw = fn(completions, **kwargs)
        return [r * weight for r in raw]
    weighted.__name__ = fn.__name__
    return weighted


# ============================================================
# === DATA LOADING — Modify sampling strategy here
# ============================================================

def load_training_data(tokenizer):
    """Load and prepare SFT + GRPO training data."""
    train_df = pl.read_csv(TRAIN_CSV)
    train_df = train_df.with_columns(
        pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('qtype')
    )

    # Exclude validation data (same seed=0 as prepare.py)
    val_df = stratified_sample(train_df, VAL_SAMPLES_PER_TYPE, seed=0)
    val_ids = set(val_df['id'].to_list())
    remaining = train_df.filter(~pl.col('id').is_in(val_ids))

    # SFT data
    sft_df = stratified_sample(remaining, SFT_SAMPLES_PER_TYPE, seed=42)
    sft_ids = set(sft_df['id'].to_list())

    sft_dataset = Dataset.from_pandas(sft_df.drop('qtype').to_pandas())
    sft_dataset = sft_dataset.map(
        lambda ex: build_sft_text(ex, tokenizer),
        remove_columns=sft_dataset.column_names,
    )

    # GRPO data (non-overlapping with SFT)
    grpo_pool = remaining.filter(~pl.col('id').is_in(sft_ids))
    grpo_df = stratified_sample(grpo_pool, GRPO_SAMPLES_PER_TYPE, seed=123)

    grpo_data = []
    for row in grpo_df.iter_rows(named=True):
        grpo_data.append({
            'prompt': [{'role': 'user', 'content': row['prompt'] + METRIC_SUFFIX}],
            'answer': str(row['answer']),
        })
    grpo_dataset = Dataset.from_list(grpo_data)

    return sft_dataset, grpo_dataset


# ============================================================
# === MAIN — Do not modify below this line ===
# ============================================================

def main():
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print(f'Loading tokenizer: {MODEL_ID}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    print('Preparing training data...')
    sft_dataset, grpo_dataset = load_training_data(tokenizer)
    val_data = load_val_data()
    print(f'SFT: {len(sft_dataset)}, GRPO: {len(grpo_dataset)}, Val: {len(val_data)}')

    # Load model
    print(f'Loading model: {MODEL_ID}')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Disable fast path
    for name, mod in sys.modules.items():
        if 'modeling_nemotron_h' in name:
            mod.is_fast_path_available = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Phase 1: SFT Warmup ----
    print('\n=== Phase 1: SFT Warmup ===')
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LR,
        logging_steps=10,
        bf16=True,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=0.0,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        warmup_ratio=WARMUP_RATIO,
        save_strategy='no',
        report_to='none',
        dataset_text_field='text',
        max_length=SFT_MAX_SEQ_LEN,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        args=sft_args,
    )

    print(f'Training {len(sft_dataset)} samples x {SFT_EPOCHS} epoch...')
    sft_trainer.train()

    # Save SFT adapter as fallback
    model.save_pretrained(OUTPUT_DIR)
    print(f'SFT adapter saved to {OUTPUT_DIR} (fallback)')

    del sft_trainer
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Phase 2: GRPO Training ----
    print('\n=== Phase 2: GRPO Training ===')

    # Monkeypatch tokenizer for enable_thinking
    if not getattr(tokenizer, '_thinking_patched', False):
        _orig_apply_chat_template = tokenizer.apply_chat_template

        def _apply_with_thinking(*args, **kwargs):
            kwargs.setdefault('enable_thinking', True)
            try:
                return _orig_apply_chat_template(*args, **kwargs)
            except TypeError:
                kwargs.pop('enable_thinking', None)
                return _orig_apply_chat_template(*args, **kwargs)

        tokenizer.apply_chat_template = _apply_with_thinking
        tokenizer._thinking_patched = True

    # TRL compatibility
    if not hasattr(model, 'warnings_issued'):
        model.warnings_issued = {}

    # Build weighted reward functions
    reward_funcs = [
        make_weighted_reward(correctness_reward, W_CORRECTNESS),
        make_weighted_reward(format_reward, W_FORMAT),
        make_weighted_reward(reasoning_reward, W_REASONING),
        make_weighted_reward(category_specific_reward, W_CATEGORY_BONUS),
    ]

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=GRPO_NUM_GENERATIONS,
        generation_batch_size=GRPO_NUM_GENERATIONS,
        max_completion_length=GRPO_MAX_COMPLETION,
        beta=GRPO_BETA,
        temperature=GRPO_TEMPERATURE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=GRPO_EPOCHS,
        learning_rate=GRPO_LR,
        bf16=True,
        max_grad_norm=0.5,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        warmup_ratio=WARMUP_RATIO,
        logging_steps=5,
        save_strategy='no',
        report_to='none',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
    )

    est_steps = len(grpo_dataset) // (BATCH_SIZE * GRAD_ACCUM)
    print(f'GRPO: {len(grpo_dataset)} prompts, {GRPO_NUM_GENERATIONS} gens/prompt, ~{est_steps} steps')

    try:
        grpo_trainer.train()
        grpo_trainer.model.save_pretrained(OUTPUT_DIR)
        print(f'GRPO adapter saved to {OUTPUT_DIR}')
    except Exception as e:
        print(f'GRPO failed: {e}')
        print(f'SFT fallback adapter still at {OUTPUT_DIR}')
        if 'size of tensor' in str(e).lower():
            print(f'\n--- GRPO TENSOR MISMATCH DIAGNOSIS ---')
            print(f'Known incompatibility: Nemotron hybrid Mamba/MoE + TRL GRPOTrainer.')
            print(f'See: https://github.com/huggingface/trl/issues/3681')
            print(f'See: https://github.com/unslothai/unsloth/issues/3387')
            print(f'Possible fixes:')
            print(f'  1. Upgrade TRL: pip install -U trl')
            print(f'  2. Use 4-bit quantization (changes tensor shapes)')
            print(f'  3. Use Unsloth patched GRPOTrainer')
            print(f'  4. SFT-only is still a valid approach (current fallback)')
            print(f'--- END DIAGNOSIS ---')

    del grpo_trainer
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Evaluate ----
    model.config.use_cache = False  # Nemotron cache has bugs; disable for safety
    print('\nEvaluating...')
    overall, by_type = evaluate_model(
        model, tokenizer, val_data,
        max_new_tokens=EVAL_MAX_NEW_TOKENS,
        batch_size=EVAL_BATCH_SIZE,
    )

    # Print results
    elapsed = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'Overall accuracy: {overall:.4f}')
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        acc = stats['correct'] / max(stats['total'], 1)
        print(f'  {qtype}: {acc:.4f} ({stats["correct"]}/{stats["total"]})')
    print(f'Time: {elapsed:.0f}s')
    print(f'{"="*60}')

    if torch.cuda.is_available():
        print(f'Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB')

    # === Debug: print raw outputs for failing categories ===
    print('\n=== Debug: Raw outputs for weak categories ===')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    for example in val_data:
        qtype = example['qtype']
        stats = by_type.get(qtype, {})
        acc = stats.get('correct', 0) / max(stats.get('total', 1), 1)
        if acc < 0.5:  # Only debug categories scoring below 50%
            messages = [{'role': 'user', 'content': example['prompt'] + METRIC_SUFFIX}]
            ids = _tokenize_prompt(messages, tokenizer).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            pred = extract_boxed_answer(response)
            match = answers_match(pred, example['answer']) if pred else False
            print(f"\n[{qtype}] gold={example['answer']}, pred={pred}, match={match}")
            print(f"  raw: {response[:400]}")

    # === THE METRIC LINE — autoresearch parses this ===
    print(f'METRIC: {overall:.4f}')

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
