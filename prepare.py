"""
prepare.py — One-time data preparation and evaluation harness (READ-ONLY).

Run once before starting autoresearch:
    python prepare.py

This downloads the model, prepares the data splits, and verifies everything works.
The evaluation function is also defined here and imported by train.py.
"""

import os
import sys
import re
import math
import json
import subprocess

import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


# === Paths ===
MODEL_ID = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
DATA_DIR = './data'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
VAL_JSON = os.path.join(DATA_DIR, 'val_split.json')
RESULTS_TSV = 'results.tsv'

# === Validation split (fixed, never changes) ===
VAL_SAMPLES_PER_TYPE = 5  # 6 * 5 = 30 samples (small for fast iteration)


def classify_type(prompt_text):
    """Classify a puzzle prompt into one of 6 categories."""
    p = prompt_text.lower()
    if 'bit manipulation' in p or '8-bit binary' in p: return 'bit_ops'
    elif 'encrypt' in p or 'decrypt' in p: return 'cipher'
    elif 'gravitational' in p or 'falling distance' in p: return 'gravity'
    elif 'numeral system' in p: return 'numeral'
    elif 'transformation rules' in p: return 'symbol'
    elif 'unit conversion' in p or 'convert the following measurement' in p: return 'unit_conv'
    return 'unknown'


def extract_boxed_answer(text):
    """Extract the last non-empty \\boxed{} content from model output."""
    if text is None:
        return None
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        for m in reversed(matches):
            if m.strip():
                return m.strip()
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return None
    depth = 0
    start = idx + len('\\boxed{')
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                content = text[start:i].strip()
                return content if content else None
            depth -= 1
    content = text[start:].strip()
    return content if content else None


def answers_match(pred, gold):
    """Check if prediction matches gold."""
    if pred is None:
        return False
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass
    return pred.strip().lower() == gold.strip().lower()


METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def _tokenize_prompt(messages, tokenizer):
    """Tokenize a single chat message, returning a 1D tensor of input_ids."""
    for kwargs in [{'enable_thinking': True}, {}]:
        try:
            result = tokenizer.apply_chat_template(
                messages, return_tensors='pt', add_generation_prompt=True, **kwargs
            )
            if hasattr(result, 'input_ids'):
                return result['input_ids'].squeeze(0)
            elif isinstance(result, dict):
                return result['input_ids'].squeeze(0)
            else:
                return result.squeeze(0)
        except TypeError:
            continue
    # Fallback: tokenize manually
    text = f'<|im_start|>user\n{messages[0]["content"]}<|im_end|>\n<|im_start|>assistant\n'
    return tokenizer(text, return_tensors='pt')['input_ids'].squeeze(0)


def evaluate_model(model, tokenizer, val_data, max_new_tokens=256, batch_size=4):
    """Evaluate on validation set with batched greedy decoding.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        val_data: List of dicts with 'prompt', 'answer', 'qtype'.
        max_new_tokens: Max tokens to generate per sample.
        batch_size: Number of samples to generate in parallel.

    Returns:
        (overall_accuracy, per_type_dict)
    """
    model.eval()
    results = {'total': 0, 'correct': 0, 'by_type': {}}

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = next(model.parameters()).device

    # Ensure tokenizer has pad token and pads on the left (for batched generation)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    # Process in batches
    for batch_start in range(0, len(val_data), batch_size):
        batch = val_data[batch_start:batch_start + batch_size]

        # Tokenize all prompts in the batch
        all_input_ids = []
        for example in batch:
            messages = [{'role': 'user', 'content': example['prompt'] + METRIC_SUFFIX}]
            ids = _tokenize_prompt(messages, tokenizer)
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)
            all_input_ids.append(ids)

        # Pad to same length (left-padded for generation)
        prompt_lengths = [ids.shape[0] for ids in all_input_ids]
        max_len = max(prompt_lengths)
        padded_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, ids in enumerate(all_input_ids):
            pad_len = max_len - ids.shape[0]
            padded_ids[i, pad_len:] = ids
            attention_mask[i, pad_len:] = 1

        padded_ids = padded_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=padded_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and score each sample
        for i, example in enumerate(batch):
            generated_ids = outputs[i, max_len:]  # Everything after the padded prompt
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            pred = extract_boxed_answer(response)
            gold = example['answer']
            qtype = example['qtype']
            is_correct = answers_match(pred, gold)

            results['total'] += 1
            results['correct'] += int(is_correct)

            if qtype not in results['by_type']:
                results['by_type'][qtype] = {'total': 0, 'correct': 0}
            results['by_type'][qtype]['total'] += 1
            results['by_type'][qtype]['correct'] += int(is_correct)

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    overall = results['correct'] / max(results['total'], 1)
    return overall, results['by_type']


def load_val_data():
    """Load the fixed validation split."""
    with open(VAL_JSON, 'r') as f:
        return json.load(f)


def stratified_sample(df, n_per_type, seed):
    """Sample n_per_type from each category."""
    dfs = []
    for qtype in df['qtype'].unique().to_list():
        subset = df.filter(pl.col('qtype') == qtype)
        n = min(n_per_type, len(subset))
        dfs.append(subset.sample(n=n, seed=seed))
    return pl.concat(dfs)


# === One-time preparation ===
if __name__ == '__main__':
    print('=== Autoresearch SFT+GRPO: Preparation ===')

    # Check data exists
    if not os.path.exists(TRAIN_CSV):
        print(f'ERROR: {TRAIN_CSV} not found.')
        print(f'Please copy train.csv to {DATA_DIR}/')
        sys.exit(1)

    # Create validation split (fixed, deterministic)
    print('Creating validation split...')
    train_df = pl.read_csv(TRAIN_CSV)
    train_df = train_df.with_columns(
        pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('qtype')
    )

    val_df = stratified_sample(train_df, VAL_SAMPLES_PER_TYPE, seed=0)
    val_data = []
    for row in val_df.iter_rows(named=True):
        val_data.append({
            'prompt': row['prompt'],
            'answer': str(row['answer']),
            'qtype': classify_type(row['prompt']),
        })

    with open(VAL_JSON, 'w') as f:
        json.dump(val_data, f)
    print(f'Saved {len(val_data)} validation samples to {VAL_JSON}')

    # Verify model is downloadable
    print(f'\nVerifying model access: {MODEL_ID}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f'Tokenizer loaded. Vocab size: {len(tokenizer)}')

    # Initialize results.tsv
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, 'w') as f:
            f.write('commit\tmetric\tbit_ops\tcipher\tgravity\tnumeral\tsymbol\tunit_conv\tstatus\tdescription\n')
        print(f'Initialized {RESULTS_TSV}')

    print('\n=== Preparation complete ===')
    print(f'Validation: {len(val_data)} samples ({VAL_SAMPLES_PER_TYPE} per type)')
    print(f'Next: run "python train.py" to verify baseline, then start autoresearch.')
