"""
adapter_sanity_check.py — load the trained LoRA adapter onto a fresh BF16
base and run one inference, in a separate Python process from train.py.

Per program.md § Validation Contract item 5: "Adapter-on-fresh-BF16-base
sanity check — load adapter onto a fresh BF16 base from a *separate
Python process* (not the same process that just trained) and verify a
sample inference works. This is the actual scoring deployment path; most
likely silent-break point."

Reference load pattern: train.py:379-385 (model load) + :386 (fast-path
disable, F-001) + :536 (use_cache=False, F-001).
"""

import os
import sys
import json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from prepare import MODEL_ID, METRIC_SUFFIX, _tokenize_prompt, extract_boxed_answer

ADAPTER_DIR = './adapter'
VAL_JSON = './data/val_split.json'


def main():
    if not os.path.exists(ADAPTER_DIR):
        print(f'FAIL: adapter dir {ADAPTER_DIR!r} does not exist — train.py did not save an adapter.')
        sys.exit(1)
    if not os.path.exists(VAL_JSON):
        print(f'FAIL: validation split {VAL_JSON!r} not found — prepare.py did not run.')
        sys.exit(1)

    with open(VAL_JSON) as f:
        val = json.load(f)
    if not val:
        print('FAIL: validation split is empty.')
        sys.exit(1)
    example = val[0]
    print(f'Sanity-checking adapter on val example qtype={example["qtype"]!r}, gold={example["answer"]!r}')

    print(f'Loading tokenizer: {MODEL_ID}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f'Loading fresh BF16 base: {MODEL_ID}')
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # F-001 defenses — same as train.py:386, :536.
    for name, mod in sys.modules.items():
        if 'modeling_nemotron_h' in name:
            mod.is_fast_path_available = False
    base.config.use_cache = False

    print(f'Loading LoRA adapter from {ADAPTER_DIR}')
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    messages = [{'role': 'user', 'content': example['prompt'] + METRIC_SUFFIX}]
    ids = _tokenize_prompt(messages, tokenizer).unsqueeze(0).to(device)

    print(f'Generating (max_new_tokens=512)...')
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = out[0, ids.shape[1]:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    n_chars = len(text)
    n_tokens = completion_ids.shape[0]
    boxed = extract_boxed_answer(text)
    has_think_tag = '<think>' in text or '</think>' in text

    print('=' * 60)
    print(f'Output length: {n_tokens} tokens / {n_chars} chars')
    print(f'Has <think> tag: {has_think_tag}')
    print(f'Extracted \\boxed{{}}: {boxed!r}')
    print(f'Gold answer: {example["answer"]!r}')
    print('=' * 60)
    print('--- BEGIN OUTPUT (first 800 chars) ---')
    print(text[:800])
    print('--- END OUTPUT ---')

    # Pass criteria: non-empty + structurally plausible.
    # Structural plausibility = at least one of (a) has a \boxed{} we could
    # extract, (b) has the <think> reasoning tag from enable_thinking=True,
    # (c) is non-trivially long. Any single one is enough — we're not
    # measuring METRIC here, just confirming the adapter loads and produces
    # token output through the same code path the eval harness uses.
    if n_chars == 0:
        print('FAIL: output is empty. Adapter likely silently broken.')
        sys.exit(2)
    if boxed is None and not has_think_tag and n_chars < 50:
        print(f'FAIL: output is non-empty but lacks \\boxed{{}}, lacks <think>, and is short ({n_chars} chars). Likely degenerate.')
        sys.exit(3)
    print('PASS: adapter-on-fresh-base sanity check successful.')


if __name__ == '__main__':
    main()
