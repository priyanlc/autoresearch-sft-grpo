"""
sanity_check.py — Adapter-on-fresh-BF16-base validation (Validation Contract point 5).

Loads the trained adapter from OUTPUT_DIR onto a fresh BF16 base in this
*separate* Python process (not the same process that trained), runs inference
on one validation example, and verifies the output is non-empty and structurally
plausible (parseable boxed answer).

This is the actual scoring deployment path — a silent break here means the
adapter+base combination doesn't generate, even if train.py produced a METRIC.

Run AFTER `python train.py` completes:
    python sanity_check.py

Exit 0: sanity check passed.
Exit 1: silent break — log to FRICTION.md.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from prepare import (
    MODEL_ID,
    METRIC_SUFFIX,
    extract_boxed_answer,
    answers_match,
    load_val_data,
    _tokenize_prompt,
)

OUTPUT_DIR = './adapter'           # matches train.py:86
EVAL_MAX_NEW_TOKENS = 512          # matches train.py:76


def main():
    print('=== Sanity check: adapter on fresh BF16 base ===')
    print(f'Adapter: {OUTPUT_DIR}')
    print(f'Base:    {MODEL_ID}')

    val_data = load_val_data()
    if not val_data:
        print('FAIL: no validation data — run prepare.py first.')
        sys.exit(1)
    example = val_data[0]
    print(f'Sample qtype: {example["qtype"]}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # F-001: disable Mamba fast path (matches train.py:386-389)
    for name, mod in sys.modules.items():
        if 'modeling_nemotron_h' in name:
            mod.is_fast_path_available = False

    # F-001: disable KV cache before generation (matches train.py:536)
    base.config.use_cache = False

    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    model.eval()

    messages = [{'role': 'user', 'content': example['prompt'] + METRIC_SUFFIX}]
    ids = _tokenize_prompt(messages, tokenizer).unsqueeze(0).to('cuda')

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    pred = extract_boxed_answer(response)
    match = answers_match(pred, example['answer']) if pred else False

    print('\n--- Response (first 400 chars) ---')
    print(response[:400])
    print('--- Result ---')
    print(f'Predicted answer: {pred}')
    print(f'Gold answer:      {example["answer"]}')
    print(f'Answers match:    {match}')

    if not response.strip():
        print('\nFAIL: empty response. Adapter+base combo cannot generate.')
        sys.exit(1)
    if pred is None:
        print('\nFAIL: response generated but no parseable \\boxed{} answer.')
        print('      Adapter loads and generates, but output is structurally suspect.')
        print('      Investigate before trusting METRIC from train.py.')
        sys.exit(1)

    print('\nPASS: adapter loads onto fresh BF16 base, generates non-empty,')
    print('      structurally valid output. Deployment path is alive.')
    sys.exit(0)


if __name__ == '__main__':
    main()
