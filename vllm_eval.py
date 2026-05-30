"""
vllm_eval.py — vLLM-based eval for the autoresearch loop (T2.14, closes Gap A).

Loads the base model + LoRA adapter via vLLM and evaluates on the fixed val_split.
This is the scoring-path-aligned eval: the Kaggle scorer also runs vLLM with
LoRA at the competition spec (max_lora_rank=32, max_tokens=7680, temperature=0,
gpu_memory_utilization=0.85, max_model_len=8192, max_num_seqs=64), so the
METRIC printed here should match the LB METRIC ±0 for the same adapter.

**Why this exists** (07-train-py-gap-analysis.md § Gap A): in-process HF
Transformers eval is wall-clock-dominated by F-001 (Nemotron's
HybridMambaAttentionDynamicCache has shape bugs, forcing use_cache=False).
A 30-sample eval takes ~3 h. vLLM uses its own KV-cache implementation
that bypasses F-001 entirely — same 30-sample eval should land at ~5 min.

Usage (standalone, against a saved adapter):
    python vllm_eval.py                       # uses ./adapter
    python vllm_eval.py --adapter ./adapter   # explicit
    python vllm_eval.py --max-new-tokens 512  # match train.py EVAL_MAX_NEW_TOKENS

Called automatically from train.py when USE_VLLM_EVAL=True (default False
until first verification per branch hygiene).

Output format mirrors train.py's end-of-run block so the autoresearch
METRIC parser is unchanged:

    Overall accuracy: 0.XXXX
      bit_ops: X.XXXX (k/n)
      ...
    Time: NNNNs
    METRIC: 0.XXXX

Install (NOT in requirements.txt — torch/cu version coupling is tight,
see F-013):

    uv pip install 'vllm>=0.6.6,<0.7' --no-deps
    uv pip install ray  # if vllm complains about ray missing on first run

If vllm fails to import or LLM() OOMs, leave USE_VLLM_EVAL=False — the
in-process HF eval in train.py is the documented fallback.
"""

import argparse
import sys
import time

from transformers import AutoTokenizer

from prepare import (
    MODEL_ID, load_val_data, extract_boxed_answer, answers_match,
    METRIC_SUFFIX,
)


def _build_prompt_texts(tokenizer, val_data):
    """Apply the chat template to each val prompt, matching
    prepare.py:_tokenize_prompt's enable_thinking=True path so the prompt
    string passed to vLLM is byte-identical to what evaluate_model produces
    via apply_chat_template. The fallback path mirrors build_sft_text's."""
    prompt_texts = []
    for example in val_data:
        messages = [{'role': 'user', 'content': example['prompt'] + METRIC_SUFFIX}]
        for kwargs in [{'enable_thinking': True}, {}]:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs,
                )
                prompt_texts.append(text)
                break
            except TypeError:
                continue
        else:
            prompt_texts.append(
                f'<|im_start|>user\n{messages[0]["content"]}<|im_end|>\n<|im_start|>assistant\n'
            )
    return prompt_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default='./adapter',
                        help='Path to the saved LoRA adapter dir (default ./adapter)')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Max tokens to generate per sample (default 512, matches train.py EVAL_MAX_NEW_TOKENS)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85,
                        help='vLLM gpu_memory_utilization (default 0.85, competition spec)')
    parser.add_argument('--max-model-len', type=int, default=8192,
                        help='vLLM max_model_len (default 8192, competition spec)')
    parser.add_argument('--max-num-seqs', type=int, default=64,
                        help='vLLM max_num_seqs (default 64, competition spec)')
    args = parser.parse_args()

    start_time = time.time()

    # Heavy imports inside main() so --help works without paying the
    # multi-minute vllm import cost.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f'Loading tokenizer: {MODEL_ID}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    val_data = load_val_data()
    print(f'Loaded {len(val_data)} val samples', flush=True)

    prompt_texts = _build_prompt_texts(tokenizer, val_data)

    # Init vLLM. trust_remote_code=True for Nemotron-H's custom modeling.
    # The competition-spec params (gpu_memory_utilization, max_model_len,
    # max_num_seqs, max_lora_rank) match the Kaggle scorer so this METRIC
    # is directly comparable to the LB.
    print(f'\nInitialising vLLM (model={MODEL_ID})...', flush=True)
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        dtype='bfloat16',
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    # Greedy decoding (temperature=0, top_p=1.0) — matches competition spec
    # and HF in-process evaluate_model's do_sample=False.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )
    lora_request = LoRARequest('autoresearch-adapter', 1, args.adapter)

    print(f'Generating ({len(prompt_texts)} prompts, max_new_tokens={args.max_new_tokens})...', flush=True)
    gen_start = time.time()
    outputs = llm.generate(prompt_texts, sampling_params, lora_request=lora_request)
    gen_elapsed = time.time() - gen_start
    print(f'Generation complete in {gen_elapsed:.0f}s ({gen_elapsed/max(len(prompt_texts),1):.2f}s/sample)', flush=True)

    # Score per-category (same logic as prepare.py:evaluate_model and
    # train.py's debug-print section).
    by_type = {}
    correct = 0
    total = 0
    failing = []

    for example, out in zip(val_data, outputs):
        response = out.outputs[0].text
        pred = extract_boxed_answer(response)
        gold = example['answer']
        qtype = example['qtype']
        is_correct = answers_match(pred, gold)

        total += 1
        correct += int(is_correct)
        if qtype not in by_type:
            by_type[qtype] = {'total': 0, 'correct': 0}
        by_type[qtype]['total'] += 1
        by_type[qtype]['correct'] += int(is_correct)

        if not is_correct:
            failing.append({
                'qtype': qtype, 'gold': gold, 'pred': pred,
                'raw': response[:400],
            })

    overall = correct / max(total, 1)
    elapsed = time.time() - start_time

    # End-of-run block, format matches train.py so the autoresearch parser
    # downstream sees no change.
    print(f'\n{"="*60}')
    print(f'Overall accuracy: {overall:.4f}')
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        acc = stats['correct'] / max(stats['total'], 1)
        print(f'  {qtype}: {acc:.4f} ({stats["correct"]}/{stats["total"]})')
    print(f'Time: {elapsed:.0f}s (gen-only: {gen_elapsed:.0f}s)')
    print(f'{"="*60}')

    # Debug: weak categories (matches train.py's <0.5 threshold)
    print('\n=== Debug: Raw outputs for weak categories ===')
    for f in failing:
        qtype_acc = by_type[f['qtype']]['correct'] / max(by_type[f['qtype']]['total'], 1)
        if qtype_acc < 0.5:
            print(f"\n[{f['qtype']}] gold={f['gold']!r}, pred={f['pred']!r}")
            print(f"  raw: {f['raw']}")

    # === THE METRIC LINE — autoresearch parses this ===
    print(f'METRIC: {overall:.4f}')


if __name__ == '__main__':
    main()
