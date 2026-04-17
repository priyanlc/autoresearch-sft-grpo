"""
train.py — SFT warmup + GRPO training script for autoresearch iteration.
NVFP4 Blackwell branch: FPQuantConfig + synthetic data + cosine reward.

This is the ONLY file the AI agent should modify.
Each run: loads model, SFT warmup, GRPO training, evaluates, prints METRIC.

Usage:
    python train.py
"""

import os
import sys
import gc
import re
import math
import json
import time
import random as _rng_module
import string as _string

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TRITON_JIT_DISABLE_OPT'] = '1'

import torch
import torch.nn as nn
import polars as pl
import wandb
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

# NVFP4 imports — graceful fallback
try:
    from transformers import FPQuantConfig
    _FPQUANT_AVAILABLE = True
except ImportError:
    _FPQUANT_AVAILABLE = False

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
SFT_SAMPLES_PER_TYPE = 200        # 6 * 200 = 1200 real SFT samples
GRPO_SAMPLES_PER_TYPE = 150       # 6 * 150 = 900 GRPO prompts

# Synthetic data — generated in-process with perfect CoT
SYNTH_SAMPLES_PER_TYPE = 500      # 6 * 500 = 3000 synthetic SFT samples
USE_SYNTHETIC = True

# SFT Phase
SFT_LR = 5e-5                    # Lower than 2e-4, more stable (from colab-faster)
SFT_EPOCHS = 1
SFT_MAX_SEQ_LEN = 2048           # Allows longer CoT

# GRPO Phase
GRPO_LR = 5e-6
GRPO_EPOCHS = 1
GRPO_NUM_GENERATIONS = 4
GRPO_MAX_COMPLETION = 1024
GRPO_TEMPERATURE = 0.7
GRPO_BETA = 0.0                  # No KL penalty → no reference model (~60GB saved)
SKIP_GRPO = False                # Set True to skip GRPO (SFT-only fallback)

# LoRA
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0               # Must be 0.0 for FPQuantLinear compatibility

# Training (shared)
BATCH_SIZE = 2                   # NVFP4 frees ~43GB → can use larger batch
GRAD_ACCUM = 4
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 0.1              # Aggressive clipping (from colab-faster)
WEIGHT_DECAY = 0.1               # Regularization (from colab-faster)

# Quantization
USE_NVFP4 = True                 # FPQuantConfig NVFP4 on Blackwell

# Evaluation
EVAL_MAX_NEW_TOKENS = 512
EVAL_BATCH_SIZE = 1

# SFT CoT
USE_COT = True

# Weights & Biases
WANDB_PROJECT = 'nemotron-reasoning'
WANDB_RUN_NAME = None              # Auto-generated if None

# Output
OUTPUT_DIR = './adapter'


# ============================================================
# === SFT PROMPT FORMAT — Controls what model learns
# ============================================================

_COT_BY_TYPE = {
    'bit_ops': "Let me analyze each input-output pair to identify the bit transformation rule.",
    'gravity': "I need to determine g from d = 0.5*g*t^2 using the first example, then apply it.",
    'unit_conv': "I need to find the conversion factor from the examples and apply it.",
    'cipher': "I need to build the substitution mapping from the ciphertext-plaintext pairs.",
    'numeral': "I need to convert the number to Roman numerals by breaking it into components.",
    'symbol': "I need to identify the transformation rules from the examples.",
}
_COT_DEFAULT = "Let me analyze the pattern in the given examples and apply it to solve the problem."


def _build_dynamic_cot(qtype, prompt, answer):
    """Build a category-specific reasoning trace for real data."""
    if qtype == 'gravity':
        m = re.search(r't\s*=\s*([\d.]+).*?distance\s*=\s*([\d.]+)', prompt, re.DOTALL | re.IGNORECASE)
        if m:
            t, d = float(m.group(1)), float(m.group(2))
            g = 2 * d / (t * t) if t > 0 else 0
            return f"Using d = 0.5*g*t^2: g = 2*{d}/{t}^2 = {g:.4f}. Answer: {answer}"
    elif qtype == 'unit_conv':
        m = re.search(r'([\d.]+)\s*[a-zA-Z]+\s*becomes\s*([\d.]+)', prompt)
        if m:
            inp, out = float(m.group(1)), float(m.group(2))
            factor = out / inp if inp > 0 else 1
            return f"Conversion factor = {out}/{inp} = {factor:.4f}. Answer: {answer}"
    elif qtype == 'cipher':
        pairs = re.findall(r'([a-z\s]+?)\s*->\s*([a-z\s]+)', prompt.lower())
        mapping = {}
        for c, p in pairs:
            c, p = c.strip(), p.strip()
            if len(c) == len(p):
                for ci, pi in zip(c, p):
                    if ci.isalpha() and pi.isalpha(): mapping[ci] = pi
        if mapping:
            ms = ', '.join(f'{k}->{v}' for k, v in sorted(mapping.items()))
            return f"Substitution map: {ms}. Decrypting gives: {answer}"
    elif qtype == 'bit_ops':
        return f"Comparing bit patterns to identify the operation. Result: {answer}"
    return None


def build_sft_text(example, tokenizer):
    """Format a training example for SFT.
    Uses generated_cot for synthetic data, dynamic CoT for real data."""
    user_msg = example['prompt'] + METRIC_SUFFIX

    if USE_COT:
        if 'generated_cot' in example and example['generated_cot']:
            cot = example['generated_cot']
        else:
            qtype = classify_type(example['prompt'])
            dynamic = _build_dynamic_cot(qtype, example['prompt'], example['answer'])
            cot = dynamic if dynamic else _COT_BY_TYPE.get(qtype, _COT_DEFAULT)
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
                messages, tokenize=False, add_generation_prompt=False, **kwargs)
            return {'text': text}
        except Exception:
            continue
    return {'text': f'<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>'}


# ============================================================
# === SYNTHETIC DATA GENERATORS — Perfect CoT for all 6 types
# ============================================================

_rng = _rng_module.Random(42)

_NOUNS = ["alice","queen","king","knight","princess","dragon","wizard","castle",
    "mirror","garden","forest","valley","tower","bridge","river","door","key",
    "book","secret","puzzle","treasure","crown","sword","shield","map","scroll",
    "potion","crystal","lantern","student","teacher","cat","turtle","rabbit",
    "phoenix","unicorn","fox","owl","raven","wolf","bear","fairy","elf","hatter"]
_ADJS = ["golden","silver","ancient","magical","mysterious","curious","dark",
    "bright","hidden","enchanted","brave","wise","clever","gentle","fierce"]
_VERBS = ["discovers","creates","follows","watches","explores","reads",
    "imagines","draws","finds","opens","guards","chases","builds","solves"]
_PREPS = ["above","around","beyond","inside","near","through","under"]
_SYMBOLS = list("!@#$%^&*'\"[]:;|\\?<>/-+~`")
_NONSTANDARD_OPS = [
    ('add', lambda a,b: a+b), ('sub', lambda a,b: a-b),
    ('mul', lambda a,b: a*b), ('absdiff', lambda a,b: abs(a-b)),
    ('rsub', lambda a,b: b-a), ('sumsq', lambda a,b: a*a+b*b),
]

def _make_phrase():
    pats = [
        lambda: f"{_rng.choice(_NOUNS)} {_rng.choice(_VERBS)} {_rng.choice(_NOUNS)}",
        lambda: f"{_rng.choice(_NOUNS)} {_rng.choice(_VERBS)} the {_rng.choice(_ADJS)} {_rng.choice(_NOUNS)}",
        lambda: f"the {_rng.choice(_ADJS)} {_rng.choice(_NOUNS)} {_rng.choice(_VERBS)}",
        lambda: f"{_rng.choice(_NOUNS)} {_rng.choice(_VERBS)} {_rng.choice(_PREPS)} {_rng.choice(_NOUNS)}",
    ]
    return _rng.choice(pats)()

def _make_cipher():
    alpha = list(_string.ascii_lowercase)
    shuf = alpha.copy(); _rng.shuffle(shuf)
    return dict(zip(alpha, shuf))

def _encrypt(text, cipher):
    return ''.join(cipher.get(c, c) for c in text.lower())

def _make_bit_op():
    op = _rng.choice(['not','shl','shr','rotl','rotr','xor','and','or','rev','swapnib','swappair'])
    if op == 'not': return lambda x: x ^ 0xFF
    elif op == 'shl':
        n = _rng.randint(1, 3); return lambda x, n=n: (x << n) & 0xFF
    elif op == 'shr':
        n = _rng.randint(1, 3); return lambda x, n=n: (x >> n) & 0xFF
    elif op == 'rotl':
        n = _rng.randint(1, 7); return lambda x, n=n: ((x << n) | (x >> (8-n))) & 0xFF
    elif op == 'rotr':
        n = _rng.randint(1, 7); return lambda x, n=n: ((x >> n) | (x << (8-n))) & 0xFF
    elif op == 'xor':
        c = _rng.randint(1, 255); return lambda x, c=c: x ^ c
    elif op == 'and':
        c = _rng.randint(1, 255); return lambda x, c=c: x & c
    elif op == 'or':
        c = _rng.randint(1, 255); return lambda x, c=c: x | c
    elif op == 'rev': return lambda x: int(f'{x:08b}'[::-1], 2)
    elif op == 'swapnib': return lambda x: ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)
    elif op == 'swappair': return lambda x: ((x & 0x55) << 1) | ((x & 0xAA) >> 1)

def _make_bit_transform():
    ops = [_make_bit_op() for _ in range(_rng.randint(1, 3))]
    def transform(x):
        for op in ops: x = op(x)
        return x
    return transform

def _make_sym_map():
    pool = _SYMBOLS.copy(); _rng.shuffle(pool)
    n = _rng.randint(8, min(15, len(pool)//2))
    return dict(zip(pool[:n], pool[n:2*n]))

def _make_sym_expr(mapping, length=None):
    if length is None: length = _rng.randint(3, 6)
    keys = list(mapping.keys())
    expr = []
    for _ in range(length):
        if _rng.random() < 0.3 and expr and expr[-1] not in {'+','-','*'}:
            expr.append(_rng.choice(['+','-','*']))
        else: expr.append(_rng.choice(keys))
    return ''.join(expr)

def _apply_sym_map(expr, mapping):
    return ''.join(mapping.get(c, c) for c in expr)

def _int_to_roman(num):
    result = ''
    for val, sym in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),
                      (90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
        while num >= val: result += sym; num -= val
    return result

def synth_gravity():
    g = round(_rng.uniform(5.0, 20.0), 2)
    n = _rng.randint(3, 5)
    times = [round(_rng.uniform(1.0, 5.0), 2)] + [round(_rng.uniform(0.5, 5.0), 2) for _ in range(n)]
    ex_t, q_t = times[:-1], times[-1]
    ex_d = [round(0.5 * g * t * t, 2) for t in ex_t]
    lines = ["In Alice's Wonderland, the gravitational constant has been secretly changed. "
             "Here are some example observations:"]
    for t, d in zip(ex_t, ex_d): lines.append(f"For t = {t}s, distance = {d} m")
    lines.append(f"Now, determine the falling distance for t = {q_t}s given d = 0.5*g*t^2.")
    t0, d0 = ex_t[0], ex_d[0]
    g_derived = round(2 * d0 / (t0 ** 2), 2)
    ans = round(0.5 * g_derived * q_t ** 2, 2)
    cot = (f"From d = 0.5*g*t^2, I can solve for g using the first example:\n"
           f"g = 2*d/t^2 = 2*{d0}/{t0}^2 = {g_derived}\n"
           f"Now applying to t = {q_t}s:\nd = 0.5 * {g_derived} * {q_t}^2 = {ans}")
    return '\n'.join(lines), str(ans), cot

def synth_unit_conv():
    factor = round(_rng.uniform(0.2, 3.0), 4)
    n = _rng.randint(3, 5)
    vals = [round(_rng.uniform(10.0, 50.0), 2)] + [round(_rng.uniform(5.0, 50.0), 2) for _ in range(n)]
    ex_v, q_v = vals[:-1], vals[-1]
    ex_c = [round(v * factor, 2) for v in ex_v]
    lines = ["In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:"]
    for v, c in zip(ex_v, ex_c): lines.append(f"{v} m becomes {c}")
    lines.append(f"Now, convert the following measurement: {q_v} m")
    v0, c0 = ex_v[0], ex_c[0]
    df = round(c0 / v0, 4)
    ans = round(q_v * df, 2)
    cot = (f"Looking at the first example: {v0} m becomes {c0}\n"
           f"The conversion factor is {c0} / {v0} = {df}\n"
           f"Applying to {q_v} m: {q_v} * {df} = {ans}")
    return '\n'.join(lines), str(ans), cot

def synth_numeral():
    n = _rng.randint(3, 5)
    nums = _rng.sample(range(1, 101), n + 1)
    ex_n, q_n = nums[:-1], nums[-1]
    lines = ["In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
             "Some examples are given below:"]
    for num in ex_n: lines.append(f"{num} -> {_int_to_roman(num)}")
    lines.append(f"Now, write the number {q_n} in the Wonderland numeral system.")
    ans = _int_to_roman(q_n)
    rem = q_n; steps = []
    for val, sym in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),
                      (90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
        while rem >= val: steps.append(f"{rem} >= {val} -> {sym}"); rem -= val
    cot = f"The examples show Roman numeral conversion. To convert {q_n}:\n" + '\n'.join(steps) + f"\nResult: {ans}"
    return '\n'.join(lines), ans, cot

def synth_cipher():
    cipher = _make_cipher()
    reverse_cipher = {v: k for k, v in cipher.items()}  # plaintext -> ciphertext
    n = _rng.randint(3, 5)
    ex_phrases = [_make_phrase() for _ in range(n)]

    # Build reverse map from examples (ciphertext -> plaintext)
    rmap = {}
    for phrase in ex_phrases:
        enc = _encrypt(phrase, cipher)
        for c, p in zip(enc, phrase.lower()):
            if c.isalpha() and p.isalpha(): rmap[c] = p

    # Regenerate query phrase until ALL its encrypted chars are in rmap
    # This ensures the CoT can fully derive the answer from examples
    for _ in range(100):
        q_phrase = _make_phrase()
        eq = _encrypt(q_phrase, cipher)
        if all(ch in rmap or not ch.isalpha() for ch in eq):
            break

    lines = ["In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:"]
    for phrase in ex_phrases: lines.append(f"{_encrypt(phrase, cipher)} -> {phrase}")
    eq = _encrypt(q_phrase, cipher)
    lines.append(f"Now, decrypt the following text: {eq}")
    map_str = ', '.join(f'{k}->{v}' for k, v in sorted(rmap.items()))
    dec_steps = []
    for ch in eq:
        if ch == ' ': dec_steps.append(' ')
        elif ch in rmap: dec_steps.append(f"{ch}->{rmap[ch]}")
        else: dec_steps.append(f"{ch}->?")
    cot = (f"Building substitution map from examples:\n{map_str}\n\n"
           f"Decrypting '{eq}':\n{' '.join(dec_steps)}\nResult: {q_phrase}")
    return '\n'.join(lines), q_phrase, cot

def synth_bit_ops():
    transform = _make_bit_transform()
    n = _rng.randint(7, 10)
    inputs = _rng.sample(range(256), n + 1)
    examples, query = inputs[:-1], inputs[-1]
    lines = ["In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
             "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
             "and possibly majority or choice functions.", "",
             "Here are some examples of input -> output:"]
    for inp in examples: lines.append(f"{inp:08b} -> {transform(inp):08b}")
    lines.append(f"\nNow, determine the output for: {query:08b}")
    ans = f"{transform(query):08b}"
    analysis = []
    for inp in examples[:3]:
        out = transform(inp)
        analysis.append(f"  {inp:08b} -> {out:08b}  (XOR diff: {inp^out:08b})")
    cot = (f"Analyzing input-output pairs to find the pattern:\n"
           + '\n'.join(analysis)
           + f"\n\nApplying the discovered rule to {query:08b}:\n"
           f"{query:08b} -> {transform(query):08b}\nResult: {ans}")
    return '\n'.join(lines), ans, cot

def synth_symbol():
    if _rng.random() < 0.56:
        # Substitution variant
        mapping = _make_sym_map()
        n = _rng.randint(3, 5)
        lines = ["In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
                 "Below are a few examples:"]
        pairs = []
        for _ in range(n):
            expr = _make_sym_expr(mapping)
            transformed = _apply_sym_map(expr, mapping)
            lines.append(f"{expr} = {transformed}")
            pairs.append((expr, transformed))

        # Build observed mapping from examples only
        dmap = {}
        for e, t in pairs:
            for ci, co in zip(e, t):
                if ci in mapping: dmap[ci] = co

        # Build query using ONLY symbols that appeared in examples
        observed_keys = [k for k in dmap.keys()]
        if not observed_keys:
            observed_keys = list(mapping.keys())  # fallback
        length = _rng.randint(3, 6)
        query_chars = []
        for _ in range(length):
            if _rng.random() < 0.3 and query_chars and query_chars[-1] not in {'+','-','*'}:
                query_chars.append(_rng.choice(['+','-','*']))
            else:
                query_chars.append(_rng.choice(observed_keys))
        query = ''.join(query_chars)
        ans = _apply_sym_map(query, mapping)
        lines.append(f"Now, determine the result for: {query}")

        ms = ', '.join(f'{k}->{v}' for k, v in sorted(dmap.items()))
        steps = [f"{ch}->{dmap[ch]}" if ch in dmap else f"{ch}(unchanged)" for ch in query]
        cot = f"Character substitution mapping from examples:\n{ms}\n\nApplying to '{query}':\n{', '.join(steps)}\nResult: {ans}"
    else:
        # Arithmetic variant
        n_ops = _rng.randint(1, 3)
        chosen = _rng.sample(_NONSTANDARD_OPS, n_ops)
        syms = _rng.sample(_SYMBOLS, n_ops)
        sym_to_op = {s: (nm, fn) for s, (nm, fn) in zip(syms, chosen)}
        n = _rng.randint(3, 5)
        lines = ["In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
                 "Below are a few examples:"]
        ex_info = []
        # Ensure every symbol appears at least once in examples
        for s in syms:
            nm, fn = sym_to_op[s]
            a, b = _rng.randint(1, 99), _rng.randint(1, 99)
            lines.append(f"{a}{s}{b} = {fn(a,b)}")
            ex_info.append((s, a, b, fn(a,b), nm))
        # Fill remaining examples
        for _ in range(n - len(syms)):
            s = _rng.choice(syms); nm, fn = sym_to_op[s]
            a, b = _rng.randint(1, 99), _rng.randint(1, 99)
            lines.append(f"{a}{s}{b} = {fn(a,b)}")
            ex_info.append((s, a, b, fn(a,b), nm))

        # Query uses a symbol that appeared in examples
        seen_syms = list(set(s for s, _, _, _, _ in ex_info))
        qs = _rng.choice(seen_syms); qnm, qfn = sym_to_op[qs]
        qa, qb = _rng.randint(1, 99), _rng.randint(1, 99)
        qr = qfn(qa, qb)
        lines.append(f"Now, determine the result for: {qa}{qs}{qb}")
        ans = str(qr)
        op_lines = []
        for s in syms:
            nm, _ = sym_to_op[s]
            exs = [(a,b,r) for (ss,a,b,r,_) in ex_info if ss == s]
            if exs:
                a,b,r = exs[0]
                op_lines.append(f"  '{s}': {a}{s}{b} = {r} -> {nm}")
        cot = (f"Identifying operations from examples:\n" + '\n'.join(op_lines)
               + f"\n\nFor {qa}{qs}{qb}: '{qs}' means {qnm}, so {qa} {qs} {qb} = {qr}\nResult: {ans}")
    return '\n'.join(lines), ans, cot

_SYNTH_GENERATORS = {
    'gravity': synth_gravity, 'unit_conv': synth_unit_conv,
    'numeral': synth_numeral, 'cipher': synth_cipher,
    'bit_ops': synth_bit_ops, 'symbol': synth_symbol,
}


# ============================================================
# === REWARD FUNCTIONS — Cosine-scaled reward (from colab-faster)
# ============================================================

def _get_content(completion):
    """Coerce completion to string."""
    if isinstance(completion, str): return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get('content', '')
    return str(completion)


_reward_tokenizer = None  # Set in main() after tokenizer is loaded

def cosine_reward(completions, ground_truth, **kwargs):
    """Cosine-scaled accuracy reward using token length (not char length).
    Correct: 0.1→1.0 (shorter=better). Wrong: -0.1→-1.0. No boxed: 0→-0.5."""
    max_len = GRPO_MAX_COMPLETION
    rewards = []
    for comp, gold in zip(completions, ground_truth):
        content = _get_content(comp)
        pred = extract_boxed_answer(content)
        # Use token count for accurate progress calculation
        if _reward_tokenizer is not None:
            clen = len(_reward_tokenizer.encode(content, add_special_tokens=False))
        else:
            clen = len(content) // 4  # fallback: ~4 chars per token
        progress = min(clen / max(max_len, 1), 1.0)
        cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        if pred is not None and answers_match(pred, str(gold)):
            rewards.append(0.1 + 0.9 * cos_scale)
        elif pred is not None:
            rewards.append(-0.1 - 0.9 * (1.0 - cos_scale))
        else:
            rewards.append(-0.5 * progress)
    return rewards


def format_reward(completions, ground_truth, **kwargs):
    """Format reward gated on correctness.
    Correct + boxed = +1.0, correct + no boxed = 0.0, wrong = 0.0.
    This prevents wrong-but-boxed answers from getting positive reward."""
    rewards = []
    for comp, gold in zip(completions, ground_truth):
        content = _get_content(comp)
        pred = extract_boxed_answer(content)
        if pred is not None and answers_match(pred, str(gold)):
            rewards.append(1.0)   # Correct and properly formatted
        else:
            rewards.append(0.0)   # Wrong or no boxed — no format bonus
    return rewards


def category_reward(completions, prompts, ground_truth, **kwargs):
    """Category-aware format validation, gated on correctness.
    Only gives bonus if the answer is correct AND matches category format.
    Wrong answers always get 0.0 — prevents rewarding plausible-looking wrong answers."""
    rewards = []
    for comp, p, gold in zip(completions, prompts, ground_truth):
        content = _get_content(comp)
        prompt_text = p[0]['content'] if isinstance(p, list) else str(p)
        qtype = classify_type(prompt_text)
        pred = extract_boxed_answer(content)

        # Gate on correctness: wrong answers get 0.0
        if pred is None or not answers_match(pred, str(gold)):
            rewards.append(0.0)
            continue

        # Correct answer — check category-specific format
        bonus = 0.0
        if qtype == 'bit_ops' and re.match(r'^[01]{8}$', pred): bonus = 0.5
        elif qtype in ('gravity','unit_conv'):
            try: float(pred); bonus = 0.5
            except ValueError: bonus = -0.5
        elif qtype == 'cipher' and re.match(r'^[a-z\s]+$', pred): bonus = 0.5
        elif qtype == 'numeral' and re.match(r'^[IVXLCDM]+$', pred): bonus = 0.5
        elif qtype == 'symbol' and pred and len(pred) <= 4: bonus = 0.3
        rewards.append(bonus)
    return rewards


# ============================================================
# === DATA LOADING — Real + Synthetic
# ============================================================

def load_training_data(tokenizer):
    """Load and prepare SFT + GRPO training data."""
    train_df = pl.read_csv(TRAIN_CSV)
    train_df = train_df.with_columns(
        pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('qtype')
    )

    # Exclude validation data
    val_df = stratified_sample(train_df, VAL_SAMPLES_PER_TYPE, seed=0)
    val_ids = set(val_df['id'].to_list())
    remaining = train_df.filter(~pl.col('id').is_in(val_ids))

    # SFT real data
    sft_df = stratified_sample(remaining, SFT_SAMPLES_PER_TYPE, seed=42)
    sft_ids = set(sft_df['id'].to_list())

    sft_real = Dataset.from_pandas(sft_df.drop('qtype').to_pandas())
    sft_real = sft_real.map(
        lambda ex: build_sft_text(ex, tokenizer),
        remove_columns=sft_real.column_names,
    )

    # Synthetic data with perfect CoT
    if USE_SYNTHETIC:
        synth_rows = []
        for cat, gen_fn in _SYNTH_GENERATORS.items():
            for i in range(SYNTH_SAMPLES_PER_TYPE):
                prompt, answer, cot = gen_fn()
                synth_rows.append({'id': f'synth_{cat}_{i:05d}', 'prompt': prompt,
                                   'answer': answer, 'generated_cot': cot, 'category': cat})
        sft_synth = Dataset.from_list(synth_rows)
        sft_synth = sft_synth.map(
            lambda ex: build_sft_text(ex, tokenizer),
            remove_columns=sft_synth.column_names,
        )
        sft_dataset = concatenate_datasets([sft_real, sft_synth]).shuffle(seed=42)
        print(f'SFT: {len(sft_real)} real + {len(sft_synth)} synthetic = {len(sft_dataset)} total')
    else:
        sft_dataset = sft_real

    # GRPO data (non-overlapping, uses ground_truth not answer)
    grpo_pool = remaining.filter(~pl.col('id').is_in(sft_ids))
    grpo_df = stratified_sample(grpo_pool, GRPO_SAMPLES_PER_TYPE, seed=123)

    grpo_data = []
    for row in grpo_df.iter_rows(named=True):
        grpo_data.append({
            'prompt': [{'role': 'user', 'content': row['prompt'] + METRIC_SUFFIX}],
            'ground_truth': str(row['answer']),
        })
    grpo_dataset = Dataset.from_list(grpo_data)

    return sft_dataset, grpo_dataset


# ============================================================
# === MAIN
# ============================================================

def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Weights & Biases
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            'sft_samples_per_type': SFT_SAMPLES_PER_TYPE,
            'grpo_samples_per_type': GRPO_SAMPLES_PER_TYPE,
            'synth_samples_per_type': SYNTH_SAMPLES_PER_TYPE,
            'use_synthetic': USE_SYNTHETIC,
            'sft_lr': SFT_LR,
            'sft_epochs': SFT_EPOCHS,
            'sft_max_seq_len': SFT_MAX_SEQ_LEN,
            'grpo_lr': GRPO_LR,
            'grpo_epochs': GRPO_EPOCHS,
            'grpo_num_generations': GRPO_NUM_GENERATIONS,
            'grpo_max_completion': GRPO_MAX_COMPLETION,
            'grpo_temperature': GRPO_TEMPERATURE,
            'grpo_beta': GRPO_BETA,
            'skip_grpo': SKIP_GRPO,
            'lora_rank': LORA_RANK,
            'lora_alpha': LORA_ALPHA,
            'lora_dropout': LORA_DROPOUT,
            'batch_size': BATCH_SIZE,
            'grad_accum': GRAD_ACCUM,
            'max_grad_norm': MAX_GRAD_NORM,
            'weight_decay': WEIGHT_DECAY,
            'use_nvfp4': USE_NVFP4,
            'use_cot': USE_COT,
        },
    )
    print(f'W&B run: {wandb.run.url}')

    # Load tokenizer
    print(f'Loading tokenizer: {MODEL_ID}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set tokenizer for cosine reward token-length calculation
    global _reward_tokenizer
    _reward_tokenizer = tokenizer

    # Prepare data
    print('Preparing training data...')
    sft_dataset, grpo_dataset = load_training_data(tokenizer)
    val_data = load_val_data()
    print(f'SFT: {len(sft_dataset)}, GRPO: {len(grpo_dataset)}, Val: {len(val_data)}')

    # ---- Load model ----
    print(f'Loading model: {MODEL_ID}')
    load_kwargs = dict(device_map={'': 0}, trust_remote_code=True)

    if USE_NVFP4:
        if not _FPQUANT_AVAILABLE:
            raise RuntimeError(
                'USE_NVFP4=True but FPQuantConfig is not available. '
                'Install fp_quant and qutlass: pip install fp_quant qutlass --no-build-isolation. '
                'Requires Blackwell GPU + CUDA 12.8+.'
            )
        load_kwargs['quantization_config'] = FPQuantConfig(
            forward_dtype="nvfp4",
            backward_dtype="bf16",
            pseudoquantization=False,
        )
        load_kwargs['dtype'] = torch.bfloat16
        print('Loading in NVFP4 (Blackwell FP-Quant)...')

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

    # Disable fast path
    for name, mod in sys.modules.items():
        if 'modeling_nemotron_h' in name:
            mod.is_fast_path_available = False

    # Enable gradient checkpointing for quantized models
    if USE_NVFP4:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print('Enabled gradient checkpointing + input grads for quantized model')

    # ---- Patch FPQuantLinear for PEFT compatibility ----
    # PEFT has no native FPQuantLinear support. We patch __bases__ AFTER model
    # load so FPQuantLinear.__init__ already ran normally during from_pretrained.
    if USE_NVFP4 and _FPQUANT_AVAILABLE:
        from fp_quant import FPQuantLinear
        _orig_init = FPQuantLinear.__init__
        FPQuantLinear.__bases__ = (nn.Linear,) + tuple(
            b for b in FPQuantLinear.__bases__ if b not in (nn.Module, nn.Linear))
        FPQuantLinear.__init__ = _orig_init
        print(f'Patched FPQuantLinear: is_Linear={issubclass(FPQuantLinear, nn.Linear)}')

        target_modules = [name for name, mod in model.named_modules()
                          if isinstance(mod, FPQuantLinear)]
        print(f'LoRA targeting {len(target_modules)} FPQuantLinear modules')
    else:
        target_modules = 'all-linear'

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated()/1024**3
        total = torch.cuda.get_device_properties(0).total_memory/1024**3
        print(f'VRAM: {mem:.1f}/{total:.0f} GB ({mem/total*100:.0f}%)')

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
        weight_decay=WEIGHT_DECAY,
        optim='adamw_8bit',
        lr_scheduler_type='cosine',
        warmup_ratio=WARMUP_RATIO,
        save_strategy='no',
        report_to='wandb',
        dataset_text_field='text',
        max_length=SFT_MAX_SEQ_LEN,
        packing=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        args=sft_args,
    )

    print(f'Training {len(sft_dataset)} samples (packing=True)...')
    sft_trainer.train()

    # Save SFT adapter as fallback
    model.save_pretrained(OUTPUT_DIR)
    print(f'SFT adapter saved to {OUTPUT_DIR} (fallback)')

    del sft_trainer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f'VRAM after SFT cleanup: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

    # ---- Phase 2: GRPO Training ----
    _grpo_succeeded = False

    if SKIP_GRPO:
        print('\nGRPO skipped (SKIP_GRPO=True). Using SFT adapter.')
    else:
        print('\n=== Phase 2: GRPO Training ===')

        # Tokenizer patch for GRPO
        if not getattr(tokenizer, '_thinking_patched', False):
            _orig_act = tokenizer.apply_chat_template
            def _act_with_thinking(*args, **kwargs):
                kwargs.setdefault('enable_thinking', True)
                try: return _orig_act(*args, **kwargs)
                except TypeError:
                    kwargs.pop('enable_thinking', None)
                    return _orig_act(*args, **kwargs)
            tokenizer.apply_chat_template = _act_with_thinking
            tokenizer._thinking_patched = True

        if not hasattr(model, 'warnings_issued'):
            model.warnings_issued = {}

        reward_funcs = [cosine_reward, format_reward, category_reward]

        # GRPO batch size must be divisible by num_generations (TRL requirement)
        grpo_batch = max(GRPO_NUM_GENERATIONS, BATCH_SIZE - (BATCH_SIZE % GRPO_NUM_GENERATIONS) if BATCH_SIZE >= GRPO_NUM_GENERATIONS else GRPO_NUM_GENERATIONS)

        grpo_config = GRPOConfig(
            output_dir=OUTPUT_DIR,
            num_generations=GRPO_NUM_GENERATIONS,
            generation_batch_size=GRPO_NUM_GENERATIONS,
            max_completion_length=GRPO_MAX_COMPLETION,
            beta=GRPO_BETA,
            temperature=GRPO_TEMPERATURE,
            per_device_train_batch_size=grpo_batch,
            gradient_accumulation_steps=1,
            num_train_epochs=GRPO_EPOCHS,
            learning_rate=GRPO_LR,
            bf16=True,
            max_grad_norm=MAX_GRAD_NORM,
            weight_decay=WEIGHT_DECAY,
            optim='adamw_8bit',
            lr_scheduler_type='cosine',
            warmup_steps=5,
            logging_steps=5,
            save_strategy='no',
            report_to='wandb',
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            remove_unused_columns=False,
        )

        # Wrap trainer construction + training in try block so config
        # validation errors also fall back to SFT gracefully
        print(f'GRPO: {len(grpo_dataset)} prompts, {GRPO_NUM_GENERATIONS} gens/prompt, batch={grpo_batch}')
        try:
            grpo_trainer = GRPOTrainer(
                model=model,
                reward_funcs=reward_funcs,
                args=grpo_config,
                train_dataset=grpo_dataset,
                processing_class=tokenizer,
            )
            grpo_trainer.train()
            grpo_trainer.model.save_pretrained(OUTPUT_DIR)
            print(f'GRPO adapter saved to {OUTPUT_DIR}')
            _grpo_succeeded = True
        except Exception as e:
            print(f'GRPO failed: {e}')
            print(f'Reloading SFT fallback adapter from {OUTPUT_DIR}...')
            # Reload the clean SFT adapter to avoid evaluating partially-mutated weights
            from peft import PeftModel
            del grpo_trainer
            del model
            gc.collect()
            torch.cuda.empty_cache()
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
            for name, mod in sys.modules.items():
                if 'modeling_nemotron_h' in name:
                    mod.is_fast_path_available = False
            model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
            print(f'SFT adapter reloaded successfully.')
            grpo_trainer = None  # already deleted above

        if grpo_trainer is not None:
            del grpo_trainer
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f'VRAM after GRPO cleanup: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

    # ---- Evaluate ----
    model.config.use_cache = False
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
    if _grpo_succeeded:
        print(f'GRPO: succeeded')
    elif SKIP_GRPO:
        print(f'GRPO: skipped')
    else:
        print(f'GRPO: failed (using SFT fallback)')
    print(f'{"="*60}')

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated()/1024**3
        print(f'Peak VRAM: {peak_vram:.1f} GB')

    # Log final results to W&B
    wandb_log = {
        'metric': overall,
        'elapsed_seconds': elapsed,
        'peak_vram_gb': peak_vram,
        'grpo_succeeded': _grpo_succeeded,
        'grpo_skipped': SKIP_GRPO,
    }
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        acc = stats['correct'] / max(stats['total'], 1)
        wandb_log[f'accuracy/{qtype}'] = acc
    wandb.log(wandb_log)

    # Log summary metrics (visible in W&B runs table)
    wandb.run.summary['metric'] = overall
    wandb.run.summary['elapsed_seconds'] = elapsed
    wandb.run.summary['grpo_succeeded'] = _grpo_succeeded
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        wandb.run.summary[f'accuracy/{qtype}'] = stats['correct'] / max(stats['total'], 1)

    wandb.finish()
    print('W&B run finished.')

    # === THE METRIC LINE — autoresearch parses this ===
    print(f'METRIC: {overall:.4f}')

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
