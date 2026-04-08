"""Quick eval of the saved SFT adapter."""
import os, sys, torch, time, gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TRITON_JIT_DISABLE_OPT'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from prepare import MODEL_ID, evaluate_model, load_val_data, extract_boxed_answer, answers_match, METRIC_SUFFIX, _tokenize_prompt

OUTPUT_DIR = './adapter'

print(f'Loading tokenizer: {MODEL_ID}', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

val_data = load_val_data()

print(f'Loading model: {MODEL_ID}', flush=True)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

# Disable fast path and patch MoE
for name, mod in sys.modules.items():
    if 'modeling_nemotron_h' in name:
        mod.is_fast_path_available = False

# Load adapter
print(f'Loading adapter from {OUTPUT_DIR}', flush=True)
model = PeftModel.from_pretrained(model, OUTPUT_DIR)
model.config.use_cache = False

print(f'\nEvaluating...', flush=True)
overall, by_type = evaluate_model(model, tokenizer, val_data, max_new_tokens=128, batch_size=1)

print(f'\n{"="*60}')
print(f'Overall accuracy: {overall:.4f}')
for qtype in sorted(by_type.keys()):
    stats = by_type[qtype]
    acc = stats['correct'] / max(stats['total'], 1)
    print(f'  {qtype}: {acc:.4f} ({stats["correct"]}/{stats["total"]})')
print(f'{"="*60}')
print(f'METRIC: {overall:.4f}')
