"""
check_install.py — Verify all dependencies are installed and GPU is available.

Run after pip install:
    python check_install.py
"""

import sys

def check(name, import_name=None):
    """Try to import a package and print its version."""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'ok')
        print(f'  {name:20s} {version}')
        return True
    except ImportError as e:
        print(f'  {name:20s} MISSING — {e}')
        return False


def main():
    print('=== Dependency Check ===\n')

    all_ok = True

    # Core
    print('Core:')
    all_ok &= check('torch')
    all_ok &= check('transformers')
    all_ok &= check('accelerate')

    # Fine-tuning
    print('\nFine-tuning:')
    all_ok &= check('peft')
    all_ok &= check('trl')
    all_ok &= check('datasets')

    # Model-specific
    print('\nModel-specific:')
    all_ok &= check('mamba_ssm')
    all_ok &= check('causal_conv1d')
    all_ok &= check('sentencepiece')

    # Data
    print('\nData:')
    all_ok &= check('polars')

    # GPU
    print('\nGPU:')
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f'  GPU: {name} ({mem:.0f} GB)')
            if mem < 40:
                print(f'  WARNING: {mem:.0f} GB may be tight for SFT+GRPO. Recommend 80GB+.')
        else:
            print('  CUDA not available!')
            all_ok = False
    except Exception as e:
        print(f'  GPU check failed: {e}')
        all_ok = False

    # Data files
    print('\nData files:')
    import os
    for f in ['data/train.csv', 'data/test.csv']:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024
            print(f'  {f:20s} {size:.0f} KB')
        else:
            print(f'  {f:20s} MISSING')
            all_ok = False

    # Transformers version check
    print('\nVersion checks:')
    try:
        import transformers
        v = transformers.__version__
        if v != '4.51.3':
            print(f'  WARNING: transformers=={v}, recommended 4.51.3')
            print(f'    Other versions may crash with Nemotron. Run:')
            print(f'    pip install transformers==4.51.3')
            print(f'    rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/NVIDIA_hyphen_Nemotron*')
        else:
            print(f'  transformers version OK ({v})')
    except ImportError:
        pass

    # Summary
    print(f'\n{"="*40}')
    if all_ok:
        print('All checks passed. Ready to run:')
        print('  python prepare.py   # one-time setup')
        print('  python train.py     # training + eval')
    else:
        print('Some checks FAILED. Fix the issues above before running.')
        sys.exit(1)


if __name__ == '__main__':
    main()
