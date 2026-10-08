"""Shared test helpers. Tests are self-contained: they build a tiny random
checkpoint rather than depending on a trained model."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessgpt import tokenizer as T          # noqa: E402
from chessgpt.backends import _import_nanogpt  # noqa: E402


def tiny_checkpoint(dirpath=None, n_layer=2, n_head=2, n_embd=64,
                    block_size=96, bias=False, seed=0):
    """Save a randomly-initialized nanoGPT checkpoint; return its path."""
    import torch
    GPT, GPTConfig = _import_nanogpt()
    torch.manual_seed(seed)
    args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                block_size=block_size, bias=bias, vocab_size=T.VOCAB_SIZE,
                dropout=0.0)
    model = GPT(GPTConfig(**args))
    model.eval()
    dirpath = dirpath or tempfile.mkdtemp(prefix='chessgpt-test-')
    path = os.path.join(dirpath, 'ckpt.pt')
    torch.save({'model': model.state_dict(), 'model_args': args,
                'iter_num': 0, 'best_val_loss': 9.9, 'config': {}}, path)
    return path


def run_tests(module_globals):
    """Minimal runner so tests work with or without pytest installed."""
    fns = [(k, v) for k, v in sorted(module_globals.items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'  ok   {name}')
        except Exception as e:
            failed += 1
            import traceback
            print(f'  FAIL {name}: {e}')
            traceback.print_exc()
    print(f'{len(fns)-failed}/{len(fns)} passed')
    return 1 if failed else 0
