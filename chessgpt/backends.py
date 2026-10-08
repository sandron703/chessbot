"""
Model backends. Both expose the same two-line interface used by masking.py:

    backend.next_logits(prefix_ids, extensions) -> (len(extensions), vocab)
    backend.block_size

TorchBackend reads a nanoGPT checkpoint and is what you evaluate with on the
desktop. NumpyBackend (runtime_np.py) reads an exported .gptc and is what runs
on the board. Everything above this line -- masking, the engine, the UCI loop,
the whole evaluation suite -- is backend-agnostic, so the board runs exactly
the code the desktop was measured on.
"""

import os
import sys

import numpy as np

from . import tokenizer as T

NANOGPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'nanoGPT')


def _import_nanogpt():
    if NANOGPT_DIR not in sys.path:
        sys.path.insert(0, NANOGPT_DIR)
    from model import GPT, GPTConfig          # noqa: E402
    return GPT, GPTConfig


class TorchBackend:
    def __init__(self, ckpt_path, device=None, dtype='float32'):
        import torch
        GPT, GPTConfig = _import_nanogpt()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch = torch
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model_args = ckpt['model_args']
        model = GPT(GPTConfig(**self.model_args))
        sd = ckpt['model']
        for k in list(sd):
            if k.startswith('_orig_mod.'):
                sd[k[len('_orig_mod.'):]] = sd.pop(k)
        model.load_state_dict(sd)
        model.eval().to(device)
        if dtype == 'float16' and device.startswith('cuda'):
            model = model.half()
        self.model = model
        self.block_size = self.model_args['block_size']
        self.vocab_size = self.model_args['vocab_size']
        self.iter_num = ckpt.get('iter_num')
        self.best_val_loss = float(ckpt.get('best_val_loss', float('nan')))
        self.config = ckpt.get('config', {})

    @property
    def n_params(self):
        return self.model.get_num_params()

    def next_logits(self, prefix, extensions):
        torch = self.torch
        lens = {len(e) for e in extensions}
        assert len(lens) == 1, 'extensions in one call must share a length'
        base = T.crop_prefix(prefix, self.block_size)
        seqs = [base + list(e) for e in extensions]
        x = torch.tensor(seqs, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
        return logits[:, -1, :].float().cpu().numpy().astype(np.float32)


def load_backend_or_exit(path, device=None):
    """load_backend, but report a missing model as a one-line CLI error."""
    try:
        return load_backend(path, device=device)
    except FileNotFoundError as e:
        sys.exit(str(e))


def load_backend(path, device=None):
    """Dispatch on extension: .pt -> TorchBackend, .gptc -> NumpyBackend."""
    if not os.path.exists(path):
        available = []
        for d, pat in (('out', '.gptc'), (NANOGPT_DIR, 'ckpt.pt')):
            if os.path.isdir(d):
                available += [os.path.join(d, f) for f in sorted(os.listdir(d))
                              if f.endswith(pat)]
        for name in sorted(os.listdir(NANOGPT_DIR)) if os.path.isdir(NANOGPT_DIR) else []:
            ck = os.path.join(NANOGPT_DIR, name, 'ckpt.pt')
            if name.startswith('out-') and os.path.exists(ck):
                available.append(ck)
        msg = [f'no model at {path!r}.']
        if available:
            msg.append('Models you do have: ' + ', '.join(available))
        msg.append('To make one: make data && make prepare && make train && make export')
        raise FileNotFoundError('  '.join(msg))
    if path.endswith('.gptc'):
        from .runtime_np import NumpyBackend
        return NumpyBackend(path)
    return TorchBackend(path, device=device)
