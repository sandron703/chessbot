"""
Dependency-light inference for a `.gptc` model: numpy + a KV cache, nothing
else. This is the code path that actually runs on the UNO Q's Cortex-A53 side,
and it doubles as the reference implementation the optional C runtime in
runtime/ is checked against.

Weights are dequantized to fp32 once at load time. int8 in the file buys file
size and a future C/NEON kernel; numpy has no int8 GEMM to exploit, and a 3M
parameter model is ~13 MB in fp32, which is nothing against 2 GB of LPDDR4.

One forward path covers both prefill (many tokens, empty cache) and extension
(one or two tokens on top of a cache). Extensions do NOT write to the cache, so
the move scorer can try a dozen candidate origins against one shared prefix
without copying the cache even once.
"""

import math
import time

import numpy as np

from . import gptc
from . import tokenizer as T

SQRT2 = math.sqrt(2.0)


def erf(x):
    """Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 -- keeps GELU matched to
    torch's exact (non-tanh) GELU without pulling in scipy."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / SQRT2))


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    y = y * w
    return y + b if b is not None else y


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


class Cache:
    __slots__ = ('k', 'v', 'n')

    def __init__(self, n_layer, n_head, block_size, head_size):
        self.k = np.zeros((n_layer, n_head, block_size, head_size), dtype=np.float32)
        self.v = np.zeros((n_layer, n_head, block_size, head_size), dtype=np.float32)
        self.n = 0

    def reset(self):
        self.n = 0


class GptcModel:
    def __init__(self, path):
        cfg, tensors, _ = gptc.read(path, dequantize=True)
        self.cfg = cfg
        self.t = tensors
        self.n_layer = cfg['n_layer']
        self.n_head = cfg['n_head']
        self.n_embd = cfg['n_embd']
        self.block_size = cfg['block_size']
        self.vocab_size = cfg['vocab_size']
        self.bias = cfg['bias']
        self.head_size = self.n_embd // self.n_head
        self.scale = 1.0 / math.sqrt(self.head_size)
        self.path = path

    # --- parameter access -------------------------------------------------
    def _p(self, name):
        return self.t[name]

    def _b(self, name):
        return self.t[name] if self.bias else None

    def new_cache(self):
        return Cache(self.n_layer, self.n_head, self.block_size, self.head_size)

    # --- the single forward path ------------------------------------------
    def forward(self, ids, cache, write=False):
        """Run `ids` on top of `cache` (which holds cache.n earlier tokens).

        Returns logits for the final token, shape (vocab_size,). With
        write=True the new keys/values are appended to the cache.
        """
        ids = np.asarray(ids, dtype=np.int64)
        m = len(ids)
        n = cache.n
        if n + m > self.block_size:
            raise ValueError(f'context overflow: {n}+{m} > {self.block_size}')

        H, hs, C = self.n_head, self.head_size, self.n_embd
        pos = np.arange(n, n + m)
        x = self._p('wte')[ids] + self._p('wpe')[pos]        # (m, C)

        # causal mask over absolute positions: row i may see column j <= n+i
        if m > 1:
            cols = np.arange(n + m)
            mask = cols[None, :] > (n + np.arange(m))[:, None]  # (m, n+m) True=blocked
        else:
            mask = None

        for l in range(self.n_layer):
            p = f'h.{l}.'
            h = layer_norm(x, self._p(p + 'ln_1.weight'), self._b(p + 'ln_1.bias'))

            qkv = h @ self._p(p + 'attn.c_attn.weight').T
            if self.bias:
                qkv = qkv + self._b(p + 'attn.c_attn.bias')
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(m, H, hs).transpose(1, 0, 2)       # (H, m, hs)
            k = k.reshape(m, H, hs).transpose(1, 0, 2)
            v = v.reshape(m, H, hs).transpose(1, 0, 2)

            if n:
                kf = np.concatenate([cache.k[l, :, :n, :], k], axis=1)
                vf = np.concatenate([cache.v[l, :, :n, :], v], axis=1)
            else:
                kf, vf = k, v
            if write:
                cache.k[l, :, n:n + m, :] = k
                cache.v[l, :, n:n + m, :] = v

            att = (q @ kf.transpose(0, 2, 1)) * self.scale    # (H, m, n+m)
            if mask is not None:
                att = np.where(mask[None, :, :], -np.inf, att)
            att = softmax(att, axis=-1)
            y = (att @ vf).transpose(1, 0, 2).reshape(m, C)

            y = y @ self._p(p + 'attn.c_proj.weight').T
            if self.bias:
                y = y + self._b(p + 'attn.c_proj.bias')
            x = x + y

            h = layer_norm(x, self._p(p + 'ln_2.weight'), self._b(p + 'ln_2.bias'))
            f = h @ self._p(p + 'mlp.c_fc.weight').T
            if self.bias:
                f = f + self._b(p + 'mlp.c_fc.bias')
            f = gelu(f)
            f = f @ self._p(p + 'mlp.c_proj.weight').T
            if self.bias:
                f = f + self._b(p + 'mlp.c_proj.bias')
            x = x + f

        if write:
            cache.n = n + m

        last = layer_norm(x[-1], self._p('ln_f.weight'), self._b('ln_f.bias'))
        return last @ self._p('wte').T                        # weight tying


class NumpyBackend:
    """The `next_logits(prefix, extensions)` interface over GptcModel.

    Holds one cache for the current prefix and reuses it across calls,
    advancing it incrementally when the prefix grows (the normal case while a
    game is played) and rebuilding only when the prefix diverges or the game
    outgrows block_size and the context has to be cropped.
    """

    def __init__(self, path_or_model):
        self.model = (path_or_model if isinstance(path_or_model, GptcModel)
                      else GptcModel(path_or_model))
        self.block_size = self.model.block_size
        self.vocab_size = self.model.vocab_size
        self._cache = self.model.new_cache()
        self._ids = []          # the (possibly cropped) tokens in the cache
        self._crop = 0
        self._last_logits = None
        self.stats = {'prefill_tokens': 0, 'rebuilds': 0, 'ext_calls': 0,
                      'seconds': 0.0}

    def _ensure(self, prefix, max_ext):
        crop = T.crop_offset(prefix, self.block_size)
        eff = list(prefix[crop:])

        if crop != self._crop:
            self._crop, self._ids, self._last_logits = crop, [], None
            self._cache.reset()
            self.stats['rebuilds'] += 1

        shared = 0
        for a, b in zip(self._ids, eff):
            if a != b:
                break
            shared += 1
        if shared < len(self._ids):                    # diverged -> rebuild
            self._cache.reset()
            self._ids, self._last_logits = [], None
            self.stats['rebuilds'] += 1
            shared = 0

        delta = eff[shared:]
        if delta:
            self._last_logits = self.model.forward(delta, self._cache, write=True)
            self._ids = eff
            self.stats['prefill_tokens'] += len(delta)
        elif self._last_logits is None and eff:
            self._cache.reset()
            self._last_logits = self.model.forward(eff, self._cache, write=True)
            self._ids = eff

    def next_logits(self, prefix, extensions):
        t0 = time.perf_counter()
        lens = {len(e) for e in extensions}
        assert len(lens) == 1, 'extensions in one call must share a length'
        self._ensure(prefix, max(lens))
        out = np.empty((len(extensions), self.vocab_size), dtype=np.float32)
        for i, ext in enumerate(extensions):
            if not ext:
                out[i] = self._last_logits
            else:
                out[i] = self.model.forward(ext, self._cache, write=False)
                self.stats['ext_calls'] += 1
        self.stats['seconds'] += time.perf_counter() - t0
        return out
