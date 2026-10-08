"""
The `.gptc` on-device model container.

Layout:

    [0:4]    magic  "GPTC"
    [4:8]    u32 version (1)
    [8:12]   u32 json_len
    [12:16]  u32 data_offset   (16 + json_len, rounded up to 64)
    [16:...] UTF-8 JSON: {"config": {...}, "tensors": [{name,dtype,shape,offset,nbytes,scale_offset?}]}
    [data_offset:] tensor blobs, 64-byte aligned, in the canonical order below

Big tensors (the attention and MLP projections -- >95% of the parameters) are
stored as int8 with one fp32 scale per output row. Embeddings, LayerNorms and
biases stay fp32: they are a rounding error in file size and the output head is
exactly where quantization noise would cost the most logit fidelity.

The canonical tensor order is fixed, so a C runtime can mmap the blob and walk
it with the offsets from `export.py --emit-header` and never parse JSON.
"""

import json
import os
import struct

import numpy as np

MAGIC = b'GPTC'
VERSION = 1
ALIGN = 64

QUANT_SUFFIXES = (
    'attn.c_attn.weight',
    'attn.c_proj.weight',
    'mlp.c_fc.weight',
    'mlp.c_proj.weight',
)


def canonical_order(n_layer, bias):
    names = ['wte', 'wpe']
    for l in range(n_layer):
        p = f'h.{l}.'
        names.append(p + 'ln_1.weight')
        if bias:
            names.append(p + 'ln_1.bias')
        names.append(p + 'attn.c_attn.weight')
        if bias:
            names.append(p + 'attn.c_attn.bias')
        names.append(p + 'attn.c_proj.weight')
        if bias:
            names.append(p + 'attn.c_proj.bias')
        names.append(p + 'ln_2.weight')
        if bias:
            names.append(p + 'ln_2.bias')
        names.append(p + 'mlp.c_fc.weight')
        if bias:
            names.append(p + 'mlp.c_fc.bias')
        names.append(p + 'mlp.c_proj.weight')
        if bias:
            names.append(p + 'mlp.c_proj.bias')
    names.append('ln_f.weight')
    if bias:
        names.append('ln_f.bias')
    return names


def should_quantize(name):
    return any(name.endswith(s) for s in QUANT_SUFFIXES)


def quantize_rows(w):
    """Per-output-row symmetric int8. w is [out, in] -> (int8[out,in], fp32[out])."""
    w = np.asarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1)
    amax[amax == 0] = 1.0
    scale = (amax / 127.0).astype(np.float32)
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def dequantize_rows(q, scale):
    return q.astype(np.float32) * scale[:, None]


def _pad(n, align=ALIGN):
    return (n + align - 1) // align * align


def write(path, config, tensors, quant=True):
    """Write a .gptc. `tensors` maps name -> fp32 ndarray, in canonical order."""
    names = canonical_order(config['n_layer'], config['bias'])
    missing = [n for n in names if n not in tensors]
    if missing:
        raise KeyError(f'missing tensors: {missing[:4]}')

    directory = []
    blobs = []
    off = 0
    for name in names:
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        entry = {'name': name, 'shape': list(arr.shape), 'offset': off}
        if quant and should_quantize(name) and arr.ndim == 2:
            q, scale = quantize_rows(arr)
            entry['dtype'] = 'int8'
            entry['nbytes'] = int(q.nbytes)
            blobs.append(q.tobytes())
            off = _pad(off + q.nbytes)
            entry['scale_offset'] = off
            entry['scale_nbytes'] = int(scale.nbytes)
            blobs.append(b'\x00' * (off - entry['offset'] - q.nbytes))
            blobs.append(scale.tobytes())
            off = _pad(off + scale.nbytes)
            blobs.append(b'\x00' * (off - entry['scale_offset'] - scale.nbytes))
        else:
            entry['dtype'] = 'fp32'
            entry['nbytes'] = int(arr.nbytes)
            blobs.append(arr.tobytes())
            off = _pad(off + arr.nbytes)
            blobs.append(b'\x00' * (off - entry['offset'] - arr.nbytes))
        directory.append(entry)

    meta = json.dumps({'config': config, 'tensors': directory},
                      separators=(',', ':')).encode('utf-8')
    data_offset = _pad(16 + len(meta))
    head = MAGIC + struct.pack('<III', VERSION, len(meta), data_offset)
    with open(path, 'wb') as f:
        f.write(head)
        f.write(meta)
        f.write(b'\x00' * (data_offset - 16 - len(meta)))
        for b in blobs:
            f.write(b)
    return os.path.getsize(path)


def read(path, dequantize=True):
    """Read a .gptc -> (config, {name: fp32 ndarray}, directory)."""
    with open(path, 'rb') as f:
        head = f.read(16)
        if head[:4] != MAGIC:
            raise ValueError(f'{path}: not a .gptc file')
        version, json_len, data_offset = struct.unpack('<III', head[4:16])
        if version != VERSION:
            raise ValueError(f'{path}: unsupported .gptc version {version}')
        meta = json.loads(f.read(json_len).decode('utf-8'))
        f.seek(data_offset)
        blob = f.read()

    config = meta['config']
    tensors = {}
    for e in meta['tensors']:
        shape = tuple(e['shape'])
        if e['dtype'] == 'int8':
            q = np.frombuffer(blob, dtype=np.int8, count=int(np.prod(shape)),
                              offset=e['offset']).reshape(shape)
            scale = np.frombuffer(blob, dtype=np.float32, count=shape[0],
                                  offset=e['scale_offset'])
            tensors[e['name']] = dequantize_rows(q, scale) if dequantize else (q, scale)
        else:
            tensors[e['name']] = np.frombuffer(
                blob, dtype=np.float32, count=int(np.prod(shape)),
                offset=e['offset']).reshape(shape).copy()
    return config, tensors, meta['tensors']
