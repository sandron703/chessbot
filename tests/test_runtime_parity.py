"""The on-device runtime must compute the same thing as PyTorch.

This is the test that lets you trust a board-side result. It covers the plain
forward pass, the KV cache, the extension path used by the move scorer, and the
context-crop path that kicks in once a game outgrows block_size. fp32 export is
held to near-exactness; int8 is allowed quantization noise but must not change
which move comes out on top.
"""

import os
import sys
import tempfile

import numpy as np

from _util import run_tests, tiny_checkpoint
from chessgpt import export
from chessgpt import tokenizer as T
from chessgpt.backends import TorchBackend
from chessgpt.engine import ChessGPT
from chessgpt.runtime_np import NumpyBackend

BLOCK = 96
_FIX = {}


def fixture():
    if not _FIX:
        d = tempfile.mkdtemp(prefix='chessgpt-parity-')
        ckpt = tiny_checkpoint(d, block_size=BLOCK, seed=4)
        f32 = os.path.join(d, 'm-fp32.gptc')
        i8 = os.path.join(d, 'm-int8.gptc')
        export.main(['--ckpt', ckpt, '--out', f32, '--fp32'])
        export.main(['--ckpt', ckpt, '--out', i8])
        _FIX.update(torch=TorchBackend(ckpt, device='cpu'),
                    fp32=NumpyBackend(f32), int8=NumpyBackend(i8),
                    ckpt=ckpt, dir=d)
    return _FIX


def long_prefix(n):
    """A valid token prefix of ~n tokens from a real legal game."""
    import chess
    board = chess.Board()
    ucis = []
    rng = np.random.default_rng(0)
    while len(ucis) * 2 < n and not board.is_game_over():
        moves = list(board.legal_moves)
        mv = moves[int(rng.integers(len(moves)))]
        ucis.append(mv.uci())
        board.push(mv)
    return T.encode_game(ucis, '1-0', eos=False)[:n]


def test_fp32_matches_torch():
    f = fixture()
    worst = 0.0
    for n in (3, 10, 50, BLOCK - 4, BLOCK, BLOCK + 7, 2 * BLOCK + 1):
        pre = long_prefix(n)
        if len(pre) < 3:
            continue
        for exts in ([[]], [[8], [12], [48]], [[8, 24]]):
            a = f['torch'].next_logits(pre, exts)
            b = f['fp32'].next_logits(pre, exts)
            worst = max(worst, float(np.abs(a - b).max()))
    assert worst < 1e-4, f'fp32 runtime diverges from torch by {worst:.2e}'


def test_int8_preserves_argmax():
    f = fixture()
    worst = 0.0
    mismatch = 0
    total = 0
    for n in (10, 50, BLOCK + 7):
        pre = long_prefix(n)
        exts = [[], ]
        a = f['torch'].next_logits(pre, exts)
        b = f['int8'].next_logits(pre, exts)
        worst = max(worst, float(np.abs(a - b).max()))
        mismatch += int(a.argmax(1)[0] != b.argmax(1)[0])
        total += 1
    assert worst < 0.1, f'int8 noise unexpectedly large: {worst:.3e}'
    assert mismatch == 0, f'{mismatch}/{total} argmax changed under int8'


def test_cache_reuse_matches_fresh_cache():
    """Walking a game forward through the incremental cache must give the same
    logits as rebuilding from scratch at every ply."""
    f = fixture()
    warm = f['fp32']          # reuses one cache, advancing it ply by ply
    model = warm.model
    ids = long_prefix(2 * BLOCK)
    for n in range(4, len(ids), 7):
        pre = ids[:n]
        incremental = warm.next_logits(pre, [[]])[0]
        cache = model.new_cache()
        scratch = model.forward(T.crop_prefix(pre, model.block_size),
                                cache, write=True)
        assert np.abs(incremental - scratch).max() < 1e-4, n


def test_same_move_chosen_by_both_backends():
    f = fixture()
    import chess
    board_hist = []
    board = chess.Board()
    rng = np.random.default_rng(1)
    for _ in range(30):
        moves = list(board.legal_moves)
        mv = moves[int(rng.integers(len(moves)))]
        board_hist.append(mv.uci())
        board.push(mv)

    picks = {}
    for name in ('torch', 'fp32', 'int8'):
        player = ChessGPT(f[name], temperature=0.0)
        player.set_position('startpos', board_hist)
        picks[name] = player.select_move().uci()
    assert picks['torch'] == picks['fp32'], picks
    assert picks['torch'] == picks['int8'], picks


def test_gptc_roundtrip_preserves_shapes():
    from chessgpt import gptc
    f = fixture()
    cfg, tensors, directory = gptc.read(f['dir'] + '/m-int8.gptc')
    assert cfg['n_layer'] == 2 and cfg['vocab_size'] == T.VOCAB_SIZE
    assert cfg['block_size'] == BLOCK
    assert tensors['wte'].shape == (T.VOCAB_SIZE, cfg['n_embd'])
    assert tensors['wpe'].shape == (BLOCK, cfg['n_embd'])
    names = [e['name'] for e in directory]
    assert names == gptc.canonical_order(cfg['n_layer'], cfg['bias'])
    assert all(e['offset'] % gptc.ALIGN == 0 for e in directory)
    quantized = [e for e in directory if e['dtype'] == 'int8']
    assert len(quantized) == 4 * cfg['n_layer']


if __name__ == '__main__':
    sys.exit(run_tests(dict(globals())))
