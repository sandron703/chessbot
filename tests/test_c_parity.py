"""Hold the C runtime to the numpy reference.

Skips itself if runtime/gptc-cli has not been built (`make -C runtime`), so it
never blocks the Python-only path -- but once you build the C runtime, this is
what says it is trustworthy. Both int8 and fp32 exports are checked, with the
same prefix/extension/crop patterns the move scorer actually uses.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

from _util import run_tests, tiny_checkpoint
from chessgpt import export
from chessgpt import tokenizer as T
from chessgpt.runtime_np import NumpyBackend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, 'runtime', 'gptc-cli')
BLOCK = 96


class Skip(Exception):
    pass


class CRuntime:
    """Drive runtime/gptc-cli over its line protocol."""

    def __init__(self, model_path):
        self.p = subprocess.Popen([CLI, model_path], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)

    def _cmd(self, line):
        self.p.stdin.write(line + '\n')
        self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def info(self):
        return dict(zip(*[iter(self._cmd('info').split())] * 2))

    def reset(self):
        self._cmd('reset')

    def commit(self, ids):
        if ids:
            assert self._cmd('commit ' + ' '.join(map(str, ids))).startswith('ok')

    def logits(self, ids):
        out = self._cmd('logits ' + ' '.join(map(str, ids)))
        parts = out.split()
        assert parts[0] != 'err', out
        n = int(parts[0])
        return np.array([float(v) for v in parts[1:n + 1]], dtype=np.float64)

    def close(self):
        try:
            self._cmd('quit')
        except Exception:
            pass
        self.p.terminate()


_FIX = {}


def fixture():
    if not _FIX:
        if not os.path.exists(CLI):
            raise Skip('runtime/gptc-cli not built -- run `make -C runtime`')
        d = tempfile.mkdtemp(prefix='chessgpt-cparity-')
        ckpt = tiny_checkpoint(d, block_size=BLOCK, seed=9)
        paths = {}
        for tag, extra in (('fp32', ['--fp32']), ('int8', [])):
            paths[tag] = os.path.join(d, f'm-{tag}.gptc')
            export.main(['--ckpt', ckpt, '--out', paths[tag]] + extra)
        _FIX.update(paths=paths, dir=d)
    return _FIX


def game_prefix(n):
    import chess
    board = chess.Board()
    ucis = []
    rng = np.random.default_rng(0)
    while len(ucis) * 2 < n + 4 and not board.is_game_over():
        moves = list(board.legal_moves)
        mv = moves[int(rng.integers(len(moves)))]
        ucis.append(mv.uci())
        board.push(mv)
    return T.encode_game(ucis, '1-0', eos=False)[:n]


def _compare(tag):
    f = fixture()
    path = f['paths'][tag]
    npb = NumpyBackend(path)
    c = CRuntime(path)
    try:
        worst = 0.0
        for n in (4, 20, 60, BLOCK - 4):
            prefix = game_prefix(n)
            if len(prefix) < 4:
                continue
            eff = T.crop_prefix(prefix, BLOCK)

            # logits at the end of the prefix. The C side is asked for them by
            # committing all but the last token and running the last one
            # uncommitted, which also exercises the commit/no-commit boundary.
            a = npb.next_logits(prefix, [[]])[0]
            c.reset()
            c.commit(eff[:-1])
            b = c.logits([eff[-1]])
            worst = max(worst, float(np.abs(a - b).max()))

            # one- and two-token extensions, the move-scorer pattern
            for exts in ([[8], [12], [48]], [[8, 24]]):
                A = npb.next_logits(prefix, exts)
                c.reset()
                c.commit(eff)
                for i, ext in enumerate(exts):
                    B = c.logits(ext)
                    worst = max(worst, float(np.abs(A[i] - B).max()))
        return worst
    finally:
        c.close()


def test_c_fp32_matches_numpy():
    try:
        worst = _compare('fp32')
    except Skip as e:
        print(f'  (skipped: {e})')
        return
    print(f'  max |dlogit| vs numpy: {worst:.2e}')
    assert worst < 1e-3, f'C fp32 diverges from numpy by {worst:.3e}'


def test_c_int8_matches_numpy():
    try:
        worst = _compare('int8')
    except Skip as e:
        print(f'  (skipped: {e})')
        return
    print(f'  max |dlogit| vs numpy: {worst:.2e}')
    assert worst < 1e-3, f'C int8 diverges from numpy by {worst:.3e}'


def test_c_reports_matching_config():
    try:
        f = fixture()
    except Skip as e:
        print(f'  (skipped: {e})')
        return
    c = CRuntime(f['paths']['int8'])
    try:
        info = c.info()
        assert int(info['vocab_size']) == T.VOCAB_SIZE
        assert int(info['block_size']) == BLOCK
        assert int(info['n_layer']) == 2
    finally:
        c.close()


if __name__ == '__main__':
    sys.exit(run_tests(dict(globals())))
