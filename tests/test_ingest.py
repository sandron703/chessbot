"""Byte-range PGN splitting must be exact.

The first version of this lost nothing but duplicated one game per shard
boundary -- invisible in a 425k-game archive unless you count, and it quietly
inflates the dataset with near-duplicate windows. These tests pin the
boundary behaviour down with worker counts that force splits mid-game,
mid-header and exactly on a game start.
"""

import os
import random
import shutil
import sys
import tempfile

import chess

from _util import run_tests
from chessgpt import ingest
from chessgpt.games import iter_games


def make_pgn(path, n_games, seed=0):
    """Write n_games of random-but-legal chess with varied lengths."""
    rng = random.Random(seed)
    expected = []
    with open(path, 'w') as f:
        for g in range(n_games):
            board = chess.Board()
            ucis = []
            target = rng.randrange(2, 40)
            while len(ucis) < target and not board.is_game_over():
                moves = list(board.legal_moves)
                mv = moves[rng.randrange(len(moves))]
                ucis.append(mv.uci())
                board.push(mv)
            result = ['1-0', '0-1', '1/2-1/2'][g % 3]
            expected.append(' '.join(ucis))
            f.write(f'[Event "Test game {g}"]\n')
            f.write(f'[White "w{g}"]\n[Black "b{g}"]\n')
            f.write(f'[Result "{result}"]\n')
            f.write(f'[WhiteElo "{2000 + g % 500}"]\n')
            f.write(f'[BlackElo "{2100 + g % 400}"]\n')
            f.write(f'[Termination "{"Normal" if g % 4 else "Time forfeit"}"]\n')
            f.write('\n')
            # SAN movetext, wrapped, as real PGN is
            board = chess.Board()
            san = []
            for i, u in enumerate(ucis):
                mv = chess.Move.from_uci(u)
                if i % 2 == 0:
                    san.append(f'{i//2+1}.')
                san.append(board.san(mv))
                board.push(mv)
            san.append(result)
            line = ''
            for tok in san:
                if len(line) + len(tok) + 1 > 76:
                    f.write(line + '\n')
                    line = ''
                line += (' ' if line else '') + tok
            f.write(line + '\n\n')
    return expected


def _ingest(pgn, workers, tmp):
    out = os.path.join(tmp, f'shards{workers}')
    if os.path.isdir(out):
        shutil.rmtree(out)
    ingest.main(['--pgn', pgn, '--out', out, '--workers', str(workers)])
    return [r['moves'] for r in iter_games(out)]


def test_split_is_exact_for_many_worker_counts():
    tmp = tempfile.mkdtemp(prefix='chessgpt-ingest-')
    try:
        pgn = os.path.join(tmp, 'games.pgn')
        expected = make_pgn(pgn, 120, seed=5)
        for workers in (1, 2, 3, 5, 8, 17, 64):
            got = _ingest(pgn, workers, tmp)
            assert len(got) == len(expected), \
                f'{workers} workers: {len(got)} games, expected {len(expected)}'
            assert sorted(got) == sorted(expected), \
                f'{workers} workers: game contents differ'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_more_workers_than_games_loses_nothing():
    """Every worker's range is then smaller than one game -- the case most
    likely to drop or double a game at a boundary."""
    tmp = tempfile.mkdtemp(prefix='chessgpt-ingest-')
    try:
        pgn = os.path.join(tmp, 'few.pgn')
        expected = make_pgn(pgn, 3, seed=11)
        got = _ingest(pgn, 24, tmp)
        assert sorted(got) == sorted(expected), (len(got), len(expected))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_headers_survive_the_roundtrip():
    tmp = tempfile.mkdtemp(prefix='chessgpt-ingest-')
    try:
        pgn = os.path.join(tmp, 'g.pgn')
        make_pgn(pgn, 40, seed=3)
        out = os.path.join(tmp, 'sh')
        ingest.main(['--pgn', pgn, '--out', out, '--workers', '4'])
        recs = list(iter_games(out))
        assert len(recs) == 40
        for r in recs:
            assert r['result'] in ('1-0', '0-1', '1/2-1/2')
            assert isinstance(r['white_elo'], int) and r['white_elo'] >= 2000
            assert r['termination'] in ('Normal', 'Time forfeit')
            assert r['plies'] == len(r['moves'].split())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_filters_apply():
    from chessgpt.prepare import encode_records
    recs = [
        {'moves': 'e2e4 e7e5 g1f3 b8c6 f1b5 g8f6', 'result': '1-0', 'plies': 6,
         'white_elo': 2500, 'black_elo': 2400, 'termination': 'Normal'},
        {'moves': 'd2d4 d7d5 c2c4 e7e6 b1c3 g8f6', 'result': '0-1', 'plies': 6,
         'white_elo': 1800, 'black_elo': 2400, 'termination': 'Normal'},
        {'moves': 'e2e4 c7c5 g1f3 d7d6 d2d4 c5d4', 'result': '1-0', 'plies': 6,
         'white_elo': 2500, 'black_elo': 2500, 'termination': 'Time forfeit'},
    ]
    keep, stats = encode_records(recs, 4, 0, False, None)
    assert stats['kept'] == 3

    keep, stats = encode_records(recs, 4, 0, False, None, min_elo=2400)
    assert stats['kept'] == 2 and stats['drop_elo_low'] == 1

    keep, stats = encode_records(recs, 4, 0, False, None,
                                 terminations={'Normal'})
    assert stats['kept'] == 2 and stats['drop_termination'] == 1

    keep, stats = encode_records(recs, 4, 0, False, None,
                                 min_elo=2400, terminations={'Normal'})
    assert stats['kept'] == 1


if __name__ == '__main__':
    sys.exit(run_tests(dict(globals())))
