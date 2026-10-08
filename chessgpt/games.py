"""
Game-record IO.

One canonical record shape flows through the whole pipeline:

    {"moves": "e2e4 e7e5 g1f3", "result": "1-0", "plies": 3, ...}

`iter_games` reads that shape from .jsonl / .jsonl.zst, and also reads .pgn /
.pgn.zst directly, so a Lichess dump can be dropped in later without touching
any other stage.
"""

import io
import json
import os

try:
    import zstandard
except ImportError:                       # pragma: no cover
    zstandard = None


def _open_text_read(path):
    if path.endswith('.zst'):
        if zstandard is None:
            raise RuntimeError('reading .zst needs `pip install zstandard`')
        fh = open(path, 'rb')
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


class GameWriter:
    """Append-only JSONL writer, zstd-compressed by extension or by `compress`.

    `compress` is explicit because stages write to a `.part` file and rename on
    success, which would otherwise defeat extension sniffing.
    """

    def __init__(self, path, level=10, compress=None):
        self.path = path
        if compress is None:
            compress = path.endswith('.zst')
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        if compress:
            if zstandard is None:
                raise RuntimeError('writing .zst needs `pip install zstandard`')
            self._raw = open(path, 'wb')
            self._cctx = zstandard.ZstdCompressor(level=level)
            self._writer = self._cctx.stream_writer(self._raw)
            self._fh = io.TextIOWrapper(self._writer, encoding='utf-8')
        else:
            self._raw = None
            self._fh = open(path, 'w', encoding='utf-8')
        self.n = 0

    def write(self, record):
        self._fh.write(json.dumps(record, separators=(',', ':')) + '\n')
        self.n += 1

    def close(self):
        self._fh.flush()
        self._fh.close()
        if self._raw is not None:
            self._writer.close()
            self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _iter_jsonl(path):
    with _open_text_read(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_pgn(path, min_plies=1):
    """Stream a PGN file into canonical records (needs python-chess)."""
    import chess.pgn
    with _open_text_read(path) as fh:
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                continue
            if game is None:
                return
            board = game.board()
            ucis = []
            try:
                for mv in game.mainline_moves():
                    ucis.append(mv.uci())
                    board.push(mv)
            except Exception:
                continue
            if len(ucis) < min_plies:
                continue
            h = game.headers

            def _elo(key):
                try:
                    return int(h.get(key, ''))
                except ValueError:
                    return None

            yield {
                'moves': ' '.join(ucis),
                'result': h.get('Result', '*'),
                'plies': len(ucis),
                'white_elo': _elo('WhiteElo'),
                'black_elo': _elo('BlackElo'),
                'termination': h.get('Termination'),
                'time_control': h.get('TimeControl'),
                'eco': h.get('ECO'),
                'source': os.path.basename(path),
            }


def iter_games(paths, min_plies=1):
    """Yield canonical records from any mix of jsonl/pgn files or directories."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    for path in expand(paths):
        base = path[:-4] if path.endswith('.zst') else path
        if base.endswith('.pgn'):
            yield from _iter_pgn(path, min_plies=min_plies)
        else:
            for rec in _iter_jsonl(path):
                if rec.get('plies', len(rec['moves'].split())) >= min_plies:
                    yield rec


def expand(paths):
    """Expand directories into their game files, sorted for reproducibility."""
    out = []
    for p in paths:
        p = str(p)
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(('.jsonl', '.jsonl.zst', '.pgn', '.pgn.zst')):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out
