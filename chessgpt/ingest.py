"""
Stage 1b -- convert a PGN archive into the pipeline's canonical game shards.

`prepare.py` can read a .pgn directly, but python-chess parses a few hundred
games a second, so a 400 MB Lichess dump costs ~20 minutes *every time* you
re-run with different filters. This stage pays that once, in parallel, and
leaves behind .jsonl.zst shards that re-prepare in seconds.

The file is split by byte range; each worker seeks to its offset, skips forward
to the next game boundary, and stops when a game starts past its end -- so no
game is parsed twice or dropped.

    python -m chessgpt.ingest --pgn data/lichess_elite_2020-06.pgn \
        --out data/lichess_elite --workers 18
"""

import argparse
import io
import os
import sys
import time
from multiprocessing import Process, Queue

from .games import GameWriter

EVENT = b'[Event '


def _raw_games(path, start, end):
    """Yield (game_text, start_offset) for games beginning in [start, end)."""
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        fh.seek(start)
        off = start
        if start:
            # we probably landed mid-game; discard until the next header block
            for line in fh:
                off += len(line)
                if line.startswith(EVENT):
                    buf, game_start, have_moves = [line], off - len(line), False
                    break
            else:
                return
        else:
            buf, game_start, have_moves = [], 0, False

        for line in fh:
            if line.startswith(EVENT) and have_moves:
                # Stop BEFORE yielding a game that starts at or past our end:
                # that game belongs to the next worker, which seeks to `end`
                # and aligns forward onto exactly this header. Yielding it here
                # duplicates one game per shard boundary.
                if game_start >= end:
                    return
                yield b''.join(buf), game_start
                buf, game_start, have_moves = [line], off, False
            else:
                buf.append(line)
                if line.strip() and not line.startswith(b'['):
                    have_moves = True
            off += len(line)
            if off >= size:
                break
        if buf and have_moves and game_start < end:
            yield b''.join(buf), game_start


def _record(game, path):
    board = game.board()
    ucis = []
    for mv in game.mainline_moves():
        ucis.append(mv.uci())
        board.push(mv)
    h = game.headers

    def elo(key):
        try:
            return int(h.get(key, ''))
        except ValueError:
            return None

    return {
        'moves': ' '.join(ucis),
        'result': h.get('Result', '*'),
        'plies': len(ucis),
        'white_elo': elo('WhiteElo'),
        'black_elo': elo('BlackElo'),
        'termination': h.get('Termination'),
        'time_control': h.get('TimeControl'),
        'eco': h.get('ECO'),
        'source': os.path.basename(path),
    }


def worker(wid, path, start, end, out_dir, min_plies, progress):
    import chess.pgn
    shard = os.path.join(out_dir, f'shard-{wid:03d}.jsonl.zst')
    tmp = shard + '.part'
    n = bad = 0
    with GameWriter(tmp, compress=True) as w:
        for text, _ in _raw_games(path, start, end):
            try:
                game = chess.pgn.read_game(io.StringIO(text.decode('utf-8', 'replace')))
                if game is None:
                    continue
                rec = _record(game, path)
            except Exception:
                bad += 1
                continue
            if rec['plies'] < min_plies:
                continue
            w.write(rec)
            n += 1
            if n % 500 == 0:
                progress.put(('n', wid, 500))
    os.replace(tmp, shard)
    progress.put(('done', wid, (n, bad)))


def main(argv=None):
    p = argparse.ArgumentParser(description='PGN archive -> canonical game shards')
    p.add_argument('--pgn', required=True, help='.pgn file (uncompressed)')
    p.add_argument('--out', required=True, help='output shard directory')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--min-plies', type=int, default=0,
                   help='0 keeps everything -- ingest is a lossless format '
                        'conversion, and prepare.py does the filtering, where '
                        'changing your mind costs seconds instead of minutes')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)

    if args.pgn.endswith('.zst'):
        sys.exit('decompress first (zstd -d); byte-range splitting needs random '
                 'access. `prepare.py --src file.pgn.zst` reads it directly but '
                 'single-threaded.')
    os.makedirs(args.out, exist_ok=True)
    existing = [f for f in os.listdir(args.out) if f.endswith('.jsonl.zst')]
    if existing and not args.overwrite:
        sys.exit(f'{args.out} already holds {len(existing)} shards '
                 f'(--overwrite to redo)')

    size = os.path.getsize(args.pgn)
    bounds = [size * w // args.workers for w in range(args.workers + 1)]
    progress = Queue()
    procs = [Process(target=worker,
                     args=(w, args.pgn, bounds[w], bounds[w + 1], args.out,
                           args.min_plies, progress))
             for w in range(args.workers)]
    t0 = time.time()
    for pr in procs:
        pr.start()

    total = bad = finished = 0
    while finished < len(procs):
        kind, wid, val = progress.get()
        if kind == 'n':
            total += val
            el = time.time() - t0
            sys.stderr.write(f'\r{total:,} games  {total/max(el,1e-9):.0f}/s  '
                             f'{el:.0f}s   ')
            sys.stderr.flush()
        else:
            n, b = val
            total += n % 500
            bad += b
            finished += 1
    for pr in procs:
        pr.join()
    sys.stderr.write('\n')

    el = time.time() - t0
    print(f'{total:,} games -> {args.out} in {el:.0f}s ({total/el:.0f}/s)')
    if bad:
        print(f'{bad:,} games failed to parse and were dropped')
    print(f'next: python -m chessgpt.prepare --src {args.out} --dataset chess_uci')


if __name__ == '__main__':
    main()
