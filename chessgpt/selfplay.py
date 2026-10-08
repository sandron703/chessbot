"""
Stage 1 -- generate training games by Stockfish self-play.

The one thing that makes or breaks this stage is DIVERSITY. Stockfish at a
fixed depth is deterministic, so naive self-play produces the same game over
and over and the dataset collapses. Two knobs fight that:

  * `--opening-plies N` with `--opening-multipv K`: for the first N plies the
    move is sampled from Stockfish's top-K at a softmax temperature, so games
    fan out through sensible-but-varied openings instead of random junk.
  * `--play-multipv K` (optional): keeps a little sampling noise in the
    middlegame too.

Games are written as one JSONL shard per worker so the stage is resumable and
embarrassingly parallel.

    python -m chessgpt.selfplay --games 20000 --workers 16 --depth 6 \
        --out data/selfplay

"""

import argparse
import math
import os
import random
import sys
import time
from multiprocessing import Process, Queue

import chess
import chess.engine

from .games import GameWriter
from .stockfish import find_stockfish

MATE = 10000


def _limit(args):
    if args.nodes:
        return chess.engine.Limit(nodes=args.nodes)
    if args.movetime:
        return chess.engine.Limit(time=args.movetime / 1000.0)
    return chess.engine.Limit(depth=args.depth)


def _softmax_choice(scores, temp, rng):
    """Sample an index from centipawn scores at the given temperature."""
    if temp <= 0:
        return max(range(len(scores)), key=lambda i: scores[i])
    m = max(scores)
    w = [math.exp((s - m) / temp) for s in scores]
    total = sum(w)
    r = rng.random() * total
    acc = 0.0
    for i, wi in enumerate(w):
        acc += wi
        if r <= acc:
            return i
    return len(scores) - 1


def _pov_cp(info, turn):
    sc = info['score'].pov(turn)
    return sc.score(mate_score=MATE)


def play_game(engine, args, rng):
    """Play one self-play game. Returns a canonical game record."""
    board = chess.Board()
    ucis = []
    opening_limit = chess.engine.Limit(depth=args.opening_depth)
    main_limit = _limit(args)

    resign_run = 0
    quiet_run = 0
    term = 'natural'
    last_cp = 0

    while not board.is_game_over(claim_draw=True):
        if len(ucis) >= args.max_plies:
            term = 'max-plies'
            break

        in_opening = len(ucis) < args.opening_plies
        multipv = args.opening_multipv if in_opening else args.play_multipv
        temp = args.opening_temp if in_opening else args.play_temp
        limit = opening_limit if in_opening else main_limit

        n_legal = board.legal_moves.count()
        k = max(1, min(multipv, n_legal))
        infos = engine.analyse(board, limit, multipv=k)
        if isinstance(infos, dict):
            infos = [infos]
        infos = [i for i in infos if i.get('pv')]
        if not infos:
            term = 'engine-no-pv'
            break

        if len(infos) == 1 or temp <= 0:
            pick = 0
        else:
            pick = _softmax_choice([_pov_cp(i, board.turn) for i in infos], temp, rng)
        move = infos[pick]['pv'][0]
        last_cp = _pov_cp(infos[0], chess.WHITE)

        ucis.append(move.uci())
        board.push(move)

        # --- adjudication: stop games whose outcome is already decided -----
        if abs(last_cp) >= args.resign_cp:
            resign_run += 1
            if resign_run >= args.resign_plies:
                term = 'adjudicated-win'
                break
        else:
            resign_run = 0

        if len(ucis) >= args.draw_after and abs(last_cp) <= args.draw_cp:
            quiet_run += 1
            if quiet_run >= args.draw_plies:
                term = 'adjudicated-draw'
                break
        else:
            quiet_run = 0

    if term == 'natural':
        result = board.result(claim_draw=True)
    elif term == 'adjudicated-win':
        result = '1-0' if last_cp > 0 else '0-1'
    else:
        result = '1/2-1/2'

    return {
        'moves': ' '.join(ucis),
        'result': result,
        'plies': len(ucis),
        'term': term,
        'final_cp': last_cp,
    }


def worker(wid, n_games, args, progress):
    rng = random.Random(args.seed * 1000003 + wid)
    shard = os.path.join(args.out, f'shard-{wid:03d}.jsonl.zst')
    if os.path.exists(shard) and not args.overwrite:
        progress.put(('skip', wid, n_games))
        return

    engine_path = find_stockfish(args.engine)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        opts = {'Threads': 1, 'Hash': args.hash_mb}
        if args.skill is not None:
            opts['Skill Level'] = args.skill
        for k, v in opts.items():
            try:
                engine.configure({k: v})
            except Exception:
                pass

        tmp = shard + '.part'
        with GameWriter(tmp, compress=True) as w:
            for _ in range(n_games):
                rec = play_game(engine, args, rng)
                rec['worker'] = wid
                w.write(rec)
                progress.put(('game', wid, rec['plies']))
        os.replace(tmp, shard)
    finally:
        engine.close()
    progress.put(('done', wid, n_games))


def main(argv=None):
    p = argparse.ArgumentParser(description='Stockfish self-play game generator')
    p.add_argument('--out', default='data/selfplay', help='output shard directory')
    p.add_argument('--games', type=int, default=1000, help='total games to generate')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--engine', default=None, help='path to the stockfish binary')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--overwrite', action='store_true', help='regenerate existing shards')
    p.add_argument('--hash-mb', type=int, default=16)
    p.add_argument('--skill', type=int, default=None,
                   help="Stockfish 'Skill Level' 0-20; lower makes weaker, more human-ish data")

    g = p.add_argument_group('search limit (pick one)')
    g.add_argument('--depth', type=int, default=6)
    g.add_argument('--nodes', type=int, default=None)
    g.add_argument('--movetime', type=float, default=None, help='milliseconds per move')

    g = p.add_argument_group('diversity')
    g.add_argument('--opening-plies', type=int, default=8)
    g.add_argument('--opening-depth', type=int, default=6)
    g.add_argument('--opening-multipv', type=int, default=4)
    g.add_argument('--opening-temp', type=float, default=60.0,
                   help='centipawn softmax temperature in the opening')
    g.add_argument('--play-multipv', type=int, default=1)
    g.add_argument('--play-temp', type=float, default=20.0)

    g = p.add_argument_group('adjudication')
    g.add_argument('--max-plies', type=int, default=300)
    g.add_argument('--resign-cp', type=int, default=900)
    g.add_argument('--resign-plies', type=int, default=6)
    g.add_argument('--draw-cp', type=int, default=10)
    g.add_argument('--draw-plies', type=int, default=24)
    g.add_argument('--draw-after', type=int, default=60)

    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    print(f'stockfish: {find_stockfish(args.engine)}')

    per = [args.games // args.workers] * args.workers
    for i in range(args.games % args.workers):
        per[i] += 1

    progress = Queue()
    procs = [Process(target=worker, args=(w, per[w], args, progress))
             for w in range(args.workers) if per[w] > 0]
    for pr in procs:
        pr.start()

    done = plies = skipped = 0
    target = sum(per)
    t0 = time.time()
    finished = 0
    while finished < len(procs):
        kind, wid, val = progress.get()
        if kind == 'game':
            done += 1
            plies += val
            if done % 25 == 0 or done == target:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (target - done) / rate if rate else 0
                sys.stderr.write(
                    f'\r{done}/{target} games  {plies/max(done,1):.1f} plies/game  '
                    f'{rate:.1f} games/s  eta {eta/60:.1f} min   ')
                sys.stderr.flush()
        elif kind == 'skip':
            skipped += val
            finished += 1
        elif kind == 'done':
            finished += 1
    for pr in procs:
        pr.join()
    sys.stderr.write('\n')

    print(f'wrote {done} games to {args.out} '
          f'({plies} plies, {plies/max(done,1):.1f} per game) in {time.time()-t0:.0f}s')
    if skipped:
        print(f'skipped {skipped} games in already-complete shards (--overwrite to redo)')


if __name__ == '__main__':
    main()
