"""
Stage 2 -- tokenize games into the train.bin / val.bin / meta.pkl trio that
nanoGPT's train.py reads, with no modification to nanoGPT itself.

The output stream is games concatenated back to back:

    <bos> <win-w> e2 e4 e7 e5 ... <eos> <bos> <draw> d2 d4 ... <eos> ...

nanoGPT samples random fixed-length windows out of that stream, so the model
sees both full games (from <bos>) and mid-game windows. That is deliberate: at
inference a game can outgrow block_size and the context has to be cropped, and
training on unaligned windows is exactly what makes the cropped case work.

The train/val split is by GAME, not by token, so no game straddles the split.

    python -m chessgpt.prepare --src data/selfplay --dataset chess_uci
"""

import argparse
import hashlib
import os
import pickle
import sys
from collections import Counter

import numpy as np

from . import tokenizer as T
from .games import GameWriter, iter_games

DEFAULT_DATASET_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'nanoGPT', 'data')


def encode_records(records, min_plies, max_plies, dedupe, results,
                   min_elo=0, max_elo=0, terminations=None):
    """Encode game records -> [(uint16 ids, record)], plus filtering stats.

    The Elo and termination filters matter for human PGN archives. A time
    forfeit is labelled 1-0/0-1 even when the position was fine, which is noise
    for the <result> conditioning token -- `--termination Normal` drops those.
    """
    seen = set()
    stats = Counter()
    out = []
    for rec in records:
        stats['seen'] += 1
        moves = rec['moves'].split()
        if len(moves) < min_plies:
            stats['drop_short'] += 1
            continue
        if min_elo or max_elo:
            elos = [rec.get('white_elo'), rec.get('black_elo')]
            if any(e is None for e in elos):
                stats['drop_no_elo'] += 1
                continue
            if min_elo and min(elos) < min_elo:
                stats['drop_elo_low'] += 1
                continue
            if max_elo and max(elos) > max_elo:
                stats['drop_elo_high'] += 1
                continue
        if terminations and (rec.get('termination') or '') not in terminations:
            stats['drop_termination'] += 1
            continue
        if max_plies and len(moves) > max_plies:
            moves = moves[:max_plies]
            stats['truncated'] += 1
        result = rec.get('result', '*')
        if result not in T.RESULT_ID:
            stats['drop_no_result'] += 1
            continue
        if results and result not in results:
            stats['drop_result_filter'] += 1
            continue
        if dedupe:
            h = hashlib.blake2b(rec['moves'].encode(), digest_size=16).digest()
            if h in seen:
                stats['drop_dupe'] += 1
                continue
            seen.add(h)
        try:
            ids = T.encode_game(moves, result)
        except Exception:
            stats['drop_encode_error'] += 1
            continue
        rec = dict(rec, moves=' '.join(moves), plies=len(moves))
        out.append((np.asarray(ids, dtype=np.uint16), rec))
        stats['kept'] += 1
        stats['plies'] += len(moves)
        stats['tokens'] += len(ids)
        stats['result_' + result] += 1
    return out, stats


def main(argv=None):
    p = argparse.ArgumentParser(description='tokenize games -> nanoGPT .bin files')
    p.add_argument('--src', nargs='+', default=['data/selfplay'],
                   help='game files or directories (.jsonl[.zst] or .pgn[.zst])')
    p.add_argument('--dataset', default='chess_uci',
                   help='dataset name; written to <nanoGPT>/data/<dataset>/')
    p.add_argument('--dataset-root', default=DEFAULT_DATASET_ROOT)
    p.add_argument('--val-games', type=int, default=2000,
                   help='games held out for validation (capped at 10%% of the set)')
    p.add_argument('--min-plies', type=int, default=10)
    p.add_argument('--max-plies', type=int, default=0, help='0 = no cap')
    p.add_argument('--max-games', type=int, default=0, help='0 = all')
    p.add_argument('--results', nargs='*', default=None,
                   help="keep only these results, e.g. --results 1-0 0-1")
    p.add_argument('--min-elo', type=int, default=0,
                   help='require BOTH players at or above this rating')
    p.add_argument('--max-elo', type=int, default=0,
                   help='require BOTH players at or below this rating')
    p.add_argument('--termination', nargs='*', default=None,
                   help="keep only these PGN Termination values, e.g. "
                        "--termination Normal (drops time forfeits, whose "
                        "result label does not reflect the position)")
    p.add_argument('--no-dedupe', action='store_true')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--block-size', type=int, default=384,
                   help='reported only, to show what fraction of games fits in context')
    args = p.parse_args(argv)

    out_dir = os.path.join(args.dataset_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    records = iter_games(args.src, min_plies=args.min_plies)
    if args.max_games:
        import itertools
        records = itertools.islice(records, args.max_games)

    games, stats = encode_records(
        records, args.min_plies, args.max_plies,
        dedupe=not args.no_dedupe,
        results=set(args.results) if args.results else None,
        min_elo=args.min_elo, max_elo=args.max_elo,
        terminations=set(args.termination) if args.termination else None)

    if not games:
        sys.exit('no games survived filtering -- check --src')

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(games))
    games = [games[i] for i in order]

    n_val = min(args.val_games, max(1, len(games) // 10))
    val, train = games[:n_val], games[n_val:]

    # The held-out game records go out alongside the bins: every position-based
    # metric in evaluate.py needs real boards, not token ids, and this keeps the
    # split reproducible without re-deriving it from a seed.
    val_games_path = os.path.join(out_dir, 'val_games.jsonl.zst')
    with GameWriter(val_games_path) as w:
        for _, rec in val:
            w.write(rec)
    print(f'val games: {len(val):,} records -> {val_games_path}')

    for name, split in (('train', [g for g, _ in train]), ('val', [g for g, _ in val])):
        path = os.path.join(out_dir, f'{name}.bin')
        total = sum(len(g) for g in split)
        arr = np.memmap(path, dtype=np.uint16, mode='w+', shape=(total,))
        i = 0
        for g in split:
            arr[i:i + len(g)] = g
            i += len(g)
        arr.flush()
        del arr
        print(f'{name}: {len(split):,} games, {total:,} tokens -> {path}')

    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(T.meta(), f)

    lens = np.array([len(g) for g, _ in games])
    print()
    print(f'vocab_size          : {T.VOCAB_SIZE} ({T.N_REAL_TOKENS} in use)')
    print(f'games kept          : {stats["kept"]:,} of {stats["seen"]:,} seen')
    for k in ('drop_short', 'drop_dupe', 'drop_no_result', 'drop_result_filter',
              'drop_no_elo', 'drop_elo_low', 'drop_elo_high', 'drop_termination',
              'drop_encode_error', 'truncated'):
        if stats[k]:
            print(f'  {k:<18}: {stats[k]:,}')
    print(f'results             : ' + '  '.join(
        f'{r}={stats["result_" + r]:,}' for r in ('1-0', '0-1', '1/2-1/2')))
    print(f'plies/game          : {stats["plies"]/stats["kept"]:.1f}')
    print(f'tokens/game         : {lens.mean():.1f} (median {np.median(lens):.0f}, '
          f'p90 {np.percentile(lens, 90):.0f}, p99 {np.percentile(lens, 99):.0f}, '
          f'max {lens.max()})')
    print(f'total tokens        : {stats["tokens"]:,}')
    print()
    print('block_size coverage (pick the smallest that covers ~p99 -- attention')
    print('cost is quadratic in block_size, and padding context is wasted compute):')
    for bs in (128, 192, 256, 320, 384, 512):
        pct = float((lens <= bs).mean()) * 100
        mark = '  <-- --block-size' if bs == args.block_size else ''
        print(f'  {bs:>4}: {pct:5.1f}%{mark}')
    print()
    print('next: cd ../nanoGPT && python train.py '
          f'"../prototype./configs/train_chess_small.py"')


if __name__ == '__main__':
    main()
