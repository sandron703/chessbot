"""
Stage 4 -- the evaluation suite. Loss alone says almost nothing about whether a
chess model is any good, so this measures five different things:

  loss       cross-entropy on held-out games, broken down by SLOT. The from-slot
             and to-slot losses behave very differently and the split is the
             fastest way to see what the model is actually learning.
  legal      how often the model's own UNMASKED choice is a legal move. This is
             the honest measure of how much chess the network internalized, as
             opposed to how much work the legality mask is doing for it.
  agreement  top-1 agreement with Stockfish, and mean centipawn loss of the
             move the model picks. Playing strength, position by position.
  match      actual games against Stockfish at a fixed skill level -> an Elo
             estimate with an error bar.
  bench      ms/move, forward passes per move, peak RSS. Run this ON the UNO Q;
             it is the number that decides whether the model ships.

    python -m chessgpt.evaluate all --model out/chess-small.gptc
"""

import argparse
import math
import os
import random
import sys
import time

import chess
import numpy as np

from . import masking
from . import tokenizer as T
from .backends import load_backend
from .engine import ChessGPT
from .games import iter_games
from .prepare import DEFAULT_DATASET_ROOT
from .stockfish import find_stockfish


def dataset_dir(dataset, root=None):
    return os.path.join(root or DEFAULT_DATASET_ROOT, dataset)


def load_val_games(dataset, root=None, limit=0):
    path = os.path.join(dataset_dir(dataset, root), 'val_games.jsonl.zst')
    if not os.path.exists(path):
        sys.exit(f'{path} not found -- run chessgpt.prepare first')
    games = []
    for rec in iter_games(path):
        games.append(rec)
        if limit and len(games) >= limit:
            break
    return games


def sample_positions(games, n, rng, min_ply=4, max_ply=200):
    """Random (board, history) pairs drawn from held-out games."""
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        rec = games[rng.randrange(len(games))]
        moves = rec['moves'].split()
        hi = min(len(moves) - 1, max_ply)
        if hi <= min_ply:
            continue
        ply = rng.randrange(min_ply, hi)
        board = chess.Board()
        ok = True
        for uci in moves[:ply]:
            try:
                board.push(chess.Move.from_uci(uci))
            except Exception:
                ok = False
                break
        if not ok or board.is_game_over():
            continue
        out.append((board, moves[:ply], moves[ply]))
    return out


def elo_from_score(score, n):
    """Elo difference implied by a match score, with a 1-sigma band."""
    if n == 0:
        return float('nan'), float('nan')
    eps = 1.0 / (2 * n + 2)
    s = min(max(score, eps), 1 - eps)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    se = math.sqrt(max(s * (1 - s), eps) / n)
    d = 400.0 / (math.log(10) * s * (1 - s)) * se
    return elo, d


# --- loss -------------------------------------------------------------------

SLOT_NAMES = {0: 'from', 1: 'to', 2: 'promo', 3: 'eos', 4: 'result'}


def slot_labels(ids):
    """Per-position label for what each token IS (used to bucket its loss)."""
    labels = []
    pending = False
    for i in ids:
        i = int(i)
        if i == T.EOS:
            labels.append(3)
        elif i in T.RESULT_IDS:
            labels.append(4)
        elif i == T.BOS:
            labels.append(-1)
        elif i in T.ID_PROMO:
            labels.append(2)
        else:
            labels.append(1 if pending else 0)
            pending = not pending
    return labels


def cmd_loss(args):
    import torch
    import torch.nn.functional as F
    from .backends import TorchBackend

    if not args.ckpt:
        sys.exit('loss needs --ckpt (a nanoGPT checkpoint); it reads full logits')
    be = TorchBackend(args.ckpt, device=args.device)
    model, device, bs = be.model, be.device, be.block_size
    games = load_val_games(args.dataset, args.dataset_root, limit=args.games)

    tot = np.zeros(5)
    cnt = np.zeros(5)
    total_loss = 0.0
    total_n = 0
    for rec in games:
        ids = T.encode_game(rec['moves'].split(), rec['result'])[:bs + 1]
        if len(ids) < 8:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x, y)
        per = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                              reduction='none').float().cpu().numpy()
        labs = np.array(slot_labels(ids[1:]))
        total_loss += per.sum()
        total_n += len(per)
        for s in range(5):
            m = labs == s
            if m.any():
                tot[s] += per[m].sum()
                cnt[s] += m.sum()

    print(f'checkpoint      : {args.ckpt}')
    print(f'iter / val_loss : {be.iter_num} / {be.best_val_loss:.4f} (from ckpt)')
    print(f'params          : {be.n_params/1e6:.2f}M')
    print(f'held-out games  : {len(games):,}')
    print(f'loss (all)      : {total_loss/total_n:.4f}   ppl {math.exp(total_loss/total_n):.2f}')
    for s in range(5):
        if cnt[s]:
            l = tot[s] / cnt[s]
            print(f'loss ({SLOT_NAMES[s]:<7}) : {l:.4f}   ppl {math.exp(l):6.2f}   '
                  f'n={int(cnt[s]):,}')
    return {'loss': total_loss / total_n}


# --- legality ---------------------------------------------------------------

def cmd_legal(args):
    be = load_backend(args.model, device=args.device)
    player = ChessGPT(be, temperature=args.temperature, condition=args.condition)
    games = load_val_games(args.dataset, args.dataset_root, limit=args.games or 5000)
    rng = random.Random(args.seed)
    positions = sample_positions(games, args.positions, rng)

    well = legal = agree = n = 0
    for board, hist, _ in positions:
        player.set_position('startpos', hist)
        prefix = player.prefix_ids()
        uci, wf = masking.generate_unconstrained_move(
            be, prefix, temperature=args.temperature,
            rng=np.random.default_rng(rng.randrange(1 << 30)))
        n += 1
        well += int(wf)
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            continue
        is_legal = mv in board.legal_moves
        legal += int(is_legal)
        if is_legal:
            best = player.select_move()
            agree += int(best is not None and best.uci() == uci)

    print(f'positions                    : {n:,}')
    print(f'well-formed (argmax a square): {100*well/n:.1f}%')
    print(f'legal, unmasked              : {100*legal/n:.1f}%   <-- what the net knows')
    print(f'  of those, == masked top-1  : {100*agree/max(legal,1):.1f}%')
    print(f'legal, masked                : 100.0%  (by construction)')
    return {'legal_unmasked': legal / n}


# --- stockfish-based metrics -----------------------------------------------

def _sf(args):
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish(args.stockfish))
    engine.configure({'Threads': 1, 'Hash': 64})
    return engine


def _sf_limit(args):
    import chess.engine
    if args.sf_nodes:
        return chess.engine.Limit(nodes=args.sf_nodes)
    if args.sf_movetime:
        return chess.engine.Limit(time=args.sf_movetime / 1000.0)
    return chess.engine.Limit(depth=args.sf_depth)


def cmd_agreement(args):
    be = load_backend(args.model, device=args.device)
    player = ChessGPT(be, temperature=0.0, condition=args.condition)
    games = load_val_games(args.dataset, args.dataset_root, limit=args.games or 5000)
    rng = random.Random(args.seed)
    positions = sample_positions(games, args.positions, rng)

    engine = _sf(args)
    limit = _sf_limit(args)
    MATE = 10000
    try:
        match_sf = match_data = n = 0
        losses = []
        for board, hist, played in positions:
            player.set_position('startpos', hist)
            mv = player.select_move()
            if mv is None:
                continue
            n += 1
            info = engine.analyse(board, limit)
            sf_best = info['pv'][0]
            cp_best = info['score'].pov(board.turn).score(mate_score=MATE)
            match_sf += int(mv == sf_best)
            match_data += int(mv.uci() == played)

            after = board.copy()
            after.push(mv)
            if after.is_game_over():
                res = after.result()
                cp_ours = 0 if res == '1/2-1/2' else (
                    MATE if (res == '1-0') == (board.turn == chess.WHITE) else -MATE)
            else:
                cp_ours = -engine.analyse(after, limit)['score'].pov(
                    after.turn).score(mate_score=MATE)
            losses.append(max(0, cp_best - cp_ours))
    finally:
        engine.close()

    losses = np.array(losses)
    print(f'positions                : {n:,}')
    print(f'top-1 == stockfish       : {100*match_sf/n:.1f}%')
    print(f'top-1 == move played     : {100*match_data/n:.1f}%')
    print(f'centipawn loss  mean     : {losses.mean():.0f}')
    print(f'                median   : {np.median(losses):.0f}')
    print(f'                p90      : {np.percentile(losses, 90):.0f}')
    print(f'blunders (>300cp)        : {100*(losses>300).mean():.1f}%')
    return {'acpl': float(losses.mean()), 'sf_top1': match_sf / n}


def cmd_match(args):
    be = load_backend(args.model, device=args.device)
    player = ChessGPT(be, temperature=args.temperature, top_k=args.top_k,
                      condition=args.condition, seed=args.seed)
    engine = _sf(args)
    limit = _sf_limit(args)
    try:
        engine.configure({'Skill Level': args.skill})
    except Exception:
        print(f'warning: engine rejected Skill Level {args.skill}', file=sys.stderr)

    w = d = l = 0
    plies = 0
    t0 = time.time()
    try:
        for g in range(args.match_games):
            our_color = chess.WHITE if g % 2 == 0 else chess.BLACK
            player.new_game()
            board = player.board
            while not board.is_game_over(claim_draw=True) and len(player.history) < args.max_plies:
                if board.turn == our_color:
                    mv = player.select_move(side=our_color)
                    if mv is None:
                        break
                    player.push(mv)
                else:
                    mv = engine.play(board, limit).move
                    if mv is None:
                        break
                    player.push(mv)
            plies += len(player.history)
            if board.is_game_over(claim_draw=True):
                res = board.result(claim_draw=True)
            else:
                res = '1/2-1/2'
            if res == '1/2-1/2':
                d += 1
            elif (res == '1-0') == (our_color == chess.WHITE):
                w += 1
            else:
                l += 1
            sys.stderr.write(f'\rgame {g+1}/{args.match_games}  +{w} ={d} -{l}   ')
            sys.stderr.flush()
    finally:
        engine.close()
        sys.stderr.write('\n')

    n = w + d + l
    score = (w + 0.5 * d) / max(n, 1)
    elo, band = elo_from_score(score, n)
    print(f'opponent      : stockfish skill {args.skill}, {_sf_limit(args)}')
    print(f'games         : {n}  (+{w} ={d} -{l})')
    print(f'score         : {100*score:.1f}%')
    print(f'elo vs that   : {elo:+.0f} +/- {band:.0f}')
    print(f'avg game      : {plies/max(n,1):.0f} plies, {(time.time()-t0)/max(n,1):.1f}s')
    return {'score': score, 'elo_diff': elo}


# --- bench ------------------------------------------------------------------

def cmd_bench(args):
    be = load_backend(args.model, device=args.device)
    player = ChessGPT(be, temperature=0.0, condition=args.condition)
    games = load_val_games(args.dataset, args.dataset_root, limit=200) \
        if args.dataset else []
    if games:
        rng = random.Random(args.seed)
        positions = [p for p in sample_positions(games, args.bench_moves, rng,
                                                 min_ply=10, max_ply=120)]
    else:
        positions = []
    if not positions:                      # no dataset handy: just play from scratch
        player.new_game()
        positions = []
        for _ in range(args.bench_moves):
            positions.append((player.board.copy(), list(player.history), None))
            mv = player.select_move()
            if mv is None:
                break
            player.push(mv)

    # warm up (first call pays for BLAS init and the prefix build)
    player.set_position('startpos', positions[0][1])
    player.select_move()

    times = []
    for board, hist, _ in positions:
        player.set_position('startpos', hist)
        t0 = time.perf_counter()
        player.select_move()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    rss = None
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM'):
                    rss = int(line.split()[1]) / 1024
    except Exception:
        pass
    # 'evaluate all' imports torch for the loss metric, which dwarfs the
    # runtime's own footprint -- don't quote a contaminated number.
    torch_loaded = 'torch' in sys.modules

    print(f'model              : {args.model}')
    print(f'backend            : {type(be).__name__}')
    print(f'block_size         : {be.block_size}')
    print(f'moves timed        : {len(times)}')
    print(f'ms/move   mean     : {times.mean():.1f}')
    print(f'          median   : {np.median(times):.1f}')
    print(f'          p95      : {np.percentile(times, 95):.1f}')
    print(f'          max      : {times.max():.1f}')
    if hasattr(be, 'stats'):
        s = be.stats
        print(f'prefill tokens     : {s["prefill_tokens"]:,}')
        print(f'extension forwards : {s["ext_calls"]:,} '
              f'({s["ext_calls"]/max(len(times),1):.1f} per move)')
        print(f'cache rebuilds     : {s["rebuilds"]}')
    if rss and not torch_loaded:
        print(f'peak RSS           : {rss:.0f} MB')
    elif rss:
        print(f'peak RSS           : {rss:.0f} MB  (includes torch -- rerun '
              f'`evaluate bench` alone for the runtime-only figure)')
    return {'ms_per_move': float(times.mean())}


# --- cli --------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description='evaluate a chess nanoGPT')
    p.add_argument('cmd', choices=['loss', 'legal', 'agreement', 'match', 'bench', 'all'])
    p.add_argument('--model', default=None, help='.gptc or ckpt.pt (inference metrics)')
    p.add_argument('--ckpt', default=None, help='nanoGPT ckpt.pt (needed by `loss`)')
    p.add_argument('--dataset', default='chess_uci')
    p.add_argument('--dataset-root', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--games', type=int, default=0, help='held-out games to read')
    p.add_argument('--positions', type=int, default=300)
    p.add_argument('--bench-moves', type=int, default=30)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-k', type=int, default=0)
    p.add_argument('--condition', default='win', choices=['win', 'draw', 'none'])

    g = p.add_argument_group('stockfish opponent')
    g.add_argument('--stockfish', default=None)
    g.add_argument('--sf-depth', type=int, default=8)
    g.add_argument('--sf-nodes', type=int, default=None)
    g.add_argument('--sf-movetime', type=float, default=None)
    g.add_argument('--skill', type=int, default=0, help='Stockfish Skill Level 0-20')
    g.add_argument('--match-games', type=int, default=40)
    g.add_argument('--max-plies', type=int, default=300)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.model is None:
        args.model = args.ckpt
    if args.ckpt is None and args.model and args.model.endswith('.pt'):
        args.ckpt = args.model

    def rule(title):
        print()
        print(f'=== {title} ' + '=' * max(0, 60 - len(title)))

    if args.cmd == 'all':
        if args.ckpt:
            rule('loss')
            cmd_loss(args)
        rule('legality (unmasked)')
        cmd_legal(args)
        rule('bench')
        cmd_bench(args)
        try:
            find_stockfish(args.stockfish)
        except FileNotFoundError as e:
            print(f'\nskipping stockfish metrics: {e}')
            return
        rule('agreement vs stockfish')
        cmd_agreement(args)
        rule('match vs stockfish')
        cmd_match(args)
        return

    {'loss': cmd_loss, 'legal': cmd_legal, 'agreement': cmd_agreement,
     'match': cmd_match, 'bench': cmd_bench}[args.cmd](args)


if __name__ == '__main__':
    main()
