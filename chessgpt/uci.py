"""
Stage 5 -- a UCI engine over stdin/stdout.

    python -m chessgpt.uci --model out/chess-small.gptc

That makes the model a first-class chess engine: any GUI can load it, and
cutechess-cli can play it a few hundred games against Stockfish at a fixed
skill level, which is how evaluate.py turns it into an Elo number.

Supported: uci, isready, setoption, ucinewgame, position (startpos|fen, moves),
go (movetime/depth/infinite are accepted and ignored -- one forward pass per
move is the whole search), stop, quit. Non-standard additions: `board` and
`top` print the position and the ranked move list, which make manual play in a
terminal bearable.
"""

import argparse
import sys

import chess

from .backends import load_backend_or_exit
from .engine import ChessGPT

NAME = 'nanoGPT-chess'
AUTHOR = 'prototype pipeline'


def _out(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


def run(model_path, temperature=0.0, top_k=0, condition='win', device=None):
    backend = load_backend_or_exit(model_path, device=device)
    player = ChessGPT(backend, temperature=temperature, top_k=top_k,
                      condition=condition)

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue
        parts = cmd.split()
        head = parts[0]

        if head == 'uci':
            _out(f'id name {NAME}')
            _out(f'id author {AUTHOR}')
            _out('option name Temperature type spin default '
                 f'{int(temperature*100)} min 0 max 500')
            _out(f'option name TopK type spin default {top_k} min 0 max 64')
            _out('option name Conditioning type combo default '
                 f'{condition} var win var draw var none')
            _out('uciok')

        elif head == 'isready':
            _out('readyok')

        elif head == 'setoption':
            try:
                i = parts.index('name')
                j = parts.index('value')
                name = ' '.join(parts[i + 1:j]).lower()
                value = ' '.join(parts[j + 1:])
            except ValueError:
                continue
            if name == 'temperature':
                player.temperature = float(value) / 100.0
            elif name == 'topk':
                player.top_k = int(value)
            elif name == 'conditioning':
                player.condition = value.lower()

        elif head == 'ucinewgame':
            player.new_game()

        elif head == 'position':
            if 'moves' in parts:
                mi = parts.index('moves')
                spec, moves = parts[1:mi], parts[mi + 1:]
            else:
                spec, moves = parts[1:], []
            if spec and spec[0] == 'fen':
                player.set_position(' '.join(spec[1:]), moves)
            else:
                player.set_position('startpos', moves)

        elif head == 'go':
            move = player.select_move()
            if move is None:
                _out('bestmove 0000')
                continue
            for mv, p in player.ranked(5):
                _out(f'info string {mv.uci()} p={p:.3f}')
            _out(f'info depth 1 score cp 0 pv {move.uci()}')
            _out(f'bestmove {move.uci()}')

        elif head == 'stop':
            pass

        elif head == 'board':          # convenience, not UCI
            _out(str(player.board))
            _out(player.board.fen())

        elif head == 'top':            # convenience, not UCI
            for mv, p in player.ranked(10):
                _out(f'{mv.uci()} {p:.4f}')

        elif head in ('quit', 'exit'):
            return

        elif head == 'ponderhit':
            pass


def main(argv=None):
    p = argparse.ArgumentParser(description='UCI engine wrapping a chess nanoGPT')
    p.add_argument('--model', required=True, help='.gptc file or nanoGPT ckpt.pt')
    p.add_argument('--temperature', type=float, default=0.0,
                   help='0 = always the model-argmax legal move')
    p.add_argument('--top-k', type=int, default=0)
    p.add_argument('--condition', default='win', choices=['win', 'draw', 'none'])
    p.add_argument('--device', default=None)
    args = p.parse_args(argv)
    run(args.model, args.temperature, args.top_k, args.condition, args.device)


if __name__ == '__main__':
    main()
