"""
Play the model in a terminal. Not part of the pipeline proper -- but being able
to sit down and play the thing is how you notice the failure modes that no
metric names ("it develops fine then throws a piece away on move 12").

    python -m chessgpt.play --model out/chess-small.gptc            # you are white
    python -m chessgpt.play --model out/chess-small.gptc --black
    python -m chessgpt.play --model out/chess-small.gptc --self     # model vs model

Enter moves as UCI (e2e4) or SAN (Nf3). Commands: top, undo, fen, quit.
"""

import argparse
import sys

import chess

from .backends import load_backend_or_exit
from .engine import ChessGPT


def show(board, player):
    print()
    print(board.unicode(invert_color=True, borders=False)
          if sys.stdout.encoding and 'UTF' in sys.stdout.encoding.upper() else board)
    print(f'  {"white" if board.turn else "black"} to move   '
          f'ply {len(player.history)}')


def main(argv=None):
    p = argparse.ArgumentParser(description='play a chess nanoGPT in the terminal')
    p.add_argument('--model', required=True)
    p.add_argument('--black', action='store_true', help='you play black')
    p.add_argument('--self', dest='selfplay', action='store_true')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-k', type=int, default=0)
    p.add_argument('--condition', default='win', choices=['win', 'draw', 'none'])
    p.add_argument('--device', default=None)
    args = p.parse_args(argv)

    player = ChessGPT(load_backend_or_exit(args.model, device=args.device),
                      temperature=args.temperature, top_k=args.top_k,
                      condition=args.condition)
    player.new_game()
    human = None if args.selfplay else (chess.BLACK if args.black else chess.WHITE)

    while not player.board.is_game_over(claim_draw=True):
        board = player.board
        show(board, player)

        if human is not None and board.turn == human:
            try:
                raw = input('your move > ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if raw in ('quit', 'exit'):
                break
            if raw == 'fen':
                print(board.fen())
                continue
            if raw == 'top':
                for mv, pr in player.ranked(8):
                    print(f'  {mv.uci():6} {pr:.3f}')
                continue
            if raw == 'undo':
                for _ in range(2):
                    if player.history:
                        player.board.pop()
                        player.history.pop()
                continue
            move = None
            for parse in (board.parse_uci, board.parse_san):
                try:
                    move = parse(raw)
                    break
                except Exception:
                    continue
            if move is None or move not in board.legal_moves:
                print('  not a legal move (UCI like e2e4, or SAN like Nf3)')
                continue
            player.push(move)
        else:
            ranked = player.ranked(3)
            mv = player.select_move()
            if mv is None:
                break
            san = board.san(mv)
            player.push(mv)
            top = '  '.join(f'{m.uci()}:{pr:.2f}' for m, pr in ranked)
            print(f'model plays {san} ({mv.uci()})   [{top}]')

    print()
    print(player.board)
    print(f'result: {player.board.result(claim_draw=True)}')
    print('moves:', ' '.join(player.history))


if __name__ == '__main__':
    main()
