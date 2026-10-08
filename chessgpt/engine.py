"""
The player: a backend + legality-constrained decoding + the bookkeeping that
keeps a token prefix in sync with a chess.Board.

Result conditioning is worth a note. Games are tokenized as
`<bos> <result> moves...`, so at inference we prime the model with the result
we want -- "the side I am playing wins". That costs nothing and measurably
shifts the model toward the sharper moves it saw in won games.
"""

import chess
import numpy as np

from . import masking
from . import tokenizer as T


class ChessGPT:
    def __init__(self, backend, temperature=0.0, top_k=0, condition='win', seed=None):
        self.backend = backend
        self.temperature = temperature
        self.top_k = top_k
        self.condition = condition            # 'win' | 'draw' | 'none'
        self.rng = np.random.default_rng(seed)
        self.board = chess.Board()
        self.history = []                     # UCI moves since the root
        self.root_is_startpos = True

    # --- position bookkeeping --------------------------------------------
    def new_game(self):
        self.board = chess.Board()
        self.history = []
        self.root_is_startpos = True

    def set_position(self, fen=None, moves=()):
        self.board = chess.Board() if fen in (None, 'startpos') else chess.Board(fen)
        self.root_is_startpos = fen in (None, 'startpos')
        self.history = []
        for uci in moves:
            self.push(uci)

    def push(self, uci):
        move = chess.Move.from_uci(uci) if isinstance(uci, str) else uci
        self.board.push(move)
        self.history.append(move.uci())

    def condition_token(self, side=None):
        if self.condition == 'none':
            return None
        if self.condition == 'draw':
            return T.DRAW
        side = self.board.turn if side is None else side
        return T.WIN_W if side == chess.WHITE else T.WIN_B

    def prefix_ids(self, side=None):
        """Token prefix for the current position.

        A position given as a bare FEN has no move history to feed the model;
        we still emit <bos> so the sequence is well formed, and the legality
        mask does the rest. Strength suffers -- the model reasons from move
        history -- which is why the UCI driver keeps the full game when it can.
        """
        ids = [T.BOS]
        tok = self.condition_token(side)
        if tok is not None:
            ids.append(tok)
        for uci in self.history:
            ids.extend(T.encode_move(chess.Move.from_uci(uci)))
        return ids

    # --- move selection ---------------------------------------------------
    def scored_moves(self, side=None):
        return masking.score_legal_moves(self.backend, self.prefix_ids(side), self.board)

    def select_move(self, side=None):
        scored = self.scored_moves(side)
        if not scored:
            return None
        return masking.pick(scored, temperature=self.temperature,
                            top_k=self.top_k, rng=self.rng)

    def ranked(self, n=5, side=None):
        """Top-n (move, probability) pairs, normalized over the legal set."""
        return masking.move_probabilities(self.scored_moves(side))[:n]

    def play_move(self, side=None):
        move = self.select_move(side)
        if move is not None:
            self.push(move)
        return move
