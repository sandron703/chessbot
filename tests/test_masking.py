"""The masking layer is the prototype's safety net: whatever the network
believes, the move that comes out must be legal chess. These tests use a
RANDOM, untrained model on purpose -- if masking only works for a good model
it isn't masking."""

import random
import sys

import chess
import numpy as np

from _util import run_tests, tiny_checkpoint
from chessgpt import masking
from chessgpt import tokenizer as T
from chessgpt.backends import TorchBackend
from chessgpt.engine import ChessGPT

_BACKEND = None


def backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = TorchBackend(tiny_checkpoint(), device='cpu')
    return _BACKEND


def random_positions(n, seed=3, max_ply=40):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        hist = []
        for _ in range(rng.randrange(1, max_ply)):
            moves = list(board.legal_moves)
            if not moves:
                break
            mv = moves[rng.randrange(len(moves))]
            hist.append(mv.uci())
            board.push(mv)
        if not board.is_game_over():
            out.append((board, hist))
    return out


def test_masks_agree_with_python_chess():
    for board, _ in random_positions(25):
        froms = masking.from_square_ids(board)
        assert froms == sorted({m.from_square for m in board.legal_moves})
        for f in froms:
            tos = masking.to_square_ids(board, f)
            assert tos
            for t in tos:
                assert any(m.from_square == f and m.to_square == t
                           for m in board.legal_moves)


def test_mask_logits_blocks_everything_else():
    logits = np.arange(T.VOCAB_SIZE, dtype=np.float32)
    masked = masking.mask_logits(logits, [3, 7])
    assert np.isneginf(masked[0]) and np.isneginf(masked[79])
    assert masked[3] == 3 and masked[7] == 7
    assert int(np.argmax(masked)) == 7


def test_scoring_covers_every_legal_move_exactly_once():
    be = backend()
    for board, hist in random_positions(12, seed=11):
        player = ChessGPT(be)
        player.set_position('startpos', hist)
        scored = player.scored_moves()
        assert len(scored) == board.legal_moves.count()
        assert {m.uci() for m, _ in scored} == {m.uci() for m in board.legal_moves}
        s = [v for _, v in scored]
        assert s == sorted(s, reverse=True), 'scored moves must be best-first'


def test_probabilities_normalize():
    be = backend()
    player = ChessGPT(be)
    player.set_position('startpos', ['e2e4', 'e7e5'])
    probs = masking.move_probabilities(player.scored_moves())
    assert abs(sum(p for _, p in probs) - 1.0) < 1e-9
    assert all(p >= 0 for _, p in probs)


def test_selected_move_is_always_legal():
    be = backend()
    for board, hist in random_positions(20, seed=5):
        for temp in (0.0, 1.0):
            player = ChessGPT(be, temperature=temp, seed=1)
            player.set_position('startpos', hist)
            mv = player.select_move()
            assert mv is not None
            assert mv in player.board.legal_moves, mv


def test_full_random_game_stays_legal():
    """Play a whole game model-vs-model. Any illegality or crash shows up here,
    including the block_size crop path on a long game."""
    be = backend()
    player = ChessGPT(be, temperature=1.0, seed=2)
    player.new_game()
    while not player.board.is_game_over(claim_draw=True) and len(player.history) < 160:
        mv = player.select_move()
        assert mv in player.board.legal_moves
        player.push(mv)
    assert len(player.history) > 10
    assert T.decode_moves(player.prefix_ids()) == player.history


def test_promotion_slot_is_masked():
    be = backend()
    player = ChessGPT(be)
    player.set_position('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
    scored = player.scored_moves()
    promos = {m.uci() for m, _ in scored if m.promotion}
    assert promos == {'a7a8q', 'a7a8r', 'a7a8b', 'a7a8n'}
    assert masking.promotion_ids(player.board, chess.A7, chess.A8) == sorted(T.PROMO_IDS)
    assert masking.promotion_ids(player.board, chess.E1, chess.E2) == []


if __name__ == '__main__':
    sys.exit(run_tests(dict(globals())))
