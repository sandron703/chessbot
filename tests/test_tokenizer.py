"""The tokenizer must be lossless on real games, including promotions,
castling and en passant -- a silent encoding bug here would poison every
downstream stage and look like 'the model just isn't learning'."""

import os
import random
import sys

import chess

from _util import run_tests
from chessgpt import tokenizer as T


def test_vocab_is_consistent():
    assert T.N_REAL_TOKENS <= T.VOCAB_SIZE
    assert len(T.ITOS) == T.VOCAB_SIZE
    assert len(T.STOI) == T.VOCAB_SIZE
    assert sorted(T.ITOS) == list(range(T.VOCAB_SIZE))
    for i in range(64):
        assert T.ITOS[i] == chess.square_name(i)
    assert len(set(T.SQUARE_IDS) | {T.PAD, T.BOS, T.EOS}
               | set(T.PROMO_IDS) | set(T.RESULT_IDS)) == T.N_REAL_TOKENS


def test_roundtrip_random_games():
    rng = random.Random(7)
    for _ in range(40):
        board = chess.Board()
        ucis = []
        while not board.is_game_over() and len(ucis) < 120:
            moves = list(board.legal_moves)
            mv = moves[rng.randrange(len(moves))]
            ucis.append(mv.uci())
            board.push(mv)
        ids = T.encode_game(ucis, '1/2-1/2')
        assert T.decode_moves(ids) == ucis
        assert ids[0] == T.BOS and ids[-1] == T.EOS


def test_roundtrip_promotions_and_castling():
    # white can promote on either wing and castle on either side
    board = chess.Board('4k3/P6P/8/8/8/8/8/R3K2R w KQ - 0 1')
    cases = ['a7a8q', 'a7a8n', 'h7h8r', 'h7h8b', 'e1g1', 'e1c1']
    for uci in cases:
        mv = chess.Move.from_uci(uci)
        assert mv in board.legal_moves, uci
        ids = T.encode_move(mv)
        assert len(ids) == (3 if mv.promotion else 2)
        assert T.decode_moves(ids) == [uci]


def test_roundtrip_en_passant():
    board = chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')
    mv = chess.Move.from_uci('e5d6')
    assert mv in board.legal_moves
    assert T.decode_moves(T.encode_move(mv)) == ['e5d6']


def test_next_slot_tracks_parity():
    ids = [T.BOS, T.WIN_W]
    assert T.next_slot(ids) == T.SLOT_FROM
    ids.append(12)
    assert T.next_slot(ids) == T.SLOT_TO
    ids.append(28)
    assert T.next_slot(ids) == T.SLOT_FROM
    ids.append(T.PROMO_ID[chess.QUEEN])          # promo must not flip parity
    assert T.next_slot(ids) == T.SLOT_FROM


def test_token_count_formula():
    ucis = ['e2e4', 'e7e5', 'g1f3']
    assert len(T.encode_game(ucis, '1-0')) == T.tokens_per_game(3)
    assert len(T.encode_game(['a7a8q'], '1-0')) == T.tokens_per_game(1, 1)


def test_crop_starts_on_a_from_slot():
    ids = T.encode_game(['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5'], '1-0', eos=False)
    for bs in range(6, len(ids) + 4):
        cropped = T.crop_prefix(ids, bs)
        assert len(cropped) + T.MAX_MOVE_TOKENS <= max(bs, len(cropped) + 3)
        if len(cropped) < len(ids):
            assert T.next_slot(ids[:len(ids) - len(cropped)]) == T.SLOT_FROM


def test_decode_rejects_malformed_streams():
    for bad in ([12], [T.PROMO_ID[chess.QUEEN]], [12, 28, 12]):
        try:
            T.decode_moves(bad)
        except ValueError:
            continue
        raise AssertionError(f'should have rejected {bad}')


if __name__ == '__main__':
    sys.exit(run_tests(dict(globals())))
