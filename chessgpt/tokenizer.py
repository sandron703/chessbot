"""
Square-pair UCI tokenizer.

A game is a flat sequence of tokens:

    <bos> <result> from to  from to  from to =q  ... <eos>

Every move contributes exactly two tokens (origin square, destination square),
plus a third token when it is a promotion. That gives ~2.05 tokens per ply, a
vocabulary of 74 (padded to 80), and -- the reason this scheme was chosen -- a
decoding process that can be *constrained* to legal chess at every single step:
at a `from` slot only squares holding a movable piece are allowed, at a `to`
slot only destinations legal from the chosen origin. See masking.py.

Square ids follow python-chess: a1=0, b1=1, ... h8=63.
"""

import chess

# --- vocabulary ------------------------------------------------------------

SQUARE_TOKENS = [chess.square_name(s) for s in range(64)]  # 'a1'..'h8'

PAD = 64
BOS = 65
EOS = 66

PROMO_BASE = 67                      # =q =r =b =n
PROMO_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
PROMO_TOKENS = ['=q', '=r', '=b', '=n']
PROMO_ID = {p: PROMO_BASE + i for i, p in enumerate(PROMO_PIECES)}
ID_PROMO = {v: k for k, v in PROMO_ID.items()}

RESULT_BASE = 71                     # <win-w> <win-b> <draw>
WIN_W, WIN_B, DRAW = 71, 72, 73
RESULT_TOKENS = ['<win-w>', '<win-b>', '<draw>']
RESULT_ID = {'1-0': WIN_W, '0-1': WIN_B, '1/2-1/2': DRAW}

N_REAL_TOKENS = 74
# nanoGPT likes a vocab that is a multiple of 64 for tensor-core efficiency;
# the padding ids are never emitted by the encoder.
VOCAB_SIZE = 80

ITOS = {}
ITOS.update({i: t for i, t in enumerate(SQUARE_TOKENS)})
ITOS[PAD] = '<pad>'
ITOS[BOS] = '<bos>'
ITOS[EOS] = '<eos>'
ITOS.update({PROMO_BASE + i: t for i, t in enumerate(PROMO_TOKENS)})
ITOS.update({RESULT_BASE + i: t for i, t in enumerate(RESULT_TOKENS)})
for i in range(N_REAL_TOKENS, VOCAB_SIZE):
    ITOS[i] = f'<unused{i}>'
STOI = {t: i for i, t in ITOS.items()}

SQUARE_IDS = list(range(64))
PROMO_IDS = [PROMO_BASE + i for i in range(4)]
RESULT_IDS = [WIN_W, WIN_B, DRAW]

# --- slot state machine ----------------------------------------------------
# Decoding needs to know which kind of token comes next. Only three states
# matter, and they are a pure function of the tokens emitted so far.
SLOT_FROM = 'from'
SLOT_TO = 'to'
SLOT_PROMO = 'promo'


def result_token(result, side=None):
    """Map a PGN result string to its conditioning token id.

    `side` (chess.WHITE/BLACK) is accepted for the inference-time use of this
    function, where we condition the model on "the side I am playing wins".
    """
    if result in RESULT_ID:
        return RESULT_ID[result]
    if side is chess.WHITE:
        return WIN_W
    if side is chess.BLACK:
        return WIN_B
    return DRAW


def encode_move(move):
    """One move -> 2 or 3 token ids. Does not validate legality."""
    ids = [move.from_square, move.to_square]
    if move.promotion:
        ids.append(PROMO_ID[move.promotion])
    return ids


def encode_game(uci_moves, result=None, bos=True, eos=True):
    """A list of UCI move strings -> token ids.

    `result` is a PGN result string ('1-0', '0-1', '1/2-1/2'); when given it is
    emitted as a conditioning token directly after <bos>.
    """
    ids = []
    if bos:
        ids.append(BOS)
    if result is not None:
        ids.append(RESULT_ID[result])
    for uci in uci_moves:
        ids.extend(encode_move(chess.Move.from_uci(uci)))
    if eos:
        ids.append(EOS)
    return ids


def decode_tokens(ids):
    """Token ids -> list of human-readable token strings."""
    return [ITOS[int(i)] for i in ids]


def decode_moves(ids):
    """Token ids -> list of UCI move strings.

    Control and result tokens are skipped. Raises ValueError on a stream that
    is not a well-formed from/to/promo sequence.
    """
    moves = []
    pending = None
    for i in ids:
        i = int(i)
        if i in (BOS, EOS, PAD) or i in RESULT_IDS:
            continue
        if i in ID_PROMO:
            if not moves or pending is not None:
                raise ValueError('promotion token in a from/to slot')
            moves[-1] += 'qrbn'[PROMO_IDS.index(i)]
            continue
        if i >= 64:
            raise ValueError(f'unexpected token id {i}')
        if pending is None:
            pending = i
        else:
            moves.append(chess.square_name(pending) + chess.square_name(i))
            pending = None
    if pending is not None:
        raise ValueError('stream ends inside a move (dangling from-square)')
    return moves


def next_slot(ids):
    """Which slot the *next* token fills, given the stream so far.

    Used by the constrained sampler to pick the right legality mask. A promo
    slot is never "expected" -- it is optional -- so this returns SLOT_FROM
    after a complete from/to pair and the caller consults the board to decide
    whether a promotion token is also permitted.
    """
    pending = False
    for i in ids:
        i = int(i)
        if i < 64:
            pending = not pending
    return SLOT_TO if pending else SLOT_FROM


def tokens_per_game(n_plies, n_promotions=0, result=True):
    return 1 + int(result) + 2 * n_plies + n_promotions + 1


def meta():
    """The dict nanoGPT's train.py/sample.py expect in data/<set>/meta.pkl."""
    return {
        'vocab_size': VOCAB_SIZE,
        'itos': dict(ITOS),
        'stoi': dict(STOI),
        'tokenizer': 'square_pair_uci_v1',
    }


# --- context cropping ------------------------------------------------------
# A long game outgrows block_size, so the prefix has to be cropped. Both
# backends use this one rule so their logits stay identical:
#
#   * keep the last (block_size - max_ext) prefix tokens, leaving room for the
#     from/to/promo tokens a move needs appended on top. Budgeting a fixed
#     max_ext (rather than the current call's) keeps the crop point stable
#     across the several calls that score one move, which is what lets the
#     KV cache survive them.
#   * then nudge the crop forward until the window starts at a from-slot, so
#     the context never begins half way through a move.
#
# The cropped window has no <bos> or result token. That is fine: training
# samples random windows out of a concatenated stream, so mid-game windows
# without a game header are exactly what the model saw.

MAX_MOVE_TOKENS = 3


def crop_offset(prefix, block_size, max_ext=MAX_MOVE_TOKENS):
    n = len(prefix)
    crop = max(0, n + max_ext - block_size)
    if crop and next_slot(prefix[:crop]) != SLOT_FROM:
        crop += 1
    return crop


def crop_prefix(prefix, block_size, max_ext=MAX_MOVE_TOKENS):
    return list(prefix[crop_offset(prefix, block_size, max_ext):])
