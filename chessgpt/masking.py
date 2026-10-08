"""
Legality-constrained decoding.

This is the payoff of square-pair tokenization. A small GPT trained only on
move sequences will happily propose illegal moves; here it simply cannot,
because at every slot the candidate set is intersected with what chess allows:

    from-slot  -> squares holding a piece that has at least one legal move
    to-slot    -> legal destinations from the chosen origin
    promo-slot -> promotion pieces legal for that (from, to) pair

Rather than sampling the origin and then being stuck with it, we score every
legal move exactly:

    log P(move) = log P(from) + log P(to | from) [+ log P(promo | from, to)]

and normalize over the legal set. That needs one forward pass for the origin
distribution plus one batched pass over the distinct origins (plus a rare third
for promotions) -- and it is the model's true conditional distribution
restricted to legal chess, not a greedy approximation.
"""

import math
from collections import defaultdict

import chess
import numpy as np

from . import tokenizer as T


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


# --- mask construction -----------------------------------------------------

def from_square_ids(board):
    """Token ids allowed in a from-slot: origins with >=1 legal move."""
    return sorted({m.from_square for m in board.legal_moves})


def to_square_ids(board, from_square):
    """Token ids allowed in a to-slot, given the origin."""
    return sorted({m.to_square for m in board.legal_moves
                   if m.from_square == from_square})


def promotion_ids(board, from_square, to_square):
    """Promo token ids allowed for this (from, to); empty if not a promotion."""
    return sorted({T.PROMO_ID[m.promotion] for m in board.legal_moves
                   if m.from_square == from_square
                   and m.to_square == to_square and m.promotion})


def allowed_ids(board, slot, from_square=None, to_square=None):
    """The mask for any slot, as a list of token ids."""
    if slot == T.SLOT_FROM:
        return from_square_ids(board)
    if slot == T.SLOT_TO:
        return to_square_ids(board, from_square)
    if slot == T.SLOT_PROMO:
        return promotion_ids(board, from_square, to_square)
    raise ValueError(slot)


def mask_logits(logits, allowed):
    """Return a copy of `logits` with everything outside `allowed` at -inf."""
    out = np.full_like(np.asarray(logits, dtype=np.float64), -np.inf)
    idx = np.asarray(allowed, dtype=np.int64)
    out[idx] = np.asarray(logits, dtype=np.float64)[idx]
    return out


# --- exact scoring over the legal move set ---------------------------------

def score_legal_moves(backend, prefix_ids, board):
    """Score every legal move. Returns [(chess.Move, raw_logprob)], best first.

    `backend.next_logits(prefix, extensions)` must return an array of shape
    (len(extensions), vocab) holding the logits at the final position of each
    `prefix + extension`; all extensions in one call have equal length.
    """
    legal = list(board.legal_moves)
    if not legal:
        return []

    by_from = defaultdict(list)
    for m in legal:
        by_from[m.from_square].append(m)
    froms = sorted(by_from)

    lp_from = log_softmax(backend.next_logits(prefix_ids, [[]])[0])

    to_logits = backend.next_logits(prefix_ids, [[f] for f in froms])
    lp_to = {f: log_softmax(to_logits[i]) for i, f in enumerate(froms)}

    promo_pairs = sorted({(m.from_square, m.to_square) for m in legal if m.promotion})
    lp_promo = {}
    if promo_pairs:
        pl = backend.next_logits(prefix_ids, [[f, t] for f, t in promo_pairs])
        for i, pair in enumerate(promo_pairs):
            lp_promo[pair] = log_softmax(pl[i])

    scored = []
    for m in legal:
        s = lp_from[m.from_square] + lp_to[m.from_square][m.to_square]
        if m.promotion:
            s += lp_promo[(m.from_square, m.to_square)][T.PROMO_ID[m.promotion]]
        scored.append((m, float(s)))
    scored.sort(key=lambda x: -x[1])
    return scored


def pick(scored, temperature=0.0, top_k=0, rng=None):
    """Choose from scored moves: argmax at temperature 0, else softmax sample."""
    if not scored:
        return None
    if temperature <= 0:
        return scored[0][0]
    cand = scored[:top_k] if top_k else scored
    logits = np.array([s for _, s in cand]) / temperature
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    rng = rng or np.random
    return cand[int(rng.choice(len(cand), p=p))][0]


def move_probabilities(scored):
    """Normalize raw log-probs over the legal set -> (move, probability)."""
    if not scored:
        return []
    m = max(s for _, s in scored)
    w = [(mv, math.exp(s - m)) for mv, s in scored]
    z = sum(x for _, x in w)
    return [(mv, x / z) for mv, x in w]


# --- unconstrained generation (for measuring what the model learned) -------

def generate_unconstrained_move(backend, prefix_ids, temperature=0.0, rng=None):
    """Emit one move with NO legality mask.

    Returns (uci, well_formed), where `well_formed` is True if the network's own
    unmasked choice in both slots was a square token. We then force the slots to
    be squares anyway so a UCI string always comes back -- the caller checks it
    against the board. The legal rate of these moves is the honest measure of
    how much chess the network itself has internalized, as opposed to how much
    the mask is doing for it.
    """
    rng = rng or np.random
    ids = []
    well_formed = True

    def step(seq):
        nonlocal well_formed
        logits = np.asarray(backend.next_logits(prefix_ids, [seq])[0], dtype=np.float64)
        if int(np.argmax(logits)) not in range(64):
            well_formed = False
        logits = mask_logits(logits, T.SQUARE_IDS)   # only "must be a square"
        if temperature <= 0:
            return int(np.argmax(logits))
        p = np.exp(log_softmax(logits / max(temperature, 1e-6)))
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    f = step(ids)
    ids.append(f)
    t = step(ids)
    ids.append(t)
    return chess.square_name(f) + chess.square_name(t), well_formed
