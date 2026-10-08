"""Locating the Stockfish binary.

Kept separate from selfplay.py so evaluate.py can find an engine without
pulling the data-generation stage (and its multiprocessing machinery) onto the
board.
"""

import os
import shutil


def find_stockfish(explicit=None):
    """Locate a Stockfish binary: flag > env > vendored > PATH."""
    candidates = [
        explicit,
        os.environ.get('STOCKFISH_PATH'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'bin', 'stockfish'),
        shutil.which('stockfish'),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        'Stockfish not found. Run scripts/get_stockfish.sh, or pass '
        '--engine /path/to/stockfish, or set STOCKFISH_PATH.')
