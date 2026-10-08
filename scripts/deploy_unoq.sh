#!/usr/bin/env bash
# Stage 7 -- push the inference side to the UNO Q's Linux (Cortex-A53) half.
#
#   ./scripts/deploy_unoq.sh arduino@uno-q.local ~/chessgpt out/chess-small.gptc
#
# Only the model file, the tokenizer/masking/runtime modules and numpy+chess go
# over. No torch, no training code, no dataset.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${1:?usage: deploy_unoq.sh user@host [remote_dir] [model.gptc]}
REMOTE_DIR=${2:-~/chessgpt}
MODEL=${3:-out/chess-small.gptc}

[[ -f $MODEL ]] || { echo "no such model: $MODEL (run 'make export')" >&2; exit 1; }

echo "== creating $TARGET:$REMOTE_DIR"
ssh "$TARGET" "mkdir -p $REMOTE_DIR/chessgpt"

echo "== syncing runtime"
rsync -az --delete \
  chessgpt/__init__.py chessgpt/tokenizer.py chessgpt/masking.py \
  chessgpt/gptc.py chessgpt/runtime_np.py chessgpt/backends.py \
  chessgpt/engine.py chessgpt/uci.py chessgpt/play.py chessgpt/games.py \
  chessgpt/evaluate.py chessgpt/prepare.py chessgpt/stockfish.py \
  "$TARGET:$REMOTE_DIR/chessgpt/"

echo "== syncing model ($(du -h "$MODEL" | cut -f1))"
rsync -az "$MODEL" "$TARGET:$REMOTE_DIR/$(basename "$MODEL")"

echo "== installing deps"
ssh "$TARGET" "python3 -c 'import numpy, chess' 2>/dev/null || \
  pip3 install --break-system-packages --quiet numpy chess"

echo "== smoke test on the board"
ssh "$TARGET" "cd $REMOTE_DIR && printf 'uci\nposition startpos moves e2e4 e7e5\ngo\nquit\n' | \
  python3 -m chessgpt.uci --model $(basename "$MODEL") | tail -3"

cat <<EOF

deployed. On the board:
  cd $REMOTE_DIR
  python3 -m chessgpt.play --model $(basename "$MODEL")        # play it
  python3 -m chessgpt.uci  --model $(basename "$MODEL")        # UCI on stdio
  python3 -m chessgpt.evaluate bench --model $(basename "$MODEL") --bench-moves 40
EOF
