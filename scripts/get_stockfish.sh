#!/usr/bin/env bash
# Fetch a static Stockfish build into prototype./bin/stockfish.
# Self-play is the slowest stage of the pipeline, so prefer the newest build
# your CPU supports (the official releases are named by instruction set).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin

if command -v stockfish >/dev/null 2>&1; then
  echo "system stockfish already on PATH: $(command -v stockfish)"
fi

ARCH=x86-64-avx2
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then ARCH=x86-64-avx512; fi
if ! grep -q avx2 /proc/cpuinfo 2>/dev/null; then ARCH=x86-64-modern; fi

VER=${SF_VERSION:-17.1}
URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_${VER}/stockfish-ubuntu-${ARCH}.tar"
echo "downloading $URL"
curl -fL "$URL" -o /tmp/sf.tar
tar -xf /tmp/sf.tar -C /tmp
find /tmp/stockfish -type f -name 'stockfish*' -perm -u+x | head -1 | xargs -I{} cp {} bin/stockfish
chmod +x bin/stockfish
rm -rf /tmp/sf.tar /tmp/stockfish
./bin/stockfish --help 2>/dev/null | head -2 || true
echo "ok -> $(pwd)/bin/stockfish"
