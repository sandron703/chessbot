# nanoGPT chess on the Arduino UNO Q — prototype pipeline

Seven stages, from "no data" to "a UCI engine running on the board". Every
stage is a standalone module you can run by hand; the `Makefile` just records
the wiring and the knobs that matter.

```
 1 selfplay  stockfish self-play ........... data/selfplay/shard-*.jsonl.zst
 2 prepare   tokenize ..................... ../nanoGPT/data/chess_uci/{train,val}.bin
                                            + meta.pkl + val_games.jsonl.zst
 3 train     nanoGPT, unmodified .......... ../nanoGPT/out-chess-small/ckpt.pt
 4 evaluate  loss / legality / ACPL / Elo / bench
 5 uci       the model as a chess engine on stdio
 6 export    quantize ..................... out/chess-small.gptc
 7 deploy    rsync to the UNO Q + bench there
```

`make smoke` runs all seven on 400 games in a few minutes. Run it first — it
fails loudly if any stage stops fitting the one next to it.

## Quick start

```sh
pip install -r requirements.txt     # chess, zstandard, numpy, tqdm
./scripts/get_stockfish.sh          # -> bin/stockfish
make smoke                          # prove the pipeline works end to end
make test                           # tokenizer / masking / runtime parity

make data GAMES=220000              # ~1.6 h on 18 cores
make prepare
make train                          # MODEL=small (default) or medium
make export
make eval
make play                           # sit down and play it
make deploy BOARD=uno-q.local
```

## The hardware reality

The UNO Q has two brains. The model runs on the **Qualcomm Dragonwing QRB2210**
side: quad Cortex-A53 @ 2 GHz, 2 or 4 GB LPDDR4, running Debian. That is a
small Linux computer, so the on-board inference path is ordinary Python —
`numpy` plus `python-chess`, no torch — and a 3 M parameter model is a 3.7 MB
file against 2 GB of RAM. The **STM32U585** MCU (786 KB SRAM) is for board I/O:
reed switches, LEDs, a clock. It is not where the transformer lives.

This means the hard part of "GPT on a microcontroller" isn't memory here. It is
getting a 3 M parameter model to play chess at all — which is a data and
tokenization problem, and that is what this pipeline is built around.

## Two design decisions that shape everything

### Square-pair tokenization

A move is two tokens: origin square, destination square (plus a third for a
promotion). Vocabulary 74, padded to 80. ~2.05 tokens per ply, ~287 tokens for
an average game.

```
1.e4 e5 2.Nf3  ->  <bos> <win-w> e2 e4  e7 e5  g1 f3  ...  <eos>
```

| scheme | vocab | tokens/move | embed+head at n_embd=256 |
|---|---|---|---|
| **square pair (this)** | **80** | **2.05** | **20 K** |
| character-level PGN | ~32 | ~7 | 8 K |
| one token per move | ~1968 | 1.0 | 504 K |

A full move vocabulary would spend 16% of the small model's entire budget on
the embedding table. Character-level PGN is cheap per token but quadruples
sequence length. Square pairs sit in between — and they buy the thing below,
which neither alternative gives you.

### Legality-constrained decoding

A 3 M parameter model will propose illegal moves. With square-pair tokens it
simply cannot, because each slot's candidate set is intersected with what chess
allows:

```
from-slot   -> squares holding a piece with >= 1 legal move
to-slot     -> legal destinations from the chosen origin
promo-slot  -> promotion pieces legal for that (from, to)
```

And rather than sampling an origin and getting stuck with it, `masking.py`
scores *every* legal move exactly:

```
log P(move) = log P(from) + log P(to | from) [+ log P(promo | from, to)]
```

then normalizes over the legal set. The normalizer is constant across moves, so
ranking on the raw joint log-probability is the model's true conditional
distribution restricted to legal chess — not a greedy approximation. Cost is
one forward pass for the origin distribution plus one batched pass over the
distinct origins: ~8 forwards per move in practice, all sharing one KV cache.

`evaluate.py legal` reports the **unmasked** legal rate separately, because
that is the honest measure of how much chess the network itself learned versus
how much the mask is doing for it.

## Stage notes

### 1. Self-play (`chessgpt/selfplay.py`)

The failure mode here is a collapsed dataset: Stockfish at fixed depth is
deterministic, so naive self-play writes the same game thousands of times. Two
knobs prevent that:

- `--opening-plies 8 --opening-multipv 4 --opening-temp 60` — the first 8 plies
  are sampled from Stockfish's top-4 at a centipawn softmax temperature, so
  games fan out through sensible-but-varied openings rather than random junk.
- `--play-temp 20` keeps a little sampling noise in the middlegame.

Games are adjudicated (`--resign-cp 900`, draw after 60 plies of near-zero
eval) so the set isn't padded with dead endgames, one shard per worker so the
stage is resumable, and `--skill` is there if you want deliberately weaker,
more human-looking data.

Measured on 18 cores at depth 6: **~38 games/s, 141 plies/game** (~287 tokens).

### 2. Prepare (`chessgpt/prepare.py`)

Writes `train.bin` / `val.bin` / `meta.pkl` exactly where nanoGPT's `train.py`
looks, so **nanoGPT stays unmodified** — it is a clean upstream clone and can
be pulled. Games are concatenated back to back; nanoGPT samples random windows
out of the stream, so the model sees both full games and mid-game windows. That
is deliberate: a long game outgrows `block_size` at inference and the context
gets cropped, and training on unaligned windows is what makes the cropped case
work.

The train/val split is **by game**, and the held-out game records are written
alongside the bins as `val_games.jsonl.zst` — every position-based metric needs
real boards, not token ids.

### 3. Train

```sh
cd ../nanoGPT && python train.py "../prototype./configs/train_chess_small.py"
```

| config | layers | heads | n_embd | params | int8 file | KV @ 384 ctx | tokens wanted | self-play time |
|---|---|---|---|---|---|---|---|---|
| `smoke` | 2 | 2 | 64 | 0.1 M | 0.16 MB | 0.1 MB | — | seconds |
| `small` | 4 | 4 | 256 | 3.2 M | 3.7 MB | 3.1 MB | ~63 M (220 K games) | ~1.6 h |
| `medium` | 8 | 8 | 512 | 25 M | 26 MB | 13 MB | ~504 M (1.8 M games) | ~13 h |

`medium` is the size at which published PGN-trained GPTs reach roughly
1300–1500 Elo. Start with `small`; generate `medium`'s data while `small`
trains. `block_size` is 384 (~190 plies), which covers most games whole.

**Training is not the slow part.** Measured on an RTX 4050 Laptop (6 GB):

| config | ms/iter | achieved | full run |
|---|---|---|---|
| `small`, `compile=False` | 61 | 9.6 TFLOP/s | ~12 min |
| `small`, `compile=True` | 49 | 12.0 TFLOP/s | **~10 min** |
| `medium`, `compile=True` | ~520 (est.) | | ~6 h |

Self-play dominates: ~1.6 h of data generation for ~10 min of `small` training.
If `small` is overfitting, generate more games rather than training longer —
12k iters is already ~4.7 epochs over a 63 M-token set.

**Two readouts that will fool you:**

- **`mfu` is meaningless here.** nanoGPT hardcodes `flops_promised = 312e12`
  (A100 bf16 peak) at `model.py:301`. On a 4050 that denominator is ~6× too
  large, so a *fully saturated* GPU reports `mfu 3.9%`. Judge throughput by
  ms/iter, or by `nvidia-smi dmon -s pucm` in another terminal — a real
  `small` run sits at SM 100%, memory bus 97%, 70 W, 2.5 GB VRAM.
- **`make smoke` looks like it is running on CPU.** It isn't; the config is so
  tiny (2 layers, 64 dim, batch 16) that each step is ~3 ms / 0.5 TFLOP/s. The
  GPU finishes faster than `nvidia-smi` samples, so it reads 0% at idle power.
  The whole 200-iteration run is under a second of GPU work.

### 4. Evaluate (`chessgpt/evaluate.py`)

Loss alone says almost nothing about a chess model, so there are five metrics:

```sh
python -m chessgpt.evaluate all --ckpt ../nanoGPT/out-chess-small/ckpt.pt \
    --model out/chess-small.gptc
```

| metric | what it tells you |
|---|---|
| `loss` | cross-entropy on held-out games, **split by slot**. from-slot and to-slot behave very differently; the split is the fastest read on what the model is learning |
| `legal` | unmasked legal-move rate — how much chess the network itself knows |
| `agreement` | top-1 agreement with Stockfish, mean/median/p90 centipawn loss, blunder rate |
| `match` | real games vs Stockfish at a fixed skill level → Elo ± band |
| `bench` | ms/move, forwards per move, peak RSS — **run this on the board** |

### 5. UCI (`chessgpt/uci.py`)

```sh
python -m chessgpt.uci --model out/chess-small.gptc
```

Makes the model a real chess engine: any GUI can load it, and `cutechess-cli`
can run a few hundred games against Stockfish. Non-standard `board` and `top`
commands print the position and the ranked move list for manual play.

One limitation worth knowing: the model reasons from move history, so a
`position fen ...` with no `moves` list loses its context. The mask still keeps
it legal, but it plays worse. The UCI driver keeps the full game whenever the
caller provides it.

### 6. Export (`chessgpt/export.py`)

Produces `.gptc`: a 16-byte header, a JSON tensor directory, then 64-byte
aligned blobs in a fixed canonical order. The four big projections per layer
(>95% of parameters) are int8 with one fp32 scale per output row; embeddings,
LayerNorms and biases stay fp32 — they are a rounding error in file size and
the tied output head is where quantization noise would cost the most logit
fidelity. `--emit-header` also writes a C header with every byte offset.

Measured: int8 quantization moves logits by ~3e-3 and does not change the
chosen move in any tested position.

### 7. Deploy

```sh
make deploy BOARD=uno-q.local
```

Sends the `.gptc` and the eleven runtime modules — no torch, no training code,
no dataset — then runs a UCI smoke test over SSH. On the board:

```sh
python3 -m chessgpt.play --model chess-small.gptc
python3 -m chessgpt.evaluate bench --model chess-small.gptc --bench-moves 40
```

`runtime/` holds an optional C implementation of the same forward pass, for
when you want no Python on the board or a NEON int8 kernel. It is held to the
numpy reference by `tests/test_c_parity.py` (currently agreeing to ~1e-7).

## Layout

```
chessgpt/
  tokenizer.py    square-pair vocab, encode/decode, context cropping
  games.py        canonical game records; .jsonl[.zst] and .pgn[.zst] readers
  stockfish.py    finds the engine binary (flag > env > bin/ > PATH)
  selfplay.py     stage 1
  prepare.py      stage 2
  masking.py      legality masks + exact scoring over the legal move set
  backends.py     TorchBackend  (desktop)  -- one next_logits() interface
  runtime_np.py   NumpyBackend  (board)    -- same interface, KV cache
  gptc.py         the .gptc container
  export.py       stage 6
  engine.py       the player: backend + masking + board bookkeeping
  uci.py          stage 5
  play.py         terminal UI
  evaluate.py     stage 4
configs/          nanoGPT configs: smoke / small / medium
runtime/          optional C runtime + its own README
scripts/          get_stockfish.sh, deploy_unoq.sh
tests/            tokenizer, masking, runtime parity, C parity
```

Everything above `backends.py` is backend-agnostic, so the board runs exactly
the code the desktop was measured on.

## What to expect

`smoke` trains a 0.1 M model for 200 iterations on 400 games. It plays badly —
that is the point; it proves the plumbing, not the chess. Rough signposts for a
real `small` run:

- **from-slot loss falling below to-slot loss** is the first sign it is tracking
  the board rather than memorizing move frequencies.
- **unmasked legal rate** is the number to watch. Single digits means the
  network knows nothing and the mask is carrying it. Published PGN-trained GPTs
  get well above 90% at ~25 M parameters; a 3 M model will land lower.
- **ACPL and the match score** are the only measures of strength that matter.
  Start the match at `--skill 0 --sf-nodes 1000` and raise it when you win.

## Open questions this prototype is meant to answer

1. How much does 3.2 M parameters cost in strength versus 25 M on the same data?
2. Does result conditioning (`--condition win`) measurably help? It is free.
3. Is depth-6 self-play the right teacher, or does weaker, more varied data
   (`--skill 5`) generalize better at this model size?
4. Does the model need board state fed to it, or is move history enough at this
   scale? If (2) and (3) plateau badly, this is the next thing to try — and it
   would mean changing the tokenizer, which is why it is kept in one file.
