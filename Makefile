# Pipeline driver. Every stage is also a plain module you can run by hand --
# this file just records the wiring and the knobs that matter.
#
#   make smoke        prove all seven stages connect (a few minutes, CPU-ok)
#   make test         unit tests (tokenizer, masking, runtime parity)
#   make ingest       PGN archive -> games      (stage 1a, parallel)
#   make data         stockfish self-play -> games (stage 1b, optional)
#   make prepare      games -> train/val.bin   (stage 2)
#   make train        nanoGPT training         (stage 3)
#   make eval         the five metrics         (stage 4)
#   make export       ckpt -> .gptc            (stage 6)
#   make play         play it in your terminal
#   make deploy       rsync the runtime to the UNO Q and bench it there

SHELL     := /bin/bash
PY        ?= ../.venv/bin/python
NANOGPT   ?= ../nanoGPT

# small | medium -- an inline comment here would leak its leading
# whitespace into the variable and break every path built from it.
MODEL     ?= small
DATASET   ?= chess_uci
OUTDIR    ?= $(NANOGPT)/out-chess-$(MODEL)
CKPT      ?= $(OUTDIR)/ckpt.pt
GPTC      ?= out/chess-$(MODEL).gptc

# stage 1 -- where games come from. GAMEDIR is what prepare reads, so point it
# at whichever source you are using (or at a directory holding both).
WORKERS   ?= $(shell nproc --ignore=2)

# 1a: a PGN archive, e.g. a Lichess Elite monthly dump. One pass of ~425k
# games is ~72M tokens, which is already the small model's full token budget.
PGN       ?= data/lichess_elite_2020-06.pgn
GAMEDIR   ?= data/lichess_elite

# 1b: stockfish self-play, if you want engine-quality data instead of human.
GAMES     ?= 210000
DEPTH     ?= 6
SELFPLAY  ?= data/selfplay

# stage 2 filters. Time forfeits carry a 1-0/0-1 label that does not reflect
# the position, which is noise for the <result> conditioning token.
PREP_ARGS ?= --termination Normal --block-size 384

# stage 4 -- evaluation opponent
SKILL     ?= 0
SF_NODES  ?= 2000
MATCH     ?= 100
POSITIONS ?= 500

# stage 7 -- the board
BOARD     ?= uno-q.local
BOARD_USER?= arduino
BOARD_DIR ?= ~/chessgpt

.PHONY: help smoke test ingest data prepare train eval export play uci deploy bench-board clean distclean stockfish

help:
	@sed -n '4,13p' Makefile | sed 's/^#   //'

stockfish: bin/stockfish
bin/stockfish:
	./scripts/get_stockfish.sh

# --- stage 1a: PGN archive -------------------------------------------------
ingest:
	$(PY) -m chessgpt.ingest --pgn $(PGN) --out $(GAMEDIR) --workers $(WORKERS)

# --- stage 1b: stockfish self-play (optional) ------------------------------
data: bin/stockfish
	$(PY) -m chessgpt.selfplay --out $(SELFPLAY) --games $(GAMES) \
		--workers $(WORKERS) --depth $(DEPTH)

# --- stage 2 ---------------------------------------------------------------
prepare:
	$(PY) -m chessgpt.prepare --src $(GAMEDIR) --dataset $(DATASET) $(PREP_ARGS)

# --- stage 3 ---------------------------------------------------------------
train:
	cd $(NANOGPT) && "$(abspath $(PY))" train.py \
		"$(abspath configs/train_chess_$(MODEL).py)"

# --- stage 4 ---------------------------------------------------------------
eval:
	$(PY) -m chessgpt.evaluate all --ckpt $(CKPT) --model $(GPTC) \
		--dataset $(DATASET) --positions $(POSITIONS) \
		--match-games $(MATCH) --skill $(SKILL) --sf-nodes $(SF_NODES)

# --- stage 6 ---------------------------------------------------------------
export:
	$(PY) -m chessgpt.export --ckpt $(CKPT) --out $(GPTC) --emit-header

# --- play ------------------------------------------------------------------
play:
	$(PY) -m chessgpt.play --model $(GPTC)

uci:
	$(PY) -m chessgpt.uci --model $(GPTC)

# --- stage 7 ---------------------------------------------------------------
deploy:
	./scripts/deploy_unoq.sh $(BOARD_USER)@$(BOARD) $(BOARD_DIR) $(GPTC)

bench-board:
	ssh $(BOARD_USER)@$(BOARD) "cd $(BOARD_DIR) && python3 -m chessgpt.evaluate \
		bench --model $(notdir $(GPTC)) --bench-moves 40"

# --- dev -------------------------------------------------------------------
test:
	@cd tests && for t in test_*.py; do echo "== $$t"; \
		"$(abspath $(PY))" $$t || exit 1; done

# A complete pass over every stage on a few hundred games. This is the thing to
# run after touching anything: it fails loudly if a stage stops fitting the one
# either side of it.
smoke: bin/stockfish
	$(PY) -m chessgpt.selfplay --out data/smoke --games 400 --workers $(WORKERS) \
		--depth 5 --overwrite
	$(PY) -m chessgpt.prepare --src data/smoke --dataset chess_uci_smoke \
		--val-games 40 --block-size 128
	cd $(NANOGPT) && "$(abspath $(PY))" train.py \
		"$(abspath configs/train_chess_smoke.py)"
	$(PY) -m chessgpt.export --ckpt $(NANOGPT)/out-chess-smoke/ckpt.pt \
		--out out/chess-smoke.gptc --emit-header
	$(PY) -m chessgpt.evaluate all --ckpt $(NANOGPT)/out-chess-smoke/ckpt.pt \
		--model out/chess-smoke.gptc --dataset chess_uci_smoke \
		--positions 60 --bench-moves 20 --match-games 4 --sf-nodes 200
	@echo
	@echo "smoke passed: all seven stages connect."

clean:
	rm -rf out/*.gptc out/*.h data/smoke $(NANOGPT)/data/chess_uci_smoke \
		$(NANOGPT)/out-chess-smoke

distclean: clean
	rm -rf $(GAMEDIR) $(SELFPLAY) $(NANOGPT)/data/$(DATASET) bin
