# C runtime (optional)

`runtime_np.py` is the reference and is what the pipeline measures. This is the
same forward pass in C, for when you want:

- no Python on the board — a single static binary, ~1 MB RSS instead of ~40 MB
- a NEON int8 kernel under `matvec()`, which is where the real A53 speedup is
- a starting point if you ever retarget the STM32U585 side (see the caveat below)

It is two files plus a driver, depends only on `libm`, and parses nothing but
the `.gptc` file.

## Build

```sh
make                                              # native
make CROSS=aarch64-linux-gnu- ARCHFLAGS='-mcpu=cortex-a53'   # for the UNO Q
```

Building it *on* the board works too and avoids the cross-toolchain entirely.

## Prove it is correct before you trust it

```sh
make -C runtime
cd tests && python test_c_parity.py
```

That diffs the C logits against `runtime_np.py` across prefix-only calls,
one- and two-token extensions, and the commit/no-commit boundary. It currently
agrees to ~1e-7. If you optimize `matvec`, this is the test that tells you
whether you broke the model.

Two details that silently break parity if you change them:

- **GELU must be the exact erf form** (`0.5x(1+erf(x/√2))`), not the tanh
  approximation. nanoGPT uses `nn.GELU()`, whose default is exact.
- **The output head is tied to `wte`.** There is no separate `lm_head` tensor in
  the file; the final matvec reuses the token embedding matrix.

## What is here and what is not

`gptc_forward(m, ids, n, commit)` is the whole API. `commit=0` runs tokens
against the KV cache without advancing it, which is what lets the move scorer
try a dozen candidate origin squares on one shared prefix.

Not here: move generation. Legality masking needs a chess move generator, and
on the Linux side `python-chess` already does that job well. The intended split
is Python for the game logic and masking, C only if the forward pass becomes
the bottleneck — measure with `chessgpt.evaluate bench` on the board first.
A fully self-contained C engine additionally needs a movegen; `tests/` would
then grow a legality fuzz test against python-chess.

## About the STM32U585

The MCU half of the UNO Q has 786 KB SRAM and 2 MB flash. This code would need:
int8 weights kept quantized (no dequantize-on-load — that is the first thing
`gptc_open` does and it is what blows the budget), a quantized KV cache, and a
model around 300 K parameters. That is a different project; the numbers are in
the top-level README.
