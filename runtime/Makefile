# Build the C runtime. Native by default; cross-compile for the UNO Q's
# Cortex-A53 with:
#
#   make CROSS=aarch64-linux-gnu- ARCHFLAGS='-mcpu=cortex-a53'
#
# Or just build it on the board -- it is two files and needs only libm.

CROSS     ?=
CC        := $(CROSS)gcc
ARCHFLAGS ?= -march=native
CFLAGS    ?= -O3 -ffast-math -funroll-loops $(ARCHFLAGS) -Wall -Wextra
LDLIBS    := -lm

all: gptc-cli

gptc-cli: main.c gptc.c gptc.h
	$(CC) $(CFLAGS) -o $@ main.c gptc.c $(LDLIBS)

clean:
	rm -f gptc-cli

.PHONY: all clean
