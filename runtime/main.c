/* gptc-cli -- a line protocol over the C runtime.
 *
 * It exists so tests/test_c_parity.py can diff this implementation against
 * runtime_np.py token for token, and so you can time the forward pass on the
 * board without Python in the picture.
 *
 *   reset                 clear the KV cache
 *   commit <id>...        run tokens and KEEP them in the cache
 *   logits <id>...        run tokens WITHOUT keeping them; print the logits
 *   info                  print config and resident parameter bytes
 *   bench <n> <ids...>    time n repeats of a non-committing forward
 *   quit
 */
#include "gptc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_IDS 1024

static int parse_ids(char *rest, int *ids) {
    int n = 0;
    for (char *tok = strtok(rest, " \t\r\n"); tok && n < MAX_IDS;
         tok = strtok(NULL, " \t\r\n"))
        ids[n++] = (int)strtol(tok, NULL, 10);
    return n;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gptc\n", argv[0]);
        return 2;
    }
    GptcModel *m = gptc_open(argv[1]);
    if (!m) return 1;

    char line[16384];
    int ids[MAX_IDS];
    while (fgets(line, sizeof line, stdin)) {
        char *sp = strchr(line, ' ');
        char *rest = sp ? sp + 1 : NULL;
        if (sp) *sp = 0;
        char *cmd = strtok(line, " \t\r\n");
        if (!cmd) continue;

        if (!strcmp(cmd, "quit")) break;

        if (!strcmp(cmd, "reset")) {
            gptc_reset(m);
            printf("ok %d\n", gptc_committed(m));

        } else if (!strcmp(cmd, "info")) {
            printf("n_layer %d n_head %d n_embd %d block_size %d vocab_size %d "
                   "bias %d params_mb %.2f committed %d\n",
                   m->n_layer, m->n_head, m->n_embd, m->block_size,
                   m->vocab_size, m->has_bias, m->bytes / 1e6,
                   gptc_committed(m));

        } else if (!strcmp(cmd, "commit")) {
            int n = rest ? parse_ids(rest, ids) : 0;
            const float *l = gptc_forward(m, ids, n, 1);
            printf(l ? "ok %d\n" : "err %d\n", gptc_committed(m));

        } else if (!strcmp(cmd, "logits")) {
            int n = rest ? parse_ids(rest, ids) : 0;
            const float *l = gptc_forward(m, ids, n, 0);
            if (!l) { printf("err\n"); continue; }
            printf("%d", m->vocab_size);
            for (int i = 0; i < m->vocab_size; i++) printf(" %.7e", l[i]);
            printf("\n");

        } else if (!strcmp(cmd, "bench")) {
            int n = rest ? parse_ids(rest, ids) : 0;
            if (n < 2) { printf("err\n"); continue; }
            int reps = ids[0];
            double t0 = now_ms();
            for (int r = 0; r < reps; r++) gptc_forward(m, ids + 1, n - 1, 0);
            double dt = now_ms() - t0;
            printf("%d forwards of %d token(s): %.3f ms total, %.3f ms each\n",
                   reps, n - 1, dt, dt / reps);

        } else {
            printf("err unknown command %s\n", cmd);
        }
        fflush(stdout);
    }
    gptc_free(m);
    return 0;
}
