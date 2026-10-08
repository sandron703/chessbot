/* gptc -- dependency-free inference for a .gptc chess GPT.
 *
 * Why this exists: runtime_np.py already runs fine on the UNO Q's Cortex-A53
 * side, and it is the reference. This is the same forward pass in C for when
 * you want no Python on the board (~1 MB RSS instead of ~40 MB), a single
 * static binary, or a NEON int8 kernel underneath.
 *
 * The contract is the .gptc file and nothing else, so this stays correct as
 * long as tests/test_c_parity.py keeps passing against the numpy version.
 *
 * Usage:
 *     GptcModel *m = gptc_open("chess-small.gptc");
 *     gptc_reset(m);
 *     gptc_forward(m, prefix_ids, n_prefix, 1);          // fill the KV cache
 *     const float *l = gptc_forward(m, &from_sq, 1, 0);  // try a candidate
 *     gptc_free(m);
 *
 * commit=0 runs the tokens against the cache WITHOUT advancing it, which is
 * what lets the move scorer try a dozen origin squares on one shared prefix.
 */
#ifndef GPTC_H
#define GPTC_H

#include <stddef.h>

typedef struct {
    float *ln1_w, *ln1_b;
    float *attn_w, *attn_b;   /* [3C, C] */
    float *proj_w, *proj_b;   /* [C, C]  */
    float *ln2_w, *ln2_b;
    float *fc_w,  *fc_b;      /* [4C, C] */
    float *fcp_w, *fcp_b;     /* [C, 4C] */
} GptcLayer;

typedef struct {
    int n_layer, n_head, n_embd, block_size, vocab_size, has_bias;
    int head_size;

    float *wte;               /* [vocab, C], also the tied output head */
    float *wpe;               /* [block, C] */
    GptcLayer *layers;
    float *lnf_w, *lnf_b;

    float *kcache, *vcache;   /* [n_layer][n_head][block][head_size] */
    int n;                    /* committed tokens in the cache */

    /* scratch */
    float *x, *h, *qkv, *att, *ff, *logits;

    size_t bytes;             /* resident parameter bytes, for reporting */
} GptcModel;

GptcModel *gptc_open(const char *path);
void gptc_free(GptcModel *m);
void gptc_reset(GptcModel *m);
int gptc_committed(const GptcModel *m);

/* Run `n` tokens on top of the cache. Returns logits for the LAST token,
 * a pointer into the model (valid until the next call). NULL on overflow. */
const float *gptc_forward(GptcModel *m, const int *ids, int n, int commit);

#endif /* GPTC_H */
