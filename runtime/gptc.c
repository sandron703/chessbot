#include "gptc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- helpers */

static void *xalloc(size_t n) {
    void *p = calloc(1, n);
    if (!p) { fprintf(stderr, "gptc: out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

/* y[out] = bias[out] + W[out,:] . x  (nn.Linear stores [out, in]) */
static void matvec(float *y, const float *W, const float *b,
                   const float *x, int out, int in) {
    for (int o = 0; o < out; o++) {
        const float *w = W + (size_t)o * in;
        float s = b ? b[o] : 0.0f;
        for (int i = 0; i < in; i++) s += w[i] * x[i];
        y[o] = s;
    }
}

static void layer_norm(float *y, const float *x, const float *w,
                       const float *b, int n) {
    float mu = 0.0f;
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mu; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++)
        y[i] = (x[i] - mu) * inv * w[i] + (b ? b[i] : 0.0f);
}

static void softmax_inplace(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* torch's nn.GELU() default is the exact erf form, not the tanh approximation.
 * Using the wrong one costs ~1e-3 per activation and quietly breaks parity. */
static float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

/* ------------------------------------------------- minimal .gptc json scan */
/* We emit this JSON ourselves (no whitespace, no escapes), so a scanner is
 * enough and keeps the runtime free of a JSON dependency. */

static int json_int(const char *js, const char *key, int dflt) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(js, pat);
    if (!p) return dflt;
    p += strlen(pat);
    if (*p == 't') return 1;
    if (*p == 'f') return 0;
    return (int)strtol(p, NULL, 10);
}

typedef struct {
    char name[72];
    int quant;
    size_t off, nbytes;
    long scale_off;
    int shape[2], ndim;
} Entry;

static const char *scan_entry(const char *p, Entry *e) {
    p = strstr(p, "{\"name\":\"");
    if (!p) return NULL;
    p += 9;
    const char *q = strchr(p, '"');
    size_t len = (size_t)(q - p);
    if (len >= sizeof e->name) len = sizeof e->name - 1;
    memcpy(e->name, p, len);
    e->name[len] = 0;

    const char *end = strchr(q, '}');
    char buf[384];
    size_t n = (size_t)(end - q);
    if (n >= sizeof buf) n = sizeof buf - 1;
    memcpy(buf, q, n);
    buf[n] = 0;

    e->quant = strstr(buf, "\"dtype\":\"int8\"") != NULL;
    e->off = (size_t)json_int(buf, "offset", 0);
    e->nbytes = (size_t)json_int(buf, "nbytes", 0);
    e->scale_off = strstr(buf, "scale_offset") ? json_int(buf, "scale_offset", -1) : -1;

    const char *sh = strstr(buf, "\"shape\":[");
    e->ndim = 0;
    if (sh) {
        sh += 9;
        while (e->ndim < 2 && (*sh >= '0' && *sh <= '9')) {
            e->shape[e->ndim++] = (int)strtol(sh, (char **)&sh, 10);
            if (*sh == ',') sh++;
        }
    }
    return end;
}

/* Dequantize (or copy) one tensor into a fresh float array. */
static float *load_tensor(const unsigned char *blob, const Entry *e, size_t *bytes) {
    size_t count = e->quant ? e->nbytes : e->nbytes / 4;
    float *out = xalloc(count * sizeof(float));
    *bytes += count * sizeof(float);
    if (!e->quant) {
        memcpy(out, blob + e->off, e->nbytes);
        return out;
    }
    const signed char *q = (const signed char *)(blob + e->off);
    const float *scale = (const float *)(blob + e->scale_off);
    int rows = e->shape[0], cols = e->ndim > 1 ? e->shape[1] : 1;
    for (int r = 0; r < rows; r++) {
        float s = scale[r];
        for (int c = 0; c < cols; c++) out[(size_t)r * cols + c] = q[(size_t)r * cols + c] * s;
    }
    return out;
}

/* ------------------------------------------------------------------- open */

GptcModel *gptc_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gptc: cannot open %s\n", path); return NULL; }
    unsigned char head[16];
    if (fread(head, 1, 16, f) != 16 || memcmp(head, "GPTC", 4)) {
        fprintf(stderr, "gptc: %s is not a .gptc file\n", path);
        fclose(f); return NULL;
    }
    unsigned version, json_len, data_off;
    memcpy(&version, head + 4, 4);
    memcpy(&json_len, head + 8, 4);
    memcpy(&data_off, head + 12, 4);
    if (version != 1) {
        fprintf(stderr, "gptc: unsupported version %u\n", version);
        fclose(f); return NULL;
    }
    char *js = xalloc(json_len + 1);
    if (fread(js, 1, json_len, f) != json_len) { fclose(f); free(js); return NULL; }

    fseek(f, 0, SEEK_END);
    long total = ftell(f);
    size_t blob_len = (size_t)total - data_off;
    unsigned char *blob = xalloc(blob_len);
    fseek(f, (long)data_off, SEEK_SET);
    if (fread(blob, 1, blob_len, f) != blob_len) {
        fclose(f); free(js); free(blob); return NULL;
    }
    fclose(f);

    GptcModel *m = xalloc(sizeof *m);
    m->n_layer = json_int(js, "n_layer", 0);
    m->n_head = json_int(js, "n_head", 0);
    m->n_embd = json_int(js, "n_embd", 0);
    m->block_size = json_int(js, "block_size", 0);
    m->vocab_size = json_int(js, "vocab_size", 0);
    m->has_bias = json_int(js, "bias", 0);
    m->head_size = m->n_embd / m->n_head;
    m->layers = xalloc((size_t)m->n_layer * sizeof(GptcLayer));

    const char *p = strstr(js, "\"tensors\":");
    Entry e;
    while (p && (p = scan_entry(p, &e))) {
        float *t = load_tensor(blob, &e, &m->bytes);
        const char *nm = e.name;
        if (!strcmp(nm, "wte")) { m->wte = t; continue; }
        if (!strcmp(nm, "wpe")) { m->wpe = t; continue; }
        if (!strcmp(nm, "ln_f.weight")) { m->lnf_w = t; continue; }
        if (!strcmp(nm, "ln_f.bias")) { m->lnf_b = t; continue; }
        int l = -1;
        const char *rest = NULL;
        if (sscanf(nm, "h.%d.", &l) == 1 && (rest = strchr(nm + 2, '.'))) {
            rest++;
            GptcLayer *L = &m->layers[l];
            if (!strcmp(rest, "ln_1.weight")) L->ln1_w = t;
            else if (!strcmp(rest, "ln_1.bias")) L->ln1_b = t;
            else if (!strcmp(rest, "ln_2.weight")) L->ln2_w = t;
            else if (!strcmp(rest, "ln_2.bias")) L->ln2_b = t;
            else if (!strcmp(rest, "attn.c_attn.weight")) L->attn_w = t;
            else if (!strcmp(rest, "attn.c_attn.bias")) L->attn_b = t;
            else if (!strcmp(rest, "attn.c_proj.weight")) L->proj_w = t;
            else if (!strcmp(rest, "attn.c_proj.bias")) L->proj_b = t;
            else if (!strcmp(rest, "mlp.c_fc.weight")) L->fc_w = t;
            else if (!strcmp(rest, "mlp.c_fc.bias")) L->fc_b = t;
            else if (!strcmp(rest, "mlp.c_proj.weight")) L->fcp_w = t;
            else if (!strcmp(rest, "mlp.c_proj.bias")) L->fcp_b = t;
            else { fprintf(stderr, "gptc: unknown tensor %s\n", nm); free(t); }
            continue;
        }
        fprintf(stderr, "gptc: unknown tensor %s\n", nm);
        free(t);
    }
    free(js);
    free(blob);

    size_t kv = (size_t)m->n_layer * m->n_head * m->block_size * m->head_size;
    m->kcache = xalloc(kv * sizeof(float));
    m->vcache = xalloc(kv * sizeof(float));
    int C = m->n_embd;
    m->x = xalloc(C * sizeof(float));
    m->h = xalloc(C * sizeof(float));
    m->qkv = xalloc(3 * C * sizeof(float));
    m->att = xalloc((size_t)m->block_size * sizeof(float));
    m->ff = xalloc(4 * (size_t)C * sizeof(float));
    m->logits = xalloc((size_t)m->vocab_size * sizeof(float));
    m->n = 0;
    return m;
}

void gptc_free(GptcModel *m) {
    if (!m) return;
    free(m->wte); free(m->wpe); free(m->lnf_w); free(m->lnf_b);
    for (int l = 0; l < m->n_layer; l++) {
        GptcLayer *L = &m->layers[l];
        free(L->ln1_w); free(L->ln1_b); free(L->ln2_w); free(L->ln2_b);
        free(L->attn_w); free(L->attn_b); free(L->proj_w); free(L->proj_b);
        free(L->fc_w); free(L->fc_b); free(L->fcp_w); free(L->fcp_b);
    }
    free(m->layers);
    free(m->kcache); free(m->vcache);
    free(m->x); free(m->h); free(m->qkv); free(m->att); free(m->ff); free(m->logits);
    free(m);
}

void gptc_reset(GptcModel *m) { m->n = 0; }
int gptc_committed(const GptcModel *m) { return m->n; }

/* ---------------------------------------------------------------- forward */

static void forward_one(GptcModel *m, int id, int pos) {
    const int C = m->n_embd, H = m->n_head, hs = m->head_size;
    const float scale = 1.0f / sqrtf((float)hs);
    const size_t per_layer = (size_t)H * m->block_size * hs;

    for (int i = 0; i < C; i++)
        m->x[i] = m->wte[(size_t)id * C + i] + m->wpe[(size_t)pos * C + i];

    for (int l = 0; l < m->n_layer; l++) {
        GptcLayer *L = &m->layers[l];
        layer_norm(m->h, m->x, L->ln1_w, L->ln1_b, C);
        matvec(m->qkv, L->attn_w, L->attn_b, m->h, 3 * C, C);

        float *q = m->qkv, *k = m->qkv + C, *v = m->qkv + 2 * C;
        float *kc = m->kcache + (size_t)l * per_layer;
        float *vc = m->vcache + (size_t)l * per_layer;

        for (int hh = 0; hh < H; hh++) {
            size_t base = (size_t)hh * m->block_size * hs;
            memcpy(kc + base + (size_t)pos * hs, k + hh * hs, hs * sizeof(float));
            memcpy(vc + base + (size_t)pos * hs, v + hh * hs, hs * sizeof(float));

            const float *qh = q + hh * hs;
            for (int t = 0; t <= pos; t++) {
                const float *kt = kc + base + (size_t)t * hs;
                float s = 0.0f;
                for (int i = 0; i < hs; i++) s += qh[i] * kt[i];
                m->att[t] = s * scale;
            }
            softmax_inplace(m->att, pos + 1);

            float *out = m->h + hh * hs;          /* reuse h as the attn output */
            memset(out, 0, hs * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *vt = vc + base + (size_t)t * hs;
                float a = m->att[t];
                for (int i = 0; i < hs; i++) out[i] += a * vt[i];
            }
        }

        matvec(m->qkv, L->proj_w, L->proj_b, m->h, C, C);
        for (int i = 0; i < C; i++) m->x[i] += m->qkv[i];

        layer_norm(m->h, m->x, L->ln2_w, L->ln2_b, C);
        matvec(m->ff, L->fc_w, L->fc_b, m->h, 4 * C, C);
        for (int i = 0; i < 4 * C; i++) m->ff[i] = gelu(m->ff[i]);
        matvec(m->qkv, L->fcp_w, L->fcp_b, m->ff, C, 4 * C);
        for (int i = 0; i < C; i++) m->x[i] += m->qkv[i];
    }

    layer_norm(m->h, m->x, m->lnf_w, m->lnf_b, C);
    matvec(m->logits, m->wte, NULL, m->h, m->vocab_size, C);   /* tied head */
}

const float *gptc_forward(GptcModel *m, const int *ids, int n, int commit) {
    if (n <= 0 || m->n + n > m->block_size) return NULL;
    for (int i = 0; i < n; i++) forward_one(m, ids[i], m->n + i);
    if (commit) m->n += n;
    return m->logits;
}
