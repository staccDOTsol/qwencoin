/* qsha: executor of the SHA-256 circuit compiled into Qwen3.5 weights. See qsha.h. */
#include "qsha.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "qsha_circuit.h"

struct qsha {
    double res[4][QS_NDIMS];        /* residual of the last 4 tokens (the conv window) */
    int cur;                        /* res[cur]: the current token */
    double state[QS_STATE];         /* DeltaNet rows: per value head, rows x columns */
    unsigned char present[QS_NROWS];
    uint64_t tokens;
};

static __thread uint64_t tls_tokens;

size_t qsha_sizeof(void) { return sizeof(qsha); }
int qsha_ndims(void) { return QS_NDIMS; }
uint64_t qsha_tokens(void) { return tls_tokens; }
void qsha_reset(qsha* s) { memset(s, 0, sizeof *s); }
void qsha_copy(qsha* dst, const qsha* src) { memcpy(dst, src, sizeof *dst); }
qsha* qsha_new(void) {
    qsha* s = (qsha*)malloc(sizeof *s);
    if (s) qsha_reset(s);
    return s;
}
void qsha_free(qsha* s) { free(s); }
const double* qsha_residual(const qsha* s) { return s->res[s->cur]; }

static inline double dot(int off, int n, const double* m) {
    double v = 0;
    for (int i = off; i < off + n; i++) v += qs_ecoef[i] * m[qs_eidx[i]];
    return v;
}

static inline void add_row(int r, double* m) {
    for (int i = qs_foff[r]; i < qs_foff[r + 1]; i++) m[qs_fidx[i]] += qs_fval[i];
}

/* The embedding: ONE plus the token's features (a family's base row and its payload bits). */
static void embed(int tok, double* m) {
    memset(m, 0, sizeof(double) * QS_NDIMS);
    if (tok < 0 || tok >= QS_VOCAB || (tok >= 256 + QS_NCTL && tok < QS_V_BASE)) return; /* not a chip token */
    m[0] = 1.0;
    int fam, pay;
    if (tok < 256) { fam = 0; pay = tok; }
    else if (tok < QS_V_BASE) { add_row(QS_CTL0 + tok - 256, m); return; }
    else if (tok < QS_F_BASE) { fam = 1; pay = tok - QS_V_BASE; }
    else { fam = 2; pay = tok - QS_F_BASE; }
    int r = qs_fam[2 * fam], nb = qs_fam[2 * fam + 1];
    add_row(r, m);
    for (int i = 0; i < nb; i++)
        if (pay >> i & 1) add_row(r + 1 + i, m);
}

/* A q / k / v channel: its projection at the token `lag` positions back, through relu. */
static inline double chan(const int* c, double (*res)[QS_NDIMS], int cur) {
    return dot(c[1], c[2], res[(cur - c[3]) & 3]);
}

static void run_heads(qsha* s, int first, int count) {
    double (*res)[QS_NDIMS] = s->res;
    int cur = s->cur;
    double* m = res[cur];
    for (int hi = first; hi < first + count; hi++) {
        const int* h = qs_head + 6 * hi;
        /* Gates first: a write needs beta = 1 and a read needs an open output gate. A head
           that neither writes nor reads at this token leaves its key and query unevaluated;
           what they would select is multiplied by an exact 0 either way. */
        double z[2][16];
        int writes[2], reads[2], any_w = 0, any_r = 0;
        for (int x = 0; x < h[5]; x++) {
            const int* v = qs_vhead + 14 * (h[4] + x);
            if (v[3]) { /* decay: keep, halve or clear the whole table */
                int nrows = v[10], ncols = v[11];
                double* st = s->state + v[12];
                unsigned char* pres = s->present + v[13];
                double a = dot(v[2], v[3], m);
                if (a == QS_DECAY_CLEAR) memset(pres, 0, nrows);
                else if (a == QS_DECAY_HALF) {
                    for (int r = 0; r < nrows; r++)
                        if (pres[r])
                            for (int k = 0; k < ncols; k++) st[r * ncols + k] *= 0.5;
                }
            }
            writes[x] = v[4] < 0 || dot(v[4], v[5], m) > 0;
            reads[x] = 0;
            for (int i = 0; i < v[7]; i++) {
                const int* ze = qs_zexpr + 2 * (v[6] + i);
                double g = dot(ze[0], ze[1], m);
                z[x][i] = g > 0 ? g : 0;
                reads[x] |= g > 0;
            }
            any_w |= writes[x];
            any_r |= reads[x];
        }
        int qrow[2], nq = 0, krow = -1;
        if (any_r)
            for (int i = 0; i < h[1]; i++) {
                const int* c = qs_chan + 4 * (h[0] + i);
                if (chan(c, res, cur) > 0 && nq < 2) qrow[nq++] = c[0];
            }
        if (any_w)
            for (int i = 0; i < h[3]; i++) {
                const int* c = qs_chan + 4 * (h[2] + i);
                if (chan(c, res, cur) > 0) krow = c[0];
            }
        for (int x = 0; x < h[5]; x++) {
            const int* v = qs_vhead + 14 * (h[4] + x);
            int ncols = v[11];
            double* st = s->state + v[12];
            unsigned char* pres = s->present + v[13];
            if (krow >= 0 && writes[x]) { /* delta rule with a one-hot key: overwrite one row */
                double* row = st + krow * ncols;
                memset(row, 0, sizeof(double) * ncols);
                for (int i = 0; i < v[1]; i++) {
                    const int* c = qs_chan + 4 * (v[0] + i);
                    double xv = chan(c, res, cur);
                    if (xv > 0) row[c[0]] = xv;
                }
                pres[krow] = 1;
            }
            if (!nq || !reads[x]) continue;
            const double* r0 = pres[qrow[0]] ? st + qrow[0] * ncols : NULL;
            const double* r1 = nq > 1 && pres[qrow[1]] ? st + qrow[1] * ncols : NULL;
            if (!r0) { r0 = r1; r1 = NULL; }
            if (!r0) continue;
            double b0 = r0[0] + (r1 ? r1[0] : 0), b1 = r0[1] + (r1 ? r1[1] : 0);
            double ref = b0 * b0 + b1 * b1;
            if (ref == 0) continue; /* no ballast: the output norm returns 0 */
            double mult = 1.0 / sqrt(ref / QS_HD); /* the head's output RMSNorm */
            for (int i = v[8]; i < v[8] + v[9]; i++) {
                const int* o = qs_vout + 3 * i;
                double zz = z[x][o[2]];
                if (zz == 0) continue;
                double xv = r0[o[0]] + (r1 ? r1[o[0]] : 0);
                if (xv != 0) m[o[1]] += xv * mult * zz * qs_voutw[i];
            }
        }
    }
}

static void run_mlp(qsha* s, const int* sub) {
    double* m = s->res[s->cur];
    double base[QS_MAXBASE], val[QS_NUNITS];
    int act[QS_NUNITS], n = 0;
    for (int i = 0; i < sub[4]; i++) { /* each distinct pre-activation sum, once */
        const int* e = qs_base + 2 * (sub[3] + i);
        base[i] = dot(e[0], e[1], m);
    }
    for (int u = sub[1]; u < sub[1] + sub[2]; u++) { /* a * relu(b) */
        const int* t = qs_unit + 5 * u;
        double b = base[t[0]] + qs_ucst[u] * m[0];
        if (b <= 0) continue;
        double a = dot(t[1], t[2], m);
        if (a == 0) continue;
        act[n] = u;
        val[n++] = a * b;
    }
    for (int i = 0; i < n; i++) { /* the down projection, added to the residual */
        const int* t = qs_unit + 5 * act[i];
        for (int j = t[3]; j < t[3] + t[4]; j++) m[qs_odim[j]] += qs_ocoef[j] * val[i];
    }
}

/* The output head: the token whose code agrees with the residual on every bit. */
static int decode(const double* m) {
    int fam = (m[qs_code[0]] > 0) | (m[qs_code[1]] > 0) << 1, pay = 0;
    for (int i = 2; i < QS_NCODE; i++) pay |= (m[qs_code[i]] > 0) << (i - 2);
    switch (fam) {
        case 0: return pay < 256 ? pay : -1;
        case 1: return QS_V_BASE + pay;
        case 2: return QS_F_BASE + pay;
        default: return pay < QS_NCTL ? 256 + pay : -1;
    }
}

int qsha_feed(qsha* s, int tok) {
    s->cur = (s->cur + 1) & 3;
    embed(tok, s->res[s->cur]);
    for (int i = 0; i < QS_NSUB; i++) {
        const int* sub = qs_sub + 5 * i;
        if (sub[0]) run_mlp(s, sub);
        else run_heads(s, sub[1], sub[2]);
    }
    s->tokens++;
    tls_tokens++;
    return decode(s->res[s->cur]);
}

/* ---- drivers: the host answers IN and nothing else ---------------------------------------- */

typedef struct {
    qsha* s;
    int pred;      /* the model's prediction after the last token */
    int cv_left;   /* value tokens of the chaining-value record still to come */
    int n_left;    /* value tokens of the block-count record still to come */
    uint32_t cv[8];
    uint64_t n;
    long tokens;
} run_t;

static void step(run_t* r, int tok) {
    if (tok == QS_TOK_RC) r->cv_left = 16;
    else if (tok == QS_TOK_RN) { r->n_left = 4; r->n = 0; }
    else if (tok >= QS_V_BASE) {
        uint32_t p = (uint32_t)(tok - QS_V_BASE) & 0xFFFF;
        if (r->cv_left) {
            int i = 16 - r->cv_left--;
            r->cv[i / 2] = i % 2 ? r->cv[i / 2] | p : p << 16;
        } else if (r->n_left) {
            r->n = r->n << 16 | p;
            r->n_left--;
        }
    }
    r->pred = qsha_feed(r->s, tok);
    r->tokens++;
}

static qsha g_fresh;      /* the state after SHA and the model's own IV / n / BLK tokens */
static uint32_t g_iv[8];
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void init_fresh(void) {
    run_t r = {&g_fresh, 0, 16, 0, {0}, 0, 0};
    qsha_reset(&g_fresh);
    step(&r, QS_TOK_SHA); /* the cv record that follows SHA is the IV, printed by the model */
    while (r.pred != QS_TOK_IN && r.pred >= 0) step(&r, r.pred);
    memcpy(g_iv, r.cv, sizeof g_iv);
}

void qsha_iv(uint32_t state[8]) {
    pthread_once(&g_once, init_fresh);
    memcpy(state, g_iv, sizeof g_iv);
}

static __thread qsha tls_session;

static void put_digest(const uint32_t cv[8], uint8_t out[32]) {
    for (int i = 0; i < 8; i++) {
        out[4 * i] = cv[i] >> 24; out[4 * i + 1] = cv[i] >> 16;
        out[4 * i + 2] = cv[i] >> 8; out[4 * i + 3] = cv[i];
    }
}

/* Feed bytes then END whenever the model asks for input, until it halts. */
static long drain(run_t* r, const uint8_t* msg, size_t len, int end) {
    size_t i = 0;
    int ended = 0;
    for (;;) {
        int tok = r->pred;
        if (tok == QS_TOK_IN) {
            if (i < len) tok = msg[i++];
            else if (end && !ended) { tok = QS_TOK_END; ended = 1; }
            else return i == len && !end ? r->tokens : -1;
        } else if (tok == QS_TOK_HALT) {
            return i == len && ended ? r->tokens : -1;
        } else if (tok < 0) {
            return -1;
        }
        if (tok == QS_TOK_BLK && !end && i == len) { step(r, tok); return r->tokens; }
        step(r, tok);
    }
}

long qsha_sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
    pthread_once(&g_once, init_fresh);
    qsha* s = &tls_session;
    qsha_copy(s, &g_fresh); /* prefix cache: SHA and the IV record are already in the state */
    run_t r = {s, QS_TOK_IN, 0, 0, {0}, 0, 0};
    long n = drain(&r, msg, len, 1);
    if (n < 0) return n;
    put_digest(r.cv, out);
    return n;
}

static void resume(run_t* r, const uint32_t state[8], uint64_t nblk) {
    qsha_reset(r->s);
    step(r, QS_TOK_RES);
    for (int i = 0; i < 8; i++) {
        step(r, QS_V_BASE + (state[i] >> 16));
        step(r, QS_F_BASE + (state[i] & 0xFFFF));
    }
    for (int i = 3; i > 0; i--) step(r, QS_V_BASE + (int)(nblk >> (16 * i) & 0xFFFF));
    step(r, QS_F_BASE + (int)(nblk & 0xFFFF));
    if (r->pred == QS_TOK_BLK) step(r, QS_TOK_BLK); /* the model closes the record itself */
}

long qsha_compress(uint32_t state[8], uint64_t* nblk, const uint8_t block[64]) {
    run_t r = {&tls_session, 0, 0, 0, {0}, 0, 0};
    resume(&r, state, *nblk);
    if (r.pred != QS_TOK_IN) return -1;
    long n = drain(&r, block, 64, 0);
    if (n < 0) return n;
    memcpy(state, r.cv, sizeof r.cv);
    *nblk = r.n;
    return n;
}

long qsha_finish(const uint32_t state[8], uint64_t nblk, const uint8_t* tail, size_t len, uint8_t out[32]) {
    if (len >= 64) return -1;
    run_t r = {&tls_session, 0, 0, 0, {0}, 0, 0};
    resume(&r, state, nblk);
    if (r.pred != QS_TOK_IN) return -1;
    long n = drain(&r, tail, len, 1);
    if (n < 0) return n;
    put_digest(r.cv, out);
    return n;
}
