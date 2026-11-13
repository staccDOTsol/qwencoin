/* qsha: executor of the SHA-256 circuit compiled into Qwen3.5 weights (hashformer).
 *
 * A session is the model's recurrent state: the Gated DeltaNet rows and the conv windows. Feeding a
 * token runs every sublayer of the circuit on it and returns the token the model predicts next.
 * The host only supplies message bytes and END; the IV, padding, length, rounds and digest are
 * the model's own tokens. The arithmetic is the layers' arithmetic on the nonzero weights, and
 * equals the IR evaluator on every residual dim at every position (tests/check_exec.py).
 */
#ifndef QSHA_H
#define QSHA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct qsha qsha;

size_t qsha_sizeof(void);
qsha* qsha_new(void);
void qsha_free(qsha* s);
void qsha_reset(qsha* s);
void qsha_copy(qsha* dst, const qsha* src);
/* Run the circuit on one token; returns the predicted next token (-1: no token has that code). */
int qsha_feed(qsha* s, int token);
const double* qsha_residual(const qsha* s); /* residual of the last token, QS_NDIMS values */
int qsha_ndims(void);

/* SHA-256 of a message, the model driving everything but the bytes. Returns tokens run, <0 on error. */
long qsha_sha256(const uint8_t* msg, size_t len, uint8_t out[32]);
/* The IV as the model emits it after SHA. */
void qsha_iv(uint32_t state[8]);
/* One compression from a midstate (RES): state and block count are updated in place. */
long qsha_compress(uint32_t state[8], uint64_t* nblk, const uint8_t block[64]);
/* Finish a message from a midstate: < 64 tail bytes, then END; the model pads and prints. */
long qsha_finish(const uint32_t state[8], uint64_t nblk, const uint8_t* tail, size_t n, uint8_t out[32]);
/* Tokens the calling thread's model has run since the thread started. */
uint64_t qsha_tokens(void);

#ifdef __cplusplus
}
#endif
#endif
