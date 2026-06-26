#include "micro_kernels.h"

/*< Fully-integer (fixed-point) PatchEmbed LayerNorm for v9. Reads the int8 conv's int32 accumulator,
 *  folds the per-channel dequant, normalises over C, and runs mean/variance/rsqrt/affine entirely in
 *  integer (int64) - no f32/double in the datapath. norm1 emits int8 (-> int8 GELU); norm2 emits f32
 *  (the only f32, at the very output, for the f32 encoder residual). Q-formats LN_* in micro_kernels.h.
 *  rsqrt is the v8 range-reduced PWL in fixed point, returning a full-precision mantissa + exponent k
 *  (so it stays accurate across the whole variance range). Proven in /tmp/v9_intexact.py
 *  (norm2 ~64 dB, norm1 int8 ~50 dB floor, all stages int64-safe). >*/

/* 1/sqrt(real_t), real_t = t/2^(2F), t>0 int64: returns rm (Q.LN_QG mantissa), *kout = exponent k.
 * inv_std_real = (rm/2^LN_QG) * 2^(F-k).  m4 in [1,4): m4*2^LN_QM = t>>(2k-LN_QM). */
static int64_t ln_int_rsqrt(int64_t t, int *kout,
        const int32_t *knot, const int32_t *sl, const int32_t *ic, uint32_t nseg)
{
    const int p  = 63 - __builtin_clzll((uint64_t)t);     /* floor(log2 t) */
    const int k  = p >> 1;
    const int sh = 2 * k - (int)LN_QM;
    const int64_t m4q = (sh >= 0) ? (t >> sh) : (t << (-sh));
    uint32_t s = 0U;
    while (s + 1U < nseg && m4q >= (int64_t)knot[s + 1U]) ++s;
    const int64_t r = (int64_t)sl[s] * m4q + ((int64_t)ic[s] << LN_QM);   /* Q.(LN_QC+LN_QM) */
    *kout = k;
    return r >> (LN_QC + LN_QM - LN_QG);                                  /* Q.LN_QG */
}

/* core: yq_out[c] (Q.LN_QY) for column l. dbuf/yq_out are caller int64[C] scratch (C<=16). */
static void ln_core_col(const int32_t *acc, uint32_t C, uint32_t L, uint32_t l, uint32_t log2C,
        const int32_t *deq_mul, int deq_sh, const int32_t *gamma_q,
        const int32_t *knot, const int32_t *sl, const int32_t *ic, uint32_t nseg,
        int64_t *dbuf, int64_t *yq_out)
{
    int64_t xqbuf[16];
    int64_t sum = 0;
    for (uint32_t c = 0; c < C; ++c) {
        const int64_t xq = ((int64_t)acc[(size_t)c * L + l] * deq_mul[c]) >> deq_sh;  /* Q.F */
        xqbuf[c] = xq; sum += xq;
    }
    const int64_t mean = sum >> log2C;
    int64_t ss = 0;
    for (uint32_t c = 0; c < C; ++c) { const int64_t d = xqbuf[c] - mean; dbuf[c] = d; ss += d * d; }
    const int64_t V = (ss >> log2C) + LN_EPSP;                            /* Q.2F + eps' */
    int k;
    const int64_t rm = ln_int_rsqrt(V, &k, knot, sl, ic, nseg);           /* Q.LN_QG */
    const int shn = (int)LN_QG + k - (int)LN_QN;
    for (uint32_t c = 0; c < C; ++c) {
        const int64_t dr = dbuf[c] * rm;
        const int64_t n  = (shn >= 0) ? (dr >> shn) : (dr << (-shn));     /* Q.LN_QN */
        const int64_t y  = n * gamma_q[c];                               /* Q.(LN_QN+LN_QC) */
        yq_out[c] = y >> (LN_QN + LN_QC - LN_QY);                         /* Q.LN_QY */
    }
}

/* norm1: int32 acc -> int8 at scale S_ln (output quant + beta folded into amul/ash/bfix). */
void layernorm_int_q_to_int8(const int32_t *acc, int8_t *y,
        const int32_t *deq_mul, int deq_sh, const int32_t *gamma_q,
        int32_t amul, int ash, const int32_t *bfix,
        const int32_t *knot, const int32_t *sl, const int32_t *ic, uint32_t nseg,
        uint32_t C, uint32_t L)
{
    if (C == 0U || L == 0U) return;
    const uint32_t log2C = (uint32_t)__builtin_ctz(C);
    const int64_t am = amul;
    const int64_t rnd = (int64_t)1 << (ash - 1);
    int64_t dbuf[16], yq[16];
    for (uint32_t l = 0; l < L; ++l) {
        ln_core_col(acc, C, L, l, log2C, deq_mul, deq_sh, gamma_q, knot, sl, ic, nseg, dbuf, yq);
        for (uint32_t c = 0; c < C; ++c) {
            int64_t q = (yq[c] * am + bfix[c] + rnd) >> ash;
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            y[(size_t)c * L + l] = (int8_t)q;
        }
    }
}

/* norm2: int32 acc -> f32 (output dequant = >>Q.LN_QY to float + beta). */
void layernorm_int_q_to_f32(const int32_t *acc, float *y,
        const int32_t *deq_mul, int deq_sh, const int32_t *gamma_q, const float *beta,
        const int32_t *knot, const int32_t *sl, const int32_t *ic, uint32_t nseg,
        uint32_t C, uint32_t L)
{
    if (C == 0U || L == 0U) return;
    const uint32_t log2C = (uint32_t)__builtin_ctz(C);
    const float inv_qy = 1.0f / (float)(1U << LN_QY);
    int64_t dbuf[16], yq[16];
    for (uint32_t l = 0; l < L; ++l) {
        ln_core_col(acc, C, L, l, log2C, deq_mul, deq_sh, gamma_q, knot, sl, ic, nseg, dbuf, yq);
        for (uint32_t c = 0; c < C; ++c)
            y[(size_t)c * L + l] = (float)yq[c] * inv_qy + beta[c];
    }
}
