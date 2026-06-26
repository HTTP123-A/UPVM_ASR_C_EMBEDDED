/* New in sq_int8_v6. */

#include "micro_kernels.h"
/* 3. Per-channel INTEGER REQUANT (fixed-point / dyadic multiplier) — the integer-only
 * sibling of requant_int32_to_int8_perchannel. It maps the int32 accumulator to int8
 * with NO floating point at runtime, replacing the f32 `lrintf((float)acc * scale)` with
 * one integer multiply + one rounding add + one arithmetic right shift:
 *
 *     q[c,l] = clamp( ( (int64)x[c,l] * mul[c] + round_bias ) >> shift[c],  -128, 127 )
 *     round_bias = (shift[c] > 0) ? (1 << (shift[c]-1)) : 0
 *
 * This is the gemmlowp / TFLite "quantized multiplier" technique: the per-channel real
 * scale requant_scale[c] = deq_scale[c] / act_scale_next[c] is approximated by a dyadic
 * rational  mul[c] / 2^shift[c]  (so  acc * scale  ~=  acc * mul >> shift). The pair
 * (mul[c], shift[c]) is precomputed per channel by weight_extractor_sq_int8_v6.py, which
 * searches the shift to minimize the error vs the exact f32 requant (typically the maximal
 * shift that keeps mul < 2^31, i.e. the most multiplier precision).
 *
 * The product is accumulated in int64 (acc < 2^31, mul < 2^31 => product < 2^62, no
 * overflow); the rounding add gives round-half-up, so results differ from the f32 lrintf
 * (round-half-even) only at exact ties, bounded to +-1 LSB. The [-128,127] clamp matches
 * the f32 kernel exactly.
 *
 * x: [C, L] int32   y: [C, L] int8   mul/shift: [C]
 */
void requant_int32_to_int8_perchannel_fixed(const int32_t *x,
                                            int8_t        *y,
                                            uint32_t       C,
                                            uint32_t       L,
                                            const int32_t *mul,
                                            const int32_t *shift)
{
    for (uint32_t c = 0; c < C; ++c) {
        const int64_t m    = (int64_t)mul[c];
        const int32_t sh   = shift[c];
        const int64_t bias = (sh > 0) ? ((int64_t)1 << (sh - 1)) : 0;
        const int32_t *xc  = x + (size_t)c * L;
        int8_t        *yc  = y + (size_t)c * L;

        for (uint32_t l = 0; l < L; ++l) {
            const int64_t p = (int64_t)xc[l] * m + bias;
            int32_t q = (int32_t)((sh > 0) ? (p >> sh) : p);   /* arithmetic shift (signed) */
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            yc[l] = (int8_t)q;
        }
    }
}
