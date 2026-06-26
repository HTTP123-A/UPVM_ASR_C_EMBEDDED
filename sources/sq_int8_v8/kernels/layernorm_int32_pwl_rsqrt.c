#include "micro_kernels.h"

/*< Fused int32 dequant + LayerNorm with a range-reduced piece-wise-linear rsqrt (v8).
 *
 *  Replaces the (dequant_int32_to_f32_perchannel -> layernorm_f32) pair for the PatchEmbed
 *  norms: it reads the int8 conv's int32 accumulator directly, applies the per-channel dequant
 *  scale, normalizes over C, and approximates inv_std = 1/sqrt(var+eps) with a LUT + linear
 *  interpolation on the variance - no sqrtf and no divide, the FPGA/ASIC-friendly O(1) form.
 *
 *  rsqrt(t), t = var + eps > 0, via range reduction (frexpf == exponent read / leading-zero
 *  count on hardware; ldexpf == a shift):
 *      frexpf(t) -> t = mant * 2^e2, mant in [0.5, 1)
 *      normalise to m' in [1,4) with an EVEN exponent 2k:
 *          e2 odd : m' = mant*2, k = (e2-1)/2      (m' in [1,2))
 *          e2 even: m' = mant*4, k = (e2-2)/2      (m' in [2,4))
 *      rsqrt(t) = 2^-k * rsqrt(m');  rsqrt(m') ~= slope[i]*m' + intercept[i] on segment i.
 *  Because the mantissa is always reduced to [1,4), ONE small table covers every variance
 *  magnitude (no per-layer range calibration). Segment breakpoints are adaptive (denser where
 *  the curvature is high) and the per-segment lines are min-max (Chebyshev) fits - all tuned and
 *  stored by weight_extractor_sq_int8_v8.py.
 *
 *  Only the rsqrt is approximated; the mean/variance (double accumulation) and the dequant
 *  multiply match the f32 path bit-for-bit, so v8 diverges from v7 solely by this PWL rsqrt
 *  (PatchEmbed LayerNorm SNR ~= 64 dB at PWL_RSQRT_SEGMENTS = 10, far above the model floor).
 */
void layernorm_int32_pwl_rsqrt(const int32_t *acc,           /* [C, L] int32 conv accumulator */
                               float         *y,             /* [C, L] f32 output             */
                               const float   *deq,           /* [C] per-channel dequant scale */
                               const float   *gamma,         /* [C] */
                               const float   *beta,          /* [C] */
                               uint32_t       C,
                               uint32_t       L,
                               float          eps,
                               const float   *pwl_knot,      /* [n_seg + 1] breakpoints on [1,4) */
                               const float   *pwl_slope,     /* [n_seg] */
                               const float   *pwl_intercept, /* [n_seg] */
                               uint32_t       n_seg)
{
    if (C == 0U || L == 0U || n_seg == 0U) return;

    const float invC = 1.0f / (float)C;

    for (uint32_t l = 0; l < L; ++l) {
        /* --- E(x),  x[c] = acc[c,l] * deq[c]  (fused dequant) --- */
        double m = 0.0;
        for (uint32_t c = 0; c < C; ++c) {
            m += (double)((float)acc[(size_t)c * L + l] * deq[c]);
        }
        m *= (double)invC;

        /* --- V(x) --- */
        double v = 0.0;
        for (uint32_t c = 0; c < C; ++c) {
            const double d = (double)((float)acc[(size_t)c * L + l] * deq[c]) - m;
            v += d * d;
        }
        v *= (double)invC;

        const float mean = (float)m;

        /* --- inv_std = rsqrt(v + eps) via range-reduced PWL (LUT + linear interp) --- */
        const float t = (float)v + eps;                 /* > 0 since eps > 0 */
        int e2;
        const float mant = frexpf(t, &e2);              /* t = mant * 2^e2, mant in [0.5,1) */
        float mp;
        int   k;
        if (e2 & 1) { mp = mant * 2.0f; k = (e2 - 1) >> 1; }   /* m' in [1,2) */
        else        { mp = mant * 4.0f; k = (e2 - 2) >> 1; }   /* m' in [2,4) */

        uint32_t i = 0U;                                /* segment: largest i with mp >= knot[i] */
        while (i + 1U < n_seg && mp >= pwl_knot[i + 1U]) ++i;
        const float r       = pwl_slope[i] * mp + pwl_intercept[i];   /* rsqrt(m') */
        const float inv_std = ldexpf(r, -k);                          /* * 2^-k    */

        /* --- Weight & Bias --- */
        for (uint32_t c = 0; c < C; ++c) {
            const float xc = (float)acc[(size_t)c * L + l] * deq[c];
            y[(size_t)c * L + l] = (xc - mean) * inv_std * gamma[c] + beta[c];
        }
    }
}
