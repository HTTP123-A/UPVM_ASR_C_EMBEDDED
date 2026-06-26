/* Split from sources/sq_int8_v0/micro_kernels.c
 * v10: AVX-VNNI (vpdpbusd, 256-bit VEX) vectorization. BIT-EXACT to the scalar v9
 * kernel — same int32 accumulator, no numeric change (see identity below).
 * Falls back to the original scalar body when not compiled with -mavxvnni.
 */

#include "micro_kernels.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <string.h>

/* <1. Point-wise Convolution INT8 >
 * x:      [C_IN, H, W]         int8
 * weight: [C_OUT, C_IN]        int8  (OC-major, same layout as F32 version)
 * bias:   [C_OUT]              int32 (pre-scaled: round(bias_f32 / (a_scale * w_scale[oc]))) or NULL
 * y:      [C_OUT, H, W]        int32 — caller applies dequant_int32_to_f32_perchannel
 *
 * Also used for linear layers: call with H=1, W=1 (L=1).
 * W8A8 symmetric, w_zp=0 in checkpoint — no zero-point correction in inner loop.
 *
 * VNNI note: y[o,pix] = sum_c w[o,c]*x[c,pix]. The activation layout is [C_IN,L] so the
 * contraction dim C_IN is strided by L; vpdpbusd needs 4 contraction bytes contiguous per
 * lane, so we transpose an 8-pixel block into an on-stack tile [C_IN/4][8 pixels][4 ch].
 * vpdpbusd is u8 x s8, but activations are signed, so we bias x by +128 (-> uint8) and
 * subtract the per-output compensation 128*sum_c w[o,c]:
 *     sum_c (x[c]+128)*w[o,c]  -  128*sum_c w[o,c]  =  sum_c x[c]*w[o,c]
 * => integer-exact, identical result to the scalar accumulator. No model buffer/API change.
 */
void pointwise_conv2d_int8(const int8_t  *restrict x,
                           const int8_t  *restrict weight,
                           const int32_t *restrict bias,
                           int32_t       *restrict y,
                           uint32_t C_IN,
                           uint32_t H,
                           uint32_t W,
                           uint32_t C_OUT)
{
    const uint32_t L = H * W;

#if defined(__AVXVNNI__)
    const uint32_t ngrp = (C_IN + 3u) / 4u;     /* channel groups of 4 (zero-padded tail) */

    /* base[o] = bias[o] - 128 * sum_c w[o,c]  (offset compensation, computed once) */
    int32_t base[C_OUT];
    for (uint32_t o = 0; o < C_OUT; ++o) {
        const int8_t *wrow = weight + (size_t)o * C_IN;
        int32_t wsum = 0;
        for (uint32_t c = 0; c < C_IN; ++c) wsum += (int32_t)wrow[c];
        base[o] = (bias ? bias[o] : 0) - 128 * wsum;
    }

    const __m256i off = _mm256_set1_epi8((char)0x80);   /* add 0x80 == +128 bias -> uint8 */
    __m256i xt[ngrp];                                   /* packed activation tile, reused */

    uint32_t pix = 0;
    for (; pix + 8u <= L; pix += 8u) {
        /* Pack: xt[g] lane p (pixel pix+p) = bytes [x[4g+0..3, pix+p] + 128]. */
        for (uint32_t g = 0; g < ngrp; ++g) {
            const uint32_t c0 = g * 4u;
            __m128i r[4];
            for (uint32_t j = 0; j < 4u; ++j) {
                if (c0 + j < C_IN) {
                    int64_t v;
                    memcpy(&v, x + (size_t)(c0 + j) * L + pix, 8);   /* 8 contiguous pixels */
                    r[j] = _mm_cvtsi64_si128(v);
                } else {
                    r[j] = _mm_setzero_si128();                      /* zero-pad channel */
                }
            }
            const __m128i a01 = _mm_unpacklo_epi8(r[0], r[1]);
            const __m128i a23 = _mm_unpacklo_epi8(r[2], r[3]);
            const __m128i lo  = _mm_unpacklo_epi16(a01, a23);        /* pixels 0..3 */
            const __m128i hi  = _mm_unpackhi_epi16(a01, a23);        /* pixels 4..7 */
            xt[g] = _mm256_add_epi8(_mm256_set_m128i(hi, lo), off);  /* +128 -> uint8 */
        }

        for (uint32_t o = 0; o < C_OUT; ++o) {
            const int8_t *wrow = weight + (size_t)o * C_IN;
            __m256i acc = _mm256_set1_epi32(base[o]);
            for (uint32_t g = 0; g < ngrp; ++g) {
                const uint32_t c0 = g * 4u;
                int32_t w4 = 0;                                      /* 4 signed weight bytes */
                uint8_t *wp = (uint8_t *)&w4;
                for (uint32_t j = 0; j < 4u && c0 + j < C_IN; ++j)
                    wp[j] = (uint8_t)wrow[c0 + j];                   /* tail bytes stay 0 */
                acc = _mm256_dpbusd_avx_epi32(acc, xt[g], _mm256_set1_epi32(w4));
            }
            _mm256_storeu_si256((__m256i *)(y + (size_t)o * L + pix), acc);
        }
    }

    /* tail pixels (L % 8) — scalar, identical math */
    for (; pix < L; ++pix) {
        for (uint32_t o = 0; o < C_OUT; ++o) {
            const int8_t *wrow = weight + (size_t)o * C_IN;
            int32_t acc = bias ? bias[o] : 0;
            for (uint32_t c = 0; c < C_IN; ++c)
                acc += (int32_t)wrow[c] * (int32_t)x[(size_t)c * L + pix];
            y[(size_t)o * L + pix] = acc;
        }
    }
#else
    for (uint32_t o = 0; o < C_OUT; ++o) {
        int32_t       *restrict yo   = y      + (size_t)o * L;
        const int8_t  *restrict wrow = weight + (size_t)o * C_IN;
        const int32_t b = bias ? bias[o] : 0;

        for (uint32_t pix = 0; pix < L; ++pix) {
            int32_t acc = b;
            for (uint32_t c = 0; c < C_IN; ++c) {
                acc += (int32_t)wrow[c] * (int32_t)x[(size_t)c * L + pix];
            }
            yo[pix] = acc;
        }
    }
#endif
}
