/* Split from sources/sq_int8_v0/micro_kernels.c
 * v10: AVX-VNNI vectorization. BIT-EXACT to the scalar v9 kernel.
 * Same +128-offset / weight-sum-compensation identity as pointwise_conv2d_int8;
 * the only difference is the virtual [x_inout | x_res] concatenation (2*C_IN channels).
 * Falls back to the original scalar body when not compiled with -mavxvnni.
 */

#include "micro_kernels.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <string.h>

/* <3. Half-split input point-wise convolution INT8 - Pytorch equivalent: skip_handler >
 * Concatenates [x_inout | x_res] virtually (C_IN_TOTAL = 2*C_IN) then 1x1 conv.
 * x_inout: [C_IN, H, W]          int8  (first C_IN input channels)
 * x_res:   [C_IN, H, W]          int8  (second C_IN input channels)
 * weight:  [C_OUT, 2*C_IN, 1, 1] int8  (OC-major, flattened to [C_OUT, 2*C_IN])
 * bias:    [C_OUT]                int32 (pre-scaled) or NULL
 * y:       [C_OUT, H, W]          int32 — caller applies dequant_int32_to_f32_perchannel
 */
void pointwise_conv2d_split2_int8(const int8_t  *restrict x_inout,
                                  const int8_t  *restrict x_res,
                                  const int8_t  *restrict weight,
                                  const int32_t *restrict bias,
                                  int32_t       *restrict y,
                                  uint32_t C_IN,
                                  uint32_t H,
                                  uint32_t W,
                                  uint32_t C_OUT)
{
    const uint32_t L          = H * W;
    const uint32_t C_IN_TOTAL = 2U * C_IN;

#if defined(__AVXVNNI__)
    const uint32_t ngrp = (C_IN_TOTAL + 3u) / 4u;   /* groups of 4 over the concatenated axis */

    /* base[o] = bias[o] - 128 * sum_c w[o,c]  over all 2*C_IN channels */
    int32_t base[C_OUT];
    for (uint32_t o = 0; o < C_OUT; ++o) {
        const int8_t *wrow = weight + (size_t)o * C_IN_TOTAL;
        int32_t wsum = 0;
        for (uint32_t c = 0; c < C_IN_TOTAL; ++c) wsum += (int32_t)wrow[c];
        base[o] = (bias ? bias[o] : 0) - 128 * wsum;
    }

    const __m256i off = _mm256_set1_epi8((char)0x80);
    __m256i xt[ngrp];

    uint32_t pix = 0;
    for (; pix + 8u <= L; pix += 8u) {
        for (uint32_t g = 0; g < ngrp; ++g) {
            const uint32_t c0 = g * 4u;
            __m128i r[4];
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t ct = c0 + j;             /* index into virtual 2*C_IN axis */
                if (ct < C_IN_TOTAL) {
                    const int8_t *src = (ct < C_IN) ? (x_inout + (size_t)ct * L)
                                                    : (x_res   + (size_t)(ct - C_IN) * L);
                    int64_t v;
                    memcpy(&v, src + pix, 8);
                    r[j] = _mm_cvtsi64_si128(v);
                } else {
                    r[j] = _mm_setzero_si128();
                }
            }
            const __m128i a01 = _mm_unpacklo_epi8(r[0], r[1]);
            const __m128i a23 = _mm_unpacklo_epi8(r[2], r[3]);
            const __m128i lo  = _mm_unpacklo_epi16(a01, a23);
            const __m128i hi  = _mm_unpackhi_epi16(a01, a23);
            xt[g] = _mm256_add_epi8(_mm256_set_m128i(hi, lo), off);
        }

        for (uint32_t o = 0; o < C_OUT; ++o) {
            const int8_t *wrow = weight + (size_t)o * C_IN_TOTAL;
            __m256i acc = _mm256_set1_epi32(base[o]);
            for (uint32_t g = 0; g < ngrp; ++g) {
                const uint32_t c0 = g * 4u;
                int32_t w4 = 0;
                uint8_t *wp = (uint8_t *)&w4;
                for (uint32_t j = 0; j < 4u && c0 + j < C_IN_TOTAL; ++j)
                    wp[j] = (uint8_t)wrow[c0 + j];
                acc = _mm256_dpbusd_avx_epi32(acc, xt[g], _mm256_set1_epi32(w4));
            }
            _mm256_storeu_si256((__m256i *)(y + (size_t)o * L + pix), acc);
        }
    }

    /* tail pixels (L % 8) — scalar, identical math */
    for (; pix < L; ++pix) {
        for (uint32_t o = 0; o < C_OUT; ++o) {
            const int8_t *wrow = weight + (size_t)o * C_IN_TOTAL;
            int32_t acc = bias ? bias[o] : 0;
            for (uint32_t c = 0; c < C_IN; ++c)
                acc += (int32_t)wrow[c]        * (int32_t)x_inout[(size_t)c * L + pix];
            for (uint32_t c = 0; c < C_IN; ++c)
                acc += (int32_t)wrow[C_IN + c] * (int32_t)x_res[(size_t)c * L + pix];
            y[(size_t)o * L + pix] = acc;
        }
    }
#else
    for (uint32_t o = 0; o < C_OUT; ++o) {
        int32_t       *restrict yo   = y      + (size_t)o * L;
        const int8_t  *restrict wrow = weight + (size_t)o * C_IN_TOTAL;
        const int32_t b = bias ? bias[o] : 0;

        for (uint32_t pix = 0; pix < L; ++pix) {
            int32_t acc = b;

            /* first half: channels from x_inout */
            for (uint32_t c = 0; c < C_IN; ++c) {
                acc += (int32_t)wrow[c]        * (int32_t)x_inout[(size_t)c * L + pix];
            }
            /* second half: channels from x_res */
            for (uint32_t c = 0; c < C_IN; ++c) {
                acc += (int32_t)wrow[C_IN + c] * (int32_t)x_res[(size_t)c * L + pix];
            }

            yo[pix] = acc;
        }
    }
#endif
}
