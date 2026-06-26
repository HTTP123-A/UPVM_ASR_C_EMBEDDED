/* Split from sources/sq_int8_v0/micro_kernels.c
 * v10: AVX2 vectorization. BIT-EXACT to the scalar v9 kernel.
 * Depthwise has no cross-channel contraction, so VNNI is not the right tool; instead we
 * accumulate each output row in place over the contiguous width W (stride=1, same-pad):
 * per (kh,kw) tap, add tap * (a shifted slice of the input row) to the output row, with
 * SIMD widening multiply. Integer addition is associative, so reordering the k*k taps from
 * "per output pixel" to "per tap across the row" yields the identical int32 result.
 * Falls back to the original scalar body when not compiled for AVX2.
 */

#include "micro_kernels.h"
#if defined(__AVX2__)
#include <immintrin.h>
#include <string.h>
#endif

/* <2. Depth-wise Convolution 2D INT8 >
 * Depthwise, N=1, stride=1, same-padding (pad = kernel_size/2).
 * x:      [C, H, W]       int8
 * weight: [C, k, k]       int8
 * bias:   [C]             int32 (pre-scaled) or NULL
 * y:      [C, H, W]       int32 — caller applies dequant_int32_to_f32_perchannel
 *
 * H_out = H, W_out = W (same-pad, stride=1).
 */
void depthwise_conv2d_int8(const int8_t  *restrict x,
                            const int8_t  *restrict weight,
                            const int32_t *restrict bias,
                            int32_t       *restrict y,
                            uint32_t C,
                            uint32_t H,
                            uint32_t W,
                            uint32_t kernel_size)
{
    const uint32_t pad = kernel_size / 2;   /* same-pad for stride=1 */
    const uint32_t K2  = kernel_size * kernel_size;
    const uint32_t HW  = H * W;

#if defined(__AVX2__)
    for (uint32_t c = 0; c < C; ++c) {
        const int8_t  *x_c = x      + (size_t)c * HW;
        const int8_t  *k_c = weight + (size_t)c * K2;
        int32_t       *y_c = y      + (size_t)c * HW;
        const int32_t  b   = bias ? bias[c] : 0;

        for (uint32_t ho = 0; ho < H; ++ho) {
            const int32_t h_base = (int32_t)ho - (int32_t)pad;
            int32_t *restrict yrow = y_c + (size_t)ho * W;

            /* init row to bias */
            {
                const __m256i vb = _mm256_set1_epi32(b);
                uint32_t w = 0;
                for (; w + 8u <= W; w += 8u) _mm256_storeu_si256((__m256i *)(yrow + w), vb);
                for (; w < W; ++w) yrow[w] = b;
            }

            for (uint32_t kh = 0; kh < kernel_size; ++kh) {
                const int32_t ih = h_base + (int32_t)kh;
                if ((uint32_t)ih >= H) continue;
                const int8_t *xih = x_c + (size_t)(uint32_t)ih * W;

                for (uint32_t kw = 0; kw < kernel_size; ++kw) {
                    const int s   = (int)kw - (int)pad;     /* input col offset = wo + s */
                    const int32_t tap = (int32_t)k_c[kh * kernel_size + kw];

                    /* valid output cols: 0 <= wo + s < W */
                    const uint32_t lo = (s < 0) ? (uint32_t)(-s) : 0u;
                    uint32_t hi;
                    if (s > 0) { const uint32_t su = (uint32_t)s; hi = (W > su) ? (W - su) : 0u; }
                    else       { hi = W; }
                    if (lo >= hi) continue;

                    const __m256i vt = _mm256_set1_epi32(tap);
                    uint32_t w = lo;
                    for (; w + 8u <= hi; w += 8u) {
                        int64_t tmp;
                        memcpy(&tmp, xih + (int)w + s, 8);                 /* 8 contiguous int8 */
                        const __m256i xi   = _mm256_cvtepi8_epi32(_mm_cvtsi64_si128(tmp));
                        const __m256i prod = _mm256_mullo_epi32(xi, vt);
                        __m256i acc = _mm256_loadu_si256((__m256i *)(yrow + w));
                        acc = _mm256_add_epi32(acc, prod);
                        _mm256_storeu_si256((__m256i *)(yrow + w), acc);
                    }
                    for (; w < hi; ++w)
                        yrow[w] += tap * (int32_t)xih[(int)w + s];
                }
            }
        }
    }
#else
    for (uint32_t c = 0; c < C; ++c) {
        const int8_t  *x_c = x      + (size_t)c * HW;
        const int8_t  *k_c = weight + (size_t)c * K2;
        int32_t       *y_c = y      + (size_t)c * HW;
        const int32_t  b   = bias ? bias[c] : 0;

        for (uint32_t ho = 0; ho < H; ++ho) {
            const int32_t h_base = (int32_t)ho - (int32_t)pad;

            for (uint32_t wo = 0; wo < W; ++wo) {
                const int32_t w_base = (int32_t)wo - (int32_t)pad;
                int32_t acc = b;

                for (uint32_t kh = 0; kh < kernel_size; ++kh) {
                    const int32_t ih = h_base + (int32_t)kh;
                    if ((uint32_t)ih >= H) continue;

                    const uint32_t x_row = (uint32_t)ih * W;
                    const uint32_t k_row = kh * kernel_size;

                    for (uint32_t kw = 0; kw < kernel_size; ++kw) {
                        const int32_t iw = w_base + (int32_t)kw;
                        if ((uint32_t)iw >= W) continue;
                        acc += (int32_t)x_c[x_row + (uint32_t)iw]
                             * (int32_t)k_c[k_row + kw];
                    }
                }
                y_c[ho * W + wo] = acc;
            }
        }
    }
#endif
}
