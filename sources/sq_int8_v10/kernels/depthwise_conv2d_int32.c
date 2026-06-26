/* Depthwise Conv2d on the int32 accumulator (sq_int8_v2) - the int32 sibling of
 * depthwise_conv2d_f32. Same NCHW (N=1) layout, same H_out/W_out formula, same
 * bias==NULL => 0 convention, but pure int32 so the encoder MLP "SumPool" can run
 * on the raw conv accumulator BEFORE dequant (replacing v1's dequant -> f32 sumpool,
 * exactly as the f32 build used depthwise_conv2d_f32 with mlp_sumpool_w).
 *
 * weight == NULL is treated as unit taps (all 1) -> a pure window sum. The
 * checkpoint's mlp_sumpool_w is exactly 1.0 everywhere (verified for every
 * pvss_ds_weight_enc{0,1,2}_{mag,pha}), so the v2 main MLP sumpool calls this with
 * weight=NULL and is bit-exact to summing the 4 neighbours. The per-output-channel
 * dequant scale is > 0, so it factors out of the sum and the pool is valid in int32
 * (no overflow: a few-million sum vs INT32_MAX). NOTE: not bit-identical to the f32
 * sumpool -- v2 sums in int32 then rounds once at dequant, whereas the f32 path
 * rounds each product; v2 is the more accurate of the two.
 *
 * In-place is permitted (y may alias x) for downsampling configs (stride >=
 * kernel_size), as used here for the 2x2 stride-2 sumpool: the output is strictly
 * compacted and produced in increasing index order, so a written cell is never an
 * input still needed by a later output. (Hence no `restrict` on x / y.)
 *
 * x: [C,H,W]  weight: [C,k,k] or NULL  bias: [C] or NULL  y: [C,H_out,W_out]
 *   H_out = floor((H + 2*pad - kernel_size) / stride) + 1   (W_out likewise)
 *
 * v10: AVX2 vectorization of the stride==1, non-aliasing case only (per-row accumulate
 * over the contiguous output width). The real use here is the stride-2 2x2 sumpool, whose
 * output columns are strided in the input (and which may alias y==x); that case keeps the
 * exact scalar path. Both paths are bit-exact to v9.
 */

#include "micro_kernels.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* <3. Depth-wise convolution 2d on int32 accumulator (weight NULL => unit/sum) > */
void depthwise_conv2d_int32(const int32_t *x,
                            const int32_t *weight,
                            const int32_t *bias,
                            int32_t *y,
                            uint32_t C,
                            uint32_t H,
                            uint32_t W,
                            uint32_t kernel_size,
                            uint32_t stride,
                            uint32_t pad)
{
    if (stride == 0U) return;

    const int32_t h_numer = (int32_t)H + 2 * (int32_t)pad - (int32_t)kernel_size;
    const int32_t w_numer = (int32_t)W + 2 * (int32_t)pad - (int32_t)kernel_size;
    if (h_numer < 0 || w_numer < 0) return;

    const uint32_t H_out = (uint32_t)(h_numer / (int32_t)stride + 1);
    const uint32_t W_out = (uint32_t)(w_numer / (int32_t)stride + 1);
    if (H_out == 0U || W_out == 0U) return;

    const uint32_t HW  = H * W;
    const uint32_t K2  = kernel_size * kernel_size;
    const uint32_t HWo = H_out * W_out;

#if defined(__AVX2__)
    if (stride == 1U) {                 /* contiguous output width -> SIMD over W_out */
        for (uint32_t c = 0; c < C; ++c) {
            const int32_t *x_c = x + (size_t)c * HW;
            const int32_t *k_c = weight ? weight + (size_t)c * K2 : NULL;
            int32_t       *y_c = y + (size_t)c * HWo;
            const int32_t  b   = bias ? bias[c] : 0;

            for (uint32_t ho = 0; ho < H_out; ++ho) {
                const int32_t h_base = (int32_t)ho - (int32_t)pad;
                int32_t *yrow = y_c + (size_t)ho * W_out;

                {   /* init row to bias */
                    const __m256i vb = _mm256_set1_epi32(b);
                    uint32_t w = 0;
                    for (; w + 8u <= W_out; w += 8u) _mm256_storeu_si256((__m256i *)(yrow + w), vb);
                    for (; w < W_out; ++w) yrow[w] = b;
                }

                for (uint32_t kh = 0; kh < kernel_size; ++kh) {
                    const int32_t ih = h_base + (int32_t)kh;
                    if ((uint32_t)ih >= H) continue;
                    const int32_t *xih = x_c + (size_t)(uint32_t)ih * W;
                    const uint32_t k_row = kh * kernel_size;

                    for (uint32_t kw = 0; kw < kernel_size; ++kw) {
                        const int   s   = (int)kw - (int)pad;     /* input col = wo + s */
                        const int32_t tap = k_c ? k_c[k_row + kw] : 1;

                        const uint32_t lo = (s < 0) ? (uint32_t)(-s) : 0u;
                        uint32_t hi = W_out;
                        if (s > 0) { const uint32_t su = (uint32_t)s; hi = (W > su) ? (W - su) : 0u; }
                        else       { hi = (W_out < W) ? W_out : W; }
                        if (hi > W_out) hi = W_out;
                        if (lo >= hi) continue;

                        const __m256i vt = _mm256_set1_epi32(tap);
                        uint32_t w = lo;
                        for (; w + 8u <= hi; w += 8u) {
                            const __m256i xi   = _mm256_loadu_si256((const __m256i *)(xih + (int)w + s));
                            const __m256i prod = _mm256_mullo_epi32(xi, vt);
                            __m256i acc = _mm256_loadu_si256((__m256i *)(yrow + w));
                            acc = _mm256_add_epi32(acc, prod);
                            _mm256_storeu_si256((__m256i *)(yrow + w), acc);
                        }
                        for (; w < hi; ++w)
                            yrow[w] += xih[(int)w + s] * tap;
                    }
                }
            }
        }
        return;
    }
#endif

    /* scalar: general stride (incl. the stride-2 sumpool, y may alias x), and non-AVX2 */
    for (uint32_t c = 0; c < C; ++c) {
        const int32_t *x_c = x + (size_t)c * HW;
        const int32_t *k_c = weight ? weight + (size_t)c * K2 : NULL;
        int32_t *y_c = y + (size_t)c * HWo;
        const int32_t b = bias ? bias[c] : 0;

        for (uint32_t ho = 0; ho < H_out; ++ho) {
            const int32_t h_base = (int32_t)(ho * stride) - (int32_t)pad;

            for (uint32_t wo = 0; wo < W_out; ++wo) {
                const int32_t w_base = (int32_t)(wo * stride) - (int32_t)pad;
                int32_t acc = b;

                for (uint32_t kh = 0; kh < kernel_size; ++kh) {
                    const int32_t ih = h_base + (int32_t)kh;
                    if ((uint32_t)ih >= H) continue;

                    const uint32_t x_row = (uint32_t)ih * W;
                    const uint32_t k_row = kh * kernel_size;

                    for (uint32_t kw = 0; kw < kernel_size; ++kw) {
                        const int32_t iw = w_base + (int32_t)kw;
                        if ((uint32_t)iw >= W) continue;

                        const int32_t wv = k_c ? k_c[k_row + kw] : 1;
                        acc += x_c[x_row + (uint32_t)iw] * wv;
                    }
                }

                y_c[ho * W_out + wo] = acc;
            }
        }
    }
}
