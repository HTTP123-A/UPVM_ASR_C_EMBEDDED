/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
/* <1. Point-wise Convolution INT8 >
 * x:      [C_IN, H, W]         int8
 * weight: [C_OUT, C_IN]        int8  (OC-major, same layout as F32 version)
 * bias:   [C_OUT]              int32 (pre-scaled: round(bias_f32 / (a_scale * w_scale[oc]))) or NULL
 * y:      [C_OUT, H, W]        int32 — caller applies dequant_int32_to_f32_perchannel
 *
 * Also used for linear layers: call with H=1, W=1 (L=1).
 * W8A8 symmetric, w_zp=0 in checkpoint — no zero-point correction in inner loop.
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
}
