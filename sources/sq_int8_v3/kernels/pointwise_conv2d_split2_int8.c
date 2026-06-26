/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
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
}
