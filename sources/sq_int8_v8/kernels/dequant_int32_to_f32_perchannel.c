/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
/* 1. Per-channel dequant: int32 accumulator → float
 *Used after pointwise_conv2d_int8, depthwise_conv2d_int8, conv2d_int8.
 * Each output channel oc has its own combined scale:
 *   scale_oc = a_scale_in * w_scale[oc]
 * y_f32[oc, p] = y_i32[oc, p] * scale_oc
 */
void dequant_int32_to_f32_perchannel(const int32_t *x,         /* [C, L] */
                                      float         *y,         /* [C, L] */
                                      uint32_t       C,
                                      uint32_t       L,
                                      const float   *scales)    /* [C], one per output channel */
{
    for (uint32_t c = 0; c < C; ++c) {
        const float sc = scales[c];
        const int32_t *xc = x + (size_t)c * L;
        float         *yc = y + (size_t)c * L;
        for (uint32_t l = 0; l < L; ++l) {
            yc[l] = (float)xc[l] * sc;
        }
    }
}
