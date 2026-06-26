/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
/*< 4. Conv 2D - INT8 >
 *  W8A8 symmetric, no zero-point (matches checkpoint format).
 *  Outputs int32 raw accumulator — caller applies dequant scale.
 *
 *  Same signature pattern as pointwise_conv2d_int8 / depthwise_conv2d_int8:
 *    x      : int8,  [C_IN,  H_IN,  W_IN]
 *    weight : int8,  [C_OUT, C_IN,  K_SIZE, K_SIZE]   (OC-major)
 *    bias   : int32, [C_OUT] or NULL  (pre-scaled: bias_i32 = bias_f32 / (a_scale * w_scale))
 *    y      : int32, [C_OUT, H_OUT, W_OUT]
 */
void conv2d_int8(const int8_t  *x,
                 const int8_t  *weight,
                 const int32_t *bias,
                 int32_t       *y,
                 uint32_t C_IN,
                 uint32_t H_IN,
                 uint32_t W_IN,
                 uint32_t C_OUT,
                 uint32_t H_OUT,
                 uint32_t W_OUT,
                 uint32_t K_SIZE,
                 uint32_t STRIDE,
                 uint32_t PAD)
{
    for (uint32_t oc = 0; oc < C_OUT; ++oc) {
        const int8_t *w_oc = weight + (size_t)oc * C_IN * K_SIZE * K_SIZE;

        for (uint32_t oh = 0; oh < H_OUT; ++oh) {
            const int32_t ih0 = (int32_t)(oh * STRIDE) - (int32_t)PAD;

            for (uint32_t ow = 0; ow < W_OUT; ++ow) {
                const int32_t iw0 = (int32_t)(ow * STRIDE) - (int32_t)PAD;
                int32_t acc = bias ? bias[oc] : 0;

                for (uint32_t ic = 0; ic < C_IN; ++ic) {
                    const int8_t *w_oc_ic = w_oc + (size_t)ic * K_SIZE * K_SIZE;

                    for (uint32_t kh = 0; kh < K_SIZE; ++kh) {
                        const int32_t ih = ih0 + (int32_t)kh;
                        if ((uint32_t)ih >= H_IN) continue;

                        const size_t x_row = ((size_t)ic * H_IN + (uint32_t)ih) * W_IN;
                        const size_t w_row = (size_t)kh * K_SIZE;

                        for (uint32_t kw = 0; kw < K_SIZE; ++kw) {
                            const int32_t iw = iw0 + (int32_t)kw;
                            if ((uint32_t)iw >= W_IN) continue;

                            acc += (int32_t)x[x_row + (uint32_t)iw]
                                 * (int32_t)w_oc_ic[w_row + kw];
                        }
                    }
                }

                y[((size_t)oc * H_OUT + oh) * W_OUT + ow] = acc;
            }
        }
    }
}
