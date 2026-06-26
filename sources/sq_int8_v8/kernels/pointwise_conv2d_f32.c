/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
/* <1. Point-wise Convolution - Pytorch equivalent: in_proj, out_proj, conv1 (mlp), conv2 (mlp) > 
 * Shape: [C_IN, H, W] --> [C_OUT, H, W]
 *
 * Notes: This is a 2D convolution with kernel size 1x1 (point-wise)
 */
void pointwise_conv2d_f32(const float *restrict x,
                        const float *restrict weight,
                        const float *restrict bias,
                        float *restrict y,
                        uint32_t C_IN,
                        uint32_t H,
                        uint32_t W,
                        uint32_t C_OUT)
{
    const uint32_t L = H * W;

    for (uint32_t o = 0; o < C_OUT; ++o) {
        float *restrict yo = y + (size_t)o * L;
        const float *restrict wrow = weight + (size_t)o * C_IN;
        const float b = bias ? bias[o] : 0.0f;

        for (uint32_t pix = 0; pix < L; ++pix) {
            float acc = b;
            for (uint32_t c = 0; c < C_IN; ++c) {
                acc += wrow[c] * x[(size_t)c * L + pix];
            }
            yo[pix] = acc;
        }
    }
}
