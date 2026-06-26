/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
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
}
