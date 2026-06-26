/* Split from sources/naives_mp/micro_kernels.c */

#include "micro_kernels.h"
/* <2. Depth-wise convolution 2d - Pytorch equivalent: conv2d (ss2d) > */
/* Depthwise Conv2d, NCHW with N=1
 * x:      [C, H, W]
 * weight: [C, k, k]
 * bias:   [C] or NULL
 * y:      [C, H_out, W_out]
 *
 * H_out = floor((H + 2*pad - kernel_size) / stride) + 1
 * W_out = floor((W + 2*pad - kernel_size) / stride) + 1
 */
void depthwise_conv2d_f32(const float *restrict x,
                        const float *restrict weight,
                        const float *restrict bias,
                        float *restrict y,
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

    const uint32_t HW = H * W;
    const uint32_t K2 = kernel_size * kernel_size;
    const uint32_t HWo = H_out * W_out;

    for (uint32_t c = 0; c < C; ++c) {
        const float *x_c = x + (size_t)c * HW;
        const float *k_c = weight + (size_t)c * K2;
        float *y_c = y + (size_t)c * HWo;
        const float b = bias ? bias[c] : 0.0f;

        for (uint32_t ho = 0; ho < H_out; ++ho) {
            const int32_t h_base = (int32_t)(ho * stride) - (int32_t)pad;

            for (uint32_t wo = 0; wo < W_out; ++wo) {
                const int32_t w_base = (int32_t)(wo * stride) - (int32_t)pad;
                float acc = b;

                for (uint32_t kh = 0; kh < kernel_size; ++kh) {
                    const int32_t ih = h_base + (int32_t)kh;
                    if ((uint32_t)ih >= H) continue;

                    const uint32_t x_row = (uint32_t)ih * W;
                    const uint32_t k_row = kh * kernel_size;

                    for (uint32_t kw = 0; kw < kernel_size; ++kw) {
                        const int32_t iw = w_base + (int32_t)kw;
                        if ((uint32_t)iw >= W) continue;

                        acc += x_c[x_row + (uint32_t)iw] * k_c[k_row + kw];
                    }
                }

                y_c[ho * W_out + wo] = acc;
            }
        }
    }
}
