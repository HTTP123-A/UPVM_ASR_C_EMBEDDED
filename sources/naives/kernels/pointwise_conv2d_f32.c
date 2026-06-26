/* Split from sources\naives\micro_kernels.c */

#include "micro_kernels.h" 
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

