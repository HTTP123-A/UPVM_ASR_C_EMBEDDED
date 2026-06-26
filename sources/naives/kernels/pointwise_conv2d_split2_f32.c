/* Split from sources\naives\micro_kernels.c */

#include "micro_kernels.h" 
void pointwise_conv2d_split2_f32(const float *restrict x_inout,   // [C, H, W]
                                const float *restrict x_res,     // [C, H, W]
                                const float *restrict weight,    // [C, 2*C]
                                const float *restrict bias,      // [C] or NULL
                                float *restrict y,               // [C, H, W]
                                uint32_t C_IN,
                                uint32_t H,
                                uint32_t W,
                                uint32_t C_OUT)
{
    const uint32_t L = H * W;
    const uint32_t C_IN_TOTAL = 2U * C_IN;

    for (uint32_t o = 0; o < C_OUT; ++o) {
        float *restrict yo = y + (size_t)o * L;
        const float *restrict wrow = weight + (size_t)o * C_IN_TOTAL;
        const float b = bias ? bias[o] : 0.0f;

        for (uint32_t pix = 0; pix < L; ++pix) {
            float acc = b;

            // First half channels from inout
            for (uint32_t c = 0; c < C_IN; ++c) {
                acc += wrow[c] * x_inout[(size_t)c * L + pix];
            }

            // Second half channels from global residual
            for (uint32_t c = 0; c < C_IN; ++c) {
                acc += wrow[C_IN + c] * x_res[(size_t)c * L + pix];
            }

            yo[pix] = acc;
        }
    }
}

/*< 4. Group point-wise convolution 1d - Pytorch equivalent: >*/

