/* Split from sources/naives_mp/micro_kernels.c */

#include "micro_kernels.h"
/*< 4. Group point-wise convolution 1d - Pytorch equivalent: >*/
void grouped_pointwise_conv1d_f32(const float *x,
                                const float *weights,
                                const float *bias,
                                float *y,
                                uint32_t SUB_D_IN,
                                uint32_t K,
                                uint32_t L,
                                uint32_t SUB_D_OUT)
{
    for (uint32_t g = 0; g < K; ++g) {
        for (uint32_t o = 0; o < SUB_D_OUT; ++o) {
            const size_t go = (size_t)g * SUB_D_OUT + o;
            const size_t out_offset = go * L;
            const size_t w_offset = go * SUB_D_IN;
            const float b = bias ? bias[go] : 0.0f;

            for (uint32_t l = 0; l < L; ++l) {
                float acc = b;
                for (uint32_t i = 0; i < SUB_D_IN; ++i) {
                    const size_t in_idx = ((size_t)g * SUB_D_IN + i) * L + l;
                    acc += x[in_idx] * weights[w_offset + i];
                }
                y[out_offset + l] = acc;
            }
        }
    }
}
