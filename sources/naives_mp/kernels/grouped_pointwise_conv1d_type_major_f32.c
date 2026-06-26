/* Split from sources/naives_mp/micro_kernels.c */

#include "micro_kernels.h"
/*< 5. Group point-wise convolution 1d (type major) - Pytorch equivalent: >*/
void grouped_pointwise_conv1d_type_major_f32(const float *x,
                                              const float *weights,
                                              const float *bias,
                                              float *y,
                                              uint32_t SUB_D_IN,
                                              uint32_t K,
                                              uint32_t L,
                                              uint32_t R,
                                              uint32_t N)
{
    const uint32_t SUB_D_OUT = R + 2U * N;

    /* y layout:
       dts: [K, R, L] at y[0 ... K*R*L)
       Bs : [K, N, L] at y[K*R*L ... K*(R+N)*L)
       Cs : [K, N, L] at y[K*(R+N)*L ... K*(R+2N)*L)
    */
    const size_t dts_base = 0;
    const size_t Bs_base  = (size_t)K * R * L;
    const size_t Cs_base  = (size_t)K * (R + N) * L;

    for (uint32_t g = 0; g < K; ++g) {
        for (uint32_t o = 0; o < SUB_D_OUT; ++o) {
            const size_t go = (size_t)g * SUB_D_OUT + o;
            const size_t w_offset = go * SUB_D_IN;
            const float b = bias ? bias[go] : 0.0f;

            for (uint32_t l = 0; l < L; ++l) {
                float acc = b;
                for (uint32_t i = 0; i < SUB_D_IN; ++i) {
                    const size_t in_idx = ((size_t)g * SUB_D_IN + i) * L + l;
                    acc += x[in_idx] * weights[w_offset + i];
                }

                size_t dst_idx;
                if (o < R) {
                    /* dts */
                    const uint32_t r = o;
                    dst_idx = dts_base + ((size_t)g * R + r) * L + l;
                } else if (o < R + N) {
                    /* Bs */
                    const uint32_t n = o - R;
                    dst_idx = Bs_base + ((size_t)g * N + n) * L + l;
                } else {
                    /* Cs */
                    const uint32_t n = o - (R + N);
                    dst_idx = Cs_base + ((size_t)g * N + n) * L + l;
                }

                y[dst_idx] = acc;
            }
        }
    }
}
