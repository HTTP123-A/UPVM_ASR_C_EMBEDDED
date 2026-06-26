/* Split from sources\naives\micro_kernels.c */

#include "micro_kernels.h" 
void conv2d_f32(const float *x,
                const float *weight,
                const float *bias,
                float *y,
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
        const float *w_oc = weight + (size_t)oc * C_IN * K_SIZE * K_SIZE;

        for (uint32_t oh = 0; oh < H_OUT; ++oh) {
            const int32_t ih0 = (int32_t)(oh * STRIDE) - (int32_t)PAD;

            for (uint32_t ow = 0; ow < W_OUT; ++ow) {
                const int32_t iw0 = (int32_t)(ow * STRIDE) - (int32_t)PAD;
                float acc = bias ? bias[oc] : 0.0f;

                for (uint32_t ic = 0; ic < C_IN; ++ic) {
                    const float *w_oc_ic = w_oc + (size_t)ic * K_SIZE * K_SIZE;

                    for (uint32_t kh = 0; kh < K_SIZE; ++kh) {
                        const int32_t ih = ih0 + (int32_t)kh;
                        if ((uint32_t)ih >= H_IN) continue;

                        const size_t x_row = ((size_t)ic * H_IN + (uint32_t)ih) * W_IN;
                        const size_t w_row = (size_t)kh * K_SIZE;

                        for (uint32_t kw = 0; kw < K_SIZE; ++kw) {
                            const int32_t iw = iw0 + (int32_t)kw;
                            if ((uint32_t)iw >= W_IN) continue;

                            acc += x[x_row + (uint32_t)iw] * w_oc_ic[w_row + kw];
                        }
                    }
                }

                y[((size_t)oc * H_OUT + oh) * W_OUT + ow] = acc;
            }
        }
    }
}

/* --- INT8 --- */
/* <1. Point-wise Convolution - Pytorch equivalent: in_proj, out_proj, conv1 (mlp), conv2 (mlp) > */

