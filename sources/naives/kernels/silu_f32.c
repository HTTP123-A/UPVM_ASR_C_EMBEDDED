/* Split from sources\naives\micro_kernels.c */

#include "micro_kernels.h" 
void silu_f32(float *restrict z, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        float x = z[i];
        float s;

        if (x >= 0.0f) {
            s = 1.0f / (1.0f + expf(-x));
        } else {
            float e = expf(x);
            s = e / (1.0f + e);
        }

        z[i] = x * s;
    }
}

/*< 2. GeLU (inplace) - Pytorch equivalent: torch.nn.GeLU >*/

