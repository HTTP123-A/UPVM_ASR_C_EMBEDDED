/* Split from sources\naives\micro_kernels.c */

#include "micro_kernels.h" 
void gelu_f32(float *restrict x, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        float xi = x[i];
        float ei = erff(xi * M_SQRT1_2);
        x[i] = 0.5f * xi * (1.0f + ei);
    }
}

/*< 3. ReLU (inplace) - Pytorch equivalent: torch.nn.ReLU >*/

