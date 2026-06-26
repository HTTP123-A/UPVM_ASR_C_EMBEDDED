/* Split from sources/naives_mp/micro_kernels.c */

#include "micro_kernels.h"
/*< 2. GeLU (inplace) - Pytorch equivalent: torch.nn.GeLU >*/
void gelu_f32(float *restrict x, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        float xi = x[i];
        float ei = erff(xi * M_SQRT1_2);
        x[i] = 0.5f * xi * (1.0f + ei);
    }
}
