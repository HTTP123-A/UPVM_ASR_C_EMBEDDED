/* Split from sources\naives\micro_kernels.c */

#include "micro_kernels.h" 
void elemwise_mul_f32(float *x, const float *z, uint32_t N)
{
    for (uint32_t i = 0; i < N; ++i) {
        x[i] = x[i] * z[i];
    }
}


/* <9. SSM core - Pytorch equivalent: SelectiveScanCore (can be Pytroch/CUDA) > */

