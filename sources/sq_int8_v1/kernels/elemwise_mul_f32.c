/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
/* <7. Element-wise mul (inplace) - Pytorch equivalent: Hadamard product c = a .* b > */
void elemwise_mul_f32(float *x, const float *z, uint32_t N)
{
    for (uint32_t i = 0; i < N; ++i) {
        x[i] = x[i] * z[i];
    }
}
