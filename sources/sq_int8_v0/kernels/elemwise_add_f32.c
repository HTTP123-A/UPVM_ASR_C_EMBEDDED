/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
/*< 6. Element-wise add (inplace) - Pytorch equivalent: Hadamard addition c = a + b >*/
void elemwise_add_f32(float *x, const float *r, uint32_t N)
{
    for (size_t i = 0; i < N; ++i) {
        x[i] = x[i] + r[i];
    }
}
