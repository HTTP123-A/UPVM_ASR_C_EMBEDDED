/* Split from sources\naives\micro_kernels.c */

#include "micro_kernels.h" 
void elemwise_add_f32(float *x, const float *r, uint32_t N)
{
    for (size_t i = 0; i < N; ++i) {
        x[i] = x[i] + r[i];
    }
}

/* <8. Element-wise mul (inplace) - Pytorch equivalent: Hadamard product c = a .* b > */

