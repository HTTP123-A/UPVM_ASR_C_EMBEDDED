/* Split from sources/naives_mp/micro_kernels.c */

#include "micro_kernels.h"
/* <2. Cross Merge - Pytorch equivalent: CrossMerge > */
/* Buffer layout:
 *   buf[0*D*L ... 1*D*L-1] = s0
 *   buf[1*D*L ... 2*D*L-1] = s1
 *   buf[2*D*L ... 3*D*L-1] = s2
 *   buf[3*D*L ... 4*D*L-1] = s3
 * Output is written in-place to s0 block.
 */
void cross_merge_f32(float *x,
                    uint32_t C,
                    uint32_t H,
                    uint32_t W)
{
    const uint32_t L = H * W;

    for (uint32_t c = 0; c < C; ++c) {
        float*       s0 = x + ((size_t)(0 * C + c) * L);  /* output */
        const float* s1 = x + ((size_t)(1 * C + c) * L);
        const float* s2 = x + ((size_t)(2 * C + c) * L);
        const float* s3 = x + ((size_t)(3 * C + c) * L);

        for (uint32_t h = 0; h < H; ++h) {
            uint32_t l0 = h * W;
            uint32_t l1 = h;

            for (uint32_t w = 0; w < W; ++w, ++l0, l1 += H) {
                const float a = s0[l0] + s2[L - 1 - l0];
                const float b = s1[l1] + s3[L - 1 - l1];
                s0[l0] = a + b;
            }
        }
    }
}
