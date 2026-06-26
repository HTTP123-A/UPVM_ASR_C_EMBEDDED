/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
/* <1. Cross Scan - Pytorch equivalent: CrossScan > */
void cross_scan_f32(float *x,
                    uint32_t C,
                    uint32_t H,
                    uint32_t W)
{
    const uint32_t L = H * W;    

    for (uint32_t c = 0; c < C; ++c) {
        const float *xc = x + (size_t)c * L;                  // block 0 (s0 = x)
        float *s1 = x + ((size_t)(1 * C + c) * L);            // block 1
        float *s2 = x + ((size_t)(2 * C + c) * L);            // block 2
        float *s3 = x + ((size_t)(3 * C + c) * L);            // block 3

        for (uint32_t h = 0; h < H; ++h) {
            const float *row = xc + (size_t)h * W;
            uint32_t l0 = h * W;
            uint32_t l1 = h;
            for (uint32_t w = 0; w < W; ++w, ++l0, l1 += H) {
                const float v = row[w];
                s1[l1]         = v;
                s2[L - 1 - l0] = v;
                s3[L - 1 - l1] = v;
            }
        }
    }
}
