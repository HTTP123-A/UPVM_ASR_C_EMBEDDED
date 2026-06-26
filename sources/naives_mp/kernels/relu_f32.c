/* Split from sources/naives_mp/micro_kernels.c */

#include "micro_kernels.h"
/*< 3. ReLU (inplace) - Pytorch equivalent: torch.nn.ReLU >*/
void relu_f32(const float *x, float *y, uint32_t n, int inplace)
{
    if (!x) return;

    if (inplace) {
        /* inplace mode: x and y are treated as the same buffer */
        float *dst = (float *)x;
        for (uint32_t i = 0; i < n; ++i) {
            if (dst[i] < 0.0f) {
                dst[i] = 0.0f;
            }
        }
    } else {
        /* out-of-place mode: y must be provided */
        if (!y) return;
        for (uint32_t i = 0; i < n; ++i) {
            const float v = x[i];
            y[i] = (v < 0.0f) ? 0.0f : v;
        }
    }
}
