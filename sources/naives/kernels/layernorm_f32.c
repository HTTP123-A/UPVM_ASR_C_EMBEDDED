/* Split from sources\naives\micro_kernels.c */

#include "micro_kernels.h" 
void layernorm_f32(float *x,
                    const float *gamma,
                    const float *beta,
                    uint32_t C,
                    uint32_t L,
                    float eps)
{
    if (C == 0 || L == 0) return;

    const float invC = 1.0f / (float)C;

    for (uint32_t l = 0; l < L; ++l) {
        /* --- E(x) --- */
        double m = 0.0;
        float *px = x + l;  /* points at x[0,l] */
        for (uint32_t c = 0; c < C; ++c, px += L) {
            m += (double)(*px);
        }
        m *= (double)invC;

        /* --- V(x) --- */
        double v = 0.0;
        px = x + l;
        for (uint32_t c = 0; c < C; ++c, px += L) {
            const double d = (double)(*px) - m;
            v += d * d;
        }
        v *= (double)invC;

        const float mean    = (float)m;
        const float inv_std = 1.0f / sqrtf((float)v + eps);

        /* --- Weight & Bias --- */
        px = x + l;
        for (uint32_t c = 0; c < C; ++c, px += L) {
            const float xn = ((*px) - mean) * inv_std;
            *px = xn * gamma[c] + beta[c];
        }
    }
}

/* --- INT8 --- */

/* ---------------------------------------------- */
/* V. MEMORY */
/* --- F16 --- */
/*< 1. xz chunk - Pytorch equivalent: x , z = x.chunk(2, dim=1) >*/

