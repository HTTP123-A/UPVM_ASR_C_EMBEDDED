/* Split from sources/sq_int8_v0/micro_kernels.c */

#include "micro_kernels.h"
/* 2. Per-channel activation quant: float → int8
 * For layers whose a_scale is per-input-channel [C] in the checkpoint
 * (SmoothQuant-migrated: 104 of 110 qpacks in gen_ptq_int8pack.pth).
 * q[c,l] = clamp( round(x[c,l] / scales[c]) + zp, -128, 127 )
 *
 * scales[c] must be the loader-folded effective scale, combining the
 * runtime smoothing division and the JLSS multiplier in one pass
 * (see utils/ptq_w8a8.py frozen forward):
 *   scales[c] = sq_s[c] * a_scale[c] * a_scale_mul
 * (sq_s is 1 for grouped/depthwise layers - never folded there.)
 */
void quant_f32_to_int8_perchannel(const float   *x,       /* [C, L] */
                                   int8_t        *y,       /* [C, L] */
                                   uint32_t       C,
                                   uint32_t       L,
                                   const float   *scales,  /* [C], a_scale from checkpoint */
                                   int32_t        zp)      /* typically 0 for symmetric */
{
    for (uint32_t c = 0; c < C; ++c) {
        const float inv = 1.0f / scales[c];
        const float *xc = x + (size_t)c * L;
        int8_t      *yc = y + (size_t)c * L;

        for (uint32_t l = 0; l < L; ++l) {
            int32_t q = (int32_t)lrintf(xc[l] * inv) + zp;
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            yc[l] = (int8_t)q;
        }
    }
}
