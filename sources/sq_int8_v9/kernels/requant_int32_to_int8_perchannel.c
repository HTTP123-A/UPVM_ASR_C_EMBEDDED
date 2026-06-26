/* New in sq_int8_v4. */

#include "micro_kernels.h"
/* 2. Per-channel REQUANT: int32 accumulator -> int8, fusing a layer's dequant
 * (int32 -> f32) with the next layer's activation quant (f32 -> int8) into a
 * single f32 scale + round + clamp + cast.
 *
 *   unfused (v0..v3):  f[c,l] = (float)acc[c,l] * deq_scale[c]                 (dequant)
 *                      q[c,l] = clamp(lrintf(f[c,l] / act_scale_next[c]), -128, 127)  (quant)
 *   fused   (v4):      q[c,l] = clamp(lrintf((float)acc[c,l] * requant_scale[c]), -128, 127)
 *
 * The export tool precomputes one correctly-rounded f32 ratio per channel:
 *   requant_scale[c] = deq_scale[c] / act_scale_next[c]
 *
 * Channel c is the shared dimension (layer-1 out-channel == layer-2 in-channel),
 * so the per-out-channel dequant scale and the per-in-channel quant scale align.
 * Symmetric (zp = 0). NOT bit-identical to dequant->quant: it removes one f32
 * rounding from the value path, so results differ only at rounding ties
 * (empirically < 3e-6 of values, always by +-1 LSB). Rounding (lrintf) and the
 * [-128, 127] clamp match quant_f32_to_int8_perchannel exactly.
 */
void requant_int32_to_int8_perchannel(const int32_t *x,             /* [C, L] int32 acc */
                                      int8_t        *y,             /* [C, L] int8      */
                                      uint32_t       C,
                                      uint32_t       L,
                                      const float   *requant_scale) /* [C] = deq / act_next */
{
    for (uint32_t c = 0; c < C; ++c) {
        const float s = requant_scale[c];
        const int32_t *xc = x + (size_t)c * L;
        int8_t        *yc = y + (size_t)c * L;

        for (uint32_t l = 0; l < L; ++l) {
            int32_t q = (int32_t)lrintf((float)xc[l] * s);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            yc[l] = (int8_t)q;
        }
    }
}
