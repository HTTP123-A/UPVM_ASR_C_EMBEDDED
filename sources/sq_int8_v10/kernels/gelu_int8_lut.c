#include "micro_kernels.h"

/*< v9 int8 GELU via a per-channel 256-entry LUT. Each entry absorbs dequant(S_ln) + GELU +
 *  quant(conv2.act_scale[c]) in one table (built at export):
 *     lut[c*256 + (q+128)] = clamp(round( GELU(q*S_ln) / conv2.act_scale[c] )), q in [-128,127].
 *  int8 in (norm1 output at S_ln) -> int8 out (conv2 input). In-place safe (y may alias x). >*/
void gelu_int8_lut(int8_t *x, const int8_t *lut, uint32_t C, uint32_t L)
{
    for (uint32_t c = 0; c < C; ++c) {
        const int8_t *lc = lut + (size_t)c * 256;
        int8_t       *xc = x + (size_t)c * L;
        for (uint32_t l = 0; l < L; ++l)
            xc[l] = lc[(int)xc[l] + 128];
    }
}
