/* INT8-path ReLU on the int32 conv accumulator (sq_int8_v1).
 *
 * Fused MLP ReLU: applied to the raw int32 accumulator of the quantized MLP fc1
 * BEFORE per-channel dequant, replacing the v0 "dequant -> relu_f32" sequence.
 * Bit-identical to v0 because every dequant scale is strictly positive
 * (deq_scale[oc] = a_scale * a_scale_mul * w_scale[oc] > 0), so for s > 0:
 *   relu(acc * s) = max(acc * s, 0) = s * max(acc, 0) = s * relu_int32(acc).
 * (Verified: all 104 exported *_w_dequant_scale.bin tensors are > 0.)
 *
 * Symmetric W8A8, zero-point 0, so 0 maps to f32 0.0 after dequant either way. */

#include "micro_kernels.h"
/*< 4. ReLU on int32 accumulator (inplace) - clamp negatives to 0 >*/
void relu_int32(int32_t *x, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        if (x[i] < 0) {
            x[i] = 0;
        }
    }
}
