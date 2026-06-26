/* Mixed-precision (f32 + int8) patch-embed block.
 * conv1/conv2 run int8 (quant -> conv2d_int8 -> int32 accumulator). v8 FUSES each conv's
 * per-channel dequant into the following LayerNorm, which uses a range-reduced piece-wise-linear
 * rsqrt (layernorm_int32_pwl_rsqrt) in place of the f32 sqrtf - no separate dequant pass. gelu f32. */

#include "macro_kernels.h"
/*< 1. Patchembed: [1, F, T] => [16, H, W] >*/
void patch_embed_int8(
    /* ----- External activation/work buffers ---------------- */
                    float *embed_inout_buf,
                    float *embed_act_buf,
                    int8_t *quant_buf,
                    int32_t *acc_buf,
    /* ----- weights struct --------------------------------- */
                    const struct embed_weight* embed_weight,
    /* ----- Shape information ------------------------------ */
                    uint32_t C_IN, uint32_t C_MID, uint32_t C_EMBED,
                    uint32_t H_IN, uint32_t W_IN, uint32_t K_SIZE,
                    uint32_t STRIDE, uint32_t PAD)
{
    /* Output dim after each conv holders */
    uint32_t H_OUT_MID, W_OUT_MID, L_OUT_MID;
    uint32_t H_OUT, W_OUT, L_OUT;

    /* Calculate shape */
    H_OUT_MID = (H_IN + 2U * PAD - K_SIZE) / STRIDE + 1U;
    W_OUT_MID = (W_IN + 2U * PAD - K_SIZE) / STRIDE + 1U;
    L_OUT_MID = H_OUT_MID * W_OUT_MID;

    H_OUT = (H_OUT_MID + 2U * PAD - K_SIZE) / STRIDE + 1U;
    W_OUT = (W_OUT_MID + 2U * PAD - K_SIZE) / STRIDE + 1U;
    L_OUT = H_OUT * W_OUT;

    /* Conv1 (int8): [1,H,W] -> [C_MID, H/2, W/2] : quant -> conv (dequant fused into norm1) */
    {
        const struct qconv_int8 *q = &embed_weight->conv1;
        quant_f32_to_int8_perchannel(embed_inout_buf, quant_buf, C_IN, H_IN * W_IN, q->act_scale, 0);
        conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf,
                    C_IN, H_IN, W_IN, C_MID, H_OUT_MID, W_OUT_MID, K_SIZE, STRIDE, PAD);
    }

    /*< v8: fused int32 dequant + LayerNorm (PWL rsqrt); reads conv1's int32 accumulator directly >*/
    layernorm_int32_pwl_rsqrt(acc_buf, embed_act_buf,
                embed_weight->conv1.deq_scale,
                embed_weight->norm1_w,
                embed_weight->norm1_b,
                C_MID, L_OUT_MID, 1e-5f,
                embed_weight->pwl_rsqrt_knot,
                embed_weight->pwl_rsqrt_slope,
                embed_weight->pwl_rsqrt_intercept,
                PWL_RSQRT_SEGMENTS);

    /*< GeLU >*/
    gelu_f32(embed_act_buf, (uint32_t)C_MID * L_OUT_MID);

    /* Conv2 (int8): [C_MID, H/2, W/2] -> [C_EMBED, H/4, W/4] : quant -> conv (dequant fused into norm2) */
    {
        const struct qconv_int8 *q = &embed_weight->conv2;
        quant_f32_to_int8_perchannel(embed_act_buf, quant_buf, C_MID, H_OUT_MID * W_OUT_MID, q->act_scale, 0);
        conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf,
                    C_MID, H_OUT_MID, W_OUT_MID, C_EMBED, H_OUT, W_OUT, K_SIZE, STRIDE, PAD);
    }

    /*< v8: fused int32 dequant + LayerNorm (PWL rsqrt); reads conv2's int32 accumulator directly >*/
    layernorm_int32_pwl_rsqrt(acc_buf, embed_inout_buf,
                embed_weight->conv2.deq_scale,
                embed_weight->norm2_w,
                embed_weight->norm2_b,
                C_EMBED, L_OUT, 1e-5f,
                embed_weight->pwl_rsqrt_knot,
                embed_weight->pwl_rsqrt_slope,
                embed_weight->pwl_rsqrt_intercept,
                PWL_RSQRT_SEGMENTS);
}
