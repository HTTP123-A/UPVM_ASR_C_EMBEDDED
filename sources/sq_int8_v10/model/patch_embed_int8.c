/* v9 fully-integer PatchEmbed: conv1(int8) -> I-LayerNorm(int8 out @ S_ln) -> int8 GELU LUT ->
 * conv2(int8, reads the int8 directly) -> I-LayerNorm(f32 out). The norm1->GELU->conv2 chain runs
 * entirely in int8 through quant_buf (no f32 staging, no separate conv2 quant); the only f32 in the
 * datapath is norm2's output (for the f32 encoder residual). embed_act_buf is unused in v9. */

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
    (void)embed_act_buf;   /* v9: int8 norm1->GELU->conv2 route through quant_buf; f32 staging unused */

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

    /* Conv1 (int8): [1,H,W] -> [C_MID, H/2, W/2] : quant -> conv -> int32 acc */
    {
        const struct qconv_int8 *q = &embed_weight->conv1;
        quant_f32_to_int8_perchannel(embed_inout_buf, quant_buf, C_IN, H_IN * W_IN, q->act_scale, 0);
        conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf,
                    C_IN, H_IN, W_IN, C_MID, H_OUT_MID, W_OUT_MID, K_SIZE, STRIDE, PAD);
    }

    /*< v9: norm1 fully-integer LayerNorm -> int8 (@ S_ln) straight into quant_buf >*/
    layernorm_int_q_to_int8(acc_buf, quant_buf,
                embed_weight->norm1_deq_mul, embed_weight->norm1_deq_sh[0], embed_weight->norm1_gamma_q,
                embed_weight->norm1_amul[0], embed_weight->norm1_ash[0], embed_weight->norm1_bfix,
                embed_weight->rsqrt_knot_q, embed_weight->rsqrt_sl_q, embed_weight->rsqrt_ic_q, LN_SEG,
                C_MID, L_OUT_MID);

    /*< v9: int8 GELU LUT (per-channel; fuses dequant+GELU+quant for conv2), in-place in quant_buf >*/
    gelu_int8_lut(quant_buf, embed_weight->gelu_lut, C_MID, L_OUT_MID);

    /* Conv2 (int8): [C_MID, H/2, W/2] -> [C_EMBED, H/4, W/4] : reads quant_buf int8 directly (no quant) */
    {
        const struct qconv_int8 *q = &embed_weight->conv2;
        conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf,
                    C_MID, H_OUT_MID, W_OUT_MID, C_EMBED, H_OUT, W_OUT, K_SIZE, STRIDE, PAD);
    }

    /*< v9: norm2 fully-integer LayerNorm -> f32 (for the f32 encoder residual) >*/
    layernorm_int_q_to_f32(acc_buf, embed_inout_buf,
                embed_weight->norm2_deq_mul, embed_weight->norm2_deq_sh[0], embed_weight->norm2_gamma_q,
                embed_weight->norm2_b,
                embed_weight->rsqrt_knot_q, embed_weight->rsqrt_sl_q, embed_weight->rsqrt_ic_q, LN_SEG,
                C_EMBED, L_OUT);
}
