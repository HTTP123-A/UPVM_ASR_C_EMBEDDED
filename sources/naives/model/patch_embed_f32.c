/* Split from sources\naives\macro_kernels.c */

#include "macro_kernels.h" 
void patch_embed_f32(
    /* ----- External activation/work buffers ---------------- */
                    float *embed_inout_buf,
                    float *embed_act_buf,
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

    /* Conv1: [1,H,W] -> [C_EMBED/2, H/2, W/2] */
    conv2d_f32(embed_inout_buf,
                embed_weight->conv1_w,
                embed_weight->conv1_b,
                embed_act_buf,
                C_IN, H_IN, W_IN,
                C_MID, H_OUT_MID, W_OUT_MID,
                K_SIZE, STRIDE, PAD);
    
    /*< LayerNorm >*/
    layernorm_f32(embed_act_buf,
                embed_weight->norm1_w,
                embed_weight->norm1_b,
                C_MID, L_OUT_MID, 1e-5f);

    /*< GeLU >*/
    gelu_f32(embed_act_buf, (uint32_t)C_MID * L_OUT_MID);

    /* Conv2: [C_EMBED/2, H/2, W/2] -> [C_EMBED, H/4, W/4] */
    conv2d_f32(embed_act_buf,
                embed_weight->conv2_w,
                embed_weight->conv2_b,
                embed_inout_buf,
                C_MID, H_OUT_MID, W_OUT_MID,
                C_EMBED, H_OUT, W_OUT,
                K_SIZE, STRIDE, PAD);

    /*< LayerNorm >*/
    layernorm_f32(embed_inout_buf,
                embed_weight->norm2_w,
                embed_weight->norm2_b,
                C_EMBED, L_OUT, 1e-5f);
}

/*< 2. PVSS Block for Downstream - Pytorch equivalent: PVSS_Block_DS_PVM >*/

