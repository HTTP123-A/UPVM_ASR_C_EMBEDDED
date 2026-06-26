/* Split from sources/naives_mp/model.c */

#include "model.h"
/* ---------------------------------------------- */
/*< 1. Weight & Other buffers init >*/
void model_buf_init_f32(
    /* ----- weights ----------------------------------------- */
                        struct upvm_asr_weight** model_weight,
                        char *weight_path,
    /* ----- External activation/work buffers ---------------- */
                        float **model_inout_buf_mag, float **model_inout_buf_pha,
                        float **model_internal_res_buf_mag, float **model_internal_res_buf_pha,
                        float **model_global_res_buf_mag, float **model_global_res_buf_pha,
                        float **model_act_buf_mag, float **model_act_buf_pha,
                        float **model_low_freq_buf_mag, float **model_low_freq_buf_pha,
                        float **hidden_state_buf_mag, float **hidden_state_buf_pha)
{
    /* Inout buf */
    *model_inout_buf_mag = calloc((size_t)INOUT_BUF_ELEMENT, sizeof(float));
    *model_inout_buf_pha = calloc((size_t)INOUT_BUF_ELEMENT, sizeof(float));

    /* Internal residual */
    *model_internal_res_buf_mag = calloc((size_t)INTERNAL_RES_BUF_ELEMENT, sizeof(float));
    *model_internal_res_buf_pha = calloc((size_t)INTERNAL_RES_BUF_ELEMENT, sizeof(float));

    /* Global residual */
    *model_global_res_buf_mag = calloc((size_t)(GLOBAL_RES_BUF_ELEMENT), sizeof(float));
    *model_global_res_buf_pha = calloc((size_t)(GLOBAL_RES_BUF_ELEMENT - SKIP_OFFSET_PATCH_EMBED), sizeof(float));

    /* Activation buf */
    *model_act_buf_mag = calloc((size_t)ACT_BUF_ELEMENT, sizeof(float));
    *model_act_buf_pha = calloc((size_t)ACT_BUF_ELEMENT, sizeof(float));

    /* Low freq buf */
    *model_low_freq_buf_mag = calloc((size_t)LOW_FREQ_ELEMENT, sizeof(float));
    *model_low_freq_buf_pha = calloc((size_t)LOW_FREQ_ELEMENT, sizeof(float));

    /* Hidden state */
    *hidden_state_buf_mag = calloc((size_t)HIDDEN_STATE_BUF_ELEMENT, sizeof(float));
    *hidden_state_buf_pha = calloc((size_t)HIDDEN_STATE_BUF_ELEMENT, sizeof(float));

    /* ---------------------------------------------------------------------------------------------------------------- */
    /* Weight */
    if (weight_path == NULL) {
        die_weight_io("weight_path is NULL", NULL);
    }

    *model_weight = (struct upvm_asr_weight *)calloc(1U, sizeof(**model_weight));
    if (*model_weight == NULL) {
        die_weight_io("failed to allocate upvm_asr_weight", NULL);
    }

    load_embed_weight_stage(&(*model_weight)->embed_weight_mag,
                            weight_path,
                            "embed_weight_mag",
                            C_IN_PATCH_EMBED, C_MID_PATCH_EMBED, C_OUT_PATCH_EMBED, K_SIZE_PATCH_EMBED);
    load_embed_weight_stage(&(*model_weight)->embed_weight_pha,
                            weight_path,
                            "embed_weight_pha",
                            C_IN_PATCH_EMBED, C_MID_PATCH_EMBED, C_OUT_PATCH_EMBED, K_SIZE_PATCH_EMBED);

    load_pvss_ds_weight_stage(&(*model_weight)->pvss_ds_weight_enc0_mag,
                              weight_path,
                              "pvss_ds_weight_enc0_mag",
                              C_ENC_0, R_ENC_0, N_ENC_0, R2N_ENC_0, DW_KERNEL_SZ_ENC_0,
                              MLP_HIDDEN_ENC_0, SUB_D_INNER_ENC_0, SUB_D_PROJECTION_ENC_0, SUB_C_ENC_0);
    load_pvss_ds_weight_stage(&(*model_weight)->pvss_ds_weight_enc0_pha,
                              weight_path,
                              "pvss_ds_weight_enc0_pha",
                              C_ENC_0, R_ENC_0, N_ENC_0, R2N_ENC_0, DW_KERNEL_SZ_ENC_0,
                              MLP_HIDDEN_ENC_0, SUB_D_INNER_ENC_0, SUB_D_PROJECTION_ENC_0, SUB_C_ENC_0);

    load_pvss_ds_weight_stage(&(*model_weight)->pvss_ds_weight_enc1_mag,
                              weight_path,
                              "pvss_ds_weight_enc1_mag",
                              C_ENC_1, R_ENC_1, N_ENC_1, R2N_ENC_1, DW_KERNEL_SZ_ENC_1,
                              MLP_HIDDEN_ENC_1, SUB_D_INNER_ENC_1, SUB_D_PROJECTION_ENC_1, SUB_C_ENC_1);
    load_pvss_ds_weight_stage(&(*model_weight)->pvss_ds_weight_enc1_pha,
                              weight_path,
                              "pvss_ds_weight_enc1_pha",
                              C_ENC_1, R_ENC_1, N_ENC_1, R2N_ENC_1, DW_KERNEL_SZ_ENC_1,
                              MLP_HIDDEN_ENC_1, SUB_D_INNER_ENC_1, SUB_D_PROJECTION_ENC_1, SUB_C_ENC_1);

    load_pvss_ds_weight_stage(&(*model_weight)->pvss_ds_weight_enc2_mag,
                              weight_path,
                              "pvss_ds_weight_enc2_mag",
                              C_ENC_2, R_ENC_2, N_ENC_2, R2N_ENC_2, DW_KERNEL_SZ_ENC_2,
                              MLP_HIDDEN_ENC_2, SUB_D_INNER_ENC_2, SUB_D_PROJECTION_ENC_2, SUB_C_ENC_2);
    load_pvss_ds_weight_stage(&(*model_weight)->pvss_ds_weight_enc2_pha,
                              weight_path,
                              "pvss_ds_weight_enc2_pha",
                              C_ENC_2, R_ENC_2, N_ENC_2, R2N_ENC_2, DW_KERNEL_SZ_ENC_2,
                              MLP_HIDDEN_ENC_2, SUB_D_INNER_ENC_2, SUB_D_PROJECTION_ENC_2, SUB_C_ENC_2);

    load_pvss_latent_weight_stage(&(*model_weight)->pvss_latent_weight_mag,
                                  weight_path,
                                  "pvss_latent_weight_mag",
                                  C_LATENT, R_LATENT, N_LATENT, R2N_LATENT, DW_KERNEL_SZ_LATENT,
                                  MLP_HIDDEN_LATENT, SUB_D_INNER_LATENT, SUB_D_PROJECTION_LATENT, SUB_C_LATENT);
    load_pvss_latent_weight_stage(&(*model_weight)->pvss_latent_weight_pha,
                                  weight_path,
                                  "pvss_latent_weight_pha",
                                  C_LATENT, R_LATENT, N_LATENT, R2N_LATENT, DW_KERNEL_SZ_LATENT,
                                  MLP_HIDDEN_LATENT, SUB_D_INNER_LATENT, SUB_D_PROJECTION_LATENT, SUB_C_LATENT);

    load_pvss_us_skip_weight_stage(&(*model_weight)->pvss_us_skip_weight_mag,
                                   weight_path,
                                   "pvss_us_skip_weight_mag");
    load_pvss_us_skip_weight_stage(&(*model_weight)->pvss_us_skip_weight_pha,
                                   weight_path,
                                   "pvss_us_skip_weight_pha");

    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_dec0_mag,
                              weight_path,
                              "pvss_us_weight_dec0_mag",
                              C_DEC_0, C_DEC_0, R_DEC_0, N_DEC_0, R2N_DEC_0, DW_KERNEL_SZ_DEC_0,
                              MLP_HIDDEN_DEC_0, SUB_D_INNER_DEC_0, SUB_D_PROJECTION_DEC_0, SUB_C_DEC_0);
    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_dec0_pha,
                              weight_path,
                              "pvss_us_weight_dec0_pha",
                              C_DEC_0, C_DEC_0, R_DEC_0, N_DEC_0, R2N_DEC_0, DW_KERNEL_SZ_DEC_0,
                              MLP_HIDDEN_DEC_0, SUB_D_INNER_DEC_0, SUB_D_PROJECTION_DEC_0, SUB_C_DEC_0);

    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_dec1_mag,
                              weight_path,
                              "pvss_us_weight_dec1_mag",
                              C_DEC_1, C_DEC_1, R_DEC_1, N_DEC_1, R2N_DEC_1, DW_KERNEL_SZ_DEC_1,
                              MLP_HIDDEN_DEC_1, SUB_D_INNER_DEC_1, SUB_D_PROJECTION_DEC_1, SUB_C_DEC_1);
    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_dec1_pha,
                              weight_path,
                              "pvss_us_weight_dec1_pha",
                              C_DEC_1, C_DEC_1, R_DEC_1, N_DEC_1, R2N_DEC_1, DW_KERNEL_SZ_DEC_1,
                              MLP_HIDDEN_DEC_1, SUB_D_INNER_DEC_1, SUB_D_PROJECTION_DEC_1, SUB_C_DEC_1);

    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_dec2_mag,
                              weight_path,
                              "pvss_us_weight_dec2_mag",
                              C_DEC_2, C_DEC_2, R_DEC_2, N_DEC_2, R2N_DEC_2, DW_KERNEL_SZ_DEC_2,
                              MLP_HIDDEN_DEC_2, SUB_D_INNER_DEC_2, SUB_D_PROJECTION_DEC_2, SUB_C_DEC_2);
    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_dec2_pha,
                              weight_path,
                              "pvss_us_weight_dec2_pha",
                              C_DEC_2, C_DEC_2, R_DEC_2, N_DEC_2, R2N_DEC_2, DW_KERNEL_SZ_DEC_2,
                              MLP_HIDDEN_DEC_2, SUB_D_INNER_DEC_2, SUB_D_PROJECTION_DEC_2, SUB_C_DEC_2);

    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_out0_mag,
                              weight_path,
                              "pvss_us_weight_out0_mag",
                              C_OUT_0, SUB_C_OUT_0, R_OUT_0, N_OUT_0, R2N_OUT_0, DW_KERNEL_SZ_OUT_0,
                              MLP_HIDDEN_OUT_0, SUB_D_INNER_OUT_0, SUB_D_PROJECTION_OUT_0, SUB_C_OUT_0);
    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_out0_pha,
                              weight_path,
                              "pvss_us_weight_out0_pha",
                              C_OUT_0, SUB_C_OUT_0, R_OUT_0, N_OUT_0, R2N_OUT_0, DW_KERNEL_SZ_OUT_0,
                              MLP_HIDDEN_OUT_0, SUB_D_INNER_OUT_0, SUB_D_PROJECTION_OUT_0, SUB_C_OUT_0);

    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_out1_mag,
                              weight_path,
                              "pvss_us_weight_out1_mag",
                              C_OUT_1, C_OUT_1, R_OUT_1, N_OUT_1, R2N_OUT_1, DW_KERNEL_SZ_OUT_1,
                              MLP_HIDDEN_OUT_1, SUB_D_INNER_OUT_1, SUB_D_PROJECTION_OUT_1, SUB_C_OUT_1);
    load_pvss_us_weight_stage(&(*model_weight)->pvss_us_weight_out1_pha,
                              weight_path,
                              "pvss_us_weight_out1_pha",
                              C_OUT_1, C_OUT_1, R_OUT_1, N_OUT_1, R2N_OUT_1, DW_KERNEL_SZ_OUT_1,
                              MLP_HIDDEN_OUT_1, SUB_D_INNER_OUT_1, SUB_D_PROJECTION_OUT_1, SUB_C_OUT_1);

    load_vss_output_weight_stage(&(*model_weight)->vss_output3_weight_mag,
                                 weight_path,
                                 "vss_output3_weight_mag",
                                 C_OUT_3, R_OUT_3, N_OUT_3, R2N_OUT_3, DW_KERNEL_SZ_OUT_3,
                                 MLP_HIDDEN_OUT_3, D_INNER_OUT_3, D_PROJECTION_OUT_3);
    load_vss_output_weight_stage(&(*model_weight)->vss_output3_weight_pha,
                                 weight_path,
                                 "vss_output3_weight_pha",
                                 C_OUT_3, R_OUT_3, N_OUT_3, R2N_OUT_3, DW_KERNEL_SZ_OUT_3,
                                 MLP_HIDDEN_OUT_3, D_INNER_OUT_3, D_PROJECTION_OUT_3);

}
