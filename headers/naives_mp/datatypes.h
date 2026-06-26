#ifndef __DATATYPES__
#define __DATATYPES__
/* ===================== INCLUDE SECTION ===================== */
#include <time.h>
#include <stddef.h>
#include <stdint.h>


/* ===================== DEFINE ============================== */


/* ===================== DATATYPES =========================== */
/*< 1. PatchEmbed data structure >*/
struct embed_weight {
    /* --- conv1 --- */
    const float* conv1_w;
    const float* conv1_b;
    /* --- norm1 --- */
    const float* norm1_w;
    const float* norm1_b;
    /* --- conv2 --- */
    const float* conv2_w;
    const float* conv2_b;
    /* --- norm2 --- */
    const float* norm2_w;
    const float* norm2_b;
};

/*< 2. PVSS DS data structure >*/
struct pvss_ds_weight {
    /* --- SSM --- */
        /*< in norm >*/
    const float* ssm_in_norm_w;
    const float* ssm_in_norm_b;
        /*< input projection >*/
    const float* ssm_in_projection_w;
        /*< input depthwise >*/
    const float* ssm_dw_conv2d_w;
    const float* ssm_dw_conv2d_b;
        /*< input group pointwise (type major) >*/
    const float* ssm_dw_conv1d_w;
        /*< input group pointwise >*/
    const float* ssm_dt_projection_w;
        /*< As, Ds >*/
    const float* ssm_A;
    const float* ssm_Ds;
    const float* ssm_delta_bias;
        /*< output norm >*/
    const float* ssm_out_norm_w;
    const float* ssm_out_norm_b;
        /*< output projection >*/
    const float* ssm_out_projection_w;

    /* --- MLP --- */
        /*< mlp norm >*/
    const float* mlp_norm_w;
    const float* mlp_norm_b;
        /*< mlp fc1 >*/
    const float* mlp_fc1_w;
    const float* mlp_fc1_b;
        /*< mlp fc2 >*/
    const float* mlp_fc2_w;
    const float* mlp_fc2_b;
        /*< mlp res fc >*/
    const float* mlp_fc_res_w;
    const float* mlp_fc_res_b;
        /*< sumpool weight >*/
    const float* mlp_sumpool_w;

    /* --- Downstream (optional) --- */
    const float* mlp_dim_reduce_w;  // hidden -> 2C (last_vss)
    const float* skip_reduce_w;     // C -> 2C (last_vss)
};

/*< 3. PVSS Latent data structure >*/
struct pvss_latent_weight {
    /* --- SSM --- */
        /*< in norm >*/
    const float* ssm_in_norm_w;
    const float* ssm_in_norm_b;
        /*< input projection >*/
    const float* ssm_in_projection_w;
        /*< input depthwise >*/
    const float* ssm_dw_conv2d_w;
    const float* ssm_dw_conv2d_b;
        /*< input group pointwise (type major) >*/
    const float* ssm_dw_conv1d_w;
        /*< input group pointwise >*/
    const float* ssm_dt_projection_w;
        /*< As, Ds >*/
    const float* ssm_A;
    const float* ssm_Ds;
    const float* ssm_delta_bias;
        /*< output norm >*/
    const float* ssm_out_norm_w;
    const float* ssm_out_norm_b;
        /*< output projection >*/
    const float* ssm_out_projection_w;

    /* --- MLP --- */
        /*< mlp norm >*/
    const float* mlp_norm_w;
    const float* mlp_norm_b;
        /*< mlp fc1 >*/
    const float* mlp_fc1_w;
    const float* mlp_fc1_b;
        /*< mlp fc2 >*/
    const float* mlp_fc2_w;
    const float* mlp_fc2_b;
};

/*< 4. PVSS Upstream data structure >*/
struct pvss_us_weight {
    /* --- SSM --- */
        /*< in norm >*/
    const float* ssm_in_norm_w;
    const float* ssm_in_norm_b;
        /*< input projection >*/
    const float* ssm_in_projection_w;
        /*< input depthwise >*/
    const float* ssm_dw_conv2d_w;
    const float* ssm_dw_conv2d_b;
        /*< input group pointwise (type major) >*/
    const float* ssm_dw_conv1d_w;
        /*< input group pointwise >*/
    const float* ssm_dt_projection_w;
        /*< As, Ds >*/
    const float* ssm_A;
    const float* ssm_Ds;
    const float* ssm_delta_bias;
        /*< output norm >*/
    const float* ssm_out_norm_w;
    const float* ssm_out_norm_b;
        /*< output projection >*/
    const float* ssm_out_projection_w;

    /* --- MLP --- */
        /*< mlp norm >*/
    const float* mlp_norm_w;
    const float* mlp_norm_b;
        /*< mlp fc1 >*/
    const float* mlp_fc1_w;
    const float* mlp_fc1_b;
        /*< mlp fc2 >*/
    const float* mlp_fc2_w;
    const float* mlp_fc2_b;

    /* --- PE  --- */
        /*< pe pointwise >*/
    const float* pe_w;
        /*< pe norm >*/
    const float* pe_norm_w;
    const float* pe_norm_b;
};

/*< 5. PVSS Output3 data structure >*/
struct vss_output_weight {
    /* --- Conv2D  */
    const float* pre_conv2d_w;
    const float* pre_conv2d_b;

    /* --- SSM --- */
        /*< in norm >*/
    const float* ssm_in_norm_w;
    const float* ssm_in_norm_b;
        /*< input projection >*/
    const float* ssm_in_projection_w;
        /*< input depthwise >*/
    const float* ssm_dw_conv2d_w;
    const float* ssm_dw_conv2d_b;
        /*< input group pointwise (type major) >*/
    const float* ssm_dw_conv1d_w;
        /*< input group pointwise >*/
    const float* ssm_dt_projection_w;
        /*< As, Ds >*/
    const float* ssm_A;
    const float* ssm_Ds;
    const float* ssm_delta_bias;
        /*< output norm >*/
    const float* ssm_out_norm_w;
    const float* ssm_out_norm_b;
        /*< output projection >*/
    const float* ssm_out_projection_w;

    /* --- MLP --- */
        /*< mlp norm >*/
    const float* mlp_norm_w;
    const float* mlp_norm_b;
        /*< mlp fc1 >*/
    const float* mlp_fc1_w;
    const float* mlp_fc1_b;
        /*< mlp fc2 >*/
    const float* mlp_fc2_w;
    const float* mlp_fc2_b;
};

/*< 6. PVSS Upstream skip structure >*/
struct pvss_us_skip_weight {
    /*< Dec 0 >*/
    const float* skip_conv_dec0_w;
    const float* skip_conv_dec0_b;

    /*< Dec 1 >*/
    const float* skip_conv_dec1_w;
    const float* skip_conv_dec1_b;

    /*< Dec 2 >*/
    const float* skip_conv_dec2_w;
    const float* skip_conv_dec2_b;

    /*< Output 0 >*/
    const float* skip_conv_out0_w;
    const float* skip_conv_out0_b;
};

/*< 7. Model's weight structure >*/
struct upvm_asr_weight {
    /* --- weight ------- */
        /*< 1. PatchEmbed >*/
    struct embed_weight embed_weight_mag;
    struct embed_weight embed_weight_pha;
        /*< 2. Enc 0 >*/
    struct pvss_ds_weight pvss_ds_weight_enc0_mag;
    struct pvss_ds_weight pvss_ds_weight_enc0_pha;
        /*< 3. Enc 1 >*/
    struct pvss_ds_weight pvss_ds_weight_enc1_mag;
    struct pvss_ds_weight pvss_ds_weight_enc1_pha;
        /*< 4. Enc 2 >*/
    struct pvss_ds_weight pvss_ds_weight_enc2_mag;
    struct pvss_ds_weight pvss_ds_weight_enc2_pha;

        /*< 5. Latent >*/
    struct pvss_latent_weight pvss_latent_weight_mag;
    struct pvss_latent_weight pvss_latent_weight_pha;

        /*< 6. Dec 0 >*/
    struct pvss_us_weight pvss_us_weight_dec0_mag;
    struct pvss_us_weight pvss_us_weight_dec0_pha;

        /*< 7. Dec 1 >*/
    struct pvss_us_weight pvss_us_weight_dec1_mag;
    struct pvss_us_weight pvss_us_weight_dec1_pha;

        /*< 8. Dec 2 >*/
    struct pvss_us_weight pvss_us_weight_dec2_mag;
    struct pvss_us_weight pvss_us_weight_dec2_pha;

        /*< 9. Out 0 >*/
    struct pvss_us_weight pvss_us_weight_out0_mag;
    struct pvss_us_weight pvss_us_weight_out0_pha;

        /*< 10. Out 1 >*/
    struct pvss_us_weight pvss_us_weight_out1_mag;
    struct pvss_us_weight pvss_us_weight_out1_pha;

        /*< 11. Out 3 >*/
    struct vss_output_weight vss_output3_weight_mag;
    struct vss_output_weight vss_output3_weight_pha;

        /* 12. Skip handler */
    struct pvss_us_skip_weight pvss_us_skip_weight_mag;
    struct pvss_us_skip_weight pvss_us_skip_weight_pha;
};

/* --------------------------------------------------------------------------------------------- */
/* 1. Weight file struct */
struct weight_file_spec {
    const char *member_name;
    size_t numel;
    const float **slot;
};

/* --------------------------------------------------------------------------------------------- */
/* 1. Stage timing structure */
struct timing_record_points {
    /* Start time */
    struct timespec start_time;
    /* STFT */
    struct timespec stft_time;

    /* Copy low-freq time */
    struct timespec low_frequency_mag_time;
    struct timespec low_frequency_pha_time;
    /* Spectrogram global residual */
    struct timespec spectrogram_res_mag_time;

    /* PatchEmbed */
    struct timespec patch_embed_mag_time;
    struct timespec patch_embed_pha_time;
        /* Patchembed global residual */
    struct timespec patch_embed_res_mag_time;
    struct timespec patch_embed_res_pha_time;
        /* Patchembed interaction */
    struct timespec patch_embed_interaction_mag_time;
    struct timespec patch_embed_interaction_pha_time;

    /* Encoder 0 */
    struct timespec encoder0_mag_time;
    struct timespec encoder0_pha_time;
        /* Encoder 0 global residual */
    struct timespec encoder0_res_mag_time;
    struct timespec encoder0_res_pha_time;
        /* Encoder 0 interaction */
    struct timespec encoder0_interaction_mag_time;
    struct timespec encoder0_interaction_pha_time;

    /* Encoder 1 */
    struct timespec encoder1_mag_time;
    struct timespec encoder1_pha_time;
        /* Encoder 1 global residual */
    struct timespec encoder1_res_mag_time;
    struct timespec encoder1_res_pha_time;
        /* Encoder 1 interaction */
    struct timespec encoder1_interaction_mag_time;
    struct timespec encoder1_interaction_pha_time;

    /* Encoder 2 */
    struct timespec encoder2_mag_time;
    struct timespec encoder2_pha_time;
        /* Encoder 2 global residual */
    struct timespec encoder2_res_mag_time;
    struct timespec encoder2_res_pha_time;
        /* Encoder 2 interaction */
    struct timespec encoder2_interaction_mag_time;
    struct timespec encoder2_interaction_pha_time;

    /* Latent */
    struct timespec latent_mag_time;
    struct timespec latent_pha_time;
        /* Latent interaction */
    struct timespec latent_interaction_mag_time;
    struct timespec latent_interaction_pha_time;

    /* Decoder 0 */
        /* Decoder 0 global residual */
    struct timespec decoder0_res_mag_time;
    struct timespec decoder0_res_pha_time;
        /* Decoder 0 PVSS time */
    struct timespec decoder0_mag_time;
    struct timespec decoder0_pha_time;    
        /* Decoder 0 interaction */
    struct timespec decoder0_interaction_mag_time;
    struct timespec decoder0_interaction_pha_time;

    /* Decoder 1 */
        /* Decoder 1 global residual */
    struct timespec decoder1_res_mag_time;
    struct timespec decoder1_res_pha_time;
        /* Decoder 1 PVSS time */
    struct timespec decoder1_mag_time;
    struct timespec decoder1_pha_time;    
        /* Decoder 1 interaction */
    struct timespec decoder1_interaction_mag_time;
    struct timespec decoder1_interaction_pha_time;

    /* Decoder 2 */
        /* Decoder 2 global residual */
    struct timespec decoder2_res_mag_time;
    struct timespec decoder2_res_pha_time;
        /* Decoder 2 PVSS time */
    struct timespec decoder2_mag_time;
    struct timespec decoder2_pha_time;    
        /* Decoder 2 interaction */
    struct timespec decoder2_interaction_mag_time;
    struct timespec decoder2_interaction_pha_time;
    
    /* Output 0 */
        /* Output 0 global residual */
    struct timespec output0_res_mag_time;
    struct timespec output0_res_pha_time;
        /* Output 0 PVSS time */
    struct timespec output0_mag_time;
    struct timespec output0_pha_time;

    /* Output 1 */    
    struct timespec output1_mag_time;
    struct timespec output1_pha_time;

    /* Output 3 */
        /* Output 3 PVSS time */
    struct timespec output3_mag_time;
    struct timespec output3_pha_time;
        /* Final output residual time */
    struct timespec output3_res_mag_time;

    /* Low-Freq replacement */
    struct timespec final_low_freq_mag_time;
    struct timespec final_low_freq_pha_time;

    /* iSTFT */
    struct timespec istft_time;

    /* Total time */
    struct timespec processing_time;
};

/* 2. Detail timing structure */
struct detail_timing_record_points {
    /* Start time */
    struct timespec start_time;
    /* STFT */
    struct timespec stft_time;

    /* Copy low-freq time */
    struct timespec low_frequency_mag_time;
    struct timespec low_frequency_pha_time;
    /* Spectrogram global residual */
    struct timespec spectrogram_res_mag_time;

    /* PatchEmbed */
    struct timespec patch_embed_mag_time;
    struct timespec patch_embed_pha_time;
        /* Patchembed global residual */
    struct timespec patch_embed_res_mag_time;
    struct timespec patch_embed_res_pha_time;
        /* Patchembed interaction */
    struct timespec patch_embed_interaction_mag_time;
    struct timespec patch_embed_interaction_pha_time;

    /* Encoder 0 */
    struct timespec encoder0_pvss_mag_time;
    struct timespec encoder0_mlp_mag_time;
    struct timespec encoder0_mag_time;

    struct timespec encoder0_pvss_pha_time;
    struct timespec encoder0_mlp_pha_time;
    struct timespec encoder0_pha_time;
        /* Encoder 0 global residual */
    struct timespec encoder0_res_mag_time;
    struct timespec encoder0_res_pha_time;
        /* Encoder 0 interaction */
    struct timespec encoder0_interaction_mag_time;
    struct timespec encoder0_interaction_pha_time;

    /* Encoder 1 */
    struct timespec encoder1_pvss_mag_time;
    struct timespec encoder1_mlp_mag_time;
    struct timespec encoder1_mag_time;

    struct timespec encoder1_pvss_pha_time;
    struct timespec encoder1_mlp_pha_time;
    struct timespec encoder1_pha_time;
        /* Encoder 1 global residual */
    struct timespec encoder1_res_mag_time;
    struct timespec encoder1_res_pha_time;
        /* Encoder 1 interaction */
    struct timespec encoder1_interaction_mag_time;
    struct timespec encoder1_interaction_pha_time;

    /* Encoder 2 */
    struct timespec encoder2_pvss_mag_time;
    struct timespec encoder2_mlp_mag_time;
    struct timespec encoder2_mag_time;

    struct timespec encoder2_pvss_pha_time;
    struct timespec encoder2_mlp_pha_time;
    struct timespec encoder2_pha_time;
        /* Encoder 2 global residual */
    struct timespec encoder2_res_mag_time;
    struct timespec encoder2_res_pha_time;
        /* Encoder 2 interaction */
    struct timespec encoder2_interaction_mag_time;
    struct timespec encoder2_interaction_pha_time;

    /* Latent */
    struct timespec latent_pvss_mag_time;
    struct timespec latent_mlp_mag_time;
    struct timespec latent_mag_time;

    struct timespec latent_pvss_pha_time;
    struct timespec latent_mlp_pha_time;
    struct timespec latent_pha_time;
        /* Latent interaction */
    struct timespec latent_interaction_mag_time;
    struct timespec latent_interaction_pha_time;

    /* Decoder 0 */
        /* Decoder 0 global residual */
    struct timespec decoder0_pvss_mag_time;
    struct timespec decoder0_mlp_mag_time;
    struct timespec decoder0_res_mag_time;

    struct timespec decoder0_pvss_pha_time;
    struct timespec decoder0_mlp_pha_time;
    struct timespec decoder0_res_pha_time;
        /* Decoder 0 PVSS time */
    struct timespec decoder0_mag_time;
    struct timespec decoder0_pha_time;    
        /* Decoder 0 interaction */
    struct timespec decoder0_interaction_mag_time;
    struct timespec decoder0_interaction_pha_time;

    /* Decoder 1 */
        /* Decoder 1 global residual */
    struct timespec decoder1_pvss_mag_time;
    struct timespec decoder1_mlp_mag_time;
    struct timespec decoder1_res_mag_time;

    struct timespec decoder1_pvss_pha_time;
    struct timespec decoder1_mlp_pha_time;
    struct timespec decoder1_res_pha_time;
        /* Decoder 1 PVSS time */
    struct timespec decoder1_mag_time;
    struct timespec decoder1_pha_time;    
        /* Decoder 1 interaction */
    struct timespec decoder1_interaction_mag_time;
    struct timespec decoder1_interaction_pha_time;

    /* Decoder 2 */
        /* Decoder 2 global residual */
    struct timespec decoder2_pvss_mag_time;
    struct timespec decoder2_mlp_mag_time;
    struct timespec decoder2_res_mag_time;

    struct timespec decoder2_pvss_pha_time;
    struct timespec decoder2_mlp_pha_time;
    struct timespec decoder2_res_pha_time;
        /* Decoder 2 PVSS time */
    struct timespec decoder2_mag_time;
    struct timespec decoder2_pha_time;    
        /* Decoder 2 interaction */
    struct timespec decoder2_interaction_mag_time;
    struct timespec decoder2_interaction_pha_time;
    
    /* Output 0 */
        /* Output 0 global residual */
    struct timespec output0_pvss_mag_time;
    struct timespec output0_mlp_mag_time;
    struct timespec output0_res_mag_time;

    struct timespec output0_pvss_pha_time;
    struct timespec output0_mlp_pha_time;
    struct timespec output0_res_pha_time;
        /* Output 0 PVSS time */
    struct timespec output0_mag_time;
    struct timespec output0_pha_time;

    /* Output 1 */
    struct timespec output1_pvss_mag_time;
    struct timespec output1_mlp_mag_time;
    struct timespec output1_mag_time;

    struct timespec output1_pvss_pha_time;
    struct timespec output1_mlp_pha_time;
    struct timespec output1_pha_time;

    /* Output 3 */
        /* Output 3 PVSS time */
    struct timespec output3_pvss_mag_time;
    struct timespec output3_mlp_mag_time;
    struct timespec output3_mag_time;

    struct timespec output3_pvss_pha_time;
    struct timespec output3_mlp_pha_time;
    struct timespec output3_pha_time;
        /* Final output residual time */
    struct timespec output3_res_mag_time;

    /* Low-Freq replacement */
    struct timespec final_low_freq_mag_time;
    struct timespec final_low_freq_pha_time;

    /* iSTFT */
    struct timespec istft_time;

    /* Total time */
    struct timespec processing_time;
};


#endif /* __DATATYPES__ */