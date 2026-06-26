#ifndef __DATATYPES__
#define __DATATYPES__
/* ===================== INCLUDE SECTION ===================== */
#include <time.h>
#include <stddef.h>
#include <stdint.h>


/* ===================== DEFINE ============================== */


/* ===================== DATATYPES =========================== */
/*< 0. Quantized conv / linear layer (W8A8 symmetric, gen_ptq_int8pack.pth) >
 * Pure INT8 layer - no fallback, no runtime switching. A member of this type
 * means: this layer IS quantized in the checkpoint. Layers that are float in
 * the checkpoint use plain `const float*` members instead, so every struct
 * below is an exact compile-time image of its checkpoint block.
 *
 * Members (loader fold rules: see PROGRESS.md section 1b):
 *   w_i8          quantized weights, same memory layout as the f32 weight
 *   deq_scale     [C_OUT] per-out-channel scale, int32 accumulator -> f32
 *   bias_i32      [C_OUT] pre-scaled bias round(bias_f32 / deq_scale[oc]), or NULL
 *   act_scale     input activation quant scale(s), always [C_IN]
 *   act_scale_len = C_IN (the 6 per-tensor layers are exactly the C_IN==1 ones).
 *                 The runtime always quantizes with quant_f32_to_int8_perchannel
 *                 (per-channel at C_IN==1 is bit-identical to per-tensor), so this
 *                 field is informational / load-time validation, not a kernel switch.
 */
struct qconv_int8 {
    const int8_t  *w_i8;
    const float   *deq_scale;
    const int32_t *bias_i32;
    const float   *act_scale;
    uint32_t       act_scale_len;
};

/*< 1. PatchEmbed data structure >*/
struct embed_weight {
    /* --- conv1 (ckpt: patch_embed_*.0, conv2d k3 s2, int8) --- */
    struct qconv_int8 conv1;
    /* --- norm1 --- */
    const float* norm1_w;
    const float* norm1_b;
    /* --- conv2 (ckpt: patch_embed_*.5, conv2d k3 s2, int8) --- */
    struct qconv_int8 conv2;
    /* --- norm2 --- */
    const float* norm2_w;
    const float* norm2_b;
};

/*< 2. PVSS DS data structure (ckpt: layers_encoder_*.0/.1/.2) >*/
struct pvss_ds_weight {
    /* --- SSM --- */
        /*< in norm >*/
    const float* ssm_in_norm_w;
    const float* ssm_in_norm_b;
        /*< input projection (ckpt: ss2d.in_proj, 1x1, int8, no bias) >*/
    struct qconv_int8 ssm_in_projection;
        /*< input depthwise (ckpt: ss2d.conv2d, k3 s1, int8) >*/
    struct qconv_int8 ssm_dw_conv2d;
        /*< input group pointwise (type major) (ckpt: ss2d.x_proj_conv, f32) >*/
    const float* ssm_dw_conv1d_w;
        /*< input group pointwise (ckpt: ss2d.dt_proj_conv, f32) >*/
    const float* ssm_dt_projection_w;
        /*< As, Ds >*/
    const float* ssm_A;
    const float* ssm_Ds;
    const float* ssm_delta_bias;
        /*< output norm >*/
    const float* ssm_out_norm_w;
    const float* ssm_out_norm_b;
        /*< output projection (ckpt: ss2d.out_proj, 1x1, int8, no bias) >*/
    struct qconv_int8 ssm_out_projection;

    /* --- MLP --- */
        /*< mlp norm >*/
    const float* mlp_norm_w;
    const float* mlp_norm_b;
        /*< mlp fc1 (ckpt: mlp.fc1, int8) >*/
    struct qconv_int8 mlp_fc1;
        /*< mlp fc2 - main path (ckpt: mlp.fc2.mlp_dim_redu, 1x1 C->2C, int8, no bias) >*/
    struct qconv_int8 mlp_fc2;
        /*< mlp res fc - residual path (ckpt: skip_proj.skip_dim_redu, 1x1 C->2C, f32 -
         * NOT quantized; the quantized ckpt module `mlp.dim_reduce` is unused by the C
         * model, see weight_extractor.py / IMPLEMENTATION_NOTES.md) >*/
    const float* mlp_fc_res_w;
    const float* mlp_fc_res_b;
        /*< sumpool weight (ckpt: mlp.fc2.mlp_sumpool, learned k2 s2 depthwise, f32) >*/
    const float* mlp_sumpool_w;

    /* --- Downstream (optional, unused dummies in this config) --- */
    const float* mlp_dim_reduce_w;  // hidden -> 2C (last_vss)
    const float* skip_reduce_w;     // C -> 2C (last_vss)
};

/*< 3. PVSS Latent data structure (ckpt: layers_encoder_*.3) >*/
struct pvss_latent_weight {
    /* --- SSM --- */
        /*< in norm >*/
    const float* ssm_in_norm_w;
    const float* ssm_in_norm_b;
        /*< input projection (ckpt: ss2d.in_proj, 1x1, int8, no bias) >*/
    struct qconv_int8 ssm_in_projection;
        /*< input depthwise (ckpt: ss2d.conv2d, k3 s1, int8) >*/
    struct qconv_int8 ssm_dw_conv2d;
        /*< input group pointwise (type major) (ckpt: ss2d.x_proj_conv, f32) >*/
    const float* ssm_dw_conv1d_w;
        /*< input group pointwise (ckpt: ss2d.dt_proj_conv, f32) >*/
    const float* ssm_dt_projection_w;
        /*< As, Ds >*/
    const float* ssm_A;
    const float* ssm_Ds;
    const float* ssm_delta_bias;
        /*< output norm >*/
    const float* ssm_out_norm_w;
    const float* ssm_out_norm_b;
        /*< output projection (ckpt: ss2d.out_proj, 1x1, int8, no bias) >*/
    struct qconv_int8 ssm_out_projection;

    /* --- MLP --- */
        /*< mlp norm >*/
    const float* mlp_norm_w;
    const float* mlp_norm_b;
        /*< mlp fc1 (ckpt: mlp.fc1, linear, int8) >*/
    struct qconv_int8 mlp_fc1;
        /*< mlp fc2 (ckpt: mlp.fc2, linear, int8) >*/
    struct qconv_int8 mlp_fc2;
};

/*< 4. PVSS Upstream data structure - decoder only (ckpt: layers_decoder_*.1/.2/.3) >*/
struct pvss_us_weight {
    /* --- SSM --- */
        /*< in norm >*/
    const float* ssm_in_norm_w;
    const float* ssm_in_norm_b;
        /*< input projection (ckpt: ss2d.in_proj, 1x1, int8, no bias) >*/
    struct qconv_int8 ssm_in_projection;
        /*< input depthwise (ckpt: ss2d.conv2d, k3 s1, int8) >*/
    struct qconv_int8 ssm_dw_conv2d;
        /*< input group pointwise (type major) (ckpt: ss2d.x_proj_conv, f32) >*/
    const float* ssm_dw_conv1d_w;
        /*< input group pointwise (ckpt: ss2d.dt_proj_conv, f32) >*/
    const float* ssm_dt_projection_w;
        /*< As, Ds >*/
    const float* ssm_A;
    const float* ssm_Ds;
    const float* ssm_delta_bias;
        /*< output norm >*/
    const float* ssm_out_norm_w;
    const float* ssm_out_norm_b;
        /*< output projection (ckpt: ss2d.out_proj, 1x1, int8, no bias) >*/
    struct qconv_int8 ssm_out_projection;

    /* --- MLP --- */
        /*< mlp norm >*/
    const float* mlp_norm_w;
    const float* mlp_norm_b;
        /*< mlp fc1 (ckpt: mlp.fc1, linear, int8) >*/
    struct qconv_int8 mlp_fc1;
        /*< mlp fc2 (ckpt: mlp.fc2, linear, int8) >*/
    struct qconv_int8 mlp_fc2;

    /* --- PE  --- */
        /*< pe pointwise (ckpt: sampler.expand, 1x1, int8, no bias) >*/
    struct qconv_int8 pe;
        /*< pe norm (ckpt: sampler.norm, f32) >*/
    const float* pe_norm_w;
    const float* pe_norm_b;
};

/*< 5a. VSS Output0 data structure (ckpt: output_layer_*.0)
 * Exact checkpoint image: whole ss2d is f32; mlp.fc1/fc2 (with bias),
 * sampler.expand (no bias) are int8. No block norms in the checkpoint
 * (norm is Identity for this block). Its skip_handler lives in
 * pvss_us_skip_weight (skip_conv_out0, int8). >*/
struct vss_output0_weight {
    /* --- SSM (all f32) --- */
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
        /*< mlp fc1 (ckpt: mlp.fc1, linear, int8) >*/
    struct qconv_int8 mlp_fc1;
        /*< mlp fc2 (ckpt: mlp.fc2, linear, int8) >*/
    struct qconv_int8 mlp_fc2;

    /* --- PE  --- */
        /*< pe pointwise (ckpt: sampler.expand, 1x1, int8, no bias) >*/
    struct qconv_int8 pe;
        /*< pe norm (ckpt: sampler.norm, f32) >*/
    const float* pe_norm_w;
    const float* pe_norm_b;
};

/*< 5b. VSS Output1 data structure (ckpt: output_layer_*.1)
 * Same as output0 plus the block norms (ckpt: blocks.0.norm / norm2, f32);
 * no skip_handler for this block. >*/
struct vss_output1_weight {
    /* --- SSM --- */
        /*< in norm (ckpt: blocks.0.norm, f32) >*/
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
        /*< mlp norm (ckpt: blocks.0.norm2, f32) >*/
    const float* mlp_norm_w;
    const float* mlp_norm_b;
        /*< mlp fc1 (ckpt: mlp.fc1, linear, int8) >*/
    struct qconv_int8 mlp_fc1;
        /*< mlp fc2 (ckpt: mlp.fc2, linear, int8) >*/
    struct qconv_int8 mlp_fc2;

    /* --- PE  --- */
        /*< pe pointwise (ckpt: sampler.expand, 1x1, int8, no bias) >*/
    struct qconv_int8 pe;
        /*< pe norm (ckpt: sampler.norm, f32) >*/
    const float* pe_norm_w;
    const float* pe_norm_b;
};

/*< 5c. VSS Output3 data structure (ckpt: output_layer_*.3, pre conv = output_layer_*.2)
 * Exact checkpoint image: pre conv + whole ss2d are f32; only mlp.fc1/fc2
 * (with bias) are int8. No block norms (Identity in PyTorch for output3). >*/
struct vss_output3_weight {
    /* --- Conv2D (ckpt: output_layer_*.2, 1x1 4C->C, f32)  */
    const float* pre_conv2d_w;
    const float* pre_conv2d_b;

    /* --- SSM (all f32) --- */
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
        /*< mlp fc1 (ckpt: mlp.fc1, linear, int8) >*/
    struct qconv_int8 mlp_fc1;
        /*< mlp fc2 (ckpt: mlp.fc2, linear, int8) >*/
    struct qconv_int8 mlp_fc2;
};

/*< 6. PVSS Upstream skip structure (ckpt: skip_handler, 1x1 2C->C, int8 with bias) >*/
struct pvss_us_skip_weight {
    /*< Dec 0 (ckpt: layers_decoder_*.1.skip_handler) >*/
    struct qconv_int8 skip_conv_dec0;

    /*< Dec 1 (ckpt: layers_decoder_*.2.skip_handler) >*/
    struct qconv_int8 skip_conv_dec1;

    /*< Dec 2 (ckpt: layers_decoder_*.3.skip_handler) >*/
    struct qconv_int8 skip_conv_dec2;

    /*< Output 0 (ckpt: output_layer_*.0.skip_handler) >*/
    struct qconv_int8 skip_conv_out0;
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
    struct vss_output0_weight vss_output0_weight_mag;
    struct vss_output0_weight vss_output0_weight_pha;

        /*< 10. Out 1 >*/
    struct vss_output1_weight vss_output1_weight_mag;
    struct vss_output1_weight vss_output1_weight_pha;

        /*< 11. Out 3 >*/
    struct vss_output3_weight vss_output3_weight_mag;
    struct vss_output3_weight vss_output3_weight_pha;

        /* 12. Skip handler */
    struct pvss_us_skip_weight pvss_us_skip_weight_mag;
    struct pvss_us_skip_weight pvss_us_skip_weight_pha;
};

/* --------------------------------------------------------------------------------------------- */
/* 1. Weight file struct (f32 members) */
struct weight_file_spec {
    const char *member_name;
    size_t numel;
    const float **slot;
};

/* 2. Weight file struct (one int8 quantized layer: w_i8 / deq_scale / bias_i32 / act_scale) */
struct qconv_file_spec {
    const char *member_name;     /* base name; loader appends _w / _w_quant_scale / _w_dequant_scale / _b_i32 */
    size_t w_numel;              /* int8 weight element count */
    size_t c_out;                /* deq_scale / bias_i32 length */
    size_t act_scale_len;        /* C_IN (per-channel) or 1 (per-tensor) */
    int    has_bias;             /* 0 -> bias_i32 stays NULL */
    struct qconv_int8 *slot;
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
