/* Split from sources/sq_int8_v0/model.c */

#include "model.h"
/*< 2. Full dual-stream UPVM-ASR (mixed precision f32 + int8) >*/
void dual_stream_unet_mp_int8(float* audio, // [L_AUDIO]
    /* ----- weights ----------------------------------------- */
                        const struct upvm_asr_weight* model_weight,
    /* ----- External activation/work buffers ---------------- */
                        /* ----- Model ----- */
                        float *model_inout_buf_mag, float *model_inout_buf_pha,
                        float *model_internal_res_buf_mag, float *model_internal_res_buf_pha,
                        float *model_global_res_buf_mag, float *model_global_res_buf_pha,
                        float *model_act_buf_mag, float *model_act_buf_pha,
                        float *model_low_freq_buf_mag, float *model_low_freq_buf_pha,
                        float *hidden_state_buf_mag, float *hidden_state_buf_pha,
    /* ----- Quant/accumulator scratch ----------------------- */
                        int8_t *quant_buf_mag, int8_t *quant_buf_pha,
                        int32_t *acc_buf_mag, int32_t *acc_buf_pha,
    /* ----- STFT -------------------------------------------- */
                        float *hann_window_buf, float *frame, float *ola, float* wss,
                        fftwf_complex *spec, fftwf_plan *p_stft, fftwf_plan *p_istft, uint32_t win_offset)
{
    /*< ---------------------------------------------------------------------------------------------------------------------- >*/
    /*< STFT >*/
    stft_f32(audio, model_inout_buf_mag, model_inout_buf_pha,
            hann_window_buf, frame, spec, p_stft,
            AUDIO_LENGTH, N_FFT, WIN_LENGTH,
            HOP_LENGTH, FRAME_SIZE, NUM_OF_FRAME, win_offset, INV_SQRT_NFFT);

    /*< openMP for dual-stream:  >*/
#pragma omp parallel num_threads(2) default(shared)
{
    /*< Ensure we have enough thread >*/
    #pragma omp single
    {
        int nth = omp_get_num_threads();
        if (nth < 2) {
            fprintf(stderr, "FATAL: need >=2 OpenMP threads, got %d\n", nth);
            fflush(stderr);
            abort();
        }
    }

    /*< Get the thread id >*/
    const uint32_t tid = (uint32_t)omp_get_thread_num();

        /*< Copy low freq >*/
    if (tid == 0) {
        memcpy(model_low_freq_buf_mag, model_inout_buf_mag, LOW_FREQ_ELEMENT * sizeof(float)); // Copy from 0th to HF * 512 elements

        /*< spectrogram global residual >*/ // Copy from the W_IN_PATCH_EMBED offset to the end of spectrogram since the we have to left W_IN_PATCH_EMBED ~ 512 for 1st freq copy back low freq at the end
        memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_SPECTROGRAM, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_IN_PATCH_EMBED * L_IN_PATCH_EMBED * sizeof(float));
    }
    else {
        memcpy(model_low_freq_buf_pha, model_inout_buf_pha, LOW_FREQ_ELEMENT * sizeof(float));
    }

    /*< ---------------------------------------------------------------------------------------------------------------------- >*/
    /*< --- PatchEmbed --- >*/
    if (tid == 0) {
        /*< Mag >*/
        patch_embed_int8(model_inout_buf_mag  + (size_t)W_IN_PATCH_EMBED, // Similar, reserved W_IN_PATCH_EMBED elements for 1st freq
                    model_act_buf_mag,
                    quant_buf_mag, acc_buf_mag,
                    (const struct embed_weight*) &(model_weight->embed_weight_mag),
                    C_IN_PATCH_EMBED, C_MID_PATCH_EMBED, C_OUT_PATCH_EMBED,
                    H_IN_PATCH_EMBED, W_IN_PATCH_EMBED, K_SIZE_PATCH_EMBED,
                    STRIDE_PATCH_EMBED, PAD_PATCH_EMBED);

        /*< PatchEmbed global residual >*/
        memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_ENC_0 * L_ENC_0 * sizeof(float));
    }
    else {
        /*< Pha >*/
        patch_embed_int8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                    model_act_buf_pha,
                    quant_buf_pha, acc_buf_pha,
                    (const struct embed_weight*) &(model_weight->embed_weight_pha),
                    C_IN_PATCH_EMBED, C_MID_PATCH_EMBED, C_OUT_PATCH_EMBED,
                    H_IN_PATCH_EMBED, W_IN_PATCH_EMBED, K_SIZE_PATCH_EMBED,
                    STRIDE_PATCH_EMBED, PAD_PATCH_EMBED);

        /*< PatchEmbed global residual >*/
        memcpy(model_global_res_buf_pha + (size_t)SKIP_OFFSET_PATCH_EMBED - SKIP_OFFSET_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, C_ENC_0 * L_ENC_0 * sizeof(float));
    }

#pragma omp barrier
#pragma omp single
{
        /*< Interaction >*/
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_0 * L_ENC_0); // mag = mag + phase
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_0 * L_ENC_0); // phase = phase + mag
}

    /*< --- Encoder 0 --- >*/
    if (tid == 0) {
        /*< Mag >*/
        pvss_ds_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                model_internal_res_buf_mag,
                model_act_buf_mag,
                hidden_state_buf_mag,
                quant_buf_mag, acc_buf_mag,
                (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc0_mag),
                C_ENC_0, H_ENC_0, W_ENC_0, L_ENC_0,
                R_ENC_0, N_ENC_0, R2N_ENC_0,
                DW_KERNEL_SZ_ENC_0, MLP_HIDDEN_ENC_0,
                SUB_D_INNER_ENC_0, SUB_D_PROJECTION_ENC_0,
                SUB_C_ENC_0, REDUCTION_FACTOR_ENC_0, CHUNK_SZ_ENC_0,
                INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_0, (size_t)HID_BUF_SLICE_ENC_0);
        /*< Encoder 0 global residual >*/
        memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_0, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_ENC_1 * L_ENC_1 * sizeof(float));
    }
    else {
        /*< Pha >*/
        pvss_ds_mp_int8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                model_internal_res_buf_pha,
                model_act_buf_pha,
                hidden_state_buf_pha,
                quant_buf_pha, acc_buf_pha,
                (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc0_pha),
                C_ENC_0, H_ENC_0, W_ENC_0, L_ENC_0,
                R_ENC_0, N_ENC_0, R2N_ENC_0,
                DW_KERNEL_SZ_ENC_0, MLP_HIDDEN_ENC_0,
                SUB_D_INNER_ENC_0, SUB_D_PROJECTION_ENC_0,
                SUB_C_ENC_0, REDUCTION_FACTOR_ENC_0, CHUNK_SZ_ENC_0,
                INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_0, (size_t)HID_BUF_SLICE_ENC_0);
        /*< Encoder 0 global residual >*/
        memcpy(model_global_res_buf_pha + (size_t)SKIP_OFFSET_ENC_0 - SKIP_OFFSET_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, C_ENC_1 * L_ENC_1 * sizeof(float));
    }

#pragma omp barrier
#pragma omp single
{
        /*< Interaction >*/
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_1 * L_ENC_1); // mag = mag + phase
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_1 * L_ENC_1); // phase = phase + mag
}

    /*< --- Encoder 1 --- >*/
    if (tid == 0) {
        /*< Mag >*/
        pvss_ds_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                model_internal_res_buf_mag,
                model_act_buf_mag,
                hidden_state_buf_mag,
                quant_buf_mag, acc_buf_mag,
                (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc1_mag),
                C_ENC_1, H_ENC_1, W_ENC_1, L_ENC_1,
                R_ENC_1, N_ENC_1, R2N_ENC_1,
                DW_KERNEL_SZ_ENC_1, MLP_HIDDEN_ENC_1,
                SUB_D_INNER_ENC_1, SUB_D_PROJECTION_ENC_1,
                SUB_C_ENC_1, REDUCTION_FACTOR_ENC_1, CHUNK_SZ_ENC_1,
                INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_1, (size_t)HID_BUF_SLICE_ENC_1);
        /*< Encoder 1 global residual >*/
        memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_1, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_ENC_2 * L_ENC_2 * sizeof(float));
    }
    else {
        /*< Pha >*/
        pvss_ds_mp_int8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                model_internal_res_buf_pha,
                model_act_buf_pha,
                hidden_state_buf_pha,
                quant_buf_pha, acc_buf_pha,
                (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc1_pha),
                C_ENC_1, H_ENC_1, W_ENC_1, L_ENC_1,
                R_ENC_1, N_ENC_1, R2N_ENC_1,
                DW_KERNEL_SZ_ENC_1, MLP_HIDDEN_ENC_1,
                SUB_D_INNER_ENC_1, SUB_D_PROJECTION_ENC_1,
                SUB_C_ENC_1, REDUCTION_FACTOR_ENC_1, CHUNK_SZ_ENC_1,
                INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_1, (size_t)HID_BUF_SLICE_ENC_1);
        /*< Encoder 1 global residual >*/
        memcpy(model_global_res_buf_pha + (size_t)SKIP_OFFSET_ENC_1 - SKIP_OFFSET_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, C_ENC_2 * L_ENC_2 * sizeof(float));
    }


#pragma omp barrier
#pragma omp single
{
        /*< Interaction >*/
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_2 * L_ENC_2); // mag = mag + phase
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_2 * L_ENC_2); // phase = phase + mag
}

    /*< --- Encoder 2 --- >*/
    if (tid == 0) {
        /*< Mag >*/
        pvss_ds_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                model_internal_res_buf_mag,
                model_act_buf_mag,
                hidden_state_buf_mag,
                quant_buf_mag, acc_buf_mag,
                (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc2_mag),
                C_ENC_2, H_ENC_2, W_ENC_2, L_ENC_2,
                R_ENC_2, N_ENC_2, R2N_ENC_2,
                DW_KERNEL_SZ_ENC_2, MLP_HIDDEN_ENC_2,
                SUB_D_INNER_ENC_2, SUB_D_PROJECTION_ENC_2,
                SUB_C_ENC_2, REDUCTION_FACTOR_ENC_2, CHUNK_SZ_ENC_2,
                INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_2, (size_t)HID_BUF_SLICE_ENC_2);
        /*< Encoder 2 global residual >*/
        memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_2, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_LATENT * L_LATENT * sizeof(float));
    }
    else {
        /*< Pha >*/
        pvss_ds_mp_int8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                model_internal_res_buf_pha,
                model_act_buf_pha,
                hidden_state_buf_pha,
                quant_buf_pha, acc_buf_pha,
                (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc2_pha),
                C_ENC_2, H_ENC_2, W_ENC_2, L_ENC_2,
                R_ENC_2, N_ENC_2, R2N_ENC_2,
                DW_KERNEL_SZ_ENC_2, MLP_HIDDEN_ENC_2,
                SUB_D_INNER_ENC_2, SUB_D_PROJECTION_ENC_2,
                SUB_C_ENC_2, REDUCTION_FACTOR_ENC_2, CHUNK_SZ_ENC_2,
                INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_2, (size_t)HID_BUF_SLICE_ENC_2);
        /*< Encoder 2 global residual >*/
        memcpy(model_global_res_buf_pha + (size_t)SKIP_OFFSET_ENC_2 - SKIP_OFFSET_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, C_LATENT * L_LATENT * sizeof(float));
    }

#pragma omp barrier
#pragma omp single
{
        /*< Interaction >*/
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_LATENT * L_LATENT); // mag = mag + phase
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_LATENT * L_LATENT); // phase = phase + mag
}

    /*< ---------------------------------------------------------------------------------------------------------------------- >*/
    /*< --- Latent ------ >*/
    if (tid == 0) {
        /*< Mag >*/
        pvss_latent_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                        model_internal_res_buf_mag,
                        model_act_buf_mag,
                        hidden_state_buf_mag,
                        quant_buf_mag, acc_buf_mag,
                        (const struct pvss_latent_weight*) &(model_weight->pvss_latent_weight_mag),
                        C_LATENT, H_LATENT, W_LATENT, L_LATENT,
                        R_LATENT, N_LATENT, R2N_LATENT,
                        DW_KERNEL_SZ_LATENT, MLP_HIDDEN_LATENT,
                        SUB_D_INNER_LATENT, SUB_D_PROJECTION_LATENT,
                        SUB_C_LATENT, REDUCTION_FACTOR_LATENT, CHUNK_SZ_LATENT,
                        INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_LATENT, (size_t)HID_BUF_SLICE_LATENT);
    }
    else {
        /*< Pha >*/
        pvss_latent_mp_int8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                        model_internal_res_buf_pha,
                        model_act_buf_pha,
                        hidden_state_buf_pha,
                        quant_buf_pha, acc_buf_pha,
                        (const struct pvss_latent_weight*) &(model_weight->pvss_latent_weight_pha),
                        C_LATENT, H_LATENT, W_LATENT, L_LATENT,
                        R_LATENT, N_LATENT, R2N_LATENT,
                        DW_KERNEL_SZ_LATENT, MLP_HIDDEN_LATENT,
                        SUB_D_INNER_LATENT, SUB_D_PROJECTION_LATENT,
                        SUB_C_LATENT, REDUCTION_FACTOR_LATENT, CHUNK_SZ_LATENT,
                        INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_LATENT, (size_t)HID_BUF_SLICE_LATENT);
    }

    /*< Interaction >*/
#pragma omp barrier
#pragma omp single
{
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_LATENT * L_LATENT); // mag = mag + phase
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_LATENT * L_LATENT); // phase = phase + mag
}

    /*< ---------------------------------------------------------------------------------------------------------------------- >*/
    /*< --- Decoder 0 --- >*/
    if (tid == 0) {
        /*< Skip handler (int8): quant(x_inout) + quant(x_res) -> split2 -> dequant >*/
        {
            const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_mag.skip_conv_dec0;
            quant_f32_to_int8_perchannel(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                                        quant_buf_mag, C_DEC_0, L_DEC_0, q->act_scale, 0);
            quant_f32_to_int8_perchannel(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_2,
                                        quant_buf_mag + (size_t)C_DEC_0 * L_DEC_0, C_DEC_0, L_DEC_0, q->act_scale + C_DEC_0, 0);
            pointwise_conv2d_split2_int8(quant_buf_mag, quant_buf_mag + (size_t)C_DEC_0 * L_DEC_0,
                                        q->w_i8, q->bias_i32, acc_buf_mag, C_DEC_0, H_DEC_0, W_DEC_0, C_DEC_0);
            dequant_int32_to_f32_perchannel(acc_buf_mag,
                                        model_inout_buf_mag + (size_t)C_DEC_0 * L_DEC_0 + W_IN_PATCH_EMBED, C_DEC_0, L_DEC_0, q->deq_scale);
        }

        /*< Mag >*/
        pvss_us_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_mag,
                    model_act_buf_mag,
                    hidden_state_buf_mag,
                    quant_buf_mag, acc_buf_mag,
                    (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec0_mag),
                    C_DEC_0, H_DEC_0, W_DEC_0, L_DEC_0,
                    R_DEC_0, N_DEC_0, R2N_DEC_0,
                    DW_KERNEL_SZ_DEC_0, MLP_HIDDEN_DEC_0,
                    SUB_D_INNER_DEC_0, SUB_D_PROJECTION_DEC_0,
                    SUB_C_DEC_0, REDUCTION_FACTOR_DEC_0, CHUNK_SZ_DEC_0,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_0, (size_t)HID_BUF_SLICE_DEC_0);
    }
    else {
        /*< Skip handler (int8) >*/
        {
            const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_pha.skip_conv_dec0;
            quant_f32_to_int8_perchannel(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                                        quant_buf_pha, C_DEC_0, L_DEC_0, q->act_scale, 0);
            quant_f32_to_int8_perchannel(model_global_res_buf_pha + (size_t)SKIP_OFFSET_ENC_2 - SKIP_OFFSET_PATCH_EMBED,
                                        quant_buf_pha + (size_t)C_DEC_0 * L_DEC_0, C_DEC_0, L_DEC_0, q->act_scale + C_DEC_0, 0);
            pointwise_conv2d_split2_int8(quant_buf_pha, quant_buf_pha + (size_t)C_DEC_0 * L_DEC_0,
                                        q->w_i8, q->bias_i32, acc_buf_pha, C_DEC_0, H_DEC_0, W_DEC_0, C_DEC_0);
            dequant_int32_to_f32_perchannel(acc_buf_pha,
                                        model_inout_buf_pha + (size_t)C_DEC_0 * L_DEC_0 + W_IN_PATCH_EMBED, C_DEC_0, L_DEC_0, q->deq_scale);
        }

        /*< Pha >*/
        pvss_us_mp_int8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_pha,
                    model_act_buf_pha,
                    hidden_state_buf_pha,
                    quant_buf_pha, acc_buf_pha,
                    (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec0_pha),
                    C_DEC_0, H_DEC_0, W_DEC_0, L_DEC_0,
                    R_DEC_0, N_DEC_0, R2N_DEC_0,
                    DW_KERNEL_SZ_DEC_0, MLP_HIDDEN_DEC_0,
                    SUB_D_INNER_DEC_0, SUB_D_PROJECTION_DEC_0,
                    SUB_C_DEC_0, REDUCTION_FACTOR_DEC_0, CHUNK_SZ_DEC_0,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_0, (size_t)HID_BUF_SLICE_DEC_0);
    }

#pragma omp barrier
#pragma omp single
{
        /*< Interaction >*/
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_DEC_1 * L_DEC_1); // mag = mag + phase
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_DEC_1 * L_DEC_1); // phase = phase + mag
}

    /*< --- Decoder 1 --- >*/
    if (tid == 0) {
        /*< Skip handler (int8) >*/
        {
            const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_mag.skip_conv_dec1;
            quant_f32_to_int8_perchannel(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                                        quant_buf_mag, C_DEC_1, L_DEC_1, q->act_scale, 0);
            quant_f32_to_int8_perchannel(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_1,
                                        quant_buf_mag + (size_t)C_DEC_1 * L_DEC_1, C_DEC_1, L_DEC_1, q->act_scale + C_DEC_1, 0);
            pointwise_conv2d_split2_int8(quant_buf_mag, quant_buf_mag + (size_t)C_DEC_1 * L_DEC_1,
                                        q->w_i8, q->bias_i32, acc_buf_mag, C_DEC_1, H_DEC_1, W_DEC_1, C_DEC_1);
            dequant_int32_to_f32_perchannel(acc_buf_mag,
                                        model_inout_buf_mag + (size_t)C_DEC_1 * L_DEC_1 + W_IN_PATCH_EMBED, C_DEC_1, L_DEC_1, q->deq_scale);
        }
        /*< Mag >*/
        pvss_us_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_mag,
                    model_act_buf_mag,
                    hidden_state_buf_mag,
                    quant_buf_mag, acc_buf_mag,
                    (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec1_mag),
                    C_DEC_1, H_DEC_1, W_DEC_1, L_DEC_1,
                    R_DEC_1, N_DEC_1, R2N_DEC_1,
                    DW_KERNEL_SZ_DEC_1, MLP_HIDDEN_DEC_1,
                    SUB_D_INNER_DEC_1, SUB_D_PROJECTION_DEC_1,
                    SUB_C_DEC_1, REDUCTION_FACTOR_DEC_1, CHUNK_SZ_DEC_1,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_1, (size_t)HID_BUF_SLICE_DEC_1);
    }
    else {
        /*< Skip handler (int8) >*/
        {
            const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_pha.skip_conv_dec1;
            quant_f32_to_int8_perchannel(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                                        quant_buf_pha, C_DEC_1, L_DEC_1, q->act_scale, 0);
            quant_f32_to_int8_perchannel(model_global_res_buf_pha + (size_t)SKIP_OFFSET_ENC_1 - SKIP_OFFSET_PATCH_EMBED,
                                        quant_buf_pha + (size_t)C_DEC_1 * L_DEC_1, C_DEC_1, L_DEC_1, q->act_scale + C_DEC_1, 0);
            pointwise_conv2d_split2_int8(quant_buf_pha, quant_buf_pha + (size_t)C_DEC_1 * L_DEC_1,
                                        q->w_i8, q->bias_i32, acc_buf_pha, C_DEC_1, H_DEC_1, W_DEC_1, C_DEC_1);
            dequant_int32_to_f32_perchannel(acc_buf_pha,
                                        model_inout_buf_pha + (size_t)C_DEC_1 * L_DEC_1 + W_IN_PATCH_EMBED, C_DEC_1, L_DEC_1, q->deq_scale);
        }

        /*< Pha >*/
        pvss_us_mp_int8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_pha,
                    model_act_buf_pha,
                    hidden_state_buf_pha,
                    quant_buf_pha, acc_buf_pha,
                    (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec1_pha),
                    C_DEC_1, H_DEC_1, W_DEC_1, L_DEC_1,
                    R_DEC_1, N_DEC_1, R2N_DEC_1,
                    DW_KERNEL_SZ_DEC_1, MLP_HIDDEN_DEC_1,
                    SUB_D_INNER_DEC_1, SUB_D_PROJECTION_DEC_1,
                    SUB_C_DEC_1, REDUCTION_FACTOR_DEC_1, CHUNK_SZ_DEC_1,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_1, (size_t)HID_BUF_SLICE_DEC_1);
    }

#pragma omp barrier
#pragma omp single
{
        /*< Interaction >*/
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_DEC_2 * L_DEC_2); // mag = mag + phase
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_DEC_2 * L_DEC_2); // phase = phase + mag
}

    /*< --- Decoder 2 --- >*/
    if (tid == 0) {
        /*< Skip handler (int8) >*/
        {
            const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_mag.skip_conv_dec2;
            quant_f32_to_int8_perchannel(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                                        quant_buf_mag, C_DEC_2, L_DEC_2, q->act_scale, 0);
            quant_f32_to_int8_perchannel(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_0,
                                        quant_buf_mag + (size_t)C_DEC_2 * L_DEC_2, C_DEC_2, L_DEC_2, q->act_scale + C_DEC_2, 0);
            pointwise_conv2d_split2_int8(quant_buf_mag, quant_buf_mag + (size_t)C_DEC_2 * L_DEC_2,
                                        q->w_i8, q->bias_i32, acc_buf_mag, C_DEC_2, H_DEC_2, W_DEC_2, C_DEC_2);
            dequant_int32_to_f32_perchannel(acc_buf_mag,
                                        model_inout_buf_mag + (size_t)C_DEC_2 * L_DEC_2 + W_IN_PATCH_EMBED, C_DEC_2, L_DEC_2, q->deq_scale);
        }
        /*< Mag >*/
        pvss_us_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                        model_internal_res_buf_mag,
                        model_act_buf_mag,
                        hidden_state_buf_mag,
                        quant_buf_mag, acc_buf_mag,
                        (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec2_mag),
                        C_DEC_2, H_DEC_2, W_DEC_2, L_DEC_2,
                        R_DEC_2, N_DEC_2, R2N_DEC_2,
                        DW_KERNEL_SZ_DEC_2, MLP_HIDDEN_DEC_2,
                        SUB_D_INNER_DEC_2, SUB_D_PROJECTION_DEC_2,
                        SUB_C_DEC_2, REDUCTION_FACTOR_DEC_2, CHUNK_SZ_DEC_2,
                        INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_2, (size_t)HID_BUF_SLICE_DEC_2);
    }
    else {
        /*< Skip handler (int8) >*/
        {
            const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_pha.skip_conv_dec2;
            quant_f32_to_int8_perchannel(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                                        quant_buf_pha, C_DEC_2, L_DEC_2, q->act_scale, 0);
            quant_f32_to_int8_perchannel(model_global_res_buf_pha + (size_t)SKIP_OFFSET_ENC_0 - SKIP_OFFSET_PATCH_EMBED,
                                        quant_buf_pha + (size_t)C_DEC_2 * L_DEC_2, C_DEC_2, L_DEC_2, q->act_scale + C_DEC_2, 0);
            pointwise_conv2d_split2_int8(quant_buf_pha, quant_buf_pha + (size_t)C_DEC_2 * L_DEC_2,
                                        q->w_i8, q->bias_i32, acc_buf_pha, C_DEC_2, H_DEC_2, W_DEC_2, C_DEC_2);
            dequant_int32_to_f32_perchannel(acc_buf_pha,
                                        model_inout_buf_pha + (size_t)C_DEC_2 * L_DEC_2 + W_IN_PATCH_EMBED, C_DEC_2, L_DEC_2, q->deq_scale);
        }

        /*< Pha >*/
        pvss_us_mp_int8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_pha,
                    model_act_buf_pha,
                    hidden_state_buf_pha,
                    quant_buf_pha, acc_buf_pha,
                    (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec2_pha),
                    C_DEC_2, H_DEC_2, W_DEC_2, L_DEC_2,
                    R_DEC_2, N_DEC_2, R2N_DEC_2,
                    DW_KERNEL_SZ_DEC_2, MLP_HIDDEN_DEC_2,
                    SUB_D_INNER_DEC_2, SUB_D_PROJECTION_DEC_2,
                    SUB_C_DEC_2, REDUCTION_FACTOR_DEC_2, CHUNK_SZ_DEC_2,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_2, (size_t)HID_BUF_SLICE_DEC_2);
    }

#pragma omp barrier
#pragma omp single
{
        /*< Interaction >*/
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_OUT_0 * L_OUT_0); // mag = mag + phase
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_OUT_0 * L_OUT_0); // phase = phase + mag
}

    /*< ---------------------------------------------------------------------------------------------------------------------- >*/
    /*< --- Output 0 ---- >*/
    if (tid == 0) {
        /*< Skip handler (int8) >*/
        {
            const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_mag.skip_conv_out0;
            quant_f32_to_int8_perchannel(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                                        quant_buf_mag, C_OUT_0, L_OUT_0, q->act_scale, 0);
            quant_f32_to_int8_perchannel(model_global_res_buf_mag + (size_t)SKIP_OFFSET_PATCH_EMBED,
                                        quant_buf_mag + (size_t)C_OUT_0 * L_OUT_0, C_OUT_0, L_OUT_0, q->act_scale + C_OUT_0, 0);
            pointwise_conv2d_split2_int8(quant_buf_mag, quant_buf_mag + (size_t)C_OUT_0 * L_OUT_0,
                                        q->w_i8, q->bias_i32, acc_buf_mag, C_OUT_0, H_OUT_0, W_OUT_0, C_OUT_0);
            dequant_int32_to_f32_perchannel(acc_buf_mag,
                                        model_inout_buf_mag + (size_t)C_OUT_0 * L_OUT_0 + W_IN_PATCH_EMBED, C_OUT_0, L_OUT_0, q->deq_scale);
        }

        /*< Mag >*/
        vss_output0_mp_f32i8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_mag,
                    model_act_buf_mag,
                    hidden_state_buf_mag,
                    quant_buf_mag, acc_buf_mag,
                    (const struct vss_output0_weight*) &(model_weight->vss_output0_weight_mag),
                    C_OUT_0, H_OUT_0, W_OUT_0, L_OUT_0,
                    R_OUT_0, N_OUT_0, R2N_OUT_0,
                    DW_KERNEL_SZ_OUT_0, MLP_HIDDEN_OUT_0,
                    SUB_D_INNER_OUT_0, SUB_D_PROJECTION_OUT_0,
                    SUB_C_OUT_0, REDUCTION_FACTOR_OUT_0, CHUNK_SZ_OUT_0,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_OUT_0, (size_t)HID_BUF_SLICE_OUT_0);
    }
    else {
        /*< Skip handler (int8) >*/
        {
            const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_pha.skip_conv_out0;
            quant_f32_to_int8_perchannel(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                                        quant_buf_pha, C_OUT_0, L_OUT_0, q->act_scale, 0);
            quant_f32_to_int8_perchannel(model_global_res_buf_pha + (size_t)SKIP_OFFSET_PATCH_EMBED - SKIP_OFFSET_PATCH_EMBED,
                                        quant_buf_pha + (size_t)C_OUT_0 * L_OUT_0, C_OUT_0, L_OUT_0, q->act_scale + C_OUT_0, 0);
            pointwise_conv2d_split2_int8(quant_buf_pha, quant_buf_pha + (size_t)C_OUT_0 * L_OUT_0,
                                        q->w_i8, q->bias_i32, acc_buf_pha, C_OUT_0, H_OUT_0, W_OUT_0, C_OUT_0);
            dequant_int32_to_f32_perchannel(acc_buf_pha,
                                        model_inout_buf_pha + (size_t)C_OUT_0 * L_OUT_0 + W_IN_PATCH_EMBED, C_OUT_0, L_OUT_0, q->deq_scale);
        }

        /*< Pha >*/
        vss_output0_mp_f32i8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_pha,
                    model_act_buf_pha,
                    hidden_state_buf_pha,
                    quant_buf_pha, acc_buf_pha,
                    (const struct vss_output0_weight*) &(model_weight->vss_output0_weight_pha),
                    C_OUT_0, H_OUT_0, W_OUT_0, L_OUT_0,
                    R_OUT_0, N_OUT_0, R2N_OUT_0,
                    DW_KERNEL_SZ_OUT_0, MLP_HIDDEN_OUT_0,
                    SUB_D_INNER_OUT_0, SUB_D_PROJECTION_OUT_0,
                    SUB_C_OUT_0, REDUCTION_FACTOR_OUT_0, CHUNK_SZ_OUT_0,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_OUT_0, (size_t)HID_BUF_SLICE_OUT_0);
    }

    /*< --- Output 1 ---- >*/
    if (tid == 0) {
        /*< Mag >*/
        vss_output1_mp_f32i8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_mag,
                    model_act_buf_mag,
                    hidden_state_buf_mag,
                    quant_buf_mag, acc_buf_mag,
                    (const struct vss_output1_weight*) &(model_weight->vss_output1_weight_mag),
                    C_OUT_1, H_OUT_1, W_OUT_1, L_OUT_1,
                    R_OUT_1, N_OUT_1, R2N_OUT_1,
                    DW_KERNEL_SZ_OUT_1, MLP_HIDDEN_OUT_1,
                    SUB_D_INNER_OUT_1, SUB_D_PROJECTION_OUT_1,
                    SUB_C_OUT_1, REDUCTION_FACTOR_OUT_1, CHUNK_SZ_OUT_1,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_OUT_1, (size_t)HID_BUF_SLICE_OUT_1);
    }
    else {
        /*< Pha >*/
        vss_output1_mp_f32i8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                    model_internal_res_buf_pha,
                    model_act_buf_pha,
                    hidden_state_buf_pha,
                    quant_buf_pha, acc_buf_pha,
                    (const struct vss_output1_weight*) &(model_weight->vss_output1_weight_pha),
                    C_OUT_1, H_OUT_1, W_OUT_1, L_OUT_1,
                    R_OUT_1, N_OUT_1, R2N_OUT_1,
                    DW_KERNEL_SZ_OUT_1, MLP_HIDDEN_OUT_1,
                    SUB_D_INNER_OUT_1, SUB_D_PROJECTION_OUT_1,
                    SUB_C_OUT_1, REDUCTION_FACTOR_OUT_1, CHUNK_SZ_OUT_1,
                    INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_OUT_1, (size_t)HID_BUF_SLICE_OUT_1);
    }

    /*< --- Output 3 ---- >*/
    if (tid == 0) {
        /*< Mag >*/
        vss_output3_f32i8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED,
                        model_internal_res_buf_mag,
                        model_act_buf_mag,
                        hidden_state_buf_mag,
                        quant_buf_mag, acc_buf_mag,
                        (const struct vss_output3_weight*) &(model_weight->vss_output3_weight_mag),
                        C_OUT_3, H_OUT_3, W_OUT_3, L_OUT_3,
                        R_OUT_3, N_OUT_3, R2N_OUT_3,
                        DW_KERNEL_SZ_OUT_3, MLP_HIDDEN_OUT_3,
                        D_INNER_OUT_3, D_PROJECTION_OUT_3, CHUNK_SZ_OUT_3);

        /*< Residual add on mag (final [1, 512, 512]) >*/
        elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_global_res_buf_mag + (size_t)SKIP_OFFSET_SPECTROGRAM, (size_t)C_IN_PATCH_EMBED * L_IN_PATCH_EMBED); // mag = mag + residual (spectrogram)

        /*< --- Freq Replace  >*/
        memcpy(model_inout_buf_mag, model_low_freq_buf_mag, LOW_FREQ_ELEMENT * sizeof(float)); // Copy back low freq to the 0th ~ HF * 512 elements

    }
    else {
        /*< Pha >*/
        vss_output3_f32i8(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED,
                        model_internal_res_buf_pha,
                        model_act_buf_pha,
                        hidden_state_buf_pha,
                        quant_buf_pha, acc_buf_pha,
                        (const struct vss_output3_weight*) &(model_weight->vss_output3_weight_pha),
                        C_OUT_3, H_OUT_3, W_OUT_3, L_OUT_3,
                        R_OUT_3, N_OUT_3, R2N_OUT_3,
                        DW_KERNEL_SZ_OUT_3, MLP_HIDDEN_OUT_3,
                        D_INNER_OUT_3, D_PROJECTION_OUT_3, CHUNK_SZ_OUT_3);

        /*< --- Freq Replace  >*/
        memcpy(model_inout_buf_pha, model_low_freq_buf_pha, LOW_FREQ_ELEMENT * sizeof(float));
    }
} // End main openMP

    /*< ---------------------------------------------------------------------------------------------------------------------- >*/
    /*< iSTFT >*/
    // This part is for iSTFT create [1, 513, 512]
    istft_f32(model_inout_buf_mag, model_inout_buf_pha, audio,
            hann_window_buf, frame, ola, wss, spec, p_istft,
            AUDIO_LENGTH, N_FFT, WIN_LENGTH, HOP_LENGTH, PAD,
            OLA_LEN, FRAME_SIZE, NUM_OF_FRAME, win_offset, INV_SQRT_NFFT);
}
