#ifndef __MODEL_H__
#define __MODEL_H__

/* ===================== INCLUDE SECTION ===================== */
/*<1. Built-in header >*/
#include <omp.h>
#include <errno.h>
#include <stdio.h>
#include <fftw3.h>
#include <stdlib.h>
#include <string.h>

/*<2. User headers >*/
#include "stft.h"
#include "datatypes.h"
#include "weight_loader_f32_i8.h"
#include "micro_kernels.h"
#include "macro_kernels.h"

/* ===================== DEFINE ============================== */
/*< 1. Macro functions >*/
#define FLOOR_DIV_U(a, b)       ((a) / (b))
#define CEIL_DIV_U(a, b)        (((a) + (b) - 1U) / (b))
#define ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))

/*< 2. Global macros >*/
#define INPUT_FREQ              16000U
#define OUTPUT_FREQ             48000U
#define DT_RANK_DIVISOR         16U
#define SSM_RATIO_NUM           2U
#define SSM_RATIO_DEN           1U
#define LOW_FREQ_HF             (((1U + (N_FFT / 2U)) * INPUT_FREQ) / OUTPUT_FREQ)
#define INNER_MP_PARALLEL       16U

/*< 3. STFT Configuration >*/
#define AUDIO_LENGTH            122640U
#define OVERLAP_LENGTH          2000U
#define N_FFT                   1024U
#define WIN_LENGTH              1024U
#define HOP_LENGTH              240U
#define FRAME_SIZE              (CEIL_DIV_U(N_FFT, 2) + 1U)
#define NUM_OF_FRAME            (1U + FLOOR_DIV_U(AUDIO_LENGTH, HOP_LENGTH))
#define INV_SQRT_NFFT           (1.0f / sqrtf((float)N_FFT))
    /* Additional for iSTFT */
#define PAD                     (N_FFT / 2U)
#define OLA_LEN                 (N_FFT + (HOP_LENGTH * (NUM_OF_FRAME - 1U)))

/*< 4. Layer >*/
    /*< (0) Input spectrograms >*/
#define SKIP_OFFSET_SPECTROGRAM 0U

    /*< (1) Patch Embedding >*/
#define C_IN_PATCH_EMBED        1U
#define C_MID_PATCH_EMBED       8U
#define C_OUT_PATCH_EMBED       16U
#define H_IN_PATCH_EMBED        512U
#define W_IN_PATCH_EMBED        512U
#define L_IN_PATCH_EMBED        (H_IN_PATCH_EMBED * W_IN_PATCH_EMBED)
#define K_SIZE_PATCH_EMBED      3U
#define STRIDE_PATCH_EMBED      2U
#define PAD_PATCH_EMBED         1U
#define SKIP_OFFSET_PATCH_EMBED (SKIP_OFFSET_SPECTROGRAM + (C_IN_PATCH_EMBED * L_IN_PATCH_EMBED))

    /*< (2) Encoder 0 >*/
#define C_ENC_0                 16U
#define H_ENC_0                 128U
#define W_ENC_0                 128U
#define L_ENC_0                 (H_ENC_0 * W_ENC_0)
#define REDUCTION_FACTOR_ENC_0  16U
#define SUB_C_ENC_0             (C_ENC_0 / REDUCTION_FACTOR_ENC_0)
#define R_ENC_0                 CEIL_DIV_U(SUB_C_ENC_0, DT_RANK_DIVISOR)
#define N_ENC_0                 1U
#define R2N_ENC_0               ((R_ENC_0) + 2U * (N_ENC_0))
#define DW_KERNEL_SZ_ENC_0      3U
#define MLP_HIDDEN_ENC_0        C_ENC_0
#define SUB_D_INNER_ENC_0       ((SUB_C_ENC_0 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_ENC_0  (SUB_D_INNER_ENC_0 * 2U)
#define CHUNK_SZ_ENC_0          64U
#define SKIP_OFFSET_ENC_0       (SKIP_OFFSET_PATCH_EMBED + (C_ENC_0 * L_ENC_0))
#define ACT_BUF_SLICE_ENC_0     ((4U * R2N_ENC_0 + 9U * SUB_D_INNER_ENC_0) * L_ENC_0)
#define HID_BUF_SLICE_ENC_0     (4U * SUB_D_INNER_ENC_0 * N_ENC_0)

    /*< (3) Encoder 1 >*/
#define C_ENC_1                 32U
#define H_ENC_1                 64U
#define W_ENC_1                 64U
#define L_ENC_1                 (H_ENC_1 * W_ENC_1)
#define REDUCTION_FACTOR_ENC_1  16U
#define SUB_C_ENC_1             (C_ENC_1 / REDUCTION_FACTOR_ENC_1)
#define R_ENC_1                 CEIL_DIV_U(SUB_C_ENC_1, DT_RANK_DIVISOR)
#define N_ENC_1                 1U
#define R2N_ENC_1               ((R_ENC_1) + 2U * (N_ENC_1))
#define DW_KERNEL_SZ_ENC_1      3U
#define MLP_HIDDEN_ENC_1        C_ENC_1
#define SUB_D_INNER_ENC_1       ((SUB_C_ENC_1 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_ENC_1  (SUB_D_INNER_ENC_1 * 2U)
#define CHUNK_SZ_ENC_1          64U
#define SKIP_OFFSET_ENC_1       (SKIP_OFFSET_ENC_0 + (C_ENC_1 * L_ENC_1))
#define ACT_BUF_SLICE_ENC_1     ((4U * R2N_ENC_1 + 9U * SUB_D_INNER_ENC_1) * L_ENC_1)
#define HID_BUF_SLICE_ENC_1     (4U * SUB_D_INNER_ENC_1 * N_ENC_1)

    /*< (4) Encoder 2 >*/
#define C_ENC_2                 64U
#define H_ENC_2                 32U
#define W_ENC_2                 32U
#define L_ENC_2                 (H_ENC_2 * W_ENC_2)
#define REDUCTION_FACTOR_ENC_2  16U
#define SUB_C_ENC_2             (C_ENC_2 / REDUCTION_FACTOR_ENC_2)
#define R_ENC_2                 CEIL_DIV_U(SUB_C_ENC_2, DT_RANK_DIVISOR)
#define N_ENC_2                 1U
#define R2N_ENC_2               ((R_ENC_2) + 2U * (N_ENC_2))
#define DW_KERNEL_SZ_ENC_2      3U
#define MLP_HIDDEN_ENC_2        C_ENC_2
#define SUB_D_INNER_ENC_2       ((SUB_C_ENC_2 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_ENC_2  (SUB_D_INNER_ENC_2 * 2U)
#define CHUNK_SZ_ENC_2          64U
#define SKIP_OFFSET_ENC_2       (SKIP_OFFSET_ENC_1 + (C_ENC_2 * L_ENC_2))
#define ACT_BUF_SLICE_ENC_2     ((4U * R2N_ENC_2 + 9U * SUB_D_INNER_ENC_2) * L_ENC_2)
#define HID_BUF_SLICE_ENC_2     (4U * SUB_D_INNER_ENC_2 * N_ENC_2)

    /*< (5) Latent >*/
#define C_LATENT                128U
#define H_LATENT                16U
#define W_LATENT                16U
#define L_LATENT                (H_LATENT * W_LATENT)
#define REDUCTION_FACTOR_LATENT 16U
#define SUB_C_LATENT            (C_LATENT / REDUCTION_FACTOR_LATENT)
#define R_LATENT                CEIL_DIV_U(SUB_C_LATENT, DT_RANK_DIVISOR)
#define N_LATENT                1U
#define R2N_LATENT              ((R_LATENT) + 2U * (N_LATENT))
#define DW_KERNEL_SZ_LATENT     3U
#define MLP_HIDDEN_LATENT       C_LATENT
#define SUB_D_INNER_LATENT      ((SUB_C_LATENT * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_LATENT (SUB_D_INNER_LATENT * 2U)
#define CHUNK_SZ_LATENT         64U
#define ACT_BUF_SLICE_LATENT    ((4U * R2N_LATENT + 9U * SUB_D_INNER_LATENT) * L_LATENT)
#define HID_BUF_SLICE_LATENT    (4U * SUB_D_INNER_LATENT * N_LATENT)

    /*< (6) Decoder 0 >*/
#define C_DEC_0                 128U
#define H_DEC_0                 16U
#define W_DEC_0                 16U
#define L_DEC_0                 (H_DEC_0 * W_DEC_0)
#define REDUCTION_FACTOR_DEC_0  8U
#define SUB_C_DEC_0             (C_DEC_0 / REDUCTION_FACTOR_DEC_0)
#define R_DEC_0                 CEIL_DIV_U(SUB_C_DEC_0, DT_RANK_DIVISOR)
#define N_DEC_0                 1U
#define R2N_DEC_0               ((R_DEC_0) + 2U * (N_DEC_0))
#define DW_KERNEL_SZ_DEC_0      3U
#define MLP_HIDDEN_DEC_0        C_DEC_0
#define SUB_D_INNER_DEC_0       ((SUB_C_DEC_0 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_DEC_0  (SUB_D_INNER_DEC_0 * 2U)
#define CHUNK_SZ_DEC_0          64U
#define ACT_BUF_SLICE_DEC_0     ((4U * R2N_DEC_0 + 9U * SUB_D_INNER_DEC_0) * L_DEC_0)
#define HID_BUF_SLICE_DEC_0     (4U * SUB_D_INNER_DEC_0 * N_DEC_0)

    /*< (7) Decoder 1 >*/
#define C_DEC_1                 64U
#define H_DEC_1                 32U
#define W_DEC_1                 32U
#define L_DEC_1                 (H_DEC_1 * W_DEC_1)
#define REDUCTION_FACTOR_DEC_1  8U
#define SUB_C_DEC_1             (C_DEC_1 / REDUCTION_FACTOR_DEC_1)
#define R_DEC_1                 CEIL_DIV_U(SUB_C_DEC_1, DT_RANK_DIVISOR)
#define N_DEC_1                 1U
#define R2N_DEC_1               ((R_DEC_1) + 2U * (N_DEC_1))
#define DW_KERNEL_SZ_DEC_1      3U
#define MLP_HIDDEN_DEC_1        C_DEC_1
#define SUB_D_INNER_DEC_1       ((SUB_C_DEC_1 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_DEC_1  (SUB_D_INNER_DEC_1 * 2U)
#define CHUNK_SZ_DEC_1          64U
#define ACT_BUF_SLICE_DEC_1     ((4U * R2N_DEC_1 + 9U * SUB_D_INNER_DEC_1) * L_DEC_1)
#define HID_BUF_SLICE_DEC_1     (4U * SUB_D_INNER_DEC_1 * N_DEC_1)

    /*< (8) Decoder 2 >*/
#define C_DEC_2                 32U
#define H_DEC_2                 64U
#define W_DEC_2                 64U
#define L_DEC_2                 (H_DEC_2 * W_DEC_2)
#define REDUCTION_FACTOR_DEC_2  8U
#define SUB_C_DEC_2             (C_DEC_2 / REDUCTION_FACTOR_DEC_2)
#define R_DEC_2                 CEIL_DIV_U(SUB_C_DEC_2, DT_RANK_DIVISOR)
#define N_DEC_2                 1U
#define R2N_DEC_2               ((R_DEC_2) + 2U * (N_DEC_2))
#define DW_KERNEL_SZ_DEC_2      3U
#define MLP_HIDDEN_DEC_2        C_DEC_2
#define SUB_D_INNER_DEC_2       ((SUB_C_DEC_2 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_DEC_2  (SUB_D_INNER_DEC_2 * 2U)
#define CHUNK_SZ_DEC_2          64U
#define ACT_BUF_SLICE_DEC_2     ((4U * R2N_DEC_2 + 9U * SUB_D_INNER_DEC_2) * L_DEC_2)
#define HID_BUF_SLICE_DEC_2     (4U * SUB_D_INNER_DEC_2 * N_DEC_2)

    /*< (9) Output 0 >*/
#define C_OUT_0                 16U
#define H_OUT_0                 128U
#define W_OUT_0                 128U
#define L_OUT_0                 (H_OUT_0 * W_OUT_0)
#define REDUCTION_FACTOR_OUT_0  4U
#define SUB_C_OUT_0             (C_OUT_0 / REDUCTION_FACTOR_OUT_0)
#define R_OUT_0                 CEIL_DIV_U(SUB_C_OUT_0, DT_RANK_DIVISOR)
#define N_OUT_0                 1U
#define R2N_OUT_0               ((R_OUT_0) + 2U * (N_OUT_0))
#define DW_KERNEL_SZ_OUT_0      3U
#define MLP_HIDDEN_OUT_0        (C_OUT_0 * 2U)
#define SUB_D_INNER_OUT_0       ((SUB_C_OUT_0 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_OUT_0  (SUB_D_INNER_OUT_0 * 2U)
#define CHUNK_SZ_OUT_0          64U
#define ACT_BUF_SLICE_OUT_0     ((4U * R2N_OUT_0 + 9U * SUB_D_INNER_OUT_0) * L_OUT_0)
#define HID_BUF_SLICE_OUT_0     (4U * SUB_D_INNER_OUT_0 * N_OUT_0)

    /*< (10) Output 1 >*/
#define C_OUT_1                 8U
#define H_OUT_1                 256U
#define W_OUT_1                 256U
#define L_OUT_1                 (H_OUT_1 * W_OUT_1)
#define C_OUT_1_OUT             CEIL_DIV_U(C_OUT_1, 2)
#define H_OUT_1_OUT             (H_OUT_1 * 2U)
#define W_OUT_1_OUT             (W_OUT_1 * 2U)
#define L_OUT_1_OUT             (H_OUT_1_OUT * W_OUT_1_OUT)
#define REDUCTION_FACTOR_OUT_1  4U
#define SUB_C_OUT_1             (C_OUT_1 / REDUCTION_FACTOR_OUT_1)
#define R_OUT_1                 CEIL_DIV_U(SUB_C_OUT_1, DT_RANK_DIVISOR)
#define N_OUT_1                 1U
#define R2N_OUT_1               ((R_OUT_1) + 2U * (N_OUT_1))
#define DW_KERNEL_SZ_OUT_1      3U
#define MLP_HIDDEN_OUT_1        (C_OUT_1 * 2U)
#define SUB_D_INNER_OUT_1       ((SUB_C_OUT_1 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define SUB_D_PROJECTION_OUT_1  (SUB_D_INNER_OUT_1 * 2U)
#define CHUNK_SZ_OUT_1          64U
#define ACT_BUF_SLICE_OUT_1     ((4U * R2N_OUT_1 + 9U * SUB_D_INNER_OUT_1) * L_OUT_1)
#define HID_BUF_SLICE_OUT_1     (4U * SUB_D_INNER_OUT_1 * N_OUT_1)

    /*< (11) Output 3 >*/
#define C_OUT_3                 1U
#define H_OUT_3                 512U
#define W_OUT_3                 512U
#define L_OUT_3                 (H_OUT_3 * W_OUT_3)
#define R_OUT_3                 CEIL_DIV_U(C_OUT_3, DT_RANK_DIVISOR)
#define N_OUT_3                 1U
#define R2N_OUT_3               ((R_OUT_3) + 2U * (N_OUT_3))
#define DW_KERNEL_SZ_OUT_3      3U
#define MLP_HIDDEN_OUT_3        (C_OUT_3 * 2U)
#define D_INNER_OUT_3           ((C_OUT_3 * SSM_RATIO_NUM) / SSM_RATIO_DEN)
#define D_PROJECTION_OUT_3      (D_INNER_OUT_3 * 2U)
#define CHUNK_SZ_OUT_3          64U

/*< 5. Buffer >*/
    /*< 1. Inout / Activation buffers >*/
#define INOUT_BUF_ELEMENT           (2U * C_OUT_1 * L_OUT_1 + W_IN_PATCH_EMBED) // OUTPUT 1 is largest
#define ACT_BUF_ELEMENT             ((4U * (R_OUT_3 + 2U * N_OUT_3) + 9U * D_INNER_OUT_3) * L_OUT_3 * CEIL_DIV_U(INNER_MP_PARALLEL,2U)) // OUTPUT 3 is largest (even if the parallel is counted)
#define LOW_FREQ_ELEMENT            (LOW_FREQ_HF * W_IN_PATCH_EMBED)
    /*< 2. Residual buffers >*/
#define INTERNAL_RES_BUF_ELEMENT    (C_OUT_1 * L_OUT_1) // Check again later
#define GLOBAL_RES_BUF_ELEMENT      ((C_IN_PATCH_EMBED * H_IN_PATCH_EMBED * W_IN_PATCH_EMBED) +\
                                     (C_ENC_0  * H_ENC_0  * W_ENC_0) +\
                                     (C_ENC_1  * H_ENC_1  * W_ENC_1) +\
                                     (C_ENC_2  * H_ENC_2  * W_ENC_2) +\
                                     (C_LATENT * H_LATENT * W_LATENT))
    /*< 3. Hidden buffers >*/
#define HIDDEN_STATE_BUF_ELEMENT    (4U * SUB_D_INNER_DEC_0 * N_DEC_0 * INNER_MP_PARALLEL) // DECODER 0 is the largest
    /*< 4. Quant/accumulator scratch (mixed precision) >*/
    /* Same element count as the f32 activation buffer: it is the largest single
     * block working set and is sliced the same way (per-thread by act_buf_slice_sz
     * in the parallel blocks; base for the sequential MLP/PE/output/patch-embed and
     * the split2 skip handlers), so it bounds every quantized layer's quant input
     * (int8) and conv accumulator (int32). */
#define QUANT_BUF_ELEMENT           ACT_BUF_ELEMENT   // int8 activation scratch
#define ACC_BUF_ELEMENT             ACT_BUF_ELEMENT   // int32 accumulator scratch

/* ===================== LOCAL FUNCTION ====================== */


/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/* ---------------------------------------------- */
/*< 1. Weight & Other buffers init >*/
void model_buf_init_f32_i8(
    /* ----- weights ----------------------------------------- */
                        struct upvm_asr_weight** model_weight,
                        char *weight_path,
    /* ----- External activation/work buffers ---------------- */
                        float **model_inout_buf_mag, float **model_inout_buf_pha,
                        float **model_internal_res_buf_mag, float **model_internal_res_buf_pha,
                        float **model_global_res_buf_mag, float **model_global_res_buf_pha,
                        float **model_act_buf_mag, float **model_act_buf_pha,
                        float **model_low_freq_buf_mag, float **model_low_freq_buf_pha,
                        float **hidden_state_buf_mag, float **hidden_state_buf_pha,
    /* ----- Quant/accumulator scratch (mixed precision) ----- */
                        int8_t **quant_buf_mag, int8_t **quant_buf_pha,
                        int32_t **acc_buf_mag, int32_t **acc_buf_pha);

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
                        /* ----- Quant/accumulator scratch ----- */
                        int8_t *quant_buf_mag, int8_t *quant_buf_pha,
                        int32_t *acc_buf_mag, int32_t *acc_buf_pha,
                        /* ----- STFT ------ */
                        float *hann_window_buf, float *frame, float *ola, float* wss,
                        fftwf_complex *spec, fftwf_plan *p_stft, fftwf_plan *p_istft, uint32_t win_offset);

/*< 3. Full dual-stream UPVM-ASR with timing profile (mixed precision f32 + int8) >*/
void dual_stream_unet_timing_profiling_mp_int8(float* audio, // [L_AUDIO]
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
                        /* ----- Quant/accumulator scratch ----- */
                        int8_t *quant_buf_mag, int8_t *quant_buf_pha,
                        int32_t *acc_buf_mag, int32_t *acc_buf_pha,
    /* ----- STFT -------------------------------------------- */
                        float *hann_window_buf, float *frame, float *ola, float* wss,
                        fftwf_complex *spec, fftwf_plan *p_stft, fftwf_plan *p_istft, uint32_t win_offset,
    /* ----- Time -------------------------------------------- */
                        struct timing_record_points *timing_record);

/*< 4. Full dual-stream UPVM-ASR with detail timing profile (mixed precision f32 + int8) >*/
void dual_stream_unet_detail_timing_profiling_mp_int8(float* audio, // [L_AUDIO]
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
                        /* ----- Quant/accumulator scratch ----- */
                        int8_t *quant_buf_mag, int8_t *quant_buf_pha,
                        int32_t *acc_buf_mag, int32_t *acc_buf_pha,
    /* ----- STFT -------------------------------------------- */
                        float *hann_window_buf, float *frame, float *ola, float* wss,
                        fftwf_complex *spec, fftwf_plan *p_stft, fftwf_plan *p_istft, uint32_t win_offset,
    /* ----- Time -------------------------------------------- */
                        struct detail_timing_record_points *detail_timing_record);

#endif /* __MODEL_H__ */