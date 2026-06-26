#ifndef _UPVM_ASR_MACRO_KERNELS_H_
#define _UPVM_ASR_MACRO_KERNELS_H_

/* ===================== INCLUDE SECTION ===================== */
/* <1. Built-in header > */
#include <omp.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>

/* <2. User headers > */
#include "datatypes.h"
#include "micro_kernels.h"

/* ===================== DEFINE ============================== */
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440F
#endif

/* Each block (macro kernel) runs its quantized layers as an explicit
 * quant -> int8 conv -> dequant sequence over the micro kernels:
 *   quant_f32_to_int8_perchannel   (act_scale is always [C_IN]; the 6 per-tensor
 *                                   layers are exactly the C_IN==1 ones, for which
 *                                   per-channel quant is bit-identical to per-tensor,
 *                                   so the kernel choice is deterministic - no
 *                                   act_scale_len branch is needed)
 *   pointwise/depthwise/conv2d_int8 (int32 accumulator, bias_i32 folded inside)
 *   dequant_int32_to_f32_perchannel ([C_OUT] deq_scale)
 * `quant_buf` (int8 input scratch) and `acc_buf` (int32 accumulator scratch) are
 * caller-provided; float islands around each layer stay f32. */

/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/* ---------------------------------------------- */
/* I. COMPUTATION (mixed precision f32 + int8) */
/* --- F32/INT8 BLOCKS -- */
/*< 1. Patchembed: [1, F, T] => [16, H, W] (conv1/conv2 int8; norm/gelu f32) >*/
void patch_embed_int8(
    /* ----- External activation/work buffers ---------------- */
                    float *embed_inout_buf,
                    float *embed_act_buf,
                    int8_t *quant_buf,          // int8 activation scratch (max of conv1/conv2)
                    int32_t *acc_buf,           // int32 accumulator scratch
    /* ----- weights struct --------------------------------- */
                    const struct embed_weight* embed_weight,
    /* ----- Shape information ------------------------------ */
                    uint32_t C_IN, uint32_t C_MID, uint32_t C_EMBED,
                    uint32_t H_IN, uint32_t W_IN, uint32_t K_SIZE,
                    uint32_t STRIDE, uint32_t PAD);

/*< 2. PVSS Block for Downstream (encoder); ss2d trio + fc1/fc2 int8, islands f32 >*/
void pvss_ds_mp_int8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,        // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
                int8_t *quant_buf,          // int8 scratch, sliced per thread by act_buf_slice_sz
                int32_t *acc_buf,           // int32 scratch, sliced per thread by act_buf_slice_sz
    /* ----- weights ---------------------------------------- */
                const struct pvss_ds_weight* pvss_ds_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz);

/*< 3. PVSS Block for Latent; ss2d trio + fc1/fc2 int8, islands f32 >*/
void pvss_latent_mp_int8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,        // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct pvss_latent_weight* pvss_latent_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz);

/*< 4. PVSS Block for Decoder (deterministic: input offset + norms always on);
 *    ss2d trio + fc1/fc2 + pe int8, islands f32 >*/
void pvss_us_mp_int8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,        // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct pvss_us_weight* pvss_us_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz);

/*< 5. VSS Output0 (no block norms, input offset); ss2d f32, fc1/fc2/pe int8 >*/
void vss_output0_mp_f32i8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,
                float *hidden_state_buf,
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct vss_output0_weight* vss_output0_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION,
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz);

/*< 6. VSS Output1 (block norms on, no input offset); ss2d f32, fc1/fc2/pe int8 >*/
void vss_output1_mp_f32i8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,
                float *hidden_state_buf,
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct vss_output1_weight* vss_output1_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION,
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz);

/*< 7. VSS Output3 (pre_conv + ss2d f32, fc1/fc2 int8, no grouped loop) >*/
void vss_output3_f32i8(
    /* ----- External activation/work buffers ---------------- */
                float *vss_inout_buf,      // input: [4*C,H,W], output written to front as [C,H,W]
                float *vss_residual_buf,   // [C,H,W]
                float *vss_act_buf,        // scratch for SSM/MLP path
                float *hidden_state_buf,   // 4 * D_INNER * N
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ----------------------------------------- */
                const struct vss_output3_weight *vss_output3_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t D_INNER, uint32_t D_PROJECTION,  // D_PROJECTION = 2 * D_INNER
                uint16_t chunk_sz);

/* --- TIMING VARIANTS (mixed precision f32 + int8) -- */
/* Each timing block == its NORMAL block above + two extra outputs (pvss_time,
 * mlp_time): clock_gettime is sampled at the SSM/PVSS->MLP boundary and at the
 * function end, so the detail-stage profile can split PVSS vs MLP. Identical
 * compute to the NORMAL block — only used by model_detail.c (DETAIL mode). */
/*< 8. PVSS Block for Downstream (encoder) - timing variant >*/
void pvss_ds_timing_mp_int8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,        // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct pvss_ds_weight* pvss_ds_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz,
    /* ----- Time ------------------------------------------- */
                struct timespec *pvss_time, struct timespec *mlp_time);

/*< 9. PVSS Block for Latent - timing variant >*/
void pvss_latent_timing_mp_int8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,        // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct pvss_latent_weight* pvss_latent_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz,
    /* ----- Time ------------------------------------------- */
                struct timespec *pvss_time, struct timespec *mlp_time);

/*< 10. PVSS Block for Decoder - timing variant (deterministic: input offset + norms always on) >*/
void pvss_us_timing_mp_int8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,        // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct pvss_us_weight* pvss_us_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz,
    /* ----- Time ------------------------------------------- */
                struct timespec *pvss_time, struct timespec *mlp_time);

/*< 11. VSS Output0 - timing variant (no block norms, input offset; ss2d f32, fc1/fc2/pe int8) >*/
void vss_output0_timing_mp_f32i8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,
                float *hidden_state_buf,
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct vss_output0_weight* vss_output0_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION,
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz,
    /* ----- Time ------------------------------------------- */
                struct timespec *pvss_time, struct timespec *mlp_time);

/*< 12. VSS Output1 - timing variant (block norms on, no input offset; ss2d f32, fc1/fc2/pe int8) >*/
void vss_output1_timing_mp_f32i8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,
                float *hidden_state_buf,
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct vss_output1_weight* vss_output1_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION,
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz,
    /* ----- Time ------------------------------------------- */
                struct timespec *pvss_time, struct timespec *mlp_time);

/*< 13. VSS Output3 - timing variant (pre_conv + ss2d f32, fc1/fc2 int8, no grouped loop) >*/
void vss_output3_timing_f32i8(
    /* ----- External activation/work buffers ---------------- */
                float *vss_inout_buf,      // input: [4*C,H,W], output written to front as [C,H,W]
                float *vss_residual_buf,   // [C,H,W]
                float *vss_act_buf,        // scratch for SSM/MLP path
                float *hidden_state_buf,   // 4 * D_INNER * N
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ----------------------------------------- */
                const struct vss_output3_weight *vss_output3_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t D_INNER, uint32_t D_PROJECTION,  // D_PROJECTION = 2 * D_INNER
                uint16_t chunk_sz,
    /* ----- Time ------------------------------------------- */
                struct timespec *pvss_time, struct timespec *mlp_time);

#endif /* __UPVM_ASR_MACRO_KERNELS_H__ */
