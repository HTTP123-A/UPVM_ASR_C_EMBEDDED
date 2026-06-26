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

/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/* ---------------------------------------------- */
/* I. COMPUTATION */
/* --- F32 --- */
/*< 1. Patchembed: [1, F, T] => [16, H, W] >*/
void patch_embed_f32(
    /* ----- External activation/work buffers ---------------- */
                    float *embed_inout_buf,
                    float *embed_act_buf,
    /* ----- weights struct --------------------------------- */
                    const struct embed_weight* embed_weight,
    /* ----- Shape information ------------------------------ */
                    uint32_t C_IN, uint32_t C_MID, uint32_t C_EMBED,
                    uint32_t H_IN, uint32_t W_IN, uint32_t K_SIZE,
                    uint32_t STRIDE, uint32_t PAD);

/*< 2. PVSS Block for Downstream - Pytorch equivalent: PVSS_Block_DS_PVM (for encoder) >*/
void pvss_ds_mp_f32(
    /* ----- External activation/work buffers ---------------- */                
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,      // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
    /* ----- weights (direct pointers, no offset/struct) ----- */
                const struct pvss_ds_weight* pvss_ds_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz);

/*< 3. PVSS Block for Latent - Pytorch equivalent: PVSS_Block_DS_PVM (for Latent) >*/
void pvss_latent_mp_f32(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,      // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
    /* ----- weights (direct pointers, no offset/struct) ----- */
                const struct pvss_latent_weight* pvss_latent_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz);

/*< 4. PVSS Block for Decoder - Pytorch equivalent: PVSS_Block_PVM (for decoder) >*/
void pvss_us_mp_f32(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,        // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
    /* ----- weights (direct pointers, no offset/struct) ----- */
                const struct pvss_us_weight* pvss_us_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz, uint32_t input_selection, uint32_t norm_ena,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz);

/*< 5. VSS Block for Output - Pytorch equivalent: VSS_Block (for output3) >*/
void vss_output3_f32(
    /* ----- External activation/work buffers ---------------- */
                float *vss_inout_buf,      // input: [4*C,H,W], output written to front as [C,H,W]
                float *vss_residual_buf,   // [C,H,W]
                float *vss_act_buf,        // scratch for SSM/MLP path
                float *hidden_state_buf,   // 4 * D_INNER * N
    /* ----- weights ----------------------------------------- */
                const struct vss_output_weight *vss_output_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t D_INNER, uint32_t D_PROJECTION,  // D_PROJECTION = 2 * D_INNER
                uint16_t chunk_sz);

    /* --- TIMING VARIANT -- */
/*< 6. PVSS Block for Downstream - Pytorch equivalent: PVSS_Block_DS_PVM (for encoder) >*/
void pvss_ds_timing_mp_f32(
    /* ----- External activation/work buffers ---------------- */                
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,      // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
    /* ----- weights (direct pointers, no offset/struct) ----- */
                const struct pvss_ds_weight* pvss_ds_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz,
                struct timespec *pvss_time, struct timespec *mlp_time);

/*< 7. PVSS Block for Latent - Pytorch equivalent: PVSS_Block_DS_PVM (for Latent) >*/
void pvss_latent_timing_mp_f32(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,      // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
    /* ----- weights (direct pointers, no offset/struct) ----- */
                const struct pvss_latent_weight* pvss_latent_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz,
                struct timespec *pvss_time, struct timespec *mlp_time);

/*< 8. PVSS Block for Decoder - Pytorch equivalent: PVSS_Block_PVM (for decoder) >*/
void pvss_us_timing_mp_f32(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,        // (4 * (2 * SUB_D_INNER + R2N) + SUB_D_INNER) * L
                float *hidden_state_buf,    // 4 * SUB_D_INNER*N
    /* ----- weights (direct pointers, no offset/struct) ----- */
                const struct pvss_us_weight* pvss_us_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION, // SUB_D_PROJECTION = 2 * SUB_D_INNER
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz, uint32_t input_selection, uint32_t norm_ena,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz,
                struct timespec *pvss_time, struct timespec *mlp_time);

/*< 9. VSS Block for Output - Pytorch equivalent: VSS_Block (for output3) >*/
void vss_output3_timing_f32(
    /* ----- External activation/work buffers ---------------- */
                float *vss_inout_buf,      // input: [4*C,H,W], output written to front as [C,H,W]
                float *vss_residual_buf,   // [C,H,W]
                float *vss_act_buf,        // scratch for SSM/MLP path
                float *hidden_state_buf,   // 4 * D_INNER * N
    /* ----- weights ----------------------------------------- */
                const struct vss_output_weight *vss_output_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t D_INNER, uint32_t D_PROJECTION,  // D_PROJECTION = 2 * D_INNER
                uint16_t chunk_sz, struct timespec *pvss_time, struct timespec *mlp_time);

#endif /* __UPVM_ASR_MACRO_KERNELS_H__ */