/* Timing variant of vss_output3_f32i8 (sq_int8_v0, DETAIL profile).
 * Identical mixed-precision compute to model/vss_output3_f32i8.c (pre_conv +
 * ss2d f32, mlp fc1/fc2 int8, no grouped loop / no block norms); adds pvss_time
 * (after the SSM residual add) and mlp_time (function end). */

#include "macro_kernels.h"
/*< 13. VSS Block for Output - timing variant (output3) >*/
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
                uint32_t D_INNER, uint32_t D_PROJECTION,
                uint16_t chunk_sz,
    /* ----- Time ------------------------------------------- */
                struct timespec *pvss_time, struct timespec *mlp_time)
{
    /* ---------------- Pre-conv (f32): [4C,H,W] -> [C,H,W] ---------------- */
    pointwise_conv2d_f32(vss_inout_buf,
                        vss_output3_weight->pre_conv2d_w,
                        vss_output3_weight->pre_conv2d_b,
                        vss_residual_buf,
                        4U * C, H, W, C);

    /* ---------------- SSM branch (f32) ---------------- */
    /* In-projection: [C,H,W] -> [2*D_INNER,H,W] = [x,z] */
    pointwise_conv2d_f32(vss_residual_buf,
                        vss_output3_weight->ssm_in_projection_w,
                        NULL,
                        vss_act_buf,
                        C, H, W, D_PROJECTION);

    /* DW-CONV on x chunk */
    depthwise_conv2d_f32(vss_act_buf,
                         vss_output3_weight->ssm_dw_conv2d_w,
                         vss_output3_weight->ssm_dw_conv2d_b,
                         vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
                         D_INNER, H, W, DW_CONV2D_KERNEL_SIZE, 1U, 1U);

    /* Activations on x and z */
    relu_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L, NULL, D_INNER * L, 1);  // x
    relu_f32(vss_act_buf + (size_t)D_INNER * L,
             vss_act_buf + (size_t)4U * (2U * D_INNER + R2N) * L,
             D_INNER * L, 0);  // z

    /* Cross-scan on x */
    cross_scan_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L, D_INNER, H, W);

    /* Grouped pointwise (type-major): produce [dts; Bs; Cs] */
    grouped_pointwise_conv1d_type_major_f32(
        vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
        vss_output3_weight->ssm_dw_conv1d_w,
        NULL,
        vss_act_buf,
        D_INNER, 4U, L, R, N);

    /* dt projection */
    grouped_pointwise_conv1d_f32(
        vss_act_buf,
        vss_output3_weight->ssm_dt_projection_w,
        NULL,
        vss_act_buf + (size_t)4U * R2N * L,
        R, 4U, L, D_INNER);

    /* Selective scan core */
    selective_scan_f32(
        vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
        vss_act_buf + (size_t)4U * R2N * L,
        vss_output3_weight->ssm_A,
        vss_act_buf + (size_t)4U * R * L,
        vss_act_buf + (size_t)4U * (R + N) * L,
        vss_output3_weight->ssm_Ds,
        vss_output3_weight->ssm_delta_bias,
        hidden_state_buf,
        4U, D_INNER, N, L, chunk_sz);

    /* Cross-merge + out_norm */
    cross_merge_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L, D_INNER, H, W);

    layernorm_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
                  vss_output3_weight->ssm_out_norm_w,
                  vss_output3_weight->ssm_out_norm_b,
                  D_INNER, L, 1e-5f);

    /* x * z */
    elemwise_mul_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
                     vss_act_buf + (size_t)4U * (2U * D_INNER + R2N) * L,
                     D_INNER * L);

    /* Out projection (f32): [D_INNER,H,W] -> [C,H,W] */
    pointwise_conv2d_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
                        vss_output3_weight->ssm_out_projection_w,
                        NULL,
                        vss_inout_buf,
                        D_INNER, H, W, C);

    /* SSM residual add */
    elemwise_add_f32(vss_inout_buf, vss_residual_buf, C * L);

    /* Get PVSS time */
    clock_gettime(CLOCK_MONOTONIC, pvss_time);

    /* ---------------- MLP branch ---------------- */
    memcpy(vss_residual_buf, vss_inout_buf, (size_t)C * L * sizeof(float));

    /* fc1 (int8): [C,H,W] -> [MLP_HIDDEN_DIM,H,W] : quant -> conv -> dequant
     * (C==1 here: a per-tensor act_scale layer, covered by per-channel quant with C=1) */
    {
        const struct qconv_int8 *q = &vss_output3_weight->mlp_fc1;
        quant_f32_to_int8_perchannel(vss_inout_buf, quant_buf, C, L, q->act_scale, 0);
        pointwise_conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf, C, H, W, MLP_HIDDEN_DIM);
        relu_int32(acc_buf, MLP_HIDDEN_DIM * L);   /* fused MLP ReLU on int32 acc (deq_scale>0 => exact vs dequant->relu_f32) */
        dequant_int32_to_f32_perchannel(acc_buf, vss_act_buf + (size_t)C * L, MLP_HIDDEN_DIM, L, q->deq_scale);
    }

    /* fc2 (int8): [MLP_HIDDEN_DIM,H,W] -> [C,H,W] : quant -> conv -> dequant */
    {
        const struct qconv_int8 *q = &vss_output3_weight->mlp_fc2;
        quant_f32_to_int8_perchannel(vss_act_buf + (size_t)C * L, quant_buf, MLP_HIDDEN_DIM, L, q->act_scale, 0);
        pointwise_conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf, MLP_HIDDEN_DIM, H, W, C);
        dequant_int32_to_f32_perchannel(acc_buf, vss_inout_buf, C, L, q->deq_scale);
    }

    /* MLP residual add */
    elemwise_add_f32(vss_inout_buf, vss_residual_buf, C * L);

    /* Get MLP time */
    clock_gettime(CLOCK_MONOTONIC, mlp_time);
}
