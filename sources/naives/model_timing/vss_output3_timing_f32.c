/* Split from sources\naives\macro_kernels.c */

#include "macro_kernels.h" 
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
                uint16_t chunk_sz, struct timespec *pvss_time, struct timespec *mlp_time)
{
    /* ---------------- Pre-conv (from output1 to output3 input) ---------------- */
    /* [4C,H,W] -> [C,H,W] */
    pointwise_conv2d_f32(vss_inout_buf,
                        vss_output_weight->pre_conv2d_w,
                        vss_output_weight->pre_conv2d_b,
                        vss_residual_buf,
                        4U * C, H, W, C);

    /* ---------------- SSM branch ---------------- */
    /* Residual for SSM block (pre-norm is Identity at output3) */
    /* vss_residual_buf already holds the reduced input [C,H,W]. */

    /* In-projection: [C,H,W] -> [2*D_INNER,H,W] = [x,z] */
    pointwise_conv2d_f32(vss_residual_buf,
                        vss_output_weight->ssm_in_projection_w,
                        NULL,
                        vss_act_buf,
                        C, H, W, D_PROJECTION);

    /* DW-CONV on x chunk */
    depthwise_conv2d_f32(vss_act_buf,
                         vss_output_weight->ssm_dw_conv2d_w,
                         vss_output_weight->ssm_dw_conv2d_b,
                         vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
                         D_INNER, H, W, DW_CONV2D_KERNEL_SIZE, 1U, 1U);

    /* Activations on x and z */
    relu_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
             NULL,
             D_INNER * L, 1);  // x

    relu_f32(vss_act_buf + (size_t)D_INNER * L,
             vss_act_buf + (size_t)4U * (2U * D_INNER + R2N) * L,
             D_INNER * L, 0);  // z

    /* Cross-scan on x */
    cross_scan_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L, D_INNER, H, W);

    /* Grouped pointwise (type-major): produce [dts; Bs; Cs] */
    grouped_pointwise_conv1d_type_major_f32(
        vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
        vss_output_weight->ssm_dw_conv1d_w,
        NULL,
        vss_act_buf,
        D_INNER,
        4U, L, R, N);

    /* dt projection */
    grouped_pointwise_conv1d_f32(
        vss_act_buf,
        vss_output_weight->ssm_dt_projection_w,
        NULL,
        vss_act_buf + (size_t)4U * R2N * L,
        R, 4U, L, D_INNER);

    /* Selective scan core */
    selective_scan_f32(
        vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
        vss_act_buf + (size_t)4U * R2N * L,
        vss_output_weight->ssm_A,
        vss_act_buf + (size_t)4U * R * L,
        vss_act_buf + (size_t)4U * (R + N) * L,
        vss_output_weight->ssm_Ds,
        vss_output_weight->ssm_delta_bias,
        hidden_state_buf,
        4U, D_INNER, N, L, chunk_sz);

    /* Cross-merge + out_norm */
    cross_merge_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L, D_INNER, H, W);

    layernorm_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
                  vss_output_weight->ssm_out_norm_w,
                  vss_output_weight->ssm_out_norm_b,
                  D_INNER, L, 1e-5f);

    /* x * z */
    elemwise_mul_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
                     vss_act_buf + (size_t)4U * (2U * D_INNER + R2N) * L,
                     D_INNER * L);

    /* Out projection: [D_INNER,H,W] -> [C,H,W] */
    pointwise_conv2d_f32(vss_act_buf + (size_t)4U * (D_INNER + R2N) * L,
                        vss_output_weight->ssm_out_projection_w,
                        NULL,
                        vss_inout_buf,
                        D_INNER, H, W, C);

    /* SSM residual add */
    elemwise_add_f32(vss_inout_buf, vss_residual_buf, C * L);

    /* Get PVSS time */
    clock_gettime(CLOCK_MONOTONIC, pvss_time);

    /* ---------------- MLP branch ---------------- */
    memcpy(vss_residual_buf, vss_inout_buf, (size_t)C * L * sizeof(float));

    /* fc1: [C,H,W] -> [MLP_HIDDEN_DIM,H,W] */
    pointwise_conv2d_f32(vss_inout_buf,
                        vss_output_weight->mlp_fc1_w,
                        vss_output_weight->mlp_fc1_b,
                        vss_act_buf + (size_t)C * L,
                        C, H, W, MLP_HIDDEN_DIM);

    /* activation */
    relu_f32(vss_act_buf + (size_t)C * L,
             NULL,
             MLP_HIDDEN_DIM * L, 1);

    /* fc2: [MLP_HIDDEN_DIM,H,W] -> [C,H,W] */
    pointwise_conv2d_f32(vss_act_buf + (size_t)C * L,
                        vss_output_weight->mlp_fc2_w,
                        vss_output_weight->mlp_fc2_b,
                        vss_inout_buf,
                        MLP_HIDDEN_DIM, H, W, C);

    /* MLP residual add */
    elemwise_add_f32(vss_inout_buf, vss_residual_buf, C * L);

    /* Get MLP time */
    clock_gettime(CLOCK_MONOTONIC, mlp_time);
}

/* --- INT8 -- */

