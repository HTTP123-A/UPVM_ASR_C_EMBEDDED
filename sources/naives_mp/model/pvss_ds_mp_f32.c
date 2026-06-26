/* Split from sources/naives_mp/macro_kernels.c */

#include "macro_kernels.h"
/*< 2. PVSS Block for Downstream - Pytorch equivalent: PVSS_Block_DS_PVM >*/                                        // ----- ==== ------- ==== ------- ===== ------- ====== -----
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
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz)
{
    /* ---------------- SSM branch ---------------- */
    /*< Pre-norm >*/
    /*< Pre-norm: in-place >*/
    layernorm_f32(pvss_inout_buf,
                pvss_ds_weight->ssm_in_norm_w,
                pvss_ds_weight->ssm_in_norm_b,
                C, L, 1e-5f);

    /*< SSM Residual >*/
    memcpy(pvss_residual_buf, pvss_inout_buf, (size_t)C * L * sizeof(float));


#pragma omp parallel num_threads((int)mp_parallel_factor)
{
    /*< Get thread id & total # of active threads in this group >*/
    const uint32_t tid = (uint32_t)omp_get_thread_num();
    const uint32_t nth = (uint32_t)omp_get_num_threads();

    /*< Get buffer slice pointer >*/
    float *act_buf_slice_ptr = pvss_act_buf + (size_t)tid * act_buf_slice_sz;
    float *hid_buf_state_slice_ptr = hidden_state_buf + (size_t)tid * hid_buf_slice_sz;

    /*< Get loop index >*/
    const uint32_t g_start_idx = (tid * reduce_factor) / nth;
    const uint32_t g_end_idx   = ((tid + 1U) * reduce_factor) / nth;

    for (uint32_t g = g_start_idx; g < g_end_idx; g++) {
        /*< In-projection - [x,z] channels >*/
        pointwise_conv2d_f32(pvss_inout_buf + (size_t)g * SUB_C * L,
                            pvss_ds_weight->ssm_in_projection_w,
                            NULL,
                            act_buf_slice_ptr,
                            SUB_C, H, W, SUB_D_PROJECTION);

        /*< DW-CONV (x) >*/
        depthwise_conv2d_f32(act_buf_slice_ptr,
                            pvss_ds_weight->ssm_dw_conv2d_w,
                            pvss_ds_weight->ssm_dw_conv2d_b,
                            act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                            SUB_D_INNER, H, W, DW_CONV2D_KERNEL_SIZE, 1U, 1U);

        /*< ReLU both x & z >*/
        relu_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                NULL,
                SUB_D_INNER * L, 1); // X chunk

        relu_f32(act_buf_slice_ptr + (size_t)SUB_D_INNER * L,
                act_buf_slice_ptr + (size_t)4 * (2 * SUB_D_INNER + R2N) * L,
                SUB_D_INNER * L, 0); // Z chunk

        /*< CrossScan on x >*/
        cross_scan_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L, SUB_D_INNER, H, W);

        /*< Grouped point-wise convolution >*/
        grouped_pointwise_conv1d_type_major_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                                                pvss_ds_weight->ssm_dw_conv1d_w,
                                                NULL,
                                                act_buf_slice_ptr,
                                                SUB_D_INNER,
                                                4, L, R, N);

        /*< dts grouped point-wise convolution >
         *
         * After this grouped point-wise conv:
         *      act_buf_slice_ptr[0 --> (4 * R2N * L - 1)]: [dts; Bs; Cs]
         *      act_buf_slice_ptr[(4 * R2N * L) --> (4 * R2N * L) + (4 * D_INNER * L)]: [~dts~; Bs; Cs; gpw_conv_dts]
         */
        grouped_pointwise_conv1d_f32(act_buf_slice_ptr,
                                    pvss_ds_weight->ssm_dt_projection_w,
                                    NULL,
                                    act_buf_slice_ptr + (size_t)4 * R2N * L,
                                    R, 4, L, SUB_D_INNER);

        /*< selective scan core - inplace >*/
        selective_scan_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                            act_buf_slice_ptr + (size_t)4 * R2N * L,     // grouped point-wise conv-ed dts
                            pvss_ds_weight->ssm_A,                  // As
                            act_buf_slice_ptr + (size_t)4 * R * L,       // Bs
                            act_buf_slice_ptr + (size_t)4 * (R + N) * L, // Cs
                            pvss_ds_weight->ssm_Ds,                 // Ds
                            pvss_ds_weight->ssm_delta_bias,         // delta_bias
                            hid_buf_state_slice_ptr,                       // hidden_state
                            4, SUB_D_INNER, N, L, chunk_sz);

        /*< CrossMerge on output of selective scan - inplace >*/
        cross_merge_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L, SUB_D_INNER, H, W);

        /*< LayerNorm after cross merge >*/
        layernorm_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                    pvss_ds_weight->ssm_out_norm_w,
                    pvss_ds_weight->ssm_out_norm_b,
                    SUB_D_INNER, L, 1e-5f);

        /*< Elementwise mul after selective-scan & LN >*/
        elemwise_mul_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,   // x after cross merge/norm
                        act_buf_slice_ptr + (size_t)4 * (2 * SUB_D_INNER + R2N) * L, // z
                        SUB_D_INNER * L);

        /*< Out projection - [x] >*/
        pointwise_conv2d_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                            pvss_ds_weight->ssm_out_projection_w,
                            NULL,
                            pvss_inout_buf + (size_t)g * SUB_C * L,
                            SUB_D_INNER, H, W, SUB_C);

        /*< Residual add >*/
        elemwise_add_f32(pvss_inout_buf + (size_t)g * SUB_C * L,
                        pvss_residual_buf + (size_t)g * SUB_C * L,
                        SUB_C * L);
    }
}

    /* ---------------- MLP branch ---------------- */
    /*< Downsampling dimensions >*/
    const uint32_t H2 = H / 2U;
    const uint32_t W2 = W / 2U;
    const uint32_t C2 = C * 2U;
    const uint32_t L2 = H2 * W2;

    /*< MLP Residual >*/
    memcpy(pvss_residual_buf, pvss_inout_buf, (size_t)C * L * sizeof(float));

    /*< Pre-norm MLP >*/
    layernorm_f32(pvss_inout_buf,
                pvss_ds_weight->mlp_norm_w,
                pvss_ds_weight->mlp_norm_b,
                C, L, 1e-5f);

    /*< MLP input projection: [C, H, W] -> [MLP_HIDDEN_DIM, H, W] >*/
    pointwise_conv2d_f32(pvss_inout_buf,
                        pvss_ds_weight->mlp_fc1_w,
                        pvss_ds_weight->mlp_fc1_b,
                        pvss_act_buf + (size_t)C * L,
                        C, H, W, MLP_HIDDEN_DIM);

    /*< MLP ReLU >*/
    relu_f32(pvss_act_buf + (size_t)C * L,
            NULL,
            MLP_HIDDEN_DIM * L, 1);

    /*< MLP sumpool: [MLP_HIDDEN_DIM, H, W] -> [MLP_HIDDEN_DIM, H/2, W/2] >*/
    depthwise_conv2d_f32(pvss_act_buf + (size_t)C * L,
                        pvss_ds_weight->mlp_sumpool_w,
                        NULL,
                        pvss_act_buf,
                        MLP_HIDDEN_DIM, H, W, 2U, 2U, 0U);

    /*< MLP res sumpool: [C, H, W] -> [C, H/2, W/2] >*/
    depthwise_conv2d_f32(pvss_residual_buf,
                        pvss_ds_weight->mlp_sumpool_w,
                        NULL,
                        pvss_act_buf + (size_t)MLP_HIDDEN_DIM * L2,
                        C, H, W, 2U, 2U, 0U);

    /*< MLP output projection: [MLP_HIDDEN_DIM, H/2, W/2] -> [2C, H/2, W/2] >*/
    pointwise_conv2d_f32(pvss_act_buf,
                        pvss_ds_weight->mlp_fc2_w,
                        pvss_ds_weight->mlp_fc2_b,
                        pvss_inout_buf,
                        MLP_HIDDEN_DIM, H2, W2, C2);

    /*< MLP res output projection: [C, H/2, W/2] -> [2C, H/2, W/2] >*/
    pointwise_conv2d_f32(pvss_act_buf + (size_t)MLP_HIDDEN_DIM * L2,
                        pvss_ds_weight->mlp_fc_res_w,
                        pvss_ds_weight->mlp_fc_res_b,
                        pvss_residual_buf,
                        C, H2, W2, C2);

    /*< Residual add >*/
    elemwise_add_f32(pvss_inout_buf,
                    pvss_residual_buf,
                    C2 * L2);
}
