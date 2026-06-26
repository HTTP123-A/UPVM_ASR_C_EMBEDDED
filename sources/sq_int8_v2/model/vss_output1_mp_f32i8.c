/* Mixed-precision (f32 + int8) VSS Output1 block for sq_int8_v0.
 * Output stage: the whole ss2d runs f32; only mlp fc1 / fc2 and sampler.expand
 * (pe) run int8 as an explicit quant -> int8 conv -> dequant sequence.
 * Deterministic image of output_layer_*.1: block norms present (ssm_in_norm /
 * mlp_norm), no skip handler (input read from base). */

#include "macro_kernels.h"
/*< 6. VSS Output1 - Pytorch equivalent: output_layer_*.1 (no skip, with norms) >*/
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
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz)
{
    /* ---------------- SSM branch (f32) ---------------- */
    /*< Pre-norm: in-place (input read from base, no skip handler) >*/
    layernorm_f32(pvss_inout_buf,
                vss_output1_weight->ssm_in_norm_w,
                vss_output1_weight->ssm_in_norm_b,
                C, L, 1e-5f);

    /*< SSM Residual >*/
    memcpy(pvss_residual_buf, pvss_inout_buf, (size_t)C * L * sizeof(float));

#pragma omp parallel num_threads((int)mp_parallel_factor)
{
    const uint32_t tid = (uint32_t)omp_get_thread_num();
    const uint32_t nth = (uint32_t)omp_get_num_threads();

    float *act_buf_slice_ptr = pvss_act_buf + (size_t)tid * act_buf_slice_sz;
    float *hid_buf_state_slice_ptr = hidden_state_buf + (size_t)tid * hid_buf_slice_sz;

    const uint32_t g_start_idx = (tid * reduce_factor) / nth;
    const uint32_t g_end_idx   = ((tid + 1U) * reduce_factor) / nth;

    for (uint32_t g = g_start_idx; g < g_end_idx; g++) {
        /*< In-projection (f32) >*/
        pointwise_conv2d_f32(pvss_inout_buf + (size_t)g * SUB_C * L,
                            vss_output1_weight->ssm_in_projection_w,
                            NULL,
                            act_buf_slice_ptr,
                            SUB_C, H, W, SUB_D_PROJECTION);

        /*< DW-CONV (f32, x) >*/
        depthwise_conv2d_f32(act_buf_slice_ptr,
                            vss_output1_weight->ssm_dw_conv2d_w,
                            vss_output1_weight->ssm_dw_conv2d_b,
                            act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                            SUB_D_INNER, H, W, DW_CONV2D_KERNEL_SIZE, 1U, 1U);

        /*< ReLU both x & z >*/
        relu_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                NULL, SUB_D_INNER * L, 1);
        relu_f32(act_buf_slice_ptr + (size_t)SUB_D_INNER * L,
                act_buf_slice_ptr + (size_t)4 * (2 * SUB_D_INNER + R2N) * L,
                SUB_D_INNER * L, 0);

        /*< CrossScan on x >*/
        cross_scan_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L, SUB_D_INNER, H, W);

        /*< Grouped point-wise convolution (x_proj) >*/
        grouped_pointwise_conv1d_type_major_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                                                vss_output1_weight->ssm_dw_conv1d_w,
                                                NULL,
                                                act_buf_slice_ptr,
                                                SUB_D_INNER, 4, L, R, N);

        /*< dts grouped point-wise convolution (dt_proj) >*/
        grouped_pointwise_conv1d_f32(act_buf_slice_ptr,
                                    vss_output1_weight->ssm_dt_projection_w,
                                    NULL,
                                    act_buf_slice_ptr + (size_t)4 * R2N * L,
                                    R, 4, L, SUB_D_INNER);

        /*< selective scan core - inplace >*/
        selective_scan_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                            act_buf_slice_ptr + (size_t)4 * R2N * L,
                            vss_output1_weight->ssm_A,
                            act_buf_slice_ptr + (size_t)4 * R * L,
                            act_buf_slice_ptr + (size_t)4 * (R + N) * L,
                            vss_output1_weight->ssm_Ds,
                            vss_output1_weight->ssm_delta_bias,
                            hid_buf_state_slice_ptr,
                            4, SUB_D_INNER, N, L, chunk_sz);

        /*< CrossMerge - inplace >*/
        cross_merge_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L, SUB_D_INNER, H, W);

        /*< LayerNorm after cross merge >*/
        layernorm_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                    vss_output1_weight->ssm_out_norm_w,
                    vss_output1_weight->ssm_out_norm_b,
                    SUB_D_INNER, L, 1e-5f);

        /*< Elementwise mul (x * z) >*/
        elemwise_mul_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                        act_buf_slice_ptr + (size_t)4 * (2 * SUB_D_INNER + R2N) * L,
                        SUB_D_INNER * L);

        /*< Out projection (f32) >*/
        pointwise_conv2d_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                            vss_output1_weight->ssm_out_projection_w,
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
    /*< MLP Residual >*/
    memcpy(pvss_residual_buf, pvss_inout_buf, (size_t)C * L * sizeof(float));

    /*< Pre-norm MLP >*/
    layernorm_f32(pvss_inout_buf,
                vss_output1_weight->mlp_norm_w,
                vss_output1_weight->mlp_norm_b,
                C, L, 1e-5f);

    /*< MLP input projection (int8): [C,H,W] -> [MLP_HIDDEN_DIM,H,W] : quant -> conv -> dequant >*/
    {
        const struct qconv_int8 *q = &vss_output1_weight->mlp_fc1;
        quant_f32_to_int8_perchannel(pvss_inout_buf, quant_buf, C, L, q->act_scale, 0);
        pointwise_conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf, C, H, W, MLP_HIDDEN_DIM);
        relu_int32(acc_buf, MLP_HIDDEN_DIM * L);   /* fused MLP ReLU on int32 acc (deq_scale>0 => exact vs dequant->relu_f32) */
        dequant_int32_to_f32_perchannel(acc_buf, pvss_act_buf + (size_t)C * L, MLP_HIDDEN_DIM, L, q->deq_scale);
    }

    /*< MLP output projection (int8): [MLP_HIDDEN_DIM,H,W] -> [C,H,W] : quant -> conv -> dequant >*/
    {
        const struct qconv_int8 *q = &vss_output1_weight->mlp_fc2;
        quant_f32_to_int8_perchannel(pvss_act_buf + (size_t)C * L, quant_buf, MLP_HIDDEN_DIM, L, q->act_scale, 0);
        pointwise_conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf, MLP_HIDDEN_DIM, H, W, C);
        dequant_int32_to_f32_perchannel(acc_buf, pvss_act_buf, C, L, q->deq_scale);
    }

    /*< Residual add >*/
    elemwise_add_f32(pvss_act_buf, pvss_residual_buf, C * L);

    /* ---------------- PatchExpanding ------------ */
    const uint32_t C_EXP = C * 2U;
    const uint32_t C_OUT = C / 2U;
    const uint32_t H_OUT = H * 2U;
    const uint32_t W_OUT = W * 2U;
    const uint32_t L_OUT = H_OUT * W_OUT;

    /*< PatchExpanding (int8): [C,H,W] -> [2C,H,W] : quant -> conv -> dequant >*/
    {
        const struct qconv_int8 *q = &vss_output1_weight->pe;
        quant_f32_to_int8_perchannel(pvss_act_buf, quant_buf, C, L, q->act_scale, 0);
        pointwise_conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf, C, H, W, C_EXP);
        dequant_int32_to_f32_perchannel(acc_buf, pvss_act_buf + (size_t)C * L, C_EXP, L, q->deq_scale);
    }

    /*< Pixel Shuffle (f32): [2C, H, W] -> [C/2, 2H, 2W] >*/
    const float *src = pvss_act_buf + (size_t)C * L;
    for (uint32_t oc = 0; oc < C_OUT; ++oc) {
        const uint32_t base_in_c = oc * 4U;
        for (uint32_t h = 0; h < H; ++h) {
            const uint32_t oh = h * 2U;
            for (uint32_t w = 0; w < W; ++w) {
                const uint32_t ow = w * 2U;
                const size_t in_idx0 = ((size_t)(base_in_c + 0U) * H + h) * W + w;
                const size_t in_idx1 = ((size_t)(base_in_c + 1U) * H + h) * W + w;
                const size_t in_idx2 = ((size_t)(base_in_c + 2U) * H + h) * W + w;
                const size_t in_idx3 = ((size_t)(base_in_c + 3U) * H + h) * W + w;
                const size_t out_base = ((size_t)oc * H_OUT + oh) * W_OUT + ow;
                pvss_inout_buf[out_base] = src[in_idx0];
                pvss_inout_buf[out_base + 1U] = src[in_idx1];
                pvss_inout_buf[out_base + W_OUT] = src[in_idx2];
                pvss_inout_buf[out_base + W_OUT + 1U] = src[in_idx3];
            }
        }
    }

    /*< PatchExpanding norm (f32) >*/
    layernorm_f32(pvss_inout_buf,
                vss_output1_weight->pe_norm_w,
                vss_output1_weight->pe_norm_b,
                C_OUT, L_OUT, 1e-5f);
}
