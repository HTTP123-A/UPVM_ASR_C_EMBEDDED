/* Mixed-precision (f32 + int8) PVSS upstream (decoder) block for sq_int8_v0.
 * Decoder-only & deterministic: input is always read from the skip-handler
 * offset (+C*L) and both block norms are always applied (no input_selection /
 * norm_ena flags). int8 layers (explicit quant -> int8 conv -> dequant on the
 * micro kernels): ss2d in_proj / dw_conv2d / out_proj (per group), mlp fc1 / fc2,
 * sampler.expand (pe). f32 islands: norms, x_proj/dt, selective scan, cross
 * scan/merge, gated mul, residual add, pixel shuffle, pe_norm. */

#include "macro_kernels.h"
/*< 4. PVSS Block for Decoder - Pytorch equivalent: PVSS_Block_PVM (for decoder) >*/
void pvss_us_mp_int8(
    /* ----- External activation/work buffers ---------------- */
                float *pvss_inout_buf,
                float *pvss_residual_buf,
                float *pvss_act_buf,
                float *hidden_state_buf,
                int8_t *quant_buf,
                int32_t *acc_buf,
    /* ----- weights ---------------------------------------- */
                const struct pvss_us_weight* pvss_us_weight,
    /* ----- Shape information ------------------------------- */
                uint32_t C, uint32_t H, uint32_t W, uint32_t L,
                uint32_t R, uint32_t N, uint32_t R2N,
                uint32_t DW_CONV2D_KERNEL_SIZE, uint32_t MLP_HIDDEN_DIM,
                uint32_t SUB_D_INNER, uint32_t SUB_D_PROJECTION,
                uint32_t SUB_C, uint32_t reduce_factor, uint16_t chunk_sz,
                uint32_t mp_parallel_factor, size_t act_buf_slice_sz, size_t hid_buf_slice_sz)
{
    /* ---------------- SSM branch ---------------- */
    /*< Input is the skip-handler output (offset by C*L) >*/
    float *pvss_inout_buf_ptr = pvss_inout_buf + (size_t)C * L;

    /*< Pre-norm: in-place >*/
    layernorm_f32(pvss_inout_buf_ptr,
                pvss_us_weight->ssm_in_norm_w,
                pvss_us_weight->ssm_in_norm_b,
                C, L, 1e-5f);

    /*< SSM Residual >*/
    memcpy(pvss_residual_buf, pvss_inout_buf_ptr, (size_t)C * L * sizeof(float));

#pragma omp parallel num_threads((int)mp_parallel_factor)
{
    /*< Get thread id & total # of active threads in this group >*/
    const uint32_t tid = (uint32_t)omp_get_thread_num();
    const uint32_t nth = (uint32_t)omp_get_num_threads();

    /*< Get buffer slice pointer >*/
    float *act_buf_slice_ptr = pvss_act_buf + (size_t)tid * act_buf_slice_sz;
    float *hid_buf_state_slice_ptr = hidden_state_buf + (size_t)tid * hid_buf_slice_sz;
    int8_t  *q_slice  = quant_buf + (size_t)tid * act_buf_slice_sz;
    int32_t *acc_slice = acc_buf  + (size_t)tid * act_buf_slice_sz;

    /*< Get loop index >*/
    const uint32_t g_start_idx = (tid * reduce_factor) / nth;
    const uint32_t g_end_idx   = ((tid + 1U) * reduce_factor) / nth;

    for (uint32_t g = g_start_idx; g < g_end_idx; g++) {
        /*< In-projection (int8): quant -> conv -> { requant x | dequant z } (v5) >*/
        {
            const struct qconv_int8 *q = &pvss_us_weight->ssm_in_projection;
            const float *x = pvss_inout_buf_ptr + (size_t)g * SUB_C * L;
            quant_f32_to_int8_perchannel(x, q_slice, SUB_C, L, q->act_scale, 0);
            pointwise_conv2d_int8(q_slice, q->w_i8, q->bias_i32, acc_slice, SUB_C, H, W, SUB_D_PROJECTION);
            /* v5: x branch [0,SUB_D_INNER) requant'd straight to int8 (in_proj dequant + dw_conv quant
             *     fused: requant_scale = in_proj.deq / dw_conv.act_next), reusing q_slice in place.
             *     z branch [SUB_D_INNER,SUB_D_PROJECTION) keeps its f32 dequant. */
            requant_int32_to_int8_perchannel_fixed(acc_slice, q_slice, SUB_D_INNER, L, pvss_us_weight->ssm_requant_mul, pvss_us_weight->ssm_requant_shift); /* v7: integer fixed-point requant (mul/shift ~= deq/act_next); replaces v5 f32 SSM requant */
            dequant_int32_to_f32_perchannel(acc_slice + (size_t)SUB_D_INNER * L,
                                            act_buf_slice_ptr + (size_t)SUB_D_INNER * L,
                                            SUB_D_PROJECTION - SUB_D_INNER, L,
                                            q->deq_scale + SUB_D_INNER);
        }

        /*< DW-CONV (int8, x): conv -> relu(int32) -> dequant; quant fused into the in_proj requant (v5) >*/
        {
            const struct qconv_int8 *q = &pvss_us_weight->ssm_dw_conv2d;
            float *y = act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L;
            /* v5: x is already int8 in q_slice from the in_proj requant above (dw_conv quant fused away). */
            depthwise_conv2d_int8(q_slice, q->w_i8, q->bias_i32, acc_slice, SUB_D_INNER, H, W, DW_CONV2D_KERNEL_SIZE);
            relu_int32(acc_slice, SUB_D_INNER * L);   /* fused SSM x ReLU on int32 acc (deq_scale>0 => exact vs dequant->relu_f32) */
            dequant_int32_to_f32_perchannel(acc_slice, y, SUB_D_INNER, L, q->deq_scale);
        }

        /*< ReLU z (x already ReLU'd on the int32 dw-conv acc above) >*/
        relu_f32(act_buf_slice_ptr + (size_t)SUB_D_INNER * L,
                act_buf_slice_ptr + (size_t)4 * (2 * SUB_D_INNER + R2N) * L,
                SUB_D_INNER * L, 0); // Z chunk

        /*< CrossScan on x >*/
        cross_scan_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L, SUB_D_INNER, H, W);

        /*< Grouped point-wise convolution >*/
        grouped_pointwise_conv1d_type_major_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                                                pvss_us_weight->ssm_dw_conv1d_w,
                                                NULL,
                                                act_buf_slice_ptr,
                                                SUB_D_INNER,
                                                4, L, R, N);

        /*< dts grouped point-wise convolution >*/
        grouped_pointwise_conv1d_f32(act_buf_slice_ptr,
                                    pvss_us_weight->ssm_dt_projection_w,
                                    NULL,
                                    act_buf_slice_ptr + (size_t)4 * R2N * L,
                                    R, 4, L, SUB_D_INNER);

        /*< selective scan core - inplace >*/
        selective_scan_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                            act_buf_slice_ptr + (size_t)4 * R2N * L,        // grouped point-wise conv-ed dts
                            pvss_us_weight->ssm_A,                          // As
                            act_buf_slice_ptr + (size_t)4 * R * L,          // Bs
                            act_buf_slice_ptr + (size_t)4 * (R + N) * L,    // Cs
                            pvss_us_weight->ssm_Ds,                         // Ds
                            pvss_us_weight->ssm_delta_bias,                 // delta_bias
                            hid_buf_state_slice_ptr,                        // hidden_state
                            4, SUB_D_INNER, N, L, chunk_sz);

        /*< CrossMerge on output of selective scan - inplace >*/
        cross_merge_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L, SUB_D_INNER, H, W);

        /*< LayerNorm after cross merge >*/
        layernorm_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,
                    pvss_us_weight->ssm_out_norm_w,
                    pvss_us_weight->ssm_out_norm_b,
                    SUB_D_INNER, L, 1e-5f);

        /*< Elementwise mul after selective-scan & LN >*/
        elemwise_mul_f32(act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L,   // x after cross merge/norm
                        act_buf_slice_ptr + (size_t)4 * (2 * SUB_D_INNER + R2N) * L, // z
                        SUB_D_INNER * L);

        /*< Out projection (int8): quant -> conv -> dequant >*/
        {
            const struct qconv_int8 *q = &pvss_us_weight->ssm_out_projection;
            const float *x = act_buf_slice_ptr + (size_t)4 * (SUB_D_INNER + R2N) * L;
            quant_f32_to_int8_perchannel(x, q_slice, SUB_D_INNER, L, q->act_scale, 0);
            pointwise_conv2d_int8(q_slice, q->w_i8, q->bias_i32, acc_slice, SUB_D_INNER, H, W, SUB_C);
            dequant_int32_to_f32_perchannel(acc_slice, pvss_inout_buf + (size_t)g * SUB_C * L, SUB_C, L, q->deq_scale);
        }

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
                pvss_us_weight->mlp_norm_w,
                pvss_us_weight->mlp_norm_b,
                C, L, 1e-5f);

    /*< MLP input projection (int8): [C,H,W] -> [MLP_HIDDEN_DIM,H,W] : quant -> conv -> relu -> requant >*/
    {
        const struct qconv_int8 *q = &pvss_us_weight->mlp_fc1;
        quant_f32_to_int8_perchannel(pvss_inout_buf, quant_buf, C, L, q->act_scale, 0);
        pointwise_conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf, C, H, W, MLP_HIDDEN_DIM);
        relu_int32(acc_buf, MLP_HIDDEN_DIM * L);   /* fused MLP ReLU on int32 acc (deq_scale>0 => exact vs dequant->relu_f32) */
        requant_int32_to_int8_perchannel_fixed(acc_buf, quant_buf, MLP_HIDDEN_DIM, L, pvss_us_weight->mlp_requant_mul, pvss_us_weight->mlp_requant_shift); /* v6: integer fixed-point requant (mul/shift ~= deq/act_next); replaces v4 f32 requant */
    }

    /*< MLP output projection (int8): [MLP_HIDDEN_DIM,H,W] -> [C,H,W] : conv -> dequant (quant folded into fc1's requant) >*/
    {
        const struct qconv_int8 *q = &pvss_us_weight->mlp_fc2;
        pointwise_conv2d_int8(quant_buf, q->w_i8, q->bias_i32, acc_buf, MLP_HIDDEN_DIM, H, W, C);
        dequant_int32_to_f32_perchannel(acc_buf, pvss_act_buf, C, L, q->deq_scale);
    }

    /*< Residual add >*/
    elemwise_add_f32(pvss_act_buf,
                    pvss_residual_buf,
                    C * L);

    /* ---------------- PatchExpanding ------------ */
    /*< Output dimension >*/
    const uint32_t C_EXP = C * 2U;
    const uint32_t C_OUT = C / 2U;
    const uint32_t H_OUT = H * 2U;
    const uint32_t W_OUT = W * 2U;
    const uint32_t L_OUT = H_OUT * W_OUT;

    /*< PatchExpanding (int8): [C,H,W] -> [2C,H,W] : quant -> conv -> dequant >*/
    {
        const struct qconv_int8 *q = &pvss_us_weight->pe;
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

                const uint32_t in_c0 = base_in_c + 0U;
                const uint32_t in_c1 = base_in_c + 1U;
                const uint32_t in_c2 = base_in_c + 2U;
                const uint32_t in_c3 = base_in_c + 3U;

                const size_t in_idx0 = ((size_t)in_c0 * H + h) * W + w;
                const size_t in_idx1 = ((size_t)in_c1 * H + h) * W + w;
                const size_t in_idx2 = ((size_t)in_c2 * H + h) * W + w;
                const size_t in_idx3 = ((size_t)in_c3 * H + h) * W + w;

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
                pvss_us_weight->pe_norm_w,
                pvss_us_weight->pe_norm_b,
                C_OUT, L_OUT, 1e-5f);
}
