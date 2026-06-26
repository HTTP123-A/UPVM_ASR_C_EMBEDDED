/* Memory-profiling forward for sq_int8_v0 (see headers/sq_int8_v0/model_memory.h).
 *
 * Runs the REAL model/ blocks and kernels (single magnitude stream, sequential),
 * and before each block logs every micro-op's activation/weight footprint via the
 * mem_profile logger. The op list + shapes mirror model/ exactly; the profile is
 * data-independent (a pure function of the static shapes). Each op is tagged with
 * the engine it maps to in the heterogeneous scheme:
 *   CPU-F32     : large/contiguous float kernels (norms, the whole SS2D scan core,
 *                 cross-scan/merge, gating, residuals, sumpool, fc_res, shuffle, FFT)
 *   INT8-ACCEL  : the int8 conv/matmul (projection, patch-embed, MLP fc, pe, skip)
 *   CPU->ACCEL  : quant  (f32 -> int8) feeding the accelerator
 *   ACCEL->CPU  : dequant(int32 -> f32) draining the accelerator
 */

#include "model_memory.h"

/* ===================== PER-OP LOGGING HELPERS ===================== */

/*< one f32 (CPU) op >*/
static void f32op(const char *stage, const char *op, uint32_t rep,
                  uint64_t in_elems, uint64_t w_bytes, uint64_t out_elems)
{
    mem_profile_log(stage, op, "CPU-F32", rep, "f32", in_elems, w_bytes, "f32", out_elems);
}

/*< one int8 layer = quant (CPU->ACCEL) + int8 conv (ACCEL) + dequant (ACCEL->CPU).
 *  weight = int8 weights + int32 bias + f32 (act_scale[c_in] + deq_scale[c_out]). >*/
static void qc(const char *stage, const char *base, uint32_t rep,
               uint64_t in_elems, uint64_t out_elems,
               uint32_t c_in, uint32_t c_out, uint64_t w_i8)
{
    char nm[96];
    const uint64_t wb = w_i8 + (uint64_t)c_out * 4U + ((uint64_t)c_in + c_out) * 4U;
    snprintf(nm, sizeof(nm), "%s.quant",    base); mem_profile_log(stage, nm, "CPU->ACCEL", rep, "f32",   in_elems,  0U, "int8",  in_elems);
    snprintf(nm, sizeof(nm), "%s.int8conv", base); mem_profile_log(stage, nm, "INT8-ACCEL", rep, "int8",  in_elems,  wb, "int32", out_elems);
    snprintf(nm, sizeof(nm), "%s.dequant",  base); mem_profile_log(stage, nm, "ACCEL->CPU", rep, "int32", out_elems, 0U, "f32",   out_elems);
}

/*< the F32 SS2D scan core (shared by the int8 and the f32-ss2d blocks).
 *  N is the SSM state dim (1 here); D = sub d_inner; R2N = R + 2N. >*/
static void log_ss2d_core_f32(const char *stage, uint32_t L, uint32_t D,
                              uint32_t R, uint32_t N, uint32_t R2N, uint32_t rep,
                              int relu_x_int32)
{
    const uint64_t Ll = L;
    if (relu_x_int32) /* v3: SSM x ReLU fused onto the int32 dw-conv accumulator (before dequant) */
        mem_profile_log(stage, "relu_x(int32)", "INT8-ACCEL", rep, "int32", (uint64_t)D * Ll, 0U, "int32", (uint64_t)D * Ll);
    else
        f32op(stage, "relu_x",                 rep, (uint64_t)D * Ll, 0U, (uint64_t)D * Ll);
    f32op(stage, "relu_z",                 rep, (uint64_t)D * Ll, 0U, (uint64_t)D * Ll);
    f32op(stage, "cross_scan",             rep, (uint64_t)4 * D * Ll, 0U, (uint64_t)4 * D * Ll);
    f32op(stage, "x_proj(grouped_conv1d)", rep, (uint64_t)4 * D * Ll, (uint64_t)4 * R2N * D * 4U, (uint64_t)4 * R2N * Ll);
    f32op(stage, "dt_proj(grouped_conv1d)",rep, (uint64_t)4 * R * Ll, (uint64_t)4 * D * R * 4U,   (uint64_t)4 * D * Ll);
    /* selective scan: xy[4D,L] in-place; reads dts[4D,L], Bs/Cs[4N,L]; state[4D*N];
     * weights As[4D*N]+Ds[4D]+delta_bias[4D] */
    f32op(stage, "selective_scan",         rep, (uint64_t)(8 * D + 8 * N) * Ll,
          (uint64_t)(4 * D * N + 8 * D) * 4U, (uint64_t)4 * D * Ll);
    f32op(stage, "cross_merge",            rep, (uint64_t)4 * D * Ll, 0U, (uint64_t)D * Ll);
    f32op(stage, "ssm_out_norm(LN)",       rep, (uint64_t)D * Ll, (uint64_t)2 * D * 4U, (uint64_t)D * Ll);
    f32op(stage, "gated_mul(x*z)",         rep, (uint64_t)D * Ll, 0U, (uint64_t)D * Ll);
}

/*< SS2D with INT8 projections (encoder / latent / decoder) >*/
static void log_ss2d_int8(const char *stage, uint32_t C, uint32_t L,
                          uint32_t SUB_C, uint32_t D, uint32_t R, uint32_t N,
                          uint32_t R2N, uint32_t red, uint32_t dwk)
{
    const uint64_t Ll = L;
    f32op(stage, "ssm_pre_norm(LN)",  1U, (uint64_t)C * Ll, (uint64_t)2 * C * 4U, (uint64_t)C * Ll);
    f32op(stage, "ssm_residual_copy", 1U, (uint64_t)C * Ll, 0U, (uint64_t)C * Ll);
    qc(stage, "in_proj",   red, (uint64_t)SUB_C * Ll, (uint64_t)(2 * D) * Ll, SUB_C, 2 * D, (uint64_t)(2 * D) * SUB_C);
    qc(stage, "dw_conv2d", red, (uint64_t)D * Ll,     (uint64_t)D * Ll,       D,     D,     (uint64_t)D * dwk * dwk);
    log_ss2d_core_f32(stage, L, D, R, N, R2N, red, 1); /* int8 path: relu_x runs int32 on the dw-conv acc */
    qc(stage, "out_proj",  red, (uint64_t)D * Ll, (uint64_t)SUB_C * Ll, D, SUB_C, (uint64_t)SUB_C * D);
    f32op(stage, "ssm_residual_add", red, (uint64_t)SUB_C * Ll, 0U, (uint64_t)SUB_C * Ll);
}

/*< SS2D fully F32 (output0 / output1); has_norm=0 for output0 (Identity) >*/
static void log_ss2d_f32(const char *stage, uint32_t C, uint32_t L,
                         uint32_t SUB_C, uint32_t D, uint32_t R, uint32_t N,
                         uint32_t R2N, uint32_t red, uint32_t dwk, uint32_t has_norm)
{
    const uint64_t Ll = L;
    if (has_norm) {
        f32op(stage, "ssm_pre_norm(LN)", 1U, (uint64_t)C * Ll, (uint64_t)2 * C * 4U, (uint64_t)C * Ll);
    }
    f32op(stage, "ssm_residual_copy", 1U, (uint64_t)C * Ll, 0U, (uint64_t)C * Ll);
    f32op(stage, "in_proj(pw/f32)",   red, (uint64_t)SUB_C * Ll, (uint64_t)(2 * D) * SUB_C * 4U, (uint64_t)(2 * D) * Ll);
    f32op(stage, "dw_conv2d(f32)",    red, (uint64_t)D * Ll, (uint64_t)D * dwk * dwk * 4U, (uint64_t)D * Ll);
    log_ss2d_core_f32(stage, L, D, R, N, R2N, red, 0); /* f32 output stage: relu_x stays f32 */
    f32op(stage, "out_proj(pw/f32)",  red, (uint64_t)D * Ll, (uint64_t)SUB_C * D * 4U, (uint64_t)SUB_C * Ll);
    f32op(stage, "ssm_residual_add",  red, (uint64_t)SUB_C * Ll, 0U, (uint64_t)SUB_C * Ll);
}

/*< MLP, downsampling variant (encoder): C->2C, spatial /4. MLP_HIDDEN = C >*/
static void log_mlp_ds(const char *stage, uint32_t C, uint32_t L, uint32_t MLPh)
{
    const uint32_t C2 = 2U * C;
    const uint64_t L2 = (uint64_t)L / 4U;
    const uint64_t Ll = L;
    f32op(stage, "mlp_residual_copy", 1U, (uint64_t)C * Ll, 0U, (uint64_t)C * Ll);
    f32op(stage, "mlp_norm(LN)",      1U, (uint64_t)C * Ll, (uint64_t)2 * C * 4U, (uint64_t)C * Ll);
    /* fc1 (v4): quant -> int8 conv -> relu_int32 -> SumPool(int32) -> requant(int32->int8, compacted L/4) */
    const uint64_t fc1_wb = (uint64_t)MLPh * C + (uint64_t)MLPh * 4U + (uint64_t)C * 4U; /* w_i8 + bias + act_scale (deq_scale now read by requant) */
    mem_profile_log(stage, "mlp_fc1.quant",      "CPU->ACCEL", 1U, "f32",   (uint64_t)C * Ll,    0U,     "int8",  (uint64_t)C * Ll);
    mem_profile_log(stage, "mlp_fc1.int8conv",   "INT8-ACCEL", 1U, "int8",  (uint64_t)C * Ll,    fc1_wb, "int32", (uint64_t)MLPh * Ll);
    mem_profile_log(stage, "mlp_relu(int32)",    "INT8-ACCEL", 1U, "int32", (uint64_t)MLPh * Ll, 0U,     "int32", (uint64_t)MLPh * Ll);
    mem_profile_log(stage, "mlp_sumpool(dwconv_int32)", "INT8-ACCEL", 1U, "int32", (uint64_t)MLPh * Ll, 0U, "int32", (uint64_t)MLPh * L2);
    mem_profile_log(stage, "mlp_fc1.requant",    "INT8-ACCEL", 1U, "int32", (uint64_t)MLPh * L2, (uint64_t)MLPh * 4U, "int8", (uint64_t)MLPh * L2); /* v4: fused dequant+quant, requant_scale[MLPh] */
    f32op(stage, "mlp_sumpool_res(dwconv)", 1U, (uint64_t)C * Ll, (uint64_t)C * 4U * 4U, (uint64_t)C * L2);
    /* fc2 (v4): int8 conv -> dequant (quant folded into fc1's requant; no act_scale read) */
    const uint64_t fc2_wb = (uint64_t)C2 * MLPh + (uint64_t)C2 * 4U; /* w_i8 + deq_scale (encoder fc2 has no bias) */
    mem_profile_log(stage, "mlp_fc2.int8conv",   "INT8-ACCEL", 1U, "int8",  (uint64_t)MLPh * L2, fc2_wb, "int32", (uint64_t)C2 * L2);
    mem_profile_log(stage, "mlp_fc2.dequant",    "ACCEL->CPU", 1U, "int32", (uint64_t)C2 * L2,   0U,     "f32",   (uint64_t)C2 * L2);
    f32op(stage, "mlp_fc_res(pw)",    1U, (uint64_t)C * L2, (uint64_t)C2 * C * 4U, (uint64_t)C2 * L2);
    f32op(stage, "mlp_residual_add",  1U, (uint64_t)C2 * L2, 0U, (uint64_t)C2 * L2);
}

/*< MLP, plain variant (latent / decoder / output): C->C. has_norm=0 for output0.
 *  fused=1 (v4: latent/decoder): fc1 dequant + fc2 quant collapse into one int32->int8
 *  requant; fused=0 (output stages, untouched): classic quant->conv->dequant on both. >*/
static void log_mlp_plain(const char *stage, uint32_t C, uint32_t L, uint32_t MLPh, uint32_t has_norm, uint32_t fused)
{
    const uint64_t Ll = L;
    f32op(stage, "mlp_residual_copy", 1U, (uint64_t)C * Ll, 0U, (uint64_t)C * Ll);
    if (has_norm) {
        f32op(stage, "mlp_norm(LN)", 1U, (uint64_t)C * Ll, (uint64_t)2 * C * 4U, (uint64_t)C * Ll);
    }
    if (fused) {
        /* fc1 (v4): quant -> int8 conv -> relu_int32 -> requant(int32->int8) */
        const uint64_t fc1_wb = (uint64_t)MLPh * C + (uint64_t)MLPh * 4U + (uint64_t)C * 4U; /* w_i8 + bias + act_scale (deq read by requant) */
        mem_profile_log(stage, "mlp_fc1.quant",    "CPU->ACCEL", 1U, "f32",   (uint64_t)C * Ll,    0U,     "int8",  (uint64_t)C * Ll);
        mem_profile_log(stage, "mlp_fc1.int8conv", "INT8-ACCEL", 1U, "int8",  (uint64_t)C * Ll,    fc1_wb, "int32", (uint64_t)MLPh * Ll);
        mem_profile_log(stage, "mlp_relu(int32)",  "INT8-ACCEL", 1U, "int32", (uint64_t)MLPh * Ll, 0U,     "int32", (uint64_t)MLPh * Ll);
        mem_profile_log(stage, "mlp_fc1.requant",  "INT8-ACCEL", 1U, "int32", (uint64_t)MLPh * Ll, (uint64_t)MLPh * 4U, "int8", (uint64_t)MLPh * Ll); /* v4: fused dequant+quant, requant_scale[MLPh] */
        /* fc2 (v4): int8 conv -> dequant (quant folded into fc1's requant; no act_scale read) */
        const uint64_t fc2_wb = (uint64_t)C * MLPh + (uint64_t)C * 4U + (uint64_t)C * 4U; /* w_i8 + bias + deq_scale */
        mem_profile_log(stage, "mlp_fc2.int8conv", "INT8-ACCEL", 1U, "int8",  (uint64_t)MLPh * Ll, fc2_wb, "int32", (uint64_t)C * Ll);
        mem_profile_log(stage, "mlp_fc2.dequant",  "ACCEL->CPU", 1U, "int32", (uint64_t)C * Ll,    0U,     "f32",   (uint64_t)C * Ll);
    } else {
        qc(stage, "mlp_fc1", 1U, (uint64_t)C * Ll, (uint64_t)MLPh * Ll, C, MLPh, (uint64_t)MLPh * C);
        mem_profile_log(stage, "mlp_relu(int32)", "INT8-ACCEL", 1U, "int32", (uint64_t)MLPh * Ll, 0U, "int32", (uint64_t)MLPh * Ll); /* v1: ReLU fused on int32 acc before dequant */
        qc(stage, "mlp_fc2", 1U, (uint64_t)MLPh * Ll, (uint64_t)C * Ll, MLPh, C, (uint64_t)C * MLPh);
    }
    f32op(stage, "mlp_residual_add", 1U, (uint64_t)C * Ll, 0U, (uint64_t)C * Ll);
}

/*< PatchExpanding (decoder / output0 / output1): pe int8 C->2C, shuffle, pe_norm >*/
static void log_patch_expand(const char *stage, uint32_t C, uint32_t L)
{
    const uint32_t C_EXP = 2U * C;
    const uint32_t C_OUT = C / 2U;
    const uint64_t L_OUT = (uint64_t)L * 4U;
    const uint64_t Ll = L;
    qc(stage, "pe(patch_expand)", 1U, (uint64_t)C * Ll, (uint64_t)C_EXP * Ll, C, C_EXP, (uint64_t)C_EXP * C);
    f32op(stage, "pixel_shuffle", 1U, (uint64_t)C_EXP * Ll, 0U, (uint64_t)C_EXP * Ll);
    f32op(stage, "pe_norm(LN)",   1U, (uint64_t)C_OUT * L_OUT, (uint64_t)2 * C_OUT * 4U, (uint64_t)C_OUT * L_OUT);
}

/*< Decoder / output0 skip handler: quant(x) + quant(res) + split2 int8 + dequant >*/
static void log_skip(const char *stage, uint32_t C, uint32_t L)
{
    const uint64_t Ll = L;
    const uint64_t wb = (uint64_t)C * (2U * C) + (uint64_t)C * 4U + ((uint64_t)2U * C + C) * 4U;
    mem_profile_log(stage, "skip.quant_x",   "CPU->ACCEL", 1U, "f32",   (uint64_t)C * Ll, 0U, "int8",  (uint64_t)C * Ll);
    mem_profile_log(stage, "skip.quant_res", "CPU->ACCEL", 1U, "f32",   (uint64_t)C * Ll, 0U, "int8",  (uint64_t)C * Ll);
    mem_profile_log(stage, "skip.split2_int8conv", "INT8-ACCEL", 1U, "int8", (uint64_t)2 * C * Ll, wb, "int32", (uint64_t)C * Ll);
    mem_profile_log(stage, "skip.dequant",   "ACCEL->CPU", 1U, "int32", (uint64_t)C * Ll, 0U, "f32",   (uint64_t)C * Ll);
}

/*< Patch embed: conv1 [1,512,512]->[8,256,256], conv2 [8,256,256]->[16,128,128] >*/
static void log_patch_embed(const char *stage)
{
    const uint64_t in1 = (uint64_t)C_IN_PATCH_EMBED * H_IN_PATCH_EMBED * W_IN_PATCH_EMBED;        /* 1*512*512 */
    const uint64_t mid = (uint64_t)C_MID_PATCH_EMBED * (H_IN_PATCH_EMBED / 2U) * (W_IN_PATCH_EMBED / 2U); /* 8*256*256 */
    const uint64_t out = (uint64_t)C_OUT_PATCH_EMBED * (H_IN_PATCH_EMBED / 4U) * (W_IN_PATCH_EMBED / 4U); /* 16*128*128 */
    qc(stage, "conv1", 1U, in1, mid, C_IN_PATCH_EMBED, C_MID_PATCH_EMBED,
       (uint64_t)C_MID_PATCH_EMBED * C_IN_PATCH_EMBED * K_SIZE_PATCH_EMBED * K_SIZE_PATCH_EMBED);
    f32op(stage, "norm1(LN)", 1U, mid, (uint64_t)2 * C_MID_PATCH_EMBED * 4U, mid);
    f32op(stage, "gelu",      1U, mid, 0U, mid);
    qc(stage, "conv2", 1U, mid, out, C_MID_PATCH_EMBED, C_OUT_PATCH_EMBED,
       (uint64_t)C_OUT_PATCH_EMBED * C_MID_PATCH_EMBED * K_SIZE_PATCH_EMBED * K_SIZE_PATCH_EMBED);
    f32op(stage, "norm2(LN)", 1U, out, (uint64_t)2 * C_OUT_PATCH_EMBED * 4U, out);
}

/*< Output3: pre_conv f32 (4C->C) + f32 SS2D (no group loop) + int8 fc1/fc2 (no norm) >*/
static void log_output3(const char *stage)
{
    const uint32_t C = C_OUT_3, D = D_INNER_OUT_3, R = R_OUT_3, N = N_OUT_3, R2N = R2N_OUT_3, MLPh = MLP_HIDDEN_OUT_3;
    const uint64_t Ll = L_OUT_3;
    f32op(stage, "pre_conv(4C->C/pw)", 1U, (uint64_t)4 * C * Ll, (uint64_t)C * (4U * C) * 4U, (uint64_t)C * Ll);
    f32op(stage, "in_proj(pw/f32)",    1U, (uint64_t)C * Ll, (uint64_t)(2 * D) * C * 4U, (uint64_t)(2 * D) * Ll);
    f32op(stage, "dw_conv2d(f32)",     1U, (uint64_t)D * Ll, (uint64_t)D * DW_KERNEL_SZ_OUT_3 * DW_KERNEL_SZ_OUT_3 * 4U, (uint64_t)D * Ll);
    log_ss2d_core_f32(stage, L_OUT_3, D, R, N, R2N, 1U, 0); /* f32 output3: relu_x stays f32 */
    f32op(stage, "out_proj(pw/f32)",   1U, (uint64_t)D * Ll, (uint64_t)C * D * 4U, (uint64_t)C * Ll);
    f32op(stage, "ssm_residual_add",   1U, (uint64_t)C * Ll, 0U, (uint64_t)C * Ll);
    f32op(stage, "mlp_residual_copy",  1U, (uint64_t)C * Ll, 0U, (uint64_t)C * Ll);
    qc(stage, "mlp_fc1", 1U, (uint64_t)C * Ll, (uint64_t)MLPh * Ll, C, MLPh, (uint64_t)MLPh * C);
    mem_profile_log(stage, "mlp_relu(int32)", "INT8-ACCEL", 1U, "int32", (uint64_t)MLPh * Ll, 0U, "int32", (uint64_t)MLPh * Ll); /* v1: ReLU fused on int32 acc before dequant */
    qc(stage, "mlp_fc2", 1U, (uint64_t)MLPh * Ll, (uint64_t)C * Ll, MLPh, C, (uint64_t)C * MLPh);
    f32op(stage, "mlp_residual_add", 1U, (uint64_t)C * Ll, 0U, (uint64_t)C * Ll);
}

/*< per-stage interaction: mag += pha ; pha += mag >*/
static void log_interact(const char *stage, uint32_t C, uint32_t L)
{
    f32op(stage, "interaction_mag(+=pha)", 1U, (uint64_t)C * L, 0U, (uint64_t)C * L);
    f32op(stage, "interaction_pha(+=mag)", 1U, (uint64_t)C * L, 0U, (uint64_t)C * L);
}

/* ===================== PROFILING FORWARD ===================== */
void dual_stream_unet_mem_int8(float* audio,
                        const struct upvm_asr_weight* model_weight,
                        float *model_inout_buf_mag, float *model_inout_buf_pha,
                        float *model_internal_res_buf_mag, float *model_internal_res_buf_pha,
                        float *model_global_res_buf_mag, float *model_global_res_buf_pha,
                        float *model_act_buf_mag, float *model_act_buf_pha,
                        float *model_low_freq_buf_mag, float *model_low_freq_buf_pha,
                        float *hidden_state_buf_mag, float *hidden_state_buf_pha,
                        int8_t *quant_buf_mag, int8_t *quant_buf_pha,
                        int32_t *acc_buf_mag, int32_t *acc_buf_pha,
                        float *hann_window_buf, float *frame, float *ola, float* wss,
                        fftwf_complex *spec, fftwf_plan *p_stft, fftwf_plan *p_istft, uint32_t win_offset,
                        const char *csv_path)
{
    (void)model_inout_buf_pha; (void)model_internal_res_buf_pha; (void)model_global_res_buf_pha;
    (void)model_act_buf_pha;   (void)model_low_freq_buf_pha;     (void)hidden_state_buf_pha;
    (void)quant_buf_pha;       (void)acc_buf_pha;

    mem_profile_begin(csv_path);

    /*< STFT (FFTW, CPU-F32) >*/
    f32op("stft", "STFT(fftw)", 1U, (uint64_t)AUDIO_LENGTH, 0U, (uint64_t)H_IN_PATCH_EMBED * W_IN_PATCH_EMBED);
    stft_f32(audio, model_inout_buf_mag, model_inout_buf_pha,
            hann_window_buf, frame, spec, p_stft,
            AUDIO_LENGTH, N_FFT, WIN_LENGTH, HOP_LENGTH, FRAME_SIZE, NUM_OF_FRAME, win_offset, INV_SQRT_NFFT);

    /*< Low-freq copy + spectrogram global residual (mag) >*/
    f32op("pre", "low_freq_copy", 1U, (uint64_t)LOW_FREQ_ELEMENT, 0U, (uint64_t)LOW_FREQ_ELEMENT);
    memcpy(model_low_freq_buf_mag, model_inout_buf_mag, LOW_FREQ_ELEMENT * sizeof(float));
    f32op("pre", "spectrogram_residual_copy", 1U, (uint64_t)C_IN_PATCH_EMBED * L_IN_PATCH_EMBED, 0U, (uint64_t)C_IN_PATCH_EMBED * L_IN_PATCH_EMBED);
    memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_SPECTROGRAM, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_IN_PATCH_EMBED * L_IN_PATCH_EMBED * sizeof(float));

    /*< Patch Embed >*/
    log_patch_embed("patch_embed");
    patch_embed_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_act_buf_mag, quant_buf_mag, acc_buf_mag,
                    (const struct embed_weight*) &(model_weight->embed_weight_mag),
                    C_IN_PATCH_EMBED, C_MID_PATCH_EMBED, C_OUT_PATCH_EMBED,
                    H_IN_PATCH_EMBED, W_IN_PATCH_EMBED, K_SIZE_PATCH_EMBED, STRIDE_PATCH_EMBED, PAD_PATCH_EMBED);
    f32op("patch_embed", "patchembed_residual_copy", 1U, (uint64_t)C_ENC_0 * L_ENC_0, 0U, (uint64_t)C_ENC_0 * L_ENC_0);
    memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_ENC_0 * L_ENC_0 * sizeof(float));
    log_interact("patch_embed", C_ENC_0, L_ENC_0);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_0 * L_ENC_0);
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_0 * L_ENC_0);

    /*< Encoder 0 >*/
    log_ss2d_int8("enc0", C_ENC_0, L_ENC_0, SUB_C_ENC_0, SUB_D_INNER_ENC_0, R_ENC_0, N_ENC_0, R2N_ENC_0, REDUCTION_FACTOR_ENC_0, DW_KERNEL_SZ_ENC_0);
    log_mlp_ds("enc0", C_ENC_0, L_ENC_0, MLP_HIDDEN_ENC_0);
    pvss_ds_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc0_mag),
            C_ENC_0, H_ENC_0, W_ENC_0, L_ENC_0, R_ENC_0, N_ENC_0, R2N_ENC_0, DW_KERNEL_SZ_ENC_0, MLP_HIDDEN_ENC_0,
            SUB_D_INNER_ENC_0, SUB_D_PROJECTION_ENC_0, SUB_C_ENC_0, REDUCTION_FACTOR_ENC_0, CHUNK_SZ_ENC_0,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_0, (size_t)HID_BUF_SLICE_ENC_0);
    f32op("enc0", "global_residual_copy", 1U, (uint64_t)C_ENC_1 * L_ENC_1, 0U, (uint64_t)C_ENC_1 * L_ENC_1);
    memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_0, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_ENC_1 * L_ENC_1 * sizeof(float));
    log_interact("enc0", C_ENC_1, L_ENC_1);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_1 * L_ENC_1);
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_1 * L_ENC_1);

    /*< Encoder 1 >*/
    log_ss2d_int8("enc1", C_ENC_1, L_ENC_1, SUB_C_ENC_1, SUB_D_INNER_ENC_1, R_ENC_1, N_ENC_1, R2N_ENC_1, REDUCTION_FACTOR_ENC_1, DW_KERNEL_SZ_ENC_1);
    log_mlp_ds("enc1", C_ENC_1, L_ENC_1, MLP_HIDDEN_ENC_1);
    pvss_ds_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc1_mag),
            C_ENC_1, H_ENC_1, W_ENC_1, L_ENC_1, R_ENC_1, N_ENC_1, R2N_ENC_1, DW_KERNEL_SZ_ENC_1, MLP_HIDDEN_ENC_1,
            SUB_D_INNER_ENC_1, SUB_D_PROJECTION_ENC_1, SUB_C_ENC_1, REDUCTION_FACTOR_ENC_1, CHUNK_SZ_ENC_1,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_1, (size_t)HID_BUF_SLICE_ENC_1);
    f32op("enc1", "global_residual_copy", 1U, (uint64_t)C_ENC_2 * L_ENC_2, 0U, (uint64_t)C_ENC_2 * L_ENC_2);
    memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_1, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_ENC_2 * L_ENC_2 * sizeof(float));
    log_interact("enc1", C_ENC_2, L_ENC_2);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_2 * L_ENC_2);
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_ENC_2 * L_ENC_2);

    /*< Encoder 2 >*/
    log_ss2d_int8("enc2", C_ENC_2, L_ENC_2, SUB_C_ENC_2, SUB_D_INNER_ENC_2, R_ENC_2, N_ENC_2, R2N_ENC_2, REDUCTION_FACTOR_ENC_2, DW_KERNEL_SZ_ENC_2);
    log_mlp_ds("enc2", C_ENC_2, L_ENC_2, MLP_HIDDEN_ENC_2);
    pvss_ds_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct pvss_ds_weight*) &(model_weight->pvss_ds_weight_enc2_mag),
            C_ENC_2, H_ENC_2, W_ENC_2, L_ENC_2, R_ENC_2, N_ENC_2, R2N_ENC_2, DW_KERNEL_SZ_ENC_2, MLP_HIDDEN_ENC_2,
            SUB_D_INNER_ENC_2, SUB_D_PROJECTION_ENC_2, SUB_C_ENC_2, REDUCTION_FACTOR_ENC_2, CHUNK_SZ_ENC_2,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_ENC_2, (size_t)HID_BUF_SLICE_ENC_2);
    f32op("enc2", "global_residual_copy", 1U, (uint64_t)C_LATENT * L_LATENT, 0U, (uint64_t)C_LATENT * L_LATENT);
    memcpy(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_2, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, C_LATENT * L_LATENT * sizeof(float));
    log_interact("enc2", C_LATENT, L_LATENT);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_LATENT * L_LATENT);
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_LATENT * L_LATENT);

    /*< Latent >*/
    log_ss2d_int8("latent", C_LATENT, L_LATENT, SUB_C_LATENT, SUB_D_INNER_LATENT, R_LATENT, N_LATENT, R2N_LATENT, REDUCTION_FACTOR_LATENT, DW_KERNEL_SZ_LATENT);
    log_mlp_plain("latent", C_LATENT, L_LATENT, MLP_HIDDEN_LATENT, 1U, 1U);
    pvss_latent_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct pvss_latent_weight*) &(model_weight->pvss_latent_weight_mag),
            C_LATENT, H_LATENT, W_LATENT, L_LATENT, R_LATENT, N_LATENT, R2N_LATENT, DW_KERNEL_SZ_LATENT, MLP_HIDDEN_LATENT,
            SUB_D_INNER_LATENT, SUB_D_PROJECTION_LATENT, SUB_C_LATENT, REDUCTION_FACTOR_LATENT, CHUNK_SZ_LATENT,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_LATENT, (size_t)HID_BUF_SLICE_LATENT);
    log_interact("latent", C_LATENT, L_LATENT);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_LATENT * L_LATENT);
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_LATENT * L_LATENT);

    /*< Decoder 0 (skip handler + US block) >*/
    log_skip("dec0", C_DEC_0, L_DEC_0);
    {
        const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_mag.skip_conv_dec0;
        quant_f32_to_int8_perchannel(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, quant_buf_mag, C_DEC_0, L_DEC_0, q->act_scale, 0);
        quant_f32_to_int8_perchannel(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_2, quant_buf_mag + (size_t)C_DEC_0 * L_DEC_0, C_DEC_0, L_DEC_0, q->act_scale + C_DEC_0, 0);
        pointwise_conv2d_split2_int8(quant_buf_mag, quant_buf_mag + (size_t)C_DEC_0 * L_DEC_0, q->w_i8, q->bias_i32, acc_buf_mag, C_DEC_0, H_DEC_0, W_DEC_0, C_DEC_0);
        dequant_int32_to_f32_perchannel(acc_buf_mag, model_inout_buf_mag + (size_t)C_DEC_0 * L_DEC_0 + W_IN_PATCH_EMBED, C_DEC_0, L_DEC_0, q->deq_scale);
    }
    log_ss2d_int8("dec0", C_DEC_0, L_DEC_0, SUB_C_DEC_0, SUB_D_INNER_DEC_0, R_DEC_0, N_DEC_0, R2N_DEC_0, REDUCTION_FACTOR_DEC_0, DW_KERNEL_SZ_DEC_0);
    log_mlp_plain("dec0", C_DEC_0, L_DEC_0, MLP_HIDDEN_DEC_0, 1U, 1U);
    log_patch_expand("dec0", C_DEC_0, L_DEC_0);
    pvss_us_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec0_mag),
            C_DEC_0, H_DEC_0, W_DEC_0, L_DEC_0, R_DEC_0, N_DEC_0, R2N_DEC_0, DW_KERNEL_SZ_DEC_0, MLP_HIDDEN_DEC_0,
            SUB_D_INNER_DEC_0, SUB_D_PROJECTION_DEC_0, SUB_C_DEC_0, REDUCTION_FACTOR_DEC_0, CHUNK_SZ_DEC_0,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_0, (size_t)HID_BUF_SLICE_DEC_0);
    log_interact("dec0", C_DEC_1, L_DEC_1);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_DEC_1 * L_DEC_1);
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_DEC_1 * L_DEC_1);

    /*< Decoder 1 >*/
    log_skip("dec1", C_DEC_1, L_DEC_1);
    {
        const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_mag.skip_conv_dec1;
        quant_f32_to_int8_perchannel(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, quant_buf_mag, C_DEC_1, L_DEC_1, q->act_scale, 0);
        quant_f32_to_int8_perchannel(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_1, quant_buf_mag + (size_t)C_DEC_1 * L_DEC_1, C_DEC_1, L_DEC_1, q->act_scale + C_DEC_1, 0);
        pointwise_conv2d_split2_int8(quant_buf_mag, quant_buf_mag + (size_t)C_DEC_1 * L_DEC_1, q->w_i8, q->bias_i32, acc_buf_mag, C_DEC_1, H_DEC_1, W_DEC_1, C_DEC_1);
        dequant_int32_to_f32_perchannel(acc_buf_mag, model_inout_buf_mag + (size_t)C_DEC_1 * L_DEC_1 + W_IN_PATCH_EMBED, C_DEC_1, L_DEC_1, q->deq_scale);
    }
    log_ss2d_int8("dec1", C_DEC_1, L_DEC_1, SUB_C_DEC_1, SUB_D_INNER_DEC_1, R_DEC_1, N_DEC_1, R2N_DEC_1, REDUCTION_FACTOR_DEC_1, DW_KERNEL_SZ_DEC_1);
    log_mlp_plain("dec1", C_DEC_1, L_DEC_1, MLP_HIDDEN_DEC_1, 1U, 1U);
    log_patch_expand("dec1", C_DEC_1, L_DEC_1);
    pvss_us_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec1_mag),
            C_DEC_1, H_DEC_1, W_DEC_1, L_DEC_1, R_DEC_1, N_DEC_1, R2N_DEC_1, DW_KERNEL_SZ_DEC_1, MLP_HIDDEN_DEC_1,
            SUB_D_INNER_DEC_1, SUB_D_PROJECTION_DEC_1, SUB_C_DEC_1, REDUCTION_FACTOR_DEC_1, CHUNK_SZ_DEC_1,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_1, (size_t)HID_BUF_SLICE_DEC_1);
    log_interact("dec1", C_DEC_2, L_DEC_2);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_DEC_2 * L_DEC_2);
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_DEC_2 * L_DEC_2);

    /*< Decoder 2 >*/
    log_skip("dec2", C_DEC_2, L_DEC_2);
    {
        const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_mag.skip_conv_dec2;
        quant_f32_to_int8_perchannel(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, quant_buf_mag, C_DEC_2, L_DEC_2, q->act_scale, 0);
        quant_f32_to_int8_perchannel(model_global_res_buf_mag + (size_t)SKIP_OFFSET_ENC_0, quant_buf_mag + (size_t)C_DEC_2 * L_DEC_2, C_DEC_2, L_DEC_2, q->act_scale + C_DEC_2, 0);
        pointwise_conv2d_split2_int8(quant_buf_mag, quant_buf_mag + (size_t)C_DEC_2 * L_DEC_2, q->w_i8, q->bias_i32, acc_buf_mag, C_DEC_2, H_DEC_2, W_DEC_2, C_DEC_2);
        dequant_int32_to_f32_perchannel(acc_buf_mag, model_inout_buf_mag + (size_t)C_DEC_2 * L_DEC_2 + W_IN_PATCH_EMBED, C_DEC_2, L_DEC_2, q->deq_scale);
    }
    log_ss2d_int8("dec2", C_DEC_2, L_DEC_2, SUB_C_DEC_2, SUB_D_INNER_DEC_2, R_DEC_2, N_DEC_2, R2N_DEC_2, REDUCTION_FACTOR_DEC_2, DW_KERNEL_SZ_DEC_2);
    log_mlp_plain("dec2", C_DEC_2, L_DEC_2, MLP_HIDDEN_DEC_2, 1U, 1U);
    log_patch_expand("dec2", C_DEC_2, L_DEC_2);
    pvss_us_mp_int8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct pvss_us_weight*) &(model_weight->pvss_us_weight_dec2_mag),
            C_DEC_2, H_DEC_2, W_DEC_2, L_DEC_2, R_DEC_2, N_DEC_2, R2N_DEC_2, DW_KERNEL_SZ_DEC_2, MLP_HIDDEN_DEC_2,
            SUB_D_INNER_DEC_2, SUB_D_PROJECTION_DEC_2, SUB_C_DEC_2, REDUCTION_FACTOR_DEC_2, CHUNK_SZ_DEC_2,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_DEC_2, (size_t)HID_BUF_SLICE_DEC_2);
    log_interact("dec2", C_OUT_0, L_OUT_0);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, (size_t)C_OUT_0 * L_OUT_0);
    elemwise_add_f32(model_inout_buf_pha + (size_t)W_IN_PATCH_EMBED, model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, (size_t)C_OUT_0 * L_OUT_0);

    /*< Output 0 (skip handler + f32-ss2d block, no block norms) >*/
    log_skip("out0", C_OUT_0, L_OUT_0);
    {
        const struct qconv_int8 *q = &model_weight->pvss_us_skip_weight_mag.skip_conv_out0;
        quant_f32_to_int8_perchannel(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, quant_buf_mag, C_OUT_0, L_OUT_0, q->act_scale, 0);
        quant_f32_to_int8_perchannel(model_global_res_buf_mag + (size_t)SKIP_OFFSET_PATCH_EMBED, quant_buf_mag + (size_t)C_OUT_0 * L_OUT_0, C_OUT_0, L_OUT_0, q->act_scale + C_OUT_0, 0);
        pointwise_conv2d_split2_int8(quant_buf_mag, quant_buf_mag + (size_t)C_OUT_0 * L_OUT_0, q->w_i8, q->bias_i32, acc_buf_mag, C_OUT_0, H_OUT_0, W_OUT_0, C_OUT_0);
        dequant_int32_to_f32_perchannel(acc_buf_mag, model_inout_buf_mag + (size_t)C_OUT_0 * L_OUT_0 + W_IN_PATCH_EMBED, C_OUT_0, L_OUT_0, q->deq_scale);
    }
    log_ss2d_f32("out0", C_OUT_0, L_OUT_0, SUB_C_OUT_0, SUB_D_INNER_OUT_0, R_OUT_0, N_OUT_0, R2N_OUT_0, REDUCTION_FACTOR_OUT_0, DW_KERNEL_SZ_OUT_0, 0U);
    log_mlp_plain("out0", C_OUT_0, L_OUT_0, MLP_HIDDEN_OUT_0, 0U, 0U);
    log_patch_expand("out0", C_OUT_0, L_OUT_0);
    vss_output0_mp_f32i8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct vss_output0_weight*) &(model_weight->vss_output0_weight_mag),
            C_OUT_0, H_OUT_0, W_OUT_0, L_OUT_0, R_OUT_0, N_OUT_0, R2N_OUT_0, DW_KERNEL_SZ_OUT_0, MLP_HIDDEN_OUT_0,
            SUB_D_INNER_OUT_0, SUB_D_PROJECTION_OUT_0, SUB_C_OUT_0, REDUCTION_FACTOR_OUT_0, CHUNK_SZ_OUT_0,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_OUT_0, (size_t)HID_BUF_SLICE_OUT_0);

    /*< Output 1 (f32-ss2d block, with block norms) >*/
    log_ss2d_f32("out1", C_OUT_1, L_OUT_1, SUB_C_OUT_1, SUB_D_INNER_OUT_1, R_OUT_1, N_OUT_1, R2N_OUT_1, REDUCTION_FACTOR_OUT_1, DW_KERNEL_SZ_OUT_1, 1U);
    log_mlp_plain("out1", C_OUT_1, L_OUT_1, MLP_HIDDEN_OUT_1, 1U, 0U);
    log_patch_expand("out1", C_OUT_1, L_OUT_1);
    vss_output1_mp_f32i8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct vss_output1_weight*) &(model_weight->vss_output1_weight_mag),
            C_OUT_1, H_OUT_1, W_OUT_1, L_OUT_1, R_OUT_1, N_OUT_1, R2N_OUT_1, DW_KERNEL_SZ_OUT_1, MLP_HIDDEN_OUT_1,
            SUB_D_INNER_OUT_1, SUB_D_PROJECTION_OUT_1, SUB_C_OUT_1, REDUCTION_FACTOR_OUT_1, CHUNK_SZ_OUT_1,
            INNER_MP_PARALLEL, (size_t)ACT_BUF_SLICE_OUT_1, (size_t)HID_BUF_SLICE_OUT_1);

    /*< Output 3 (pre_conv + f32-ss2d + int8 fc1/fc2) + final residual + freq replace >*/
    log_output3("out3");
    vss_output3_f32i8(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_internal_res_buf_mag, model_act_buf_mag, hidden_state_buf_mag, quant_buf_mag, acc_buf_mag,
            (const struct vss_output3_weight*) &(model_weight->vss_output3_weight_mag),
            C_OUT_3, H_OUT_3, W_OUT_3, L_OUT_3, R_OUT_3, N_OUT_3, R2N_OUT_3, DW_KERNEL_SZ_OUT_3, MLP_HIDDEN_OUT_3,
            D_INNER_OUT_3, D_PROJECTION_OUT_3, CHUNK_SZ_OUT_3);
    f32op("out3", "final_spectrogram_residual_add", 1U, (uint64_t)C_IN_PATCH_EMBED * L_IN_PATCH_EMBED, 0U, (uint64_t)C_IN_PATCH_EMBED * L_IN_PATCH_EMBED);
    elemwise_add_f32(model_inout_buf_mag + (size_t)W_IN_PATCH_EMBED, model_global_res_buf_mag + (size_t)SKIP_OFFSET_SPECTROGRAM, (size_t)C_IN_PATCH_EMBED * L_IN_PATCH_EMBED);
    f32op("out3", "freq_replace(low_freq_copy_back)", 1U, (uint64_t)LOW_FREQ_ELEMENT, 0U, (uint64_t)LOW_FREQ_ELEMENT);
    memcpy(model_inout_buf_mag, model_low_freq_buf_mag, LOW_FREQ_ELEMENT * sizeof(float));

    /*< iSTFT (FFTW, CPU-F32) >*/
    f32op("istft", "iSTFT(fftw)", 1U, (uint64_t)2 * H_IN_PATCH_EMBED * W_IN_PATCH_EMBED, 0U, (uint64_t)AUDIO_LENGTH);
    istft_f32(model_inout_buf_mag, model_inout_buf_pha, audio, hann_window_buf, frame, ola, wss, spec, p_istft,
            AUDIO_LENGTH, N_FFT, WIN_LENGTH, HOP_LENGTH, PAD, OLA_LEN, FRAME_SIZE, NUM_OF_FRAME, win_offset, INV_SQRT_NFFT);

    mem_profile_end();
}
