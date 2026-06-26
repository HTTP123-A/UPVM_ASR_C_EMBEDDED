#ifndef _UPVM_ASR_MICRO_KERNELS_H_
#define _UPVM_ASR_MICRO_KERNELS_H_

/* ===================== INCLUDE SECTION ===================== */
/* <1. Built-in header > */
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

/* <2. User headers > */

/* ===================== DEFINE ============================== */
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440F
#endif

/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/* ---------------------------------------------- */
/* I. PERFORM CROSS SCAN */
/* --- F32 --- */
/* <1. Cross Scan - Pytorch equivalent: CrossScan > */
void cross_scan_f32(float *x,
                    uint32_t C,
                    uint32_t H,
                    uint32_t W);

/* <2. Cross Merge - Pytorch equivalent: CrossMerge > */
void cross_merge_f32(float *x,
                    uint32_t C,
                    uint32_t H,
                    uint32_t W);

/* ---------------------------------------------- */
/* II. COMPUTATION */
/* --- F32 --- */
/*< 1. Point-wise Convolution - Pytorch equivalent: in_proj, out_proj, conv1 (mlp), conv2 (mlp) >*/
void pointwise_conv2d_f32(const float *restrict x,
                        const float *restrict weight,
                        const float *restrict bias,
                        float *restrict y,
                        uint32_t C_IN, uint32_t H, uint32_t W, uint32_t C_OUT);

/*< 2. Depth-wise convolution 2d - Pytorch equivalent: conv2d (ss2d) >*/
void depthwise_conv2d_f32(const float *restrict x,
                        const float *restrict weight,
                        const float *restrict bias,
                        float *restrict y,
                        uint32_t C, uint32_t H, uint32_t W,
                        uint32_t kernel_size, uint32_t stride, uint32_t pad);

/*< 3. Group point-wise convolution 1d - Pytorch equivalent: >*/
void grouped_pointwise_conv1d_f32(const float *x,
                                const float *weights,
                                const float *bias,
                                float *y,
                                uint32_t SUB_D_IN, uint32_t K, uint32_t L, uint32_t SUB_D_OUT);

/*< 4. Group point-wise convolution 1d (type major) - Pytorch equivalent: >*/
void grouped_pointwise_conv1d_type_major_f32(const float *x,
                                            const float *weights,
                                            const float *bias,
                                            float *y,
                                            uint32_t SUB_D_IN, uint32_t K, uint32_t L, uint32_t R, uint32_t N);

/* <5. Element-wise add (inplace) - Pytorch equivalent: Hadamard addition c = a + b > */
void elemwise_add_f32(float *x, const float *r, uint32_t N);

/* <6. Element-wise mul (inplace) - Pytorch equivalent: Hadamard product c = a .* b > */
void elemwise_mul_f32(float *x, const float *z, uint32_t N);

/*< 7. SSM core - Selective scan (naive FP16 path, f32 state/weights for stability) >*/
void selective_scan_f32(float *xy,                 /* [G*D, L], in-place input/output */
                        const float *dts,          /* [G*D, L] */
                        const float *As,           /* [G*D, N] */
                        const float *Bs,           /* [G, N, L] */
                        const float *Cs,           /* [G, N, L] */
                        const float *Ds,           /* [G*D] */
                        const float *delta_bias,   /* [G*D] */
                        float *hidden_state_buf,   /* [G*D*N], caller-managed f32 state workspace */
                        uint32_t K,
                        uint32_t D,
                        uint32_t N,
                        uint32_t L,
                        uint16_t chunk_sz);

/* --- INT8 --- */
/* <1. Point-wise Convolution - Pytorch equivalent: in_proj, conv1 (mlp), conv2 (mlp), linear (H=W=1) >
 * W8A8 symmetric (w_zp = 0). bias is int32 pre-scaled by the layer's combined dequant
 * scale: round(bias_f32 / dequant_scale[oc]). For per-tensor a_scale layers that is
 * a_scale * w_scale[oc]; for per-channel a_scale layers the loader must first fold
 * a_scale[c] * a_scale_mul into the weights and re-derive w_scale (see PROGRESS.md).
 * Output is the raw int32 accumulator - caller applies dequant_int32_to_f32_perchannel. */
void pointwise_conv2d_int8(const int8_t *restrict x,
                        const int8_t *restrict weight,
                        const int32_t *restrict bias,
                        int32_t *restrict y,
                        uint32_t C_IN,
                        uint32_t H,
                        uint32_t W,
                        uint32_t C_OUT);

/* <2. Depth-wise convolution 2d - Pytorch equivalent: conv2d (ss2d) >
 * stride=1, same-padding (pad = kernel_size/2): H_out = H, W_out = W. */
void depthwise_conv2d_int8(const int8_t *restrict x,
                        const int8_t *restrict weight,
                        const int32_t *restrict bias,
                        int32_t *restrict y,
                        uint32_t C,
                        uint32_t H,
                        uint32_t W,
                        uint32_t kernel_size);

/* <3. Half-split input point-wise convolution 2d - Pytorch equivalent: skip_handler = nn.Conv2d(...) > */
void pointwise_conv2d_split2_int8(const int8_t *restrict x_inout,  // [C_IN, H, W]
                                const int8_t *restrict x_res,     // [C_IN, H, W]
                                const int8_t *restrict weight,    // [C_OUT, 2*C_IN]
                                const int32_t *restrict bias,     // [C_OUT] or NULL
                                int32_t *restrict y,              // [C_OUT, H, W]
                                uint32_t C_IN, uint32_t H, uint32_t W, uint32_t C_OUT);

/* <4. Conv 2D (regular conv2D) - Pytorch equivalent: patch_embed conv > */
void conv2d_int8(const int8_t *x,
                const int8_t *weight,
                const int32_t *bias,
                int32_t *y,
                uint32_t C_IN, uint32_t H_IN, uint32_t W_IN,
                uint32_t C_OUT, uint32_t H_OUT, uint32_t W_OUT,
                uint32_t K_SIZE, uint32_t STRIDE, uint32_t PAD);

/* ---------------------------------------------- */
/* III. ACTIVATION */
/* --- F32 --- */
/* <1. SiLU (inplace) - Pytorch equivalent: torch.nn.SiLU > */
void silu_f32(float *restrict z, uint32_t n);

/* <2. GeLU (inplace) - Pytorch equivalent: torch.nn.GeLU > */
void gelu_f32(float *x, uint32_t n);

/* <3. ReLU (inplace) - Pytorch equivalent: torch.nn.ReLU > */
void relu_f32(const float *x, float *y, uint32_t n, int inplace);

/* --- INT8 --- */
/* <4. ReLU on the int32 conv accumulator (inplace) - fused MLP fc1 ReLU applied
 *     before dequant. Bit-identical to dequant->relu_f32 because deq_scale[oc] > 0. > */
void relu_int32(int32_t *x, uint32_t n);

/* <5. Depth-wise conv2d on the int32 accumulator - int32 sibling of
 *     depthwise_conv2d_f32. weight==NULL => unit taps (pure window sum); bias==NULL
 *     => 0. Used for the fused encoder MLP SumPool (called weight=NULL since the
 *     checkpoint mlp_sumpool_w is all 1.0) before dequant; deq_scale>0 factors the
 *     scale out. In-place safe (y may alias x) for downsampling (stride>=kernel).
 *     NOT bit-identical to the f32 sumpool. > */
void depthwise_conv2d_int32(const int32_t *x,
                            const int32_t *weight,
                            const int32_t *bias,
                            int32_t *y,
                            uint32_t C, uint32_t H, uint32_t W,
                            uint32_t kernel_size, uint32_t stride, uint32_t pad);

/* ---------------------------------------------- */
/* IV. NORM */
/* --- F32 --- */
/* <1. Layer Norm > */
void layernorm_f32(float *x,
                    const float *gamma,
                    const float *beta,
                    uint32_t C, uint32_t L, float eps);

/* ---------------------------------------------- */
/* V. QUANT */
/* --- INT8 --- */
/* <1. Per-channel activation quant: float -> int8 (a_scale is [C_IN], SmoothQuant-migrated) > */
void quant_f32_to_int8_perchannel(const float *x,       /* [C, L] */
                                int8_t *y,              /* [C, L] */
                                uint32_t C,
                                uint32_t L,
                                const float *scales,    /* [C], a_scale from checkpoint */
                                int32_t zp);            /* typically 0 for symmetric */

/* <2. Per-channel REQUANT: int32 acc -> int8, fusing a layer's dequant with the
 *     next layer's activation quant into one f32 scale+round+clamp+cast.
 *     q[c,l] = clamp(lrintf((float)x[c,l] * requant_scale[c]), -128, 127),
 *     requant_scale[c] = deq_scale[c] / act_scale_next[c] (precomputed at export).
 *     Removes one f32 rounding vs dequant->quant: NOT bit-identical, differs only
 *     at rounding ties (< 3e-6 of values, by +-1 LSB). > */
void requant_int32_to_int8_perchannel(const int32_t *x,       /* [C, L] int32 acc */
                                    int8_t *y,                 /* [C, L] int8 */
                                    uint32_t C,
                                    uint32_t L,
                                    const float *requant_scale); /* [C], deq / act_next */

/* <2. v6: INTEGER-ONLY per-channel requant (fixed-point / dyadic multiplier).
 *     Replaces the f32 multiply + lrintf with int multiply + add + arithmetic shift:
 *       q[c,l] = clamp( (((int64)x[c,l] * mul[c]) + (1<<(shift[c]-1))) >> shift[c], -128, 127 )
 *     mul[c]/shift[c] approximate requant_scale[c] ~= mul[c] / 2^shift[c] (per-channel,
 *     searched & precomputed by weight_extractor_sq_int8_v6.py). No float at runtime;
 *     differs from the f32 requant only at rounding ties (<= +-1 LSB). For shift==0 the
 *     add/shift are skipped (q = x*mul). > */
void requant_int32_to_int8_perchannel_fixed(const int32_t *x,   /* [C, L] int32 acc */
                                    int8_t *y,                   /* [C, L] int8 */
                                    uint32_t C,
                                    uint32_t L,
                                    const int32_t *mul,          /* [C] fixed-point multiplier */
                                    const int32_t *shift);       /* [C] right-shift amount (>=0) */

/* ---------------------------------------------- */
/* VI. DEQUANT */
/* --- INT8 --- */
/* <1. Per-channel dequant: int32 accumulator -> float, scale_oc = a_scale_in * w_scale[oc] >
 * Every quantized layer has per-output-channel w_scale, so dequant is always per-channel. */
void dequant_int32_to_f32_perchannel(const int32_t *x,  /* [C, L] */
                                    float *y,           /* [C, L] */
                                    uint32_t C,
                                    uint32_t L,
                                    const float *scales);   /* [C], one per output channel */

#endif /* __UPVM_ASR_MICRO_KERNELS_H__ */
