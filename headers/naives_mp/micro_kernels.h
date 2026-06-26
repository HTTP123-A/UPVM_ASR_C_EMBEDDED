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

/*< 3. Half-split input point-wise convolution 2d - Pytorch equivalent: skip_handler = nn.Conv2d(...) >*/
void pointwise_conv2d_split2_f32(const float *restrict x_inout,   // [C, H, W]
                                const float *restrict x_res,     // [C, H, W]
                                const float *restrict weight,    // [C, 2*C]
                                const float *restrict bias,      // [C] or NULL
                                float *restrict y,               // [C, H, W]
                                uint32_t C_IN, uint32_t H, uint32_t W, uint32_t C_OUT);

/*< 4. Group point-wise convolution 1d - Pytorch equivalent: >*/
void grouped_pointwise_conv1d_f32(const float *x,
                                const float *weights,
                                const float *bias,
                                float *y,                           
                                uint32_t SUB_D_IN, uint32_t K, uint32_t L, uint32_t SUB_D_OUT);

/*< 5. Group point-wise convolution 1d (type major) - Pytorch equivalent: >*/
void grouped_pointwise_conv1d_type_major_f32(const float *x,
                                            const float *weights,
                                            const float *bias,
                                            float *y,
                                            uint32_t SUB_D_IN, uint32_t K, uint32_t L, uint32_t R, uint32_t N);

/* <6. Element-wise add (inplace) - Pytorch equivalent: Hadamard addition c = a + b > */
void elemwise_add_f32(float *x, const float *r, uint32_t N);

/* <7. Element-wise mul (inplace) - Pytorch equivalent: Hadamard product c = a .* b > */
void elemwise_mul_f32(float *x, const float *z, uint32_t N);

/*< 8. SSM core - Selective scan (naive FP16 path, f32 state/weights for stability) >*/
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

/*< 9. Conv 2D (regular conv2D) >*/
void conv2d_f32(const float *x,
                const float *weight,
                const float *bias,
                float *y,
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

/* ---------------------------------------------- */
/* IV. NORM */
/* --- F32 --- */
/* <1. Layer Norm > */
void layernorm_f32(float *x,
                    const float *gamma,
                    const float *beta,
                    uint32_t C, uint32_t L, float eps);

#endif /* __UPVM_ASR_MICRO_KERNELS_H__ */
