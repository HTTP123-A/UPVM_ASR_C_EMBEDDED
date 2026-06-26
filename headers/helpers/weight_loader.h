#ifndef _WEIGHT_LOADING_
#define _WEIGHT_LOADING_

/* ===================== INCLUDE SECTION ===================== */
/* <1. Built-in header > */
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* <2. User headers > */
#include "datatypes.h"

/* ===================== DEFINE ============================== */
#define WEIGHT_PATH_BUF_SIZE    4096U

/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/* ---------------------------------------------- */
/*< 1. Print a weight-loading error and terminate >*/
void die_weight_io(const char *msg, const char *path);

/*< 2. Compose one weight-file path from root/stage/member >*/
void build_weight_file_path(char *dst,
                            size_t dst_size,
                            const char *weight_root,
                            const char *stage_dir,
                            const char *member_name);

/*< 3. Return file size in bytes or terminate on failure >*/
long file_size_bytes_or_die(const char *path);

/*< 4. Allocate float32 storage for one weight tensor >*/
float *alloc_f32_or_die(size_t numel);

/*< 5. Read an exact float32 tensor file after size validation >*/
void read_f32_file_exact(const char *path, float *dst, size_t numel);

/*< 6. Load all tensor files for one stage from its folder >*/
void load_weight_stage(const char *weight_root,
                    const char *stage_dir,
                    const struct weight_file_spec *specs,
                    size_t spec_count);

/*< 7. Load all PatchEmbed weights for one stream >*/
void load_embed_weight_stage(struct embed_weight *weight,
                            const char *weight_root,
                            const char *stage_dir,
                            uint32_t c_in, uint32_t c_mid, uint32_t c_out, uint32_t kernel_size);

/*< 8. Load one encoder PVSS-DS weight set >*/
void load_pvss_ds_weight_stage(struct pvss_ds_weight *weight,
                            const char *weight_root,
                            const char *stage_dir,
                            uint32_t c,
                            uint32_t r,
                            uint32_t n,
                            uint32_t r2n,
                            uint32_t dw_kernel_size,
                            uint32_t mlp_hidden_dim,
                            uint32_t sub_d_inner,
                            uint32_t sub_d_projection,
                            uint32_t sub_c);

/*< 9. Load one latent PVSS weight set >*/
void load_pvss_latent_weight_stage(struct pvss_latent_weight *weight,
                                const char *weight_root,
                                const char *stage_dir,
                                uint32_t c,
                                uint32_t r,
                                uint32_t n,
                                uint32_t r2n,
                                uint32_t dw_kernel_size,
                                uint32_t mlp_hidden_dim,
                                uint32_t sub_d_inner,
                                uint32_t sub_d_projection,
                                uint32_t sub_c);

/*< 10. Load one decoder/output PVSS-US weight set >*/
void load_pvss_us_weight_stage(struct pvss_us_weight *weight,
                            const char *weight_root,
                            const char *stage_dir,
                            uint32_t c,
                            uint32_t norm_dim,
                            uint32_t r,
                            uint32_t n,
                            uint32_t r2n,
                            uint32_t dw_kernel_size,
                            uint32_t mlp_hidden_dim,
                            uint32_t sub_d_inner,
                            uint32_t sub_d_projection,
                            uint32_t sub_c);

/*< 11. Load one output3 VSS weight set >*/
void load_vss_output_weight_stage(struct vss_output_weight *weight,
                                const char *weight_root,
                                const char *stage_dir,
                                uint32_t c,
                                uint32_t r,
                                uint32_t n,
                                uint32_t r2n,
                                uint32_t dw_kernel_size,
                                uint32_t mlp_hidden_dim,
                                uint32_t d_inner,
                                uint32_t d_projection);

/*< 12. Load all upstream skip-handler weights for one stream >*/
void load_pvss_us_skip_weight_stage(struct pvss_us_skip_weight *weight,
                                    const char *weight_root,
                                    const char *stage_dir);

#endif /* __WEIGHT_LOADING__ */