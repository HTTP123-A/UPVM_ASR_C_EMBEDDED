/* Author: Nguyen Quang Phuong
 * Email: ngquangph@gmail.com
 *
 * Notes:
 * Dual-stream parallel VMamba ASR (C version).
 * 
 */

/* ===================== INCLUDE SECTION ===================== */
/*< 1. Built-in header >*/

/*< 2. User headers >*/
#include "model.h"
#include "weight_loader.h"

/* ===================== GLOBAL VARIABLES ==================== */


/* ===================== FUNCTION DEFINE SECTION ============= */
/* ---------------------------------------------- */
/*< 1. Print a weight-loading error and terminate >*/
void die_weight_io(const char *msg, const char *path)
{
    if (path != NULL) {
        fprintf(stderr, "%s: %s", msg, path);
        if (errno != 0) {
            fprintf(stderr, " (%s)", strerror(errno));
        }
        fputc('\n', stderr);
    } else {
        fprintf(stderr, "%s\n", msg);
    }
    exit(EXIT_FAILURE);
}

/*< 2. Compose one weight-file path from root/stage/member >*/
void build_weight_file_path(char *dst,
                            size_t dst_size,
                            const char *weight_root,
                            const char *stage_dir,
                            const char *member_name)
{
    const int written = snprintf(dst, dst_size, "%s/%s/%s.bin", weight_root, stage_dir, member_name);
    if (written < 0 || (size_t)written >= dst_size) {
        die_weight_io("weight file path is too long", NULL);
    }
}

/*< 3. Return file size in bytes or terminate on failure >*/
long file_size_bytes_or_die(const char *path)
{
    FILE *f = fopen(path, "rb");
    long size_bytes;

    if (f == NULL) {
        die_weight_io("failed to open weight file", path);
    }

    if (fseek(f, 0L, SEEK_END) != 0) {
        fclose(f);
        die_weight_io("failed to seek weight file", path);
    }

    size_bytes = ftell(f);
    fclose(f);

    if (size_bytes < 0L) {
        die_weight_io("failed to get weight file size", path);
    }

    return size_bytes;
}

/*< 4. Allocate float32 storage for one weight tensor >*/
float *alloc_f32_or_die(size_t numel)
{
    float *buf = (float *)malloc(numel * sizeof(float));
    if (buf == NULL) {
        die_weight_io("failed to allocate weight buffer", NULL);
    }
    return buf;
}

/*< 5. Read an exact float32 tensor file after size validation >*/
void read_f32_file_exact(const char *path, float *dst, size_t numel)
{
    FILE *f = fopen(path, "rb");
    const long expected_bytes = (long)(numel * sizeof(float));
    const long actual_bytes = file_size_bytes_or_die(path);

    if (actual_bytes != expected_bytes) {
        fprintf(stderr,
                "weight size mismatch for %s: expected %ld bytes (%zu floats), got %ld bytes\n",
                path,
                expected_bytes,
                numel,
                actual_bytes);
        exit(EXIT_FAILURE);
    }

    if (f == NULL) {
        die_weight_io("failed to open weight file", path);
    }

    if (fread(dst, sizeof(float), numel, f) != numel) {
        fclose(f);
        die_weight_io("failed to read weight file", path);
    }

    fclose(f);
}

/*< 6. Load all tensor files for one stage from its folder >*/
void load_weight_stage(const char *weight_root,
                    const char *stage_dir,
                    const struct weight_file_spec *specs,
                    size_t spec_count)
{
    char path_buf[WEIGHT_PATH_BUF_SIZE];
    size_t i;

    for (i = 0U; i < spec_count; ++i) {
        float *buf;

        build_weight_file_path(path_buf,
                            sizeof(path_buf),
                            weight_root,
                            stage_dir,
                            specs[i].member_name);

        buf = alloc_f32_or_die(specs[i].numel);
        read_f32_file_exact(path_buf, buf, specs[i].numel);
        *(specs[i].slot) = buf;
    }
}

/*< 7. Load all PatchEmbed weights for one stream >*/
void load_embed_weight_stage(struct embed_weight *weight,
                            const char *weight_root,
                            const char *stage_dir,
                            uint32_t c_in, uint32_t c_mid, uint32_t c_out, uint32_t kernel_size)
{
    const struct weight_file_spec specs[] = {
        {"conv1_w", (size_t)c_mid * c_in * kernel_size * kernel_size, &weight->conv1_w},
        {"conv1_b", (size_t)c_mid, &weight->conv1_b},
        {"norm1_w", (size_t)c_mid, &weight->norm1_w},
        {"norm1_b", (size_t)c_mid, &weight->norm1_b},
        {"conv2_w", (size_t)c_out * c_mid * kernel_size * kernel_size, &weight->conv2_w},
        {"conv2_b", (size_t)c_out, &weight->conv2_b},
        {"norm2_w", (size_t)c_out, &weight->norm2_w},
        {"norm2_b", (size_t)c_out, &weight->norm2_b},
    };

    load_weight_stage(weight_root, stage_dir, specs, ARRAY_LEN(specs));
}

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
                            uint32_t sub_c)
{
    const struct weight_file_spec specs[] = {
        {"ssm_in_norm_w", (size_t)c, &weight->ssm_in_norm_w},
        {"ssm_in_norm_b", (size_t)c, &weight->ssm_in_norm_b},
        {"ssm_in_projection_w", (size_t)sub_d_projection * sub_c, &weight->ssm_in_projection_w},
        {"ssm_dw_conv2d_w", (size_t)sub_d_inner * dw_kernel_size * dw_kernel_size, &weight->ssm_dw_conv2d_w},
        {"ssm_dw_conv2d_b", (size_t)sub_d_inner, &weight->ssm_dw_conv2d_b},
        {"ssm_dw_conv1d_w", (size_t)4U * r2n * sub_d_inner, &weight->ssm_dw_conv1d_w},
        {"ssm_dt_projection_w", (size_t)4U * sub_d_inner * r, &weight->ssm_dt_projection_w},
        {"ssm_A", (size_t)4U * sub_d_inner * n, &weight->ssm_A},
        {"ssm_Ds", (size_t)4U * sub_d_inner, &weight->ssm_Ds},
        {"ssm_delta_bias", (size_t)4U * sub_d_inner, &weight->ssm_delta_bias},
        {"ssm_out_norm_w", (size_t)sub_d_inner, &weight->ssm_out_norm_w},
        {"ssm_out_norm_b", (size_t)sub_d_inner, &weight->ssm_out_norm_b},
        {"ssm_out_projection_w", (size_t)sub_c * sub_d_inner, &weight->ssm_out_projection_w},
        {"mlp_norm_w", (size_t)c, &weight->mlp_norm_w},
        {"mlp_norm_b", (size_t)c, &weight->mlp_norm_b},
        {"mlp_fc1_w", (size_t)mlp_hidden_dim * c, &weight->mlp_fc1_w},
        {"mlp_fc1_b", (size_t)mlp_hidden_dim, &weight->mlp_fc1_b},
        {"mlp_fc2_w", (size_t)(2U * c) * mlp_hidden_dim, &weight->mlp_fc2_w},
        {"mlp_fc2_b", (size_t)(2U * c), &weight->mlp_fc2_b},
        {"mlp_fc_res_w", (size_t)(2U * c) * mlp_hidden_dim, &weight->mlp_fc_res_w},
        {"mlp_fc_res_b", (size_t)(2U * c), &weight->mlp_fc_res_b},
        {"mlp_sumpool_w", (size_t)c * 2U * 2U, &weight->mlp_sumpool_w},
        {"mlp_dim_reduce_w", 1U, &weight->mlp_dim_reduce_w},
        {"skip_reduce_w", 1U, &weight->skip_reduce_w},
    };

    load_weight_stage(weight_root, stage_dir, specs, ARRAY_LEN(specs));
}

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
                                uint32_t sub_c)
{
    const struct weight_file_spec specs[] = {
        {"ssm_in_norm_w", (size_t)c, &weight->ssm_in_norm_w},
        {"ssm_in_norm_b", (size_t)c, &weight->ssm_in_norm_b},
        {"ssm_in_projection_w", (size_t)sub_d_projection * sub_c, &weight->ssm_in_projection_w},
        {"ssm_dw_conv2d_w", (size_t)sub_d_inner * dw_kernel_size * dw_kernel_size, &weight->ssm_dw_conv2d_w},
        {"ssm_dw_conv2d_b", (size_t)sub_d_inner, &weight->ssm_dw_conv2d_b},
        {"ssm_dw_conv1d_w", (size_t)4U * r2n * sub_d_inner, &weight->ssm_dw_conv1d_w},
        {"ssm_dt_projection_w", (size_t)4U * sub_d_inner * r, &weight->ssm_dt_projection_w},
        {"ssm_A", (size_t)4U * sub_d_inner * n, &weight->ssm_A},
        {"ssm_Ds", (size_t)4U * sub_d_inner, &weight->ssm_Ds},
        {"ssm_delta_bias", (size_t)4U * sub_d_inner, &weight->ssm_delta_bias},
        {"ssm_out_norm_w", (size_t)sub_d_inner, &weight->ssm_out_norm_w},
        {"ssm_out_norm_b", (size_t)sub_d_inner, &weight->ssm_out_norm_b},
        {"ssm_out_projection_w", (size_t)sub_c * sub_d_inner, &weight->ssm_out_projection_w},
        {"mlp_norm_w", (size_t)c, &weight->mlp_norm_w},
        {"mlp_norm_b", (size_t)c, &weight->mlp_norm_b},
        {"mlp_fc1_w", (size_t)mlp_hidden_dim * c, &weight->mlp_fc1_w},
        {"mlp_fc1_b", (size_t)mlp_hidden_dim, &weight->mlp_fc1_b},
        {"mlp_fc2_w", (size_t)c * mlp_hidden_dim, &weight->mlp_fc2_w},
        {"mlp_fc2_b", (size_t)c, &weight->mlp_fc2_b},
    };

    load_weight_stage(weight_root, stage_dir, specs, ARRAY_LEN(specs));
}

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
                            uint32_t sub_c)
{
    const struct weight_file_spec specs[] = {
        {"ssm_in_norm_w", (size_t)norm_dim, &weight->ssm_in_norm_w},
        {"ssm_in_norm_b", (size_t)norm_dim, &weight->ssm_in_norm_b},
        {"ssm_in_projection_w", (size_t)sub_d_projection * sub_c, &weight->ssm_in_projection_w},
        {"ssm_dw_conv2d_w", (size_t)sub_d_inner * dw_kernel_size * dw_kernel_size, &weight->ssm_dw_conv2d_w},
        {"ssm_dw_conv2d_b", (size_t)sub_d_inner, &weight->ssm_dw_conv2d_b},
        {"ssm_dw_conv1d_w", (size_t)4U * r2n * sub_d_inner, &weight->ssm_dw_conv1d_w},
        {"ssm_dt_projection_w", (size_t)4U * sub_d_inner * r, &weight->ssm_dt_projection_w},
        {"ssm_A", (size_t)4U * sub_d_inner * n, &weight->ssm_A},
        {"ssm_Ds", (size_t)4U * sub_d_inner, &weight->ssm_Ds},
        {"ssm_delta_bias", (size_t)4U * sub_d_inner, &weight->ssm_delta_bias},
        {"ssm_out_norm_w", (size_t)sub_d_inner, &weight->ssm_out_norm_w},
        {"ssm_out_norm_b", (size_t)sub_d_inner, &weight->ssm_out_norm_b},
        {"ssm_out_projection_w", (size_t)sub_c * sub_d_inner, &weight->ssm_out_projection_w},
        {"mlp_norm_w", (size_t)norm_dim, &weight->mlp_norm_w},
        {"mlp_norm_b", (size_t)norm_dim, &weight->mlp_norm_b},
        {"mlp_fc1_w", (size_t)mlp_hidden_dim * c, &weight->mlp_fc1_w},
        {"mlp_fc1_b", (size_t)mlp_hidden_dim, &weight->mlp_fc1_b},
        {"mlp_fc2_w", (size_t)c * mlp_hidden_dim, &weight->mlp_fc2_w},
        {"mlp_fc2_b", (size_t)c, &weight->mlp_fc2_b},
        {"pe_w", (size_t)(2U * c) * c, &weight->pe_w},
        {"pe_norm_w", (size_t)(c / 2U), &weight->pe_norm_w},
        {"pe_norm_b", (size_t)(c / 2U), &weight->pe_norm_b},
    };

    load_weight_stage(weight_root, stage_dir, specs, ARRAY_LEN(specs));
}

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
                                uint32_t d_projection)
{
    const struct weight_file_spec specs[] = {
        {"pre_conv2d_w", (size_t)c * (4U * c), &weight->pre_conv2d_w},
        {"pre_conv2d_b", (size_t)c, &weight->pre_conv2d_b},
        {"ssm_in_norm_w", (size_t)c, &weight->ssm_in_norm_w},
        {"ssm_in_norm_b", (size_t)c, &weight->ssm_in_norm_b},
        {"ssm_in_projection_w", (size_t)d_projection * c, &weight->ssm_in_projection_w},
        {"ssm_dw_conv2d_w", (size_t)d_inner * dw_kernel_size * dw_kernel_size, &weight->ssm_dw_conv2d_w},
        {"ssm_dw_conv2d_b", (size_t)d_inner, &weight->ssm_dw_conv2d_b},
        {"ssm_dw_conv1d_w", (size_t)4U * r2n * d_inner, &weight->ssm_dw_conv1d_w},
        {"ssm_dt_projection_w", (size_t)4U * d_inner * r, &weight->ssm_dt_projection_w},
        {"ssm_A", (size_t)4U * d_inner * n, &weight->ssm_A},
        {"ssm_Ds", (size_t)4U * d_inner, &weight->ssm_Ds},
        {"ssm_delta_bias", (size_t)4U * d_inner, &weight->ssm_delta_bias},
        {"ssm_out_norm_w", (size_t)d_inner, &weight->ssm_out_norm_w},
        {"ssm_out_norm_b", (size_t)d_inner, &weight->ssm_out_norm_b},
        {"ssm_out_projection_w", (size_t)c * d_inner, &weight->ssm_out_projection_w},
        {"mlp_norm_w", (size_t)c, &weight->mlp_norm_w},
        {"mlp_norm_b", (size_t)c, &weight->mlp_norm_b},
        {"mlp_fc1_w", (size_t)mlp_hidden_dim * c, &weight->mlp_fc1_w},
        {"mlp_fc1_b", (size_t)mlp_hidden_dim, &weight->mlp_fc1_b},
        {"mlp_fc2_w", (size_t)c * mlp_hidden_dim, &weight->mlp_fc2_w},
        {"mlp_fc2_b", (size_t)c, &weight->mlp_fc2_b},
    };

    load_weight_stage(weight_root, stage_dir, specs, ARRAY_LEN(specs));
}

/*< 12. Load all upstream skip-handler weights for one stream >*/
void load_pvss_us_skip_weight_stage(struct pvss_us_skip_weight *weight,
                                    const char *weight_root,
                                    const char *stage_dir)
{
    const struct weight_file_spec specs[] = {
        {"skip_conv_dec0_w", (size_t)C_DEC_0 * (2U * C_DEC_0), &weight->skip_conv_dec0_w},
        {"skip_conv_dec0_b", (size_t)C_DEC_0, &weight->skip_conv_dec0_b},
        {"skip_conv_dec1_w", (size_t)C_DEC_1 * (2U * C_DEC_1), &weight->skip_conv_dec1_w},
        {"skip_conv_dec1_b", (size_t)C_DEC_1, &weight->skip_conv_dec1_b},
        {"skip_conv_dec2_w", (size_t)C_DEC_2 * (2U * C_DEC_2), &weight->skip_conv_dec2_w},
        {"skip_conv_dec2_b", (size_t)C_DEC_2, &weight->skip_conv_dec2_b},
        {"skip_conv_out0_w", (size_t)C_OUT_0 * (2U * C_OUT_0), &weight->skip_conv_out0_w},
        {"skip_conv_out0_b", (size_t)C_OUT_0, &weight->skip_conv_out0_b},
    };

    load_weight_stage(weight_root, stage_dir, specs, ARRAY_LEN(specs));
}