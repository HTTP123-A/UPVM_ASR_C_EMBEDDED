/* Author: Nguyen Quang Phuong
 * Email: ngquangph@gmail.com
 *
 * Notes:
 * Dual-stream parallel VMamba ASR (C version, INT8-F32 mixed precision).
 * Loads data/weight_sq_int8_v0 exported by weight_extractor_sq_int8_v0.py.
 * All quantization fold math was done at export time - this loader only reads
 * arrays (see sources/sq_int8_v0/IMPLEMENTATION_NOTES.md).
 */

/* ===================== INCLUDE SECTION ===================== */
/*< 1. Built-in header >*/

/*< 2. User headers >*/
#include "model.h"
#include "weight_loader_f32_i8.h"

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

/*< 4. Allocate storage for one weight tensor >*/
static void *alloc_bytes_or_die(size_t numel, size_t elem_size)
{
    void *buf = malloc(numel * elem_size);
    if (buf == NULL) {
        die_weight_io("failed to allocate weight buffer", NULL);
    }
    return buf;
}

float *alloc_f32_or_die(size_t numel)
{
    return (float *)alloc_bytes_or_die(numel, sizeof(float));
}

/*< 5. Read an exact tensor file after size validation >*/
static void read_file_exact(const char *path, void *dst, size_t elem_size, size_t numel)
{
    FILE *f = fopen(path, "rb");
    const long expected_bytes = (long)(numel * elem_size);
    const long actual_bytes = file_size_bytes_or_die(path);

    if (actual_bytes != expected_bytes) {
        fprintf(stderr,
                "weight size mismatch for %s: expected %ld bytes (%zu elements), got %ld bytes\n",
                path,
                expected_bytes,
                numel,
                actual_bytes);
        exit(EXIT_FAILURE);
    }

    if (f == NULL) {
        die_weight_io("failed to open weight file", path);
    }

    if (fread(dst, elem_size, numel, f) != numel) {
        fclose(f);
        die_weight_io("failed to read weight file", path);
    }

    fclose(f);
}

void read_f32_file_exact(const char *path, float *dst, size_t numel)
{
    read_file_exact(path, dst, sizeof(float), numel);
}

/*< 6. Load all float32 tensor files for one stage from its folder >*/
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

/*< 7. Load all int8 quantized layers for one stage from its folder >*/
static void *load_qconv_member_file(const char *weight_root,
                                    const char *stage_dir,
                                    const char *base_name,
                                    const char *suffix,
                                    size_t elem_size,
                                    size_t numel)
{
    char member_buf[WEIGHT_MEMBER_BUF_SIZE];
    char path_buf[WEIGHT_PATH_BUF_SIZE];
    void *buf;

    if ((size_t)snprintf(member_buf, sizeof(member_buf), "%s%s", base_name, suffix) >= sizeof(member_buf)) {
        die_weight_io("qconv member name is too long", base_name);
    }

    build_weight_file_path(path_buf, sizeof(path_buf), weight_root, stage_dir, member_buf);
    buf = alloc_bytes_or_die(numel, elem_size);
    read_file_exact(path_buf, buf, elem_size, numel);
    return buf;
}

void load_qconv_stage(const char *weight_root,
                    const char *stage_dir,
                    const struct qconv_file_spec *specs,
                    size_t spec_count)
{
    size_t i;

    for (i = 0U; i < spec_count; ++i) {
        const struct qconv_file_spec *spec = &specs[i];
        struct qconv_int8 *q = spec->slot;

        q->w_i8 = (const int8_t *)load_qconv_member_file(weight_root, stage_dir,
                        spec->member_name, "_w", sizeof(int8_t), spec->w_numel);
        q->act_scale = (const float *)load_qconv_member_file(weight_root, stage_dir,
                        spec->member_name, "_w_quant_scale", sizeof(float), spec->act_scale_len);
        q->deq_scale = (const float *)load_qconv_member_file(weight_root, stage_dir,
                        spec->member_name, "_w_dequant_scale", sizeof(float), spec->c_out);
        q->bias_i32 = (spec->has_bias != 0)
            ? (const int32_t *)load_qconv_member_file(weight_root, stage_dir,
                        spec->member_name, "_b_i32", sizeof(int32_t), spec->c_out)
            : NULL;
        q->act_scale_len = (uint32_t)spec->act_scale_len;
    }
}

/*< 8. Load all PatchEmbed weights for one stream >*/
void load_embed_weight_stage(struct embed_weight *weight,
                            const char *weight_root,
                            const char *stage_dir,
                            uint32_t c_in, uint32_t c_mid, uint32_t c_out, uint32_t kernel_size)
{
    const struct weight_file_spec f32_specs[] = {
        {"norm1_w", (size_t)c_mid, &weight->norm1_w},
        {"norm1_b", (size_t)c_mid, &weight->norm1_b},
        {"norm2_w", (size_t)c_out, &weight->norm2_w},
        {"norm2_b", (size_t)c_out, &weight->norm2_b},
    };
    const struct qconv_file_spec qconv_specs[] = {
        {"conv1", (size_t)c_mid * c_in * kernel_size * kernel_size, (size_t)c_mid, (size_t)c_in, 1, &weight->conv1},
        {"conv2", (size_t)c_out * c_mid * kernel_size * kernel_size, (size_t)c_out, (size_t)c_mid, 1, &weight->conv2},
    };

    load_weight_stage(weight_root, stage_dir, f32_specs, ARRAY_LEN(f32_specs));
    load_qconv_stage(weight_root, stage_dir, qconv_specs, ARRAY_LEN(qconv_specs));
}

/*< 9. Load one encoder PVSS-DS weight set >*/
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
    const struct weight_file_spec f32_specs[] = {
        {"ssm_in_norm_w", (size_t)c, &weight->ssm_in_norm_w},
        {"ssm_in_norm_b", (size_t)c, &weight->ssm_in_norm_b},
        {"ssm_dw_conv1d_w", (size_t)4U * r2n * sub_d_inner, &weight->ssm_dw_conv1d_w},
        {"ssm_dt_projection_w", (size_t)4U * sub_d_inner * r, &weight->ssm_dt_projection_w},
        {"ssm_A", (size_t)4U * sub_d_inner * n, &weight->ssm_A},
        {"ssm_Ds", (size_t)4U * sub_d_inner, &weight->ssm_Ds},
        {"ssm_delta_bias", (size_t)4U * sub_d_inner, &weight->ssm_delta_bias},
        {"ssm_out_norm_w", (size_t)sub_d_inner, &weight->ssm_out_norm_w},
        {"ssm_out_norm_b", (size_t)sub_d_inner, &weight->ssm_out_norm_b},
        {"ssm_requant_scale", (size_t)sub_d_inner, &weight->ssm_requant_scale},
        {"mlp_norm_w", (size_t)c, &weight->mlp_norm_w},
        {"mlp_norm_b", (size_t)c, &weight->mlp_norm_b},
        {"mlp_fc_res_w", (size_t)(2U * c) * mlp_hidden_dim, &weight->mlp_fc_res_w},
        {"mlp_fc_res_b", (size_t)(2U * c), &weight->mlp_fc_res_b},
        {"mlp_sumpool_w", (size_t)c * 2U * 2U, &weight->mlp_sumpool_w},
        {"mlp_requant_scale", (size_t)mlp_hidden_dim, &weight->mlp_requant_scale},
        {"mlp_dim_reduce_w", 1U, &weight->mlp_dim_reduce_w},
        {"skip_reduce_w", 1U, &weight->skip_reduce_w},
    };
    const struct qconv_file_spec qconv_specs[] = {
        {"ssm_in_projection", (size_t)sub_d_projection * sub_c, (size_t)sub_d_projection, (size_t)sub_c, 0, &weight->ssm_in_projection},
        {"ssm_dw_conv2d", (size_t)sub_d_inner * dw_kernel_size * dw_kernel_size, (size_t)sub_d_inner, (size_t)sub_d_inner, 1, &weight->ssm_dw_conv2d},
        {"ssm_out_projection", (size_t)sub_c * sub_d_inner, (size_t)sub_c, (size_t)sub_d_inner, 0, &weight->ssm_out_projection},
        {"mlp_fc1", (size_t)mlp_hidden_dim * c, (size_t)mlp_hidden_dim, (size_t)c, 1, &weight->mlp_fc1},
        {"mlp_fc2", (size_t)(2U * c) * mlp_hidden_dim, (size_t)(2U * c), (size_t)mlp_hidden_dim, 0, &weight->mlp_fc2},
    };

    load_weight_stage(weight_root, stage_dir, f32_specs, ARRAY_LEN(f32_specs));
    load_qconv_stage(weight_root, stage_dir, qconv_specs, ARRAY_LEN(qconv_specs));
}

/*< 10. Load one latent PVSS weight set >*/
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
    const struct weight_file_spec f32_specs[] = {
        {"ssm_in_norm_w", (size_t)c, &weight->ssm_in_norm_w},
        {"ssm_in_norm_b", (size_t)c, &weight->ssm_in_norm_b},
        {"ssm_dw_conv1d_w", (size_t)4U * r2n * sub_d_inner, &weight->ssm_dw_conv1d_w},
        {"ssm_dt_projection_w", (size_t)4U * sub_d_inner * r, &weight->ssm_dt_projection_w},
        {"ssm_A", (size_t)4U * sub_d_inner * n, &weight->ssm_A},
        {"ssm_Ds", (size_t)4U * sub_d_inner, &weight->ssm_Ds},
        {"ssm_delta_bias", (size_t)4U * sub_d_inner, &weight->ssm_delta_bias},
        {"ssm_out_norm_w", (size_t)sub_d_inner, &weight->ssm_out_norm_w},
        {"ssm_out_norm_b", (size_t)sub_d_inner, &weight->ssm_out_norm_b},
        {"ssm_requant_scale", (size_t)sub_d_inner, &weight->ssm_requant_scale},
        {"mlp_norm_w", (size_t)c, &weight->mlp_norm_w},
        {"mlp_norm_b", (size_t)c, &weight->mlp_norm_b},
        {"mlp_requant_scale", (size_t)mlp_hidden_dim, &weight->mlp_requant_scale},
    };
    const struct qconv_file_spec qconv_specs[] = {
        {"ssm_in_projection", (size_t)sub_d_projection * sub_c, (size_t)sub_d_projection, (size_t)sub_c, 0, &weight->ssm_in_projection},
        {"ssm_dw_conv2d", (size_t)sub_d_inner * dw_kernel_size * dw_kernel_size, (size_t)sub_d_inner, (size_t)sub_d_inner, 1, &weight->ssm_dw_conv2d},
        {"ssm_out_projection", (size_t)sub_c * sub_d_inner, (size_t)sub_c, (size_t)sub_d_inner, 0, &weight->ssm_out_projection},
        {"mlp_fc1", (size_t)mlp_hidden_dim * c, (size_t)mlp_hidden_dim, (size_t)c, 1, &weight->mlp_fc1},
        {"mlp_fc2", (size_t)c * mlp_hidden_dim, (size_t)c, (size_t)mlp_hidden_dim, 1, &weight->mlp_fc2},
    };

    load_weight_stage(weight_root, stage_dir, f32_specs, ARRAY_LEN(f32_specs));
    load_qconv_stage(weight_root, stage_dir, qconv_specs, ARRAY_LEN(qconv_specs));
}

/*< 11. Load one decoder PVSS-US weight set (decoder only - norms are [c]) >*/
void load_pvss_us_weight_stage(struct pvss_us_weight *weight,
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
    const struct weight_file_spec f32_specs[] = {
        {"ssm_in_norm_w", (size_t)c, &weight->ssm_in_norm_w},
        {"ssm_in_norm_b", (size_t)c, &weight->ssm_in_norm_b},
        {"ssm_dw_conv1d_w", (size_t)4U * r2n * sub_d_inner, &weight->ssm_dw_conv1d_w},
        {"ssm_dt_projection_w", (size_t)4U * sub_d_inner * r, &weight->ssm_dt_projection_w},
        {"ssm_A", (size_t)4U * sub_d_inner * n, &weight->ssm_A},
        {"ssm_Ds", (size_t)4U * sub_d_inner, &weight->ssm_Ds},
        {"ssm_delta_bias", (size_t)4U * sub_d_inner, &weight->ssm_delta_bias},
        {"ssm_out_norm_w", (size_t)sub_d_inner, &weight->ssm_out_norm_w},
        {"ssm_out_norm_b", (size_t)sub_d_inner, &weight->ssm_out_norm_b},
        {"ssm_requant_scale", (size_t)sub_d_inner, &weight->ssm_requant_scale},
        {"mlp_norm_w", (size_t)c, &weight->mlp_norm_w},
        {"mlp_norm_b", (size_t)c, &weight->mlp_norm_b},
        {"mlp_requant_scale", (size_t)mlp_hidden_dim, &weight->mlp_requant_scale},
        {"pe_norm_w", (size_t)(c / 2U), &weight->pe_norm_w},
        {"pe_norm_b", (size_t)(c / 2U), &weight->pe_norm_b},
    };
    const struct qconv_file_spec qconv_specs[] = {
        {"ssm_in_projection", (size_t)sub_d_projection * sub_c, (size_t)sub_d_projection, (size_t)sub_c, 0, &weight->ssm_in_projection},
        {"ssm_dw_conv2d", (size_t)sub_d_inner * dw_kernel_size * dw_kernel_size, (size_t)sub_d_inner, (size_t)sub_d_inner, 1, &weight->ssm_dw_conv2d},
        {"ssm_out_projection", (size_t)sub_c * sub_d_inner, (size_t)sub_c, (size_t)sub_d_inner, 0, &weight->ssm_out_projection},
        {"mlp_fc1", (size_t)mlp_hidden_dim * c, (size_t)mlp_hidden_dim, (size_t)c, 1, &weight->mlp_fc1},
        {"mlp_fc2", (size_t)c * mlp_hidden_dim, (size_t)c, (size_t)mlp_hidden_dim, 1, &weight->mlp_fc2},
        {"pe", (size_t)(2U * c) * c, (size_t)(2U * c), (size_t)c, 0, &weight->pe},
    };

    load_weight_stage(weight_root, stage_dir, f32_specs, ARRAY_LEN(f32_specs));
    load_qconv_stage(weight_root, stage_dir, qconv_specs, ARRAY_LEN(qconv_specs));
}

/*< 12. Load one output0 VSS weight set (f32 ss2d, no block norms) >*/
void load_vss_output0_weight_stage(struct vss_output0_weight *weight,
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
    const struct weight_file_spec f32_specs[] = {
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
        {"pe_norm_w", (size_t)(c / 2U), &weight->pe_norm_w},
        {"pe_norm_b", (size_t)(c / 2U), &weight->pe_norm_b},
    };
    const struct qconv_file_spec qconv_specs[] = {
        {"mlp_fc1", (size_t)mlp_hidden_dim * c, (size_t)mlp_hidden_dim, (size_t)c, 1, &weight->mlp_fc1},
        {"mlp_fc2", (size_t)c * mlp_hidden_dim, (size_t)c, (size_t)mlp_hidden_dim, 1, &weight->mlp_fc2},
        {"pe", (size_t)(2U * c) * c, (size_t)(2U * c), (size_t)c, 0, &weight->pe},
    };

    load_weight_stage(weight_root, stage_dir, f32_specs, ARRAY_LEN(f32_specs));
    load_qconv_stage(weight_root, stage_dir, qconv_specs, ARRAY_LEN(qconv_specs));
}

/*< 13. Load one output1 VSS weight set (f32 ss2d, with block norms) >*/
void load_vss_output1_weight_stage(struct vss_output1_weight *weight,
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
    const struct weight_file_spec f32_specs[] = {
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
        {"pe_norm_w", (size_t)(c / 2U), &weight->pe_norm_w},
        {"pe_norm_b", (size_t)(c / 2U), &weight->pe_norm_b},
    };
    const struct qconv_file_spec qconv_specs[] = {
        {"mlp_fc1", (size_t)mlp_hidden_dim * c, (size_t)mlp_hidden_dim, (size_t)c, 1, &weight->mlp_fc1},
        {"mlp_fc2", (size_t)c * mlp_hidden_dim, (size_t)c, (size_t)mlp_hidden_dim, 1, &weight->mlp_fc2},
        {"pe", (size_t)(2U * c) * c, (size_t)(2U * c), (size_t)c, 0, &weight->pe},
    };

    load_weight_stage(weight_root, stage_dir, f32_specs, ARRAY_LEN(f32_specs));
    load_qconv_stage(weight_root, stage_dir, qconv_specs, ARRAY_LEN(qconv_specs));
}

/*< 14. Load one output3 VSS weight set (pre conv + f32 ss2d) >*/
void load_vss_output3_weight_stage(struct vss_output3_weight *weight,
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
    const struct weight_file_spec f32_specs[] = {
        {"pre_conv2d_w", (size_t)c * (4U * c), &weight->pre_conv2d_w},
        {"pre_conv2d_b", (size_t)c, &weight->pre_conv2d_b},
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
    };
    const struct qconv_file_spec qconv_specs[] = {
        {"mlp_fc1", (size_t)mlp_hidden_dim * c, (size_t)mlp_hidden_dim, (size_t)c, 1, &weight->mlp_fc1},
        {"mlp_fc2", (size_t)c * mlp_hidden_dim, (size_t)c, (size_t)mlp_hidden_dim, 1, &weight->mlp_fc2},
    };

    load_weight_stage(weight_root, stage_dir, f32_specs, ARRAY_LEN(f32_specs));
    load_qconv_stage(weight_root, stage_dir, qconv_specs, ARRAY_LEN(qconv_specs));
}

/*< 15. Load all upstream skip-handler weights for one stream >*/
void load_pvss_us_skip_weight_stage(struct pvss_us_skip_weight *weight,
                                    const char *weight_root,
                                    const char *stage_dir)
{
    const struct qconv_file_spec qconv_specs[] = {
        {"skip_conv_dec0", (size_t)C_DEC_0 * (2U * C_DEC_0), (size_t)C_DEC_0, (size_t)(2U * C_DEC_0), 1, &weight->skip_conv_dec0},
        {"skip_conv_dec1", (size_t)C_DEC_1 * (2U * C_DEC_1), (size_t)C_DEC_1, (size_t)(2U * C_DEC_1), 1, &weight->skip_conv_dec1},
        {"skip_conv_dec2", (size_t)C_DEC_2 * (2U * C_DEC_2), (size_t)C_DEC_2, (size_t)(2U * C_DEC_2), 1, &weight->skip_conv_dec2},
        {"skip_conv_out0", (size_t)C_OUT_0 * (2U * C_OUT_0), (size_t)C_OUT_0, (size_t)(2U * C_OUT_0), 1, &weight->skip_conv_out0},
    };

    load_qconv_stage(weight_root, stage_dir, qconv_specs, ARRAY_LEN(qconv_specs));
}
