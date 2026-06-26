/* Encoder-0 latency micro-benchmark — original float32 version (naives_mp).
 *
 * Same harness as bench_int8.c but for the f32 reference block
 * pvss_ds_timing_mp_f32 (no quant/acc int8 scratch buffers, f32 weight struct).
 * See bench_int8.c / README.md for the methodology.
 *
 * argv: <mp_parallel_factor> <warmup_iters> <timed_iters> <weight_root_dir>
 * stdout (one CSV row): version,mp,iters,mean_ssm_us,mean_mlp_us,mean_tot_us,min_ssm_us,min_mlp_us,min_tot_us
 */
#include "model.h"          /* enc0 shape macros */
#include "macro_kernels.h"  /* pvss_ds_timing_mp_f32 prototype + struct pvss_ds_weight (f32) */
#include "bench_common.h"

#ifndef VERSION_TAG
#define VERSION_TAG "f32_naive_mp"
#endif

void load_pvss_ds_weight_stage(struct pvss_ds_weight *weight, const char *weight_root,
        const char *stage_dir, uint32_t c, uint32_t r, uint32_t n, uint32_t r2n,
        uint32_t dw_kernel_size, uint32_t mlp_hidden_dim, uint32_t sub_d_inner,
        uint32_t sub_d_projection, uint32_t sub_c);

int main(int argc, char **argv)
{
    const int   mp_req = argc > 1 ? atoi(argv[1]) : 1;
    const int   warm   = argc > 2 ? atoi(argv[2]) : 5;
    const int   iters  = argc > 3 ? atoi(argv[3]) : 50;
    const char *wroot  = argc > 4 ? argv[4]       : "./data/weight_f32";

    const size_t MAXT = 16;
    int mp = mp_req < 1 ? 1 : (mp_req > (int)MAXT ? (int)MAXT : mp_req);
    omp_set_max_active_levels(2);

    const size_t L       = (size_t)L_ENC_0;
    const size_t slice   = (size_t)ACT_BUF_SLICE_ENC_0;
    const size_t inout_n = (size_t)C_ENC_0 * L;

    float *inout  = malloc(inout_n * sizeof(float));
    float *inout0 = malloc(inout_n * sizeof(float));
    float *resid  = malloc(inout_n * sizeof(float));
    float *act    = calloc(MAXT * slice, sizeof(float));
    float *hid    = calloc(MAXT * (size_t)HID_BUF_SLICE_ENC_0, sizeof(float));
    if (!inout || !inout0 || !resid || !act || !hid) { fprintf(stderr, "alloc failed\n"); return 1; }

    struct pvss_ds_weight w; memset(&w, 0, sizeof(w));
    load_pvss_ds_weight_stage(&w, wroot, "pvss_ds_weight_enc0_mag",
        C_ENC_0, R_ENC_0, N_ENC_0, R2N_ENC_0, DW_KERNEL_SZ_ENC_0, MLP_HIDDEN_ENC_0,
        SUB_D_INNER_ENC_0, SUB_D_PROJECTION_ENC_0, SUB_C_ENC_0);

    fill_rand(inout0, inout_n, 0xC0FFEEULL);

    struct timespec t0, tp, tm;
    double sum_s = 0, sum_m = 0, sum_t = 0, min_s = 1e30, min_m = 1e30, min_t = 1e30;
    for (int it = -warm; it < iters; ++it) {
        memcpy(inout, inout0, inout_n * sizeof(float));
        memset(hid, 0, MAXT * (size_t)HID_BUF_SLICE_ENC_0 * sizeof(float));
        clock_gettime(CLOCK_MONOTONIC, &t0);
        pvss_ds_timing_mp_f32(inout, resid, act, hid, &w,
            C_ENC_0, H_ENC_0, W_ENC_0, L_ENC_0, R_ENC_0, N_ENC_0, R2N_ENC_0,
            DW_KERNEL_SZ_ENC_0, MLP_HIDDEN_ENC_0, SUB_D_INNER_ENC_0, SUB_D_PROJECTION_ENC_0,
            SUB_C_ENC_0, REDUCTION_FACTOR_ENC_0, (uint16_t)CHUNK_SZ_ENC_0,
            (uint32_t)mp, (size_t)ACT_BUF_SLICE_ENC_0, (size_t)HID_BUF_SLICE_ENC_0, &tp, &tm);
        if (it < 0) continue;
        const double s = ts_ns(&t0, &tp) / 1e3;
        const double m = ts_ns(&tp, &tm) / 1e3;
        const double t = ts_ns(&t0, &tm) / 1e3;
        sum_s += s; sum_m += m; sum_t += t;
        if (s < min_s) min_s = s;
        if (m < min_m) min_m = m;
        if (t < min_t) min_t = t;
    }

    printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", VERSION_TAG, mp, iters,
           sum_s / iters, sum_m / iters, sum_t / iters, min_s, min_m, min_t);
    free(inout); free(inout0); free(resid); free(act); free(hid);
    return 0;
}
