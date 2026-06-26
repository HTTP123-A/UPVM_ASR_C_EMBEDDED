/* Encoder-0 latency micro-benchmark — INT8 versions (sq_int8_v0 / sq_int8_v5).
 *
 * Built once per version with that version's -I include dir, so it picks up that
 * version's enc0 macros, struct layout and timing block (the function names are
 * identical across versions, which is exactly why each version must be its own
 * executable — see Makefile / README.md).
 *
 * Drives ONLY the first encoder block pvss_ds (encoder 0):
 *   input  [C_ENC_0, H_ENC_0, W_ENC_0] = [16, 128, 128]
 *   output [2*C_ENC_0, H/2, W/2]        = [32,  64,  64]
 *
 * It calls the version's *_timing_* variant, which timestamps the SSM->MLP
 * boundary (pvss_time) and the block end (mlp_time), so we split SSM vs MLP vs
 * total. Real enc0 weights are loaded (well-conditioned values, no denormals);
 * the input is a fixed random buffer reset every iteration so each iteration
 * performs identical work. mp_parallel_factor (the SSM group-parallelism, the
 * ONLY threaded part — the MLP runs sequentially) is a CLI arg so all three
 * versions can be compared at equal thread counts.
 *
 * argv: <mp_parallel_factor> <warmup_iters> <timed_iters> <weight_root_dir>
 * stdout (one CSV row): version,mp,iters,mean_ssm_us,mean_mlp_us,mean_tot_us,min_ssm_us,min_mlp_us,min_tot_us
 */
#include "model.h"          /* enc0 shape macros (pulls fftw3.h/stft.h headers only; never linked/called) */
#include "macro_kernels.h"  /* pvss_ds_timing_mp_int8 prototype + struct pvss_ds_weight */
#include "bench_common.h"

#ifndef VERSION_TAG
#define VERSION_TAG "sq_int8"
#endif

/* real loader from sources/<ver>/common/weight_loader_f32_i8.c */
void load_pvss_ds_weight_stage(struct pvss_ds_weight *weight, const char *weight_root,
        const char *stage_dir, uint32_t c, uint32_t r, uint32_t n, uint32_t r2n,
        uint32_t dw_kernel_size, uint32_t mlp_hidden_dim, uint32_t sub_d_inner,
        uint32_t sub_d_projection, uint32_t sub_c);

int main(int argc, char **argv)
{
    const int   mp_req = argc > 1 ? atoi(argv[1]) : 1;
    const int   warm   = argc > 2 ? atoi(argv[2]) : 5;
    const int   iters  = argc > 3 ? atoi(argv[3]) : 50;
    const char *wroot  = argc > 4 ? argv[4]       : "./data/weight_sq_int8_v0";

    const size_t MAXT = 16;                       /* covers every version's native INNER_MP_PARALLEL */
    int mp = mp_req < 1 ? 1 : (mp_req > (int)MAXT ? (int)MAXT : mp_req);
    omp_set_max_active_levels(2);

    const size_t L       = (size_t)L_ENC_0;
    const size_t slice   = (size_t)ACT_BUF_SLICE_ENC_0;
    const size_t inout_n = (size_t)C_ENC_0 * L;

    float   *inout  = malloc(inout_n * sizeof(float));
    float   *inout0 = malloc(inout_n * sizeof(float));
    float   *resid  = malloc(inout_n * sizeof(float));
    float   *act    = calloc(MAXT * slice, sizeof(float));
    float   *hid    = calloc(MAXT * (size_t)HID_BUF_SLICE_ENC_0, sizeof(float));
    int8_t  *qbuf   = calloc(MAXT * slice, sizeof(int8_t));
    int32_t *accb   = calloc(MAXT * slice, sizeof(int32_t));
    if (!inout || !inout0 || !resid || !act || !hid || !qbuf || !accb) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

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
        pvss_ds_timing_mp_int8(inout, resid, act, hid, qbuf, accb, &w,
            C_ENC_0, H_ENC_0, W_ENC_0, L_ENC_0, R_ENC_0, N_ENC_0, R2N_ENC_0,
            DW_KERNEL_SZ_ENC_0, MLP_HIDDEN_ENC_0, SUB_D_INNER_ENC_0, SUB_D_PROJECTION_ENC_0,
            SUB_C_ENC_0, REDUCTION_FACTOR_ENC_0, (uint16_t)CHUNK_SZ_ENC_0,
            (uint32_t)mp, (size_t)ACT_BUF_SLICE_ENC_0, (size_t)HID_BUF_SLICE_ENC_0, &tp, &tm);
        if (it < 0) continue;                     /* warmup */
        const double s = ts_ns(&t0, &tp) / 1e3;   /* SSM   (us) */
        const double m = ts_ns(&tp, &tm) / 1e3;   /* MLP   (us) */
        const double t = ts_ns(&t0, &tm) / 1e3;   /* total (us) */
        sum_s += s; sum_m += m; sum_t += t;
        if (s < min_s) min_s = s;
        if (m < min_m) min_m = m;
        if (t < min_t) min_t = t;
    }

    printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", VERSION_TAG, mp, iters,
           sum_s / iters, sum_m / iters, sum_t / iters, min_s, min_m, min_t);
    free(inout); free(inout0); free(resid); free(act); free(hid); free(qbuf); free(accb);
    return 0;
}
