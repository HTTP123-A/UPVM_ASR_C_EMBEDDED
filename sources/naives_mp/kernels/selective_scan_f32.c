/* Split from sources/naives_mp/micro_kernels.c */

#include "micro_kernels.h"
/* <9. SSM core - Pytorch equivalent: SelectiveScanCore (can be Pytroch/CUDA) > */
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
                        uint16_t chunk_sz)
{
    const uint32_t KD = K * D;
    uint32_t step = (chunk_sz == 0U) ? L : (uint32_t)chunk_sz;
    if (step > L) step = L;

    const size_t NL = (size_t)N * (size_t)L;

    memset(hidden_state_buf, 0, (size_t)KD * (size_t)N * sizeof(float));

    for (uint32_t g = 0; g < K; ++g) {
        const float *B_g = Bs + (size_t)g * NL;
        const float *C_g = Cs + (size_t)g * NL;

        for (uint32_t d = 0; d < D; ++d) {
            const uint32_t gd = g * D + d;

            float *xy_row = xy + (size_t)gd * L;
            const float *dts_row = dts + (size_t)gd * L;
            const float *A_row = As + (size_t)gd * N;
            float *s = hidden_state_buf + (size_t)gd * N;

            const float D_gd = Ds[gd];
            const float db_gd = delta_bias[gd];

            for (uint32_t t0 = 0; t0 < L; t0 += step) {
                const uint32_t T = (t0 + step <= L) ? step : (L - t0);

                for (uint32_t l = 0; l < T; ++l) {
                    const uint32_t t = t0 + l;

                    const float raw_dt = dts_row[t] + db_gd;
                    const float delta = (raw_dt > 20.0f) ? raw_dt : log1pf(expf(raw_dt));
                    const float u_val = xy_row[t];

                    float y_val = 0.0f;
                    const float *b_ptr = B_g + t;
                    const float *c_ptr = C_g + t;

                    for (uint32_t n = 0; n < N; ++n) {
                        const float a = expf(A_row[n] * delta);
                        s[n] = s[n] * a + (delta * u_val * (*b_ptr));
                        y_val += (*c_ptr) * s[n];

                        b_ptr += L;
                        c_ptr += L;
                    }

                    xy_row[t] = y_val + D_gd * u_val;
                }
            }
        }
    }
}
