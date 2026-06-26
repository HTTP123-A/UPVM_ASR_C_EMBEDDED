#ifndef _MODEL_MEMORY_H_
#define _MODEL_MEMORY_H_

/* Memory-profiling variant of the sq_int8_v0 forward.
 * It runs the SAME blocks/kernels as model/ (single magnitude stream, sequential)
 * and, for every micro-op, logs the input/output activation footprint (element
 * count, dtype, bytes) and the weight footprint to a CSV, tagging each op with the
 * engine it belongs to in the heterogeneous scheme (CPU-F32 / INT8-ACCEL / the
 * quant-dequant boundary). The profile is data-independent (a pure function of the
 * static shapes), so one pass is the whole report. */

#include "model.h"

/* ===================== MEMORY PROFILER LOGGER ===================== */
/*< Open the CSV and write the header (call once at the start). >*/
void mem_profile_begin(const char *csv_path);

/*< Log one micro-op. in/out bytes are computed from the dtype string
 *  ("f32"/"int32" = 4 B, "int8" = 1 B, "-" = 0). weight_bytes is passed in
 *  pre-computed (mixed dtypes). repeats = how many times the op runs in the
 *  block's group loop (per-invocation footprint is reported, not ×repeats). >*/
void mem_profile_log(const char *stage, const char *op, const char *engine,
                     uint32_t repeats,
                     const char *in_dtype, uint64_t in_elems,
                     uint64_t weight_bytes,
                     const char *out_dtype, uint64_t out_elems);

/*< Write the SUMMARY rows (peak single-op working set, per-engine sums) + close. >*/
void mem_profile_end(void);

/* ===================== PROFILING FORWARD ===================== */
/*< Single-pass, single-stream (magnitude), sequential forward that runs the real
 *  model/ blocks and writes the per-op memory report to csv_path. The phase
 *  buffers are passed only so the per-stage interaction adds can execute (they are
 *  zero — output is intentionally not used; only the memory report matters). >*/
void dual_stream_unet_mem_int8(float* audio, // [L_AUDIO]
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
                        const char *csv_path);

#endif /* _MODEL_MEMORY_H_ */
