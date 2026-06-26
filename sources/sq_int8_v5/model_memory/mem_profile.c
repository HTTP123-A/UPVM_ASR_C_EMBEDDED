/* CSV memory-profile logger for sq_int8_v0 (see headers/sq_int8_v0/model_memory.h).
 * One row per micro-op: input/output activation (elems, dtype, bytes), weight bytes,
 * the op's working-set bytes, and the engine it runs on. Trailing SUMMARY rows give
 * the peak single-op working set and per-engine sums. Single-threaded (the profiling
 * forward runs sequentially), so plain buffered writes are race-free. */

#include "model_memory.h"

static FILE    *g_fp        = NULL;
static uint32_t g_idx       = 0U;
static uint64_t g_peak_op   = 0U;   /* max single-op working set (bytes) */
static uint64_t g_sum_cpu   = 0U;   /* sum of op bytes on CPU-F32 */
static uint64_t g_sum_accel = 0U;   /* sum of op bytes on INT8-ACCEL */
static uint64_t g_sum_bound = 0U;   /* sum of op bytes on the quant/dequant boundary */

/*< dtype -> byte size >*/
static uint32_t dtype_size(const char *dt)
{
    if (dt == NULL) {
        return 0U;
    }
    if (strcmp(dt, "int8") == 0) {
        return 1U;
    }
    if (strcmp(dt, "f32") == 0 || strcmp(dt, "int32") == 0) {
        return 4U;
    }
    return 0U; /* "-" / unknown */
}

void mem_profile_begin(const char *csv_path)
{
    g_fp = fopen(csv_path, "w");
    if (g_fp == NULL) {
        fprintf(stderr, "mem_profile: cannot open %s\n", csv_path);
        exit(EXIT_FAILURE);
    }
    g_idx = 0U;
    g_peak_op = 0U;
    g_sum_cpu = 0U;
    g_sum_accel = 0U;
    g_sum_bound = 0U;

    fprintf(g_fp,
        "idx,stage,op,engine,repeats,"
        "in_dtype,in_elems,in_bytes,"
        "weight_bytes,"
        "out_dtype,out_elems,out_bytes,"
        "op_bytes,op_MB\n");
}

void mem_profile_log(const char *stage, const char *op, const char *engine,
                     uint32_t repeats,
                     const char *in_dtype, uint64_t in_elems,
                     uint64_t weight_bytes,
                     const char *out_dtype, uint64_t out_elems)
{
    const uint64_t in_b  = in_elems  * (uint64_t)dtype_size(in_dtype);
    const uint64_t out_b = out_elems * (uint64_t)dtype_size(out_dtype);
    const uint64_t op_b  = in_b + weight_bytes + out_b;
    const double   op_mb = (double)op_b / (1024.0 * 1024.0);

    if (g_fp == NULL) {
        return;
    }

    fprintf(g_fp,
        "%u,%s,%s,%s,%u,"
        "%s,%llu,%llu,"
        "%llu,"
        "%s,%llu,%llu,"
        "%llu,%.4f\n",
        g_idx, stage, op, engine, repeats,
        in_dtype, (unsigned long long)in_elems, (unsigned long long)in_b,
        (unsigned long long)weight_bytes,
        out_dtype, (unsigned long long)out_elems, (unsigned long long)out_b,
        (unsigned long long)op_b, op_mb);

    if (op_b > g_peak_op) {
        g_peak_op = op_b;
    }
    if (strncmp(engine, "INT8", 4) == 0) {
        g_sum_accel += op_b;
    } else if (strncmp(engine, "CPU-F32", 7) == 0) {
        g_sum_cpu += op_b;
    } else { /* "CPU->ACCEL" / "ACCEL->CPU" quant-dequant boundary */
        g_sum_bound += op_b;
    }
    g_idx++;
}

/*< 14-column SUMMARY row: idx,SUMMARY,label,engine,(8 empty),op_bytes,op_MB >*/
static void summary_row(const char *label, const char *engine, uint64_t bytes)
{
    fprintf(g_fp, "%u,SUMMARY,%s,%s,,,,,,,,,%llu,%.4f\n",
            g_idx++, label, engine,
            (unsigned long long)bytes, (double)bytes / 1048576.0);
}

void mem_profile_end(void)
{
    if (g_fp == NULL) {
        return;
    }

    /* Summary rows (op_bytes column carries the value). */
    summary_row("peak_single_op_working_set", "", g_peak_op);
    summary_row("sum_op_bytes_CPU_F32", "CPU-F32", g_sum_cpu);
    summary_row("sum_op_bytes_INT8_ACCEL", "INT8-ACCEL", g_sum_accel);
    summary_row("sum_op_bytes_quant_dequant_boundary", "BOUNDARY", g_sum_bound);

    printf("[mem_profile] rows=%u  peak_single_op=%.2f MB  "
           "(CPU-F32 sum=%.1f MB, INT8-ACCEL sum=%.1f MB, boundary sum=%.1f MB)\n",
           g_idx, (double)g_peak_op / 1048576.0,
           (double)g_sum_cpu / 1048576.0, (double)g_sum_accel / 1048576.0,
           (double)g_sum_bound / 1048576.0);

    fclose(g_fp);
    g_fp = NULL;
}
