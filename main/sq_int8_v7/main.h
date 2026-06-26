#ifndef _MAIN_
#define _MAIN_

/* ===================== INCLUDE SECTION ===================== */
/* <1. Built-in header > */
#include <time.h>
#include <errno.h>
#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* <2. User headers > */
#include "stft.h"
#include "model.h"
#include "model_memory.h"
#include "audio.h"
#include "datatypes.h"

/* ===================== DEFINE ============================== */
/*< 1. Paths >*/
/* v7 uses its own W8A8 weight set (data/weight_sq_int8_v7/, from weight_extractor_sq_int8_v7.py).
 * It is v6's weights PLUS a per-stage fixed-point SSM(x) requant pair (ssm_requant_mul[D] int32,
 * ssm_requant_shift[D] int32) approximating ssm_requant_scale ~= mul / 2^shift, so the SSM x-branch
 * requant runs integer-only too (v6 already did the MLP requant). All other scales/weights unchanged. */
#define WEIGHT_PATH                     "./data/weight_sq_int8_v7/"
#define DEGRADED_AUDIO_PATH             "./data/test_audio/downsample/"
#define GENERATED_AUDIO_PATH            "./data/test_audio/generated/sq_int8_v7/"

#define RESULT_PATH                     "./results/sq_int8_v7/"
#define PROCESSING_TIMING_CSV_PATH      "./results/sq_int8_v7/processing_time.csv"
#define PROCESSING_TIMING_TXT_PATH      "./results/sq_int8_v7/processing_time.txt"
#define STAGE_TIMING_CSV_PATH           "./results/sq_int8_v7/stage_timing.csv"
#define DETAIL_STAGE_TIMING_CSV_PATH    "./results/sq_int8_v7/detail_stage_timing.csv"
#define MEM_PROFILE_CSV_PATH            "./results/sq_int8_v7/memory_profile.csv"

/*< 2. Constant >*/
#define MAIN_PATH_BUF_SIZE      1024U
#define PROCESSING_TIME_SAMPLES (2937U * 5U)

/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/* ---------------------------------------------- */
/*< 1. Checking wav extension of a file >*/
static int has_wav_extension(const char *name);

/*< 2. Build the file path base on file name >*/
static void build_path(char *dst, size_t dst_size, const char *dir_path, const char *file_name);

/*< 3. Check whether the folder exist >*/
static void ensure_generated_dir_exists(void);

/*< 4. Ensure the result folder exists >*/
static void ensure_result_dir_exists(void);

/*< 5. Calculate elapsed time >*/
static double elapsed_ms(const struct timespec *start, const struct timespec *end);

/*< 6. Write timing summary >*/
static void write_timing_summary(const double *timings, uint32_t count);

/* ===================== MAIN FUNCTION SECTION =============== */
/* ---------------------------------------------- */
int main(int argc, char *argv[]);

#endif /* __MAIN__ */
