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
/* v5 uses its own W8A8 weight set (data/weight_sq_int8_v5/, from weight_extractor_sq_int8_v5.py).
 * It is v4's weights PLUS a per-stage ssm_requant_scale = ssm_in_projection.deq_scale[:SUB_D_INNER]
 * / ssm_dw_conv2d.act_scale for the fused SSM in_proj->dw_conv2d dequant+quant (REQUANT) on the
 * x branch. All other scales/weights (incl. v4's mlp_requant_scale) are unchanged. */
#define WEIGHT_PATH                     "./data/weight_sq_int8_v5/"
#define DEGRADED_AUDIO_PATH             "./data/test_audio/downsample/"
#define GENERATED_AUDIO_PATH            "./data/test_audio/generated/sq_int8_v5/"

#define RESULT_PATH                     "./results/sq_int8_v5/"
#define PROCESSING_TIMING_CSV_PATH      "./results/sq_int8_v5/processing_time.csv"
#define PROCESSING_TIMING_TXT_PATH      "./results/sq_int8_v5/processing_time.txt"
#define STAGE_TIMING_CSV_PATH           "./results/sq_int8_v5/stage_timing.csv"
#define DETAIL_STAGE_TIMING_CSV_PATH    "./results/sq_int8_v5/detail_stage_timing.csv"
#define MEM_PROFILE_CSV_PATH            "./results/sq_int8_v5/memory_profile.csv"

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
