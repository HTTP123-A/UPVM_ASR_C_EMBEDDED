/* Author: Nguyen Quang Phuong
 * Email: ngquangph@gmail.com
 *
 * Notes:
 * Dual-stream parallel VMamba ASR (C version) — mixed precision (f32 + int8),
 * SmoothQuant W8A8 checkpoint (gen_ptq_int8pack.pth). Supports MODE = NORMAL /
 * STAGE_TIMING / DETAIL_STAGE_TIMING (same timing layout as the f32 builds).
 *
 */

/* ===================== INCLUDE SECTION ===================== */
/*< 1. Built-in header >*/

/*< 2. User headers >*/
#include "main.h"

/* ===================== GLOBAL VARIABLES ==================== */
/*< 1. Buffer pointers >*/
    /* Audio buffer pointers */
float *audio;
float *per_file_audio;
    /* Weight buffer pointers */
struct upvm_asr_weight *model_weight;
    /* Activation buffer */
float *model_inout_buf_mag, *model_inout_buf_pha;
float *model_internal_res_buf_mag, *model_internal_res_buf_pha;
float *model_global_res_buf_mag, *model_global_res_buf_pha;
float *model_act_buf_mag, *model_act_buf_pha;
float *model_low_freq_buf_mag, *model_low_freq_buf_pha;
float *hidden_state_buf_mag, *hidden_state_buf_pha;
    /* Quant (int8) + accumulator (int32) scratch, per stream */
int8_t *quant_buf_mag, *quant_buf_pha;
int32_t *acc_buf_mag, *acc_buf_pha;
    /* STFT buffer pointers & structs */
        /* FFT structures */
fftwf_plan p_stft;
fftwf_plan p_istft;
fftwf_complex *spec;

        /* Buffers */
float *win;
float *ola;
float *wss;
float *frame;

/*< 2. Global integer >*/
    /* STFT */
uint32_t window_offset;
    /* Time monitoring */
struct timespec model_start_time;
struct timespec model_stop_time;
double processing_time_ms;
double processing_time_collect[PROCESSING_TIME_SAMPLES];
    /* Stage timing record*/
#ifdef _MODEL_STAGE_TIMING_
struct timing_record_points stage_timing_record;

double stft_time;
double low_freq_mag_time;
double low_freq_pha_time;
double spectrogram_res_mag_time;

double patch_embed_mag_time;
double patch_embed_pha_time;
double patch_embed_res_mag_time;
double patch_embed_res_pha_time;
double patch_embed_interaction_mag_time;
double patch_embed_interaction_pha_time;

double encoder0_mag_time;
double encoder0_pha_time;
double encoder0_res_mag_time;
double encoder0_res_pha_time;
double encoder0_interaction_mag_time;
double encoder0_interaction_pha_time;

double encoder1_mag_time;
double encoder1_pha_time;
double encoder1_res_mag_time;
double encoder1_res_pha_time;
double encoder1_interaction_mag_time;
double encoder1_interaction_pha_time;

double encoder2_mag_time;
double encoder2_pha_time;
double encoder2_res_mag_time;
double encoder2_res_pha_time;
double encoder2_interaction_mag_time;
double encoder2_interaction_pha_time;

double latent_mag_time;
double latent_pha_time;
double latent_interaction_mag_time;
double latent_interaction_pha_time;

double decoder0_res_mag_time;
double decoder0_res_pha_time;
double decoder0_mag_time;
double decoder0_pha_time;
double decoder0_interaction_mag_time;
double decoder0_interaction_pha_time;

double decoder1_res_mag_time;
double decoder1_res_pha_time;
double decoder1_mag_time;
double decoder1_pha_time;
double decoder1_interaction_mag_time;
double decoder1_interaction_pha_time;

double decoder2_res_mag_time;
double decoder2_res_pha_time;
double decoder2_mag_time;
double decoder2_pha_time;
double decoder2_interaction_mag_time;
double decoder2_interaction_pha_time;

double output0_res_mag_time;
double output0_res_pha_time;
double output0_mag_time;
double output0_pha_time;

double output1_mag_time;
double output1_pha_time;

double output3_mag_time;
double output3_pha_time;
double output3_res_mag_time;

double final_low_freq_mag_time;
double final_low_freq_pha_time;

double istft_time;

double total_time;
#endif

#ifdef _MODEL_DETAIL_STAGE_TIMING_
struct detail_timing_record_points detail_timing_record;

double stft_time;
double low_freq_mag_time;
double low_freq_pha_time;
double spectrogram_res_mag_time;

double patch_embed_mag_time;
double patch_embed_pha_time;
double patch_embed_res_mag_time;
double patch_embed_res_pha_time;
double patch_embed_interaction_mag_time;
double patch_embed_interaction_pha_time;

double encoder0_pvss_mag_time;
double encoder0_mlp_mag_time;
double encoder0_mag_time;
double encoder0_pvss_pha_time;
double encoder0_mlp_pha_time;
double encoder0_pha_time;
double encoder0_res_mag_time;
double encoder0_res_pha_time;
double encoder0_interaction_mag_time;
double encoder0_interaction_pha_time;

double encoder1_pvss_mag_time;
double encoder1_mlp_mag_time;
double encoder1_mag_time;
double encoder1_pvss_pha_time;
double encoder1_mlp_pha_time;
double encoder1_pha_time;
double encoder1_res_mag_time;
double encoder1_res_pha_time;
double encoder1_interaction_mag_time;
double encoder1_interaction_pha_time;

double encoder2_pvss_mag_time;
double encoder2_mlp_mag_time;
double encoder2_mag_time;
double encoder2_pvss_pha_time;
double encoder2_mlp_pha_time;
double encoder2_pha_time;
double encoder2_res_mag_time;
double encoder2_res_pha_time;
double encoder2_interaction_mag_time;
double encoder2_interaction_pha_time;

double latent_pvss_mag_time;
double latent_mlp_mag_time;
double latent_mag_time;
double latent_pvss_pha_time;
double latent_mlp_pha_time;
double latent_pha_time;
double latent_interaction_mag_time;
double latent_interaction_pha_time;

double decoder0_pvss_mag_time;
double decoder0_mlp_mag_time;
double decoder0_res_mag_time;
double decoder0_pvss_pha_time;
double decoder0_mlp_pha_time;
double decoder0_res_pha_time;
double decoder0_mag_time;
double decoder0_pha_time;
double decoder0_interaction_mag_time;
double decoder0_interaction_pha_time;

double decoder1_pvss_mag_time;
double decoder1_mlp_mag_time;
double decoder1_res_mag_time;
double decoder1_pvss_pha_time;
double decoder1_mlp_pha_time;
double decoder1_res_pha_time;
double decoder1_mag_time;
double decoder1_pha_time;
double decoder1_interaction_mag_time;
double decoder1_interaction_pha_time;

double decoder2_pvss_mag_time;
double decoder2_mlp_mag_time;
double decoder2_res_mag_time;
double decoder2_pvss_pha_time;
double decoder2_mlp_pha_time;
double decoder2_res_pha_time;
double decoder2_mag_time;
double decoder2_pha_time;
double decoder2_interaction_mag_time;
double decoder2_interaction_pha_time;

double output0_pvss_mag_time;
double output0_mlp_mag_time;
double output0_res_mag_time;
double output0_pvss_pha_time;
double output0_mlp_pha_time;
double output0_res_pha_time;
double output0_mag_time;
double output0_pha_time;

double output1_pvss_mag_time;
double output1_mlp_mag_time;
double output1_mag_time;
double output1_pvss_pha_time;
double output1_mlp_pha_time;
double output1_pha_time;

double output3_pvss_mag_time;
double output3_mlp_mag_time;
double output3_mag_time;
double output3_pvss_pha_time;
double output3_mlp_pha_time;
double output3_pha_time;
double output3_res_mag_time;

double final_low_freq_mag_time;
double final_low_freq_pha_time;

double istft_time;

double total_time;
#endif

/* ===================== MAIN FUNCTION SECTION =============== */
/* ---------------------------------------------- */
int main(int argc, char *argv[])
{
    DIR *dir_ptr;
    struct dirent *entry;
    struct audio_io_meta meta;

    uint32_t timing_idx;
    uint32_t seg_idx;

    char input_path[MAIN_PATH_BUF_SIZE];
    char output_path[MAIN_PATH_BUF_SIZE];

    FILE *csv_fp;
#ifdef _MODEL_STAGE_TIMING_
    FILE *stage_timing_csv_fp;
#endif
#ifdef _MODEL_DETAIL_STAGE_TIMING_
    FILE *detail_stage_timing_csv_fp;
#endif

    (void)argc;
    (void)argv;

    /*< 1. Allocate memory for buffers (f32 work + int8/int32 quant scratch) >*/
    model_buf_init_f32_i8(
        &model_weight,
        (char *)WEIGHT_PATH,
        &model_inout_buf_mag, &model_inout_buf_pha,
        &model_internal_res_buf_mag, &model_internal_res_buf_pha,
        &model_global_res_buf_mag, &model_global_res_buf_pha,
        &model_act_buf_mag, &model_act_buf_pha,
        &model_low_freq_buf_mag, &model_low_freq_buf_pha,
        &hidden_state_buf_mag, &hidden_state_buf_pha,
        &quant_buf_mag, &quant_buf_pha,
        &acc_buf_mag, &acc_buf_pha
    );

    /*< 2. Allocate memory for STFT buffers >*/
    stft_init_f32(
        &win, &frame, &spec,
        &ola, &wss, &p_stft, &p_istft,
        &window_offset, N_FFT, WIN_LENGTH, OLA_LEN, FRAME_SIZE
    );

    /*< 3. Ensure output folders exist >*/
    ensure_generated_dir_exists();
    ensure_result_dir_exists();

    /*< 4. Open timing CSV >*/
    csv_fp = fopen(PROCESSING_TIMING_CSV_PATH, "w");
    if (csv_fp == NULL) {
        fprintf(stderr, "failed to open csv file: %s\n", PROCESSING_TIMING_CSV_PATH);
        return EXIT_FAILURE;
    }

    fprintf(csv_fp, "timing_index,file_name,segment_index,processing_time_ms\n");

#ifdef _MODEL_STAGE_TIMING_
    stage_timing_csv_fp = fopen(STAGE_TIMING_CSV_PATH, "w");
    if (stage_timing_csv_fp == NULL) {
        fprintf(stderr, "failed to open stage timing csv file: %s\n", STAGE_TIMING_CSV_PATH);
        return EXIT_FAILURE;
    }

    fprintf(stage_timing_csv_fp, "timing_index,file_name,segment_index,"
        "stft_time,low_freq_mag_time,low_freq_pha_time,spectrogram_res_mag_time,"
        "patch_embed_mag_time,patch_embed_pha_time,patch_embed_res_mag_time,"
        "patch_embed_res_pha_time,patch_embed_interaction_mag_time,patch_embed_interaction_pha_time,"
        "encoder0_mag_time,encoder0_pha_time,encoder0_res_mag_time,encoder0_res_pha_time,"
        "encoder0_interaction_mag_time,encoder0_interaction_pha_time,"
        "encoder1_mag_time,encoder1_pha_time,encoder1_res_mag_time,encoder1_res_pha_time,encoder1_interaction_mag_time,"
        "encoder1_interaction_pha_time,encoder2_mag_time,encoder2_pha_time,encoder2_res_mag_time,encoder2_res_pha_time,"
        "encoder2_interaction_mag_time,encoder2_interaction_pha_time,"
        "latent_mag_time,latent_pha_time,latent_interaction_mag_time,latent_interaction_pha_time,"
        "decoder0_res_mag_time,decoder0_res_pha_time,decoder0_mag_time,decoder0_pha_time,decoder0_interaction_mag_time,decoder0_interaction_pha_time,"
        "decoder1_res_mag_time,decoder1_res_pha_time,decoder1_mag_time,decoder1_pha_time,decoder1_interaction_mag_time,decoder1_interaction_pha_time,"
        "decoder2_res_mag_time,decoder2_res_pha_time,decoder2_mag_time,decoder2_pha_time,decoder2_interaction_mag_time,decoder2_interaction_pha_time,"
        "output0_res_mag_time,output0_res_pha_time,output0_mag_time,output0_pha_time,output1_mag_time,output1_pha_time,"
        "output3_mag_time,output3_pha_time,output3_res_mag_time,final_low_freq_mag_time,final_low_freq_pha_time,"
        "istft_time,total_time\n");
#endif

#ifdef _MODEL_DETAIL_STAGE_TIMING_
    detail_stage_timing_csv_fp = fopen(DETAIL_STAGE_TIMING_CSV_PATH, "w");
    if (detail_stage_timing_csv_fp == NULL) {
        fprintf(stderr, "failed to open detail stage timing csv file: %s\n", DETAIL_STAGE_TIMING_CSV_PATH);
        return EXIT_FAILURE;
    }

    fprintf(detail_stage_timing_csv_fp,
        "timing_index,file_name,segment_index,"
        "stft_time,low_freq_mag_time,low_freq_pha_time,spectrogram_res_mag_time,"
        "patch_embed_mag_time,patch_embed_pha_time,patch_embed_res_mag_time,"
        "patch_embed_res_pha_time,patch_embed_interaction_mag_time,patch_embed_interaction_pha_time,"
        "encoder0_pvss_mag_time,encoder0_mlp_mag_time,encoder0_mag_time,"
        "encoder0_pvss_pha_time,encoder0_mlp_pha_time,encoder0_pha_time,"
        "encoder0_res_mag_time,encoder0_res_pha_time,encoder0_interaction_mag_time,encoder0_interaction_pha_time,"
        "encoder1_pvss_mag_time,encoder1_mlp_mag_time,encoder1_mag_time,"
        "encoder1_pvss_pha_time,encoder1_mlp_pha_time,encoder1_pha_time,"
        "encoder1_res_mag_time,encoder1_res_pha_time,encoder1_interaction_mag_time,encoder1_interaction_pha_time,"
        "encoder2_pvss_mag_time,encoder2_mlp_mag_time,encoder2_mag_time,"
        "encoder2_pvss_pha_time,encoder2_mlp_pha_time,encoder2_pha_time,"
        "encoder2_res_mag_time,encoder2_res_pha_time,encoder2_interaction_mag_time,encoder2_interaction_pha_time,"
        "latent_pvss_mag_time,latent_mlp_mag_time,latent_mag_time,"
        "latent_pvss_pha_time,latent_mlp_pha_time,latent_pha_time,"
        "latent_interaction_mag_time,latent_interaction_pha_time,"
        "decoder0_pvss_mag_time,decoder0_mlp_mag_time,decoder0_res_mag_time,"
        "decoder0_pvss_pha_time,decoder0_mlp_pha_time,decoder0_res_pha_time,"
        "decoder0_mag_time,decoder0_pha_time,decoder0_interaction_mag_time,decoder0_interaction_pha_time,"
        "decoder1_pvss_mag_time,decoder1_mlp_mag_time,decoder1_res_mag_time,"
        "decoder1_pvss_pha_time,decoder1_mlp_pha_time,decoder1_res_pha_time,"
        "decoder1_mag_time,decoder1_pha_time,decoder1_interaction_mag_time,decoder1_interaction_pha_time,"
        "decoder2_pvss_mag_time,decoder2_mlp_mag_time,decoder2_res_mag_time,"
        "decoder2_pvss_pha_time,decoder2_mlp_pha_time,decoder2_res_pha_time,"
        "decoder2_mag_time,decoder2_pha_time,decoder2_interaction_mag_time,decoder2_interaction_pha_time,"
        "output0_pvss_mag_time,output0_mlp_mag_time,output0_res_mag_time,"
        "output0_pvss_pha_time,output0_mlp_pha_time,output0_res_pha_time,"
        "output0_mag_time,output0_pha_time,"
        "output1_pvss_mag_time,output1_mlp_mag_time,output1_mag_time,"
        "output1_pvss_pha_time,output1_mlp_pha_time,output1_pha_time,"
        "output3_pvss_mag_time,output3_mlp_mag_time,output3_mag_time,"
        "output3_pvss_pha_time,output3_mlp_pha_time,output3_pha_time,"
        "output3_res_mag_time,final_low_freq_mag_time,final_low_freq_pha_time,istft_time,total_time\n");

#endif

    /*< 5. Open original audio folder >*/
    dir_ptr = opendir(DEGRADED_AUDIO_PATH);
    if (dir_ptr == NULL) {
        fclose(csv_fp);
        fprintf(stderr, "failed to open input dir: %s\n", DEGRADED_AUDIO_PATH);
        return EXIT_FAILURE;
    }

    /*< 6. openMP setup >*/
    omp_set_dynamic(0);
    omp_set_max_active_levels(2);

    /*< 7. Reset timing index >*/
    timing_idx = 0U;

    /*< 8. Main test loop >*/
    while ((entry = readdir(dir_ptr)) != NULL) {
        if (entry->d_name[0] == '.') {
            continue;
        }

        if (!has_wav_extension(entry->d_name)) {
            continue;
        }

        build_path(input_path, sizeof(input_path), DEGRADED_AUDIO_PATH, entry->d_name);
        build_path(output_path, sizeof(output_path), GENERATED_AUDIO_PATH, entry->d_name);

        memset(&meta, 0, sizeof(meta));
        per_file_audio = NULL;

        audio_read_test_wav_f32(input_path, &per_file_audio, &meta);

        for (seg_idx = 0U; seg_idx < meta.num_segments; ++seg_idx) {
            if (timing_idx >= PROCESSING_TIME_SAMPLES) {
                fprintf(stderr, "timing buffer overflow at %u samples\n", timing_idx);
                free(per_file_audio);
                fclose(csv_fp);
#ifdef _MODEL_STAGE_TIMING_
                fclose(stage_timing_csv_fp);
#endif
#ifdef _MODEL_DETAIL_STAGE_TIMING_
                fclose(detail_stage_timing_csv_fp);
#endif
                closedir(dir_ptr);
                return EXIT_FAILURE;
            }

            audio = per_file_audio + ((size_t)seg_idx * AUDIO_LENGTH);

            clock_gettime(CLOCK_MONOTONIC, &model_start_time);

#ifdef _MODEL_STAGE_TIMING_
            dual_stream_unet_timing_profiling_mp_int8(
                audio,
                model_weight,
                model_inout_buf_mag, model_inout_buf_pha,
                model_internal_res_buf_mag, model_internal_res_buf_pha,
                model_global_res_buf_mag, model_global_res_buf_pha,
                model_act_buf_mag, model_act_buf_pha,
                model_low_freq_buf_mag, model_low_freq_buf_pha,
                hidden_state_buf_mag, hidden_state_buf_pha,
                quant_buf_mag, quant_buf_pha,
                acc_buf_mag, acc_buf_pha,
                win, frame, ola, wss,
                spec, &p_stft, &p_istft, window_offset,
                &stage_timing_record);
#elif defined(_MODEL_DETAIL_STAGE_TIMING_)
            dual_stream_unet_detail_timing_profiling_mp_int8(
                audio,
                model_weight,
                model_inout_buf_mag, model_inout_buf_pha,
                model_internal_res_buf_mag, model_internal_res_buf_pha,
                model_global_res_buf_mag, model_global_res_buf_pha,
                model_act_buf_mag, model_act_buf_pha,
                model_low_freq_buf_mag, model_low_freq_buf_pha,
                hidden_state_buf_mag, hidden_state_buf_pha,
                quant_buf_mag, quant_buf_pha,
                acc_buf_mag, acc_buf_pha,
                win, frame, ola, wss,
                spec, &p_stft, &p_istft, window_offset,
                &detail_timing_record);
#elif defined(_MODEL_MEM_PROFILE_)
            dual_stream_unet_mem_int8(
                audio,
                model_weight,
                model_inout_buf_mag, model_inout_buf_pha,
                model_internal_res_buf_mag, model_internal_res_buf_pha,
                model_global_res_buf_mag, model_global_res_buf_pha,
                model_act_buf_mag, model_act_buf_pha,
                model_low_freq_buf_mag, model_low_freq_buf_pha,
                hidden_state_buf_mag, hidden_state_buf_pha,
                quant_buf_mag, quant_buf_pha,
                acc_buf_mag, acc_buf_pha,
                win, frame, ola, wss,
                spec, &p_stft, &p_istft, window_offset,
                MEM_PROFILE_CSV_PATH);
#else
            dual_stream_unet_mp_int8(
                audio,
                model_weight,
                model_inout_buf_mag, model_inout_buf_pha,
                model_internal_res_buf_mag, model_internal_res_buf_pha,
                model_global_res_buf_mag, model_global_res_buf_pha,
                model_act_buf_mag, model_act_buf_pha,
                model_low_freq_buf_mag, model_low_freq_buf_pha,
                hidden_state_buf_mag, hidden_state_buf_pha,
                quant_buf_mag, quant_buf_pha,
                acc_buf_mag, acc_buf_pha,
                win, frame, ola, wss,
                spec, &p_stft, &p_istft, window_offset);
#endif

            clock_gettime(CLOCK_MONOTONIC, &model_stop_time);

            processing_time_ms = elapsed_ms(&model_start_time, &model_stop_time);
            processing_time_collect[timing_idx] = processing_time_ms;

#ifdef _MODEL_STAGE_TIMING_
            stft_time = elapsed_ms(&stage_timing_record.start_time, &stage_timing_record.stft_time);
            /* --------------------- */
            low_freq_mag_time = elapsed_ms(&stage_timing_record.stft_time, &stage_timing_record.low_frequency_mag_time);
            spectrogram_res_mag_time = elapsed_ms(&stage_timing_record.low_frequency_mag_time, &stage_timing_record.spectrogram_res_mag_time);

            low_freq_pha_time = elapsed_ms(&stage_timing_record.stft_time, &stage_timing_record.low_frequency_pha_time);

            /* --------------------- */
            patch_embed_mag_time = elapsed_ms(&stage_timing_record.spectrogram_res_mag_time, &stage_timing_record.patch_embed_mag_time);
            patch_embed_res_mag_time = elapsed_ms(&stage_timing_record.patch_embed_mag_time, &stage_timing_record.patch_embed_res_mag_time);

            patch_embed_pha_time = elapsed_ms(&stage_timing_record.low_frequency_pha_time, &stage_timing_record.patch_embed_pha_time);
            patch_embed_res_pha_time = elapsed_ms(&stage_timing_record.patch_embed_pha_time, &stage_timing_record.patch_embed_res_pha_time);

            patch_embed_interaction_mag_time = elapsed_ms(&stage_timing_record.patch_embed_res_mag_time, &stage_timing_record.patch_embed_interaction_mag_time);
            patch_embed_interaction_pha_time = elapsed_ms(&stage_timing_record.patch_embed_interaction_mag_time, &stage_timing_record.patch_embed_interaction_pha_time);

            /* --------------------- */
            encoder0_mag_time = elapsed_ms(&stage_timing_record.patch_embed_interaction_pha_time, &stage_timing_record.encoder0_mag_time);
            encoder0_res_mag_time = elapsed_ms(&stage_timing_record.encoder0_mag_time, &stage_timing_record.encoder0_res_mag_time);

            encoder0_pha_time = elapsed_ms(&stage_timing_record.patch_embed_interaction_pha_time, &stage_timing_record.encoder0_pha_time);
            encoder0_res_pha_time = elapsed_ms(&stage_timing_record.encoder0_pha_time, &stage_timing_record.encoder0_res_pha_time);

            encoder0_interaction_mag_time = elapsed_ms(&stage_timing_record.encoder0_res_mag_time, &stage_timing_record.encoder0_interaction_mag_time);
            encoder0_interaction_pha_time = elapsed_ms(&stage_timing_record.encoder0_interaction_mag_time, &stage_timing_record.encoder0_interaction_pha_time);

            /* --------------------- */
            encoder1_mag_time = elapsed_ms(&stage_timing_record.encoder0_interaction_pha_time, &stage_timing_record.encoder1_mag_time);
            encoder1_res_mag_time = elapsed_ms(&stage_timing_record.encoder1_mag_time, &stage_timing_record.encoder1_res_mag_time);

            encoder1_pha_time = elapsed_ms(&stage_timing_record.encoder0_interaction_pha_time, &stage_timing_record.encoder1_pha_time);
            encoder1_res_pha_time = elapsed_ms(&stage_timing_record.encoder1_pha_time, &stage_timing_record.encoder1_res_pha_time);

            encoder1_interaction_mag_time = elapsed_ms(&stage_timing_record.encoder1_res_mag_time, &stage_timing_record.encoder1_interaction_mag_time);
            encoder1_interaction_pha_time = elapsed_ms(&stage_timing_record.encoder1_interaction_mag_time, &stage_timing_record.encoder1_interaction_pha_time);

            /* --------------------- */
            encoder2_mag_time = elapsed_ms(&stage_timing_record.encoder1_interaction_pha_time, &stage_timing_record.encoder2_mag_time);
            encoder2_res_mag_time = elapsed_ms(&stage_timing_record.encoder2_mag_time, &stage_timing_record.encoder2_res_mag_time);

            encoder2_pha_time = elapsed_ms(&stage_timing_record.encoder1_interaction_pha_time, &stage_timing_record.encoder2_pha_time);
            encoder2_res_pha_time = elapsed_ms(&stage_timing_record.encoder2_pha_time, &stage_timing_record.encoder2_res_pha_time);

            encoder2_interaction_mag_time = elapsed_ms(&stage_timing_record.encoder2_res_mag_time, &stage_timing_record.encoder2_interaction_mag_time);
            encoder2_interaction_pha_time = elapsed_ms(&stage_timing_record.encoder2_interaction_mag_time, &stage_timing_record.encoder2_interaction_pha_time);

            /* --------------------- */
            latent_mag_time = elapsed_ms(&stage_timing_record.encoder2_interaction_pha_time, &stage_timing_record.latent_mag_time);
            latent_interaction_mag_time = elapsed_ms(&stage_timing_record.latent_mag_time, &stage_timing_record.latent_interaction_mag_time);

            latent_pha_time = elapsed_ms(&stage_timing_record.encoder2_interaction_pha_time, &stage_timing_record.latent_pha_time);
            latent_interaction_pha_time = elapsed_ms(&stage_timing_record.latent_pha_time, &stage_timing_record.latent_interaction_pha_time);

            /* --------------------- */
            decoder0_res_mag_time = elapsed_ms(&stage_timing_record.latent_interaction_pha_time, &stage_timing_record.decoder0_res_mag_time);
            decoder0_mag_time = elapsed_ms(&stage_timing_record.decoder0_res_mag_time, &stage_timing_record.decoder0_mag_time);

            decoder0_res_pha_time = elapsed_ms(&stage_timing_record.latent_interaction_pha_time, &stage_timing_record.decoder0_res_pha_time);
            decoder0_pha_time = elapsed_ms(&stage_timing_record.decoder0_res_pha_time, &stage_timing_record.decoder0_pha_time);

            decoder0_interaction_mag_time = elapsed_ms(&stage_timing_record.decoder0_mag_time, &stage_timing_record.decoder0_interaction_mag_time);
            decoder0_interaction_pha_time = elapsed_ms(&stage_timing_record.decoder0_interaction_mag_time, &stage_timing_record.decoder0_interaction_pha_time);

            /* --------------------- */
            decoder1_res_mag_time = elapsed_ms(&stage_timing_record.decoder0_interaction_pha_time, &stage_timing_record.decoder1_res_mag_time);
            decoder1_mag_time = elapsed_ms(&stage_timing_record.decoder1_res_mag_time, &stage_timing_record.decoder1_mag_time);

            decoder1_res_pha_time = elapsed_ms(&stage_timing_record.decoder0_interaction_pha_time, &stage_timing_record.decoder1_res_pha_time);
            decoder1_pha_time = elapsed_ms(&stage_timing_record.decoder1_res_pha_time, &stage_timing_record.decoder1_pha_time);

            decoder1_interaction_mag_time = elapsed_ms(&stage_timing_record.decoder1_mag_time, &stage_timing_record.decoder1_interaction_mag_time);
            decoder1_interaction_pha_time = elapsed_ms(&stage_timing_record.decoder1_interaction_mag_time, &stage_timing_record.decoder1_interaction_pha_time);

            /* --------------------- */
            decoder2_res_mag_time = elapsed_ms(&stage_timing_record.decoder1_interaction_pha_time, &stage_timing_record.decoder2_res_mag_time);
            decoder2_mag_time = elapsed_ms(&stage_timing_record.decoder2_res_mag_time, &stage_timing_record.decoder2_mag_time);

            decoder2_res_pha_time = elapsed_ms(&stage_timing_record.decoder1_interaction_pha_time, &stage_timing_record.decoder2_res_pha_time);
            decoder2_pha_time = elapsed_ms(&stage_timing_record.decoder2_res_pha_time, &stage_timing_record.decoder2_pha_time);

            decoder2_interaction_mag_time = elapsed_ms(&stage_timing_record.decoder2_mag_time, &stage_timing_record.decoder2_interaction_mag_time);
            decoder2_interaction_pha_time = elapsed_ms(&stage_timing_record.decoder2_interaction_mag_time, &stage_timing_record.decoder2_interaction_pha_time);

            /* --------------------- */
            output0_res_mag_time = elapsed_ms(&stage_timing_record.decoder2_interaction_pha_time, &stage_timing_record.output0_res_mag_time);
            output0_mag_time = elapsed_ms(&stage_timing_record.output0_res_mag_time, &stage_timing_record.output0_mag_time);

            output0_res_pha_time = elapsed_ms(&stage_timing_record.decoder2_interaction_pha_time, &stage_timing_record.output0_res_pha_time);
            output0_pha_time = elapsed_ms(&stage_timing_record.output0_res_pha_time, &stage_timing_record.output0_pha_time);

            /* --------------------- */
            output1_mag_time = elapsed_ms(&stage_timing_record.output0_mag_time, &stage_timing_record.output1_mag_time);
            output1_pha_time = elapsed_ms(&stage_timing_record.output0_pha_time, &stage_timing_record.output1_pha_time);

            /* --------------------- */
            output3_mag_time = elapsed_ms(&stage_timing_record.output1_mag_time, &stage_timing_record.output3_mag_time);
            output3_res_mag_time = elapsed_ms(&stage_timing_record.output3_mag_time, &stage_timing_record.output3_res_mag_time);

            output3_pha_time = elapsed_ms(&stage_timing_record.output1_pha_time, &stage_timing_record.output3_pha_time);

            /* --------------------- */
            final_low_freq_mag_time = elapsed_ms(&stage_timing_record.output3_res_mag_time, &stage_timing_record.final_low_freq_mag_time);
            final_low_freq_pha_time = elapsed_ms(&stage_timing_record.output3_pha_time, &stage_timing_record.final_low_freq_pha_time);

            /* --------------------- */
            istft_time = elapsed_ms(&stage_timing_record.final_low_freq_mag_time, &stage_timing_record.istft_time);

            /* --------------------- */
            total_time = elapsed_ms(&stage_timing_record.start_time, &stage_timing_record.istft_time);

            /* Print the stage timing */
            fprintf(stage_timing_csv_fp,
                "%u,%s,%u,"
                "%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,"
                "%.6f,%.6f,%.6f,"
                "%.6f,%.6f,"
                "%.6f,%.6f\n",
                timing_idx,
                entry->d_name,
                seg_idx,
                stft_time,
                low_freq_mag_time,
                low_freq_pha_time,
                spectrogram_res_mag_time,
                patch_embed_mag_time,
                patch_embed_pha_time,
                patch_embed_res_mag_time,
                patch_embed_res_pha_time,
                patch_embed_interaction_mag_time,
                patch_embed_interaction_pha_time,
                encoder0_mag_time,
                encoder0_pha_time,
                encoder0_res_mag_time,
                encoder0_res_pha_time,
                encoder0_interaction_mag_time,
                encoder0_interaction_pha_time,
                encoder1_mag_time,
                encoder1_pha_time,
                encoder1_res_mag_time,
                encoder1_res_pha_time,
                encoder1_interaction_mag_time,
                encoder1_interaction_pha_time,
                encoder2_mag_time,
                encoder2_pha_time,
                encoder2_res_mag_time,
                encoder2_res_pha_time,
                encoder2_interaction_mag_time,
                encoder2_interaction_pha_time,
                latent_mag_time,
                latent_pha_time,
                latent_interaction_mag_time,
                latent_interaction_pha_time,
                decoder0_res_mag_time,
                decoder0_res_pha_time,
                decoder0_mag_time,
                decoder0_pha_time,
                decoder0_interaction_mag_time,
                decoder0_interaction_pha_time,
                decoder1_res_mag_time,
                decoder1_res_pha_time,
                decoder1_mag_time,
                decoder1_pha_time,
                decoder1_interaction_mag_time,
                decoder1_interaction_pha_time,
                decoder2_res_mag_time,
                decoder2_res_pha_time,
                decoder2_mag_time,
                decoder2_pha_time,
                decoder2_interaction_mag_time,
                decoder2_interaction_pha_time,
                output0_res_mag_time,
                output0_res_pha_time,
                output0_mag_time,
                output0_pha_time,
                output1_mag_time,
                output1_pha_time,
                output3_mag_time,
                output3_pha_time,
                output3_res_mag_time,
                final_low_freq_mag_time,
                final_low_freq_pha_time,
                istft_time,
                total_time);
#endif

#ifdef _MODEL_DETAIL_STAGE_TIMING_
            stft_time = elapsed_ms(&detail_timing_record.start_time, &detail_timing_record.stft_time);
            /* --------------------- */
            low_freq_mag_time = elapsed_ms(&detail_timing_record.stft_time, &detail_timing_record.low_frequency_mag_time);
            spectrogram_res_mag_time = elapsed_ms(&detail_timing_record.low_frequency_mag_time, &detail_timing_record.spectrogram_res_mag_time);

            low_freq_pha_time = elapsed_ms(&detail_timing_record.stft_time, &detail_timing_record.low_frequency_pha_time);

            /* --------------------- */
            patch_embed_mag_time = elapsed_ms(&detail_timing_record.spectrogram_res_mag_time, &detail_timing_record.patch_embed_mag_time);
            patch_embed_res_mag_time = elapsed_ms(&detail_timing_record.patch_embed_mag_time, &detail_timing_record.patch_embed_res_mag_time);

            patch_embed_pha_time = elapsed_ms(&detail_timing_record.low_frequency_pha_time, &detail_timing_record.patch_embed_pha_time);
            patch_embed_res_pha_time = elapsed_ms(&detail_timing_record.patch_embed_pha_time, &detail_timing_record.patch_embed_res_pha_time);

            patch_embed_interaction_mag_time = elapsed_ms(&detail_timing_record.patch_embed_res_mag_time, &detail_timing_record.patch_embed_interaction_mag_time);
            patch_embed_interaction_pha_time = elapsed_ms(&detail_timing_record.patch_embed_interaction_mag_time, &detail_timing_record.patch_embed_interaction_pha_time);

            /* --------------------- */
            encoder0_pvss_mag_time = elapsed_ms(&detail_timing_record.patch_embed_interaction_pha_time, &detail_timing_record.encoder0_pvss_mag_time);
            encoder0_mlp_mag_time = elapsed_ms(&detail_timing_record.encoder0_pvss_mag_time, &detail_timing_record.encoder0_mlp_mag_time);
                /* --- */
            encoder0_mag_time = elapsed_ms(&detail_timing_record.patch_embed_interaction_pha_time, &detail_timing_record.encoder0_mag_time);
            encoder0_res_mag_time = elapsed_ms(&detail_timing_record.encoder0_mag_time, &detail_timing_record.encoder0_res_mag_time);

            encoder0_pvss_pha_time = elapsed_ms(&detail_timing_record.patch_embed_interaction_pha_time, &detail_timing_record.encoder0_pvss_pha_time);
            encoder0_mlp_pha_time = elapsed_ms(&detail_timing_record.encoder0_pvss_pha_time, &detail_timing_record.encoder0_mlp_pha_time);
                /* --- */
            encoder0_pha_time = elapsed_ms(&detail_timing_record.patch_embed_interaction_pha_time, &detail_timing_record.encoder0_pha_time);
            encoder0_res_pha_time = elapsed_ms(&detail_timing_record.encoder0_pha_time, &detail_timing_record.encoder0_res_pha_time);

            encoder0_interaction_mag_time = elapsed_ms(&detail_timing_record.encoder0_res_mag_time, &detail_timing_record.encoder0_interaction_mag_time);
            encoder0_interaction_pha_time = elapsed_ms(&detail_timing_record.encoder0_interaction_mag_time, &detail_timing_record.encoder0_interaction_pha_time);

            /* --------------------- */
            encoder1_pvss_mag_time = elapsed_ms(&detail_timing_record.encoder0_interaction_pha_time, &detail_timing_record.encoder1_pvss_mag_time);
            encoder1_mlp_mag_time = elapsed_ms(&detail_timing_record.encoder1_pvss_mag_time, &detail_timing_record.encoder1_mlp_mag_time);
                /* --- */
            encoder1_mag_time = elapsed_ms(&detail_timing_record.encoder0_interaction_pha_time, &detail_timing_record.encoder1_mag_time);
            encoder1_res_mag_time = elapsed_ms(&detail_timing_record.encoder1_mag_time, &detail_timing_record.encoder1_res_mag_time);

            encoder1_pvss_pha_time = elapsed_ms(&detail_timing_record.encoder0_interaction_pha_time, &detail_timing_record.encoder1_pvss_pha_time);
            encoder1_mlp_pha_time = elapsed_ms(&detail_timing_record.encoder1_pvss_pha_time, &detail_timing_record.encoder1_mlp_pha_time);
                /* --- */
            encoder1_pha_time = elapsed_ms(&detail_timing_record.encoder0_interaction_pha_time, &detail_timing_record.encoder1_pha_time);
            encoder1_res_pha_time = elapsed_ms(&detail_timing_record.encoder1_pha_time, &detail_timing_record.encoder1_res_pha_time);

            encoder1_interaction_mag_time = elapsed_ms(&detail_timing_record.encoder1_res_mag_time, &detail_timing_record.encoder1_interaction_mag_time);
            encoder1_interaction_pha_time = elapsed_ms(&detail_timing_record.encoder1_interaction_mag_time, &detail_timing_record.encoder1_interaction_pha_time);

            /* --------------------- */
            encoder2_pvss_mag_time = elapsed_ms(&detail_timing_record.encoder1_interaction_pha_time, &detail_timing_record.encoder2_pvss_mag_time);
            encoder2_mlp_mag_time = elapsed_ms(&detail_timing_record.encoder2_pvss_mag_time, &detail_timing_record.encoder2_mlp_mag_time);
                /* --- */
            encoder2_mag_time = elapsed_ms(&detail_timing_record.encoder1_interaction_pha_time, &detail_timing_record.encoder2_mag_time);
            encoder2_res_mag_time = elapsed_ms(&detail_timing_record.encoder2_mag_time, &detail_timing_record.encoder2_res_mag_time);

            encoder2_pvss_pha_time = elapsed_ms(&detail_timing_record.encoder1_interaction_pha_time, &detail_timing_record.encoder2_pvss_pha_time);
            encoder2_mlp_pha_time = elapsed_ms(&detail_timing_record.encoder2_pvss_pha_time, &detail_timing_record.encoder2_mlp_pha_time);
                /* --- */
            encoder2_pha_time = elapsed_ms(&detail_timing_record.encoder1_interaction_pha_time, &detail_timing_record.encoder2_pha_time);
            encoder2_res_pha_time = elapsed_ms(&detail_timing_record.encoder2_pha_time, &detail_timing_record.encoder2_res_pha_time);

            encoder2_interaction_mag_time = elapsed_ms(&detail_timing_record.encoder2_res_mag_time, &detail_timing_record.encoder2_interaction_mag_time);
            encoder2_interaction_pha_time = elapsed_ms(&detail_timing_record.encoder2_interaction_mag_time, &detail_timing_record.encoder2_interaction_pha_time);

            /* --------------------- */
            latent_pvss_mag_time = elapsed_ms(&detail_timing_record.encoder2_interaction_pha_time, &detail_timing_record.latent_pvss_mag_time);
            latent_mlp_mag_time = elapsed_ms(&detail_timing_record.latent_pvss_mag_time, &detail_timing_record.latent_mlp_mag_time);
                /* --- */
            latent_mag_time = elapsed_ms(&detail_timing_record.encoder2_interaction_pha_time, &detail_timing_record.latent_mag_time);
            latent_interaction_mag_time = elapsed_ms(&detail_timing_record.latent_mag_time, &detail_timing_record.latent_interaction_mag_time);

            latent_pvss_pha_time = elapsed_ms(&detail_timing_record.encoder2_interaction_pha_time, &detail_timing_record.latent_pvss_pha_time);
            latent_mlp_pha_time = elapsed_ms(&detail_timing_record.latent_pvss_pha_time, &detail_timing_record.latent_mlp_pha_time);
                /* --- */
            latent_pha_time = elapsed_ms(&detail_timing_record.encoder2_interaction_pha_time, &detail_timing_record.latent_pha_time);
            latent_interaction_pha_time = elapsed_ms(&detail_timing_record.latent_pha_time, &detail_timing_record.latent_interaction_pha_time);

            /* --------------------- */
            decoder0_res_mag_time = elapsed_ms(&detail_timing_record.latent_interaction_pha_time, &detail_timing_record.decoder0_res_mag_time);
            decoder0_pvss_mag_time = elapsed_ms(&detail_timing_record.decoder0_res_mag_time, &detail_timing_record.decoder0_pvss_mag_time);
            decoder0_mlp_mag_time = elapsed_ms(&detail_timing_record.decoder0_pvss_mag_time, &detail_timing_record.decoder0_mlp_mag_time);
                /* --- */
            decoder0_mag_time = elapsed_ms(&detail_timing_record.decoder0_res_mag_time, &detail_timing_record.decoder0_mag_time);

            decoder0_res_pha_time = elapsed_ms(&detail_timing_record.latent_interaction_pha_time, &detail_timing_record.decoder0_res_pha_time);
            decoder0_pvss_pha_time = elapsed_ms(&detail_timing_record.decoder0_res_pha_time, &detail_timing_record.decoder0_pvss_pha_time);
            decoder0_mlp_pha_time = elapsed_ms(&detail_timing_record.decoder0_pvss_pha_time, &detail_timing_record.decoder0_mlp_pha_time);
                /* --- */
            decoder0_pha_time = elapsed_ms(&detail_timing_record.decoder0_res_pha_time, &detail_timing_record.decoder0_pha_time);

            decoder0_interaction_mag_time = elapsed_ms(&detail_timing_record.decoder0_mag_time, &detail_timing_record.decoder0_interaction_mag_time);
            decoder0_interaction_pha_time = elapsed_ms(&detail_timing_record.decoder0_interaction_mag_time, &detail_timing_record.decoder0_interaction_pha_time);

            /* --------------------- */
            decoder1_res_mag_time = elapsed_ms(&detail_timing_record.decoder0_interaction_pha_time, &detail_timing_record.decoder1_res_mag_time);
            decoder1_pvss_mag_time = elapsed_ms(&detail_timing_record.decoder1_res_mag_time, &detail_timing_record.decoder1_pvss_mag_time);
            decoder1_mlp_mag_time = elapsed_ms(&detail_timing_record.decoder1_pvss_mag_time, &detail_timing_record.decoder1_mlp_mag_time);
                /* --- */
            decoder1_mag_time = elapsed_ms(&detail_timing_record.decoder0_interaction_pha_time, &detail_timing_record.decoder1_mag_time);

            decoder1_res_pha_time = elapsed_ms(&detail_timing_record.decoder0_interaction_pha_time, &detail_timing_record.decoder1_res_pha_time);
            decoder1_pvss_pha_time = elapsed_ms(&detail_timing_record.decoder1_res_pha_time, &detail_timing_record.decoder1_pvss_pha_time);
            decoder1_mlp_pha_time = elapsed_ms(&detail_timing_record.decoder1_pvss_pha_time, &detail_timing_record.decoder1_mlp_pha_time);
                /* --- */
            decoder1_pha_time = elapsed_ms(&detail_timing_record.decoder0_interaction_pha_time, &detail_timing_record.decoder1_pha_time);

            decoder1_interaction_mag_time = elapsed_ms(&detail_timing_record.decoder1_mag_time, &detail_timing_record.decoder1_interaction_mag_time);
            decoder1_interaction_pha_time = elapsed_ms(&detail_timing_record.decoder1_interaction_mag_time, &detail_timing_record.decoder1_interaction_pha_time);

            /* --------------------- */
            decoder2_res_mag_time = elapsed_ms(&detail_timing_record.decoder1_interaction_pha_time, &detail_timing_record.decoder2_res_mag_time);
            decoder2_pvss_mag_time = elapsed_ms(&detail_timing_record.decoder2_res_mag_time, &detail_timing_record.decoder2_pvss_mag_time);
            decoder2_mlp_mag_time = elapsed_ms(&detail_timing_record.decoder2_pvss_mag_time, &detail_timing_record.decoder2_mlp_mag_time);
                /* --- */
            decoder2_mag_time = elapsed_ms(&detail_timing_record.decoder1_interaction_pha_time, &detail_timing_record.decoder2_mag_time);

            decoder2_res_pha_time = elapsed_ms(&detail_timing_record.decoder1_interaction_pha_time, &detail_timing_record.decoder2_res_pha_time);
            decoder2_pvss_pha_time = elapsed_ms(&detail_timing_record.decoder2_res_pha_time, &detail_timing_record.decoder2_pvss_pha_time);
            decoder2_mlp_pha_time = elapsed_ms(&detail_timing_record.decoder2_pvss_pha_time, &detail_timing_record.decoder2_mlp_pha_time);
                /* --- */
            decoder2_pha_time = elapsed_ms(&detail_timing_record.decoder1_interaction_pha_time, &detail_timing_record.decoder2_pha_time);

            decoder2_interaction_mag_time = elapsed_ms(&detail_timing_record.decoder2_mag_time, &detail_timing_record.decoder2_interaction_mag_time);
            decoder2_interaction_pha_time = elapsed_ms(&detail_timing_record.decoder2_interaction_mag_time, &detail_timing_record.decoder2_interaction_pha_time);

            /* --------------------- */
            output0_res_mag_time = elapsed_ms(&detail_timing_record.decoder2_interaction_pha_time, &detail_timing_record.output0_res_mag_time);
            output0_pvss_mag_time = elapsed_ms(&detail_timing_record.output0_res_mag_time, &detail_timing_record.output0_pvss_mag_time);
            output0_mlp_mag_time = elapsed_ms(&detail_timing_record.output0_pvss_mag_time, &detail_timing_record.output0_mlp_mag_time);
                /* --- */
            output0_mag_time = elapsed_ms(&detail_timing_record.decoder2_interaction_pha_time, &detail_timing_record.output0_mag_time);

            output0_res_pha_time = elapsed_ms(&detail_timing_record.decoder2_interaction_pha_time, &detail_timing_record.output0_res_pha_time);
            output0_pvss_pha_time = elapsed_ms(&detail_timing_record.output0_res_pha_time, &detail_timing_record.output0_pvss_pha_time);
            output0_mlp_pha_time = elapsed_ms(&detail_timing_record.output0_pvss_pha_time, &detail_timing_record.output0_mlp_pha_time);
                /* --- */
            output0_pha_time = elapsed_ms(&detail_timing_record.decoder2_interaction_pha_time, &detail_timing_record.output0_pha_time);

            /* --------------------- */
            output1_pvss_mag_time = elapsed_ms(&detail_timing_record.output0_mag_time, &detail_timing_record.output1_pvss_mag_time);
            output1_mlp_mag_time = elapsed_ms(&detail_timing_record.output1_pvss_mag_time, &detail_timing_record.output1_mlp_mag_time);
                /* --- */
            output1_mag_time = elapsed_ms(&detail_timing_record.output0_mag_time, &detail_timing_record.output1_mag_time);

            output1_pvss_pha_time = elapsed_ms(&detail_timing_record.output0_mag_time, &detail_timing_record.output1_pvss_pha_time);
            output1_mlp_pha_time = elapsed_ms(&detail_timing_record.output1_pvss_pha_time, &detail_timing_record.output1_mlp_pha_time);
                /* --- */
            output1_pha_time = elapsed_ms(&detail_timing_record.output0_mag_time, &detail_timing_record.output1_pha_time);

            /* --------------------- */
            output3_pvss_mag_time = elapsed_ms(&detail_timing_record.output1_mag_time, &detail_timing_record.output3_pvss_mag_time);
            output3_mlp_mag_time = elapsed_ms(&detail_timing_record.output3_pvss_mag_time, &detail_timing_record.output3_mlp_mag_time);
                /* --- */
            output3_mag_time = elapsed_ms(&detail_timing_record.output1_mag_time, &detail_timing_record.output3_mag_time);
            output3_res_mag_time = elapsed_ms(&detail_timing_record.output3_mag_time, &detail_timing_record.output3_res_mag_time);

            output3_pvss_pha_time = elapsed_ms(&detail_timing_record.output1_mag_time, &detail_timing_record.output3_pvss_pha_time);
            output3_mlp_pha_time = elapsed_ms(&detail_timing_record.output3_pvss_pha_time, &detail_timing_record.output3_mlp_pha_time);
                /* --- */
            output3_pha_time = elapsed_ms(&detail_timing_record.output1_mag_time, &detail_timing_record.output3_pha_time);

            /* --------------------- */
            final_low_freq_mag_time = elapsed_ms(&detail_timing_record.output3_res_mag_time, &detail_timing_record.final_low_freq_mag_time);
            final_low_freq_pha_time = elapsed_ms(&detail_timing_record.output3_pha_time, &detail_timing_record.final_low_freq_pha_time);

            /* --------------------- */
            istft_time = elapsed_ms(&detail_timing_record.final_low_freq_mag_time, &detail_timing_record.istft_time);

            /* --------------------- */
            total_time = elapsed_ms(&detail_timing_record.start_time, &detail_timing_record.istft_time);

            /* Print the detail stage timing */
            fprintf(detail_stage_timing_csv_fp,
                "%u,%s,%u,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f\n",
                timing_idx,
                entry->d_name,
                seg_idx,
                stft_time,
                low_freq_mag_time,
                low_freq_pha_time,
                spectrogram_res_mag_time,
                patch_embed_mag_time,
                patch_embed_pha_time,
                patch_embed_res_mag_time,
                patch_embed_res_pha_time,
                patch_embed_interaction_mag_time,
                patch_embed_interaction_pha_time,

                encoder0_pvss_mag_time,
                encoder0_mlp_mag_time,
                encoder0_mag_time,
                encoder0_pvss_pha_time,
                encoder0_mlp_pha_time,
                encoder0_pha_time,
                encoder0_res_mag_time,
                encoder0_res_pha_time,
                encoder0_interaction_mag_time,
                encoder0_interaction_pha_time,

                encoder1_pvss_mag_time,
                encoder1_mlp_mag_time,
                encoder1_mag_time,
                encoder1_pvss_pha_time,
                encoder1_mlp_pha_time,
                encoder1_pha_time,
                encoder1_res_mag_time,
                encoder1_res_pha_time,
                encoder1_interaction_mag_time,
                encoder1_interaction_pha_time,

                encoder2_pvss_mag_time,
                encoder2_mlp_mag_time,
                encoder2_mag_time,
                encoder2_pvss_pha_time,
                encoder2_mlp_pha_time,
                encoder2_pha_time,
                encoder2_res_mag_time,
                encoder2_res_pha_time,
                encoder2_interaction_mag_time,
                encoder2_interaction_pha_time,

                latent_pvss_mag_time,
                latent_mlp_mag_time,
                latent_mag_time,
                latent_pvss_pha_time,
                latent_mlp_pha_time,
                latent_pha_time,
                latent_interaction_mag_time,
                latent_interaction_pha_time,

                decoder0_pvss_mag_time,
                decoder0_mlp_mag_time,
                decoder0_res_mag_time,
                decoder0_pvss_pha_time,
                decoder0_mlp_pha_time,
                decoder0_res_pha_time,
                decoder0_mag_time,
                decoder0_pha_time,
                decoder0_interaction_mag_time,
                decoder0_interaction_pha_time,

                decoder1_pvss_mag_time,
                decoder1_mlp_mag_time,
                decoder1_res_mag_time,
                decoder1_pvss_pha_time,
                decoder1_mlp_pha_time,
                decoder1_res_pha_time,
                decoder1_mag_time,
                decoder1_pha_time,
                decoder1_interaction_mag_time,
                decoder1_interaction_pha_time,

                decoder2_pvss_mag_time,
                decoder2_mlp_mag_time,
                decoder2_res_mag_time,
                decoder2_pvss_pha_time,
                decoder2_mlp_pha_time,
                decoder2_res_pha_time,
                decoder2_mag_time,
                decoder2_pha_time,
                decoder2_interaction_mag_time,
                decoder2_interaction_pha_time,

                output0_pvss_mag_time,
                output0_mlp_mag_time,
                output0_res_mag_time,
                output0_pvss_pha_time,
                output0_mlp_pha_time,
                output0_res_pha_time,
                output0_mag_time,
                output0_pha_time,

                output1_pvss_mag_time,
                output1_mlp_mag_time,
                output1_mag_time,
                output1_pvss_pha_time,
                output1_mlp_pha_time,
                output1_pha_time,

                output3_pvss_mag_time,
                output3_mlp_mag_time,
                output3_mag_time,
                output3_pvss_pha_time,
                output3_mlp_pha_time,
                output3_pha_time,
                output3_res_mag_time,

                final_low_freq_mag_time,
                final_low_freq_pha_time,
                istft_time,
                total_time);
#endif

            /* Print the overall timing */
            fprintf(csv_fp, "%u,%s,%u,%.6f\n",
                    timing_idx,
                    entry->d_name,
                    seg_idx,
                    processing_time_ms);
            printf("%u,%s,%u,%.6f\n",
                    timing_idx,
                    entry->d_name,
                    seg_idx,
                    processing_time_ms);

            timing_idx++;
#ifdef _MODEL_MEM_PROFILE_
            /* The memory profile is data-independent: one segment is the whole report. */
            free(per_file_audio);
            per_file_audio = NULL;
            goto mem_done;
#endif
        }

        audio_write_test_wav_f32(output_path, per_file_audio, &meta);

        free(per_file_audio);
        per_file_audio = NULL;
    }

#ifdef _MODEL_MEM_PROFILE_
mem_done:
#endif
    closedir(dir_ptr);
    fclose(csv_fp);
#ifdef _MODEL_STAGE_TIMING_
    fclose(stage_timing_csv_fp);
#endif
#ifdef _MODEL_DETAIL_STAGE_TIMING_
    fclose(detail_stage_timing_csv_fp);
#endif

    /*< 9. Write timing summary >*/
    write_timing_summary(processing_time_collect, timing_idx);

    return 0;
}



/* ===================== FUNCTION DEFINITION ================= */
/*< 1. Checking wav extension of a file >*/
static int has_wav_extension(const char *name)
{
    size_t len;

    if (name == NULL) {
        return 0;
    }

    len = strlen(name);
    if (len < 4U) {
        return 0;
    }

    return (strcmp(name + len - 4U, ".wav") == 0);
}

/*< 2. Build the file path base on file name >*/
static void build_path(char *dst, size_t dst_size, const char *dir_path, const char *file_name)
{
    int written;

    if (dst == NULL || dir_path == NULL || file_name == NULL) {
        fprintf(stderr, "build_path received NULL argument\n");
        exit(EXIT_FAILURE);
    }

    written = snprintf(dst, dst_size, "%s%s", dir_path, file_name);
    if (written < 0 || (size_t)written >= dst_size) {
        fprintf(stderr, "path too long: %s%s\n", dir_path, file_name);
        exit(EXIT_FAILURE);
    }
}

/*< 3. Check whether the folder exist >*/
static void ensure_generated_dir_exists(void)
{
    if (mkdir(GENERATED_AUDIO_PATH, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "failed to create output dir: %s\n", GENERATED_AUDIO_PATH);
        exit(EXIT_FAILURE);
    }
}

/*< 4. Ensure the result folder exists >*/
static void ensure_result_dir_exists(void)
{
    if (mkdir(RESULT_PATH, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "failed to create result dir: %s\n", RESULT_PATH);
        exit(EXIT_FAILURE);
    }
}

/*< 5. Calculate elapsed time >*/
static double elapsed_ms(const struct timespec *start, const struct timespec *end)
{
    const double sec = (double)(end->tv_sec - start->tv_sec);
    const double nsec = (double)(end->tv_nsec - start->tv_nsec);
    return (sec * 1000.0) + (nsec / 1000000.0);
}

/*< 6. Write timing summary >*/
static void write_timing_summary(const double *timings, uint32_t count)
{
    FILE *fp;
    double sum;
    double mean;
    double var_sum;
    double std;
    uint32_t idx;

    if (timings == NULL) {
        fprintf(stderr, "write_timing_summary received NULL timings\n");
        exit(EXIT_FAILURE);
    }

    fp = fopen(PROCESSING_TIMING_TXT_PATH, "w");
    if (fp == NULL) {
        fprintf(stderr, "failed to open summary file: %s\n", PROCESSING_TIMING_TXT_PATH);
        exit(EXIT_FAILURE);
    }

    if (count == 0U) {
        fprintf(fp, "total_segments_tested: 0\n");
        fprintf(fp, "mean_processing_time_ms: 0.000000\n");
        fprintf(fp, "std_processing_time_ms: 0.000000\n");
        fclose(fp);
        return;
    }

    sum = 0.0;
    for (idx = 0U; idx < count; ++idx) {
        sum += timings[idx];
    }
    mean = sum / (double)count;

    var_sum = 0.0;
    for (idx = 0U; idx < count; ++idx) {
        const double diff = timings[idx] - mean;
        var_sum += diff * diff;
    }
    std = sqrt(var_sum / (double)count);

    fprintf(fp, "total_segments_tested: %u\n", count);
    fprintf(fp, "mean_processing_time_ms: %.6f\n", mean);
    fprintf(fp, "std_processing_time_ms: %.6f\n", std);

    fclose(fp);
}
