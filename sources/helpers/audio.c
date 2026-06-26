/* Author: Nguyen Quang Phuong
 * Email: ngquangph@gmail.com
 *
 * Notes:
 *  WAV test audio I/O helpers for the embedded UPVM-ASR pipeline.
 *
 */

/* ===================== INCLUDE SECTION ===================== */
/*< 1. Built-in header >*/
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*< 2. User headers >*/
#include "audio.h"
#include "model.h"

/* ===================== DEFINE ============================== */
#define WAV_FMT_PCM            0x0001U
#define WAV_FMT_IEEE_FLOAT     0x0003U
#define WAV_FMT_EXTENSIBLE     0xFFFEU
#define WAV_HEADER_SIZE        44U

/* ===================== FUNCTION DEFINE SECTION ============= */
/*< 1. Print audio I/O error and terminate >*/
static void die_audio_io(const char *msg, const char *path)
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

/*< 2. Read one uint16 little-endian value >*/
static uint16_t read_u16_le(const unsigned char *src)
{
    return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
}

/*< 3. Read one uint32 little-endian value >*/
static uint32_t read_u32_le(const unsigned char *src)
{
    return (uint32_t)src[0]
         | ((uint32_t)src[1] << 8)
         | ((uint32_t)src[2] << 16)
         | ((uint32_t)src[3] << 24);
}

/*< 4. Write one uint16 little-endian value >*/
static void write_u16_le(unsigned char *dst, uint16_t value)
{
    dst[0] = (unsigned char)(value & 0xFFU);
    dst[1] = (unsigned char)((value >> 8) & 0xFFU);
}

/*< 5. Write one uint32 little-endian value >*/
static void write_u32_le(unsigned char *dst, uint32_t value)
{
    dst[0] = (unsigned char)(value & 0xFFU);
    dst[1] = (unsigned char)((value >> 8) & 0xFFU);
    dst[2] = (unsigned char)((value >> 16) & 0xFFU);
    dst[3] = (unsigned char)((value >> 24) & 0xFFU);
}

/*< 6. Allocate float32 buffer or terminate >*/
static float *alloc_audio_f32_or_die(size_t numel)
{
    float *buf = (float *)calloc(numel, sizeof(float));
    if (buf == NULL) {
        die_audio_io("failed to allocate audio buffer", NULL);
    }
    return buf;
}

/*< 7. Skip one RIFF chunk payload including word alignment >*/
static void skip_chunk_or_die(FILE *fp, uint32_t chunk_size, const char *path)
{
    const long skip_size = (long)chunk_size + (long)(chunk_size & 1U);
    if (fseek(fp, skip_size, SEEK_CUR) != 0) {
        die_audio_io("failed to skip wav chunk", path);
    }
}

/*< 8. Read one RIFF chunk payload including word alignment >*/
static unsigned char *read_chunk_or_die(FILE *fp, uint32_t chunk_size, const char *path)
{
    unsigned char *buf = NULL;

    if (chunk_size > 0U) {
        buf = (unsigned char *)malloc((size_t)chunk_size);
        if (buf == NULL) {
            die_audio_io("failed to allocate wav chunk buffer", NULL);
        }
        if (fread(buf, 1U, chunk_size, fp) != chunk_size) {
            free(buf);
            die_audio_io("failed to read wav chunk", path);
        }
    }

    if ((chunk_size & 1U) != 0U) {
        if (fgetc(fp) == EOF) {
            free(buf);
            die_audio_io("failed to read wav chunk padding", path);
        }
    }

    return buf;
}

/*< 9. Decode one PCM or float sample to normalized float32 >*/
static float decode_wav_sample_f32(const unsigned char *src,
                                   uint16_t audio_format,
                                   uint16_t bits_per_sample)
{
    int32_t sample_i32;
    float sample_f32;

    if (audio_format == WAV_FMT_IEEE_FLOAT) {
        if (bits_per_sample != 32U) {
            die_audio_io("unsupported IEEE float WAV bit depth", NULL);
        }
        memcpy(&sample_f32, src, sizeof(float));
        return sample_f32;
    }

    switch (bits_per_sample) {
        case 8U:
            return ((float)src[0] - 128.0f) / 128.0f;
        case 16U:
            sample_i32 = (int16_t)read_u16_le(src);
            return (float)sample_i32 / 32768.0f;
        case 24U:
            sample_i32 = (int32_t)((uint32_t)src[0]
                                 | ((uint32_t)src[1] << 8)
                                 | ((uint32_t)src[2] << 16));
            if ((sample_i32 & 0x00800000L) != 0) {
                sample_i32 |= (int32_t)0xFF000000L;
            }
            return (float)sample_i32 / 8388608.0f;
        case 32U:
            sample_i32 = (int32_t)read_u32_le(src);
            return (float)sample_i32 / 2147483648.0f;
        default:
            die_audio_io("unsupported PCM WAV bit depth", NULL);
            return 0.0f;
    }
}

/*< 10. Decode single-channel WAV samples to float32 >*/
static float *decode_wav_single_channel_f32(const unsigned char *data_buf,
                                            uint32_t data_size,
                                            uint16_t audio_format,
                                            uint16_t bits_per_sample,
                                            uint16_t block_align,
                                            uint32_t *num_frames_out)
{
    const uint16_t bytes_per_sample = (uint16_t)(bits_per_sample / 8U);
    const uint32_t num_frames = (block_align > 0U) ? (data_size / block_align) : 0U;
    float *audio = alloc_audio_f32_or_die((num_frames > 0U) ? (size_t)num_frames : 1U);

    if (bytes_per_sample == 0U || block_align != bytes_per_sample) {
        die_audio_io("invalid single-channel WAV sample width", NULL);
    }

    for (uint32_t frame_idx = 0U; frame_idx < num_frames; ++frame_idx) {
        const unsigned char *frame_ptr = data_buf + ((size_t)frame_idx * block_align);
        audio[frame_idx] = decode_wav_sample_f32(frame_ptr, audio_format, bits_per_sample);
    }

    *num_frames_out = num_frames;
    return audio;
}

/*< 12. Clip float32 sample to [-1, 1] >*/
static float clip_audio_sample_f32(float value)
{
    if (value > 1.0f) {
        return 1.0f;
    }
    if (value < -1.0f) {
        return -1.0f;
    }
    return value;
}

/*< 13. Encode one normalized float32 sample to PCM16 >*/
static int16_t encode_pcm16_f32(float sample)
{
    const float clipped = clip_audio_sample_f32(sample);

    if (clipped >= 1.0f) {
        return (int16_t)32767;
    }
    if (clipped <= -1.0f) {
        return (int16_t)-32768;
    }
    return (int16_t)lrintf(clipped * 32767.0f);
}

/*< 14. Read degraded 48 kHz mono WAV and unfold like PyTorch tester path >*/
void audio_read_test_wav_f32(const char *input_path,
                             float **segments,
                             struct audio_io_meta *meta)
{
    FILE *fp;
    unsigned char header[12];
    unsigned char chunk_header[8];
    unsigned char *fmt_buf = NULL;
    unsigned char *data_buf = NULL;
    uint32_t fmt_size = 0U;
    uint32_t data_size = 0U;
    uint16_t audio_format = 0U;
    uint16_t num_channels = 0U;
    uint16_t bits_per_sample = 0U;
    uint16_t block_align = 0U;
    uint32_t input_sample_rate = 0U;
    uint32_t decoded_length = 0U;
    float *decoded = NULL;
    float *flat_segments = NULL;
    uint32_t segment_length;
    uint32_t num_segments;
    uint32_t padded_length;
    uint32_t pad_length;
    uint32_t segment_step;
    const uint32_t overlap_length = OVERLAP_LENGTH; /* match PyTorch TEST.OVERLAP */

    if (input_path == NULL || segments == NULL || meta == NULL) {
        die_audio_io("audio_read_test_wav_f32 received NULL argument", input_path);
    }

    *segments = NULL;
    memset(meta, 0, sizeof(*meta));

    fp = fopen(input_path, "rb");
    if (fp == NULL) {
        die_audio_io("failed to open input wav", input_path);
    }

    if (fread(header, 1U, sizeof(header), fp) != sizeof(header)) {
        fclose(fp);
        die_audio_io("failed to read wav header", input_path);
    }

    if (memcmp(header, "RIFF", 4U) != 0 || memcmp(header + 8U, "WAVE", 4U) != 0) {
        fclose(fp);
        die_audio_io("unsupported wav container", input_path);
    }

    while ((fmt_buf == NULL || data_buf == NULL) &&
           fread(chunk_header, 1U, sizeof(chunk_header), fp) == sizeof(chunk_header)) {
        const uint32_t chunk_size = read_u32_le(chunk_header + 4U);

        if (memcmp(chunk_header, "fmt ", 4U) == 0) {
            free(fmt_buf);
            fmt_buf = read_chunk_or_die(fp, chunk_size, input_path);
            fmt_size = chunk_size;
        } else if (memcmp(chunk_header, "data", 4U) == 0) {
            free(data_buf);
            data_buf = read_chunk_or_die(fp, chunk_size, input_path);
            data_size = chunk_size;
        } else {
            skip_chunk_or_die(fp, chunk_size, input_path);
        }
    }

    fclose(fp);

    if (fmt_buf == NULL || data_buf == NULL || fmt_size < 16U) {
        free(fmt_buf);
        free(data_buf);
        die_audio_io("missing required fmt/data chunk in wav", input_path);
    }

    audio_format = read_u16_le(fmt_buf);
    num_channels = read_u16_le(fmt_buf + 2U);
    input_sample_rate = read_u32_le(fmt_buf + 4U);
    block_align = read_u16_le(fmt_buf + 12U);
    bits_per_sample = read_u16_le(fmt_buf + 14U);

    if (audio_format == WAV_FMT_EXTENSIBLE) {
        if (fmt_size < 40U) {
            free(fmt_buf);
            free(data_buf);
            die_audio_io("unsupported extensible wav fmt chunk", input_path);
        }
        audio_format = read_u16_le(fmt_buf + 24U);
    }

    if ((audio_format != WAV_FMT_PCM) && (audio_format != WAV_FMT_IEEE_FLOAT)) {
        free(fmt_buf);
        free(data_buf);
        die_audio_io("unsupported wav format", input_path);
    }

    if (num_channels == 0U || block_align == 0U || input_sample_rate == 0U) {
        free(fmt_buf);
        free(data_buf);
        die_audio_io("invalid wav fmt values", input_path);
    }

    if (input_sample_rate != OUTPUT_FREQ) {
        free(fmt_buf);
        free(data_buf);
        die_audio_io("degraded wav must already be 48 kHz", input_path);
    }

    if (num_channels != 1U) {
        free(fmt_buf);
        free(data_buf);
        die_audio_io("degraded wav must already be mono", input_path);
    }

    decoded = decode_wav_single_channel_f32(data_buf,
                                            data_size,
                                            audio_format,
                                            bits_per_sample,
                                            block_align,
                                            &decoded_length);

    segment_length = AUDIO_LENGTH;

    if (overlap_length >= segment_length) {
        free(fmt_buf);
        free(data_buf);
        free(decoded);
        die_audio_io("invalid overlap: must be < AUDIO_LENGTH", input_path);
    }

    segment_step = segment_length - overlap_length;

    if (decoded_length == 0U) {
        /* Keep one zero segment for robustness on empty input. */
        pad_length = segment_length;
    } else {
        pad_length = (segment_length - (decoded_length % segment_length)) % segment_length;
        if (decoded_length < segment_length) {
            pad_length = segment_length - decoded_length;
        }
    }

    padded_length = decoded_length + pad_length;
    if (padded_length == 0U || padded_length <= segment_length) {
        num_segments = 1U;
    } else {
        /*
         * Mirror torch.Tensor.unfold(..., size=segment_length, step=segment_step):
         * num_segments = floor((padded_length - segment_length) / segment_step) + 1
         */
        num_segments = 1U + FLOOR_DIV_U((padded_length - segment_length), segment_step);
    }

    flat_segments = alloc_audio_f32_or_die((size_t)num_segments * (size_t)segment_length);

    if (decoded_length > 0U) {
        for (uint32_t seg_idx = 0U; seg_idx < num_segments; ++seg_idx) {
            const uint32_t start = seg_idx * segment_step;
            const uint32_t remaining = (start < decoded_length) ? (decoded_length - start) : 0U;
            const uint32_t copy_len = (remaining < segment_length) ? remaining : segment_length;

            if (copy_len == 0U) {
                break;
            }

            memcpy(flat_segments + ((size_t)seg_idx * segment_length),
                   decoded + (size_t)start,
                   (size_t)copy_len * sizeof(float));
        }
    }

    meta->input_sample_rate = input_sample_rate;
    meta->target_sample_rate = OUTPUT_FREQ;
    meta->segment_length = segment_length;
    meta->num_segments = num_segments;
    meta->original_length = decoded_length;
    meta->padded_length = padded_length; /* tester-style pre-unfold waveform length */
    meta->pad_length = pad_length;       /* right-pad appended before unfold */
    meta->input_channels = num_channels;
    meta->bits_per_sample = bits_per_sample;

    *segments = flat_segments;

    free(fmt_buf);
    free(data_buf);
    free(decoded);
}

/*< 15. Fold processed float32 segments and write mono PCM16 WAV >*/
void audio_write_test_wav_f32(const char *output_path,
                              const float *segments,
                              const struct audio_io_meta *meta)
{
    FILE *fp;
    unsigned char wav_header[WAV_HEADER_SIZE];
    float *folded = NULL;
    uint32_t *fold_count = NULL;

    const uint16_t num_channels = 1U;
    const uint16_t bits_per_sample = 16U;
    const uint16_t block_align = (uint16_t)(num_channels * (bits_per_sample / 8U));

    const uint32_t sample_rate = (meta != NULL && meta->target_sample_rate > 0U)
                               ? meta->target_sample_rate
                               : OUTPUT_FREQ;
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t output_length = (meta != NULL) ? meta->original_length : 0U;
    const uint32_t data_bytes = output_length * block_align;

    const uint32_t overlap_length = OVERLAP_LENGTH; /* must match read-side overlap */
    uint32_t step;
    uint32_t coverage_length;

    if (output_path == NULL || segments == NULL || meta == NULL) {
        die_audio_io("audio_write_test_wav_f32 received NULL argument", output_path);
    }

    if (meta->segment_length == 0U || meta->num_segments == 0U) {
        die_audio_io("invalid audio metadata for wav write", output_path);
    }

    if (overlap_length >= meta->segment_length && meta->num_segments > 1U) {
        die_audio_io("invalid overlap for wav fold/write", output_path);
    }

    step = (meta->num_segments > 1U)
         ? (meta->segment_length - overlap_length)
         : meta->segment_length;

    coverage_length = meta->segment_length;
    if (meta->num_segments > 1U) {
        coverage_length += (meta->num_segments - 1U) * step;
    }

    if (meta->original_length > meta->padded_length) {
        die_audio_io("invalid audio metadata for wav write", output_path);
    }
    if (coverage_length > meta->padded_length) {
        die_audio_io("invalid fold coverage for wav write", output_path);
    }

    folded = alloc_audio_f32_or_die((size_t)coverage_length);
    fold_count = (uint32_t *)calloc((size_t)coverage_length, sizeof(uint32_t));
    if (fold_count == NULL) {
        free(folded);
        die_audio_io("failed to allocate fold counter buffer", output_path);
    }

    for (uint32_t seg_idx = 0U; seg_idx < meta->num_segments; ++seg_idx) {
        const uint32_t start = (meta->num_segments > 1U) ? (seg_idx * step) : 0U;
        const float *seg_ptr = segments + ((size_t)seg_idx * meta->segment_length);

        for (uint32_t k = 0U; k < meta->segment_length; ++k) {
            const uint32_t dst = start + k;
            if (dst >= coverage_length) {
                break;
            }
            folded[dst] += seg_ptr[k];
            fold_count[dst] += 1U;
        }
    }

    fp = fopen(output_path, "wb");
    if (fp == NULL) {
        free(folded);
        free(fold_count);
        die_audio_io("failed to open output wav", output_path);
    }

    memcpy(wav_header + 0U, "RIFF", 4U);
    write_u32_le(wav_header + 4U, 36U + data_bytes);
    memcpy(wav_header + 8U, "WAVE", 4U);
    memcpy(wav_header + 12U, "fmt ", 4U);
    write_u32_le(wav_header + 16U, 16U);
    write_u16_le(wav_header + 20U, WAV_FMT_PCM);
    write_u16_le(wav_header + 22U, num_channels);
    write_u32_le(wav_header + 24U, sample_rate);
    write_u32_le(wav_header + 28U, byte_rate);
    write_u16_le(wav_header + 32U, block_align);
    write_u16_le(wav_header + 34U, bits_per_sample);
    memcpy(wav_header + 36U, "data", 4U);
    write_u32_le(wav_header + 40U, data_bytes);

    if (fwrite(wav_header, 1U, sizeof(wav_header), fp) != sizeof(wav_header)) {
        fclose(fp);
        free(folded);
        free(fold_count);
        die_audio_io("failed to write wav header", output_path);
    }

    for (uint32_t idx = 0U; idx < output_length; ++idx) {
        unsigned char sample_bytes[2];
        int16_t sample_pcm16;
        float sample_f32 = 0.0f;

        if (idx < coverage_length && fold_count[idx] > 0U) {
            sample_f32 = folded[idx] / (float)fold_count[idx];
        } else {
            /* Match PyTorch fold behavior: uncovered tail remains zero. */
            sample_f32 = 0.0f;
        }
        sample_pcm16 = encode_pcm16_f32(sample_f32);

        write_u16_le(sample_bytes, (uint16_t)sample_pcm16);
        if (fwrite(sample_bytes, 1U, sizeof(sample_bytes), fp) != sizeof(sample_bytes)) {
            fclose(fp);
            free(folded);
            free(fold_count);
            die_audio_io("failed to write wav data", output_path);
        }
    }

    fclose(fp);
    free(folded);
    free(fold_count);
}
