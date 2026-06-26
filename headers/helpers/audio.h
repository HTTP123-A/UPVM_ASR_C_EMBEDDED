#ifndef _AUDIO_
#define _AUDIO_

/* ===================== INCLUDE SECTION ===================== */
/*< 1. Built-in header >*/
#include <stddef.h>
#include <stdint.h>

/*< 2. User headers >*/

/* ===================== DATATYPES =========================== */
struct audio_io_meta {
    uint32_t input_sample_rate;
    uint32_t target_sample_rate;
    uint32_t segment_length; /* fixed per-segment length (AUDIO_LENGTH) */
    uint32_t num_segments;   /* unfold segment count with step=(segment-overlap) */
    uint32_t original_length;/* decoded waveform length before any pad */
    uint32_t padded_length;  /* tester-style right-pad length before unfold */
    uint32_t pad_length;     /* right-pad amount used to reach padded_length */
    uint16_t input_channels;
    uint16_t bits_per_sample;
};

/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/*< 1. Read degraded 48 kHz mono WAV, apply tester-style pad, and unfold >*/
void audio_read_test_wav_f32(const char *input_path,
                             float **segments,
                             struct audio_io_meta *meta);

/*< 2. Fold overlap-averaged segments and trim back to original length >*/
void audio_write_test_wav_f32(const char *output_path,
                              const float *segments,
                              const struct audio_io_meta *meta);

#endif /* _AUDIO_ */
