/* Author: Nguyen Quang Phuong
 * Email: ngquangph@gmail.com
 *
 * Notes:
 *  Implement STFT & iSTFT based on FFTW
 * 
 */

  /* ===================== INCLUDE SECTION ===================== */
/*< 1. Built-in header >*/


/*< 2. User headers >*/
#include "stft.h"

/* ===================== GLOBAL VARIABLES ==================== */


/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/* ---------------------------------------------- */
/* --- F32 --- */
/*< 1. Initialize >*/
void stft_init_f32(float **win, float **frame, fftwf_complex **spec,
                float **ola, float **wss, fftwf_plan *p_stft, fftwf_plan *p_istft,
                uint32_t *window_offset, uint32_t n_fft, uint32_t win_length, uint32_t ola_length, uint32_t frame_size)
{
    /*< Create Hann window >*/
    *win = calloc(win_length, sizeof(float));
    if (win_length == 1) {
        (*win)[0] = 1.0f;
    }
    else {
        for (uint32_t i = 0U; i < win_length; i++) {
            (*win)[i] = 0.5f - 0.5f * cosf((TWO_PI * (float)i) / (float)win_length);
        }
    }

    /*< Allocate memory for frame and spec >*/
    *frame = fftwf_alloc_real((size_t)n_fft); // Frame
    *spec = fftwf_alloc_complex((size_t)frame_size);

    /*< Allocate memory for ola and wss (for istft) >*/
    *ola = calloc((size_t)ola_length, sizeof(float));
    *wss = malloc((size_t)ola_length * sizeof(float)); // WSS have the same length with OLA scratch buffer

    /*< Initialize plan >*/
    *p_stft = fftwf_plan_dft_r2c_1d(n_fft, *frame, *spec, FFTW_MEASURE);
    *p_istft = fftwf_plan_dft_c2r_1d(n_fft, *spec, *frame, FFTW_MEASURE);

    /*< Calculate window offset >*/
    *window_offset = (uint32_t)((n_fft - win_length) / 2);
}

/*< 2. STFT >*/
void stft_f32(const float *audio, float *mag, float *pha,
            float *win, float *frame,
            fftwf_complex *spec, fftwf_plan *p,
            uint32_t length, uint32_t n_fft, uint32_t win_length,
            uint32_t hop_length, uint32_t frame_size, uint32_t num_of_frame, uint32_t win_offset, float inv_sqrt_nfft)
{
    /* Calculate STFT */
    for (uint32_t t = 0; t < num_of_frame; t++) {
        /* 1. Set the current frame to 0s (avoid overlap with the next frame) */
        memset(frame, 0, (size_t)n_fft * sizeof(float));
        /* 2. Start point in time domain for the i-th frame */
        const int start = (int)(t * hop_length - (n_fft / 2));
        /* 3. Padding mode `reflect` * Hann Window */
        for (uint32_t k = 0; k < win_length; k++) {
            int kth_sample = start + k;
            while ((kth_sample < (int)0) || (kth_sample >= (int)length)) { // reflect logic
                if (kth_sample < 0) kth_sample = -kth_sample;
                else kth_sample = 2 * length - kth_sample - 2;
            }

            frame[win_offset + k] = audio[kth_sample] * win[k]; // Hann window            
        }

        /* 4. Execute STFT */
        fftwf_execute(*p);

        /* 5. Obtain the magnitude & phase */
        for (uint32_t f = 0; f < frame_size; f++) {
            float re = spec[f][0] * inv_sqrt_nfft; // Real
            float im = spec[f][1] * inv_sqrt_nfft; // Imagine
            const size_t write_idx = (size_t)f * num_of_frame + t; // [F,T]
            const float amp = hypotf(re, im);
            mag[write_idx] = log2f(amp + STFT_EPS); 
            pha[write_idx] = (amp > 1e-5f) ? atan2f(im, re) : 0.0f;
        }
    }
}

/*< 3. iSTFT >*/
void istft_f32(const float *mag, const float *pha, float *audio,
            float *win, float *frame, float *ola, float *wss,
            fftwf_complex *spec, fftwf_plan *p,
            uint32_t length, uint32_t n_fft, uint32_t win_length, uint32_t hop_length,
            uint32_t pad, uint32_t ola_length, uint32_t frame_size, uint32_t num_of_frame, uint32_t win_offset, float inv_sqrt_nfft)
{
    /* 1. Set OLA & WSS to 0s */
    memset(ola, 0, (size_t)ola_length * sizeof(float));
    memset(wss, 0, (size_t)ola_length * sizeof(float));

    /* 2. Calculate iSTFT */
    for (uint32_t t = 0; t < num_of_frame; ++t) {
        /* 2.1. Form the complex spectrum from mag + phase */
        for (uint32_t f = 0; f < frame_size; ++f) {
            const size_t spectrum_idx = (size_t)f * num_of_frame + t;   /* [F,T] */
            const float amp = exp2f(mag[spectrum_idx]);                 /* inverse log2 */
            const float ph  = pha[spectrum_idx];                        /* Copy original - can by bypass */
            spec[f][0] = amp * cosf(ph);
            spec[f][1] = amp * sinf(ph);
        }

        /* 2.2. Execute iSTFT */
        fftwf_execute(*p);

        /* 2.3. OLA Step */
        const uint32_t start = t * hop_length;
        for (uint32_t n = 0; n < n_fft; ++n) {
            float w = 0.0f;
            if (n >= win_offset && n < (win_offset + win_length)) {
                w = win[n - win_offset];
            }

            const uint32_t pos = start + n;
            const float s = frame[n] * inv_sqrt_nfft; /* undo FFTW scaling to match normalized=True */
            ola[pos] += s * w;
            wss[pos] += w * w;
        }
    }

    /* 3. NOLA Normalization () */
    for (uint32_t i = 0; i < ola_length; ++i) {
        if (wss[i] > STFT_EPS) {
            ola[i] /= wss[i];
        } else {
            ola[i] = 0.0f;
        }
    }

    /* 4. Truncate to original audio length */
    const uint32_t valid_len = (ola_length > 2U * pad) ? (ola_length - 2U * pad) : 0U;
    const uint32_t copy_len = (length < valid_len) ? length : valid_len;

    if (copy_len > 0U) {
        memcpy(audio, ola + pad, (size_t)copy_len * sizeof(float));
    }
    if (length > copy_len) {
        memset(audio + copy_len, 0, (size_t)(length - copy_len) * sizeof(float));
    }
}

/* --- INT8 -- */