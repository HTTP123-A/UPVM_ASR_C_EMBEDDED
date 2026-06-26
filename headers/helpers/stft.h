#ifndef _STFT_
#define _STFT_

/* ===================== INCLUDE SECTION ===================== */
/* <1. Built-in header > */
#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* <2. User headers > */

/* ===================== DEFINE ============================== */
#define TWO_PI          6.2831853071795864769F
#define STFT_EPS        1e-08f

/* ===================== DATATYPES =========================== */


/* ===================== FUNCTION PROTOTYPE SECTION ========== */
/* ---------------------------------------------- */
/* --- F32 --- */
/*< 1. Init stft required struct (FFTW) >*/
void stft_init_f32(float **win, float **frame, fftwf_complex **spec,
                float **ola, float **wss, fftwf_plan *p_stft, fftwf_plan *p_istft,
                uint32_t *window_offset, uint32_t n_fft, uint32_t win_length, uint32_t ola_length, uint32_t frame_size);

/*< 2. STFT (FFTW) >*/
void stft_f32(const float *audio, float *mag, float *pha,
            float *win, float *frame,
            fftwf_complex *spec, fftwf_plan *p,            
            uint32_t length, uint32_t n_fft, uint32_t win_length,
            uint32_t hop_length, uint32_t frame_size, uint32_t num_of_frame, uint32_t win_offset, float inv_sqrt_nfft);

/*< 3. iSTFT >*/
void istft_f32(const float *mag, const float *pha, float *audio,
            float *win, float *frame, float *ola, float *wss,
            fftwf_complex *spec, fftwf_plan *p,
            uint32_t length, uint32_t n_fft, uint32_t win_length, uint32_t hop_length,
            uint32_t pad, uint32_t ola_length, uint32_t frame_size, uint32_t num_of_frame, uint32_t win_offset, float inv_sqrt_nfft);

/* --- INT8 -- */

/* ---------------------------------------------- */

#endif /* __STFT__ */