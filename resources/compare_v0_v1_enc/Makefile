# Encoder-0 latency micro-benchmark: builds one standalone executable per version
# (f32 naives_mp / sq_int8_v0 / sq_int8_v5). Separate binaries are mandatory — the
# block, kernel and struct names are identical across versions, so they cannot be
# linked into one program. Compile flags mirror the project Makefile (-O3) so the
# numbers are representative of the real build.
#
#   make ROOT=../.. FFTW_INC=/path/to/fftw3/local/include
#
# FFTW_INC only needs to expose fftw3.h as a HEADER (model.h includes it); no FFTW
# symbol is referenced, so nothing from FFTW is linked.

ROOT     ?= ../..
FFTW_INC ?= $(ROOT)/../fftw3/local/include
CC       ?= gcc
CFLAGS   ?= -std=c11 -O3 -Wall -Wextra -fopenmp
LDLIBS   ?= -lm -fopenmp

INC_COMMON = -I. -I$(ROOT)/headers/helpers -I$(FFTW_INC)

F32_SRCS = $(wildcard $(ROOT)/sources/naives_mp/kernels/*.c) \
           $(ROOT)/sources/naives_mp/model_timing/pvss_ds_timing_mp_f32.c \
           $(ROOT)/sources/naives_mp/common/weight_loader.c
V0_SRCS  = $(wildcard $(ROOT)/sources/sq_int8_v0/kernels/*.c) \
           $(ROOT)/sources/sq_int8_v0/model_timing/pvss_ds_timing_mp_int8.c \
           $(ROOT)/sources/sq_int8_v0/common/weight_loader_f32_i8.c
V5_SRCS  = $(wildcard $(ROOT)/sources/sq_int8_v5/kernels/*.c) \
           $(ROOT)/sources/sq_int8_v5/model_timing/pvss_ds_timing_mp_int8.c \
           $(ROOT)/sources/sq_int8_v5/common/weight_loader_f32_i8.c

.PHONY: all clean
all: bench_f32 bench_v0 bench_v5

bench_f32: bench_f32.c $(F32_SRCS)
	$(CC) $(CFLAGS) -I$(ROOT)/headers/naives_mp  $(INC_COMMON) -DVERSION_TAG='"f32_naive_mp"' $^ -o $@ $(LDLIBS)

bench_v0: bench_int8.c $(V0_SRCS)
	$(CC) $(CFLAGS) -I$(ROOT)/headers/sq_int8_v0 $(INC_COMMON) -DVERSION_TAG='"sq_int8_v0"' $^ -o $@ $(LDLIBS)

bench_v5: bench_int8.c $(V5_SRCS)
	$(CC) $(CFLAGS) -I$(ROOT)/headers/sq_int8_v5 $(INC_COMMON) -DVERSION_TAG='"sq_int8_v5"' $^ -o $@ $(LDLIBS)

clean:
	rm -f bench_f32 bench_v0 bench_v5
