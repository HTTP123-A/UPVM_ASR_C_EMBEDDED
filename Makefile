CC ?= gcc
UNAME_S := $(shell uname -s)

PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(PROJECT_DIR)/..)

VERSION ?= NAIVES
VERSION_DIR := $(shell printf '%s' "$(VERSION)" | tr '[:upper:]' '[:lower:]')
VERSION_DEFINE := UPVM_ASR_IMPL_$(VERSION)

MODE ?= NORMAL
MODE_DIR := $(shell printf '%s' "$(MODE)" | tr '[:upper:]' '[:lower:]')

BUILD_ROOT := $(PROJECT_DIR)/build
BUILD_DIR := $(BUILD_ROOT)/$(VERSION_DIR)/$(MODE_DIR)
RESULT_DIR := $(PROJECT_DIR)/results/$(VERSION_DIR)
ifeq ($(MODE),NORMAL)
TARGET_NAME ?= upvm_asr_$(VERSION_DIR)
else
TARGET_NAME ?= upvm_asr_$(VERSION_DIR)_$(MODE_DIR)
endif
TARGET := $(PROJECT_DIR)/$(TARGET_NAME)

HELPER_SRC_DIR := $(PROJECT_DIR)/sources/helpers
VERSION_SRC_DIR := $(PROJECT_DIR)/sources/$(VERSION_DIR)
MAIN_DIR := $(PROJECT_DIR)/main/$(VERSION_DIR)
MAIN_SRC := $(MAIN_DIR)/main.c

HELPER_SRCS := $(wildcard $(HELPER_SRC_DIR)/*.c)
VERSION_SRCS := $(shell find "$(VERSION_SRC_DIR)" -type f -name '*.c')
SRCS := $(HELPER_SRCS) $(VERSION_SRCS) $(MAIN_SRC)

OBJS := $(patsubst $(PROJECT_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

LOCAL_FFTW_PREFIX ?= $(REPO_ROOT)/fftw3/local
LOCAL_FFTW_HEADER := $(LOCAL_FFTW_PREFIX)/include/fftw3.h
LOCAL_FFTW_STATIC_LIB := $(LOCAL_FFTW_PREFIX)/lib/libfftw3f.a

ifeq ($(wildcard $(VERSION_SRC_DIR)),)
$(error VERSION='$(VERSION)' does not map to an existing source directory: $(VERSION_SRC_DIR))
endif

ifeq ($(wildcard $(PROJECT_DIR)/headers/$(VERSION_DIR)),)
$(error VERSION='$(VERSION)' does not map to an existing header directory: $(PROJECT_DIR)/headers/$(VERSION_DIR))
endif

ifeq ($(wildcard $(MAIN_SRC)),)
$(error VERSION='$(VERSION)' does not map to an existing main source: $(MAIN_SRC))
endif

ifeq ($(strip $(HELPER_SRCS)),)
$(error No helper sources found under $(HELPER_SRC_DIR))
endif

ifeq ($(strip $(VERSION_SRCS)),)
$(error No versioned sources found under $(VERSION_SRC_DIR))
endif

ifeq ($(filter $(MODE),NORMAL STAGE_TIMING DETAIL_STAGE_TIMING MEM_PROFILE),)
$(error MODE='$(MODE)' is invalid. Supported values: NORMAL, STAGE_TIMING, DETAIL_STAGE_TIMING, MEM_PROFILE)
endif

ifeq ($(wildcard $(LOCAL_FFTW_HEADER)),)
FFTW_CPPFLAGS ?=
FFTW_LDLIBS ?= -lfftw3f
else
FFTW_CPPFLAGS ?= -I$(LOCAL_FFTW_PREFIX)/include
ifeq ($(wildcard $(LOCAL_FFTW_STATIC_LIB)),)
FFTW_LDLIBS ?= -L$(LOCAL_FFTW_PREFIX)/lib -lfftw3f
else
FFTW_LDLIBS ?= $(LOCAL_FFTW_STATIC_LIB)
endif
endif

CPPFLAGS += -D_POSIX_C_SOURCE=199309L
CPPFLAGS += -D$(VERSION_DEFINE)=1
ifeq ($(MODE),STAGE_TIMING)
CPPFLAGS += -D_MODEL_STAGE_TIMING_
else ifeq ($(MODE),DETAIL_STAGE_TIMING)
CPPFLAGS += -D_MODEL_DETAIL_STAGE_TIMING_
else ifeq ($(MODE),MEM_PROFILE)
CPPFLAGS += -D_MODEL_MEM_PROFILE_
endif
CPPFLAGS += -I$(PROJECT_DIR)/headers/helpers
CPPFLAGS += -I$(PROJECT_DIR)/headers/$(VERSION_DIR)
CPPFLAGS += -I$(MAIN_DIR)
CPPFLAGS += $(FFTW_CPPFLAGS)

CSTD ?= -std=c11
OPT ?= -O3
WARN ?= -Wall -Wextra
DEPFLAGS := -MMD -MP
OPENMP_FLAGS :=
ifneq ($(filter $(VERSION_DIR),naives_mp sq_int8_v0 sq_int8_v1 sq_int8_v2 sq_int8_v3 sq_int8_v4 sq_int8_v5 sq_int8_v6 sq_int8_v7 sq_int8_v8 sq_int8_v9),)
OPENMP_FLAGS += -fopenmp
endif

CFLAGS += $(CSTD) $(OPT) $(WARN) $(DEPFLAGS) $(OPENMP_FLAGS)
LDFLAGS += $(OPENMP_FLAGS)
LDLIBS += $(FFTW_LDLIBS) -lm -pthread
ifeq ($(UNAME_S),Linux)
LDLIBS += -lrt
endif

.PHONY: all clean info prepare_dirs

all: prepare_dirs $(TARGET)

prepare_dirs:
	@mkdir -p $(BUILD_DIR) $(RESULT_DIR)

info:
	@echo "PROJECT_DIR  = $(PROJECT_DIR)"
	@echo "VERSION      = $(VERSION)"
	@echo "VERSION_DIR  = $(VERSION_DIR)"
	@echo "MODE         = $(MODE)"
	@echo "MODE_DIR     = $(MODE_DIR)"
	@echo "MAIN_DIR     = $(MAIN_DIR)"
	@echo "MAIN_SRC     = $(MAIN_SRC)"
	@echo "TARGET       = $(TARGET)"
	@echo "BUILD_DIR    = $(BUILD_DIR)"
	@echo "RESULT_DIR   = $(RESULT_DIR)"
	@echo "FFTW_HEADER  = $(LOCAL_FFTW_HEADER)"
	@echo "FFTW_LIB     = $(FFTW_LDLIBS)"

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/%.o: $(PROJECT_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Global clean: removes the whole build tree (every version/mode) and every
# upvm_asr_* binary at the project root, regardless of VERSION/MODE selectors.
# Paths are anchored to $(PROJECT_DIR); does NOT touch data/, results/ or weights.
clean:
	rm -rf $(BUILD_ROOT)
	rm -f $(PROJECT_DIR)/upvm_asr_*

-include $(DEPS)
