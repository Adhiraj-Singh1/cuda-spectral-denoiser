# Build file for the CUDA spectral denoiser.
#
#   make            build bin/denoise and bin/generate_audio_dataset
#   make data       generate the synthetic dataset in data/
#   make run        build and run with ARGS="..." (see ./bin/denoise -h)
#   make clean      remove build products
#   make clean-all  also remove generated data, outputs and logs
#
# Set ARCH to target a specific GPU, e.g. make ARCH="-arch=sm_75".

NVCC ?= nvcc
CXX ?= g++
ARCH ?=

INCLUDES := -I.
NVCCFLAGS := -std=c++17 -O2 -Wno-deprecated-gpu-targets $(INCLUDES) $(ARCH)
CXXFLAGS := -std=c++17 -O2 -Wall $(INCLUDES)
LIBS := -lcufft

BIN_DIR := bin
OBJ_DIR := obj
TARGET := $(BIN_DIR)/denoise
GENERATOR := $(BIN_DIR)/generate_audio_dataset

CU_SRCS := src/main.cu src/denoise_pipeline.cu
CC_SRCS := src/wav_io.cc src/cpu_reference.cc src/cli_options.cc \
           src/spectrogram_png.cc
HEADERS := $(wildcard src/*.h src/*.cuh)
OBJS := $(CU_SRCS:src/%.cu=$(OBJ_DIR)/%.cu.o) \
        $(CC_SRCS:src/%.cc=$(OBJ_DIR)/%.cc.o)

.PHONY: all build data run clean clean-all

all: build
build: $(TARGET) $(GENERATOR)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(LIBS)

$(OBJ_DIR)/%.cu.o: src/%.cu $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cc.o: src/%.cc $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(GENERATOR): tools/generate_audio_dataset.cc $(OBJ_DIR)/wav_io.cc.o \
              | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJ_DIR)/wav_io.cc.o -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

# 240 clips of 1.5-3 s at 16 kHz, white or pink noise at 0-10 dB SNR.
data: $(GENERATOR)
	./$(GENERATOR) -o data -c 240 -s 2026

run: build
	./$(TARGET) $(ARGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

clean-all: clean
	rm -rf data output logs/run_log.txt logs/summary.md logs/results.csv
