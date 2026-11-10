// Copyright 2026 Adhiraj Singh
//
// GPU denoising pipeline. A "chunk" of many clips is packed into one device
// buffer and processed with a fixed sequence of batched operations:
//
//   1. FrameAndWindowKernel   split every clip into overlapping Hann frames
//   2. cufftExecR2C           batched real-to-complex FFT of all frames
//   3. NoiseProfileKernel     per-clip noise spectrum from the lead-in frames
//   4. SpectralGainKernel     spectral subtraction gain on every bin
//   5. cufftExecC2R           batched inverse FFT of all frames
//   6. OverlapAddKernel       weighted overlap-add back into clips
//   7. SnrReductionKernel     per-clip signal/error energies (shared-memory
//                             reduction) for the input and output SNR
//
// Every stage is timed with CUDA events.

#ifndef SRC_DENOISE_PIPELINE_CUH_
#define SRC_DENOISE_PIPELINE_CUH_

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "src/dsp_common.h"

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t check_cuda_status = (call);                                \
    if (check_cuda_status != cudaSuccess) {                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
              cudaGetErrorString(check_cuda_status), __FILE__, __LINE__);  \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CHECK_CUFFT(call)                                                  \
  do {                                                                     \
    cufftResult check_cufft_status = (call);                               \
    if (check_cufft_status != CUFFT_SUCCESS) {                             \
      fprintf(stderr, "cuFFT error %d at %s:%d\n",                         \
              static_cast<int>(check_cufft_status), __FILE__, __LINE__);   \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

namespace denoise {

// One audio clip travelling through the pipeline.
struct ClipJob {
  std::string name;
  int sample_rate = 0;
  std::vector<float> noisy;
  std::vector<float> clean;      // Empty when no clean reference exists.
  std::vector<float> denoised;   // Filled by DenoisePipeline::Process.
  bool keep_magnitudes = false;  // Copy spectra back for the gallery.
  std::vector<float> magnitude_in;   // num_frames x bins, before the gain.
  std::vector<float> magnitude_out;  // num_frames x bins, after the gain.
  int num_frames = 0;
  int noise_frames = 0;
  // Energies computed on the GPU (only valid when clean is non-empty).
  double clean_energy = 0.0;
  double input_error_energy = 0.0;
  double output_error_energy = 0.0;
};

// Accumulated time (ms) per pipeline stage, measured with CUDA events.
struct StageTimes {
  double upload = 0.0;
  double framing = 0.0;
  double forward_fft = 0.0;
  double noise_and_gain = 0.0;
  double inverse_fft = 0.0;
  double overlap_add = 0.0;
  double snr = 0.0;
  double download = 0.0;
  double Total() const {
    return upload + framing + forward_fft + noise_and_gain + inverse_fft +
           overlap_add + snr + download;
  }
};

// Device-side description of a clip inside a chunk.
struct ClipInfo {
  int64_t sample_offset;  // Offset into the packed signal buffers.
  int length;             // Samples.
  int first_frame;        // Offset into the packed frame buffers.
  int num_frames;
  int noise_first;        // First noise frame (relative to first_frame).
  int noise_count;
  int has_clean;
};

class DenoisePipeline {
 public:
  // `capacity_frames` bounds the number of STFT frames in one chunk; it
  // determines the device memory used (about 12 KB per frame for N = 1024).
  DenoisePipeline(const DenoiseParams& params, int capacity_frames);
  ~DenoisePipeline();

  DenoisePipeline(const DenoisePipeline&) = delete;
  DenoisePipeline& operator=(const DenoisePipeline&) = delete;

  int capacity_frames() const { return capacity_frames_; }

  // Frames a clip of `length` samples needs.
  int FramesFor(int length) const {
    return NumFrames(length, params_.frame_size, params_.hop);
  }

  // Denoises all `clips` (their total frame count must not exceed
  // capacity_frames()) and fills denoised, energies and magnitudes.
  void Process(const std::vector<ClipJob*>& clips, StageTimes* times);

 private:
  void Allocate();
  void Free();

  DenoiseParams params_;
  int capacity_frames_;
  int64_t capacity_samples_;
  int bins_;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t events_[9] = {};

  // Pinned host staging buffers.
  float* host_signal_ = nullptr;
  float* host_clean_ = nullptr;
  float* host_output_ = nullptr;
  ClipInfo* host_clips_ = nullptr;
  int* host_frame_clip_ = nullptr;
  double* host_energies_ = nullptr;

  // Device buffers.
  float* dev_signal_ = nullptr;
  float* dev_clean_ = nullptr;
  float* dev_output_ = nullptr;
  float* dev_frames_ = nullptr;        // capacity_frames x N (real)
  cufftComplex* dev_spectra_ = nullptr;  // capacity_frames x bins
  float* dev_noise_ = nullptr;         // capacity_frames x bins (per clip)
  float* dev_magnitude_in_ = nullptr;  // capacity_frames x bins
  float* dev_magnitude_out_ = nullptr;
  ClipInfo* dev_clips_ = nullptr;
  int* dev_frame_clip_ = nullptr;
  double* dev_energies_ = nullptr;     // 3 per clip
};

}  // namespace denoise

#endif  // SRC_DENOISE_PIPELINE_CUH_
