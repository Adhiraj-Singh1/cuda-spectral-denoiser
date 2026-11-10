// Copyright 2026 Adhiraj Singh
//
// Single-threaded CPU implementation of the denoiser (with its own radix-2
// FFT). Used to verify the GPU output sample by sample, to measure the GPU
// speedup, and to draw the clean-signal spectrogram for the gallery.

#ifndef SRC_CPU_REFERENCE_H_
#define SRC_CPU_REFERENCE_H_

#include <complex>
#include <vector>

#include "src/dsp_common.h"

namespace denoise {

// In-place iterative radix-2 FFT. `inverse` computes the unnormalized
// inverse transform (like cuFFT). data.size() must be a power of two.
void Fft(std::vector<std::complex<float>>* data, bool inverse);

// Magnitude STFT of a clip: num_frames x NumBins(frame_size) values.
void ComputeMagnitudes(const float* signal, int length,
                       const DenoiseParams& params, int sample_rate,
                       std::vector<float>* magnitudes, int* num_frames);

// Runs the complete denoiser on one clip.
void DenoiseClipCpu(const float* signal, int length,
                    const DenoiseParams& params, int sample_rate,
                    std::vector<float>* output);

// 10 * log10(sum(clean^2) / sum((estimate - clean)^2)).
double SnrDb(const float* clean, const float* estimate, int length);

}  // namespace denoise

#endif  // SRC_CPU_REFERENCE_H_
