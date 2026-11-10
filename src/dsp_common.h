// Copyright 2026 Adhiraj Singh
//
// Signal-processing definitions shared by the CUDA kernels and the CPU
// reference: STFT framing geometry, the analysis/synthesis window and the
// spectral-subtraction gain. Keeping them in one header guarantees that the
// GPU and the CPU implement exactly the same algorithm.
//
// STFT geometry (N = frame size, H = hop, P = N - H):
//   frame f of a clip covers samples [f * H - P, f * H - P + N)
//   samples outside [0, length) are treated as zeros
//   number of frames = ceil((length + P) / H), so every sample of the clip is
//   covered by N / H frames, which keeps the overlap-add normalization stable.

#ifndef SRC_DSP_COMMON_H_
#define SRC_DSP_COMMON_H_

#include <math.h>

#ifdef __CUDACC__
#define DSP_HOST_DEVICE __host__ __device__
#else
#define DSP_HOST_DEVICE
#endif

namespace denoise {

struct DenoiseParams {
  int frame_size = 1024;          // N, a power of two.
  int hop = 256;                  // H, at most N / 2.
  float noise_ms = 250.0f;        // Noise-only lead-in used for the profile.
  float over_subtraction = 1.5f;  // alpha in the gain formula.
  float gain_floor = 0.05f;       // Lowest gain applied to any bin.
};

// Number of rfft bins for a real frame of size n.
DSP_HOST_DEVICE inline int NumBins(int frame_size) {
  return frame_size / 2 + 1;
}

DSP_HOST_DEVICE inline int NumFrames(int length, int frame_size, int hop) {
  return (length + frame_size - 1) / hop;
}

DSP_HOST_DEVICE inline int FrameStart(int frame, int frame_size, int hop) {
  return frame * hop - (frame_size - hop);
}

// Periodic Hann window, used for both analysis and synthesis.
DSP_HOST_DEVICE inline float HannWindow(int n, int frame_size) {
  return 0.5f - 0.5f * cosf(6.283185307179586f * n / frame_size);
}

// Frames that lie completely inside the noise-only lead-in of a clip. Their
// average magnitude spectrum is the noise profile of that clip.
DSP_HOST_DEVICE inline void NoiseFrameRange(int length, int noise_samples,
                                            int frame_size, int hop,
                                            int* first, int* count) {
  const int pad = frame_size - hop;
  const int num_frames = NumFrames(length, frame_size, hop);
  const int lead_in = noise_samples < length ? noise_samples : length;
  *first = (pad + hop - 1) / hop;  // First frame with start >= 0.
  const int span = lead_in - frame_size + pad;
  int last = span >= 0 ? span / hop : -1;  // Last frame with end <= lead_in.
  if (last > num_frames - 1) last = num_frames - 1;
  *count = last >= *first ? last - *first + 1 : 0;
}

// Frames whose span contains sample t: frames first..last (inclusive).
DSP_HOST_DEVICE inline void FramesCoveringSample(int t, int num_frames,
                                                 int frame_size, int hop,
                                                 int* first, int* last) {
  *first = t / hop;
  *last = (t + frame_size - hop) / hop;
  if (*last > num_frames - 1) *last = num_frames - 1;
}

// Magnitude spectral subtraction: G = max(1 - alpha * noise / |X|, floor).
DSP_HOST_DEVICE inline float SpectralGain(float magnitude, float noise,
                                          float over_subtraction,
                                          float gain_floor) {
  const float gain = 1.0f - over_subtraction * noise / (magnitude + 1e-12f);
  return gain < gain_floor ? gain_floor : gain;
}

}  // namespace denoise

#endif  // SRC_DSP_COMMON_H_
