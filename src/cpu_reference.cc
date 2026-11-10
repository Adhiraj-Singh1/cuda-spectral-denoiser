// Copyright 2026 Adhiraj Singh

#include "src/cpu_reference.h"

#include <math.h>

#include <complex>
#include <utility>
#include <vector>

#include "src/dsp_common.h"

namespace denoise {
namespace {

using Complex = std::complex<float>;

// Builds the windowed frames of a clip and returns their spectra
// (num_frames x bins).
void Stft(const float* signal, int length, const DenoiseParams& params,
          std::vector<Complex>* spectra, int* num_frames) {
  const int n = params.frame_size;
  const int bins = NumBins(n);
  *num_frames = NumFrames(length, n, params.hop);
  spectra->assign(static_cast<size_t>(*num_frames) * bins, Complex());
  std::vector<Complex> frame(n);
  for (int f = 0; f < *num_frames; ++f) {
    const int start = FrameStart(f, n, params.hop);
    for (int i = 0; i < n; ++i) {
      const int t = start + i;
      const float sample = (t >= 0 && t < length) ? signal[t] : 0.0f;
      frame[i] = Complex(sample * HannWindow(i, n), 0.0f);
    }
    Fft(&frame, false);
    for (int k = 0; k < bins; ++k) {
      (*spectra)[static_cast<size_t>(f) * bins + k] = frame[k];
    }
  }
}

}  // namespace

void Fft(std::vector<std::complex<float>>* data, bool inverse) {
  std::vector<Complex>& a = *data;
  const int n = static_cast<int>(a.size());
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  // Twiddle factors exp(-2*pi*i*k/n), computed once per FFT size.
  static thread_local std::vector<Complex> twiddles;
  if (static_cast<int>(twiddles.size()) != n / 2) {
    twiddles.resize(n / 2);
    for (int k = 0; k < n / 2; ++k) {
      const double angle = -2.0 * M_PI * k / n;
      twiddles[k] = Complex(static_cast<float>(cos(angle)),
                            static_cast<float>(sin(angle)));
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    const int stride = n / len;
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < len / 2; ++k) {
        const Complex t = twiddles[k * stride];
        const Complex w = inverse ? std::conj(t) : t;
        const Complex u = a[i + k];
        const Complex v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
      }
    }
  }
}

void ComputeMagnitudes(const float* signal, int length,
                       const DenoiseParams& params, int sample_rate,
                       std::vector<float>* magnitudes, int* num_frames) {
  (void)sample_rate;
  std::vector<Complex> spectra;
  Stft(signal, length, params, &spectra, num_frames);
  magnitudes->resize(spectra.size());
  for (size_t i = 0; i < spectra.size(); ++i) {
    (*magnitudes)[i] = std::abs(spectra[i]);
  }
}

void DenoiseClipCpu(const float* signal, int length,
                    const DenoiseParams& params, int sample_rate,
                    std::vector<float>* output) {
  const int n = params.frame_size;
  const int hop = params.hop;
  const int bins = NumBins(n);
  int num_frames = 0;
  std::vector<Complex> spectra;
  Stft(signal, length, params, &spectra, &num_frames);

  // Noise profile: mean magnitude over the frames of the lead-in.
  int noise_first = 0, noise_count = 0;
  const int noise_samples =
      static_cast<int>(params.noise_ms * 0.001f * sample_rate);
  NoiseFrameRange(length, noise_samples, n, hop, &noise_first, &noise_count);
  std::vector<float> noise(bins, 0.0f);
  for (int k = 0; k < bins; ++k) {
    float sum = 0.0f;
    for (int f = noise_first; f < noise_first + noise_count; ++f) {
      sum += std::abs(spectra[static_cast<size_t>(f) * bins + k]);
    }
    noise[k] = noise_count > 0 ? sum / noise_count : 0.0f;
  }

  // Spectral gain, inverse FFT and weighted overlap-add.
  std::vector<float> frames(static_cast<size_t>(num_frames) * n);
  std::vector<Complex> full(n);
  for (int f = 0; f < num_frames; ++f) {
    const Complex* x = &spectra[static_cast<size_t>(f) * bins];
    for (int k = 0; k < bins; ++k) {
      const float gain = SpectralGain(std::abs(x[k]), noise[k],
                                      params.over_subtraction,
                                      params.gain_floor);
      full[k] = x[k] * gain;
    }
    for (int k = bins; k < n; ++k) full[k] = std::conj(full[n - k]);
    Fft(&full, true);
    for (int i = 0; i < n; ++i) {
      frames[static_cast<size_t>(f) * n + i] = full[i].real();
    }
  }

  output->assign(length, 0.0f);
  for (int t = 0; t < length; ++t) {
    int first = 0, last = 0;
    FramesCoveringSample(t, num_frames, n, hop, &first, &last);
    float sum = 0.0f, weight = 0.0f;
    for (int f = first; f <= last; ++f) {
      const int i = t - FrameStart(f, n, hop);
      const float w = HannWindow(i, n);
      sum += frames[static_cast<size_t>(f) * n + i] * w / n;
      weight += w * w;
    }
    (*output)[t] = weight > 1e-8f ? sum / weight : 0.0f;
  }
}

double SnrDb(const float* clean, const float* estimate, int length) {
  double signal = 0.0, error = 0.0;
  for (int i = 0; i < length; ++i) {
    signal += static_cast<double>(clean[i]) * clean[i];
    const double d = static_cast<double>(estimate[i]) - clean[i];
    error += d * d;
  }
  if (error <= 0.0) return 200.0;
  return 10.0 * log10(signal / error);
}

}  // namespace denoise
