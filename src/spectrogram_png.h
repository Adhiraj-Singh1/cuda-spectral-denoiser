// Copyright 2026 Adhiraj Singh
//
// Renders magnitude spectrograms as a color PNG: several panels (e.g. clean,
// noisy, denoised) stacked vertically, time on the x axis, frequency on the
// y axis (low frequencies at the bottom), log-magnitude color scale.

#ifndef SRC_SPECTROGRAM_PNG_H_
#define SRC_SPECTROGRAM_PNG_H_

#include <string>
#include <vector>

namespace denoise {

struct SpectrogramPanel {
  const float* magnitudes = nullptr;  // frames x bins, row-major by frame.
  int frames = 0;
  int bins = 0;
};

// All panels share one dB scale (max over all panels, 80 dB range) so they
// can be compared directly.
bool WriteSpectrogramPng(const std::string& path,
                         const std::vector<SpectrogramPanel>& panels);

}  // namespace denoise

#endif  // SRC_SPECTROGRAM_PNG_H_
