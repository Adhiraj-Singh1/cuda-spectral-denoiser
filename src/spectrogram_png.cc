// Copyright 2026 Adhiraj Singh

#include "src/spectrogram_png.h"

#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

namespace denoise {
namespace {

constexpr float kRangeDb = 80.0f;
constexpr int kSeparatorRows = 6;
constexpr int kTimeScale = 3;  // Each frame is drawn 3 pixels wide.
constexpr int kFreqStep = 2;   // Two frequency bins per pixel row (max).

// Dark-to-bright color map (black, purple, red, orange, pale yellow).
void ColorFor(float t, uint8_t* rgb) {
  static const float kStops[5][3] = {{0, 0, 4},
                                     {87, 16, 110},
                                     {188, 55, 84},
                                     {249, 142, 9},
                                     {252, 255, 164}};
  t = std::max(0.0f, std::min(1.0f, t)) * 4.0f;
  const int i = std::min(3, static_cast<int>(t));
  const float f = t - i;
  for (int c = 0; c < 3; ++c) {
    rgb[c] = static_cast<uint8_t>(kStops[i][c] +
                                  (kStops[i + 1][c] - kStops[i][c]) * f);
  }
}

}  // namespace

bool WriteSpectrogramPng(const std::string& path,
                         const std::vector<SpectrogramPanel>& panels) {
  if (panels.empty()) return false;
  float max_db = -1e9f;
  int max_frames = 0;
  int total_rows = 0;
  for (const SpectrogramPanel& p : panels) {
    max_frames = std::max(max_frames, p.frames);
    total_rows += (p.bins + kFreqStep - 1) / kFreqStep;
    for (int i = 0; i < p.frames * p.bins; ++i) {
      max_db = std::max(max_db, 20.0f * log10f(p.magnitudes[i] + 1e-9f));
    }
  }
  total_rows += kSeparatorRows * (static_cast<int>(panels.size()) - 1);
  const int width = max_frames * kTimeScale;
  std::vector<uint8_t> image(static_cast<size_t>(width) * total_rows * 3, 255);

  int row0 = 0;
  for (const SpectrogramPanel& p : panels) {
    const int rows = (p.bins + kFreqStep - 1) / kFreqStep;
    for (int r = 0; r < rows; ++r) {
      const int row = row0 + (rows - 1 - r);  // Low frequencies at bottom.
      for (int x = 0; x < width; ++x) {
        const int frame = x / kTimeScale;
        uint8_t* px = &image[(static_cast<size_t>(row) * width + x) * 3];
        if (frame >= p.frames) {
          px[0] = px[1] = px[2] = 0;
          continue;
        }
        float magnitude = 0.0f;
        for (int k = r * kFreqStep; k < (r + 1) * kFreqStep && k < p.bins;
             ++k) {
          magnitude = std::max(
              magnitude, p.magnitudes[static_cast<size_t>(frame) * p.bins + k]);
        }
        const float db = 20.0f * log10f(magnitude + 1e-9f);
        ColorFor((db - (max_db - kRangeDb)) / kRangeDb, px);
      }
    }
    row0 += rows + kSeparatorRows;
  }
  return stbi_write_png(path.c_str(), width, total_rows, 3, image.data(),
                        width * 3) != 0;
}

}  // namespace denoise
