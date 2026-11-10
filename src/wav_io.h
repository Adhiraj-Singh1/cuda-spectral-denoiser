// Copyright 2026 Adhiraj Singh
//
// Minimal WAV file reading and writing (no external dependencies).
// Reads 8/16/24/32-bit PCM and 32-bit float WAV files (multi-channel files
// are mixed down to mono); writes 16-bit PCM mono.

#ifndef SRC_WAV_IO_H_
#define SRC_WAV_IO_H_

#include <string>
#include <vector>

namespace denoise {

struct AudioClip {
  int sample_rate = 0;
  std::vector<float> samples;  // Mono, nominally in [-1, 1].
};

bool ReadWav(const std::string& path, AudioClip* clip, std::string* error);
bool WriteWav16(const std::string& path, const std::vector<float>& samples,
                int sample_rate);

// Sorted paths of all .wav files in `directory`.
std::vector<std::string> ListWavFiles(const std::string& directory);

// mkdir -p.
bool MakeDirectories(const std::string& directory);

// "data/noisy/clip_0001.wav" -> "clip_0001.wav".
std::string FileName(const std::string& path);

}  // namespace denoise

#endif  // SRC_WAV_IO_H_
