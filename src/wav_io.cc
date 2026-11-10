// Copyright 2026 Adhiraj Singh

#include "src/wav_io.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace denoise {
namespace {

uint32_t ReadU32(const uint8_t* p) {
  return p[0] | (p[1] << 8) | (p[2] << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t ReadU16(const uint8_t* p) { return p[0] | (p[1] << 8); }

void WriteU32(uint32_t v, FILE* f) {
  const uint8_t b[4] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8),
                        static_cast<uint8_t>(v >> 16),
                        static_cast<uint8_t>(v >> 24)};
  fwrite(b, 1, 4, f);
}

void WriteU16(uint16_t v, FILE* f) {
  const uint8_t b[2] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8)};
  fwrite(b, 1, 2, f);
}

// Decodes one sample of the given format to a float in [-1, 1].
float DecodeSample(const uint8_t* p, int format, int bits) {
  if (format == 3 && bits == 32) {
    float v;
    memcpy(&v, p, sizeof(v));
    return v;
  }
  switch (bits) {
    case 8:
      return (static_cast<int>(p[0]) - 128) / 128.0f;
    case 16:
      return static_cast<int16_t>(ReadU16(p)) / 32768.0f;
    case 24: {
      int32_t v = p[0] | (p[1] << 8) | (p[2] << 16);
      if (v & 0x800000) v |= ~0xFFFFFF;
      return v / 8388608.0f;
    }
    case 32:
      return static_cast<int32_t>(ReadU32(p)) / 2147483648.0f;
    default:
      return 0.0f;
  }
}

}  // namespace

bool ReadWav(const std::string& path, AudioClip* clip, std::string* error) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) {
    *error = "cannot open file";
    return false;
  }
  std::vector<uint8_t> data;
  uint8_t buffer[65536];
  size_t n = 0;
  while ((n = fread(buffer, 1, sizeof(buffer), f)) > 0) {
    data.insert(data.end(), buffer, buffer + n);
  }
  fclose(f);

  if (data.size() < 12 || memcmp(data.data(), "RIFF", 4) != 0 ||
      memcmp(data.data() + 8, "WAVE", 4) != 0) {
    *error = "not a RIFF/WAVE file";
    return false;
  }
  int format = 0, channels = 0, bits = 0, rate = 0;
  const uint8_t* samples = nullptr;
  size_t sample_bytes = 0;
  size_t pos = 12;
  while (pos + 8 <= data.size()) {
    const uint8_t* chunk = data.data() + pos;
    const uint32_t size = ReadU32(chunk + 4);
    const size_t body = pos + 8;
    if (body + size > data.size() && memcmp(chunk, "data", 4) != 0) break;
    if (memcmp(chunk, "fmt ", 4) == 0 && size >= 16) {
      format = ReadU16(chunk + 8);
      channels = ReadU16(chunk + 10);
      rate = static_cast<int>(ReadU32(chunk + 12));
      bits = ReadU16(chunk + 22);
      if (format == 0xFFFE && size >= 40) format = ReadU16(chunk + 32);
    } else if (memcmp(chunk, "data", 4) == 0) {
      samples = chunk + 8;
      sample_bytes = std::min<size_t>(size, data.size() - body);
    }
    pos = body + size + (size & 1);
  }
  if (samples == nullptr || channels <= 0 || rate <= 0 ||
      (format != 1 && format != 3) ||
      (bits != 8 && bits != 16 && bits != 24 && bits != 32)) {
    *error = "unsupported or malformed WAV format";
    return false;
  }
  const int bytes_per_sample = bits / 8;
  const size_t frames = sample_bytes / (bytes_per_sample * channels);
  clip->sample_rate = rate;
  clip->samples.assign(frames, 0.0f);
  for (size_t i = 0; i < frames; ++i) {
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
      sum += DecodeSample(samples + (i * channels + c) * bytes_per_sample,
                          format, bits);
    }
    clip->samples[i] = sum / channels;
  }
  return true;
}

bool WriteWav16(const std::string& path, const std::vector<float>& samples,
                int sample_rate) {
  FILE* f = fopen(path.c_str(), "wb");
  if (f == nullptr) return false;
  const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * 2);
  fwrite("RIFF", 1, 4, f);
  WriteU32(36 + data_bytes, f);
  fwrite("WAVEfmt ", 1, 8, f);
  WriteU32(16, f);
  WriteU16(1, f);  // PCM
  WriteU16(1, f);  // mono
  WriteU32(sample_rate, f);
  WriteU32(sample_rate * 2, f);
  WriteU16(2, f);
  WriteU16(16, f);
  fwrite("data", 1, 4, f);
  WriteU32(data_bytes, f);
  for (float s : samples) {
    float v = s * 32767.0f;
    v = std::max(-32768.0f, std::min(32767.0f, v));
    WriteU16(static_cast<uint16_t>(static_cast<int16_t>(lrintf(v))), f);
  }
  const bool ok = ferror(f) == 0;
  fclose(f);
  return ok;
}

std::vector<std::string> ListWavFiles(const std::string& directory) {
  std::vector<std::string> files;
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) return files;
  while (struct dirent* entry = readdir(dir)) {
    std::string name = entry->d_name;
    if (name.size() < 5 || name[0] == '.') continue;
    std::string ext = name.substr(name.size() - 4);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".wav") files.push_back(directory + "/" + name);
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

bool MakeDirectories(const std::string& directory) {
  for (size_t pos = directory.find('/', 1); pos != std::string::npos;
       pos = directory.find('/', pos + 1)) {
    if (mkdir(directory.substr(0, pos).c_str(), 0755) != 0 &&
        errno != EEXIST) {
      return false;
    }
  }
  return mkdir(directory.c_str(), 0755) == 0 || errno == EEXIST;
}

std::string FileName(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace denoise
