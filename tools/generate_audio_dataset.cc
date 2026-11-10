// Copyright 2026 Adhiraj Singh
//
// generate_audio_dataset: writes a reproducible set of clean and noisy audio
// clips (16-bit mono WAV) plus a CSV describing them.
//
// Each clean clip is a short "melody": a sequence of harmonic notes with
// attack/decay envelopes and vibrato, starting after a silent lead-in. Each
// noisy clip adds white or pink noise at a random SNR. The lead-in contains
// only noise, which is what the denoiser uses to learn the noise spectrum.
//
// Usage: generate_audio_dataset [-o DIR] [-c COUNT] [-s SEED]
//            [-r SAMPLE_RATE] [--min_seconds F] [--max_seconds F]
//            [--min_snr F] [--max_snr F]

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "src/wav_io.h"

namespace {

constexpr double kTwoPi = 6.283185307179586;
constexpr double kLeadInSeconds = 0.35;

struct Settings {
  std::string output_dir = "data";
  int count = 240;
  unsigned int seed = 2026;
  int sample_rate = 16000;
  double min_seconds = 1.5;
  double max_seconds = 3.0;
  double min_snr = 0.0;
  double max_snr = 10.0;
};

std::vector<float> MakeMelody(int length, int sample_rate, std::mt19937* rng) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<double> x(length, 0.0);
  int pos = static_cast<int>(kLeadInSeconds * sample_rate);
  const int harmonics = 3 + static_cast<int>(unit(*rng) * 4);
  const double brightness = 0.45 + 0.3 * unit(*rng);
  while (pos < length - sample_rate / 20) {
    const int duration =
        static_cast<int>((0.12 + 0.38 * unit(*rng)) * sample_rate);
    const int semitone = static_cast<int>(unit(*rng) * 36);
    const double f0 = 110.0 * pow(2.0, semitone / 12.0);
    const double decay = 1.0 + 4.0 * unit(*rng);
    const double vibrato_rate = 4.0 + 3.0 * unit(*rng);
    for (int i = 0; i < duration && pos + i < length; ++i) {
      const double t = static_cast<double>(i) / sample_rate;
      const double envelope = std::min(1.0, t / 0.02) * exp(-decay * t);
      const double vibrato = 1.0 + 0.003 * sin(kTwoPi * vibrato_rate * t);
      double s = 0.0, amplitude = 1.0;
      for (int h = 1; h <= harmonics; ++h) {
        const double f = f0 * h * vibrato;
        if (f < 0.45 * sample_rate) s += amplitude * sin(kTwoPi * f * t);
        amplitude *= brightness;
      }
      x[pos + i] += envelope * s;
    }
    pos += static_cast<int>(duration * (0.6 + 0.4 * unit(*rng)));
  }
  double peak = 1e-9;
  for (double v : x) peak = std::max(peak, fabs(v));
  std::vector<float> out(length);
  for (int i = 0; i < length; ++i) {
    out[i] = static_cast<float>(0.5 * x[i] / peak);
  }
  return out;
}

// Pink noise with Paul Kellet's economy filter (about -3 dB per octave).
std::vector<float> MakeNoise(int length, bool pink, std::mt19937* rng) {
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> noise(length);
  float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
  for (int i = 0; i < length; ++i) {
    const float white = gauss(*rng);
    if (pink) {
      b0 = 0.99765f * b0 + white * 0.0990460f;
      b1 = 0.96300f * b1 + white * 0.2965164f;
      b2 = 0.57000f * b2 + white * 1.0526913f;
      noise[i] = b0 + b1 + b2 + white * 0.1848f;
    } else {
      noise[i] = white;
    }
  }
  return noise;
}

double Energy(const std::vector<float>& v) {
  double sum = 0.0;
  for (float x : v) sum += static_cast<double>(x) * x;
  return sum;
}

bool ParseArgs(int argc, char** argv, Settings* s) {
  static const struct option kLongOptions[] = {
      {"min_seconds", required_argument, nullptr, 1},
      {"max_seconds", required_argument, nullptr, 2},
      {"min_snr", required_argument, nullptr, 3},
      {"max_snr", required_argument, nullptr, 4},
      {nullptr, 0, nullptr, 0}};
  int opt = 0;
  while ((opt = getopt_long(argc, argv, "o:c:s:r:h", kLongOptions,
                            nullptr)) != -1) {
    switch (opt) {
      case 'o': s->output_dir = optarg; break;
      case 'c': s->count = atoi(optarg); break;
      case 's': s->seed = static_cast<unsigned int>(strtoul(optarg, nullptr,
                                                            10));
                break;
      case 'r': s->sample_rate = atoi(optarg); break;
      case 1: s->min_seconds = atof(optarg); break;
      case 2: s->max_seconds = atof(optarg); break;
      case 3: s->min_snr = atof(optarg); break;
      case 4: s->max_snr = atof(optarg); break;
      default: return false;
    }
  }
  return s->count > 0 && s->sample_rate >= 8000 &&
         s->min_seconds > kLeadInSeconds + 0.2 &&
         s->max_seconds >= s->min_seconds && s->max_snr >= s->min_snr;
}

}  // namespace

int main(int argc, char** argv) {
  Settings settings;
  if (!ParseArgs(argc, argv, &settings)) {
    fprintf(stderr,
            "Usage: %s [-o DIR] [-c COUNT] [-s SEED] [-r SAMPLE_RATE] "
            "[--min_seconds F] [--max_seconds F] [--min_snr F] "
            "[--max_snr F]\n",
            argv[0]);
    return 2;
  }
  const std::string clean_dir = settings.output_dir + "/clean";
  const std::string noisy_dir = settings.output_dir + "/noisy";
  if (!denoise::MakeDirectories(clean_dir) ||
      !denoise::MakeDirectories(noisy_dir)) {
    fprintf(stderr, "Cannot create %s\n", settings.output_dir.c_str());
    return 1;
  }
  FILE* csv = fopen((settings.output_dir + "/dataset.csv").c_str(), "w");
  if (csv != nullptr) fprintf(csv, "clip,seconds,noise_type,target_snr_db\n");

  std::mt19937 rng(settings.seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  double total_seconds = 0.0;
  for (int i = 0; i < settings.count; ++i) {
    const double seconds =
        settings.min_seconds +
        (settings.max_seconds - settings.min_seconds) * unit(rng);
    const int length = static_cast<int>(seconds * settings.sample_rate);
    std::vector<float> clean = MakeMelody(length, settings.sample_rate, &rng);
    const bool pink = unit(rng) < 0.5;
    std::vector<float> noise = MakeNoise(length, pink, &rng);
    const double snr_db =
        settings.min_snr + (settings.max_snr - settings.min_snr) * unit(rng);
    const double scale =
        sqrt(Energy(clean) / Energy(noise) / pow(10.0, snr_db / 10.0));

    std::vector<float> noisy(length);
    float peak = 1e-9f;
    for (int t = 0; t < length; ++t) {
      noisy[t] = clean[t] + static_cast<float>(scale) * noise[t];
      peak = std::max(peak, fabsf(noisy[t]));
    }
    // Scale both versions together so the noisy file never clips; this keeps
    // the SNR unchanged.
    const float gain = std::min(1.0f, 0.95f / peak);
    for (int t = 0; t < length; ++t) {
      clean[t] *= gain;
      noisy[t] *= gain;
    }

    char name[64];
    snprintf(name, sizeof(name), "clip_%04d.wav", i);
    if (!denoise::WriteWav16(clean_dir + "/" + name, clean,
                             settings.sample_rate) ||
        !denoise::WriteWav16(noisy_dir + "/" + name, noisy,
                             settings.sample_rate)) {
      fprintf(stderr, "Failed to write %s\n", name);
      return 1;
    }
    if (csv != nullptr) {
      fprintf(csv, "%s,%.3f,%s,%.2f\n", name, seconds, pink ? "pink" : "white",
              snr_db);
    }
    total_seconds += seconds;
  }
  if (csv != nullptr) fclose(csv);
  printf("Wrote %d clean/noisy clip pairs (%.1f s of audio, %d Hz) to %s\n",
         settings.count, total_seconds, settings.sample_rate,
         settings.output_dir.c_str());
  return 0;
}
