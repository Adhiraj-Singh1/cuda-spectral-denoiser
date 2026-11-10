// Copyright 2026 Adhiraj Singh

#include "src/cli_options.h"

#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

namespace denoise {
namespace {

bool ParseInt(const char* text, int* value) {
  char* end = nullptr;
  const int64_t parsed = strtoll(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseFloat(const char* text, float* value) {
  char* end = nullptr;
  const float parsed = strtof(text, &end);
  if (end == text || *end != '\0') return false;
  *value = parsed;
  return true;
}

bool IsPowerOfTwo(int v) { return v > 0 && (v & (v - 1)) == 0; }

}  // namespace

bool ParseOptions(int argc, char** argv, Options* options, bool* show_help,
                  std::string* error) {
  static const struct option kLongOptions[] = {
      {"noisy_dir", required_argument, nullptr, 'i'},
      {"clean_dir", required_argument, nullptr, 'c'},
      {"output_dir", required_argument, nullptr, 'o'},
      {"log_file", required_argument, nullptr, 'l'},
      {"summary_file", required_argument, nullptr, 'S'},
      {"results_csv", required_argument, nullptr, 'r'},
      {"frame_size", required_argument, nullptr, 'f'},
      {"hop", required_argument, nullptr, 'H'},
      {"noise_ms", required_argument, nullptr, 'N'},
      {"over_subtraction", required_argument, nullptr, 'a'},
      {"gain_floor", required_argument, nullptr, 'g'},
      {"batch_frames", required_argument, nullptr, 'b'},
      {"max_clips", required_argument, nullptr, 'm'},
      {"gallery", required_argument, nullptr, 'G'},
      {"device", required_argument, nullptr, 'd'},
      {"verify", no_argument, nullptr, 'v'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  *show_help = false;
  int hop = 0;  // 0 = frame_size / 4.
  optind = 1;
  int opt = 0;
  while ((opt = getopt_long(argc, argv, "i:c:o:l:S:r:f:H:N:a:g:b:m:G:d:vh",
                            kLongOptions, nullptr)) != -1) {
    bool ok = true;
    DenoiseParams& p = options->params;
    switch (opt) {
      case 'i': options->noisy_dir = optarg; break;
      case 'c': options->clean_dir = optarg; break;
      case 'o': options->output_dir = optarg; break;
      case 'l': options->log_file = optarg; break;
      case 'S': options->summary_file = optarg; break;
      case 'r': options->results_csv = optarg; break;
      case 'f': ok = ParseInt(optarg, &p.frame_size); break;
      case 'H': ok = ParseInt(optarg, &hop); break;
      case 'N': ok = ParseFloat(optarg, &p.noise_ms); break;
      case 'a': ok = ParseFloat(optarg, &p.over_subtraction); break;
      case 'g': ok = ParseFloat(optarg, &p.gain_floor); break;
      case 'b': ok = ParseInt(optarg, &options->batch_frames); break;
      case 'm': ok = ParseInt(optarg, &options->max_clips); break;
      case 'G': ok = ParseInt(optarg, &options->gallery_count); break;
      case 'd': ok = ParseInt(optarg, &options->device); break;
      case 'v': options->verify = true; break;
      case 'h': *show_help = true; return true;
      default:
        *error = "unknown option";
        return false;
    }
    if (!ok) {
      *error = std::string("invalid value '") + optarg + "' for -" +
               static_cast<char>(opt);
      return false;
    }
  }

  DenoiseParams& p = options->params;
  p.hop = hop > 0 ? hop : p.frame_size / 4;
  if (!IsPowerOfTwo(p.frame_size) || p.frame_size < 64 ||
      p.frame_size > 8192) {
    *error = "--frame_size must be a power of two between 64 and 8192";
  } else if (p.hop < 1 || p.hop > p.frame_size / 2) {
    *error = "--hop must be between 1 and frame_size / 2";
  } else if (p.noise_ms <= 0.0f) {
    *error = "--noise_ms must be > 0";
  } else if (p.over_subtraction < 0.0f || p.over_subtraction > 10.0f) {
    *error = "--over_subtraction must be between 0 and 10";
  } else if (p.gain_floor < 0.0f || p.gain_floor > 1.0f) {
    *error = "--gain_floor must be between 0 and 1";
  } else if (options->batch_frames < 64) {
    *error = "--batch_frames must be >= 64";
  } else if (options->max_clips < 0 || options->gallery_count < 0) {
    *error = "--max_clips and --gallery must be >= 0";
  } else {
    return true;
  }
  return false;
}

void PrintUsage(const char* program_name) {
  printf(
      "Usage: %s [options]\n"
      "Batch GPU audio denoising (STFT spectral subtraction with cuFFT).\n\n"
      "  -i, --noisy_dir DIR         noisy input .wav files (data/noisy)\n"
      "  -c, --clean_dir DIR         clean references for SNR (data/clean);\n"
      "                              pass \"\" to skip SNR measurement\n"
      "  -o, --output_dir DIR        output folder (output)\n"
      "  -l, --log_file FILE         per-clip log (logs/run_log.txt)\n"
      "  -S, --summary_file FILE     markdown summary (logs/summary.md)\n"
      "  -r, --results_csv FILE      per-clip CSV (logs/results.csv)\n"
      "  -f, --frame_size N          STFT frame, power of two (1024)\n"
      "  -H, --hop N                 STFT hop (frame_size / 4)\n"
      "  -N, --noise_ms MS           noise-only lead-in length (250)\n"
      "  -a, --over_subtraction F    spectral subtraction alpha (1.5)\n"
      "  -g, --gain_floor F          minimum gain per bin (0.05)\n"
      "  -b, --batch_frames N        STFT frames per GPU chunk (16384)\n"
      "  -m, --max_clips N           process at most N clips (all)\n"
      "  -G, --gallery N             spectrogram images to write (6)\n"
      "  -d, --device N              GPU index (GPU with most SMs)\n"
      "  -v, --verify                compare every clip with the CPU\n"
      "  -h, --help                  show this help\n",
      program_name);
}

}  // namespace denoise
