// Copyright 2026 Adhiraj Singh
//
// Command line options for the denoise executable.

#ifndef SRC_CLI_OPTIONS_H_
#define SRC_CLI_OPTIONS_H_

#include <string>

#include "src/dsp_common.h"

namespace denoise {

struct Options {
  std::string noisy_dir = "data/noisy";
  std::string clean_dir = "data/clean";  // Empty = no SNR measurement.
  std::string output_dir = "output";
  std::string log_file = "logs/run_log.txt";
  std::string summary_file = "logs/summary.md";
  std::string results_csv = "logs/results.csv";
  DenoiseParams params;
  int batch_frames = 16384;  // STFT frames per GPU chunk.
  int max_clips = 0;         // 0 = all clips.
  int gallery_count = 6;     // Spectrogram images to write.
  int device = -1;           // -1 = GPU with the most SMs.
  bool verify = false;       // Compare every clip with the CPU reference.
};

// Parses argv into `options`. Returns false and sets `error` on bad input;
// sets `show_help` for -h/--help.
bool ParseOptions(int argc, char** argv, Options* options, bool* show_help,
                  std::string* error);

void PrintUsage(const char* program_name);

}  // namespace denoise

#endif  // SRC_CLI_OPTIONS_H_
