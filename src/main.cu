// Copyright 2026 Adhiraj Singh
//
// denoise: GPU batch audio denoiser. Reads every .wav file in a folder,
// removes stationary background noise with STFT spectral subtraction on the
// GPU (cuFFT + custom kernels), and writes the cleaned clips, spectrogram
// images, a per-clip log, a CSV of results and a markdown summary.

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/cli_options.h"
#include "src/cpu_reference.h"
#include "src/denoise_pipeline.cuh"
#include "src/dsp_common.h"
#include "src/spectrogram_png.h"
#include "src/wav_io.h"

namespace denoise {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaxClipsPerChunk = 65535;  // gridDim.y limit of OverlapAdd.
constexpr float kVerifyTolerance = 1e-3f;

double MillisecondsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double EnergyRatioDb(double signal, double error) {
  if (error <= 0.0) return 200.0;
  return 10.0 * log10(signal / error);
}

std::string BaseName(const std::string& file_name) {
  const size_t dot = file_name.find_last_of('.');
  return dot == std::string::npos ? file_name : file_name.substr(0, dot);
}

// Writes every message both to stdout and to the log file.
class RunLogger {
 public:
  explicit RunLogger(const std::string& path)
      : file_(fopen(path.c_str(), "w")) {}
  ~RunLogger() {
    if (file_ != nullptr) fclose(file_);
  }
  RunLogger(const RunLogger&) = delete;
  RunLogger& operator=(const RunLogger&) = delete;

  bool ok() const { return file_ != nullptr; }

  __attribute__((format(printf, 2, 3))) void Log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    vprintf(format, args);
    if (file_ != nullptr) {
      vfprintf(file_, format, args_copy);
      fflush(file_);
    }
    va_end(args_copy);
    va_end(args);
  }

 private:
  FILE* file_;
};

struct RunStats {
  int processed = 0;
  int failed = 0;
  int chunks = 0;
  int64_t samples = 0;
  int64_t frames = 0;
  double audio_seconds = 0.0;
  double cpu_ms = 0.0;
  int verified_ok = 0;
  int verified_bad = 0;
  float max_difference = 0.0f;
  std::vector<double> input_snr;
  std::vector<double> output_snr;
  StageTimes gpu;
};

int SelectDevice(int requested, RunLogger* log) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    log->Log("ERROR: no CUDA capable GPU was found.\n");
    exit(EXIT_FAILURE);
  }
  int chosen = requested;
  if (chosen < 0 || chosen >= device_count) {
    int best = -1;
    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp prop;
      CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
      if (prop.multiProcessorCount > best) {
        best = prop.multiProcessorCount;
        chosen = i;
      }
    }
  }
  CHECK_CUDA(cudaSetDevice(chosen));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, chosen));
  log->Log("GPU devices found: %d\n", device_count);
  log->Log("Using GPU %d: %s | compute %d.%d | %d SMs | %.1f GB global mem\n",
           chosen, prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  return chosen;
}

// Loads the noisy clip and (if present) its clean reference.
bool LoadClip(const std::string& noisy_path, const Options& options,
              ClipJob* job, std::string* error) {
  AudioClip noisy;
  if (!ReadWav(noisy_path, &noisy, error)) return false;
  if (noisy.samples.empty()) {
    *error = "empty audio";
    return false;
  }
  job->name = FileName(noisy_path);
  job->sample_rate = noisy.sample_rate;
  job->noisy = std::move(noisy.samples);
  if (!options.clean_dir.empty()) {
    AudioClip clean;
    std::string ignored;
    if (ReadWav(options.clean_dir + "/" + job->name, &clean, &ignored) &&
        clean.samples.size() == job->noisy.size()) {
      job->clean = std::move(clean.samples);
    }
  }
  return true;
}

// Writes outputs for one processed clip and updates the statistics.
void FinishClip(ClipJob* job, int index, int total, const Options& options,
                RunStats* stats, FILE* csv, RunLogger* log) {
  const std::string base = BaseName(job->name);
  const int length = static_cast<int>(job->noisy.size());
  const double seconds = static_cast<double>(length) / job->sample_rate;
  const bool has_clean = job->clean.size() == job->noisy.size();

  const std::string out_path = options.output_dir + "/denoised/" + job->name;
  const bool written = WriteWav16(out_path, job->denoised, job->sample_rate);

  // SNR from the GPU reduction.
  double snr_in = 0.0, snr_out = 0.0;
  if (has_clean) {
    snr_in = EnergyRatioDb(job->clean_energy, job->input_error_energy);
    snr_out = EnergyRatioDb(job->clean_energy, job->output_error_energy);
    stats->input_snr.push_back(snr_in);
    stats->output_snr.push_back(snr_out);
  }

  std::string verify_text;
  float max_diff = 0.0f;
  if (options.verify) {
    std::vector<float> cpu;
    const Clock::time_point start = Clock::now();
    DenoiseClipCpu(job->noisy.data(), length, options.params, job->sample_rate,
                   &cpu);
    stats->cpu_ms += MillisecondsSince(start);
    for (int t = 0; t < length; ++t) {
      max_diff = std::max(max_diff, fabsf(cpu[t] - job->denoised[t]));
    }
    bool ok = max_diff <= kVerifyTolerance;
    if (has_clean) {
      // The GPU reduction must agree with a host-side SNR of the GPU output.
      const double host_snr =
          SnrDb(job->clean.data(), job->denoised.data(), length);
      ok = ok && fabs(host_snr - snr_out) < 0.01;
    }
    stats->max_difference = std::max(stats->max_difference, max_diff);
    if (ok) {
      ++stats->verified_ok;
    } else {
      ++stats->verified_bad;
    }
    char buffer[64];
    snprintf(buffer, sizeof(buffer), " | cpu max diff %.1e %s", max_diff,
             ok ? "OK" : "MISMATCH");
    verify_text = buffer;
  }

  if (job->keep_magnitudes) {
    const int bins = NumBins(options.params.frame_size);
    std::vector<float> clean_magnitudes;
    std::vector<SpectrogramPanel> panels;
    if (has_clean) {
      int frames = 0;
      ComputeMagnitudes(job->clean.data(), length, options.params,
                        job->sample_rate, &clean_magnitudes, &frames);
      panels.push_back({clean_magnitudes.data(), frames, bins});
    }
    panels.push_back({job->magnitude_in.data(), job->num_frames, bins});
    panels.push_back({job->magnitude_out.data(), job->num_frames, bins});
    WriteSpectrogramPng(
        options.output_dir + "/spectrograms/" + base + "_spectrogram.png",
        panels);
    // Listening examples: clean (if known), noisy input and denoised output.
    const std::string prefix = options.output_dir + "/examples/" + base;
    if (has_clean) WriteWav16(prefix + "_0_clean.wav", job->clean,
                              job->sample_rate);
    WriteWav16(prefix + "_1_noisy_input.wav", job->noisy, job->sample_rate);
    WriteWav16(prefix + "_2_denoised_output.wav", job->denoised,
               job->sample_rate);
  }

  if (written) {
    ++stats->processed;
  } else {
    ++stats->failed;
  }
  stats->samples += length;
  stats->frames += job->num_frames;
  stats->audio_seconds += seconds;

  if (has_clean) {
    log->Log("[%4d/%4d] %-16s %5.2f s | %3d frames | SNR %6.2f dB -> %6.2f dB "
             "(%+5.2f dB)%s%s\n",
             index + 1, total, job->name.c_str(), seconds, job->num_frames,
             snr_in, snr_out, snr_out - snr_in, verify_text.c_str(),
             written ? "" : " (WRITE FAILED)");
  } else {
    log->Log("[%4d/%4d] %-16s %5.2f s | %3d frames | no clean reference%s%s\n",
             index + 1, total, job->name.c_str(), seconds, job->num_frames,
             verify_text.c_str(), written ? "" : " (WRITE FAILED)");
  }
  if (csv != nullptr) {
    fprintf(csv, "%s,%d,%.3f,%d,", job->name.c_str(), length, seconds,
            job->num_frames);
    if (has_clean) {
      fprintf(csv, "%.3f,%.3f,%.3f,", snr_in, snr_out, snr_out - snr_in);
    } else {
      fprintf(csv, ",,,");
    }
    if (options.verify) {
      fprintf(csv, "%.3g\n", max_diff);
    } else {
      fprintf(csv, "\n");
    }
  }
}

double Mean(const std::vector<double>& v) {
  double sum = 0.0;
  for (double x : v) sum += x;
  return v.empty() ? 0.0 : sum / v.size();
}

void WriteSummary(const Options& options, const RunStats& stats, int total,
                  double wall_ms, RunLogger* log) {
  const StageTimes& g = stats.gpu;
  const double gpu_ms = g.Total();
  const double realtime = gpu_ms > 0 ? stats.audio_seconds / (gpu_ms / 1e3)
                                     : 0.0;
  std::vector<double> improvement;
  for (size_t i = 0; i < stats.input_snr.size(); ++i) {
    improvement.push_back(stats.output_snr[i] - stats.input_snr[i]);
  }
  const double min_imp =
      improvement.empty()
          ? 0.0
          : *std::min_element(improvement.begin(), improvement.end());
  const double max_imp =
      improvement.empty()
          ? 0.0
          : *std::max_element(improvement.begin(), improvement.end());

  log->Log("\n====================== Summary ======================\n");
  log->Log("Clips found / processed / failed: %d / %d / %d\n", total,
           stats.processed, stats.failed);
  log->Log("Audio processed:                  %.1f s (%" PRId64
           " samples, %" PRId64 " STFT frames)\n",
           stats.audio_seconds, stats.samples, stats.frames);
  log->Log("STFT frame / hop:                 %d / %d samples\n",
           options.params.frame_size, options.params.hop);
  log->Log("GPU chunks (batched cuFFT calls): %d\n", stats.chunks);
  log->Log("GPU time by stage (ms):\n");
  log->Log("  upload %.2f | framing %.2f | forward FFT %.2f | "
           "noise+gain %.2f\n",
           g.upload, g.framing, g.forward_fft, g.noise_and_gain);
  log->Log("  inverse FFT %.2f | overlap-add %.2f | SNR %.2f | download %.2f\n",
           g.inverse_fft, g.overlap_add, g.snr, g.download);
  log->Log("Total GPU time:                   %.2f ms\n", gpu_ms);
  log->Log("GPU speed:                        %.0fx real time\n", realtime);
  log->Log("Wall time (incl. WAV file I/O):   %.1f ms\n", wall_ms);
  if (!improvement.empty()) {
    log->Log("Mean SNR:                         %.2f dB -> %.2f dB\n",
             Mean(stats.input_snr), Mean(stats.output_snr));
    log->Log("SNR improvement mean / min / max: %+.2f / %+.2f / %+.2f dB\n",
             Mean(improvement), min_imp, max_imp);
  }
  if (options.verify) {
    log->Log("Sum of CPU reference time:        %.1f ms\n", stats.cpu_ms);
    log->Log("GPU speedup vs. 1 CPU core:       %.1fx\n",
             gpu_ms > 0 ? stats.cpu_ms / gpu_ms : 0.0);
    log->Log("Verification:                     %d OK, %d mismatched (max "
             "sample difference %.1e)\n",
             stats.verified_ok, stats.verified_bad, stats.max_difference);
  }

  FILE* md = fopen(options.summary_file.c_str(), "w");
  if (md == nullptr) return;
  fprintf(md, "# Run summary\n\n| Metric | Value |\n|---|---|\n");
  fprintf(md, "| Clips processed | %d of %d |\n", stats.processed, total);
  fprintf(md, "| Audio processed | %.1f s (%" PRId64 " STFT frames) |\n",
          stats.audio_seconds, stats.frames);
  fprintf(md, "| STFT frame / hop | %d / %d samples |\n",
          options.params.frame_size, options.params.hop);
  fprintf(md, "| GPU chunks | %d |\n", stats.chunks);
  fprintf(md, "| Total GPU time | %.2f ms |\n", gpu_ms);
  fprintf(md, "| GPU speed | %.0fx real time |\n", realtime);
  fprintf(md, "| Wall time (incl. WAV file I/O) | %.1f ms |\n", wall_ms);
  if (!improvement.empty()) {
    fprintf(md, "| Mean SNR before / after | %.2f dB / %.2f dB |\n",
            Mean(stats.input_snr), Mean(stats.output_snr));
    fprintf(md, "| SNR improvement mean / min / max | %+.2f / %+.2f / %+.2f "
            "dB |\n", Mean(improvement), min_imp, max_imp);
  }
  if (options.verify) {
    fprintf(md, "| CPU reference time | %.1f ms |\n", stats.cpu_ms);
    fprintf(md, "| GPU speedup vs. 1 CPU core | %.1fx |\n",
            gpu_ms > 0 ? stats.cpu_ms / gpu_ms : 0.0);
    fprintf(md, "| Verified against CPU | %d OK, %d mismatched |\n",
            stats.verified_ok, stats.verified_bad);
  }
  fprintf(md, "\n## GPU time by stage\n\n| Stage | ms | Share |\n");
  fprintf(md, "|---|---|---|\n");
  const struct {
    const char* name;
    double ms;
  } kStages[] = {{"Upload (H2D)", g.upload},
                 {"Framing + window", g.framing},
                 {"Forward FFT (cuFFT R2C)", g.forward_fft},
                 {"Noise profile + gain", g.noise_and_gain},
                 {"Inverse FFT (cuFFT C2R)", g.inverse_fft},
                 {"Overlap-add", g.overlap_add},
                 {"SNR reduction", g.snr},
                 {"Download (D2H)", g.download}};
  for (const auto& stage : kStages) {
    fprintf(md, "| %s | %.2f | %.1f%% |\n", stage.name, stage.ms,
            gpu_ms > 0 ? 100.0 * stage.ms / gpu_ms : 0.0);
  }
  fclose(md);
}

int Run(const Options& options) {
  for (const std::string& file :
       {options.log_file, options.summary_file, options.results_csv}) {
    const size_t slash = file.find_last_of('/');
    if (slash != std::string::npos) MakeDirectories(file.substr(0, slash));
  }
  RunLogger log(options.log_file);
  if (!log.ok()) {
    fprintf(stderr, "Cannot open log file %s\n", options.log_file.c_str());
    return EXIT_FAILURE;
  }
  const time_t now = time(nullptr);
  log.Log("CUDA spectral denoiser - run started %s", ctime(&now));
  SelectDevice(options.device, &log);

  std::vector<std::string> files = ListWavFiles(options.noisy_dir);
  if (options.max_clips > 0 &&
      static_cast<int>(files.size()) > options.max_clips) {
    files.resize(options.max_clips);
  }
  if (files.empty()) {
    log.Log("ERROR: no .wav files found in %s\n", options.noisy_dir.c_str());
    return EXIT_FAILURE;
  }
  const int total = static_cast<int>(files.size());
  for (const char* sub : {"/denoised", "/spectrograms", "/examples"}) {
    if (!MakeDirectories(options.output_dir + sub)) {
      log.Log("ERROR: cannot create %s%s\n", options.output_dir.c_str(), sub);
      return EXIT_FAILURE;
    }
  }
  FILE* csv = fopen(options.results_csv.c_str(), "w");
  if (csv != nullptr) {
    fprintf(csv, "clip,samples,seconds,frames,input_snr_db,output_snr_db,"
            "improvement_db,max_diff_vs_cpu\n");
  }

  log.Log("Input: %s (%d clips) | clean references: %s\n",
          options.noisy_dir.c_str(), total,
          options.clean_dir.empty() ? "none" : options.clean_dir.c_str());
  log.Log("STFT frame %d, hop %d | over-subtraction %.2f | gain floor %.2f | "
          "noise lead-in %.0f ms | %d frames per GPU chunk\n\n",
          options.params.frame_size, options.params.hop,
          options.params.over_subtraction, options.params.gain_floor,
          options.params.noise_ms, options.batch_frames);

  const int gallery_step =
      options.gallery_count > 0 ? std::max(1, total / options.gallery_count)
                                : 0;
  int gallery_written = 0;

  RunStats stats;
  const Clock::time_point wall_start = Clock::now();
  {
    DenoisePipeline pipeline(options.params, options.batch_frames);
    std::vector<std::unique_ptr<ClipJob>> chunk;
    std::vector<int> chunk_index;
    int chunk_frames = 0;

    auto flush = [&]() {
      if (chunk.empty()) return;
      std::vector<ClipJob*> jobs;
      for (auto& job : chunk) jobs.push_back(job.get());
      pipeline.Process(jobs, &stats.gpu);
      ++stats.chunks;
      for (size_t i = 0; i < chunk.size(); ++i) {
        FinishClip(chunk[i].get(), chunk_index[i], total, options, &stats, csv,
                   &log);
      }
      chunk.clear();
      chunk_index.clear();
      chunk_frames = 0;
    };

    for (int i = 0; i < total; ++i) {
      std::unique_ptr<ClipJob> job(new ClipJob());
      std::string error;
      if (!LoadClip(files[i], options, job.get(), &error)) {
        log.Log("[%4d/%4d] SKIPPED %s: %s\n", i + 1, total, files[i].c_str(),
                error.c_str());
        ++stats.failed;
        continue;
      }
      const int frames =
          pipeline.FramesFor(static_cast<int>(job->noisy.size()));
      if (frames > pipeline.capacity_frames()) {
        log.Log("[%4d/%4d] SKIPPED %s: clip needs %d frames, more than "
                "--batch_frames %d\n",
                i + 1, total, files[i].c_str(), frames,
                pipeline.capacity_frames());
        ++stats.failed;
        continue;
      }
      if (chunk_frames + frames > pipeline.capacity_frames() ||
          static_cast<int>(chunk.size()) >= kMaxClipsPerChunk) {
        flush();
      }
      job->keep_magnitudes = gallery_step > 0 && i % gallery_step == 0 &&
                             gallery_written < options.gallery_count;
      if (job->keep_magnitudes) ++gallery_written;
      chunk_frames += frames;
      chunk.push_back(std::move(job));
      chunk_index.push_back(i);
    }
    flush();
  }  // The pipeline frees its GPU memory here, before the device reset.
  const double wall_ms = MillisecondsSince(wall_start);
  if (csv != nullptr) fclose(csv);

  WriteSummary(options, stats, total, wall_ms, &log);
  CHECK_CUDA(cudaDeviceReset());
  return stats.failed == 0 && stats.verified_bad == 0 ? EXIT_SUCCESS
                                                      : EXIT_FAILURE;
}

}  // namespace
}  // namespace denoise

int main(int argc, char** argv) {
  denoise::Options options;
  bool show_help = false;
  std::string error;
  if (!denoise::ParseOptions(argc, argv, &options, &show_help, &error)) {
    fprintf(stderr, "Error: %s\n\n", error.c_str());
    denoise::PrintUsage(argv[0]);
    return 2;
  }
  if (show_help) {
    denoise::PrintUsage(argv[0]);
    return 0;
  }
  return denoise::Run(options);
}
