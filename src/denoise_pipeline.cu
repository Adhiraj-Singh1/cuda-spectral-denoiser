// Copyright 2026 Adhiraj Singh

#include "src/denoise_pipeline.cuh"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "src/dsp_common.h"

namespace denoise {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kReductionThreads = 256;  // Must be a power of two.

int BlocksFor(int64_t work) {
  return static_cast<int>((work + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

// One thread per (frame, sample): copies the frame's samples out of its clip
// (zero outside the clip) and applies the Hann analysis window.
__global__ void FrameAndWindowKernel(const float* __restrict__ signal,
                                     const ClipInfo* __restrict__ clips,
                                     const int* __restrict__ frame_clip,
                                     float* __restrict__ frames,
                                     int total_frames, int frame_size,
                                     int hop) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(total_frames) * frame_size) return;
  const int frame = static_cast<int>(idx / frame_size);
  const int i = static_cast<int>(idx % frame_size);
  const ClipInfo clip = clips[frame_clip[frame]];
  const int t = FrameStart(frame - clip.first_frame, frame_size, hop) + i;
  const float sample =
      (t >= 0 && t < clip.length) ? signal[clip.sample_offset + t] : 0.0f;
  frames[idx] = sample * HannWindow(i, frame_size);
}

// One thread per (clip, frequency bin): mean magnitude over the clip's
// noise-only lead-in frames.
__global__ void NoiseProfileKernel(const cufftComplex* __restrict__ spectra,
                                   const ClipInfo* __restrict__ clips,
                                   float* __restrict__ noise, int num_clips,
                                   int bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_clips * bins) return;
  const int clip_index = idx / bins;
  const int k = idx % bins;
  const ClipInfo clip = clips[clip_index];
  float sum = 0.0f;
  for (int f = 0; f < clip.noise_count; ++f) {
    const cufftComplex x =
        spectra[static_cast<int64_t>(clip.first_frame + clip.noise_first + f) *
                    bins + k];
    sum += sqrtf(x.x * x.x + x.y * x.y);
  }
  noise[idx] = clip.noise_count > 0 ? sum / clip.noise_count : 0.0f;
}

// One thread per (frame, bin): applies the spectral subtraction gain and
// records the magnitude before and after (used for the spectrogram images).
__global__ void SpectralGainKernel(cufftComplex* __restrict__ spectra,
                                   const ClipInfo* __restrict__ clips,
                                   const int* __restrict__ frame_clip,
                                   const float* __restrict__ noise,
                                   float* __restrict__ magnitude_in,
                                   float* __restrict__ magnitude_out,
                                   int total_frames, int bins,
                                   float over_subtraction, float gain_floor) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(total_frames) * bins) return;
  const int frame = static_cast<int>(idx / bins);
  const int k = static_cast<int>(idx % bins);
  const int clip_index = frame_clip[frame];
  cufftComplex x = spectra[idx];
  const float magnitude = sqrtf(x.x * x.x + x.y * x.y);
  const float noise_level = noise[static_cast<int64_t>(clip_index) * bins + k];
  const float gain =
      SpectralGain(magnitude, noise_level, over_subtraction, gain_floor);
  x.x *= gain;
  x.y *= gain;
  spectra[idx] = x;
  magnitude_in[idx] = magnitude;
  magnitude_out[idx] = magnitude * gain;
}

// One thread per output sample (grid.y = clip): weighted overlap-add of every
// frame that covers the sample, normalized by the summed squared window.
// Gathering instead of scattering means no atomics are needed.
__global__ void OverlapAddKernel(const float* __restrict__ frames,
                                 const ClipInfo* __restrict__ clips,
                                 float* __restrict__ output, int frame_size,
                                 int hop) {
  const ClipInfo clip = clips[blockIdx.y];
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= clip.length) return;
  int first = 0, last = 0;
  FramesCoveringSample(t, clip.num_frames, frame_size, hop, &first, &last);
  float sum = 0.0f;
  float weight = 0.0f;
  for (int f = first; f <= last; ++f) {
    const int i = t - FrameStart(f, frame_size, hop);
    const float w = HannWindow(i, frame_size);
    // cuFFT's inverse transform is unnormalized: divide by frame_size.
    sum += frames[static_cast<int64_t>(clip.first_frame + f) * frame_size + i] *
           w / frame_size;
    weight += w * w;
  }
  output[clip.sample_offset + t] = weight > 1e-8f ? sum / weight : 0.0f;
}

// One block per clip: shared-memory tree reduction of
// sum(clean^2), sum((noisy - clean)^2) and sum((output - clean)^2).
__global__ void SnrReductionKernel(const float* __restrict__ clean,
                                   const float* __restrict__ noisy,
                                   const float* __restrict__ output,
                                   const ClipInfo* __restrict__ clips,
                                   double* __restrict__ energies) {
  __shared__ double clean_sum[kReductionThreads];
  __shared__ double input_sum[kReductionThreads];
  __shared__ double output_sum[kReductionThreads];

  const ClipInfo clip = clips[blockIdx.x];
  const int tid = threadIdx.x;
  double c2 = 0.0, in2 = 0.0, out2 = 0.0;
  if (clip.has_clean) {
    for (int t = tid; t < clip.length; t += blockDim.x) {
      const int64_t i = clip.sample_offset + t;
      const double c = clean[i];
      const double di = noisy[i] - c;
      const double d_o = output[i] - c;
      c2 += c * c;
      in2 += di * di;
      out2 += d_o * d_o;
    }
  }
  clean_sum[tid] = c2;
  input_sum[tid] = in2;
  output_sum[tid] = out2;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      clean_sum[tid] += clean_sum[tid + stride];
      input_sum[tid] += input_sum[tid + stride];
      output_sum[tid] += output_sum[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    energies[3 * blockIdx.x] = clean_sum[0];
    energies[3 * blockIdx.x + 1] = input_sum[0];
    energies[3 * blockIdx.x + 2] = output_sum[0];
  }
}

float ElapsedMs(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

}  // namespace

DenoisePipeline::DenoisePipeline(const DenoiseParams& params,
                                 int capacity_frames)
    : params_(params),
      capacity_frames_(capacity_frames),
      capacity_samples_(static_cast<int64_t>(capacity_frames) * params.hop),
      bins_(NumBins(params.frame_size)) {
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  for (cudaEvent_t& event : events_) CHECK_CUDA(cudaEventCreate(&event));
  Allocate();
}

DenoisePipeline::~DenoisePipeline() {
  cudaStreamSynchronize(stream_);
  Free();
  for (cudaEvent_t& event : events_) cudaEventDestroy(event);
  cudaStreamDestroy(stream_);
}

void DenoisePipeline::Allocate() {
  const size_t frames = static_cast<size_t>(capacity_frames_);
  const size_t samples = static_cast<size_t>(capacity_samples_);
  const size_t spectrum_values = frames * bins_;
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_signal_),
                           samples * sizeof(float), cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_clean_),
                           samples * sizeof(float), cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_output_),
                           samples * sizeof(float), cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_clips_),
                           frames * sizeof(ClipInfo), cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_frame_clip_),
                           frames * sizeof(int), cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_energies_),
                           frames * 3 * sizeof(double), cudaHostAllocDefault));

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_signal_),
                        samples * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_clean_),
                        samples * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_output_),
                        samples * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_frames_),
                        frames * params_.frame_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_spectra_),
                        spectrum_values * sizeof(cufftComplex)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_noise_),
                        spectrum_values * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_magnitude_in_),
                        spectrum_values * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_magnitude_out_),
                        spectrum_values * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_clips_),
                        frames * sizeof(ClipInfo)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_frame_clip_),
                        frames * sizeof(int)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_energies_),
                        frames * 3 * sizeof(double)));
}

void DenoisePipeline::Free() {
  cudaFreeHost(host_signal_);
  cudaFreeHost(host_clean_);
  cudaFreeHost(host_output_);
  cudaFreeHost(host_clips_);
  cudaFreeHost(host_frame_clip_);
  cudaFreeHost(host_energies_);
  cudaFree(dev_signal_);
  cudaFree(dev_clean_);
  cudaFree(dev_output_);
  cudaFree(dev_frames_);
  cudaFree(dev_spectra_);
  cudaFree(dev_noise_);
  cudaFree(dev_magnitude_in_);
  cudaFree(dev_magnitude_out_);
  cudaFree(dev_clips_);
  cudaFree(dev_frame_clip_);
  cudaFree(dev_energies_);
}

void DenoisePipeline::Process(const std::vector<ClipJob*>& clips,
                              StageTimes* times) {
  const int n = params_.frame_size;
  const int hop = params_.hop;
  const int num_clips = static_cast<int>(clips.size());
  if (num_clips == 0) return;

  // Pack the clips into the pinned staging buffers.
  int64_t total_samples = 0;
  int total_frames = 0;
  int max_length = 0;
  for (int c = 0; c < num_clips; ++c) {
    ClipJob* job = clips[c];
    const int length = static_cast<int>(job->noisy.size());
    ClipInfo& info = host_clips_[c];
    info.sample_offset = total_samples;
    info.length = length;
    info.first_frame = total_frames;
    info.num_frames = FramesFor(length);
    const int noise_samples =
        static_cast<int>(params_.noise_ms * 0.001f * job->sample_rate);
    NoiseFrameRange(length, noise_samples, n, hop, &info.noise_first,
                    &info.noise_count);
    info.has_clean = job->clean.size() == job->noisy.size() ? 1 : 0;
    memcpy(host_signal_ + total_samples, job->noisy.data(),
           length * sizeof(float));
    if (info.has_clean) {
      memcpy(host_clean_ + total_samples, job->clean.data(),
             length * sizeof(float));
    } else {
      memset(host_clean_ + total_samples, 0, length * sizeof(float));
    }
    for (int f = 0; f < info.num_frames; ++f) {
      host_frame_clip_[total_frames + f] = c;
    }
    job->num_frames = info.num_frames;
    job->noise_frames = info.noise_count;
    total_samples += length;
    total_frames += info.num_frames;
    max_length = std::max(max_length, length);
  }
  if (total_frames > capacity_frames_) {
    fprintf(stderr, "Chunk of %d frames exceeds capacity %d\n", total_frames,
            capacity_frames_);
    exit(EXIT_FAILURE);
  }

  // Batched FFT plans for exactly this chunk's frames.
  cufftHandle forward_plan, inverse_plan;
  int fft_size = n;
  CHECK_CUFFT(cufftPlanMany(&forward_plan, 1, &fft_size, nullptr, 1, n,
                            nullptr, 1, bins_, CUFFT_R2C, total_frames));
  CHECK_CUFFT(cufftPlanMany(&inverse_plan, 1, &fft_size, nullptr, 1, bins_,
                            nullptr, 1, n, CUFFT_C2R, total_frames));
  CHECK_CUFFT(cufftSetStream(forward_plan, stream_));
  CHECK_CUFFT(cufftSetStream(inverse_plan, stream_));

  const size_t sample_bytes = total_samples * sizeof(float);
  CHECK_CUDA(cudaEventRecord(events_[0], stream_));
  CHECK_CUDA(cudaMemcpyAsync(dev_signal_, host_signal_, sample_bytes,
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync(dev_clean_, host_clean_, sample_bytes,
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync(dev_clips_, host_clips_,
                             num_clips * sizeof(ClipInfo),
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaMemcpyAsync(dev_frame_clip_, host_frame_clip_,
                             total_frames * sizeof(int),
                             cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA(cudaEventRecord(events_[1], stream_));

  const int64_t frame_samples = static_cast<int64_t>(total_frames) * n;
  FrameAndWindowKernel<<<BlocksFor(frame_samples), kThreadsPerBlock, 0,
                         stream_>>>(dev_signal_, dev_clips_, dev_frame_clip_,
                                    dev_frames_, total_frames, n, hop);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(events_[2], stream_));

  CHECK_CUFFT(cufftExecR2C(forward_plan, dev_frames_, dev_spectra_));
  CHECK_CUDA(cudaEventRecord(events_[3], stream_));

  const int64_t spectrum_values = static_cast<int64_t>(total_frames) * bins_;
  NoiseProfileKernel<<<BlocksFor(static_cast<int64_t>(num_clips) * bins_),
                       kThreadsPerBlock, 0, stream_>>>(
      dev_spectra_, dev_clips_, dev_noise_, num_clips, bins_);
  SpectralGainKernel<<<BlocksFor(spectrum_values), kThreadsPerBlock, 0,
                       stream_>>>(
      dev_spectra_, dev_clips_, dev_frame_clip_, dev_noise_, dev_magnitude_in_,
      dev_magnitude_out_, total_frames, bins_, params_.over_subtraction,
      params_.gain_floor);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(events_[4], stream_));

  CHECK_CUFFT(cufftExecC2R(inverse_plan, dev_spectra_, dev_frames_));
  CHECK_CUDA(cudaEventRecord(events_[5], stream_));

  const dim3 overlap_grid(BlocksFor(max_length), num_clips);
  OverlapAddKernel<<<overlap_grid, kThreadsPerBlock, 0, stream_>>>(
      dev_frames_, dev_clips_, dev_output_, n, hop);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(events_[6], stream_));

  SnrReductionKernel<<<num_clips, kReductionThreads, 0, stream_>>>(
      dev_clean_, dev_signal_, dev_output_, dev_clips_, dev_energies_);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(events_[7], stream_));

  CHECK_CUDA(cudaMemcpyAsync(host_output_, dev_output_, sample_bytes,
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpyAsync(host_energies_, dev_energies_,
                             num_clips * 3 * sizeof(double),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaEventRecord(events_[8], stream_));
  CHECK_CUDA(cudaStreamSynchronize(stream_));

  times->upload += ElapsedMs(events_[0], events_[1]);
  times->framing += ElapsedMs(events_[1], events_[2]);
  times->forward_fft += ElapsedMs(events_[2], events_[3]);
  times->noise_and_gain += ElapsedMs(events_[3], events_[4]);
  times->inverse_fft += ElapsedMs(events_[4], events_[5]);
  times->overlap_add += ElapsedMs(events_[5], events_[6]);
  times->snr += ElapsedMs(events_[6], events_[7]);
  times->download += ElapsedMs(events_[7], events_[8]);

  // Unpack results.
  for (int c = 0; c < num_clips; ++c) {
    ClipJob* job = clips[c];
    const ClipInfo& info = host_clips_[c];
    job->denoised.assign(host_output_ + info.sample_offset,
                         host_output_ + info.sample_offset + info.length);
    job->clean_energy = host_energies_[3 * c];
    job->input_error_energy = host_energies_[3 * c + 1];
    job->output_error_energy = host_energies_[3 * c + 2];
    if (job->keep_magnitudes) {
      const size_t count = static_cast<size_t>(info.num_frames) * bins_;
      const size_t offset = static_cast<size_t>(info.first_frame) * bins_;
      job->magnitude_in.resize(count);
      job->magnitude_out.resize(count);
      CHECK_CUDA(cudaMemcpy(job->magnitude_in.data(),
                            dev_magnitude_in_ + offset, count * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(job->magnitude_out.data(),
                            dev_magnitude_out_ + offset, count * sizeof(float),
                            cudaMemcpyDeviceToHost));
    }
  }
  cufftDestroy(forward_plan);
  cufftDestroy(inverse_plan);
}

}  // namespace denoise
