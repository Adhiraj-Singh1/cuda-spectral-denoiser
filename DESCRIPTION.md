# Project description: CUDA Spectral Denoiser

**Goal.** I wanted my capstone to be a complete, measurable signal-processing
application rather than a single kernel. The program removes stationary
background noise (hiss and hum-like noise) from a large batch of audio clips
on the GPU, and it measures how much cleaner every clip became. One run
processes 240 clips, about 9 minutes of audio. It writes the cleaned audio,
before/after spectrograms, listening examples and a per-clip results table.

**Algorithm.** STFT spectral subtraction. Each clip is cut into overlapping
1024-sample Hann-windowed frames (75% overlap) and transformed to 513
frequency bins. The first 250 ms of each clip contain only noise; their
average magnitude per bin is the noise profile. Every bin is then scaled by
G = max(1 − 1.5·noise/|X|, 0.05), the frames are transformed back, and the
audio is rebuilt with a weighted overlap-add.

**GPU implementation.**

- **Batched cuFFT.** Instead of one FFT call per clip, all frames of all
  clips in a chunk (up to 16,384 frames) go through a single `cufftPlanMany`
  real-to-complex call, and later a single complex-to-real call.
- **Five custom kernels:**
  - framing and windowing, using a frame-to-clip lookup table so clips of
    different lengths can share one batch
  - the per-clip noise profile
  - the spectral gain
  - a gather-style overlap-add: each thread reads the 4 frames covering its
    sample, so no atomics are needed
  - an SNR reduction: one block per clip, with a shared-memory tree
    reduction
- **Timing.** Every stage is timed with CUDA events.

**Correctness.** The STFT geometry, window and gain formula live in one
`__host__ __device__` header, shared with a CPU reference that has its own
radix-2 FFT. With `--verify`, every output sample is compared with the CPU
result. I also checked the algorithm against an independent NumPy
implementation, and the outputs matched within one 16-bit audio step.

**Results.** [FILL IN after the GPU run: numbers from logs/summary.md]

**Challenges and lessons learned.**

- **Edge artifacts.** My first prototype normalized the overlap-add by the
  sum of the window. Near the start and end of a clip that sum approaches
  zero, which made the output blow up. The fix was to apply the window when
  rebuilding the audio as well (weighted overlap-add, dividing by Σw²) and
  to start the frames before the clip, so every sample is covered by the
  same number of frames. After that, reconstruction with gain 1 was exact.
- **Parameter tuning.** Over-subtraction is a trade-off. α = 1.5 gave the
  best average SNR. Larger values remove more noise but also eat into the
  quieter harmonics, so the SNR drops. Longer frames (1024 instead of 512
  samples) helped by about 1–1.5 dB, because the narrower frequency bins
  separate the harmonics better from the noise.
- **Musical noise.** The spectrograms show isolated residual noise blobs
  that sound like faint twinkling. This is the classic weakness of
  spectral subtraction.
- **Batching.** Packing variable-length clips into one cuFFT batch needed a
  lookup table and careful offsets, but it means the GPU always works on
  thousands of frames at once.

**Next steps.**

- Reduce musical noise with temporal smoothing of the gain or a
  decision-directed Wiener filter.
- Track the noise profile over time, for noise that isn't stationary.
- Process live audio in real time with CUDA streams.

**How to run:** `bash run.sh` builds the code, generates the dataset and
runs the batch. All options are listed in `README.md`.
