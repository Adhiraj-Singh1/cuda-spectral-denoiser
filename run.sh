#!/usr/bin/env bash
# Builds the project, generates the dataset (first run only) and denoises
# every clip in data/noisy on the GPU. Extra arguments are passed through to
# bin/denoise, e.g.  ./run.sh --frame_size 2048 --over_subtraction 2
set -euo pipefail
cd "$(dirname "$0")"

make build

if [ -z "$(ls -A data/noisy 2>/dev/null)" ]; then
  echo "Generating dataset in data/ ..."
  make data
fi

mkdir -p logs output
./bin/denoise \
  --noisy_dir data/noisy \
  --clean_dir data/clean \
  --output_dir output \
  --log_file logs/run_log.txt \
  --summary_file logs/summary.md \
  --results_csv logs/results.csv \
  --frame_size 1024 \
  --over_subtraction 1.5 \
  --gain_floor 0.05 \
  --gallery 6 \
  --verify \
  "$@"

echo
echo "Done. Denoised audio: output/denoised | listening examples: output/examples"
echo "Spectrograms: output/spectrograms | log: logs/run_log.txt | summary: logs/summary.md"
