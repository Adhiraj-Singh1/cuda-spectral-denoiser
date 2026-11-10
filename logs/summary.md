# Run summary

| Metric | Value |
|---|---|
| Clips processed | 240 of 240 |
| Audio processed | 534.5 s (34237 STFT frames) |
| STFT frame / hop | 1024 / 256 samples |
| GPU chunks | 3 |
| Total GPU time | 18.42 ms |
| GPU speed | 29011x real time |
| Wall time (incl. WAV file I/O) | 7993.5 ms |
| Mean SNR before / after | 5.11 dB / 14.25 dB |
| SNR improvement mean / min / max | +9.14 / +5.26 / +11.29 dB |
| CPU reference time | 6636.7 ms |
| GPU speedup vs. 1 CPU core | 360.2x |
| Verified against CPU | 240 OK, 0 mismatched |

## GPU time by stage

| Stage | ms | Share |
|---|---|---|
| Upload (H2D) | 5.70 | 31.0% |
| Framing + window | 1.99 | 10.8% |
| Forward FFT (cuFFT R2C) | 1.38 | 7.5% |
| Noise profile + gain | 1.93 | 10.5% |
| Inverse FFT (cuFFT C2R) | 1.38 | 7.5% |
| Overlap-add | 1.44 | 7.8% |
| SNR reduction | 1.98 | 10.7% |
| Download (D2H) | 2.64 | 14.3% |
