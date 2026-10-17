_Mean over 5 record(s); beat-detection scores pooled over records; edges trimmed before scoring._

**SNR improvement [dB] (higher is better)**

| Noise | Input SNR [dB] | FIR | IIR (causal) | IIR (causal, lag-corrected) | IIR (filtfilt) |
|---|---:|---:|---:|---:|---:|
| baseline wander | 24 | -16.30 | -22.38 | -21.55 | -16.23 |
| baseline wander | 18 | -10.30 | -16.38 | -15.55 | -10.24 |
| baseline wander | 12 | -4.32 | -10.39 | -9.56 | -4.26 |
| baseline wander | 6 | +1.61 | -4.41 | -3.59 | +1.68 |
| baseline wander | 0 | +7.36 | +1.50 | +2.28 | +7.43 |
| electrode motion | 24 | -15.87 | -21.89 | -21.06 | -15.81 |
| electrode motion | 18 | -10.12 | -15.94 | -15.13 | -10.06 |
| electrode motion | 12 | -4.96 | -10.15 | -9.38 | -4.90 |
| electrode motion | 6 | -1.10 | -4.86 | -4.24 | -1.04 |
| electrode motion | 0 | +1.07 | -0.87 | -0.52 | +1.13 |
| muscle artefact | 24 | -15.86 | -21.92 | -21.10 | -15.79 |
| muscle artefact | 18 | -9.94 | -15.94 | -15.12 | -9.87 |
| muscle artefact | 12 | -4.26 | -10.02 | -9.22 | -4.16 |
| muscle artefact | 6 | +0.71 | -4.30 | -3.57 | +0.90 |
| muscle artefact | 0 | +4.28 | +0.75 | +1.29 | +4.59 |

**RMSE vs clean record [mV] (lower is better)**

| Noise | Input SNR [dB] | no filter | FIR | IIR (causal) | IIR (causal, lag-corrected) | IIR (filtfilt) |
|---|---:|---:|---:|---:|---:|---:|
| baseline wander | 24 | 0.022 | 0.151 | 0.282 | 0.271 | 0.151 |
| baseline wander | 18 | 0.043 | 0.152 | 0.282 | 0.271 | 0.151 |
| baseline wander | 12 | 0.086 | 0.152 | 0.282 | 0.272 | 0.151 |
| baseline wander | 6 | 0.171 | 0.153 | 0.283 | 0.272 | 0.152 |
| baseline wander | 0 | 0.341 | 0.156 | 0.285 | 0.275 | 0.155 |
| electrode motion | 24 | 0.023 | 0.153 | 0.282 | 0.272 | 0.152 |
| electrode motion | 18 | 0.046 | 0.156 | 0.284 | 0.274 | 0.155 |
| electrode motion | 12 | 0.091 | 0.169 | 0.291 | 0.281 | 0.168 |
| electrode motion | 6 | 0.182 | 0.210 | 0.316 | 0.307 | 0.209 |
| electrode motion | 0 | 0.363 | 0.321 | 0.398 | 0.391 | 0.319 |
| muscle artefact | 24 | 0.023 | 0.152 | 0.282 | 0.272 | 0.151 |
| muscle artefact | 18 | 0.045 | 0.153 | 0.283 | 0.272 | 0.152 |
| muscle artefact | 12 | 0.091 | 0.157 | 0.285 | 0.275 | 0.156 |
| muscle artefact | 6 | 0.181 | 0.173 | 0.295 | 0.285 | 0.170 |
| muscle artefact | 0 | 0.360 | 0.223 | 0.328 | 0.320 | 0.216 |

**R-peak detection: Se / PPV [%]**

| Noise | Input SNR [dB] | no filter | FIR | IIR (causal) | IIR (causal, lag-corrected) | IIR (filtfilt) |
|---|---:|---:|---:|---:|---:|---:|
| baseline wander | 24 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.2 |
| baseline wander | 18 | 99.6 / 94.1 | 99.6 / 94.1 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.1 |
| baseline wander | 12 | 99.6 / 94.1 | 99.6 / 94.1 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.1 |
| baseline wander | 6 | 99.6 / 94.1 | 99.6 / 94.1 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.1 |
| baseline wander | 0 | 99.6 / 94.0 | 99.6 / 94.0 | 99.6 / 94.1 | 99.6 / 94.1 | 99.6 / 94.0 |
| electrode motion | 24 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.2 |
| electrode motion | 18 | 99.6 / 94.0 | 99.6 / 94.0 | 99.6 / 94.0 | 99.6 / 94.0 | 99.6 / 94.0 |
| electrode motion | 12 | 99.6 / 94.0 | 99.6 / 94.0 | 99.6 / 93.9 | 99.6 / 93.9 | 99.6 / 94.0 |
| electrode motion | 6 | 99.5 / 89.2 | 99.5 / 89.2 | 99.5 / 89.3 | 99.5 / 89.2 | 99.5 / 89.2 |
| electrode motion | 0 | 98.1 / 76.2 | 98.1 / 76.2 | 98.3 / 76.6 | 98.0 / 76.4 | 98.1 / 76.3 |
| muscle artefact | 24 | 99.6 / 94.0 | 99.6 / 94.0 | 99.6 / 94.2 | 99.6 / 94.2 | 99.6 / 94.0 |
| muscle artefact | 18 | 99.6 / 94.0 | 99.6 / 94.1 | 99.6 / 94.0 | 99.6 / 94.0 | 99.6 / 94.1 |
| muscle artefact | 12 | 99.7 / 94.2 | 99.7 / 94.2 | 99.7 / 94.2 | 99.7 / 94.2 | 99.7 / 94.2 |
| muscle artefact | 6 | 99.6 / 93.8 | 99.6 / 93.8 | 99.6 / 93.9 | 99.6 / 93.9 | 99.6 / 93.8 |
| muscle artefact | 0 | 99.3 / 91.5 | 99.3 / 91.5 | 99.3 / 91.5 | 99.3 / 91.5 | 99.3 / 91.5 |
