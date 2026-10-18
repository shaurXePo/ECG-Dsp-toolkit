# References and further reading

Links only. Papers behind a publisher paywall are not redistributed in this repository; open each one through its DOI
(many universities and libraries give access).

## Why these papers

| Topic in this repo | Paper |
|---|---|
| Filter trade-offs for ECG (linear phase, distortion, overlapping spectra) | Luo & Johnston, *A review of electrocardiogram filtering*, J. Electrocardiology 43(6):486-496, 2010. DOI: [10.1016/j.jelectrocard.2010.07.007](https://doi.org/10.1016/j.jelectrocard.2010.07.007) |
| Baseline wander removal methods compared (FIR / IIR / others) | Romero et al., *Baseline wander removal methods for ECG signals: a comparative study*, arXiv:1807.11359 (open access): <https://arxiv.org/abs/1807.11359> |
| R-peak detector (`src/peaks.py` is a simplified version) | Pan & Tompkins, *A Real-Time QRS Detection Algorithm*, IEEE Trans. Biomed. Eng., 1985. DOI: [10.1109/TBME.1985.325532](https://doi.org/10.1109/TBME.1985.325532) |
| Noise recordings (bw, em, ma) used for the experiments | Moody, Muldrow & Mark, *A noise stress test for arrhythmia detectors*, Computers in Cardiology, 1984. Database: <https://physionet.org/content/nstdb/1.0.0/> |
| ECG recordings used for the experiments | Moody & Mark, *The impact of the MIT-BIH Arrhythmia Database*, IEEE Eng. Med. Biol. Mag. 20(3):45-50, 2001. Database: <https://physionet.org/content/mitdb/1.0.0/> |
| PhysioNet itself | Goldberger et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23):e215-e220, 2000. |

## Classic DSP references (as implemented here)

| Topic | Reference |
|---|---|
| Radix-2 FFT | Cooley & Tukey, *An algorithm for the machine calculation of complex Fourier series*, Math. Comp. 19:297-301, 1965 |
| Windows and leakage | Harris, *On the use of windows for harmonic analysis with the discrete Fourier transform*, Proc. IEEE 66(1):51-83, 1978 |
| Welch PSD | Welch, *The use of fast Fourier transform for the estimation of power spectra*, IEEE Trans. Audio Electroacoust. 15(2):70-73, 1967 |
| Window design, bilinear transform, Butterworth | Oppenheim & Schafer, *Discrete-Time Signal Processing* (textbook) |

## Notes
* Check each citation against the publisher's page before relying on page numbers.
* Data files are not included in this repo. Run `python data/download_data.py` to fetch them from PhysioNet.
