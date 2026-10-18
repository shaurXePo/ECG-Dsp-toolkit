# Vision add-on: ECG strips + YOLO beat detection + Streamlit app

```
raw ECG -> toolkit filters (whole excerpt) -> 4 s strip images -> YOLO boxes (class + confidence)
                                                                       -> drawn back on the ECG in Streamlit
```

## What is in here
| Path | Purpose |
|---|---|
| `vision/render.py` | filtered window -> 640x256 image, keeps the pixel <-> time mapping |
| `vision/labels.py` | MIT-BIH annotations -> YOLO boxes (AAMI classes N, S, V, F) |
| `vision/make_dataset.py` | builds the YOLO dataset with an inter-patient DS1/DS2 split and optional NSTDB noise |
| `vision/train_yolo.py` | trains YOLOv8 and evaluates on the test split (DS2) |
| `vision/infer.py` | YOLO and classical detectors, results in seconds |
| `app/streamlit_app.py` | interactive demo |

## Setup
```bash
source .venv/bin/activate
pip install -r requirements-vision.txt       # streamlit, plotly, pillow, ultralytics
```

## 1. Run the app (works immediately, no training needed)
```bash
streamlit run app/streamlit_app.py
```
Without trained weights the app uses the classical R-peak detector and says so. Choose the source
(synthetic ECG, an MIT-BIH record from `data/mitdb`, or your own CSV), add noise, pick a filter, and watch the boxes.

## 2. Build the YOLO dataset
```bash
python -m vision.make_dataset                          # clean excerpts, filtered with IIR (filtfilt)
python -m vision.make_dataset --noise-snrs 12 6        # also add noisy copies (bw, em, ma at 12 and 6 dB)
python -m vision.make_dataset --synthetic              # offline smoke test (fake labels, never report it)
```
With only the five default records you get DS1 = 101, 119, 207 and DS2 = 100, 200: good for a smoke test, too small for
real conclusions. Download more DS1/DS2 records (see `scripts` / `data/download_data.py`) and rerun. The script prints
the split and the number of boxes per class; look at it. Class F is rare and Q is excluded by default.

## 3. Train and evaluate
```bash
python -m vision.train_yolo --data datasets/ecg_yolo/dataset.yaml --epochs 30
```
Weights land in `runs/detect/ecg_yolo/weights/best.pt`, which is where the app looks by default.
Test-set (DS2) metrics are saved to `runs/detect/ecg_yolo/test_metrics.json`.

## Design decisions worth defending
* Noise is added to the whole excerpt, then filtered, then windowed: filters need seconds of context.
* Inter-patient split (de Chazal DS1/DS2). Validation records come from DS1. Noise for train/val comes from the first
  half of each NSTDB record and for test from the second half, so noise is never shared.
* Boxes are +/-0.2 s around the annotated R-peak and as tall as the trace. They locate beats; they do not delineate waves.
* Flips and colour jitter are off (a flipped ECG is a different waveform), mosaic is off by default.
* Per-window amplitude normalisation means absolute mV is not visible to the model.

## Limits to state when you present this
* Untested here: the YOLO training run and real-data detection quality. Report only numbers you produced yourself.
* Beat classification from images is not expected to beat a 1D CNN/LSTM by default; compare them on the same DS2 split.
* Five records is a smoke test, not a result.
* Ultralytics is AGPL-3.0. Educational demo, not a medical device.
