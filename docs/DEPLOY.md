# Deploying the Streamlit app (permanent public URL, free)

Target: **Streamlit Community Cloud** (needs a GitHub repo; free). Facts from Streamlit's docs at the time of writing:
apps get roughly 0.7 to 2.7 GB of memory, apps with no traffic for 12 hours go to sleep (and wake when visited), and the
Python requirements file next to the app entrypoint takes precedence over the one in the repo root. Check the current
limits on share.streamlit.io before you rely on them.

Why this repo needs a little preparation: PyTorch and Ultralytics are far too heavy for that memory limit, and the hosted
app has no `data/mitdb`. So the hosted app runs YOLO through **onnxruntime** (small) and ships two short **real ECG excerpts**.

## 1. Export the model to ONNX and VERIFY it (on your laptop, about 10 minutes)
```bash
source .venv/bin/activate
pip install onnx onnxruntime
python -m vision.export_onnx --weights runs/detect/ecg_yolo/weights/best.pt --out models/ecg_yolo_best.onnx --check 40
```
It must end with `OK: the ONNX path reproduces the Ultralytics detections.` If it says the decoder disagrees, do not deploy:
send the output to the person helping you. (If `onnx` fails to install on a very new Python, run this step in a free Colab notebook instead.)

## 2. Create the demo excerpts (test records only, so the model check is fair)
```bash
python scripts/make_demo_excerpt.py --record 100 --record 200 --seconds 60
```

## 3. Test the lean setup locally
```bash
python -m venv /tmp/lean && /tmp/lean/bin/pip install -r app/requirements.txt
/tmp/lean/bin/streamlit run app/streamlit_app.py
```
In the sidebar the source should be "Bundled MIT-BIH demo", the detector YOLO, and below the plot you should see "loaded ecg_yolo_best.onnx (onnxruntime)".

## 4. Commit and push
```bash
git add app/streamlit_app.py app/requirements.txt app/sample_data vision/infer.py vision/onnx_infer.py vision/export_onnx.py \
        scripts/make_demo_excerpt.py models tests/test_onnx_infer.py docs/DEPLOY.md
git status      # expect: models/*.onnx and *.json, app/sample_data/*.npz, no runs/, datasets/, .pt, .venv
git commit -m "Hosting kit: ONNX inference, bundled demo excerpts, lean requirements"
git push
```

## 5. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub (authorise access to the repo).
2. **Create app** > repository `shaurXePo/ECG-Dsp-toolkit`, branch `main`, main file path `app/streamlit_app.py`.
3. **Advanced settings**: choose Python **3.12** (or 3.13). Optionally pick the subdomain name.
4. **Deploy**. The first build takes a few minutes. Your URL looks like `https://<name>.streamlit.app`.
5. Put the link in the README (badge below) and on your CV.

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
```

## Things to know
* **Sleeping:** after 12 quiet hours the app sleeps. Visitors can wake it (it takes a short while). Open it yourself before sharing the link.
* **Updates:** every `git push` to `main` redeploys automatically.
* **If it fails:** use "Manage app" for the logs. "Over its resource limits" means too much memory (lower "Analyse first"); a stuck build usually means a wrong Python version or a bad requirements line: reboot the app and check the logs.
* **Licence:** Ultralytics is AGPL-3.0. Keep the repository public and state in the README that the model was trained with Ultralytics.
* **Safety:** educational demo, not a medical device. The app keeps uploaded CSVs in memory only, but do not upload patient data to a public app.
