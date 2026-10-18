"""Path/flow tests for vision.train_yolo with a fake `ultralytics` module (the real one is heavy)."""
import json
import sys
import types
from pathlib import Path


def _install_fake(monkeypatch, tmp_path, record):
    class Box:
        map50, map, ap_class_index, ap50 = 0.7, 0.5, [0, 2], [0.9, 0.4]

    class Metrics:
        box, names = Box(), {0: "N", 1: "S", 2: "V", 3: "F"}

    class YOLO:
        def __init__(self, path):
            record["init"] = path
            self.trainer = types.SimpleNamespace(save_dir=Path(record["project"]) / "ecg_yolo" if "project" in record else tmp_path)

        def train(self, **kw):
            record["train"] = kw
            self.trainer = types.SimpleNamespace(save_dir=Path(kw["project"]) / kw["name"])
            (Path(kw["project"]) / kw["name"] / "weights").mkdir(parents=True, exist_ok=True)
            (Path(kw["project"]) / kw["name"] / "weights" / "best.pt").write_bytes(b"x")

        def val(self, **kw):
            record["val"] = kw
            return Metrics()

    mod = types.ModuleType("ultralytics")
    mod.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def test_train_uses_absolute_project_and_evaluates_saved_weights(monkeypatch, tmp_path):
    from vision import train_yolo
    rec = {}
    _install_fake(monkeypatch, tmp_path, rec)
    monkeypatch.chdir(tmp_path)
    res = train_yolo.main(["--data", "d.yaml", "--epochs", "1", "--project", "runs/detect"])
    assert Path(rec["train"]["project"]).is_absolute()            # the bug: a relative path got double-prefixed
    assert rec["train"]["patience"] == 8 and rec["val"]["split"] == "test"
    out = tmp_path / "runs" / "detect" / "ecg_yolo" / "test_metrics.json"
    assert out.exists() and json.loads(out.read_text())["ap50_per_class"] == {"N": 0.9, "V": 0.4}
    assert res["map50"] == 0.7


def test_eval_only_writes_metrics_next_to_the_run(monkeypatch, tmp_path):
    from vision import train_yolo
    rec = {}
    _install_fake(monkeypatch, tmp_path, rec)
    w = tmp_path / "somerun" / "weights" / "best.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"x")
    train_yolo.main(["--eval-only", "--weights", str(w), "--data", "d.yaml"])
    assert (tmp_path / "somerun" / "test_metrics.json").exists() and "train" not in rec
