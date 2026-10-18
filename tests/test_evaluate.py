import numpy as np

from vision.evaluate import MISSED, match_detections


def test_matching_counts_and_confusion():
    ref_t, ref_c = [1.0, 2.0, 3.0, 4.0], ["N", "V", "N", "S"]
    det_t, det_c = [1.05, 2.02, 5.0], ["N", "N", "V"]          # N ok, V called N, N(3.0) missed, S missed, 1 false positive
    r = match_detections(ref_t, ref_c, det_t, det_c)
    assert r["matched"] == 2 and r["missed"] == 2 and r["false_positives"] == 1
    assert r["matrix"]["N"]["N"] == 1 and r["matrix"]["N"][MISSED] == 1
    assert r["matrix"]["V"]["N"] == 1 and r["matrix"]["S"][MISSED] == 1
    assert r["per_class"]["V"]["wrong_class"] == 1 and r["per_class"]["N"]["sensitivity"] == 0.5
    assert np.isclose(r["sens_loc"], 0.5) and np.isclose(r["ppv_loc"], 2 / 3) and r["label_aware"]


def test_tolerance_and_one_to_one():
    r = match_detections([1.0, 1.1], ["N", "N"], [1.05], ["N"], tol_s=0.15)   # one detection cannot match two beats
    assert r["matched"] == 1 and r["missed"] == 1 and r["false_positives"] == 0
    assert match_detections([1.0], ["N"], [1.2], ["N"], tol_s=0.15)["matched"] == 0


def test_classical_detector_is_location_only():
    r = match_detections([1.0, 2.0], ["N", "V"], [1.0, 2.0], ["R", "R"])
    assert not r["label_aware"] and r["sens_loc"] == 1.0 and r["ppv_loc"] == 1.0


def test_empty_inputs():
    r = match_detections([], [], [], [])
    assert r["n_ref"] == 0 and np.isnan(r["sens_loc"]) and np.isnan(r["ppv_loc"])
