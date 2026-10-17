"""Metrics, beat matching and R-peak detection."""
import numpy as np
import pytest

from src import io as ecgio
from src import metrics, noise, peaks, pipeline

RNG = np.random.default_rng(5)


def test_snr_db_known_value():
    ref = RNG.standard_normal(10000)
    n = RNG.standard_normal(10000)
    noisy, scaled, _ = noise.mix_at_snr(ref, n, 12.0)
    assert metrics.snr_db(ref, noisy) == pytest.approx(12.0, abs=1e-9)
    assert metrics.snr_improvement_db(ref, noisy, ref) > 100          # perfect output -> huge gain
    assert metrics.snr_improvement_db(ref, noisy, noisy) == pytest.approx(0.0, abs=1e-12)


def test_rmse_and_dc_invariance():
    ref = RNG.standard_normal(1000)
    assert metrics.rmse(ref, ref + 5.0) == pytest.approx(0.0, abs=1e-12)       # DC removed
    assert metrics.rmse(ref, ref + 0.1 * RNG.standard_normal(1000)) == pytest.approx(0.1, rel=0.1)


def test_trim_slice():
    with pytest.raises(ValueError):
        metrics.trim_slice(3600, 360.0, 5.0)
    s = metrics.trim_slice(36000, 360.0, 5.0)
    assert (s.start, s.stop) == (1800, 34200)


def test_match_peaks_counts():
    ref = np.array([100, 400, 700, 1000])
    det = np.array([102, 395, 705, 1500])      # last is a false positive, 1000 is missed
    assert metrics.match_peaks(ref, det, tol_samples=10) == (3, 1, 1)
    # one detection cannot match two references
    assert metrics.match_peaks(np.array([100, 105]), np.array([102]), 10) == (1, 0, 1)
    assert metrics.match_peaks(ref, np.array([], dtype=int), 10) == (0, 0, 4)


def test_peak_scores():
    s = metrics.peak_scores(90, 10, 30)
    assert s["se"] == pytest.approx(0.75) and s["ppv"] == pytest.approx(0.9)
    assert s["f1"] == pytest.approx(2 * 0.75 * 0.9 / 1.65)


def test_detector_on_clean_synthetic_ecg():
    rec = ecgio.synthetic_record(duration_s=120, seed=3)
    det = peaks.detect_r_peaks(rec["signal"], rec["fs"])
    tp, fp, fn = metrics.match_peaks(rec["ann_sample"], det, int(0.15 * rec["fs"]))
    sc = metrics.peak_scores(tp, fp, fn)
    assert sc["se"] > 0.97 and sc["ppv"] > 0.97
    # fine localisation: matched peaks are within 20 ms of the truth
    err = [np.min(np.abs(det - r)) for r in rec["ann_sample"][2:-2]]
    assert np.median(err) < 0.02 * rec["fs"]


def test_detector_survives_baseline_wander_after_filtering():
    rec = ecgio.synthetic_record(duration_s=120, seed=4)
    fs = rec["fs"]
    nz = noise.synthetic_noise("bw", len(rec["signal"]), fs, seed=1)
    noisy, _, _ = noise.mix_at_snr(rec["signal"], nz, 0.0)
    bank = pipeline.build_filter_bank(fs)
    cut = metrics.trim_slice(len(noisy), fs, 5.0)
    ref = rec["ann_sample"]
    ref = ref[(ref >= cut.start) & (ref < cut.stop)]
    for method in pipeline.METHODS:
        det = peaks.detect_r_peaks(bank.apply(method, noisy), fs)
        det = det[(det >= cut.start) & (det < cut.stop)]
        tp, fp, fn = metrics.match_peaks(ref, det, int(0.15 * fs))
        assert metrics.peak_scores(tp, fp, fn)["f1"] > 0.95, method


def test_filter_bank_improves_snr_for_bw_and_ma():
    rec = ecgio.synthetic_record(duration_s=120, seed=6)
    fs, x = rec["fs"], rec["signal"]
    bank = pipeline.build_filter_bank(fs)
    cut = metrics.trim_slice(len(x), fs, 5.0)
    for kind in ("bw", "ma"):
        nz = noise.synthetic_noise(kind, len(x), fs, seed=2)
        noisy, _, _ = noise.mix_at_snr(x, nz, 6.0)
        for method in ("FIR", "IIR (filtfilt)"):
            y = bank.apply(method, noisy)
            d = metrics.snr_improvement_db(x[cut], noisy[cut], y[cut])
            assert d > 3.0, (kind, method, d)


def test_filter_bank_unknown_method():
    bank = pipeline.build_filter_bank(360.0)
    with pytest.raises(ValueError):
        bank.apply("nope", np.zeros(100))
