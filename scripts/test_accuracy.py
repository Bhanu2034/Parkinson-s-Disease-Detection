#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final verification: test all WAV files against model with the fixed app.py logic
(no mic override - pure model prediction).
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import parselmouth
from parselmouth.praat import call
import librosa
import soundfile as sf
import tempfile

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

def _load_pkl(path):
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pickle.load(f)

model = _load_pkl(os.path.join(MODELS_DIR, "parkinson_model.pkl"))
scaler = _load_pkl(os.path.join(MODELS_DIR, "scaler.pkl"))
FEATURE_NAMES = _load_pkl(os.path.join(MODELS_DIR, "feature_names.pkl"))

def safe(v, default=0.0):
    try:
        fv = float(v)
        return default if (np.isnan(fv) or np.isinf(fv)) else fv
    except Exception:
        return default

UCI_BOUNDS = {
    'MDVP:Fo(Hz)':       (88.33,  260.11),
    'MDVP:Fhi(Hz)':      (102.15, 592.03),
    'MDVP:Flo(Hz)':      (65.48,  239.17),
    'MDVP:Jitter(%)':    (0.0017, 0.0332),
    'MDVP:Jitter(Abs)':  (5e-6,   0.00026),
    'MDVP:RAP':          (0.0007, 0.0214),
    'MDVP:PPQ':          (0.0009, 0.0196),
    'Jitter:DDP':        (0.0020, 0.0643),
    'MDVP:Shimmer':      (0.0095, 0.1191),
    'MDVP:Shimmer(dB)':  (0.085,  1.302),
    'Shimmer:APQ3':      (0.0046, 0.0565),
    'Shimmer:APQ5':      (0.0057, 0.0794),
    'MDVP:APQ':          (0.0072, 0.1378),
    'Shimmer:DDA':       (0.0136, 0.1694),
    'NHR':               (0.0006, 0.3148),
    'HNR':               (8.44,   33.05),
}

def clip_to_uci_bounds(features):
    clipped = []
    for i, name in enumerate(FEATURE_NAMES):
        lo, hi = UCI_BOUNDS.get(name, (-np.inf, np.inf))
        clipped.append(float(np.clip(features[i], lo, hi)))
    return clipped

def extract_voice_features(audio_path):
    y_audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    tmp_wav = os.path.join(tempfile.gettempdir(), "pk_verify.wav")
    sf.write(tmp_wav, y_audio, sr)
    
    sound = parselmouth.Sound(tmp_wav)
    pitch_obj = call(sound, "To Pitch", 0.0, 75, 600)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    
    Fo  = safe(call(pitch_obj, "Get mean",    0, 0, "Hertz"),  154.0)
    Fhi = safe(call(pitch_obj, "Get maximum", 0, 0, "Hertz", "Parabolic"), 197.0)
    Flo = safe(call(pitch_obj, "Get minimum", 0, 0, "Hertz", "Parabolic"), 116.0)
    
    jitter_pct  = safe(call(point_process, "Get jitter (local)",            0, 0, 0.0001, 0.02, 1.3), 0.006)
    jitter_abs  = safe(call(point_process, "Get jitter (local, absolute)",  0, 0, 0.0001, 0.02, 1.3), 4.4e-5)
    jitter_rap  = safe(call(point_process, "Get jitter (rap)",              0, 0, 0.0001, 0.02, 1.3), 0.003)
    jitter_ppq5 = safe(call(point_process, "Get jitter (ppq5)",             0, 0, 0.0001, 0.02, 1.3), 0.003)
    jitter_ddp  = safe(call(point_process, "Get jitter (ddp)",              0, 0, 0.0001, 0.02, 1.3), 0.010)
    
    shimmer_loc  = safe(call([sound, point_process], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6), 0.030)
    shimmer_db   = safe(call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 0.280)
    shimmer_apq3 = safe(call([sound, point_process], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6), 0.016)
    shimmer_apq5 = safe(call([sound, point_process], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6), 0.018)
    shimmer_apq11= safe(call([sound, point_process], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6), 0.024)
    shimmer_dda  = safe(call([sound, point_process], "Get shimmer (dda)",      0, 0, 0.0001, 0.02, 1.3, 1.6), 0.047)
    
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR = safe(call(harmonicity, "Get mean", 0, 0), 21.0)
    if HNR <= 0:
        HNR = 21.0
    NHR = float(np.clip(1.0 / (10 ** (HNR / 10.0)), 0.00065, 0.315))
    
    return [Fo, Fhi, Flo, jitter_pct, jitter_abs, jitter_rap, jitter_ppq5,
            jitter_ddp, shimmer_loc, shimmer_db, shimmer_apq3, shimmer_apq5,
            shimmer_apq11, shimmer_dda, NHR, HNR]


def predict_wav(wav_path):
    """Exact same logic as the FIXED app.py /predict_mic endpoint."""
    feats = extract_voice_features(wav_path)
    feats = clip_to_uci_bounds(feats)
    X_scaled = scaler.transform(np.array([feats]))
    prob = float(model.predict_proba(X_scaled)[0][1])
    prob_pct = round(prob * 100, 2)
    
    if prob >= 0.5:
        result = "PD Detected"
        risk = "High" if prob_pct >= 70 else "Medium"
    else:
        result = "Healthy"
        risk = "Low"
    
    return prob, prob_pct, result, risk


def main():
    test_dir = os.path.join(DATA_DIR, "test_wavs")
    healthy_dir = os.path.join(test_dir, "healthy")
    pd_dir = os.path.join(test_dir, "pd")
    
    print("\n" + "="*80)
    print(" FINAL VERIFICATION: FIXED APP.PY (no mic override)")
    print("="*80)
    
    healthy_files = sorted([f for f in os.listdir(healthy_dir) if f.endswith('.wav')])
    pd_files = sorted([f for f in os.listdir(pd_dir) if f.endswith('.wav')])
    
    # HEALTHY
    print(f"\n--- HEALTHY ({len(healthy_files)} files) --- Expected: prob < 50%")
    print(f"  {'File':32s} {'Prob%':>8s} {'Result':>16s} {'Risk':>8s} {'Status':>8s}")
    print(f"  {'-'*75}")
    h_ok = 0
    for fname in healthy_files:
        prob, pct, result, risk = predict_wav(os.path.join(healthy_dir, fname))
        ok = prob < 0.5
        h_ok += int(ok)
        print(f"  {fname:32s} {pct:8.1f} {result:>16s} {risk:>8s} {'OK' if ok else 'FAIL':>8s}")
    
    # PD
    print(f"\n--- PD ({len(pd_files)} files) --- Expected: prob >= 50%")
    print(f"  {'File':32s} {'Prob%':>8s} {'Result':>16s} {'Risk':>8s} {'Status':>8s}")
    print(f"  {'-'*75}")
    p_ok = 0
    for fname in pd_files:
        prob, pct, result, risk = predict_wav(os.path.join(pd_dir, fname))
        ok = prob >= 0.5
        p_ok += int(ok)
        print(f"  {fname:32s} {pct:8.1f} {result:>16s} {risk:>8s} {'OK' if ok else 'FAIL':>8s}")
    
    total = len(healthy_files) + len(pd_files)
    total_ok = h_ok + p_ok
    
    print(f"\n{'='*80}")
    print(f" RESULTS:")
    print(f"   Healthy: {h_ok}/{len(healthy_files)} correct ({100*h_ok/max(1,len(healthy_files)):.0f}%)")
    print(f"   PD:      {p_ok}/{len(pd_files)} correct ({100*p_ok/max(1,len(pd_files)):.0f}%)")
    print(f"   TOTAL:   {total_ok}/{total} ({100*total_ok/max(1,total):.0f}%)")
    print(f"{'='*80}")
    
    # Standalone files
    print("\n--- STANDALONE FILES ---")
    for fname in ['sample_01.wav', 'sample_02.wav', 'parkinsons_5sec.wav']:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            prob, pct, result, risk = predict_wav(fpath)
            print(f"  {fname:32s} prob={pct:.1f}% -> {result} ({risk})")

if __name__ == "__main__":
    main()
