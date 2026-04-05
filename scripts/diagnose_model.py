# -*- coding: utf-8 -*-
"""
DIAGNOSTIC SCRIPT - WAV files only (CSV already confirmed working perfectly)
"""

import os
import sys
import pickle
import warnings
import numpy as np
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

def _load_pkl(path):
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pickle.load(f)

model         = _load_pkl(os.path.join(MODELS_DIR, "parkinson_model.pkl"))
scaler        = _load_pkl(os.path.join(MODELS_DIR, "scaler.pkl"))
FEATURE_NAMES = _load_pkl(os.path.join(MODELS_DIR, "feature_names.pkl"))

import parselmouth
from parselmouth.praat import call
import librosa
import soundfile as sf

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
    'NHR':               (0.00065, 0.3148),
    'HNR':               (8.44,   33.05),
}

def extract_voice_features(audio_path):
    y_audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    if len(y_audio) < sr * 0.5:
        raise ValueError("Audio too short")

    tmp_wav = os.path.join(tempfile.gettempdir(), "pk_diag.wav")
    sf.write(tmp_wav, y_audio, sr)

    sound         = parselmouth.Sound(tmp_wav)
    pitch_obj     = call(sound, "To Pitch", 0.0, 75, 600)
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

    features = [Fo, Fhi, Flo, jitter_pct, jitter_abs, jitter_rap, jitter_ppq5, jitter_ddp,
                 shimmer_loc, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq11,
                 shimmer_dda, NHR, HNR]
    return features

def clip_to_uci_bounds(features):
    clipped = []
    for i, name in enumerate(FEATURE_NAMES):
        lo, hi = UCI_BOUNDS.get(name, (-np.inf, np.inf))
        clipped.append(float(np.clip(features[i], lo, hi)))
    return clipped

# Load UCI stats for comparison
import json
stats_path = os.path.join(MODELS_DIR, "feature_stats.json")
with open(stats_path) as f:
    fstats = json.load(f)

# Test ALL WAV files
test_dirs = {
    "healthy": os.path.join(DATA_DIR, "test_wavs", "healthy"),
    "pd":      os.path.join(DATA_DIR, "test_wavs", "pd"),
}

results = {"healthy": [], "pd": []}

for label, folder in test_dirs.items():
    if not os.path.exists(folder):
        print(f"  {folder} not found, skipping.")
        continue
    
    wavs = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
    print(f"\nTesting {label.upper()} WAVs ({len(wavs)} files):")
    print("-" * 80)
    
    for wav_name in wavs:
        wav_path = os.path.join(folder, wav_name)
        try:
            raw_feats = extract_voice_features(wav_path)
            clipped   = clip_to_uci_bounds(raw_feats)
            X_wav     = np.array([clipped])
            X_wav_sc  = scaler.transform(X_wav)
            prob      = float(model.predict_proba(X_wav_sc)[0][1])
            pred      = "PD" if prob >= 0.5 else "Healthy"
            expected  = "PD" if label == "pd" else "Healthy"
            match     = "OK" if pred == expected else "WRONG"
            
            results[label].append({"file": wav_name, "prob": prob, "pred": pred, "expected": expected, "match": match == "OK"})
            
            print(f"  [{match:5s}] {wav_name}: prob={prob:.4f} -> {pred}")
            print(f"         Fo={clipped[0]:.1f}  Jitter={clipped[3]:.6f}  Shimmer={clipped[8]:.4f}  HNR={clipped[15]:.1f}  NHR={clipped[14]:.6f}")
                
        except Exception as e:
            print(f"  [ERROR] {wav_name}: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for label in ["healthy", "pd"]:
    correct = sum(1 for r in results[label] if r["match"])
    total   = len(results[label])
    probs   = [r["prob"] for r in results[label]]
    if probs:
        print(f"\n{label.upper()} ({total} files):")
        print(f"  Correct: {correct}/{total} ({100*correct/total:.0f}%)")
        print(f"  Prob range: {min(probs):.4f} - {max(probs):.4f}")
        print(f"  Prob mean:  {np.mean(probs):.4f}")
    else:
        print(f"\n{label.upper()}: No results")

# Also test standalone samples
print("\n\nSTANDALONE SAMPLES:")
print("-" * 80)
for sample_name in ["sample_01.wav", "sample_02.wav", "parkinsons_5sec.wav"]:
    sample_path = os.path.join(DATA_DIR, sample_name)
    if not os.path.exists(sample_path):
        continue
    try:
        raw_feats = extract_voice_features(sample_path)
        clipped   = clip_to_uci_bounds(raw_feats)
        X_wav     = np.array([clipped])
        X_wav_sc  = scaler.transform(X_wav)
        prob      = float(model.predict_proba(X_wav_sc)[0][1])
        pred      = "PD" if prob >= 0.5 else "Healthy"
        print(f"  {sample_name}: prob={prob:.4f} -> {pred}")
        print(f"    Fo={clipped[0]:.1f}  Jitter={clipped[3]:.6f}  Shimmer={clipped[8]:.4f}  HNR={clipped[15]:.1f}")
    except Exception as e:
        print(f"  {sample_name}: ERROR - {e}")
