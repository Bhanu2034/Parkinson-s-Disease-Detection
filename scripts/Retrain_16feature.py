# -*- coding: utf-8 -*-
"""
PARKINSON'S MODEL RETRAINING — 16 PRAAT FEATURES ONLY
=======================================================
Trains ONLY on the 16 features that app.py actually extracts via Praat.
This eliminates the feature-count mismatch that caused the old model to
outperform the retrained one (old model was trained on 22 features but
inference was sending 16 — completely misaligned).

The 16 features used (matching UCI CSV column names):
  MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz),
  MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP,
  MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5,
  MDVP:APQ, Shimmer:DDA, NHR, HNR

Run from your project root:
  python retrain_16feature.py
"""

import os
import sys
import pickle
import json
import shutil
import numpy as np
import pandas as pd
import warnings
import tempfile

import soundfile as sf
import librosa
import parselmouth
from parselmouth.praat import call

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARNING] imbalanced-learn not installed. pip install imbalanced-learn")

SR = 22050  # Sample rate — must match extract_voice_features()

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — adjust BASE_DIR if your layout differs
# ─────────────────────────────────────────────────────────────────────────────
# NEW — go one level up from scripts/ to reach project root
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH  = os.path.join(DATA_DIR, "parkinsons.csv")

# ─────────────────────────────────────────────────────────────────────────────
# The exact 16 feature names used by app.py (and present in UCI CSV)
# Order MUST match what extract_voice_features() returns
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES_16 = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
]

UCI_BOUNDS = {
    "MDVP:Fo(Hz)":       (88.33,  260.11),
    "MDVP:Fhi(Hz)":      (102.15, 592.03),
    "MDVP:Flo(Hz)":      (65.48,  239.17),
    "MDVP:Jitter(%)":    (0.0017, 0.0332),
    "MDVP:Jitter(Abs)":  (5e-6,   0.00026),
    "MDVP:RAP":          (0.0007, 0.0214),
    "MDVP:PPQ":          (0.0009, 0.0196),
    "Jitter:DDP":        (0.0020, 0.0643),
    "MDVP:Shimmer":      (0.0095, 0.1191),
    "MDVP:Shimmer(dB)":  (0.085,  1.302),
    "Shimmer:APQ3":      (0.0046, 0.0565),
    "Shimmer:APQ5":      (0.0057, 0.0794),
    "MDVP:APQ":          (0.0072, 0.1378),
    "Shimmer:DDA":       (0.0136, 0.1694),
    "NHR":               (0.0006, 0.3148),
    "HNR":               (8.44,   33.05),
}


def safe(v, default=0.0):
    try:
        fv = float(v)
        return default if (np.isnan(fv) or np.isinf(fv)) else fv
    except Exception:
        return default


def clip_to_uci_bounds(features):
    return [float(np.clip(features[i], *UCI_BOUNDS.get(FEATURE_NAMES_16[i], (-np.inf, np.inf))))
            for i in range(len(FEATURE_NAMES_16))]


# ─────────────────────────────────────────────────────────────────────────────
# WAV feature extraction  — IDENTICAL to the fixed app.py pipeline
# (sr=22050, pitch ceiling 600 Hz to match UCI range, no pre-emphasis)
# ─────────────────────────────────────────────────────────────────────────────
def extract_voice_features(audio_path):
    """
    Extract the same 16 features as the fixed app.py.
    KEY SETTINGS (must match app.py exactly):
      - sr = 22050
      - pitch floor = 75 Hz, ceiling = 600 Hz  (UCI range)
      - HNR clipped before NHR is derived
    """
    y_audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    if len(y_audio) < sr * 0.5:
        raise ValueError("Audio too short.")

    rms = float(np.sqrt(np.mean(y_audio ** 2)))
    if rms < 0.005:
        raise ValueError("Recording level too low.")

    y_trimmed, _ = librosa.effects.trim(y_audio, top_db=25)
    if len(y_trimmed) < sr * 0.3:
        y_trimmed = y_audio

    tmp_wav = os.path.join(tempfile.gettempdir(), "pk_retrain.wav")
    sf.write(tmp_wav, y_trimmed, sr)

    sound         = parselmouth.Sound(tmp_wav)
    pitch_obj     = call(sound, "To Pitch", 0.0, 75, 600)   # 600 Hz ceiling — UCI range
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

    Fo  = safe(call(pitch_obj, "Get mean",    0, 0, "Hertz"),  154.23)
    Fhi = safe(call(pitch_obj, "Get maximum", 0, 0, "Hertz", "Parabolic"), 197.10)
    Flo = safe(call(pitch_obj, "Get minimum", 0, 0, "Hertz", "Parabolic"), 116.32)

    jitter_pct  = safe(call(point_process, "Get jitter (local)",            0, 0, 0.0001, 0.02, 1.3), 0.00545)
    jitter_abs  = safe(call(point_process, "Get jitter (local, absolute)",  0, 0, 0.0001, 0.02, 1.3), 3.30e-5)
    jitter_rap  = safe(call(point_process, "Get jitter (rap)",              0, 0, 0.0001, 0.02, 1.3), 0.00280)
    jitter_ppq5 = safe(call(point_process, "Get jitter (ppq5)",             0, 0, 0.0001, 0.02, 1.3), 0.00300)
    jitter_ddp  = safe(call(point_process, "Get jitter (ddp)",              0, 0, 0.0001, 0.02, 1.3), 0.00839)

    shimmer_loc  = safe(call([sound, point_process], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6), 0.03680)
    shimmer_db   = safe(call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 0.32500)
    shimmer_apq3 = safe(call([sound, point_process], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6), 0.01870)
    shimmer_apq5 = safe(call([sound, point_process], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6), 0.02290)
    shimmer_apq11= safe(call([sound, point_process], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6), 0.02740)
    shimmer_dda  = safe(call([sound, point_process], "Get shimmer (dda)",      0, 0, 0.0001, 0.02, 1.3, 1.6), 0.05610)

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR_raw = safe(call(harmonicity, "Get mean", 0, 0), 21.92)
    if HNR_raw <= 0:
        HNR_raw = 21.92
    HNR = float(np.clip(HNR_raw, 8.44, 33.05))   # clip FIRST
    NHR = float(np.clip(1.0 / (10 ** (HNR / 10.0)), 0.00065, 0.315))  # derive from clipped

    return [Fo, Fhi, Flo, jitter_pct, jitter_abs, jitter_rap, jitter_ppq5,
            jitter_ddp, shimmer_loc, shimmer_db, shimmer_apq3, shimmer_apq5,
            shimmer_apq11, shimmer_dda, NHR, HNR]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic WAV generation for domain-gap bridging augmentation
# ─────────────────────────────────────────────────────────────────────────────
def generate_voice_like(duration_sec, base_f0_hz, jitter_amount=0.02,
                        shimmer_amount=0.03, noise_ratio=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = int(duration_sec * SR)
    t = np.arange(n, dtype=np.float64) / SR
    phase = 2 * np.pi * base_f0_hz * t
    jitter_walk = np.cumsum(jitter_amount * (np.random.rand(n) - 0.5))
    phase = phase + jitter_walk
    env = 1.0 + shimmer_amount * (np.random.rand(n) - 0.5)
    env = np.clip(env, 0.3, 1.0)
    y = np.zeros(n)
    for h in [1, 2, 3, 4]:
        y += (1.0 / h) * np.sin(h * phase)
    y *= env
    if noise_ratio > 0:
        y = y + np.random.randn(n) * noise_ratio
    y *= 0.3 / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def generate_augmentation_wavs(tmp_dir):
    """
    Generate synthetic WAV pairs (healthy + PD) and extract their Praat features.
    These bridge the domain gap between UCI CSV measurements and real mic audio.
    Returns X_wav (n, 16), y_wav (n,).
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # (label, f0, jitter, shimmer, noise, seed)  label: 0=healthy  1=PD
    configs = [
        # ── HEALTHY (low jitter/shimmer, no noise, various F0) ──────────────
        (0, 110, 0.002, 0.006, 0.00, 1001),
        (0, 120, 0.003, 0.008, 0.00, 1002),
        (0, 130, 0.002, 0.005, 0.00, 1003),
        (0, 140, 0.004, 0.010, 0.00, 1004),
        (0, 150, 0.003, 0.007, 0.00, 1005),
        (0, 180, 0.002, 0.006, 0.00, 1101),
        (0, 200, 0.003, 0.008, 0.00, 1102),
        (0, 220, 0.002, 0.005, 0.00, 1103),
        (0, 190, 0.004, 0.010, 0.00, 1104),
        (0, 210, 0.003, 0.007, 0.00, 1105),
        # consumer-mic simulation (slightly higher jitter/shimmer but still healthy)
        (0, 150, 0.005, 0.012, 0.01, 1201),
        (0, 160, 0.006, 0.014, 0.01, 1202),
        (0, 170, 0.005, 0.011, 0.01, 1203),
        (0, 145, 0.006, 0.013, 0.02, 1204),
        (0, 185, 0.006, 0.015, 0.01, 1205),
        # ── PD (moderate) ───────────────────────────────────────────────────
        (1, 130, 0.012, 0.045, 0.08, 2001),
        (1, 140, 0.014, 0.050, 0.09, 2002),
        (1, 150, 0.011, 0.042, 0.07, 2003),
        (1, 135, 0.013, 0.048, 0.08, 2004),
        (1, 145, 0.015, 0.055, 0.10, 2005),
        (1, 160, 0.012, 0.044, 0.08, 2006),
        (1, 125, 0.014, 0.050, 0.09, 2007),
        (1, 155, 0.011, 0.043, 0.07, 2008),
        # ── PD (severe) ─────────────────────────────────────────────────────
        (1, 130, 0.022, 0.070, 0.12, 2101),
        (1, 140, 0.025, 0.080, 0.14, 2102),
        (1, 120, 0.020, 0.065, 0.11, 2103),
        (1, 135, 0.028, 0.085, 0.15, 2104),
        (1, 150, 0.024, 0.075, 0.13, 2105),
        (1, 115, 0.030, 0.090, 0.16, 2106),
        (1, 110, 0.032, 0.095, 0.17, 2107),
    ]

    X_wav, y_wav = [], []
    for idx, (label, f0, jit, shim, noise, seed) in enumerate(configs):
        wav_path = os.path.join(tmp_dir, f"synth_{idx:03d}.wav")
        y_sig = generate_voice_like(3.0, f0, jit, shim, noise, seed)
        sf.write(wav_path, y_sig, SR)
        try:
            feats = extract_voice_features(wav_path)
            feats = clip_to_uci_bounds(feats)
            X_wav.append(feats)
            y_wav.append(label)
        except Exception as e:
            print(f"  [aug] Skipping synth_{idx:03d}: {e}")

    return np.array(X_wav), np.array(y_wav)


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(" PARKINSON'S MODEL RETRAINING — 16 PRAAT FEATURES")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load UCI CSV, select only the 16 features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading UCI dataset (16 features)...")
df = pd.read_csv(CSV_PATH)
if "name" in df.columns:
    df = df.drop(columns=["name"])

missing = [c for c in FEATURE_NAMES_16 if c not in df.columns]
if missing:
    print(f"    ERROR — missing columns in CSV: {missing}")
    sys.exit(1)

X_uci = df[FEATURE_NAMES_16].values
y_uci = df["status"].values

unique, counts = np.unique(y_uci, return_counts=True)
print(f"    Rows: {len(df)}  |  Healthy: {counts[0]}  |  PD: {counts[1]}")
print(f"    Class ratio: {counts[1]/counts[0]:.1f}:1 (PD:Healthy)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Generate WAV augmentation data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Generating synthetic WAV augmentation data...")
tmp_wav_dir = os.path.join(tempfile.gettempdir(), "pk_aug_wavs")
X_wav, y_wav = generate_augmentation_wavs(tmp_wav_dir)
u_w, c_w = np.unique(y_wav, return_counts=True)
print(f"    Generated: {len(X_wav)} samples  |  Healthy: {c_w[0]}  |  PD: {c_w[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Combine UCI + WAV data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Combining UCI + WAV data...")
X_combined = np.vstack([X_uci, X_wav])
y_combined  = np.concatenate([y_uci, y_wav])
u_c, c_c = np.unique(y_combined, return_counts=True)
print(f"    Combined: {len(X_combined)} samples  |  Healthy: {c_c[0]}  |  PD: {c_c[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Fit scaler on combined data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Fitting MinMaxScaler on combined data...")
scaler = MinMaxScaler()
X_combined_scaled = scaler.fit_transform(X_combined)
X_uci_scaled      = scaler.transform(X_uci)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SMOTE / class-weight balancing
# ─────────────────────────────────────────────────────────────────────────────
if HAS_SMOTE:
    print("\n[5] Applying SMOTE to balance classes...")
    min_class_count = min(c_c)
    sm = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
    X_resampled, y_resampled = sm.fit_resample(X_combined_scaled, y_combined)
    u_r, c_r = np.unique(y_resampled, return_counts=True)
    print(f"    After SMOTE: {dict(zip(u_r.tolist(), c_r.tolist()))}")
else:
    print("\n[5] No SMOTE — using class_weight='balanced' in models.")
    X_resampled, y_resampled = X_combined_scaled, y_combined

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Train and evaluate candidate models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Training candidate models (10-fold stratified CV)...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

candidates = {
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=3, random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=6, min_samples_leaf=3, random_state=42,
    ),
    "SVM_rbf": SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True, random_state=42,
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=42,
    ),
}

results       = {}
trained_models = {}

for name, clf in candidates.items():
    bal_scores = cross_val_score(clf, X_resampled, y_resampled,
                                 cv=cv, scoring="balanced_accuracy")
    roc_scores = cross_val_score(clf, X_resampled, y_resampled,
                                 cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(clf, X_combined_scaled, y_combined,
                                 cv=cv, scoring="accuracy")
    clf.fit(X_resampled, y_resampled)
    trained_models[name] = clf
    results[name] = {
        "balanced_accuracy": round(bal_scores.mean(), 4),
        "roc_auc":           round(roc_scores.mean(), 4),
        "accuracy":          round(acc_scores.mean(), 4),
    }
    print(f"  {name}:")
    print(f"    Balanced Acc = {bal_scores.mean():.4f}  "
          f"ROC-AUC = {roc_scores.mean():.4f}  "
          f"Acc = {acc_scores.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Pick best model by balanced accuracy
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Selecting best model...")
best_name = max(results, key=lambda k: results[k]["balanced_accuracy"])
best_clf  = trained_models[best_name]
print(f"    Best: {best_name}  (balanced_acc={results[best_name]['balanced_accuracy']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Probability calibration
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Calibrating probabilities (Platt scaling)...")
calibrated_clf = CalibratedClassifierCV(best_clf, cv=5, method="sigmoid")
calibrated_clf.fit(X_resampled, y_resampled)

# Sanity check on UCI data
proba_uci_all = calibrated_clf.predict_proba(X_uci_scaled)[:, 1]
h_proba = proba_uci_all[y_uci == 0]
p_proba = proba_uci_all[y_uci == 1]
print(f"    UCI Healthy  -> mean prob = {h_proba.mean():.4f}  "
      f"range [{h_proba.min():.4f}, {h_proba.max():.4f}]")
print(f"    UCI PD       -> mean prob = {p_proba.mean():.4f}  "
      f"range [{p_proba.min():.4f}, {p_proba.max():.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Find optimal threshold on combined data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Finding optimal decision threshold...")
proba_combined_all = calibrated_clf.predict_proba(X_combined_scaled)[:, 1]
best_thresh, best_bal = 0.5, 0.0
for thresh in np.arange(0.30, 0.75, 0.01):
    preds = (proba_combined_all >= thresh).astype(int)
    bal   = balanced_accuracy_score(y_combined, preds)
    if bal > best_bal:
        best_bal, best_thresh = bal, thresh

print(f"    Optimal threshold = {best_thresh:.2f}  (balanced acc = {best_bal:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Final evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Final evaluation on UCI data...")
preds_uci_opt = (proba_uci_all >= best_thresh).astype(int)
cm  = confusion_matrix(y_uci, preds_uci_opt)
print(f"    Accuracy:          {accuracy_score(y_uci, preds_uci_opt):.4f}")
print(f"    Balanced Accuracy: {balanced_accuracy_score(y_uci, preds_uci_opt):.4f}")
print(f"    ROC-AUC:           {roc_auc_score(y_uci, proba_uci_all):.4f}")
print(f"    Confusion matrix:")
print(f"      Healthy: TN={cm[0,0]}  FP={cm[0,1]}  "
      f"({100*cm[0,0]/(cm[0,0]+cm[0,1]+1e-9):.1f}% correct)")
print(f"      PD:      FN={cm[1,0]}  TP={cm[1,1]}  "
      f"({100*cm[1,1]/(cm[1,0]+cm[1,1]+1e-9):.1f}% correct)")
print(classification_report(y_uci, preds_uci_opt, target_names=["Healthy", "PD"]))

print("\n[10b] WAV augmentation data accuracy...")
proba_wav_all = calibrated_clf.predict_proba(scaler.transform(X_wav))[:, 1]
preds_wav     = (proba_wav_all >= best_thresh).astype(int)
wav_acc = (preds_wav == y_wav).mean()
print(f"    WAV accuracy: {wav_acc:.4f}  ({(preds_wav==y_wav).sum()}/{len(y_wav)})")
for i in range(len(y_wav)):
    expected = "Healthy" if y_wav[i] == 0 else "PD"
    got      = "Healthy" if preds_wav[i] == 0 else "PD"
    ok       = "✓" if y_wav[i] == preds_wav[i] else "✗"
    print(f"    synth_{i:03d}: expected={expected:7s}  prob={proba_wav_all[i]:.3f}  "
          f"got={got:7s}  {ok}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: Save artifacts
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Saving model artifacts...")
os.makedirs(MODEL_DIR, exist_ok=True)

for fname in ["parkinson_model.pkl", "scaler.pkl", "feature_names.pkl"]:
    src = os.path.join(MODEL_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(MODEL_DIR, fname + ".bak"))
        print(f"    Backed up {fname}")

with open(os.path.join(MODEL_DIR, "parkinson_model.pkl"), "wb") as f:
    pickle.dump(calibrated_clf, f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(FEATURE_NAMES_16, f)

# Feature stats (healthy vs PD means, for reference)
healthy_df_uci = df[df["status"] == 0]
pd_df_uci      = df[df["status"] == 1]
feature_stats  = {}
for fn in FEATURE_NAMES_16:
    feature_stats[fn] = {
        "healthy_mean": float(healthy_df_uci[fn].mean()),
        "healthy_std":  float(healthy_df_uci[fn].std()),
        "pd_mean":      float(pd_df_uci[fn].mean()),
        "pd_std":       float(pd_df_uci[fn].std()),
        "overall_min":  float(df[fn].min()),
        "overall_max":  float(df[fn].max()),
    }
with open(os.path.join(MODEL_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

model_meta = {
    "best_model_name":       best_name,
    "n_features":            16,
    "feature_names":         FEATURE_NAMES_16,
    "balanced_accuracy_cv":  results[best_name]["balanced_accuracy"],
    "roc_auc_cv":            results[best_name]["roc_auc"],
    "accuracy_cv":           results[best_name]["accuracy"],
    "final_uci_accuracy":    round(accuracy_score(y_uci, preds_uci_opt), 4),
    "final_uci_balanced_acc":round(balanced_accuracy_score(y_uci, preds_uci_opt), 4),
    "final_roc_auc":         round(roc_auc_score(y_uci, proba_uci_all), 4),
    "optimal_threshold":     round(float(best_thresh), 4),
    "n_uci_samples":         int(len(df)),
    "n_wav_augmentation":    int(len(X_wav)),
    "smote_applied":         HAS_SMOTE,
    "all_model_results":     results,
}
with open(os.path.join(MODEL_DIR, "model_results.json"), "w") as f:
    json.dump(model_meta, f, indent=2)

print(f"\nSaved to {MODEL_DIR}:")
print("  parkinson_model.pkl   (calibrated, 16-feature, WAV-augmented)")
print("  scaler.pkl")
print("  feature_names.pkl     (16 features only)")
print("  feature_stats.json")
print("  model_results.json")

print("\n" + "=" * 65)
print(" RETRAINING COMPLETE")
print(f" Best model:          {best_name}")
print(f" Optimal threshold:   {best_thresh:.2f}")
print(f" UCI Balanced Acc:    {balanced_accuracy_score(y_uci, preds_uci_opt):.1%}")
print(f" UCI ROC-AUC:         {roc_auc_score(y_uci, proba_uci_all):.4f}")
print(f" WAV accuracy:        {wav_acc:.1%}")
print("=" * 65)
print("\nNEXT STEPS:")
print("  1. Copy retrain_16feature.py to your project root and run it.")
print("  2. Replace app.py with the fixed version (sr=22050, pitch ceil=600).")
print("  3. Verify with test_accuracy.py.")