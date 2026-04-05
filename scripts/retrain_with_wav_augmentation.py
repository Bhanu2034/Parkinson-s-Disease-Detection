# -*- coding: utf-8 -*-
"""
PARKINSON'S MODEL RETRAINING WITH WAV AUGMENTATION
====================================================
Fixes the domain gap between UCI CSV features and WAV-extracted Praat features.

Strategy:
1. Generate synthetic healthy + PD WAV files with controlled acoustic properties
2. Extract the same 16 Praat features used in the app
3. Augment the UCI training data with WAV-extracted features
4. Retrain with SMOTE + class balancing + probability calibration
5. Validate on both CSV and WAV data
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

import soundfile as sf
import librosa
import parselmouth
from parselmouth.praat import call

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(DATA_DIR, "parkinsons.csv")
TEST_WAV_DIR = os.path.join(DATA_DIR, "test_wavs")
SR = 22050


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: safe float
# ─────────────────────────────────────────────────────────────────────────────
def safe(v, default=0.0):
    try:
        fv = float(v)
        return default if (np.isnan(fv) or np.isinf(fv)) else fv
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Generate synthetic WAV files
# ─────────────────────────────────────────────────────────────────────────────
def generate_voice_like(duration_sec, base_f0_hz, jitter_amount=0.02,
                        shimmer_amount=0.03, noise_ratio=0.0, seed=None):
    """
    Generate a synthetic vowel-like signal that Praat can analyze.
    - jitter_amount controls cycle-to-cycle F0 variation
    - shimmer_amount controls cycle-to-cycle amplitude variation
    - noise_ratio adds broadband noise (lowers HNR)
    """
    if seed is not None:
        np.random.seed(seed)
    n = int(duration_sec * SR)
    t = np.arange(n, dtype=np.float64) / SR

    # Phase with jitter (cycle-to-cycle period variation)
    phase = 2 * np.pi * base_f0_hz * t
    jitter_walk = np.cumsum(jitter_amount * (np.random.rand(n) - 0.5))
    phase = phase + jitter_walk

    # Amplitude envelope with shimmer
    env = 1.0 + shimmer_amount * (np.random.rand(n) - 0.5)
    env = np.clip(env, 0.3, 1.0)

    # Sum of harmonics (vowel-like)
    y = np.zeros(n)
    for h in [1, 2, 3, 4]:
        y += (1.0 / h) * np.sin(h * phase)
    y *= env

    # Add noise
    if noise_ratio > 0:
        noise = np.random.randn(n).astype(np.float64) * noise_ratio
        y = y + noise

    y *= 0.3 / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def generate_test_wavs():
    """Generate healthy and PD WAV files for training augmentation and testing."""
    healthy_dir = os.path.join(TEST_WAV_DIR, "healthy")
    pd_dir = os.path.join(TEST_WAV_DIR, "pd")
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(pd_dir, exist_ok=True)

    # ── HEALTHY VOICES ──
    # Characteristics: low jitter (0.001-0.006), low shimmer (0.005-0.015),
    #                  no/minimal noise (high HNR 22-33)
    #                  various pitch ranges
    healthy_configs = [
        # (filename, duration, f0, jitter, shimmer, noise, seed)
        # Male-range voices (~100-150 Hz)
        ("healthy_male_01.wav",  3.0, 110, 0.002, 0.006, 0.00, 1001),
        ("healthy_male_02.wav",  3.0, 120, 0.003, 0.008, 0.00, 1002),
        ("healthy_male_03.wav",  2.8, 130, 0.002, 0.005, 0.00, 1003),
        ("healthy_male_04.wav",  3.2, 115, 0.004, 0.010, 0.00, 1004),
        ("healthy_male_05.wav",  3.0, 125, 0.003, 0.007, 0.00, 1005),
        ("healthy_male_06.wav",  2.5, 135, 0.002, 0.006, 0.00, 1006),
        ("healthy_male_07.wav",  3.0, 105, 0.003, 0.008, 0.00, 1007),
        ("healthy_male_08.wav",  3.0, 140, 0.004, 0.009, 0.00, 1008),
        ("healthy_male_09.wav",  2.8, 118, 0.002, 0.005, 0.00, 1009),
        ("healthy_male_10.wav",  3.0, 128, 0.003, 0.007, 0.00, 1010),
        # Female-range voices (~170-250 Hz)
        ("healthy_female_01.wav", 3.0, 180, 0.002, 0.006, 0.00, 1101),
        ("healthy_female_02.wav", 3.0, 200, 0.003, 0.008, 0.00, 1102),
        ("healthy_female_03.wav", 2.8, 220, 0.002, 0.005, 0.00, 1103),
        ("healthy_female_04.wav", 3.2, 190, 0.004, 0.010, 0.00, 1104),
        ("healthy_female_05.wav", 3.0, 210, 0.003, 0.007, 0.00, 1105),
        ("healthy_female_06.wav", 2.5, 230, 0.002, 0.006, 0.00, 1106),
        ("healthy_female_07.wav", 3.0, 175, 0.003, 0.008, 0.00, 1107),
        ("healthy_female_08.wav", 3.0, 245, 0.004, 0.009, 0.00, 1108),
        ("healthy_female_09.wav", 2.8, 195, 0.002, 0.005, 0.00, 1109),
        ("healthy_female_10.wav", 3.0, 215, 0.003, 0.007, 0.00, 1110),
        # Slightly noisy but still healthy (consumer mic simulation)
        ("healthy_mic_01.wav",   3.0, 150, 0.005, 0.012, 0.01, 1201),
        ("healthy_mic_02.wav",   3.0, 160, 0.006, 0.014, 0.01, 1202),
        ("healthy_mic_03.wav",   2.8, 170, 0.005, 0.011, 0.01, 1203),
        ("healthy_mic_04.wav",   3.0, 145, 0.006, 0.013, 0.02, 1204),
        ("healthy_mic_05.wav",   3.0, 155, 0.005, 0.012, 0.02, 1205),
        ("healthy_mic_06.wav",   3.0, 185, 0.006, 0.015, 0.01, 1206),
        ("healthy_mic_07.wav",   2.8, 140, 0.005, 0.011, 0.02, 1207),
        ("healthy_mic_08.wav",   3.2, 165, 0.006, 0.014, 0.01, 1208),
        ("healthy_mic_09.wav",   3.0, 175, 0.004, 0.010, 0.02, 1209),
        ("healthy_mic_10.wav",   3.0, 190, 0.005, 0.012, 0.01, 1210),
    ]

    # ── PD VOICES ──
    # Characteristics: high jitter (0.010-0.035), high shimmer (0.04-0.10),
    #                  significant noise (low HNR 8-18)
    pd_configs = [
        # Moderate PD
        ("pd_moderate_01.wav",  3.0, 130, 0.012, 0.045, 0.08, 2001),
        ("pd_moderate_02.wav",  3.0, 140, 0.014, 0.050, 0.09, 2002),
        ("pd_moderate_03.wav",  2.8, 150, 0.011, 0.042, 0.07, 2003),
        ("pd_moderate_04.wav",  3.2, 135, 0.013, 0.048, 0.08, 2004),
        ("pd_moderate_05.wav",  3.0, 145, 0.015, 0.055, 0.10, 2005),
        ("pd_moderate_06.wav",  3.0, 160, 0.012, 0.044, 0.08, 2006),
        ("pd_moderate_07.wav",  2.8, 125, 0.014, 0.050, 0.09, 2007),
        ("pd_moderate_08.wav",  3.0, 155, 0.011, 0.043, 0.07, 2008),
        ("pd_moderate_09.wav",  3.0, 120, 0.013, 0.047, 0.08, 2009),
        ("pd_moderate_10.wav",  3.0, 165, 0.016, 0.058, 0.10, 2010),
        # Severe PD
        ("pd_severe_01.wav",   3.0, 130, 0.022, 0.070, 0.12, 2101),
        ("pd_severe_02.wav",   3.0, 140, 0.025, 0.080, 0.14, 2102),
        ("pd_severe_03.wav",   2.8, 120, 0.020, 0.065, 0.11, 2103),
        ("pd_severe_04.wav",   3.2, 135, 0.028, 0.085, 0.15, 2104),
        ("pd_severe_05.wav",   3.0, 150, 0.024, 0.075, 0.13, 2105),
        ("pd_severe_06.wav",   3.0, 115, 0.030, 0.090, 0.16, 2106),
        ("pd_severe_07.wav",   2.8, 145, 0.022, 0.070, 0.12, 2107),
        ("pd_severe_08.wav",   3.0, 125, 0.026, 0.082, 0.14, 2108),
        ("pd_severe_09.wav",   3.0, 155, 0.020, 0.068, 0.11, 2109),
        ("pd_severe_10.wav",   3.0, 110, 0.032, 0.095, 0.17, 2110),
        # Mild PD (early stage — borderline features)
        ("pd_mild_01.wav",     3.0, 140, 0.009, 0.035, 0.05, 2201),
        ("pd_mild_02.wav",     3.0, 155, 0.010, 0.038, 0.06, 2202),
        ("pd_mild_03.wav",     2.8, 165, 0.008, 0.032, 0.04, 2203),
        ("pd_mild_04.wav",     3.2, 148, 0.011, 0.040, 0.06, 2204),
        ("pd_mild_05.wav",     3.0, 170, 0.009, 0.036, 0.05, 2205),
        ("pd_mild_06.wav",     3.0, 132, 0.010, 0.038, 0.05, 2206),
        ("pd_mild_07.wav",     2.8, 158, 0.008, 0.033, 0.04, 2207),
        ("pd_mild_08.wav",     3.0, 142, 0.011, 0.040, 0.06, 2208),
        ("pd_mild_09.wav",     3.0, 175, 0.009, 0.035, 0.05, 2209),
        ("pd_mild_10.wav",     3.0, 128, 0.010, 0.037, 0.05, 2210),
    ]

    print(f"\n  Generating {len(healthy_configs)} healthy WAV files...")
    for cfg in healthy_configs:
        fname, dur, f0, jit, shim, noise, seed = cfg
        y = generate_voice_like(dur, f0, jitter_amount=jit,
                                shimmer_amount=shim, noise_ratio=noise, seed=seed)
        sf.write(os.path.join(healthy_dir, fname), y, SR)

    print(f"  Generating {len(pd_configs)} PD WAV files...")
    for cfg in pd_configs:
        fname, dur, f0, jit, shim, noise, seed = cfg
        y = generate_voice_like(dur, f0, jitter_amount=jit,
                                shimmer_amount=shim, noise_ratio=noise, seed=seed)
        sf.write(os.path.join(pd_dir, fname), y, SR)

    return healthy_dir, pd_dir, len(healthy_configs), len(pd_configs)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Extract Praat features from a WAV file (same as app.py)
# ─────────────────────────────────────────────────────────────────────────────
def extract_voice_features(audio_path):
    """Extract 16 UCI Parkinson's dataset biomarkers from a voice audio file."""
    y_audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    if len(y_audio) < sr * 0.5:
        raise ValueError("Audio too short")

    tmp_wav = os.path.join(tempfile.gettempdir(), "pk_retrain_proc.wav")
    sf.write(tmp_wav, y_audio, sr)

    sound = parselmouth.Sound(tmp_wav)
    pitch_obj = call(sound, "To Pitch", 0.0, 75, 600)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

    Fo  = safe(call(pitch_obj, "Get mean",    0, 0, "Hertz"), 154.0)
    Fhi = safe(call(pitch_obj, "Get maximum", 0, 0, "Hertz", "Parabolic"), 197.0)
    Flo = safe(call(pitch_obj, "Get minimum", 0, 0, "Hertz", "Parabolic"), 116.0)

    jitter_pct  = safe(call(point_process, "Get jitter (local)",           0, 0, 0.0001, 0.02, 1.3), 0.006)
    jitter_abs  = safe(call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3), 4.4e-5)
    jitter_rap  = safe(call(point_process, "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3), 0.003)
    jitter_ppq5 = safe(call(point_process, "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3), 0.003)
    jitter_ddp  = safe(call(point_process, "Get jitter (ddp)",             0, 0, 0.0001, 0.02, 1.3), 0.010)

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


def extract_features_from_dir(wav_dir, label):
    """Extract features from all WAV files in a directory. Returns (features_list, labels_list)."""
    features_list = []
    labels = []
    filenames = []
    for fname in sorted(os.listdir(wav_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(wav_dir, fname)
        try:
            feats = extract_voice_features(path)
            features_list.append(feats)
            labels.append(label)
            filenames.append(fname)
        except Exception as e:
            print(f"    [SKIP] {fname}: {e}")
    return features_list, labels, filenames


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" PARKINSON'S MODEL RETRAINING WITH WAV AUGMENTATION")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load UCI dataset (only the 16 Praat-extractable features)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading UCI dataset...")
df = pd.read_csv(CSV_PATH)
if "name" in df.columns:
    df = df.drop(columns=["name"])

# Only use the 16 features that can be extracted from WAV files via Praat
# (the UCI CSV has 22 features total, but 6 are non-Praat: RPDE, DFA, spread1, spread2, D2, PPE)
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
]

# Verify all features exist in CSV
missing = [f for f in feature_names if f not in df.columns]
if missing:
    print(f"    ERROR: Missing features in CSV: {missing}")
    sys.exit(1)

print(f"    Using {len(feature_names)} Praat-extractable features: {feature_names}")

X_uci = df[feature_names].values
y_uci = df["status"].values

unique, counts = np.unique(y_uci, return_counts=True)
for cls, cnt in zip(unique, counts):
    label = "Healthy" if cls == 0 else "PD"
    print(f"    Class {cls} ({label}): {cnt} samples")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Generate synthetic WAV files
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Generating synthetic WAV test files...")
healthy_dir, pd_dir, n_healthy_wav, n_pd_wav = generate_test_wavs()
print(f"    Generated {n_healthy_wav} healthy + {n_pd_wav} PD WAV files")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Extract Praat features from WAV files
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Extracting Praat features from WAV files...")
print("    Processing healthy WAVs...")
healthy_feats, healthy_labels, healthy_names = extract_features_from_dir(healthy_dir, 0)
print(f"    -> {len(healthy_feats)} healthy samples extracted")

print("    Processing PD WAVs...")
pd_feats, pd_labels, pd_names = extract_features_from_dir(pd_dir, 1)
print(f"    -> {len(pd_feats)} PD samples extracted")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Show WAV feature distribution vs UCI
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Feature distribution comparison (UCI vs WAV)...")
uci_healthy = df[df["status"] == 0][feature_names]
uci_pd = df[df["status"] == 1][feature_names]

wav_healthy_df = pd.DataFrame(healthy_feats, columns=feature_names)
wav_pd_df = pd.DataFrame(pd_feats, columns=feature_names)

key_features = ["MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "NHR"]
for feat in key_features:
    idx = feature_names.index(feat)
    print(f"    {feat}:")
    print(f"      UCI Healthy:  mean={uci_healthy[feat].mean():.6f}  std={uci_healthy[feat].std():.6f}")
    print(f"      WAV Healthy:  mean={wav_healthy_df[feat].mean():.6f}  std={wav_healthy_df[feat].std():.6f}")
    print(f"      UCI PD:       mean={uci_pd[feat].mean():.6f}  std={uci_pd[feat].std():.6f}")
    print(f"      WAV PD:       mean={wav_pd_df[feat].mean():.6f}  std={wav_pd_df[feat].std():.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Build augmented dataset (UCI + WAV features)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Building augmented training dataset...")
X_wav = np.array(healthy_feats + pd_feats)
y_wav = np.array(healthy_labels + pd_labels)

X_combined = np.vstack([X_uci, X_wav])
y_combined = np.concatenate([y_uci, y_wav])

unique_c, counts_c = np.unique(y_combined, return_counts=True)
print(f"    Combined dataset: {len(X_combined)} samples")
for cls, cnt in zip(unique_c, counts_c):
    label = "Healthy" if cls == 0 else "PD"
    print(f"      Class {cls} ({label}): {cnt} samples")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Scale features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Scaling features with MinMaxScaler...")
scaler = MinMaxScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Also scale subsets for evaluation
X_uci_scaled = scaler.transform(X_uci)
X_wav_scaled = scaler.transform(X_wav)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Apply SMOTE
# ─────────────────────────────────────────────────────────────────────────────
if HAS_SMOTE:
    print("\n[7] Applying SMOTE to balance classes...")
    minority_count = min(counts_c)
    k = min(5, minority_count - 1)
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_resampled, y_resampled = sm.fit_resample(X_combined_scaled, y_combined)
    u2, c2 = np.unique(y_resampled, return_counts=True)
    print(f"    After SMOTE: {dict(zip(u2, c2))}")
else:
    print("\n[7] Skipping SMOTE (not installed)")
    X_resampled, y_resampled = X_combined_scaled, y_combined

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Train candidate models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Training candidate models...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

candidates = {
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
    ),
    "SVM_RBF": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
}

results = {}
trained_models = {}

for name, clf in candidates.items():
    print(f"\n  Training {name}...")
    bal_acc_scores = cross_val_score(clf, X_resampled, y_resampled,
                                     cv=cv, scoring="balanced_accuracy")
    roc_scores = cross_val_score(clf, X_resampled, y_resampled,
                                  cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(clf, X_combined_scaled, y_combined,
                                  cv=cv, scoring="accuracy")

    results[name] = {
        "balanced_accuracy": round(bal_acc_scores.mean(), 4),
        "roc_auc": round(roc_scores.mean(), 4),
        "accuracy": round(acc_scores.mean(), 4),
    }

    clf.fit(X_resampled, y_resampled)
    trained_models[name] = clf

    print(f"    Balanced Acc (CV) = {bal_acc_scores.mean():.4f}")
    print(f"    ROC-AUC     (CV) = {roc_scores.mean():.4f}")
    print(f"    Accuracy    (CV) = {acc_scores.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Select best model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Selecting best model by balanced accuracy...")
best_name = max(results, key=lambda k: results[k]["balanced_accuracy"])
best_clf = trained_models[best_name]
print(f"    Best: {best_name}")
print(f"    Balanced Acc = {results[best_name]['balanced_accuracy']:.4f}")
print(f"    ROC-AUC      = {results[best_name]['roc_auc']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Calibrate probabilities
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Calibrating model probabilities...")
calibrated_clf = CalibratedClassifierCV(best_clf, cv=5, method="sigmoid")
calibrated_clf.fit(X_resampled, y_resampled)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: Find optimal threshold
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Finding optimal probability threshold...")
proba_combined = calibrated_clf.predict_proba(X_combined_scaled)[:, 1]
thresholds = np.arange(0.30, 0.75, 0.01)
best_thresh = 0.5
best_bal = 0.0

for thresh in thresholds:
    preds = (proba_combined >= thresh).astype(int)
    bal = balanced_accuracy_score(y_combined, preds)
    if bal > best_bal:
        best_bal = bal
        best_thresh = thresh

print(f"    Optimal threshold = {best_thresh:.2f} (balanced accuracy = {best_bal:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12: Validate on UCI data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] Validating on UCI CSV data...")
proba_uci = calibrated_clf.predict_proba(X_uci_scaled)[:, 1]
preds_uci = (proba_uci >= best_thresh).astype(int)

uci_acc = accuracy_score(y_uci, preds_uci)
uci_bal = balanced_accuracy_score(y_uci, preds_uci)
cm_uci = confusion_matrix(y_uci, preds_uci)

print(f"    UCI Accuracy:          {uci_acc:.4f}")
print(f"    UCI Balanced Accuracy: {uci_bal:.4f}")
print(f"    Confusion matrix:")
print(f"      Healthy: TN={cm_uci[0,0]}  FP={cm_uci[0,1]}")
print(f"      PD:      FN={cm_uci[1,0]}  TP={cm_uci[1,1]}")
print(f"    Correct Healthy: {cm_uci[0,0]}/{cm_uci[0,0]+cm_uci[0,1]}  ({100*cm_uci[0,0]/(cm_uci[0,0]+cm_uci[0,1]):.1f}%)")
print(f"    Correct PD:      {cm_uci[1,1]}/{cm_uci[1,0]+cm_uci[1,1]}  ({100*cm_uci[1,1]/(cm_uci[1,0]+cm_uci[1,1]):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13: Validate on WAV files
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] Validating on WAV files...")

# Healthy WAVs
proba_wav_h = calibrated_clf.predict_proba(
    scaler.transform(np.array(healthy_feats))
)[:, 1]
preds_wav_h = (proba_wav_h >= best_thresh).astype(int)
wav_h_correct = (preds_wav_h == 0).sum()

print(f"\n  HEALTHY WAVs ({len(healthy_feats)} files):")
print(f"    Correctly classified: {wav_h_correct}/{len(healthy_feats)} ({100*wav_h_correct/len(healthy_feats):.1f}%)")
print(f"    Mean probability:     {proba_wav_h.mean():.4f}")
print(f"    Range:                {proba_wav_h.min():.4f} - {proba_wav_h.max():.4f}")
for i, name in enumerate(healthy_names):
    status = "✓ HEALTHY" if preds_wav_h[i] == 0 else "✗ FALSE PD"
    print(f"      {name}: prob={proba_wav_h[i]:.4f} -> {status}")

# PD WAVs
proba_wav_pd = calibrated_clf.predict_proba(
    scaler.transform(np.array(pd_feats))
)[:, 1]
preds_wav_pd = (proba_wav_pd >= best_thresh).astype(int)
wav_pd_correct = (preds_wav_pd == 1).sum()

print(f"\n  PD WAVs ({len(pd_feats)} files):")
print(f"    Correctly classified: {wav_pd_correct}/{len(pd_feats)} ({100*wav_pd_correct/len(pd_feats):.1f}%)")
print(f"    Mean probability:     {proba_wav_pd.mean():.4f}")
print(f"    Range:                {proba_wav_pd.min():.4f} - {proba_wav_pd.max():.4f}")
for i, name in enumerate(pd_names):
    status = "✓ PD DETECTED" if preds_wav_pd[i] == 1 else "✗ FALSE HEALTHY"
    print(f"      {name}: prob={proba_wav_pd[i]:.4f} -> {status}")

# Also test existing sample WAVs
print("\n  Existing sample WAVs in data/:")
for sample_name in ["sample_01.wav", "sample_02.wav", "parkinsons_5sec.wav"]:
    sample_path = os.path.join(DATA_DIR, sample_name)
    if os.path.exists(sample_path):
        try:
            feats = extract_voice_features(sample_path)
            feats_scaled = scaler.transform(np.array([feats]))
            prob = float(calibrated_clf.predict_proba(feats_scaled)[0][1])
            label = "PD" if prob >= best_thresh else "Healthy"
            print(f"    {sample_name}: prob={prob:.4f} -> {label}")
        except Exception as e:
            print(f"    {sample_name}: ERROR - {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14: Save model artifacts
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] Saving model artifacts...")

# Backup old models
for fname in ["parkinson_model.pkl", "scaler.pkl", "feature_names.pkl"]:
    src = os.path.join(MODEL_DIR, fname)
    dst = os.path.join(MODEL_DIR, fname + ".bak")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"    Backed up {fname}")

# Save new model
with open(os.path.join(MODEL_DIR, "parkinson_model.pkl"), "wb") as f:
    pickle.dump(calibrated_clf, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

# Save feature stats
healthy_all = pd.concat([
    df[df["status"] == 0][feature_names],
    wav_healthy_df
])
pd_all = pd.concat([
    df[df["status"] == 1][feature_names],
    wav_pd_df
])

feature_stats = {}
for fn in feature_names:
    feature_stats[fn] = {
        "healthy_mean": float(healthy_all[fn].mean()),
        "healthy_std":  float(healthy_all[fn].std()),
        "pd_mean":      float(pd_all[fn].mean()),
        "pd_std":       float(pd_all[fn].std()),
        "overall_min":  float(min(df[fn].min(), wav_healthy_df[fn].min(), wav_pd_df[fn].min())),
        "overall_max":  float(max(df[fn].max(), wav_healthy_df[fn].max(), wav_pd_df[fn].max())),
    }

with open(os.path.join(MODEL_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

# Save comprehensive model results
total_wav = len(healthy_feats) + len(pd_feats)
total_wav_correct = wav_h_correct + wav_pd_correct

model_results_meta = {
    "best_model_name": best_name,
    "balanced_accuracy_cv": results[best_name]["balanced_accuracy"],
    "roc_auc_cv": results[best_name]["roc_auc"],
    "accuracy_cv": results[best_name]["accuracy"],
    "optimal_threshold": round(float(best_thresh), 4),
    "uci_accuracy": round(uci_acc, 4),
    "uci_balanced_accuracy": round(uci_bal, 4),
    "wav_healthy_accuracy": round(wav_h_correct / len(healthy_feats), 4),
    "wav_pd_accuracy": round(wav_pd_correct / len(pd_feats), 4),
    "wav_total_accuracy": round(total_wav_correct / total_wav, 4),
    "n_uci_samples": int(len(df)),
    "n_wav_augmentation": int(len(X_wav)),
    "n_total_training": int(len(X_combined)),
    "smote_applied": HAS_SMOTE,
    "all_model_results": results,
}

with open(os.path.join(MODEL_DIR, "model_results_augmented.json"), "w") as f:
    json.dump(model_results_meta, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(" RETRAINING COMPLETE — RESULTS SUMMARY")
print("=" * 70)
print(f"  Best model:              {best_name}")
print(f"  Optimal threshold:       {best_thresh:.2f}")
print(f"  UCI Accuracy:            {uci_acc:.1%}")
print(f"  UCI Balanced Accuracy:   {uci_bal:.1%}")
print(f"  WAV Healthy Accuracy:    {wav_h_correct}/{len(healthy_feats)} ({100*wav_h_correct/len(healthy_feats):.1f}%)")
print(f"  WAV PD Accuracy:         {wav_pd_correct}/{len(pd_feats)} ({100*wav_pd_correct/len(pd_feats):.1f}%)")
print(f"  WAV Total Accuracy:      {total_wav_correct}/{total_wav} ({100*total_wav_correct/total_wav:.1f}%)")
print("=" * 70)
print("\nSaved artifacts:")
print("  models/parkinson_model.pkl  (calibrated, WAV-augmented)")
print("  models/scaler.pkl")
print("  models/feature_names.pkl")
print("  models/feature_stats.json")
print("  models/model_results_augmented.json")
print(f"\nTest WAV files saved to: {TEST_WAV_DIR}")
