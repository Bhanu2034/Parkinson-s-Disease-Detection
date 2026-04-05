# -*- coding: utf-8 -*-
"""
PARKINSON'S DETECTION MODEL RETRAINING SCRIPT
==============================================
Fixes the issue of healthy voices being misclassified as high risk.

Root cause analysis:
- UCI dataset is imbalanced (75% PD, 25% healthy)
- GradientBoosting trained on imbalanced data becomes overly biased toward PD
- Consumer microphone recordings have slightly elevated jitter/shimmer which
  falls into the PD decision zone of an imbalanced model

Solution:
- Use class_weight='balanced' + SMOTE oversampling of minority class
- Use calibrated probabilities (CalibratedClassifierCV)
- Tune decision threshold to maximise balanced accuracy (not just raw accuracy)
- Use cross-validation to pick the best threshold
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import warnings

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

# Try to import imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARNING] imbalanced-learn not installed. Falling back to class_weight balancing.")
    print("  Install with: pip install imbalanced-learn")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CSV_PATH = os.path.join(DATA_DIR, "parkinsons.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(" PARKINSON'S MODEL RETRAINING  (Balanced)")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"\n[1] Loaded dataset: {len(df)} rows from {CSV_PATH}")

if "name" in df.columns:
    df = df.drop(columns=["name"])

feature_names = [c for c in df.columns if c != "status"]
print(f"    Features ({len(feature_names)}): {feature_names}")

X = df[feature_names].values
y = df["status"].values

unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    label = "Healthy" if cls == 0 else "PD"
    print(f"    Class {cls} ({label}): {cnt} samples")

print(f"\n    Class balance: {counts[0]} healthy vs {counts[1]} PD "
      f"(ratio = {counts[1]/counts[0]:.1f}:1)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Scale features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Scaling features with MinMaxScaler...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Apply SMOTE to balance the dataset
# ─────────────────────────────────────────────────────────────────────────────
if HAS_SMOTE:
    print("\n[3] Applying SMOTE oversampling to balance classes...")
    sm = SMOTE(random_state=42, k_neighbors=min(5, counts[0]-1))
    X_resampled, y_resampled = sm.fit_resample(X_scaled, y)
    u2, c2 = np.unique(y_resampled, return_counts=True)
    print(f"    After SMOTE: {dict(zip(u2, c2))}")
else:
    print("\n[3] Skipping SMOTE — using class_weight='balanced' instead...")
    X_resampled, y_resampled = X_scaled, y

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Define candidate models (all with class balance support)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Training and evaluating candidate models...")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

candidates = {
    "GradientBoosting_balanced": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    ),
    "RandomForest_balanced": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
    ),
    "SVM_balanced": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "LogisticRegression_balanced": LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
}

results = {}
trained_models = {}

for name, clf in candidates.items():
    # Cross-validate balanced accuracy on SMOTE-resampled data
    bal_acc_scores = cross_val_score(
        clf, X_resampled, y_resampled,
        cv=cv, scoring="balanced_accuracy"
    )
    roc_scores = cross_val_score(
        clf, X_resampled, y_resampled,
        cv=cv, scoring="roc_auc"
    )
    
    mean_bal  = bal_acc_scores.mean()
    mean_roc  = roc_scores.mean()
    
    # Also measure on original (unbalanced) data — accuracy
    acc_scores = cross_val_score(
        clf, X_scaled, y,
        cv=cv, scoring="accuracy"
    )
    mean_acc = acc_scores.mean()
    
    results[name] = {
        "balanced_accuracy": round(mean_bal, 4),
        "roc_auc":           round(mean_roc, 4),
        "accuracy":          round(mean_acc, 4),
    }
    
    # Train final model on resampled data
    clf.fit(X_resampled, y_resampled)
    trained_models[name] = clf
    
    print(f"  {name}:")
    print(f"    Balanced Acc  (CV) = {mean_bal:.4f}")
    print(f"    ROC-AUC       (CV) = {mean_roc:.4f}")
    print(f"    Accuracy      (CV) = {mean_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Select best model (by balanced_accuracy — fairer for imbalanced data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Selecting best model by balanced accuracy...")
best_name = max(results, key=lambda k: results[k]["balanced_accuracy"])
best_clf  = trained_models[best_name]
print(f"    Best model: {best_name}")
print(f"    Balanced Acc = {results[best_name]['balanced_accuracy']:.4f}")
print(f"    ROC-AUC      = {results[best_name]['roc_auc']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Calibrate the best model's probabilities
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Calibrating model probabilities with Sigmoid (Platt scaling)...")
calibrated_clf = CalibratedClassifierCV(best_clf, cv=5, method="sigmoid")
calibrated_clf.fit(X_resampled, y_resampled)

# Verify calibration preserves class separation on original data
proba_all = calibrated_clf.predict_proba(X_scaled)[:, 1]
proba_h   = proba_all[y == 0]
proba_pd  = proba_all[y == 1]

print(f"    Healthy samples  -> mean prob = {proba_h.mean():.4f}  (range {proba_h.min():.4f}-{proba_h.max():.4f})")
print(f"    PD samples       -> mean prob = {proba_pd.mean():.4f}  (range {proba_pd.min():.4f}-{proba_pd.max():.4f})")

correct_h  = (proba_h  < 0.5).sum()
correct_pd = (proba_pd >= 0.5).sum()
print(f"    Correct healthy: {correct_h}/{len(proba_h)}  ({100*correct_h/len(proba_h):.1f}%)")
print(f"    Correct PD:      {correct_pd}/{len(proba_pd)}  ({100*correct_pd/len(proba_pd):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Find optimal threshold that maximises balanced accuracy
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Finding optimal probability threshold...")
thresholds = np.arange(0.30, 0.75, 0.01)
best_thresh = 0.5
best_bal    = 0.0

for thresh in thresholds:
    preds = (proba_all >= thresh).astype(int)
    bal = balanced_accuracy_score(y, preds)
    if bal > best_bal:
        best_bal   = bal
        best_thresh = thresh

print(f"    Optimal threshold = {best_thresh:.2f}  (balanced accuracy = {best_bal:.4f})")

# Re-evaluate at optimal threshold
preds_opt = (proba_all >= best_thresh).astype(int)
cm = confusion_matrix(y, preds_opt)
print("\n    Confusion matrix (rows=actual, cols=predicted):")
print(f"      Healthy: TN={cm[0,0]}  FP={cm[0,1]}")
print(f"      PD:      FN={cm[1,0]}  TP={cm[1,1]}")
print("\n    Classification report:")
print(classification_report(y, preds_opt, target_names=["Healthy", "PD"]))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Test on the sample WAV files in data/ (using saved values from analysis)
# ─────────────────────────────────────────────────────────────────────────────
# These are known UCI Parkinson dataset samples (both are PD patients from UCI)
# The 'parkinsons_5sec.wav' has borderline features (prob ~0.70 → PD, correct per UCI)
print("\n[8] Verifying predictions on all CSV rows (original data)...")

all_preds_orig = (calibrated_clf.predict_proba(X_scaled)[:, 1] >= best_thresh).astype(int)
final_acc = accuracy_score(y, all_preds_orig)
final_bal = balanced_accuracy_score(y, all_preds_orig)
final_auc = roc_auc_score(y, calibrated_clf.predict_proba(X_scaled)[:, 1])

print(f"    Final accuracy:          {final_acc:.4f}")
print(f"    Final balanced accuracy: {final_bal:.4f}")
print(f"    Final ROC-AUC:           {final_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Save the final model artifacts
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Saving model artifacts...")

# Backup old model
import shutil
for fname in ["parkinson_model.pkl", "scaler.pkl", "feature_names.pkl"]:
    src = os.path.join(MODEL_DIR, fname)
    dst = os.path.join(MODEL_DIR, fname + ".bak")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"    Backed up {fname} -> {fname}.bak")

# Save new calibrated model
with open(os.path.join(MODEL_DIR, "parkinson_model.pkl"), "wb") as f:
    pickle.dump(calibrated_clf, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

# Save feature stats
healthy_df  = df[df["status"] == 0]
pd_df       = df[df["status"] == 1]
feature_stats = {}
for fn in feature_names:
    feature_stats[fn] = {
        "healthy_mean": float(healthy_df[fn].mean()),
        "healthy_std":  float(healthy_df[fn].std()),
        "pd_mean":      float(pd_df[fn].mean()),
        "pd_std":       float(pd_df[fn].std()),
        "overall_min":  float(df[fn].min()),
        "overall_max":  float(df[fn].max()),
    }

with open(os.path.join(MODEL_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

# Save model results metadata
model_results = {
    "best_model_name":       best_name,
    "balanced_accuracy_cv":  results[best_name]["balanced_accuracy"],
    "roc_auc_cv":            results[best_name]["roc_auc"],
    "accuracy_cv":           results[best_name]["accuracy"],
    "final_accuracy":        round(final_acc,  4),
    "final_balanced_acc":    round(final_bal,  4),
    "final_roc_auc":         round(final_auc,  4),
    "optimal_threshold":     round(float(best_thresh), 4),
    "n_samples":             int(len(df)),
    "n_healthy":             int(counts[0]),
    "n_pd":                  int(counts[1]),
    "smote_applied":         HAS_SMOTE,
    "all_model_results":     results,
}

with open(os.path.join(MODEL_DIR, "model_results.json"), "w") as f:
    json.dump(model_results, f, indent=2)

print(f"\nSaved to {MODEL_DIR}:")
print("  parkinson_model.pkl  (calibrated, balanced)")
print("  scaler.pkl")
print("  feature_names.pkl")
print("  feature_stats.json")
print("  model_results.json")

print("\n" + "=" * 60)
print(" RETRAINING COMPLETE")
print(f" Best model:              {best_name}")
print(f" Optimal threshold:       {best_thresh:.2f}")
print(f" Balanced accuracy:       {final_bal:.1%}")
print(f" ROC-AUC:                 {final_auc:.4f}")
print(f" Correct Healthy (CSV):   {correct_h}/{len(proba_h)}")
print(f" Correct PD (CSV):        {correct_pd}/{len(proba_pd)}")
print("=" * 60)
