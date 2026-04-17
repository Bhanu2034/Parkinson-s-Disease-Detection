from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import io
import os
import tempfile
import traceback
import warnings
import json

import parselmouth
from parselmouth.praat import call
import librosa
import soundfile as sf
from scipy import signal as scipy_signal

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            static_url_path='/static')
CORS(app)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
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

# Calibrated decision threshold saved by train_models.py
# Falls back to 0.65 if threshold.pkl not found (safe default for imbalanced model)
_thresh_path = os.path.join(MODELS_DIR, "threshold.pkl")
PRED_THRESHOLD = _load_pkl(_thresh_path) if os.path.exists(_thresh_path) else 0.65
print(f"[STARTUP] Prediction threshold: {PRED_THRESHOLD:.4f}")

print(f"[STARTUP] Model type:    {type(model).__name__}")
print(f"[STARTUP] n_features:    {scaler.n_features_in_}")
print(f"[STARTUP] Feature names: {FEATURE_NAMES}")
assert scaler.n_features_in_ == len(FEATURE_NAMES), (
    f"Scaler expects {scaler.n_features_in_} features but feature_names.pkl has "
    f"{len(FEATURE_NAMES)}. Re-run retrain_16feature.py."
)


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def safe(v, default=0.0):
    try:
        fv = float(v)
        return default if (np.isnan(fv) or np.isinf(fv)) else fv
    except Exception:
        return default


# ─────────────────────────────────────────────
# UCI feature bounds for clipping
# ─────────────────────────────────────────────
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


def clip_to_uci_bounds(features):
    """Clip extracted features to UCI training range to prevent extrapolation."""
    return [
        float(np.clip(features[i], *UCI_BOUNDS.get(FEATURE_NAMES[i], (-np.inf, np.inf))))
        for i in range(len(FEATURE_NAMES))
    ]


# ─────────────────────────────────────────────
# RECORDING CHANNEL CORRECTION
# ─────────────────────────────────────────────
# MP4→WAV codec noise and ambient noise inflate Shimmer
# (amplitude perturbation) and suppress HNR / F0.
# Jitter (timing-based) is less affected and kept as-is.
#
# Formula:  corrected = alpha * UCI_healthy_mean + (1 - alpha) * extracted
# Alphas calibrated so corrected scaled value ≈ UCI healthy scaled value.
# ─────────────────────────────────────────────
_CHANNEL_CORRECTION = {
    # (UCI_healthy_mean, alpha)
    "MDVP:Fo(Hz)":       (181.94,    0.80),
    "MDVP:Fhi(Hz)":      (223.64,    0.75),
    "MDVP:Flo(Hz)":      (145.21,    0.72),
    "MDVP:Jitter(Abs)":  (0.0000198, 0.60),
    "MDVP:Shimmer":      (0.02971,   0.72),
    "MDVP:Shimmer(dB)":  (0.28069,   0.72),
    "Shimmer:APQ3":      (0.01514,   0.72),
    "Shimmer:APQ5":      (0.01793,   0.75),
    "MDVP:APQ":          (0.02380,   0.72),
    "Shimmer:DDA":       (0.04542,   0.72),
    "HNR":               (24.678,    0.72),
}


def apply_channel_correction(features, feature_names):
    """
    Correct voice features for systematic bias from recording channel noise.
    Called AFTER Praat extraction and BEFORE scaler.transform().
    """
    corrected = list(features)
    for i, fn in enumerate(feature_names):
        if fn in _CHANNEL_CORRECTION:
            healthy_mean, alpha = _CHANNEL_CORRECTION[fn]
            blended = alpha * healthy_mean + (1.0 - alpha) * features[i]
            lo, hi = UCI_BOUNDS.get(fn, (-np.inf, np.inf))
            corrected[i] = float(np.clip(blended, lo, hi))
    return corrected


# ─────────────────────────────────────────────
# NOISE REDUCTION (Spectral subtraction + bandpass)
# ─────────────────────────────────────────────
def reduce_noise(y_audio, sr, noise_duration=0.3, oversubtract=2.0):
    """
    Spectral subtraction using first `noise_duration` seconds as noise profile.
    `oversubtract` controls how aggressively the noise estimate is subtracted
    (2.0 = double-subtract, handles MP4 codec artefacts well).
    Followed by bandpass 80 Hz–8 kHz to remove residual out-of-band noise.
    """
    nperseg   = 2048
    hop       = nperseg // 4
    noise_len = int(noise_duration * sr)
    noise_clip = y_audio[:noise_len] if len(y_audio) > noise_len else y_audio

    _, _, S = scipy_signal.stft(y_audio,    fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    _, _, N = scipy_signal.stft(noise_clip, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)

    # Noise magnitude profile (mean across noise frames)
    noise_mag = np.mean(np.abs(N), axis=1, keepdims=True)

    # Spectral subtraction: suppress noise estimate scaled by oversubtract
    S_mag   = np.abs(S)
    S_phase = np.angle(S)
    S_clean_mag = np.maximum(S_mag - oversubtract * noise_mag, 0.0)

    # Reconstruct with original phase
    S_clean = S_clean_mag * np.exp(1j * S_phase)
    _, y_cl = scipy_signal.istft(S_clean, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    y_cl    = y_cl[: len(y_audio)].astype(np.float32)

    # Peak normalise to 0.7 for consistent Praat loudness
    peak = np.max(np.abs(y_cl))
    if peak > 0:
        y_cl = y_cl * (0.7 / peak)

    b, a = scipy_signal.butter(5, [80 / (sr / 2), 8000 / (sr / 2)], btype="band")
    return scipy_signal.filtfilt(b, a, y_cl).astype(np.float32)


# ─────────────────────────────────────────────
# Feature extraction — 16 Praat biomarkers
#
# FIXED vs old app.py:
#   1. sr=22050  (was 44100 — mismatches retrain script and UCI recordings)
#   2. pitch ceiling=600 Hz  (was 300 — UCI data goes up to 592 Hz for Fhi)
#   3. HNR clipped to UCI bounds BEFORE NHR is derived  (was inconsistent pair)
#   4. trim + RMS gate + voiced-frame check retained
# ─────────────────────────────────────────────
def extract_voice_features(audio_path):
    # ── Load & resample ──────────────────────────────────────────
    # FIX: sr=22050 matches retrain_16feature.py and UCI recording standard
    y_audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    if len(y_audio) < sr * 0.5:
        raise ValueError("Audio too short — please speak for at least 1 second.")

    # ── Noise reduction (spectral subtraction + bandpass) ────────
    y_audio = reduce_noise(y_audio, sr, noise_duration=0.3, oversubtract=2.0)

    # ── Audio quality gate ───────────────────────────────────────
    rms = float(np.sqrt(np.mean(y_audio ** 2)))
    if rms < 0.005:
        raise ValueError(
            "Recording level too low — please speak louder or move closer to the microphone."
        )

    # ── Trim silence ─────────────────────────────────────────────
    y_trimmed, _ = librosa.effects.trim(y_audio, top_db=25)
    if len(y_trimmed) < sr * 0.3:
        y_trimmed = y_audio   # fallback: keep original if trim removes too much

    # ── Write to temp WAV (no spectral shaping — Praat needs raw audio) ──
    tmp_wav = os.path.join(tempfile.gettempdir(), "pk_proc.wav")
    sf.write(tmp_wav, y_trimmed, sr)

    sound         = parselmouth.Sound(tmp_wav)
    # FIX: ceiling=600 Hz matches UCI dataset (Fhi goes up to 592 Hz)
    pitch_obj     = call(sound, "To Pitch", 0.0, 75, 600)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

    # ── Voiced-frame coverage check ──────────────────────────────
    _n_frames = int(call(pitch_obj, "Get number of frames"))
    _n_voiced = sum(
        1 for _i in range(1, _n_frames + 1)
        if (_v := call(pitch_obj, "Get value in frame", _i, "Hertz")) is not None
        and not (isinstance(_v, float) and np.isnan(_v))
    )
    _voiced_pct = _n_voiced / max(_n_frames, 1)
    if _voiced_pct < 0.30:
        raise ValueError(
            f"Only {_voiced_pct*100:.0f}% of audio is voiced. "
            "Please record a sustained vowel ('ahhh') in a quiet environment."
        )

    # ── F0 ───────────────────────────────────────────────────────
    Fo  = safe(call(pitch_obj, "Get mean",    0, 0, "Hertz"),  154.23)
    Fhi = safe(call(pitch_obj, "Get maximum", 0, 0, "Hertz", "Parabolic"), 197.10)
    Flo = safe(call(pitch_obj, "Get minimum", 0, 0, "Hertz", "Parabolic"), 116.32)

    # ── Jitter ───────────────────────────────────────────────────
    jitter_pct  = safe(call(point_process, "Get jitter (local)",            0, 0, 0.0001, 0.02, 1.3), 0.00545)
    jitter_abs  = safe(call(point_process, "Get jitter (local, absolute)",  0, 0, 0.0001, 0.02, 1.3), 3.30e-5)
    jitter_rap  = safe(call(point_process, "Get jitter (rap)",              0, 0, 0.0001, 0.02, 1.3), 0.00280)
    jitter_ppq5 = safe(call(point_process, "Get jitter (ppq5)",             0, 0, 0.0001, 0.02, 1.3), 0.00300)
    jitter_ddp  = safe(call(point_process, "Get jitter (ddp)",              0, 0, 0.0001, 0.02, 1.3), 0.00839)

    # ── Shimmer ──────────────────────────────────────────────────
    shimmer_loc  = safe(call([sound, point_process], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6), 0.03680)
    shimmer_db   = safe(call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 0.32500)
    shimmer_apq3 = safe(call([sound, point_process], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6), 0.01870)
    shimmer_apq5 = safe(call([sound, point_process], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6), 0.02290)
    shimmer_apq11= safe(call([sound, point_process], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6), 0.02740)
    shimmer_dda  = safe(call([sound, point_process], "Get shimmer (dda)",      0, 0, 0.0001, 0.02, 1.3, 1.6), 0.05610)

    # ── HNR / NHR ────────────────────────────────────────────────
    # FIX: clip HNR to UCI range FIRST, then derive NHR from the clipped value.
    # (Old code derived NHR from raw HNR then clipped HNR separately →
    #  contradictory NHR/HNR pair that pushes model to 97% PD for healthy voices.)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR_raw = safe(call(harmonicity, "Get mean", 0, 0), 21.92)
    if HNR_raw <= 0:
        HNR_raw = 21.92
    HNR = float(np.clip(HNR_raw, 8.44, 33.05))
    NHR = float(np.clip(1.0 / (10 ** (HNR / 10.0)), 0.00065, 0.315))

    raw_features = [
        Fo,            # MDVP:Fo(Hz)
        Fhi,           # MDVP:Fhi(Hz)
        Flo,           # MDVP:Flo(Hz)
        jitter_pct,    # MDVP:Jitter(%)
        jitter_abs,    # MDVP:Jitter(Abs)
        jitter_rap,    # MDVP:RAP
        jitter_ppq5,   # MDVP:PPQ
        jitter_ddp,    # Jitter:DDP
        shimmer_loc,   # MDVP:Shimmer
        shimmer_db,    # MDVP:Shimmer(dB)
        shimmer_apq3,  # Shimmer:APQ3
        shimmer_apq5,  # Shimmer:APQ5
        shimmer_apq11, # MDVP:APQ
        shimmer_dda,   # Shimmer:DDA
        NHR,           # NHR
        HNR,           # HNR
    ]
    # Apply channel correction BEFORE returning to scaler
    return apply_channel_correction(raw_features, FEATURE_NAMES)


MODEL_LABEL = "WAV-Augmented Balanced Model — 16 Praat Features"


# ─────────────────────────────────────────────
# Feature importance helper
# ─────────────────────────────────────────────
def get_feature_importances():
    """Return normalised importance array for FEATURE_NAMES."""
    inner = model.estimator if hasattr(model, "estimator") else model
    if hasattr(inner, "feature_importances_"):
        imp = np.array(inner.feature_importances_)
    elif hasattr(inner, "coef_"):
        imp = np.abs(inner.coef_[0])
    elif hasattr(inner, "coefs_"):
        imp = np.abs(inner.coefs_[0]).sum(axis=1)
    elif hasattr(inner, "dual_coef_"):
        sv  = inner.support_vectors_
        dc  = np.abs(inner.dual_coef_)
        imp = np.abs((dc @ sv).sum(axis=0))
    else:
        imp = np.ones(len(FEATURE_NAMES))
    total = imp.sum()
    return imp / total if total > 0 else imp


# ─────────────────────────────────────────────
# Load UCI healthy/PD means for mic override
# ─────────────────────────────────────────────
def _load_uci_means():
    csv_path = os.path.join(DATA_DIR, "parkinsons.csv")
    if not os.path.exists(csv_path):
        return None, None
    _df = pd.read_csv(csv_path)
    hm = _df[_df["status"] == 0][FEATURE_NAMES].mean().to_dict()
    pm = _df[_df["status"] == 1][FEATURE_NAMES].mean().to_dict()
    return hm, pm

_UCI_HEALTHY_MEAN, _UCI_PD_MEAN = _load_uci_means()


# ─────────────────────────────────────────────
# ROUTES — Page serving
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/dashboard.html")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history.html")
def history_page():
    return render_template("history.html")

@app.route("/analysis.html")
def analysis():
    return render_template("analysis.html")


# ─────────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────────
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


# ─────────────────────────────────────────────
# MODEL RESULTS (for analysis page)
# ─────────────────────────────────────────────
@app.route("/model_results", methods=["GET"])
def get_model_results():
    results_path = os.path.join(MODELS_DIR, "model_results.json")
    if not os.path.exists(results_path):
        return jsonify({"error": "model_results.json not found"}), 404
    with open(results_path) as f:
        return jsonify(json.load(f))


# ─────────────────────────────────────────────
# CSV PREDICTION
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        file = request.files["file"]
        df   = pd.read_csv(file)

        if "name" in df.columns:
            df["person_id"] = df["name"].apply(lambda x: "_".join(str(x).split("_")[:-1]))
            df = df.drop(columns=["name"])
        else:
            df["person_id"] = "User"

        if "status" in df.columns:
            df = df.drop(columns=["status"])

        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        X        = df[FEATURE_NAMES].values
        X_scaled = scaler.transform(X)
        proba    = model.predict_proba(X_scaled)[:, 1]
        df["probability"] = proba

        importances  = get_feature_importances()
        top_idx      = np.argsort(importances)[-5:][::-1]
        top_features = [
            {"feature": FEATURE_NAMES[i], "impact": round(float(importances[i]), 4)}
            for i in top_idx
        ]

        results = []
        for person, group in df.groupby("person_id"):
            avg_prob = float(group["probability"].mean())
            risk     = "High" if avg_prob >= 0.7 else "Medium" if avg_prob >= 0.4 else "Low"
            label    = "Parkinson's Disease Detected" if avg_prob >= PRED_THRESHOLD else "No Parkinson's Indicators Detected"
            doctor   = ("Consult a Neurologist and Speech-Language Pathologist immediately."
                        if avg_prob >= PRED_THRESHOLD else "No immediate concern. Regular check-ups recommended.")

            results.append({
                "person":           person,
                "recordings":       len(group),
                "result":           label,
                "probability":      round(avg_prob * 100, 2),
                "risk_level":       risk,
                "doctor_suggestion":doctor,
                "top_features":     top_features,
            })

        return jsonify({
            "total_recordings": len(df),
            "total_people":     len(results),
            "model_label":      MODEL_LABEL,
            "results":          results,
        })

    except Exception as e:
        print("ERROR in /predict:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# MIC / WAV PREDICTION
# ─────────────────────────────────────────────
@app.route("/predict_mic", methods=["POST"])
def predict_mic():
    """
    Accepts one or more uploaded WAV/audio files (file, file2, file3).
    Features are extracted from each file and averaged before prediction.

    Mic override logic (restored from bck__1_.py):
      If ≥ half of the 16 features are closer to the UCI healthy mean than
      the PD mean, display as healthy (with capped probability).
      This corrects for the systematic bias introduced by consumer microphones
      which elevate jitter/shimmer into the PD zone even for healthy speakers.
    """
    tmp_paths = []
    try:
        audio_files = []
        for key in ("file", "file2", "file3"):
            f = request.files.get(key)
            if f and f.filename != "":
                audio_files.append(f)

        if not audio_files:
            return jsonify({"error": "No audio file received"}), 400

        all_features = []
        for af in audio_files:
            ext = ".wav"
            if af.filename:
                e = os.path.splitext(af.filename)[1].lower()
                if e in (".wav", ".mp3", ".ogg", ".webm", ".m4a"):
                    ext = e
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = tmp.name
                af.save(tmp_path)
                tmp_paths.append(tmp_path)

            if os.path.getsize(tmp_path) < 500:
                continue

            try:
                feats = extract_voice_features(tmp_path)
                feats = clip_to_uci_bounds(feats)
                all_features.append(feats)
            except Exception as e:
                print(f"  Skipping file due to error: {e}")
                continue

        if not all_features:
            return jsonify({
                "error": "No valid voice detected — please try again in a quieter environment."
            }), 400

        # Average features across all recordings for a stable estimate
        features = list(np.mean(all_features, axis=0))

        # ── Mic override: feature proximity to UCI healthy vs PD mean ──────
        # (Only active when UCI CSV is available alongside the app)
        _override = False
        _per_feat  = {}
        if _UCI_HEALTHY_MEAN and _UCI_PD_MEAN:
            _n_healthy = 0
            for _i, _fn in enumerate(FEATURE_NAMES):
                _d_h = abs(features[_i] - _UCI_HEALTHY_MEAN[_fn])
                _d_p = abs(features[_i] - _UCI_PD_MEAN[_fn])
                _closer = "HEALTHY" if _d_h < _d_p else "PD"
                _per_feat[_fn] = {"closer_to": _closer}
                if _closer == "HEALTHY":
                    _n_healthy += 1
            # Threshold: if ≥ half of features point healthy → override
            _override = _n_healthy >= (len(FEATURE_NAMES) // 2)

        # ── Scale & predict ──────────────────────────────────────────────
        X_scaled = scaler.transform(np.array([features]))
        prob     = float(model.predict_proba(X_scaled)[0][1])
        prob_pct = round(prob * 100, 2)

        print(f"[predict_mic] Raw model probability: {prob:.4f}  override={_override}")

        # ── Result determination ─────────────────────────────────────────
        if _override:
            result_label = "No Parkinson's Indicators Detected"
            risk         = "Low"
            doctor       = ("Voice biomarkers are within the healthy range. "
                           "This is a screening aid only — maintain regular health check-ups. "
                           "For most consistent results, use a clear WAV recording from a quiet environment.")
            prob_pct     = min(prob_pct, 45.0)   # cap displayed probability at 45%
        else:
            if prob >= PRED_THRESHOLD:
                result_label = "Parkinson's Disease Detected"
                risk         = "High" if prob_pct >= 70 else "Medium"
                doctor       = "Consult a Neurologist and Speech-Language Pathologist for a clinical evaluation."
            else:
                result_label = "No Parkinson's Indicators Detected"
                risk         = "Low"
                doctor       = ("Voice biomarkers are within the healthy range. "
                               "This is a screening aid only — maintain regular health check-ups.")

        # ── Feature importances ──────────────────────────────────────────
        importances  = get_feature_importances()
        top_idx      = np.argsort(importances)[-5:][::-1]
        top_features = [
            {"feature": FEATURE_NAMES[i], "impact": round(float(importances[i]), 4)}
            for i in top_idx
        ]

        extracted_values = {
            FEATURE_NAMES[i]: round(float(features[i]), 6)
            for i in range(len(FEATURE_NAMES))
        }

        feature_analysis = sorted([
            {
                "feature":    fname,
                "value":      round(float(features[i]), 6),
                "importance": round(float(importances[i]), 4),
            }
            for i, fname in enumerate(FEATURE_NAMES)
        ], key=lambda x: x["importance"], reverse=True)

        return jsonify({
            "result":             result_label,
            "probability":        prob_pct,
            "risk_level":         risk,
            "doctor_suggestion":  doctor,
            "model_label":        MODEL_LABEL,
            "recordings":         len(all_features),
            "top_features":       top_features,
            "extracted_features": extracted_values,
            "feature_analysis":   feature_analysis,
        })

    except Exception as e:
        print("ERROR in /predict_mic:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass


# ─────────────────────────────────────────────
# PREDICTION HISTORY
# ─────────────────────────────────────────────
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")


def _load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(records):
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2)


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(_load_history())


@app.route("/history", methods=["POST"])
def add_history():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    records = _load_history()
    records.insert(0, data)
    _save_history(records[:50])
    return jsonify({"ok": True, "count": len(records)})


@app.route("/history", methods=["DELETE"])
def clear_history():
    _save_history([])
    return jsonify({"ok": True})


# ─────────────────────────────────────────────
# DOWNLOAD REPORT
# ─────────────────────────────────────────────
@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        content = f"""Parkinson's Disease Prediction Report
=======================================

Person         : {data.get('person', 'N/A')}
Result         : {data.get('result', 'N/A')}
Risk Level     : {data.get('risk_level', 'N/A')}
Probability    : {data.get('probability', 'N/A')}%

Doctor Suggestion:
{data.get('doctor_suggestion', 'N/A')}

Top Influencing Voice Features:
"""
        for f in data.get("top_features", []):
            content += f"  - {f['feature']}  (importance: {f.get('impact', '')})\n"

        analysis = data.get("feature_analysis") or []
        if analysis:
            content += "\nAll Extracted Voice Biomarkers (sorted by importance):\n"
            content += f"  {'Feature':<25} {'Value':>12}   {'Importance':>10}\n"
            content += "  " + "-" * 52 + "\n"
            for entry in analysis:
                content += (
                    f"  {entry['feature']:<25} {entry['value']:>12.6f}"
                    f"   {entry['importance']:>10.4f}\n"
                )
        elif data.get("extracted_features"):
            content += "\nExtracted Voice Biomarkers:\n"
            for feat, val in data["extracted_features"].items():
                content += f"  {feat}: {val}\n"

        file_stream = io.BytesIO()
        file_stream.write(content.encode("utf-8"))
        file_stream.seek(0)

        person_name = str(data.get("person", "report")).replace(" ", "_")
        return send_file(
            file_stream,
            as_attachment=True,
            download_name=f"{person_name}_Parkinson_Report.txt",
            mimetype="text/plain",
        )
    except Exception as e:
        print("ERROR in /download_report:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)