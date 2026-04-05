#!/usr/bin/env python3
"""
Generate a small WAV dataset with synthetic voice-like signals for testing
the Parkinson's voice analysis app. Creates several collections with
multiple WAV files each (22050 Hz, mono, 2–4 s).
"""

import os
import numpy as np
import soundfile as sf

# Match backend: 22050 Hz
SR = 22050
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "wav_dataset")


def generate_voice_like(
    duration_sec,
    base_f0_hz,
    jitter_amount=0.02,
    shimmer_amount=0.03,
    noise_ratio=0.0,
    seed=None,
):
    """
    Generate a synthetic vowel-like signal that Praat can analyze (pitch, jitter, shimmer, HNR).
    - Higher jitter_amount/shimmer_amount -> more cycle-to-cycle variation -> higher jitter/shimmer.
    - noise_ratio > 0 adds broadband noise -> lower HNR (more "pathological").
    """
    if seed is not None:
        np.random.seed(seed)
    n = int(duration_sec * SR)
    t = np.arange(n, dtype=np.float64) / SR

    # Phase: linear + random walk for jitter (cycle-to-cycle period variation)
    phase = 2 * np.pi * base_f0_hz * t
    jitter_walk = np.cumsum(jitter_amount * (np.random.rand(n) - 0.5))
    phase = phase + jitter_walk

    # Amplitude envelope (shimmer = cycle-to-cycle amplitude variation)
    env = 1.0 + shimmer_amount * (np.random.rand(n) - 0.5)
    env = np.clip(env, 0.4, 1.0)

    # Sum of harmonics (vowel-like)
    y = np.zeros(n)
    for h in [1, 2, 3, 4]:
        y += (1.0 / h) * np.sin(h * phase)
    y *= env

    # Add noise to lower HNR (used for high-risk samples)
    if noise_ratio > 0:
        noise = np.random.randn(n).astype(np.float32) * noise_ratio
        y = y + noise

    y *= 0.3 / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    # (fname, dur, f0, jitter_amt, shimmer_amt, noise_ratio, seed)
    collections = [
        # Pitch-variety collections (neutral test set)
        {
            "name": "collection_1",
            "description": "Lower pitch (~120 Hz), 3 samples",
            "samples": [
                ("sample_01.wav", 2.5, 120, 0.015, 0.02, 0.0, 10),
                ("sample_02.wav", 3.0, 118, 0.02, 0.025, 0.0, 11),
                ("sample_03.wav", 2.8, 122, 0.018, 0.022, 0.0, 12),
            ],
        },
        {
            "name": "collection_2",
            "description": "Mid pitch (~180 Hz), 3 samples",
            "samples": [
                ("sample_01.wav", 3.0, 180, 0.02, 0.03, 0.0, 20),
                ("sample_02.wav", 2.5, 178, 0.018, 0.028, 0.0, 21),
                ("sample_03.wav", 3.2, 182, 0.022, 0.032, 0.0, 22),
            ],
        },
        {
            "name": "collection_3",
            "description": "Higher pitch (~220 Hz), 3 samples",
            "samples": [
                ("sample_01.wav", 2.8, 220, 0.025, 0.035, 0.0, 30),
                ("sample_02.wav", 3.0, 218, 0.02, 0.03, 0.0, 31),
                ("sample_03.wav", 2.6, 225, 0.028, 0.038, 0.0, 32),
            ],
        },
        # Risk-level collections (tuned so Praat extracts healthy- vs PD-like features)
        {
            "name": "collection_low_risk",
            "description": "Low risk — stable pitch, low jitter/shimmer, high HNR",
            "samples": [
                ("sample_01.wav", 3.0, 150, 0.004, 0.008, 0.0, 101),
                ("sample_02.wav", 2.8, 155, 0.005, 0.010, 0.0, 102),
                ("sample_03.wav", 3.2, 148, 0.003, 0.007, 0.0, 103),
            ],
        },
        {
            "name": "collection_medium_risk",
            "description": "Medium risk — moderate jitter/shimmer and noise",
            "samples": [
                ("sample_01.wav", 3.0, 160, 0.012, 0.035, 0.06, 201),
                ("sample_02.wav", 2.8, 158, 0.014, 0.040, 0.07, 202),
                ("sample_03.wav", 3.0, 162, 0.011, 0.032, 0.05, 203),
            ],
        },
        {
            "name": "collection_high_risk",
            "description": "High risk — high jitter/shimmer, low HNR (noise)",
            "samples": [
                ("sample_01.wav", 3.0, 165, 0.025, 0.070, 0.12, 301),
                ("sample_02.wav", 2.8, 162, 0.028, 0.075, 0.14, 302),
                ("sample_03.wav", 3.0, 168, 0.022, 0.065, 0.10, 303),
            ],
        },
    ]

    for coll in collections:
        coll_path = os.path.join(DATASET_DIR, coll["name"])
        os.makedirs(coll_path, exist_ok=True)
        print(f"Writing {coll['name']}: {coll['description']}")
        for row in coll["samples"]:
            fname, dur, f0, jit, shim, noise, seed = row
            y = generate_voice_like(
                dur, f0,
                jitter_amount=jit,
                shimmer_amount=shim,
                noise_ratio=noise,
                seed=seed,
            )
            path = os.path.join(coll_path, fname)
            sf.write(path, y, SR)
            print(f"  {fname} ({dur}s, F0≈{f0} Hz, jit={jit}, shim={shim}, noise={noise})")
    print(f"\nDataset written to: {DATASET_DIR}")


if __name__ == "__main__":
    main()
