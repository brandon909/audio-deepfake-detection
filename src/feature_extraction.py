"""Feature extraction for speech authenticity classification."""

import librosa
import numpy as np


def summarize_feature(values: np.ndarray) -> dict[str, float]:
    """Return common summary statistics for a feature array."""
    flattened = values.flatten()
    return {
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
        "median": float(np.median(flattened)),
    }


def extract_audio_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Extract spectral and cepstral audio features from a waveform."""
    features: dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    feature_groups = {
        "mfcc": mfcc,
        "delta_mfcc": delta_mfcc,
        "delta2_mfcc": delta2_mfcc,
        "spectral_centroid": librosa.feature.spectral_centroid(y=audio, sr=sample_rate),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=audio, sr=sample_rate),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(audio),
        "rms": librosa.feature.rms(y=audio),
    }

    for name, values in feature_groups.items():
        for stat_name, stat_value in summarize_feature(values).items():
            features[f"{name}_{stat_name}"] = stat_value

    return features

