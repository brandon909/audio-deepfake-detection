"""Audio loading and preprocessing helpers."""

from pathlib import Path

import librosa
import numpy as np


def load_audio(path: str | Path, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono waveform."""
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio, sr


def remove_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading and trailing silence from a waveform."""
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim audio to a fixed number of samples."""
    if len(audio) > target_length:
        return audio[:target_length]
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio

