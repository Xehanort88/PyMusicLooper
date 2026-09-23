"""Shared fixtures: synthetic tracks whose correct loop points are known in advance.

The test track is an intro followed by the same 8-note pattern repeated several times,
so any loop whose length is a whole number of patterns is seamless.
"""

import numpy as np
import pytest
import soundfile as sf

SR = 22050
BEAT_SAMPLES = SR // 2  # 120 bpm
INTRO_NOTES = [50, 53, 57, 59]
PATTERN_NOTES = [60, 64, 67, 72, 69, 65, 62, 67]
PATTERN_REPEATS = 4

INTRO_SAMPLES = len(INTRO_NOTES) * BEAT_SAMPLES
PATTERN_SAMPLES = len(PATTERN_NOTES) * BEAT_SAMPLES

# Loop points are located on ~23ms STFT frames, then nudged to a zero crossing (+/-5ms)
SAMPLE_TOLERANCE = int(0.03 * SR)


def _note(midi: int) -> np.ndarray:
    t = np.arange(BEAT_SAMPLES) / SR
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    tone = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, 5))
    attack = np.minimum(1.0, t / 0.005)
    return tone * attack * np.exp(-4 * t)


def make_track() -> np.ndarray:
    notes = INTRO_NOTES + PATTERN_NOTES * PATTERN_REPEATS + PATTERN_NOTES[:1]
    y = np.concatenate([_note(n) for n in notes])
    return (0.5 * y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture(scope="session")
def track() -> np.ndarray:
    return make_track()


@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("audio")


@pytest.fixture(scope="session")
def track_path(audio_dir, track):
    path = audio_dir / "track.wav"
    sf.write(path, track, SR, subtype="FLOAT")
    return str(path)


@pytest.fixture(scope="session")
def stereo_track_path(audio_dir, track):
    path = audio_dir / "stereo.wav"
    sf.write(path, np.stack([track, 0.8 * track], axis=1), SR, subtype="FLOAT")
    return str(path)


@pytest.fixture(scope="session")
def silent_path(audio_dir):
    path = audio_dir / "silent.wav"
    sf.write(path, np.zeros(SR, dtype=np.float32), SR)
    return str(path)


@pytest.fixture
def flac_track_path(tmp_path, track):
    """A fresh FLAC copy per test, since tagging tests write next to it."""
    path = tmp_path / "track.flac"
    sf.write(path, track, SR, format="FLAC")
    return str(path)


def assert_whole_patterns(loop_start: int, loop_end: int):
    """Asserts that a loop spans a whole number of repeated patterns."""
    length = loop_end - loop_start
    n_patterns = round(length / PATTERN_SAMPLES)
    assert n_patterns >= 1, f"loop of {length} samples is shorter than one pattern"
    assert abs(length - n_patterns * PATTERN_SAMPLES) <= SAMPLE_TOLERANCE, (
        f"loop of {length} samples is not a multiple of the {PATTERN_SAMPLES}-sample pattern"
    )
