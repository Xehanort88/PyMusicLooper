import numpy as np
import pytest
from conftest import INTRO_SAMPLES, SAMPLE_TOLERANCE, SR, assert_whole_patterns

from pymusiclooper import analysis
from pymusiclooper.analysis import (
    LoopPair,
    _calculate_loop_score,
    _calculate_subseq_beat_similarity,
    _prioritize_duration,
    _weights,
    nearest_zero_crossing,
)
from pymusiclooper.core import MusicLooper
from pymusiclooper.exceptions import LoopNotFoundError


@pytest.fixture(scope="module")
def looper(track_path):
    return MusicLooper(track_path)


@pytest.fixture(scope="module")
def pairs(looper):
    return looper.find_loop_pairs()


# --- End-to-end loop detection ---


def test_best_loop_spans_whole_patterns(pairs):
    best = pairs[0]
    assert_whole_patterns(best.loop_start, best.loop_end)
    assert best.loop_start >= INTRO_SAMPLES - SAMPLE_TOLERANCE, "loop should not start inside the intro"


def test_best_loop_scores_highly(pairs):
    assert pairs[0].score > 0.95


def test_pairs_are_sorted_ordered_and_in_bounds(looper, pairs):
    scores = [pair.score for pair in pairs[1:]]
    assert scores == sorted(scores, reverse=True)
    for pair in pairs:
        assert 0 <= pair.loop_start < pair.loop_end <= looper.mlaudio.length


def test_default_min_duration_is_35_percent_of_track(looper, pairs):
    min_samples = 0.35 * looper.mlaudio.length
    assert all(p.loop_end - p.loop_start >= min_samples - SAMPLE_TOLERANCE - SR for p in pairs)


def test_min_and_max_loop_duration_respected(looper):
    pairs = looper.find_loop_pairs(min_loop_duration=5, max_loop_duration=9)
    for pair in pairs:
        seconds = (pair.loop_end - pair.loop_start) / SR
        assert 5 - 0.05 <= seconds <= 9 + 0.05
    assert_whole_patterns(pairs[0].loop_start, pairs[0].loop_end)


def test_approx_loop_position_respected(looper):
    pairs = looper.find_loop_pairs(approx_loop_start=6.0, approx_loop_end=14.0)
    best = pairs[0]
    assert abs(best.loop_start / SR - 6.0) <= 2.05
    assert abs(best.loop_end / SR - 14.0) <= 2.05
    assert_whole_patterns(best.loop_start, best.loop_end)


def test_disable_pruning_keeps_at_least_as_many_pairs(looper, pairs):
    unpruned = looper.find_loop_pairs(disable_pruning=True)
    assert len(unpruned) >= len(pairs)


def test_impossible_constraints_raise(looper):
    with pytest.raises(LoopNotFoundError):
        looper.find_loop_pairs(min_loop_duration=60)


def test_prioritize_duration_sees_real_loop_positions(monkeypatch, looper):
    durations_seen = []
    original = analysis._prioritize_duration

    def spy(pair_list):
        durations_seen.extend(p.loop_end - p.loop_start for p in pair_list)
        return original(pair_list)

    monkeypatch.setattr(analysis, "_prioritize_duration", spy)
    looper.find_loop_pairs()

    assert durations_seen, "_prioritize_duration was not called"
    assert max(durations_seen) > 0


# --- Scoring ---


def _pair(start, end, score, loudness=0.1):
    return LoopPair(0, 0, note_distance=0.0, loudness_difference=loudness, score=score, loop_start=start, loop_end=end)


def test_prioritize_duration_prefers_longest_among_tied_scores():
    pairs = [_pair(0, 100, 1.0), _pair(0, 300, 1.0), _pair(0, 900, 0.5)]
    _prioritize_duration(pairs)
    assert pairs[0].loop_end == 300


def test_prioritize_duration_keeps_clearly_better_score_first():
    pairs = [_pair(0, 100, 0.99), _pair(0, 300, 0.90)]
    _prioritize_duration(pairs)
    assert pairs[0].loop_end == 100


def _random_chroma(n_frames=200, seed=0):
    return np.random.default_rng(seed).random((12, n_frames))


def test_identical_sequences_score_one():
    chroma = _random_chroma()
    chroma[:, 100:150] = chroma[:, 20:70]
    score = _calculate_loop_score(30, 110, chroma, test_duration=20, weights=_weights(20, start=5))
    assert score == pytest.approx(1.0)


def test_unrelated_sequences_score_lower():
    chroma = _random_chroma()
    chroma[:, 100:150] = chroma[:, 20:70]
    matching = _calculate_loop_score(30, 110, chroma, test_duration=20, weights=_weights(20, start=5))
    unrelated = _calculate_loop_score(30, 160, chroma, test_duration=20, weights=_weights(20, start=5))
    assert unrelated < matching


def test_truncated_lookbehind_weights_frames_nearest_the_loop_point():
    chroma = _random_chroma()
    # Only 3 frames exist before b1=3; they match the 3 frames before b2=100
    chroma[:, 0:3] = chroma[:, 97:100]
    weights = _weights(20, start=10)[::-1]  # heaviest weight on the frame right before the loop point

    score = _calculate_subseq_beat_similarity(3, 100, chroma, -20, weights=weights)

    assert score == pytest.approx(weights[-3:].sum() / weights.sum())


# --- Zero crossings ---


def _sine(freq=100.0, seconds=1.0, channels=1):
    t = np.arange(int(seconds * SR)) / SR
    y = np.sin(2 * np.pi * freq * t)
    return np.tile(y[:, np.newaxis], (1, channels))


def _assert_rising_zero_crossing(audio, idx):
    assert abs(audio[idx, 0]) < 0.05
    assert audio[idx + 1, 0] > audio[idx, 0]


@pytest.mark.parametrize("channels", [1, 2])
def test_zero_crossing_snaps_to_nearby_rising_crossing(channels):
    audio = _sine(channels=channels)
    idx = nearest_zero_crossing(audio, SR, 1000)
    assert abs(idx - 1000) <= SR // 200
    _assert_rising_zero_crossing(audio, idx)


def test_zero_crossing_near_start_of_audio():
    audio = _sine()
    idx = nearest_zero_crossing(audio, SR, 30)
    assert 0 <= idx <= 30 + SR // 200
    _assert_rising_zero_crossing(audio, idx)


def test_zero_crossing_keeps_index_without_a_crossing():
    audio = np.full((SR, 1), 0.8)
    assert nearest_zero_crossing(audio, SR, 1000) == 1000
