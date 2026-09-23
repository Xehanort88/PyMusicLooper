import numpy as np
import pytest
import soundfile as sf
from conftest import SR, make_track

from pymusiclooper.audio import MLAudio
from pymusiclooper.exceptions import AudioLoadError


def test_loads_mono(track_path, track):
    audio = MLAudio(track_path)
    assert audio.rate == SR
    assert audio.n_channels == 1
    assert audio.length == track.size
    assert audio.playback_audio.shape == (track.size, 1)
    assert audio.total_duration == pytest.approx(track.size / SR)
    assert audio.filename == "track.wav"


def test_loads_stereo(stereo_track_path, track):
    audio = MLAudio(stereo_track_path)
    assert audio.n_channels == 2
    assert audio.playback_audio.shape == (track.size, 2)


def test_analysis_signal_is_normalized_mono(stereo_track_path):
    audio = MLAudio(stereo_track_path)
    assert audio.audio.ndim == 1
    assert np.max(np.abs(audio.audio)) == pytest.approx(1.0)


@pytest.mark.xfail(strict=True, reason="bug: for mono input, to_mono returns the same array, so normalizing it in place also rescales the playback/export audio")
def test_playback_audio_is_untouched(track_path, track):
    audio = MLAudio(track_path)
    np.testing.assert_array_equal(audio.playback_audio[:, 0], track)


def test_silent_file_raises(silent_path):
    with pytest.raises(AudioLoadError):
        MLAudio(silent_path)


def test_non_audio_file_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not audio")
    with pytest.raises(AudioLoadError):
        MLAudio(str(path))


def test_leading_silence_is_trimmed_on_frame_boundary(tmp_path):
    silence = np.zeros(SR + 1234, dtype=np.float32)
    path = tmp_path / "padded.wav"
    sf.write(path, np.concatenate([silence, make_track()]), SR, subtype="FLOAT")

    audio = MLAudio(str(path))

    assert 0 < audio.trim_offset <= silence.size
    # apply_trim_offset round-trips through frames, which is only exact
    # because librosa trims on hop-length (512 sample) boundaries
    assert audio.trim_offset % 512 == 0
    assert audio.frames_to_samples(audio.apply_trim_offset(10)) == (
        audio.frames_to_samples(10) + audio.trim_offset
    )


def test_time_conversions(track_path):
    audio = MLAudio(track_path)
    assert audio.seconds_to_samples(1) == SR
    assert audio.samples_to_seconds(SR) == pytest.approx(1.0)
    assert audio.frames_to_samples(1) == 512
    assert audio.samples_to_frames(1024) == 2
    assert audio.samples_to_ftime(int(61.5 * SR)) == "01:01.500"
