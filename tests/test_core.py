import os

import numpy as np
import pytest
import soundfile as sf
import taglib
from conftest import INTRO_SAMPLES, PATTERN_SAMPLES, SR

from pymusiclooper.core import MusicLooper

LOOP_START = INTRO_SAMPLES
LOOP_END = INTRO_SAMPLES + 2 * PATTERN_SAMPLES


@pytest.fixture(scope="module")
def looper(track_path):
    return MusicLooper(track_path)


# --- Split / extend ---


def test_export_splits_into_intro_loop_outro(looper, track, tmp_path):
    looper.export(LOOP_START, LOOP_END, format="WAV", output_dir=str(tmp_path))

    sections = [
        sf.read(tmp_path / f"track.wav-{name}.wav")[0] for name in ("intro", "loop", "outro")
    ]

    assert [s.shape[0] for s in sections] == [LOOP_START, LOOP_END - LOOP_START, track.size - LOOP_END]
    np.testing.assert_allclose(np.concatenate(sections), looper.mlaudio.playback_audio[:, 0], atol=1e-4)


def test_extend_with_fade_out_reaches_requested_length(looper, tmp_path):
    output_path = looper.extend(LOOP_START, LOOP_END, extended_length=60, fade_length=3, format="WAV", output_dir=str(tmp_path))

    extended, rate = sf.read(output_path)
    assert rate == SR
    assert extended.shape[0] / SR == pytest.approx(60, abs=0.05)
    assert np.max(np.abs(extended[-100:])) < 0.01, "track should fade out to silence"
    assert os.path.basename(output_path) == "track.wav-extended-1m00s.wav"


def test_extend_without_fade_out_keeps_outro(looper, tmp_path):
    output_path = looper.extend(LOOP_START, LOOP_END, extended_length=60, disable_fade_out=True, format="WAV", output_dir=str(tmp_path))

    extended, _ = sf.read(output_path)
    outro = looper.mlaudio.playback_audio[LOOP_END:, 0]
    assert extended.shape[0] / SR >= 60
    np.testing.assert_allclose(extended[-outro.size:], outro, atol=1e-4)


def test_extend_shorter_than_track_raises(looper, tmp_path):
    with pytest.raises(ValueError):
        looper.extend(LOOP_START, LOOP_END, extended_length=5, output_dir=str(tmp_path))


@pytest.mark.xfail(strict=True, reason="bug: fade_length=0 slices the whole final loop section (x[-0:]) and fails to broadcast")
def test_extend_with_zero_fade_length(looper, tmp_path):
    looper.extend(LOOP_START, LOOP_END, extended_length=60, fade_length=0, format="WAV", output_dir=str(tmp_path))


# --- Loop point text export ---


def test_export_txt_appends_lines(looper, tmp_path):
    looper.export_txt(LOOP_START, LOOP_END, output_dir=str(tmp_path))
    looper.export_txt(1, 2, output_dir=str(tmp_path))

    lines = (tmp_path / "loops.txt").read_text().splitlines()
    assert lines == [f"{LOOP_START} {LOOP_END} track.wav", "1 2 track.wav"]


# --- Metadata tags ---


@pytest.mark.parametrize(
    ("start_tag", "end_tag", "stored_end"),
    [
        ("LOOP_START", "LOOP_END", LOOP_END),
        ("LOOPSTART", "LOOPLENGTH", LOOP_END - LOOP_START),
    ],
)
def test_tags_roundtrip(flac_track_path, tmp_path, start_tag, end_tag, stored_end):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    written = MusicLooper(flac_track_path).export_tags(LOOP_START, LOOP_END, start_tag, end_tag, output_dir=str(out_dir))
    assert written == (str(LOOP_START), str(stored_end))

    tagged = MusicLooper(str(out_dir / "track-tagged.flac"))
    assert tagged.read_tags(start_tag, end_tag) == (LOOP_START, LOOP_END)
    assert tagged.read_tags(None, None) == (LOOP_START, LOOP_END), "tags should be auto-detected"


def test_read_tags_without_loop_tags_raises(flac_track_path):
    with pytest.raises(ValueError):
        MusicLooper(flac_track_path).read_tags(None, None)


@pytest.mark.xfail(strict=True, reason="bug: export_tags defaults output_dir to the source file path instead of its directory")
def test_export_tags_defaults_to_source_directory(flac_track_path, tmp_path):
    MusicLooper(flac_track_path).export_tags(LOOP_START, LOOP_END, "LOOP_START", "LOOP_END")
    assert (tmp_path / "track-tagged.flac").exists()


def test_extend_copies_source_tags(flac_track_path, tmp_path):
    with taglib.File(flac_track_path, save_on_exit=True) as source:
        source.tags["TITLE"] = ["Test Track"]

    output_path = MusicLooper(flac_track_path).extend(
        LOOP_START, LOOP_END, extended_length=60, format="FLAC", output_dir=str(tmp_path)
    )

    with taglib.File(output_path) as extended:
        assert extended.tags["TITLE"] == ["Test Track"]
