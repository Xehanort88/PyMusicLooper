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


# --- Embedded loop points ---


def _write_loop_tags(path: str, loop_start: int, loop_end: int):
    with taglib.File(path, save_on_exit=True) as audio_file:
        audio_file.tags["LOOP_START"] = [str(loop_start)]
        audio_file.tags["LOOP_END"] = [str(loop_end)]


def test_find_loop_pairs_puts_embedded_tags_first(flac_track_path):
    _write_loop_tags(flac_track_path, LOOP_START + 123, LOOP_END + 123)

    pairs = MusicLooper(flac_track_path).find_loop_pairs()

    assert pairs[0].from_metadata
    assert (pairs[0].loop_start, pairs[0].loop_end) == (LOOP_START + 123, LOOP_END + 123)
    assert len(pairs) > 1, "detected loop points should still follow the embedded ones"
    assert not any(pair.from_metadata for pair in pairs[1:])


def test_find_loop_pairs_can_ignore_embedded_tags(flac_track_path):
    _write_loop_tags(flac_track_path, LOOP_START + 123, LOOP_END + 123)

    pairs = MusicLooper(flac_track_path).find_loop_pairs(use_embedded_tags=False)

    assert not any(pair.from_metadata for pair in pairs)


def test_find_loop_pairs_ignores_embedded_tags_with_approx_position(flac_track_path):
    _write_loop_tags(flac_track_path, LOOP_START + 123, LOOP_END + 123)

    pairs = MusicLooper(flac_track_path).find_loop_pairs(
        approx_loop_start=LOOP_START / SR, approx_loop_end=LOOP_END / SR
    )

    assert not any(pair.from_metadata for pair in pairs)


def test_embedded_tags_out_of_range_are_ignored(flac_track_path, track):
    _write_loop_tags(flac_track_path, LOOP_START, track.size + 1)

    assert MusicLooper(flac_track_path).read_embedded_loop_pair() is None


def test_no_embedded_tags(track_path):
    assert MusicLooper(track_path).read_embedded_loop_pair() is None


# --- Lossless trim ---


@pytest.mark.parametrize(
    ("filename", "subtype", "dtype"),
    [
        ("pcm16.wav", "PCM_16", "int16"),
        ("pcm24.flac", "PCM_24", "int32"),
        ("float.wav", "FLOAT", "float32"),
    ],
)
def test_trim_is_bit_exact(track, tmp_path, filename, subtype, dtype):
    source_path = tmp_path / filename
    sf.write(source_path, np.stack([track, 0.8 * track], axis=1), SR, subtype=subtype)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    output_path = MusicLooper(str(source_path)).trim(LOOP_END, keep_after=100, output_dir=str(out_dir))

    source, _ = sf.read(source_path, dtype=dtype)
    trimmed, rate = sf.read(output_path, dtype=dtype)
    assert rate == SR
    assert sf.info(output_path).subtype == subtype
    assert os.path.basename(output_path) == f"{os.path.splitext(filename)[0]}-trimmed{os.path.splitext(filename)[1]}"
    np.testing.assert_array_equal(trimmed, source[: LOOP_END + 100])


def test_trim_keep_after_past_track_end_keeps_whole_track(flac_track_path, track, tmp_path):
    output_path = MusicLooper(flac_track_path).trim(LOOP_END, keep_after=10 * track.size, output_dir=str(tmp_path))

    assert sf.info(output_path).frames == track.size


def test_trim_copies_source_tags(flac_track_path, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with taglib.File(flac_track_path, save_on_exit=True) as source:
        source.tags["LOOP_START"] = [str(LOOP_START)]
        source.tags["LOOP_END"] = [str(LOOP_END)]

    output_path = MusicLooper(flac_track_path).trim(LOOP_END, output_dir=str(out_dir))

    assert MusicLooper(output_path).read_tags(None, None) == (LOOP_START, LOOP_END)


@pytest.mark.parametrize("keep_after", [0, 777])
def test_trim_ogg_vorbis_without_reencoding(track, tmp_path, keep_after):
    source_path = tmp_path / "track.ogg"
    sf.write(source_path, np.stack([track, 0.8 * track], axis=1), SR, format="OGG", subtype="VORBIS")
    with taglib.File(str(source_path), save_on_exit=True) as source:
        source.tags["LOOPSTART"] = [str(LOOP_START)]
        source.tags["LOOPLENGTH"] = [str(LOOP_END - LOOP_START)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    output_path = MusicLooper(str(source_path)).trim(LOOP_END, keep_after=keep_after, output_dir=str(out_dir))

    source, _ = sf.read(source_path, dtype="float32")
    trimmed, _ = sf.read(output_path, dtype="float32")
    assert os.path.basename(output_path) == "track-trimmed.ogg"
    np.testing.assert_array_equal(trimmed, source[: LOOP_END + keep_after])
    assert MusicLooper(output_path).read_tags(None, None) == (LOOP_START, LOOP_END)


def test_trim_ogg_vorbis_past_track_end_keeps_whole_track(track, tmp_path):
    source_path = tmp_path / "track.ogg"
    sf.write(source_path, track, SR, format="OGG", subtype="VORBIS")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    output_path = MusicLooper(str(source_path)).trim(LOOP_END, keep_after=10 * track.size, output_dir=str(out_dir))

    assert open(output_path, "rb").read() == open(source_path, "rb").read()


@pytest.mark.parametrize(
    ("filename", "format", "subtype"),
    [
        ("track.opus", "OGG", "OPUS"),
        ("track.mp3", "MP3", "MPEG_LAYER_III"),
        ("ulaw.wav", "WAV", "ULAW"),
    ],
)
def test_trim_rejects_unsupported_formats(track, tmp_path, filename, format, subtype):
    source_path = tmp_path / filename
    sf.write(source_path, track, SR if subtype != "OPUS" else 48000, format=format, subtype=subtype)

    with pytest.raises(ValueError):
        MusicLooper(str(source_path)).trim(LOOP_END, output_dir=str(tmp_path))


@pytest.mark.parametrize(
    ("filename", "format", "subtype", "supported"),
    [
        ("pcm16.wav", "WAV", "PCM_16", True),
        ("float.wav", "WAV", "FLOAT", True),
        ("track.flac", "FLAC", "PCM_16", True),
        ("track.ogg", "OGG", "VORBIS", True),
        ("track.opus", "OGG", "OPUS", False),
        ("track.mp3", "MP3", "MPEG_LAYER_III", False),
        ("ulaw.wav", "WAV", "ULAW", False),
    ],
)
def test_supports_lossless_trim(track, tmp_path, filename, format, subtype, supported):
    source_path = tmp_path / filename
    sf.write(source_path, track, SR if subtype != "OPUS" else 48000, format=format, subtype=subtype)

    assert MusicLooper(str(source_path)).supports_lossless_trim() is supported


@pytest.mark.parametrize("filename", ["track.flac", "track.ogg"])
def test_export_tags_with_trim_writes_single_tagged_trimmed_file(track, tmp_path, filename):
    source_path = tmp_path / filename
    sf.write(source_path, track, SR)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    MusicLooper(str(source_path)).export_tags(
        LOOP_START, LOOP_END, "LOOPSTART", "LOOPLENGTH", output_dir=str(out_dir), trim=True, keep_after=100
    )

    track_name, extension = os.path.splitext(filename)
    assert os.listdir(out_dir) == [f"{track_name}-tagged{extension}"]
    output_path = str(out_dir / f"{track_name}-tagged{extension}")
    assert sf.info(output_path).frames == LOOP_END + 100
    assert MusicLooper(output_path).read_tags("LOOPSTART", "LOOPLENGTH") == (LOOP_START, LOOP_END)


def test_export_tags_with_trim_rejects_unsupported_formats(track, tmp_path):
    source_path = tmp_path / "track.mp3"
    sf.write(source_path, track, SR, format="MP3")

    with pytest.raises(ValueError):
        MusicLooper(str(source_path)).export_tags(LOOP_START, LOOP_END, "LOOP_START", "LOOP_END", output_dir=str(tmp_path), trim=True)


def test_ogg_vorbis_packet_durations_match_granule_positions(track, tmp_path):
    from pymusiclooper.ogg import _find_packet_cut, _read_pages

    source_path = tmp_path / "track.ogg"
    sf.write(source_path, np.stack([track, 0.8 * track], axis=1), SR, format="OGG", subtype="VORBIS")
    pages = list(_read_pages(source_path.read_bytes()))

    # Raises if the computed packet durations do not add up to the page granule positions
    for n_samples in range(1, track.size, 997):
        assert _find_packet_cut(pages, n_samples) is not None
    assert _find_packet_cut(pages, track.size + 1) is None


def test_trim_ogg_vorbis_falls_back_to_page_level_cut(track, tmp_path, monkeypatch):
    from pymusiclooper import ogg

    def fail(*args, **kwargs):
        raise ValueError("unparseable")

    monkeypatch.setattr(ogg, "_find_packet_cut", fail)
    source_path = tmp_path / "track.ogg"
    sf.write(source_path, track, SR, format="OGG", subtype="VORBIS")
    output_path = tmp_path / "trimmed.ogg"

    assert ogg.trim_vorbis(str(source_path), str(output_path), LOOP_END) == LOOP_END

    source, _ = sf.read(source_path, dtype="float32")
    trimmed, _ = sf.read(output_path, dtype="float32")
    np.testing.assert_array_equal(trimmed, source[:LOOP_END])
