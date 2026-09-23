"""Checks that trimmed Ogg Vorbis files also end on the exact sample with FFmpeg, an independent Vorbis decoder.

FFmpeg is looked up in the PML_FFMPEG environment variable, then in tools/ffmpeg/, then on the PATH;
the tests are skipped if it is not found, or if it does not honor Vorbis end trimming at all (older builds).
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import SR

from pymusiclooper.ogg import trim_vorbis


def _find_ffmpeg():
    local_ffmpeg = Path(__file__).parent.parent / "tools" / "ffmpeg" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    for candidate in (os.environ.get("PML_FFMPEG"), local_ffmpeg, shutil.which("ffmpeg")):
        if candidate and os.path.isfile(candidate):
            return str(candidate)
    return None


def _decode(ffmpeg: str, path, decoder: str, n_channels: int) -> np.ndarray:
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-c:a", decoder, "-i", str(path), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, n_channels)


@pytest.fixture(scope="module")
def ffmpeg():
    path = _find_ffmpeg()
    if path is None:
        pytest.skip("ffmpeg not found (set PML_FFMPEG or place it in tools/ffmpeg/)")
    return path


@pytest.fixture(scope="module")
def stereo_ogg_path(track, tmp_path_factory):
    path = tmp_path_factory.mktemp("ogg") / "track.ogg"
    sf.write(path, np.stack([track, 0.8 * track], axis=1), SR, format="OGG", subtype="VORBIS")
    return path


@pytest.mark.parametrize("decoder", ["vorbis", "libvorbis"])
def test_ffmpeg_decodes_trimmed_ogg_to_exact_length(ffmpeg, stereo_ogg_path, tmp_path, decoder):
    source_length = sf.info(stereo_ogg_path).frames
    try:
        source = _decode(ffmpeg, stereo_ogg_path, decoder, n_channels=2)
    except subprocess.CalledProcessError:
        pytest.skip(f"this ffmpeg build has no '{decoder}' decoder")
    if source.shape[0] != source_length:
        pytest.skip("this ffmpeg build does not honor Vorbis end trimming, even on untouched files")

    # Cut points past the first audio page (a stream whose first audio page is also its last
    # is a special case in FFmpeg, and loops cannot realistically end that early)
    for n_samples in range(source_length // 4, source_length, 4999):
        output_path = tmp_path / "trimmed.ogg"
        trim_vorbis(str(stereo_ogg_path), str(output_path), n_samples)

        trimmed = _decode(ffmpeg, output_path, decoder, n_channels=2)
        assert trimmed.shape[0] == n_samples
        np.testing.assert_array_equal(trimmed, source[:n_samples])
