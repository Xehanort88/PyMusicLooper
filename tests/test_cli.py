import re
import shutil

import pytest
from click.testing import CliRunner
from conftest import SR, assert_whole_patterns

from pymusiclooper import __version__
from pymusiclooper.cli import cli_main


@pytest.fixture
def run():
    def invoke(*args):
        result = CliRunner().invoke(cli_main, [str(arg) for arg in args])
        assert result.exception is None or isinstance(result.exception, SystemExit), result.output
        return result
    return invoke


def test_version(run):
    result = run("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_path_is_required(run):
    result = run("export-points")
    assert result.exit_code != 0


def test_export_points_to_stdout(run, track_path):
    result = run("export-points", "--path", track_path)

    assert result.exit_code == 0
    loop_start = int(re.search(r"LOOP_START: (\d+)", result.output).group(1))
    loop_end = int(re.search(r"LOOP_END: (\d+)", result.output).group(1))
    assert_whole_patterns(loop_start, loop_end)


def test_export_points_in_seconds(run, track_path):
    result = run("export-points", "--path", track_path, "--fmt", "SECONDS")

    loop_start = float(re.search(r"LOOP_START: ([\d.]+)", result.output).group(1))
    loop_end = float(re.search(r"LOOP_END: ([\d.]+)", result.output).group(1))
    assert_whole_patterns(round(loop_start * SR), round(loop_end * SR))


def test_export_points_alt_export_top(run, track_path):
    result = run("export-points", "--path", track_path, "--alt-export-top", "3")

    lines = result.output.strip().splitlines()
    assert len(lines) == 3
    assert all(len(line.split()) == 5 for line in lines)


def test_export_points_to_txt(run, track_path, tmp_path):
    run("export-points", "--path", track_path, "--export-to", "TXT", "--output-dir", tmp_path)

    lines = (tmp_path / "loops.txt").read_text().splitlines()
    assert len(lines) == 1
    assert lines[0].endswith(" track.wav")


def test_split_audio(run, track_path, tmp_path):
    run("split-audio", "--path", track_path, "--output-dir", tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "track.wav-intro.wav",
        "track.wav-loop.wav",
        "track.wav-outro.wav",
    ]


def test_extend(run, track_path, tmp_path):
    run("extend", "--path", track_path, "--output-dir", tmp_path, "--extended-length", 45, "--format", "WAV")

    assert [p.name for p in tmp_path.iterdir()] == ["track.wav-extended-0m45s.wav"]


def test_batch_export_skips_non_audio_files(run, track_path, stereo_track_path, tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    shutil.copy(track_path, in_dir)
    shutil.copy(stereo_track_path, in_dir)
    (in_dir / "notes.txt").write_text("not audio")

    result = run("export-points", "--path", in_dir, "--export-to", "TXT", "--output-dir", out_dir)

    assert result.exit_code == 0
    lines = (out_dir / "loops.txt").read_text().splitlines()
    assert sorted(line.split()[-1] for line in lines) == ["stereo.wav", "track.wav"]


def test_trim_keeps_recommended_samples_after_loop_end_by_default(run, track_path, tmp_path):
    import soundfile as sf

    from pymusiclooper.core import MusicLooper
    from pymusiclooper.handler import RECOMMENDED_KEEP_AFTER

    result = run("trim", "--path", track_path, "--output-dir", tmp_path)

    assert result.exit_code == 0
    loop_end = MusicLooper(track_path).find_loop_pairs()[0].loop_end
    assert sf.info(tmp_path / "track-trimmed.wav").frames == loop_end + RECOMMENDED_KEEP_AFTER
