import os

import pytest
import soundfile as sf
from conftest import SR

from pymusiclooper import handler
from pymusiclooper.analysis import LoopPair
from pymusiclooper.core import MusicLooper
from pymusiclooper.handler import BatchHandler, LoopExportHandler, LoopHandler


def _bare_loop_handler(n_pairs=30):
    """A LoopHandler with preset loop pairs, skipping audio loading and analysis."""
    loop_handler = LoopHandler.__new__(LoopHandler)
    loop_handler.loop_pair_list = [
        LoopPair(0, 0, note_distance=0.0, loudness_difference=0.0, score=1 - i / 100, loop_start=i, loop_end=i + 1000)
        for i in range(n_pairs)
    ]
    loop_handler.filepath = "track.wav"
    loop_handler.in_samples = True
    loop_handler._musiclooper = None
    loop_handler._progressbar = None
    return loop_handler


@pytest.fixture
def fake_input(monkeypatch):
    """Feeds the given answers to the interactive prompt, in order."""
    def feed(*answers):
        remaining = iter(answers)
        monkeypatch.setattr(handler.rich_console, "input", lambda *args, **kwargs: next(remaining))
    return feed


def test_interactive_selects_entered_index(fake_input):
    fake_input("3")
    assert _bare_loop_handler().interactive_handler() == 3


def test_interactive_reprompts_on_invalid_input(fake_input):
    fake_input("abc", "", "99", "4")
    assert _bare_loop_handler().interactive_handler() == 4


def test_interactive_more_then_select(fake_input):
    fake_input("more", "27", "1")
    assert _bare_loop_handler().interactive_handler() == 27


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, EOFError])
def test_interactive_exits_cleanly_on_interrupt(monkeypatch, interrupt):
    def raise_interrupt(*args, **kwargs):
        raise interrupt
    monkeypatch.setattr(handler.rich_console, "input", raise_interrupt)
    monkeypatch.setattr(handler.signal, "signal", lambda *args: None)
    with pytest.raises(SystemExit):
        _bare_loop_handler().interactive_handler()


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (("",), (False, 0)),
        (("n",), (False, 0)),
        (("y", ""), (True, handler.RECOMMENDED_KEEP_AFTER)),
        (("Y", "0"), (True, 0)),
        (("yes", "abc", "-5", "1.5", "10"), (True, 10)),
    ],
)
def test_trim_prompt(fake_input, answers, expected):
    fake_input(*answers)
    assert LoopExportHandler.trim_prompt(None) == expected


def test_trim_prompt_exits_cleanly_on_interrupt(monkeypatch):
    def raise_interrupt(*args, **kwargs):
        raise EOFError
    monkeypatch.setattr(handler.rich_console, "input", raise_interrupt)
    monkeypatch.setattr(handler.signal, "signal", lambda *args: None)
    with pytest.raises(SystemExit):
        LoopExportHandler.trim_prompt(None)


def _interactive_export_handler(monkeypatch, source_path, output_dir, **kwargs):
    monkeypatch.setenv("PML_INTERACTIVE_MODE", "1")
    monkeypatch.setattr(handler.rich_console, "print", lambda *args, **kwargs: None)
    return LoopExportHandler(path=str(source_path), min_duration_multiplier=0.35, output_dir=str(output_dir), **kwargs)


def test_interactive_tag_with_trim_prompt_writes_single_file(monkeypatch, fake_input, track, tmp_path):
    source_path = tmp_path / "track.flac"
    sf.write(source_path, track, SR)
    out_dir = tmp_path / "out"
    export_handler = _interactive_export_handler(monkeypatch, source_path, out_dir, tag_names=("LOOPSTART", "LOOPLENGTH"))
    loop_pair = export_handler.loop_pair_list[0]

    fake_input("0", "y", "")
    export_handler.run()

    assert os.listdir(out_dir) == ["track-tagged.flac"]
    tagged = MusicLooper(str(out_dir / "track-tagged.flac"))
    assert tagged.read_tags("LOOPSTART", "LOOPLENGTH") == (loop_pair.loop_start, loop_pair.loop_end)
    assert sf.info(out_dir / "track-tagged.flac").frames == loop_pair.loop_end + handler.RECOMMENDED_KEEP_AFTER


def test_interactive_trim_prompt_declined(monkeypatch, fake_input, track, tmp_path):
    source_path = tmp_path / "track.flac"
    sf.write(source_path, track, SR)
    out_dir = tmp_path / "out"
    export_handler = _interactive_export_handler(monkeypatch, source_path, out_dir, tag_names=("LOOP_START", "LOOP_END"))

    fake_input("0", "")
    export_handler.run()

    assert os.listdir(out_dir) == ["track-tagged.flac"]
    assert sf.info(out_dir / "track-tagged.flac").frames == track.size


def test_interactive_trim_prompt_skipped_for_unsupported_formats(monkeypatch, fake_input, track, tmp_path):
    source_path = tmp_path / "track.mp3"
    sf.write(source_path, track, SR, format="MP3")
    export_handler = _interactive_export_handler(monkeypatch, source_path, tmp_path / "out", to_stdout=True)

    # Only the loop selection is answered; a trim prompt would exhaust the answers
    fake_input("0")
    export_handler.run()


def test_interactive_trim_command_does_not_prompt_again(monkeypatch, fake_input, track, tmp_path):
    source_path = tmp_path / "track.flac"
    sf.write(source_path, track, SR)
    out_dir = tmp_path / "out"
    export_handler = _interactive_export_handler(monkeypatch, source_path, out_dir, trim=True, keep_after=7)
    loop_pair = export_handler.loop_pair_list[0]

    fake_input("0")
    export_handler.run()

    assert os.listdir(out_dir) == ["track-trimmed.flac"]
    assert sf.info(out_dir / "track-trimmed.flac").frames == loop_pair.loop_end + 7


def test_choose_loop_pair_defaults_to_best():
    loop_handler = _bare_loop_handler()
    assert loop_handler.choose_loop_pair(interactive_mode=False) is loop_handler.loop_pair_list[0]


def test_txt_export_message_names_written_file(monkeypatch, track_path, tmp_path):
    export_handler = LoopExportHandler.__new__(LoopExportHandler)
    export_handler._musiclooper = MusicLooper(track_path)
    export_handler.output_directory = str(tmp_path)
    export_handler.alt_export_top = 0
    export_handler.fmt = "samples"
    export_handler.batch_mode = False

    messages = []
    monkeypatch.setattr(handler.rich_console, "print", lambda msg, *args, **kwargs: messages.append(msg))
    export_handler.txt_export_runner(100, 200)

    written = os.listdir(tmp_path)
    assert written == ["loops.txt"]
    assert str(tmp_path / "loops.txt") in messages[0]


def test_get_files_in_directory(tmp_path):
    (tmp_path / "a.wav").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.wav").touch()

    flat = BatchHandler.get_files_in_directory(str(tmp_path))
    recursive = BatchHandler.get_files_in_directory(str(tmp_path), recursive=True)

    assert flat == [str(tmp_path / "a.wav")]
    assert sorted(recursive) == sorted([str(tmp_path / "a.wav"), str(tmp_path / "sub" / "b.wav")])
