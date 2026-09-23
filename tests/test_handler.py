import os

import pytest

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


@pytest.mark.xfail(strict=True, reason="bug: the choice made after 'more'/'all'/'reset' is discarded and the user is prompted again")
def test_interactive_more_then_select(fake_input):
    fake_input("more", "27", "1")
    assert _bare_loop_handler().interactive_handler() == 27


def test_choose_loop_pair_defaults_to_best():
    loop_handler = _bare_loop_handler()
    assert loop_handler.choose_loop_pair(interactive_mode=False) is loop_handler.loop_pair_list[0]


@pytest.mark.xfail(strict=True, reason="bug: success message names loop.txt but the file written is loops.txt")
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
