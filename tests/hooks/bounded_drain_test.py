"""Tests for BoundedDrain's line-by-line streaming of a child's pipe to the log."""

import io

from midrun.hooks.bounded_drain import BoundedDrain


class _RecordingLog:
    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, message, *args) -> None:
        self.infos.append(message.format(*args))


def _drain(data: bytes, max_bytes: int) -> tuple[list[str], BoundedDrain]:
    log = _RecordingLog()
    drain = BoundedDrain(io.BytesIO(data), tag="t", max_bytes=max_bytes, log=log)
    drain.start()
    assert drain.join(timeout=5.0)
    return log.infos, drain


def test_a_line_split_across_reads_is_one_record() -> None:
    # The first read ends inside "é" and inside the first line.
    data = ("a" * 4095 + "é line one\nline two\n").encode()
    infos, _ = _drain(data, max_bytes=1 << 20)
    assert infos == [
        "[hook t] " + "a" * 4095 + "é line one",
        "[hook t] line two",
    ]


def test_each_line_is_its_own_record() -> None:
    infos, _ = _drain(b"one\ntwo\nthree", max_bytes=1024)
    assert infos == ["[hook t] one", "[hook t] two", "[hook t] three"]


def test_a_line_cut_by_the_byte_limit_is_logged_up_to_the_limit() -> None:
    infos, drain = _drain(b"abc\ndefgh\nij\n", max_bytes=7)
    assert infos[:2] == ["[hook t] abc", "[hook t] def"]
    assert len(infos) == 3
    assert "6 more bytes not logged" in infos[2]
    assert drain.truncated
