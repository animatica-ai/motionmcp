# SPDX-License-Identifier: Apache-2.0
"""motionmcp.client.video_source: a video_reference's ``video`` from a file or
a URL. Standard library only (runs in the client-only job)."""

from __future__ import annotations

import base64

import pytest

from motionmcp.client import video_source


@pytest.mark.parametrize("name,media_type", [
    ("clip.mp4", "video/mp4"), ("clip.M4V", "video/mp4"),
    ("clip.mov", "video/quicktime"), ("clip.webm", "video/webm"),
])
def test_file_is_base64_with_its_media_type(tmp_path, name, media_type) -> None:
    raw = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256))
    path = tmp_path / name
    path.write_bytes(raw)
    got = video_source(path)
    assert set(got) == {"data", "media_type"}
    assert got["media_type"] == media_type
    assert base64.b64decode(got["data"], validate=True) == raw


def test_media_type_can_be_given(tmp_path) -> None:
    path = tmp_path / "capture.bin"
    path.write_bytes(b"x" * 10)
    assert video_source(path, media_type="video/webm")["media_type"] == "video/webm"
    with pytest.raises(ValueError):
        video_source(path)                                   # unknown extension
    with pytest.raises(ValueError):
        video_source(path, media_type="video/avi")


def test_url() -> None:
    assert video_source(url="https://cdn.example.com/a.mp4") == {"url": "https://cdn.example.com/a.mp4"}
    with pytest.raises(ValueError):
        video_source(url="http://cdn.example.com/a.mp4")
    with pytest.raises(ValueError):
        video_source(url="https://cdn.example.com/a.mp4", media_type="video/mp4")


def test_exactly_one_source(tmp_path) -> None:
    with pytest.raises(ValueError):
        video_source()
    with pytest.raises(ValueError):
        video_source(tmp_path / "a.mp4", url="https://cdn.example.com/a.mp4")


def test_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        video_source(path)


def test_matches_the_server_schema(tmp_path) -> None:
    pytest.importorskip("pydantic", reason="needs the [server] extras")
    from motionmcp import schemas
    path = tmp_path / "clip.mov"
    path.write_bytes(b"moov" * 100)
    src = schemas.VideoSource.model_validate(video_source(path))
    assert src.num_bytes == 400 and src.decoded() == b"moov" * 100
