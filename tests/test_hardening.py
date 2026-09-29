# SPDX-License-Identifier: Apache-2.0
"""The 1.2 review fixes: body size before parsing, small error envelopes,
finite numbers, skeleton errors as invalid_skeleton, the root joint in a
motion_reference, the segment gate before payloads, and /capabilities
unchanged for a backbone that takes no references."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the [server] extras")

from fastapi.testclient import TestClient  # noqa: E402

from motionmcp import ModelSpec, PROTOCOL_VERSION, VideoSource, build_app  # noqa: E402
from motionmcp.null_backbone import NullBackbone  # noqa: E402


class RefNull(NullBackbone):
    def __init__(self, segments=("motion_reference", "video_reference"),
                 max_request_bytes=None, retargets=True, **kw):
        super().__init__(**kw)
        self.segments, self.max_request_bytes, self.retargets = segments, max_request_bytes, retargets

    def capabilities(self) -> ModelSpec:
        spec = super().capabilities()
        spec.supported_segments = ["text", "unconditioned", *self.segments]
        spec.supports_retargeting = self.retargets
        if self.max_request_bytes is not None:
            spec.limits.max_request_bytes = self.max_request_bytes
        return spec


def _client(**kw) -> TestClient:
    return TestClient(build_app(RefNull(**kw)))


def _skel(c: TestClient) -> dict:
    return c.get("/capabilities").json()["models"][0]["canonical_skeleton"]


def _ref(names, t=4, **kw) -> dict:
    return {"type": "motion_reference", "duration_frames": 10, "joint_names": names,
            "rotations": [[[0, 0, 0, 1]] * len(names)] * t,
            "root_positions": [[0, 0.9, 0]] * t, "fps": 30, **kw}


def _body(skel, *segments) -> dict:
    return {"protocol_version": "1.2", "model": "null", "skeleton": skel, "segments": list(segments)}


def _post_raw(c: TestClient, text: str):
    return c.post("/generate", content=text.encode(), headers={"Content-Type": "application/json"})


# ---- NaN / Infinity -------------------------------------------------------

@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("where", ["fps", "rotation", "root"])
def test_motion_reference_numbers_must_be_finite(token, where) -> None:
    c = _client()
    skel = _skel(c)
    names = [j["name"] for j in skel["joints"]]
    seg = _ref(names)
    marker = 12345.678
    if where == "fps":
        seg["fps"] = marker
    elif where == "rotation":
        seg["rotations"] = [[[0, 0, 0, marker]] + [[0, 0, 0, 1]] * (len(names) - 1)] * 4
    else:
        seg["root_positions"] = [[0, marker, 0]] * 4
    text = json.dumps(_body(skel, seg)).replace(str(marker), token)
    assert token in text
    r = _post_raw(c, text)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "schema_validation"


@pytest.mark.parametrize("field", ["start_s", "end_s", "fps"])
def test_video_reference_numbers_must_be_finite(field) -> None:
    c = _client()
    seg = {"type": "video_reference", "duration_frames": 10,
           "video": {"url": "https://cdn.example.com/a.mp4"}, field: 12345.678}
    text = json.dumps(_body(_skel(c), seg)).replace("12345.678", "NaN")
    assert _post_raw(c, text).status_code == 422


# ---- the error envelope stays small -------------------------------------------

def test_a_huge_bad_video_data_gives_a_small_error() -> None:
    c = _client(max_request_bytes=8_000_000)
    seg = {"type": "video_reference", "duration_frames": 10,
           "video": {"data": "!" * 4_000_000, "media_type": "video/mp4"}}
    r = c.post("/generate", json=_body(_skel(c), seg))
    assert r.status_code == 422
    assert len(r.content) < 4_000, "the offending input is not echoed back"
    assert "!!!!" not in r.text


# ---- max_request_bytes before parsing ----------------------------------------

def test_body_over_max_request_bytes_is_413_from_content_length() -> None:
    c = _client(max_request_bytes=10_000)
    body = json.dumps(_body(_skel(c), {"type": "text", "prompt": "x" * 20_000,
                                       "duration_frames": 10}))
    r = _post_raw(c, body)
    assert r.status_code == 413
    err = r.json()["error"]
    assert err["code"] == "payload_too_large"
    assert err["details"] == {"max_request_bytes": 10_000, "request_bytes": len(body)}


def test_a_streamed_body_is_cut_off_at_the_cap() -> None:
    """No Content-Length (chunked): the server stops reading past the cap
    instead of buffering the whole body. Driven over raw ASGI, since the test
    client buffers a streamed body before the app sees it."""
    import asyncio

    app = build_app(RefNull(max_request_bytes=1_000_000))
    pulled, sent = [], []

    async def receive():
        pulled.append(1)
        if len(pulled) > 200:                       # 200 MB if it were all read
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": b"x" * 1_000_000, "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/generate", "raw_path": b"/generate",
             "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("c", 1),
             "headers": [(b"content-type", b"application/json"),
                         (b"transfer-encoding", b"chunked")]}
    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert start["status"] == 413
    assert json.loads(body)["error"]["code"] == "payload_too_large"
    assert len(pulled) <= 3, f"read {len(pulled)} MB for a 1 MB cap"


def test_each_model_s_own_cap_applies() -> None:
    small = RefNull(model_id="small", max_request_bytes=1_000)
    big = RefNull(model_id="big", max_request_bytes=100_000)
    c = TestClient(build_app([small, big]))
    skel = _skel(c)
    seg = {"type": "text", "prompt": "x" * 900, "duration_frames": 10}
    body = {**_body(skel, seg)}
    size = len(json.dumps({**body, "model": "big"}))
    assert 1_000 < size < 100_000
    assert c.post("/generate", json={**body, "model": "big"}).status_code == 200
    r = c.post("/generate", json={**body, "model": "small"})
    assert r.status_code == 413 and r.json()["error"]["details"]["max_request_bytes"] == 1_000


# ---- base64 size without decoding ------------------------------------------------

def test_num_bytes_is_arithmetic_not_a_decode(monkeypatch) -> None:
    raw = bytes(range(256)) * 10 + b"ab"
    data = base64.b64encode(raw).decode()
    import motionmcp.schemas as schemas
    monkeypatch.setattr(schemas.base64, "b64decode",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded")))
    src = VideoSource.model_validate({"data": data, "media_type": "video/mp4"})
    assert src.num_bytes == len(raw)
    monkeypatch.undo()
    assert src.decoded() == raw


@pytest.mark.parametrize("data", ["QUJD\nREVG", "QUJD REVG", "QUJ", "QUJD=REVG", "QU==QUJD",
                                  "QUJD-_==", "QQ===", "QUJDRA"])
def test_base64_must_be_standard_padded_one_line(data) -> None:
    with pytest.raises(ValueError):
        VideoSource.model_validate({"data": data, "media_type": "video/mp4"})


# ---- skeletons: 400 invalid_skeleton ---------------------------------------------

_TWO_ROOTS = {"joints": [
    {"name": "a", "parent": None, "rest_translation": [0, 0, 0], "rest_rotation": [0, 0, 0, 1]},
    {"name": "b", "parent": None, "rest_translation": [0, 0, 0], "rest_rotation": [0, 0, 0, 1]}]}
_LATE_PARENT = {"joints": [
    {"name": "b", "parent": "a", "rest_translation": [0, 0, 0], "rest_rotation": [0, 0, 0, 1]},
    {"name": "a", "parent": None, "rest_translation": [0, 0, 0], "rest_rotation": [0, 0, 0, 1]}]}


@pytest.mark.parametrize("bad", [_TWO_ROOTS, _LATE_PARENT])
def test_request_skeleton_topology_is_invalid_skeleton(bad) -> None:
    c = _client()
    r = c.post("/generate", json=_body(bad, {"type": "text", "prompt": "walk", "duration_frames": 5}))
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_skeleton"


def test_segment_skeleton_topology_is_invalid_skeleton() -> None:
    c = _client()
    r = c.post("/generate", json=_body(_skel(c), _ref(["a"], skeleton=_TWO_ROOTS)))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_skeleton"
    assert err["details"]["errors"][0]["loc"][:3] == ["segments", 0, "motion_reference"]


def test_a_malformed_joint_is_still_schema_validation() -> None:
    c = _client()
    skel = _skel(c)
    skel["joints"][0]["rest_rotation"] = [0, 0, 1]
    r = c.post("/generate", json=_body(skel, {"type": "text", "prompt": "walk", "duration_frames": 5}))
    assert r.status_code == 422


# ---- the root joint ---------------------------------------------------------------

def test_a_motion_reference_needs_its_skeleton_s_root() -> None:
    c = _client()
    skel = _skel(c)
    root = skel["joints"][0]["name"]
    names = [j["name"] for j in skel["joints"] if j["parent"] is not None]
    r = c.post("/generate", json=_body(skel, _ref(names)))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["details"] == {"root_joint": root, "segment": 0}


def test_the_root_is_the_segment_skeleton_s_when_it_has_one() -> None:
    c = _client()
    own = {"joints": [
        {"name": "pelvis", "parent": None, "rest_translation": [0, 1, 0], "rest_rotation": [0, 0, 0, 1]},
        {"name": "chest", "parent": "pelvis", "rest_translation": [0, .3, 0], "rest_rotation": [0, 0, 0, 1]}]}
    skel = _skel(c)
    assert c.post("/generate", json=_body(skel, _ref(["chest"], skeleton=own))).json()[
        "error"]["details"]["root_joint"] == "pelvis"
    assert c.post("/generate", json=_body(skel, _ref(["pelvis", "chest"], skeleton=own))).status_code == 200


# ---- the segment gate comes first ------------------------------------------------------

def test_unsupported_segment_before_its_payload_is_parsed() -> None:
    c = _client(segments=())
    seg = {"type": "video_reference", "duration_frames": 10,
           "video": {"data": "not base64 at all", "media_type": "video/mp4"}}
    r = c.post("/generate", json=_body(_skel(c), seg))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_segment"


def test_an_unknown_segment_type_is_unsupported_segment() -> None:
    c = _client()
    r = c.post("/generate", json=_body(_skel(c), {"type": "audio", "duration_frames": 10}))
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_segment"


# ---- /capabilities for a backbone that takes no references ---------------------------------

def test_capabilities_unchanged_for_an_old_backbone() -> None:
    before = json.loads((Path(__file__).parent / "capabilities_0_9_null.json").read_text())
    now = TestClient(build_app(NullBackbone())).get("/capabilities").json()
    assert now["protocol_version"] == PROTOCOL_VERSION == "1.2"
    assert now["models"][0].pop("supports_motion_reference_mixed") is False
    before.pop("protocol_version")
    now.pop("protocol_version")
    assert now == before


def test_mixed_flag_follows_supported_segments() -> None:
    class Claims(NullBackbone):
        def capabilities(self):
            spec = super().capabilities()
            spec.supports_motion_reference_mixed = True       # without the segment type
            return spec
    m = TestClient(build_app(Claims())).get("/capabilities").json()["models"][0]
    assert m["supports_motion_reference_mixed"] is False
    m = _client().get("/capabilities").json()["models"][0]
    assert m["supports_motion_reference_mixed"] is True
