# SPDX-License-Identifier: Apache-2.0
"""The video_reference segment (MMCP 1.2): a prompt given as a video.

Its payload has its own checks (one source, https URL or base64 data with a
media type, the trim, max_video_bytes / max_video_seconds); everything else is
a text segment's, so every request rule is tested here against the same
request with a text segment in its place (as in test_motion_reference)."""

from __future__ import annotations

import base64
import json

import pytest

pytest.importorskip("fastapi", reason="needs the [server] extras")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from motionmcp import (  # noqa: E402
    GenerateRequest,
    ModelSpec,
    VideoReferenceSegment,
    VideoSource,
    build_app,
)
from motionmcp.null_backbone import NullBackbone  # noqa: E402

_MP4 = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 4      # 1036 bytes; never decoded
_DATA = base64.b64encode(_MP4).decode("ascii")
_URL = "https://cdn.example.com/clips/walk.mp4"


class VideoNull(NullBackbone):
    """A NullBackbone that advertises video_reference (and motion_reference)
    and keeps what it got."""

    def __init__(self, max_video_bytes: int | None = 50_000_000,
                 max_video_seconds: float | None = 30.0,
                 loops: bool = False, poses: bool = False, **kw):
        super().__init__(**kw)
        self.max_video_bytes = max_video_bytes
        self.max_video_seconds = max_video_seconds
        self.loops = loops
        self.poses = poses
        self.received: list[GenerateRequest] = []

    def capabilities(self) -> ModelSpec:
        spec = super().capabilities()
        spec.supported_segments = ["text", "unconditioned", "motion_reference",
                                   "video_reference", *(["pose"] if self.poses else [])]
        spec.limits.max_video_bytes = self.max_video_bytes
        spec.limits.max_video_seconds = self.max_video_seconds
        spec.supports_loop = self.loops
        return spec

    async def generate(self, request: GenerateRequest):
        self.received.append(request)
        return await super().generate(request)


def _skeleton(client: TestClient) -> dict:
    return client.get("/capabilities").json()["models"][0]["canonical_skeleton"]


def _url_segment(t: int = 30, **kw) -> dict:
    return {"type": "video_reference", "duration_frames": t, "video": {"url": _URL}, **kw}


def _data_segment(t: int = 30, data: str = _DATA, **kw) -> dict:
    return {"type": "video_reference", "duration_frames": t,
            "video": {"data": data, "media_type": "video/mp4"}, **kw}


def _request(skel: dict, *segments: dict, **kw) -> dict:
    return {"protocol_version": "1.2", "model": "null", "skeleton": skel,
            "segments": list(segments), **kw}


@pytest.fixture
def backbone() -> VideoNull:
    return VideoNull()


@pytest.fixture
def client(backbone: VideoNull) -> TestClient:
    return TestClient(build_app(backbone))


# ---- schema -------------------------------------------------------------

def test_url_segment_parses() -> None:
    seg = VideoReferenceSegment.model_validate(_url_segment(t=48))
    assert seg.duration_frames == 48 and seg.video.url == _URL
    assert seg.video.data is None and seg.video.num_bytes is None
    assert (seg.start_s, seg.end_s, seg.fps, seg.seed) == (None,) * 4
    assert seg.trimmed_seconds is None


def test_data_segment_parses_and_is_not_decoded_as_video() -> None:
    seg = VideoReferenceSegment.model_validate(
        _data_segment(start_s=1.5, end_s=4.0, fps=29.97, seed=9))
    assert seg.video.media_type == "video/mp4"
    assert seg.video.num_bytes == len(_MP4)
    assert seg.video.decoded() == _MP4
    assert seg.trimmed_seconds == pytest.approx(2.5)
    assert (seg.fps, seg.seed) == (29.97, 9)


@pytest.mark.parametrize("media_type", ["video/mp4", "video/quicktime", "video/webm"])
def test_every_media_type(media_type) -> None:
    raw = {"data": _DATA, "media_type": media_type}
    assert VideoSource.model_validate(raw).media_type == media_type


def test_url_scheme_is_case_insensitive() -> None:
    assert VideoSource.model_validate({"url": "HTTPS://cdn.example.com:8443/a.mp4"}).url


def test_trimmed_seconds_from_end_alone() -> None:
    assert VideoReferenceSegment.model_validate(_url_segment(end_s=3)).trimmed_seconds == 3
    assert VideoReferenceSegment.model_validate(_url_segment(start_s=3)).trimmed_seconds is None


def test_url_video_cannot_be_decoded_by_the_sdk() -> None:
    with pytest.raises(ValueError):
        VideoSource.model_validate({"url": _URL}).decoded()


def test_duration_frames_is_required_as_for_text() -> None:
    raw = _url_segment()
    del raw["duration_frames"]
    with pytest.raises(ValidationError):
        VideoReferenceSegment.model_validate(raw)


@pytest.mark.parametrize("video", [
    {},                                                              # no source
    {"url": _URL, "data": _DATA, "media_type": "video/mp4"},         # both
    {"url": "http://cdn.example.com/walk.mp4"},                      # not https
    {"url": "ftp://cdn.example.com/walk.mp4"},
    {"url": "https:///walk.mp4"},                                    # no host
    {"url": "walk.mp4"},
    {"url": ""},
    {"url": "https://user:pw@cdn.example.com/walk.mp4"},             # user info
    {"url": "https://cdn.example.com/walk .mp4"},                    # whitespace
    {"url": "https://cdn.example.com/walk.mp4\r\nX: y"},            # control chars
    {"url": "https://cdn.example.com:99999/walk.mp4"},               # invalid port
    {"url": "https://cdn.example.com:0/walk.mp4"},
    {"url": "https://cdn.example.com:x/walk.mp4"},
    {"url": "https://cdn.example.com/" + "a" * 5000},                # too long
    {"url": _URL, "media_type": "video/mp4"},                        # media_type with url
    {"data": _DATA},                                                 # no media_type
    {"data": _DATA, "media_type": "video/avi"},                      # not a listed type
    {"data": _DATA, "media_type": "image/gif"},
    {"data": "not base64!", "media_type": "video/mp4"},
    {"data": _DATA[:-2], "media_type": "video/mp4"},                 # bad padding
    {"data": "", "media_type": "video/mp4"},
    {"data": "====", "media_type": "video/mp4"},                     # decodes to nothing
    {"url": _URL, "headers": {"Authorization": "x"}},                # extra field
])
def test_video_source_rejects(video: dict) -> None:
    with pytest.raises(ValidationError):
        VideoReferenceSegment.model_validate({**_url_segment(), "video": video})


@pytest.mark.parametrize("change", [
    {"duration_frames": 0},
    {"start_s": -0.1},
    {"end_s": 0},
    {"start_s": 2.0, "end_s": 2.0},                                  # start == end
    {"start_s": 3.0, "end_s": 2.0},                                  # start > end
    {"start_s": float("inf")},
    {"end_s": float("nan")},
    {"person": 0},                                                   # not in 1.2: the most prominent
    {"fps": 0},
    {"fps": -24},
    {"prompt": "walk"},                                              # extra field
    {"video": None},
])
def test_segment_rejects(change: dict) -> None:
    with pytest.raises(ValidationError):
        VideoReferenceSegment.model_validate({**_url_segment(), **change})


def test_zero_start_is_a_trim_from_the_beginning() -> None:
    seg = VideoReferenceSegment.model_validate(_url_segment(start_s=0, end_s=5))
    assert seg.trimmed_seconds == 5


@pytest.mark.parametrize("seg", [_url_segment(t=7, end_s=2.5),
                                 _data_segment(t=7, start_s=0.5, fps=24, seed=3)])
def test_json_round_trip(seg) -> None:
    raw = _request({"joints": [{"name": "a", "parent": None, "rest_translation": [0, 0, 0],
                                "rest_rotation": [0, 0, 0, 1]}]},
                   seg, {"type": "text", "prompt": "a person sits", "duration_frames": 5})
    req = GenerateRequest.model_validate(raw)
    again = GenerateRequest.model_validate(json.loads(req.model_dump_json()))
    assert again == req
    got = again.segments[0]
    assert isinstance(got, VideoReferenceSegment) and again.video_reference is got
    assert again.video_references == [got]
    assert got.video.num_bytes == req.segments[0].video.num_bytes
    assert again.total_frames == 12, "the segments' duration_frames, not the video's length"


# ---- it reaches the backbone as sent ------------------------------------

@pytest.mark.parametrize("seg", [_url_segment(t=40, seed=3, start_s=1, end_s=3),
                                 _data_segment(t=40, fps=30)])
def test_happy_path_reaches_backbone_parsed(client, backbone, seg) -> None:
    skel = _skeleton(client)
    r = client.post("/generate", json=_request(skel, seg, options={"num_samples": 3}))
    assert r.status_code == 200, r.text
    got = backbone.received[-1].video_reference
    sent = {"start_s": None, "end_s": None, "fps": None, "seed": None, **seg}
    sent["video"] = {"url": None, "data": None, "media_type": None, **seg["video"]}
    assert got.model_dump(mode="json") == json.loads(json.dumps(sent))
    ext = r.json()["extensions"]["MMCP_motion"]
    assert [s["num_frames"] for s in ext["samples"]] == [40] * 3


def test_capabilities_advertise_video_reference(client) -> None:
    m = client.get("/capabilities").json()["models"][0]
    assert "video_reference" in m["supported_segments"]
    assert m["limits"]["max_video_bytes"] == 50_000_000
    assert m["limits"]["max_video_seconds"] == 30.0


def test_gate_rejects_without_the_capability() -> None:
    c = TestClient(build_app(NullBackbone()))
    caps = c.get("/capabilities").json()["models"][0]
    assert "video_reference" not in caps["supported_segments"]
    assert "max_video_bytes" not in caps["limits"]
    assert "max_video_seconds" not in caps["limits"]
    for seg in (_url_segment(), _data_segment()):
        r = c.post("/generate", json=_request(_skeleton(c), seg))
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "unsupported_segment"
        assert "video_reference" not in err["details"]["supported_segments"]


def test_gate_is_per_type() -> None:
    """A model that takes motion references but not video references."""
    class MotionOnly(VideoNull):
        def capabilities(self) -> ModelSpec:
            spec = super().capabilities()
            spec.supported_segments = ["text", "unconditioned", "motion_reference"]
            return spec

    c = TestClient(build_app(MotionOnly()))
    r = c.post("/generate", json=_request(_skeleton(c), _url_segment()))
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_segment"


def test_schema_errors_are_422(client) -> None:
    skel = _skeleton(client)
    bad = {**_url_segment(), "video": {"url": "http://insecure.example.com/a.mp4"}}
    r = client.post("/generate", json=_request(skel, bad))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "schema_validation"


# ---- the text segment's rules -------------------------------------------

def _as_text(seg: dict) -> dict:
    """The text segment in a reference's place: same length, same seed."""
    return {"type": "text", "prompt": "a person walks", "duration_frames": seg["duration_frames"],
            **({"seed": seg["seed"]} if "seed" in seg else {})}


def _outcome(c: TestClient, body: dict):
    r = c.post("/generate", json=body)
    if r.status_code == 200:
        return 200, [s["num_frames"] for s in r.json()["extensions"]["MMCP_motion"]["samples"]]
    return r.status_code, r.json()["error"]["code"]


_REF = "ref"


def _layouts(names):
    """(segments with _REF placeholders, extra request fields) -- requests a
    text segment makes or breaks in every way the SDK checks."""
    pk = lambda f: {"type": "pose_keyframe", "frame": f, "joint_rotations": {names[0]: [0, 0, 0, 1]}}  # noqa: E731
    return [
        ([_REF], {}),
        ([_REF, _REF], {}),
        ([_REF, "text"], {}),
        (["text", _REF, "text"], {}),
        ([_REF, "unconditioned", _REF], {}),
        ([_REF, "motion_reference"], {}),
        (["motion_reference", _REF, "text"], {}),
        ([_REF], {"options": {"num_samples": 4, "seed": 1}}),
        ([_REF], {"options": {"loop": True}}),
        ([_REF, _REF], {"options": {"loop": True}}),
        ([_REF, "pose"], {}),
        ([_REF], {"options": {"guidance": {"type": "separated", "weight": [2.0, 2.0]}}}),
        ([_REF], {"options": {"transition_frames": 8}}),
        ([_REF, _REF], {"constraints": [pk(39)]}),
        ([_REF, _REF], {"constraints": [pk(40)]}),                 # frame_out_of_range
        ([_REF], {"options": {"num_samples": 99}}),                # 422 schema
        ([_REF], {"options": {"num_samples": 16}}),                # the most samples
        ([_REF] * 50, {}),                                         # max_duration_seconds
    ]


@pytest.mark.parametrize("loops", [False, True])
@pytest.mark.parametrize("i", range(18))
def test_a_video_reference_is_validated_like_a_text_segment(i, loops) -> None:
    b = VideoNull(loops=loops, poses=True)
    c = TestClient(build_app(b))
    skel = _skeleton(c)
    names = [j["name"] for j in skel["joints"]]
    segments, extra = _layouts(names)[i]
    motion = {"type": "motion_reference", "duration_frames": 11, "joint_names": names[:1],
              "rotations": [[[0, 0, 0, 1]]] * 4, "root_positions": [[0, 0.9, 0]] * 4, "fps": 30}
    others = {"text": {"type": "text", "prompt": "a person sits", "duration_frames": 17},
              "unconditioned": {"type": "unconditioned", "duration_frames": 9},
              "pose": {"type": "pose", "prompt": "a person sits"},
              "motion_reference": motion}
    refs = [(_url_segment if k % 2 else _data_segment)(t=20, seed=10 + k)
            for k in range(len(segments))]
    with_refs = [refs[k] if s == _REF else others[s] for k, s in enumerate(segments)]
    with_text = [_as_text(refs[k]) if s == _REF else others[s] for k, s in enumerate(segments)]
    got = _outcome(c, _request(skel, *with_refs, **extra))
    want = _outcome(c, _request(skel, *with_text, **extra))
    assert got == want


def test_mixed_segments_reach_the_backbone_in_order(backbone, client) -> None:
    skel = _skeleton(client)
    names = [j["name"] for j in skel["joints"]]
    segs = [{"type": "text", "prompt": "a person walks forward", "duration_frames": 30, "seed": 1},
            {"type": "motion_reference", "duration_frames": 20, "joint_names": names[:1],
             "rotations": [[[0, 0, 0, 1]]] * 5, "root_positions": [[0, 0.9, 0]] * 5,
             "fps": 30, "seed": 2},
            _url_segment(t=25, seed=3, start_s=0.5, end_s=2.5),
            _data_segment(t=15, seed=4)]
    r = client.post("/generate", json=_request(skel, *segs))
    assert r.status_code == 200, r.text
    got = backbone.received[-1]
    assert [s.type for s in got.segments] == ["text", "motion_reference", "video_reference",
                                              "video_reference"]
    assert got.video_references == [got.segments[2], got.segments[3]]
    assert got.video_reference is got.segments[2]
    assert [s.seed for s in got.segments] == [1, 2, 3, 4]
    assert got.total_frames == 90


# ---- the payload's own limits -------------------------------------------

@pytest.mark.parametrize("cap,ok", [(len(_MP4), True), (len(_MP4) - 1, False)])
def test_max_video_bytes_caps_inline_data(cap, ok) -> None:
    c = TestClient(build_app(VideoNull(max_video_bytes=cap)))
    r = c.post("/generate", json=_request(_skeleton(c), _url_segment(), _data_segment()))
    if ok:
        assert r.status_code == 200, r.text
    else:
        assert r.status_code == 413
        err = r.json()["error"]
        assert err["code"] == "payload_too_large"
        assert err["details"] == {"max_video_bytes": cap, "video_bytes": len(_MP4), "segment": 1}


def test_max_video_bytes_cannot_see_a_url() -> None:
    c = TestClient(build_app(VideoNull(max_video_bytes=1)))
    r = c.post("/generate", json=_request(_skeleton(c), _url_segment()))
    assert r.status_code == 200, "the server checks a URL's size when it fetches it"


@pytest.mark.parametrize("trim,ok", [
    ({"end_s": 10.0}, True),
    ({"start_s": 5.0, "end_s": 15.0}, True),
    ({"end_s": 10.5}, False),
    ({"start_s": 1.0, "end_s": 11.5}, False),
    ({"start_s": 100.0}, True),        # length unknown to the SDK: the server's to check
    ({}, True),
])
def test_max_video_seconds_caps_the_trimmed_length(trim, ok) -> None:
    c = TestClient(build_app(VideoNull(max_video_seconds=10.0)))
    r = c.post("/generate", json=_request(_skeleton(c), _url_segment(**trim)))
    if ok:
        assert r.status_code == 200, r.text
    else:
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "invalid_options"
        assert err["details"]["max_video_seconds"] == 10.0
        assert err["details"]["segment"] == 0
        assert err["details"]["video_seconds"] == pytest.approx(
            trim["end_s"] - trim.get("start_s", 0.0))


def test_no_video_caps_when_not_advertised() -> None:
    c = TestClient(build_app(VideoNull(max_video_bytes=None, max_video_seconds=None)))
    limits = c.get("/capabilities").json()["models"][0]["limits"]
    assert "max_video_bytes" not in limits and "max_video_seconds" not in limits
    r = c.post("/generate", json=_request(_skeleton(c), _data_segment(end_s=3600)))
    assert r.status_code == 200, r.text


def test_the_output_length_is_duration_frames_not_the_videos() -> None:
    c = TestClient(build_app(VideoNull(max_video_seconds=2.0)))
    r = c.post("/generate", json=_request(_skeleton(c), _url_segment(t=200, end_s=2.0)))
    assert r.status_code == 200, r.text
    assert r.json()["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == 200


# ---- batches --------------------------------------------------------------

def test_batch_items_with_video_references() -> None:
    c = TestClient(build_app(VideoNull(max_video_bytes=len(_MP4) - 1)))
    skel = _skeleton(c)
    good = _request(skel, _url_segment(t=8), {"type": "text", "prompt": "sit", "duration_frames": 8})
    bad = _request(skel, _data_segment(t=8))
    r = c.post("/generate", json={"protocol_version": "1.2", "requests": [good, bad, good]})
    res = r.json()["results"]
    assert "gltf" in res[0] and "gltf" in res[2]
    assert res[1]["error"]["code"] == "payload_too_large"
