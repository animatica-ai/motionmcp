# SPDX-License-Identifier: Apache-2.0
"""The motion_reference segment (MMCP 1.2): schema, request rules, the
capability gate, max_reference_frames, and the response metadata."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from motionmcp import GenerateRequest, ModelSpec, MotionReferenceSegment, build_app
from motionmcp.client.gltf_parser import parse_gltf_samples
from motionmcp.null_backbone import NullBackbone


class ReferenceNull(NullBackbone):
    """A NullBackbone that advertises motion_reference and keeps what it got."""

    def __init__(self, max_reference_frames: int | None = 300, **kw):
        super().__init__(**kw)
        self.max_reference_frames = max_reference_frames
        self.received: list[GenerateRequest] = []

    def capabilities(self) -> ModelSpec:
        spec = super().capabilities()
        spec.supported_segments = ["text", "unconditioned", "motion_reference"]
        spec.limits.max_reference_frames = self.max_reference_frames
        return spec

    async def generate(self, request: GenerateRequest):
        self.received.append(request)
        return await super().generate(request)


def _joints(client: TestClient) -> tuple[dict, list[str]]:
    skel = client.get("/capabilities").json()["models"][0]["canonical_skeleton"]
    return skel, [j["name"] for j in skel["joints"]]


def _segment(names: list[str], t: int = 10, **kw) -> dict:
    return {
        "type": "motion_reference",
        "joint_names": names,
        "rotations": [[[0.0, 0.0, 0.0, 1.0]] * len(names)] * t,
        "root_positions": [[0.0, 0.9, 0.01 * i] for i in range(t)],
        "fps": 30,
        **kw,
    }


def _request(skel: dict, *segments: dict, **kw) -> dict:
    return {"protocol_version": "1.2", "model": "null", "skeleton": skel,
            "segments": list(segments), **kw}


@pytest.fixture
def backbone() -> ReferenceNull:
    return ReferenceNull()


@pytest.fixture
def client(backbone: ReferenceNull) -> TestClient:
    return TestClient(build_app(backbone))


# ---- schema -------------------------------------------------------------

def test_segment_parses_with_defaults() -> None:
    seg = MotionReferenceSegment.model_validate(_segment(["a", "b"], t=5))
    assert seg.fidelity == 0.0 and seg.duration_frames is None
    assert seg.reference_frames == 5 and seg.output_frames == 5
    seg = MotionReferenceSegment.model_validate(_segment(["a"], t=5, duration_frames=12,
                                                         fidelity=0.05))
    assert seg.output_frames == 12 and seg.fidelity == 0.05


def test_non_unit_quaternions_are_taken_as_sent() -> None:
    raw = _segment(["a"], t=2)
    raw["rotations"] = [[[0.0, 0.0, 0.0, 1.02]], [[0.0, 0.1, 0.0, 0.99]]]
    seg = MotionReferenceSegment.model_validate(raw)
    assert seg.rotations[0][0] == (0.0, 0.0, 0.0, 1.02)


@pytest.mark.parametrize("change", [
    {"rotations": [[[0, 0, 0, 1]]]},                                  # T = 1
    {"root_positions": [[0, 0, 0]] * 3},                              # T mismatch
    {"rotations": [[[0, 0, 0, 1]]] * 4},                              # frame joint count
    {"rotations": [[[0, 0, 0, 1], [0, 0, 1]]] * 4},                   # not a quaternion
    {"joint_names": []},
    {"joint_names": ["a", "a"]},
    {"fidelity": -0.01},
    {"fidelity": 0.21},
    {"duration_frames": 0},
    {"fps": 0},
    {"prompt": "walk"},                                               # extra field
])
def test_segment_rejects(change: dict) -> None:
    raw = {**_segment(["a", "b"], t=4), **change}
    with pytest.raises(ValidationError):
        MotionReferenceSegment.model_validate(raw)


def test_json_round_trip() -> None:
    raw = _request({"joints": [{"name": "a", "parent": None, "rest_translation": [0, 0, 0],
                                "rest_rotation": [0, 0, 0, 1]}]},
                   _segment(["a"], t=3, fidelity=0.05))
    req = GenerateRequest.model_validate(raw)
    again = GenerateRequest.model_validate(json.loads(req.model_dump_json()))
    assert again == req
    seg = again.segments[0]
    assert isinstance(seg, MotionReferenceSegment) and again.motion_reference is seg
    assert seg.duration_frames is None           # left as sent, not filled in
    assert again.total_frames == 3


# ---- request rules ------------------------------------------------------

def test_happy_path_reaches_backbone_untouched(client, backbone) -> None:
    skel, names = _joints(client)
    seg = _segment(names, t=20, duration_frames=15, fidelity=0.05)
    r = client.post("/generate", json=_request(skel, seg, options={"num_samples": 3}))
    assert r.status_code == 200, r.text
    got = backbone.received[-1].motion_reference
    assert got is not None
    assert got.model_dump(mode="json") == {**seg, "fps": 30.0}
    ext = r.json()["extensions"]["MMCP_motion"]
    assert len(ext["samples"]) == 3
    assert all(s["num_frames"] == 15 for s in ext["samples"])
    assert all(s["reference"] == {"fidelity": 0.05} for s in ext["samples"])
    assert [m["reference"] for m in parse_gltf_samples(r.json())] == [{"fidelity": 0.05}] * 3


def test_text_response_has_no_reference(client) -> None:
    skel, _ = _joints(client)
    r = client.post("/generate", json=_request(
        skel, {"type": "text", "prompt": "walk", "duration_frames": 10}))
    assert r.status_code == 200
    assert "reference" not in r.json()["extensions"]["MMCP_motion"]["samples"][0]
    assert parse_gltf_samples(r.json())[0]["reference"] is None


def test_default_duration_is_reference_length(client) -> None:
    skel, names = _joints(client)
    r = client.post("/generate", json=_request(skel, _segment(names, t=12)))
    assert r.json()["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == 12


@pytest.mark.parametrize("extra", [
    "reference", {"type": "text", "prompt": "walk", "duration_frames": 10},
    {"type": "unconditioned", "duration_frames": 10},
])
def test_reference_stands_alone(client, extra) -> None:
    skel, names = _joints(client)
    other = _segment(names) if extra == "reference" else extra
    r = client.post("/generate", json=_request(skel, _segment(names), other))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_reference_with_loop_is_invalid_request(client) -> None:
    skel, names = _joints(client)
    r = client.post("/generate", json=_request(skel, _segment(names), options={"loop": True}))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_reference_joints_must_be_on_the_skeleton(client) -> None:
    skel, names = _joints(client)
    r = client.post("/generate", json=_request(skel, _segment(names[:-1] + ["NotAJoint"])))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_joint"


def test_constraints_frame_range_uses_output_length(client) -> None:
    skel, names = _joints(client)
    pk = {"type": "pose_keyframe", "frame": 14, "joint_rotations": {names[0]: [0, 0, 0, 1]}}
    ok = client.post("/generate", json=_request(
        skel, _segment(names, t=10, duration_frames=15), constraints=[pk]))
    assert ok.status_code == 200, ok.text
    bad = client.post("/generate", json=_request(
        skel, _segment(names, t=10), constraints=[pk]))
    assert bad.json()["error"]["code"] == "frame_out_of_range"


# ---- capabilities and the gate ------------------------------------------

def test_gate_rejects_without_the_capability() -> None:
    c = TestClient(build_app(NullBackbone()))
    skel, names = _joints(c)
    caps = c.get("/capabilities").json()["models"][0]
    assert "motion_reference" not in caps["supported_segments"]
    assert "max_reference_frames" not in caps["limits"]
    r = c.post("/generate", json=_request(skel, _segment(names)))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_segment"


def test_capabilities_advertise_reference(client) -> None:
    m = client.get("/capabilities").json()["models"][0]
    assert "motion_reference" in m["supported_segments"]
    assert m["limits"]["max_reference_frames"] == 300


@pytest.mark.parametrize("t,duration,ok", [
    (300, None, True), (301, None, False), (100, 300, True), (100, 301, False),
])
def test_max_reference_frames(t, duration, ok) -> None:
    b = ReferenceNull(max_reference_frames=300)
    c = TestClient(build_app(b))
    skel, names = _joints(c)
    kw = {} if duration is None else {"duration_frames": duration}
    r = c.post("/generate", json=_request(skel, _segment(names[:2], t=t, **kw)))
    if ok:
        assert r.status_code == 200, r.text
    else:
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_options"
        assert r.json()["error"]["details"]["max_reference_frames"] == 300


def test_no_frame_cap_when_not_advertised() -> None:
    c = TestClient(build_app(ReferenceNull(max_reference_frames=None)))
    skel, names = _joints(c)
    assert "max_reference_frames" not in c.get("/capabilities").json()["models"][0]["limits"]
    r = c.post("/generate", json=_request(skel, _segment(names[:2], t=400)))
    assert r.status_code == 200, r.text


def test_batch_item_with_reference(client) -> None:
    skel, names = _joints(client)
    good = _request(skel, _segment(names, t=8))
    bad = _request(skel, _segment(names), _segment(names))
    r = client.post("/generate", json={"protocol_version": "1.2", "requests": [good, bad]})
    res = r.json()["results"]
    assert "gltf" in res[0]
    assert res[1]["error"]["code"] == "invalid_request"
