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

    def __init__(self, max_reference_frames: int | None = 300, retargets: bool = False,
                 mixing: bool = False, **kw):
        super().__init__(**kw)
        self.max_reference_frames = max_reference_frames
        self.retargets = retargets
        self.mixing = mixing
        self.received: list[GenerateRequest] = []

    def capabilities(self) -> ModelSpec:
        spec = super().capabilities()
        spec.supported_segments = ["text", "unconditioned", "motion_reference"]
        spec.limits.max_reference_frames = self.max_reference_frames
        spec.supports_retargeting = self.retargets
        spec.supports_motion_reference_mixed = self.mixing
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
    seg = _segment(names, t=15, duration_frames=15, fidelity=0.05)
    r = client.post("/generate", json=_request(skel, seg, options={"num_samples": 3}))
    assert r.status_code == 200, r.text
    got = backbone.received[-1].motion_reference
    assert got is not None
    assert got.model_dump(mode="json") == {**seg, "fps": 30.0, "skeleton": None, "seed": None}
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
def test_reference_stands_alone_without_mixing(client, extra) -> None:
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


# ---- fidelity keeps the clip's length ----------------------------------

@pytest.mark.parametrize("t,duration,fidelity,ok", [
    (10, 15, 0.0, True),       # fidelity 0: any output length
    (20, 15, 0.0, True),
    (10, None, 0.05, True),    # default length is the clip's
    (10, 10, 0.05, True),
    (10, 15, 0.05, False),
    (20, 15, 0.05, False),
])
def test_fidelity_needs_the_reference_length(client, t, duration, fidelity, ok) -> None:
    skel, names = _joints(client)
    kw = {} if duration is None else {"duration_frames": duration}
    r = client.post("/generate", json=_request(skel, _segment(names, t=t, fidelity=fidelity, **kw)))
    if ok:
        assert r.status_code == 200, r.text
        assert r.json()["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == (duration or t)
    else:
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "invalid_request"
        assert err["details"]["reference_frames"] == t and err["details"]["duration_frames"] == duration


# ---- a reference on its own skeleton -------------------------------------

_MIXAMO = {"joints": [
    {"name": "mixamorig:Hips", "parent": None, "rest_translation": [0, 1, 0], "rest_rotation": [0, 0, 0, 1]},
    {"name": "mixamorig:Spine", "parent": "mixamorig:Hips", "rest_translation": [0, 0.1, 0],
     "rest_rotation": [0, 0, 0, 1]},
    {"name": "mixamorig:Head", "parent": "mixamorig:Spine", "rest_translation": [0, 0.5, 0],
     "rest_rotation": [0, 0, 0, 1]},
]}


def test_segment_skeleton_parses_and_round_trips() -> None:
    seg = MotionReferenceSegment.model_validate(
        _segment(["mixamorig:Hips", "mixamorig:Head"], t=4, skeleton=_MIXAMO))
    assert [j.name for j in seg.skeleton.joints][0] == "mixamorig:Hips"
    again = MotionReferenceSegment.model_validate(json.loads(seg.model_dump_json()))
    assert again == seg


def test_segment_skeleton_uses_the_skeleton_schema() -> None:
    bad = {"joints": [{"name": "a", "parent": None, "rest_translation": [0, 0, 0],
                       "rest_rotation": [0, 0, 0, 1]},
                      {"name": "b", "parent": None, "rest_translation": [0, 0, 0],
                       "rest_rotation": [0, 0, 0, 1]}]}                 # two roots
    with pytest.raises(ValidationError):
        MotionReferenceSegment.model_validate(_segment(["a"], t=3, skeleton=bad))


def test_reference_from_another_rig_reaches_backbone() -> None:
    b = ReferenceNull(retargets=True)
    c = TestClient(build_app(b))
    skel, _ = _joints(c)
    names = [j["name"] for j in _MIXAMO["joints"]]
    seg = _segment(names, t=12, duration_frames=40, skeleton=_MIXAMO)
    r = c.post("/generate", json=_request(skel, seg))
    assert r.status_code == 200, r.text
    got = b.received[-1].motion_reference
    assert [j.name for j in got.skeleton.joints] == names
    assert got.reference_frames == 12 and got.output_frames == 40
    assert r.json()["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == 40


def test_unknown_joint_checks_the_reference_skeleton() -> None:
    c = TestClient(build_app(ReferenceNull(retargets=True)))
    skel, request_names = _joints(c)
    # Request-skeleton names are unknown on the clip's own rig.
    r = c.post("/generate", json=_request(skel, _segment(request_names[:2], skeleton=_MIXAMO)))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unknown_joint"
    assert "mixamorig:Hips" in json.dumps(err)


def test_reference_skeleton_needs_retargeting_unless_canonical(client) -> None:
    skel, names = _joints(client)
    r = client.post("/generate", json=_request(
        skel, _segment(["mixamorig:Hips"], skeleton=_MIXAMO)))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "retargeting_unsupported"
    # The model's own skeleton, sent explicitly, is fine without retargeting.
    r = client.post("/generate", json=_request(skel, _segment(names, skeleton=skel)))
    assert r.status_code == 200, r.text


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


# ---- mixing: a reference is a prompt like a text segment (1.2) -------------

@pytest.fixture
def mixing() -> ReferenceNull:
    return ReferenceNull(mixing=True, retargets=True)


@pytest.fixture
def mclient(mixing: ReferenceNull) -> TestClient:
    return TestClient(build_app(mixing))


_TEXT = {"type": "text", "prompt": "a person walks", "duration_frames": 20}


def test_capabilities_advertise_mixing(client, mclient) -> None:
    assert client.get("/capabilities").json()["models"][0]["supports_motion_reference_mixed"] is False
    assert mclient.get("/capabilities").json()["models"][0]["supports_motion_reference_mixed"] is True


def test_two_references_make_one_take(mclient, mixing) -> None:
    skel, names = _joints(mclient)
    walk = _segment(names, t=30, duration_frames=40)
    sit = _segment(names[:3], t=12, duration_frames=25, skeleton=_MIXAMO) | {
        "joint_names": ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Head"],
        "rotations": [[[0.0, 0.0, 0.0, 1.0]] * 3] * 12}
    r = mclient.post("/generate", json=_request(skel, walk, sit, options={"num_samples": 2}))
    assert r.status_code == 200, r.text
    got = mixing.received[-1]
    assert [s.type for s in got.segments] == ["motion_reference"] * 2
    assert got.motion_reference is got.segments[0]
    assert got.motion_references == got.segments
    assert got.segments[1].skeleton is not None and got.segments[0].skeleton is None
    assert got.total_frames == 65
    ext = r.json()["extensions"]["MMCP_motion"]
    assert [s["num_frames"] for s in ext["samples"]] == [65, 65]
    assert all(s["reference"] == {"fidelity": 0.0} for s in ext["samples"])


@pytest.mark.parametrize("layout", [
    ("ref", "text"), ("text", "ref"), ("text", "ref", "text"),
    ("ref", "unconditioned", "ref"), ("ref", "ref", "ref"),
])
def test_references_mix_with_text_and_unconditioned(mclient, mixing, layout) -> None:
    skel, names = _joints(mclient)
    make = {"ref": lambda i: _segment(names, t=10, duration_frames=15 + i, seed=100 + i),
            "text": lambda i: {**_TEXT, "duration_frames": 20 + i, "seed": 7},
            "unconditioned": lambda i: {"type": "unconditioned", "duration_frames": 10 + i}}
    segs = [make[k](i) for i, k in enumerate(layout)]
    r = mclient.post("/generate", json=_request(skel, *segs))
    assert r.status_code == 200, r.text
    got = mixing.received[-1]
    assert [s.type for s in got.segments] == [
        {"ref": "motion_reference"}.get(k, k) for k in layout]
    lengths = [s["duration_frames"] for s in segs]
    assert [(s.output_frames if s.type == "motion_reference" else s.duration_frames)
            for s in got.segments] == lengths
    assert got.total_frames == sum(lengths)
    assert [s.seed for s in got.motion_references] == [
        100 + i for i, k in enumerate(layout) if k == "ref"]
    assert r.json()["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == sum(lengths)


def test_mixed_reference_needs_its_duration(mclient) -> None:
    skel, names = _joints(mclient)
    r = mclient.post("/generate", json=_request(skel, _TEXT, _segment(names, t=10)))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_request" and err["details"]["segment"] == 1


@pytest.mark.parametrize("fidelity,ok", [(0.0, True), (0.05, False)])
def test_fidelity_only_for_a_lone_reference(mclient, fidelity, ok) -> None:
    skel, names = _joints(mclient)
    ref = _segment(names, t=10, duration_frames=10, fidelity=fidelity)
    r = mclient.post("/generate", json=_request(skel, ref, _segment(names, t=10, duration_frames=10)))
    if ok:
        assert r.status_code == 200, r.text
    else:
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "invalid_request" and err["details"]["segment"] == 0
    # alone it keeps working
    r = mclient.post("/generate", json=_request(skel, ref))
    assert r.status_code == 200, r.text
    assert r.json()["extensions"]["MMCP_motion"]["samples"][0]["reference"] == {"fidelity": fidelity}


def test_mixed_references_refuse_loop_and_pose(mclient) -> None:
    skel, names = _joints(mclient)
    ref = _segment(names, t=10, duration_frames=10)
    r = mclient.post("/generate", json=_request(skel, ref, _TEXT, options={"loop": True}))
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
    r = mclient.post("/generate", json=_request(skel, ref, options={"loop": True}))
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"

    class PoseToo(ReferenceNull):
        def capabilities(self) -> ModelSpec:
            spec = super().capabilities()
            spec.supported_segments = [*spec.supported_segments, "pose"]
            return spec

    c = TestClient(build_app(PoseToo(mixing=True)))
    r = c.post("/generate", json=_request(skel, ref, {"type": "pose", "prompt": "a person sits"}))
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def test_each_mixed_reference_is_checked(mclient) -> None:
    skel, names = _joints(mclient)
    good = _segment(names, t=10, duration_frames=10)
    bad_joint = _segment(names[:-1] + ["NotAJoint"], t=10, duration_frames=10)
    r = mclient.post("/generate", json=_request(skel, good, bad_joint))
    assert r.json()["error"]["code"] == "unknown_joint"
    too_long = _segment(names[:2], t=301, duration_frames=10)
    r = mclient.post("/generate", json=_request(skel, good, too_long))
    err = r.json()["error"]
    assert err["code"] == "invalid_options" and err["details"]["segment"] == 1
    # constraints span the whole take
    pk = {"type": "pose_keyframe", "frame": 19, "joint_rotations": {names[0]: [0, 0, 0, 1]}}
    r = mclient.post("/generate", json=_request(skel, good, dict(good), constraints=[pk]))
    assert r.status_code == 200, r.text
    pk["frame"] = 20
    r = mclient.post("/generate", json=_request(skel, good, dict(good), constraints=[pk]))
    assert r.json()["error"]["code"] == "frame_out_of_range"


def test_mixed_batch_item(mclient) -> None:
    skel, names = _joints(mclient)
    mixed = _request(skel, _segment(names, t=8, duration_frames=8), _TEXT)
    r = mclient.post("/generate", json={"protocol_version": "1.2", "requests": [mixed, mixed]})
    assert all("gltf" in x for x in r.json()["results"])


def test_segment_seed_parses() -> None:
    seg = MotionReferenceSegment.model_validate(_segment(["a"], t=3, seed=5))
    assert seg.seed == 5
    assert MotionReferenceSegment.model_validate(_segment(["a"], t=3)).seed is None
