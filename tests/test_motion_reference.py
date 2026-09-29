# SPDX-License-Identifier: Apache-2.0
"""The motion_reference segment (MMCP 1.2): a prompt given as a motion.

Its payload has its own checks (shapes, joints on its skeleton,
max_reference_frames); everything else is a text segment's, so every request
rule is tested here against the same request with a text segment in its place."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="needs the [server] extras")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from motionmcp import GenerateRequest, ModelSpec, MotionReferenceSegment, build_app  # noqa: E402
from motionmcp.schemas import TextSegment  # noqa: E402
from motionmcp.null_backbone import NullBackbone  # noqa: E402


class ReferenceNull(NullBackbone):
    """A NullBackbone that advertises motion_reference and keeps what it got."""

    def __init__(self, max_reference_frames: int | None = 300, retargets: bool = False,
                 loops: bool = False, poses: bool = False, **kw):
        super().__init__(**kw)
        self.max_reference_frames = max_reference_frames
        self.retargets = retargets
        self.loops = loops
        self.poses = poses
        self.received: list[GenerateRequest] = []

    def capabilities(self) -> ModelSpec:
        spec = super().capabilities()
        spec.supported_segments = ["text", "unconditioned", "motion_reference",
                                   *(["pose"] if self.poses else [])]
        spec.limits.max_reference_frames = self.max_reference_frames
        spec.supports_retargeting = self.retargets
        spec.supports_loop = self.loops
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
        "duration_frames": t,
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

def test_segment_parses() -> None:
    seg = MotionReferenceSegment.model_validate(_segment(["a", "b"], t=5, duration_frames=12))
    assert seg.duration_frames == 12 and seg.reference_frames == 5 and seg.seed is None
    assert MotionReferenceSegment.model_validate(_segment(["a"], t=3, seed=5)).seed == 5


def test_duration_frames_is_required_as_for_text() -> None:
    raw = _segment(["a"], t=4)
    del raw["duration_frames"]
    with pytest.raises(ValidationError):
        MotionReferenceSegment.model_validate(raw)
    with pytest.raises(ValidationError):
        TextSegment.model_validate({"type": "text", "prompt": "walk"})


def test_fidelity_is_gone() -> None:
    with pytest.raises(ValidationError):
        MotionReferenceSegment.model_validate(_segment(["a"], t=3, fidelity=0.0))


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
    {"fidelity": 0.0},                                                # removed in 1.2
    {"fps": float("nan")},
    {"fps": float("inf")},
    {"rotations": [[[0, 0, 0, float("nan")], [0, 0, 0, 1]]] * 4},
    {"root_positions": [[0, float("inf"), 0]] * 4},
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
                   _segment(["a"], t=3, duration_frames=7))
    req = GenerateRequest.model_validate(raw)
    again = GenerateRequest.model_validate(json.loads(req.model_dump_json()))
    assert again == req
    seg = again.segments[0]
    assert isinstance(seg, MotionReferenceSegment) and again.motion_reference is seg
    assert again.total_frames == 7, "the segment's duration_frames, not the clip's length"


# ---- it reaches the backbone as sent ------------------------------------

def test_happy_path_reaches_backbone_untouched(client, backbone) -> None:
    skel, names = _joints(client)
    seg = _segment(names, t=15, duration_frames=40, seed=3)
    r = client.post("/generate", json=_request(skel, seg, options={"num_samples": 3}))
    assert r.status_code == 200, r.text
    got = backbone.received[-1].motion_reference
    assert got.model_dump(mode="json") == {**seg, "fps": 30.0, "skeleton": None}
    ext = r.json()["extensions"]["MMCP_motion"]
    assert [s["num_frames"] for s in ext["samples"]] == [40] * 3
    assert all("reference" not in s for s in ext["samples"]), "a response like a text one"


def test_capabilities_advertise_reference(client) -> None:
    m = client.get("/capabilities").json()["models"][0]
    assert "motion_reference" in m["supported_segments"]
    assert m["limits"]["max_reference_frames"] == 300
    assert m["supports_motion_reference_mixed"] is True, "implied: a reference mixes like text"


def test_gate_rejects_without_the_capability() -> None:
    c = TestClient(build_app(NullBackbone()))
    skel, names = _joints(c)
    caps = c.get("/capabilities").json()["models"][0]
    assert "motion_reference" not in caps["supported_segments"]
    assert "max_reference_frames" not in caps["limits"]
    assert caps["supports_motion_reference_mixed"] is False
    r = c.post("/generate", json=_request(skel, _segment(names)))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_segment"


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
    ]


@pytest.mark.parametrize("loops", [False, True])
@pytest.mark.parametrize("i", range(15))
def test_a_reference_is_validated_like_a_text_segment(i, loops) -> None:
    b = ReferenceNull(loops=loops, poses=True)
    c = TestClient(build_app(b))
    skel, names = _joints(c)
    segments, extra = _layouts(names)[i]
    others = {"text": {"type": "text", "prompt": "a person sits", "duration_frames": 17},
              "unconditioned": {"type": "unconditioned", "duration_frames": 9},
              "pose": {"type": "pose", "prompt": "a person sits"}}
    refs = [_segment(names, t=12, duration_frames=20, seed=10 + k) for k in range(len(segments))]
    with_refs = [refs[k] if s == _REF else others[s] for k, s in enumerate(segments)]
    with_text = [_as_text(refs[k]) if s == _REF else others[s] for k, s in enumerate(segments)]
    got = _outcome(c, _request(skel, *with_refs, **extra))
    want = _outcome(c, _request(skel, *with_text, **extra))
    assert got == want


def test_mixed_segments_reach_the_backbone_in_order(backbone, client) -> None:
    skel, names = _joints(client)
    segs = [_segment(names, t=10, duration_frames=15, seed=100),
            {"type": "text", "prompt": "a person sits", "duration_frames": 20, "seed": 7},
            _segment(names, t=30, duration_frames=25, seed=101)]
    r = client.post("/generate", json=_request(skel, *segs))
    assert r.status_code == 200, r.text
    got = backbone.received[-1]
    assert [s.type for s in got.segments] == ["motion_reference", "text", "motion_reference"]
    assert got.motion_references == [got.segments[0], got.segments[2]]
    assert [s.seed for s in got.segments] == [100, 7, 101]
    assert got.total_frames == 60


def test_the_output_length_is_duration_frames_not_the_clips() -> None:
    c = TestClient(build_app(ReferenceNull(max_reference_frames=50)))
    skel, names = _joints(c)
    # a clip within the cap, a take far longer than it: fine (text rules)
    r = c.post("/generate", json=_request(skel, _segment(names[:2], t=40, duration_frames=200)))
    assert r.status_code == 200, r.text
    assert r.json()["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == 200


# ---- the payload's own checks -------------------------------------------

def test_reference_joints_must_be_on_the_skeleton(client) -> None:
    skel, names = _joints(client)
    good = _segment(names)
    r = client.post("/generate", json=_request(skel, good, _segment(names[:-1] + ["NotAJoint"])))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_joint"


@pytest.mark.parametrize("t,ok", [(300, True), (301, False)])
def test_max_reference_frames_caps_the_clip(t, ok) -> None:
    c = TestClient(build_app(ReferenceNull(max_reference_frames=300)))
    skel, names = _joints(c)
    r = c.post("/generate", json=_request(skel, _segment(names[:2], t=10),
                                          _segment(names[:2], t=t, duration_frames=10)))
    if ok:
        assert r.status_code == 200, r.text
    else:
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "invalid_options"
        assert err["details"] == {"max_reference_frames": 300, "reference_frames": 301, "segment": 1}


def test_no_frame_cap_when_not_advertised() -> None:
    c = TestClient(build_app(ReferenceNull(max_reference_frames=None)))
    skel, names = _joints(c)
    assert "max_reference_frames" not in c.get("/capabilities").json()["models"][0]["limits"]
    r = c.post("/generate", json=_request(skel, _segment(names[:2], t=400, duration_frames=30)))
    assert r.status_code == 200, r.text


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
    assert got.reference_frames == 12 and got.duration_frames == 40


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


# ---- batches --------------------------------------------------------------

def test_batch_items_with_references(client) -> None:
    skel, names = _joints(client)
    good = _request(skel, _segment(names, t=8), {"type": "text", "prompt": "sit", "duration_frames": 8})
    bad = _request(skel, _segment(names[:-1] + ["NotAJoint"]))
    r = client.post("/generate", json={"protocol_version": "1.2", "requests": [good, good, bad]})
    res = r.json()["results"]
    assert "gltf" in res[0] and "gltf" in res[1]
    assert res[2]["error"]["code"] == "unknown_joint"
