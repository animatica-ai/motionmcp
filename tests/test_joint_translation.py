# SPDX-License-Identifier: Apache-2.0
"""MMCP 1.4: joints that move, curves keyed where they bend, and asking for that.

Three additions, each opt-in on the model's side, each checked here through the
server and back out through the SDK's own parser -- a document the server can
write and its own client cannot read is the failure these exist to catch.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="needs the [server] extras")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from motionmcp import GenerateRequest, ModelSpec, MotionResult, build_app
from motionmcp.client.gltf_parser import parse_gltf, xyzw_to_rotmat
from motionmcp.null_backbone import NullBackbone
from motionmcp.protocol import PROTOCOL_VERSION

T = 30
HEAD = 2  # Hips, Spine, Head, LeftFoot, RightFoot: Head hangs off Spine
SLIDE = np.linspace(0.3, 0.6, T, dtype=np.float32)  # the head slides up 30 cm


def _ry(deg: float) -> np.ndarray:
    h = np.radians(deg) / 2
    return np.array([0.0, np.sin(h), 0.0, np.cos(h)], dtype=np.float32)


class Moving(NullBackbone):
    """A rest pose whose head slides on its neck and turns about +Y."""

    def __init__(self, *, joints: bool = True, reduction: bool = False,
                 keyframes: dict | None = None):
        super().__init__()
        self.joints = joints
        self.reduction = reduction
        self.keyframes = keyframes
        self.seen: list[GenerateRequest] = []

    def capabilities(self) -> ModelSpec:
        spec = super().capabilities()
        spec.supports_joint_translation = self.joints
        spec.supports_keyframe_reduction = self.reduction
        return spec

    async def generate(self, request: GenerateRequest) -> MotionResult:
        self.seen.append(request)
        rest = await super().generate(request)
        rotations = rest.rotations.copy()
        rotations[0, :, HEAD] = [_ry(90.0 * t / (T - 1)) for t in range(T)]
        translations = np.broadcast_to(
            np.asarray([j.rest_translation for j in request.skeleton.joints], np.float32),
            (1, T, len(request.skeleton.joints), 3)).copy()
        translations[0, :, 0] = rest.root_translations[0]
        translations[0, :, HEAD, 1] = SLIDE
        return MotionResult(rotations=rotations, root_translations=rest.root_translations,
                            translations=translations, keyframes=self.keyframes)


def _post(backbone: Moving, **options):
    c = TestClient(build_app(backbone))
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "model": "null",
        "skeleton": c.get("/capabilities").json()["models"][0]["canonical_skeleton"],
        "segments": [{"type": "text", "prompt": "look up", "duration_frames": T}],
    }
    if options:
        body["options"] = options
    return c.post("/generate", json=body)


def _channels(doc: dict) -> dict:
    anim = doc["animations"][0]
    return {(ch["target"]["node"], ch["target"]["path"]): anim["samplers"][ch["sampler"]]
            for ch in anim["channels"]}


# --- joint translation ------------------------------------------------------

def test_a_moved_joint_gets_a_channel_and_reads_back_as_an_offset() -> None:
    r = _post(Moving())
    assert r.status_code == 200, r.text
    doc = r.json()
    channels = _channels(doc)
    assert (HEAD, "translation") in channels
    # only the joint that moves: the others sit at their rest offset
    assert {n for n, p in channels if p == "translation"} == {0, HEAD}

    m = parse_gltf(doc)
    assert np.allclose(m["local_translations"][:, HEAD, 1], SLIDE, atol=1e-6)
    assert np.allclose(m["local_translations"][:, 1], [0.0, 0.5, 0.0])  # Spine at rest
    # an offset is not a world position: posed_joints stays the root's alone
    assert np.allclose(m["posed_joints"][:, HEAD], 0.0)
    assert np.allclose(m["posed_joints"][:, 0], [0.0, 0.95, 0.0])
    assert np.allclose(m["local_translations"][:, 0], m["posed_joints"][:, 0])


def test_channels_are_dropped_for_a_model_that_does_not_advertise_them() -> None:
    r = _post(Moving(joints=False))
    assert r.status_code == 200, r.text
    assert {n for n, p in _channels(r.json()) if p == "translation"} == {0}


def test_translations_must_match_the_rotations() -> None:
    with pytest.raises(ValueError, match="translations must be"):
        MotionResult(rotations=np.zeros((1, T, 5, 4), np.float32),
                     root_translations=np.zeros((1, T, 3), np.float32),
                     translations=np.zeros((1, T, 4, 3), np.float32))


# --- keyframes --------------------------------------------------------------

def test_a_channel_keyed_where_it_bends_reads_back_whole() -> None:
    keys = {HEAD: {"rotation": [0, 15, T - 1], "translation": [0, T - 1]}}
    r = _post(Moving(keyframes=keys))
    assert r.status_code == 200, r.text
    doc = r.json()
    channels = _channels(doc)
    assert doc["accessors"][channels[(HEAD, "rotation")]["input"]]["count"] == 3
    assert doc["accessors"][channels[(HEAD, "translation")]["input"]]["count"] == 2
    assert doc["accessors"][channels[(1, "rotation")]["input"]]["count"] == T  # unkeyed: dense

    m = parse_gltf(doc)
    assert m["num_frames"] == T
    assert m["fps"] == pytest.approx(30.0)
    assert m["local_rot_mats"].shape == (T, 5, 3, 3)
    # What a reader plays back between the keys: slerp on the rotation, a line
    # on the translation. The head turns about one axis at a steady rate and
    # slides at one, so both give back the dense curve on every frame.
    dense = xyzw_to_rotmat(np.stack([_ry(90.0 * t / (T - 1)) for t in range(T)]))
    assert np.allclose(m["local_rot_mats"][:, HEAD], dense, atol=1e-5)
    assert np.allclose(m["local_translations"][:, HEAD, 1], SLIDE, atol=1e-5)


# --- keyframe_budget --------------------------------------------------------

def test_a_budget_is_refused_where_reduction_is_not_advertised() -> None:
    r = _post(Moving(reduction=False), keyframe_budget=0.01)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_options"


def test_a_budget_reaches_a_model_that_advertises_reduction() -> None:
    backbone = Moving(reduction=True)
    r = _post(backbone, keyframe_budget=0.01)
    assert r.status_code == 200, r.text
    assert backbone.seen[0].options.keyframe_budget == pytest.approx(0.01)


def test_no_budget_means_every_frame() -> None:
    backbone = Moving(reduction=True)
    assert _post(backbone).status_code == 200
    assert backbone.seen[0].options is None or backbone.seen[0].options.keyframe_budget is None


@pytest.mark.parametrize("budget", [0, -1.0])
def test_a_budget_is_a_positive_tolerance(budget: float) -> None:
    from motionmcp import Options
    with pytest.raises(ValidationError):
        Options(keyframe_budget=budget)
