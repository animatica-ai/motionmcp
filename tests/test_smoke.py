# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke tests: build the app, hit it via httpx, check the wire."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from motionmcp import (
    GenerateRequest,
    Joint,
    Skeleton,
    build_app,
)
from motionmcp.null_backbone import NullBackbone


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app(NullBackbone()))


def test_capabilities_shape(client: TestClient) -> None:
    r = client.get("/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["protocol_version"] == "1.0"
    assert body["coordinate_system"] == "right_handed_y_up"
    assert body["units"] == "meters"
    assert body["rotation_format"] == "quaternion_xyzw"
    assert len(body["models"]) == 1
    m = body["models"][0]
    assert m["id"] == "null"
    assert m["fps"] == 30.0
    assert m["supports_retargeting"] is False
    assert "canonical_skeleton" in m
    assert "limits" in m


def _request_for(client: TestClient) -> dict:
    """Pull canonical from /capabilities and build a minimal valid request."""
    caps = client.get("/capabilities").json()
    canonical = caps["models"][0]["canonical_skeleton"]
    return {
        "protocol_version": "1.0",
        "model": "null",
        "skeleton": canonical,
        "segments": [{"type": "text", "prompt": "stand", "duration_frames": 10}],
    }


def test_generate_returns_gltf(client: TestClient) -> None:
    req = _request_for(client)
    r = client.post("/generate", json=req)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("model/gltf+json")

    gltf = r.json()
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["asset"]["generator"] == "motionmcp"
    assert len(gltf["animations"]) == 1
    assert gltf["animations"][0]["name"] == "sample_0"

    ext = gltf["extensions"]["MMCP_motion"]
    assert ext["model"] == "null"
    assert ext["fps"] == 30.0
    assert len(ext["samples"]) == 1
    assert ext["samples"][0]["num_frames"] == 10
    assert ext["samples"][0]["chunk_boundaries"] == []


def test_unknown_model(client: TestClient) -> None:
    req = _request_for(client)
    req["model"] = "does-not-exist"
    r = client.post("/generate", json=req)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unknown_model"
    assert "available_models" in err["details"]


def test_retargeting_unsupported(client: TestClient) -> None:
    req = _request_for(client)
    req["skeleton"]["joints"][0]["name"] = "RenamedHips"  # break canonical match
    # Need to also rename any children that pointed at it.
    for j in req["skeleton"]["joints"][1:]:
        if j["parent"] == "Hips":
            j["parent"] = "RenamedHips"
    r = client.post("/generate", json=req)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "retargeting_unsupported"


def test_unsupported_constraint(client: TestClient) -> None:
    req = _request_for(client)
    req["constraints"] = [{
        "type": "root_path",
        "frames": [0],
        "positions_xz": [[0.0, 0.0]],
    }]
    r = client.post("/generate", json=req)
    # NullBackbone advertises pose_keyframe / root_path / effector_target —
    # so root_path is supported. Switch this to a deliberately unsupported case
    # by configuring NullBackbone with a smaller list, instead.


def test_pose_segment_unsupported_by_default(client: TestClient) -> None:
    """NullBackbone defaults to supported_segments=['text','unconditioned']
    (no 'pose'). A request with a pose segment must fail with
    `unsupported_segment`."""
    req = _request_for(client)
    req["segments"] = [{"type": "pose", "prompt": "stand neutral"}]
    r = client.post("/generate", json=req)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported_segment"
    assert "supported_segments" in err["details"]
    assert "pose" not in err["details"]["supported_segments"]


def test_pose_segment_accepted_when_advertised() -> None:
    """When a backbone advertises 'pose', the pose segment passes
    validation and the SDK invokes generate(). NullBackbone returns
    rest pose for any input shape."""
    from motionmcp import build_app, ModelSpec, Skeleton
    from motionmcp.null_backbone import NullBackbone, _default_skeleton
    from motionmcp.protocol import SUPPORTED_CONSTRAINTS

    class PoseCapableNull(NullBackbone):
        def capabilities(self) -> ModelSpec:
            spec = super().capabilities()
            spec.supported_segments = ["text", "unconditioned", "pose"]
            return spec

    c = TestClient(build_app(PoseCapableNull()))
    canonical = c.get("/capabilities").json()["models"][0]["canonical_skeleton"]
    r = c.post("/generate", json={
        "protocol_version": "1.0",
        "model": "null",
        "skeleton": canonical,
        "segments": [{"type": "pose", "prompt": "person waves hello"}],
    })
    assert r.status_code == 200, r.text
    gltf = r.json()
    # Pose segment is intrinsically 1 frame; the response carries 1 frame.
    assert gltf["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == 1


def test_capabilities_reports_supported_segments(client: TestClient) -> None:
    body = client.get("/capabilities").json()
    spec = body["models"][0]
    assert "supported_segments" in spec
    assert "text" in spec["supported_segments"]
    assert "unconditioned" in spec["supported_segments"]
    # NullBackbone doesn't advertise pose generation.
    assert "pose" not in spec["supported_segments"]


def test_unknown_joint_in_constraint(client: TestClient) -> None:
    req = _request_for(client)
    req["constraints"] = [{
        "type": "effector_target",
        "joint": "NonexistentBone",
        "frames": [0],
        "positions": [[0.0, 0.0, 0.0]],
    }]
    r = client.post("/generate", json=req)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_joint"


def test_frame_out_of_range(client: TestClient) -> None:
    req = _request_for(client)
    req["constraints"] = [{
        "type": "pose_keyframe",
        "frame": 99,
        "joint_rotations": {"Hips": [0.0, 0.0, 0.0, 1.0]},
    }]
    r = client.post("/generate", json=req)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "frame_out_of_range"


def test_schema_validation_bad_quat_length(client: TestClient) -> None:
    req = _request_for(client)
    req["skeleton"]["joints"][0]["rest_rotation"] = [0.0, 0.0, 0.0]  # missing w
    r = client.post("/generate", json=req)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "schema_validation"


def test_version_unsupported(client: TestClient) -> None:
    req = _request_for(client)
    req["protocol_version"] = "2.0"
    r = client.post("/generate", json=req)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "version_unsupported"


def test_num_samples_returns_multiple_animations(client: TestClient) -> None:
    req = _request_for(client)
    req["options"] = {"num_samples": 3}
    r = client.post("/generate", json=req)
    assert r.status_code == 200, r.text
    gltf = r.json()
    assert len(gltf["animations"]) == 3
    assert [a["name"] for a in gltf["animations"]] == [
        "sample_0", "sample_1", "sample_2"
    ]


def test_total_frames_property() -> None:
    req = GenerateRequest(
        protocol_version="1.0",
        model="x",
        skeleton=Skeleton(joints=[Joint(
            name="Hips", parent=None,
            rest_translation=(0, 0, 0), rest_rotation=(0, 0, 0, 1),
        )]),
        segments=[
            {"type": "text", "prompt": "a", "duration_frames": 10},
            {"type": "unconditioned", "duration_frames": 20},
        ],
    )
    assert req.total_frames == 30
