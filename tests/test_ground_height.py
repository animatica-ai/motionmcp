# SPDX-License-Identifier: Apache-2.0
"""The ``ground_height`` constraint (MMCP 1.3): its schema, and the opt-in gate."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the [server] extras")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from motionmcp import GenerateRequest, GroundHeightConstraint, ModelSpec, build_app
from motionmcp.null_backbone import NullBackbone
from motionmcp.protocol import OPTIONAL_CONSTRAINTS, PROTOCOL_VERSION, SUPPORTED_CONSTRAINTS


class GroundNull(NullBackbone):
    """A null backbone that says it understands ground heights."""

    def capabilities(self) -> ModelSpec:
        spec = super().capabilities()
        spec.supported_constraints = list(SUPPORTED_CONSTRAINTS) + ["ground_height"]
        return spec


def _request(client: TestClient, constraints: list) -> dict:
    canonical = client.get("/capabilities").json()["models"][0]["canonical_skeleton"]
    return {
        "protocol_version": "1.3",
        "model": "null",
        "skeleton": canonical,
        "segments": [{"type": "text", "prompt": "run and jump down", "duration_frames": 30}],
        "constraints": constraints,
    }


GROUND = {"type": "ground_height",
          "points": [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.5, -0.6], [0.0, 7.0, -0.6]]}


def test_protocol_is_1_3() -> None:
    assert PROTOCOL_VERSION == "1.3"
    assert "ground_height" in OPTIONAL_CONSTRAINTS
    assert "ground_height" not in SUPPORTED_CONSTRAINTS


def test_parses_as_a_constraint() -> None:
    c = GroundHeightConstraint(**GROUND)
    assert c.points[2] == (0.0, 4.5, -0.6)


@pytest.mark.parametrize("bad, why", [
    ({"points": [[0.0, 0.0]]}, "Field required|at least 3|3 items"),
    ({"points": [[0.0, 0.0, float("nan")]]}, "finite"),
    ({"points": []}, "at least 1"),
    ({"points": [[0, 0, 0]], "frames": [0]}, "Extra inputs"),
])
def test_rejects_malformed(bad: dict, why: str) -> None:
    with pytest.raises(ValidationError, match=why):
        GroundHeightConstraint(type="ground_height", **bad)


def test_refused_where_not_advertised() -> None:
    c = TestClient(build_app(NullBackbone()))
    r = c.post("/generate", json=_request(c, [GROUND]))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported_constraint"
    assert "ground_height" not in err["details"]["supported_constraints"]


def test_accepted_where_advertised() -> None:
    c = TestClient(build_app(GroundNull()))
    r = c.post("/generate", json=_request(c, [GROUND]))
    assert r.status_code == 200, r.text


def test_places_not_frames_so_no_frame_check() -> None:
    # far-away points are not an error: the route may wander, the server ignores them
    c = TestClient(build_app(GroundNull()))
    r = c.post("/generate", json=_request(c, [{**GROUND, "points": [[100.0, 100.0, -3.0]]}]))
    assert r.status_code == 200, r.text


def test_reaches_the_backbone_parsed() -> None:
    c = TestClient(build_app(GroundNull()))
    req = GenerateRequest.model_validate(_request(c, [GROUND]))
    (g,) = req.constraints
    assert isinstance(g, GroundHeightConstraint)
    assert len(g.points) == 4
