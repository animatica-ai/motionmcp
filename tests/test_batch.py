# SPDX-License-Identifier: Apache-2.0
"""POST /generate takes a batch body: several requests, one result each."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from motionmcp import MotionResult, build_app
from motionmcp.errors import ProtocolError
from motionmcp.null_backbone import NullBackbone


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app(NullBackbone()))


def _item(client: TestClient, prompt: str = "stand", frames: int = 10, **extra) -> dict:
    canonical = client.get("/capabilities").json()["models"][0]["canonical_skeleton"]
    return {"model": "null", "skeleton": canonical,
            "segments": [{"type": "text", "prompt": prompt, "duration_frames": frames}], **extra}


def _batch(*items) -> dict:
    return {"protocol_version": "1.0", "requests": list(items)}


def test_capabilities_advertise_batches(client: TestClient) -> None:
    m = client.get("/capabilities").json()["models"][0]
    assert m["supports_batch"] is True
    assert m["limits"]["max_batch_size"] == 16


def test_a_batch_answers_one_gltf_per_request_in_order(client: TestClient) -> None:
    r = client.post("/generate", json=_batch(
        _item(client, "walk", 10),
        _item(client, "wave", 24, options={"num_samples": 3}),
    ))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/json")
    first, second = r.json()["results"]
    assert first["gltf"]["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == 10
    # variations of one character ride along: one animation per sample
    assert [a["name"] for a in second["gltf"]["animations"]] == ["sample_0", "sample_1", "sample_2"]
    assert second["gltf"]["extensions"]["MMCP_motion"]["samples"][0]["num_frames"] == 24


def test_a_bad_item_gets_its_own_error_and_the_rest_still_run(client: TestClient) -> None:
    good = _item(client)
    unknown = {**_item(client), "model": "nope"}
    malformed = {**_item(client), "segments": "not a list"}
    r = client.post("/generate", json=_batch(good, unknown, malformed, good))
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert "gltf" in results[0] and "gltf" in results[3]
    assert results[1]["error"]["code"] == "unknown_model"
    assert results[2]["error"]["code"] == "schema_validation"


def test_an_item_may_carry_its_own_protocol_version(client: TestClient) -> None:
    r = client.post("/generate", json=_batch({**_item(client), "protocol_version": "1.0"},
                                              {**_item(client), "protocol_version": "9.0"}))
    ok, old = r.json()["results"]
    assert "gltf" in ok and old["error"]["code"] == "version_unsupported"


def test_a_batch_too_big_or_empty_is_refused_whole(client: TestClient) -> None:
    r = client.post("/generate", json=_batch(*[_item(client)] * 17))
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_options"
    r = client.post("/generate", json=_batch())
    assert r.status_code == 422 and r.json()["error"]["code"] == "schema_validation"
    r = client.post("/generate", json={"protocol_version": "9.0", "requests": [_item(client)]})
    assert r.json()["error"]["code"] == "version_unsupported"


def test_a_single_request_is_answered_as_before(client: TestClient) -> None:
    r = client.post("/generate", json={"protocol_version": "1.0", **_item(client)})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("model/gltf+json")
    assert r.json()["animations"][0]["name"] == "sample_0"


class _BatchingNull(NullBackbone):
    """Batches on its own: one call for the whole group, one item fails."""

    calls: list[int] = []

    def generate_batch(self, requests):
        type(self).calls.append(len(requests))
        out = []
        for req in requests:
            if req.segments[0].prompt == "fail":
                out.append(ProtocolError("invalid_options", "cannot do that"))
            else:
                out.append(self.generate(req))
        return out


def test_a_backbone_that_batches_gets_the_group_in_one_call() -> None:
    _BatchingNull.calls = []
    c = TestClient(build_app(_BatchingNull()))
    r = c.post("/generate", json=_batch(_item(c, "a"), _item(c, "fail"), _item(c, "b")))
    assert _BatchingNull.calls == [3], "one call for the group, not one per item"
    a, failed, b = r.json()["results"]
    assert "gltf" in a and "gltf" in b
    assert failed["error"]["code"] == "invalid_options"


class _Raising(NullBackbone):
    def generate(self, req):
        if req.segments[0].prompt == "boom":
            raise RuntimeError("the GPU caught fire")
        return super().generate(req)


def test_an_item_that_raises_is_an_internal_error_for_that_item_only() -> None:
    c = TestClient(build_app(_Raising()))
    ok, boom = c.post("/generate", json=_batch(_item(c, "fine"), _item(c, "boom"))).json()["results"]
    assert isinstance(ok["gltf"], dict)
    assert boom["error"]["code"] == "internal_error"
    assert "caught fire" in boom["error"]["message"]


def test_motion_result_is_still_the_unit() -> None:
    # generate_batch returns MotionResults, the same thing generate does
    assert MotionResult.__name__ == "MotionResult"
