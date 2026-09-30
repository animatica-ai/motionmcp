# SPDX-License-Identifier: Apache-2.0
"""A batch window gathers requests that arrived separately into one model call."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi", reason="needs the [server] extras")
httpx = pytest.importorskip("httpx", reason="needs the [dev] extras")

from fastapi.testclient import TestClient  # noqa: E402

from motionmcp import build_app  # noqa: E402
from motionmcp.null_backbone import NullBackbone  # noqa: E402


class _CountingBackbone(NullBackbone):
    """Records the size of every batch the server hands it."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def generate_batch(self, requests):
        self.batch_sizes.append(len(requests))
        return [self.generate(r) for r in requests]


class _PickyBackbone(_CountingBackbone):
    """Only requests of the same duration may share a call."""

    def batch_key(self, request):
        return tuple(s.duration_frames for s in request.segments)


def _request(frames: int = 10) -> dict:
    canonical = TestClient(build_app(NullBackbone())).get("/capabilities").json()[
        "models"][0]["canonical_skeleton"]
    return {"protocol_version": "1.0", "model": "null", "skeleton": canonical,
            "segments": [{"type": "text", "prompt": "walk", "duration_frames": frames}]}


def _post_together(backbone: NullBackbone, bodies: list[dict], **app_kwargs) -> list[int]:
    """Post every body at once against a fresh app; return the status codes."""
    app = build_app(backbone, **app_kwargs)

    async def run() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with app.router.lifespan_context(app):
                answers = await asyncio.gather(
                    *(client.post("/generate", json=body) for body in bodies)
                )
        return [answer.status_code for answer in answers]

    return asyncio.run(run())


def test_without_a_window_every_request_runs_on_its_own() -> None:
    backbone = _CountingBackbone()
    codes = _post_together(backbone, [_request()] * 3)
    assert codes == [200, 200, 200]
    assert backbone.batch_sizes == []


def test_a_window_gathers_requests_that_arrived_separately() -> None:
    backbone = _CountingBackbone()
    codes = _post_together(backbone, [_request()] * 3, batch_window_ms=40)
    assert codes == [200, 200, 200]
    assert backbone.batch_sizes == [3]


def test_the_batch_never_grows_past_max_batch_size() -> None:
    backbone = _CountingBackbone()
    codes = _post_together(backbone, [_request()] * 3, batch_window_ms=20, max_batch_size=2)
    assert codes == [200, 200, 200]
    # the third request waits for the next window rather than being refused
    assert backbone.batch_sizes == [2, 1]


def test_requests_of_different_keys_do_not_share_a_call() -> None:
    backbone = _PickyBackbone()
    codes = _post_together(
        backbone, [_request(10), _request(24), _request(10)], batch_window_ms=40,
    )
    assert codes == [200, 200, 200]
    assert sorted(backbone.batch_sizes) == [1, 2]
