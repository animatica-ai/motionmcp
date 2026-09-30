# SPDX-License-Identifier: Apache-2.0
"""Hold arriving requests for a moment so a model can answer several in one call.

A server without a batch window answers each ``POST /generate`` on its own: the
model runs once per request, whatever else is in flight. With a window, a
request waits up to ``batch_window_ms``, and the requests that arrived in that
moment and share a :meth:`~motionmcp.Backbone.batch_key` go to
:meth:`~motionmcp.Backbone.generate_batch` together.

One queue per ``(model, key)``, first come first served: a batch takes the
oldest waiting requests up to the cap, and whatever is left over waits for the
next window rather than being refused. A queued request has no deadline of its
own — the client's own timeout is the one that matters.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Hashable

from .backbone import Backbone
from .schemas import GenerateRequest

RunBatch = Callable[[Backbone, list], Awaitable[list]]


class RequestBatcher:
    """Collects requests per key and runs each group through ``run_batch``."""

    def __init__(self, run_batch: RunBatch, window_ms: int):
        self._run_batch = run_batch
        self._window_s = window_ms / 1000
        self._waiting: dict[Hashable, list[tuple[GenerateRequest, asyncio.Future]]] = {}
        self._drains: dict[Hashable, asyncio.Future] = {}
        self._caps: dict[Hashable, int] = {}

    async def run(
        self, key: Hashable, backbone: Backbone, request: GenerateRequest, max_batch_size: int,
    ):
        """Wait for this request's batch to run; return its result or raise its error."""
        answer: asyncio.Future = asyncio.get_event_loop().create_future()
        self._waiting.setdefault(key, []).append((request, answer))
        self._caps[key] = max_batch_size
        if key not in self._drains:
            self._drains[key] = asyncio.ensure_future(self._drain(key, backbone))
        return await answer

    async def _drain(self, key: Hashable, backbone: Backbone) -> None:
        try:
            while True:
                await asyncio.sleep(self._window_s)
                queue = self._waiting.get(key) or []
                if not queue:
                    return
                cap = self._caps[key]
                taken, self._waiting[key] = queue[:cap], queue[cap:]
                await self._run_group(backbone, taken)
        finally:
            self._drains.pop(key, None)
            if not self._waiting.get(key):
                self._waiting.pop(key, None)
                self._caps.pop(key, None)

    async def _run_group(self, backbone: Backbone, taken: list) -> None:
        try:
            outcomes = await self._run_batch(backbone, [request for request, _ in taken])
        except Exception as exc:  # noqa: BLE001 — the whole call failed; every request in it did
            outcomes = [exc] * len(taken)
        for (_, answer), outcome in zip(taken, outcomes):
            if answer.done():  # the client went away while its batch ran
                continue
            if isinstance(outcome, BaseException):
                answer.set_exception(outcome)
            else:
                answer.set_result(outcome)
