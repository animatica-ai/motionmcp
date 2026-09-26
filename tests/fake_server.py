# SPDX-License-Identifier: Apache-2.0
"""A fake MMCP server on a real socket, for wire-level client tests.

Monkeypatching ``urlopen`` (the older pattern in this suite) cannot tell an
``HTTPError`` from a ``resp.status == 202``, does not show which headers
actually went out, and never exercises ``Location`` / ``Retry-After``. This
module runs a stdlib ``ThreadingHTTPServer`` on 127.0.0.1, port 0, in a
daemon thread. The test queues the responses it wants, in order, and reads
back a log of every request the client made. The ``mmcp_server`` fixture in
``conftest.py`` starts and stops it around each test.
"""

from __future__ import annotations

import collections
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RecordedRequest:
    """One request as it arrived: method, path (with query), headers, body."""

    def __init__(self, method, path, headers, body):
        self.method = method
        self.path = path
        self.headers = headers  # dict, keys spelled as they came on the wire
        self.body = body  # bytes, b"" when there was none

    def header(self, name, default=None):
        """Case-insensitive header lookup."""
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return default

    def __repr__(self):
        return f"RecordedRequest({self.method} {self.path})"


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.0: one request per connection, so nothing lingers on keep-alive.
    protocol_version = "HTTP/1.0"

    def _serve(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        fake = self.server.fake  # type: ignore[attr-defined]
        status, headers, out = fake._take(
            RecordedRequest(self.command, self.path, dict(self.headers.items()), body))
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() != "content-length":
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(out)))
        self.send_header("Connection", "close")
        self.end_headers()
        if out:
            self.wfile.write(out)

    do_GET = _serve
    do_POST = _serve
    do_DELETE = _serve

    def log_message(self, format, *args):
        pass


class FakeMmcpServer:
    """Queue of canned responses + log of received requests."""

    def __init__(self):
        self._lock = threading.Lock()
        self._responses = collections.deque()
        self.requests = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.fake = self  # type: ignore[attr-defined]
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        # A short poll interval keeps shutdown() from stalling a whole poll step per test.
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )

    def enqueue(self, status=200, headers=None, body=b""):
        """Queue one response. *body* is bytes or str (str is UTF-8 encoded).

        Responses go out in the order queued, one per request, whatever the
        path. Content-Length is always set from *body*.
        """
        if isinstance(body, str):
            body = body.encode("utf-8")
        with self._lock:
            self._responses.append((status, dict(headers or {}), body))

    def _take(self, request):
        with self._lock:
            self.requests.append(request)
            if self._responses:
                return self._responses.popleft()
        msg = (f"fake_server: no response queued for {request.method} {request.path} "
               f"(request #{len(self.requests)})")
        return 500, {"Content-Type": "text/plain; charset=utf-8"}, msg.encode("utf-8")

    def start(self):
        self._thread.start()

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
