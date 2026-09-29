# SPDX-License-Identifier: Apache-2.0
"""motionmcp.fetch_video_url: fetching a video_reference's URL without SSRF.
Standard library only (runs in the client-only job). The network is faked:
``resolve`` and the connection factory are replaced."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time

import pytest

from motionmcp import fetch as fetch_mod
from motionmcp.errors import ProtocolError
from motionmcp.fetch import fetch_video_url, is_public_address

PUBLIC = "93.184.216.34"


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status, self._body, self._headers = status, body, headers or {}

    def getheader(self, name, default=None):
        return self._headers.get(name, default)

    def read1(self, n):
        chunk, self._body = self._body[:n], self._body[n:]
        return chunk


class FakeConnection:
    sock = None

    def __init__(self, routes, log, host, address, port, timeout):
        self.routes, self.log = routes, log
        self.host, self.address, self.port = host, address, port

    def request(self, method, path, headers):
        self.log.append((self.host, self.address, path, headers["Host"]))
        self._resp = self.routes[(self.host, path)]

    def getresponse(self):
        return self._resp

    def close(self):
        pass


@pytest.fixture
def net(monkeypatch):
    routes, log = {}, []
    monkeypatch.setattr(fetch_mod, "_connect",
                        lambda *a: FakeConnection(routes, log, *a))
    return routes, log


def _dns(table):
    return lambda host, port: table[host]


def test_fetches_a_public_url(net) -> None:
    routes, log = net
    routes[("cdn.example.com", "/a.mp4?sig=1")] = FakeResponse(body=b"x" * 1000)
    got = fetch_video_url("https://cdn.example.com/a.mp4?sig=1", max_bytes=1000,
                          resolve=_dns({"cdn.example.com": [PUBLIC]}))
    assert got == b"x" * 1000
    assert log == [("cdn.example.com", PUBLIC, "/a.mp4?sig=1", "cdn.example.com")], \
        "connected to the checked address, with the URL's host for TLS and Host"


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.0.0.5", "172.16.3.4", "192.168.1.1", "169.254.169.254",
    "100.64.0.1", "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1",
    # IPv6 forms that reach an IPv4 address: judged by that address
    "64:ff9b::a9fe:a9fe", "64:ff9b::7f00:1", "64:ff9b::a00:1",     # NAT64 -> metadata, loopback, 10/8
    "64:ff9b:1::1", "64:ff9b:1::5db8:d822",                          # NAT64 local-use: never
    "::127.0.0.1", "::a9fe:a9fe", "::10.0.0.1",                      # IPv4-compatible
    "2002:7f00:1::1", "2002:a9fe:a9fe::",                            # 6to4
    "2001:0:4136:e378:8000:63bf:80ff:fffe",                          # Teredo, client 127.0.0.1
])
def test_non_public_addresses_are_refused(net, address) -> None:
    assert not is_public_address(address)
    with pytest.raises(ProtocolError) as e:
        fetch_video_url("https://evil.example.com/v.mp4", max_bytes=10,
                        resolve=_dns({"evil.example.com": [address]}))
    assert e.value.code == "invalid_request"
    assert net[1] == [], "never connected"


@pytest.mark.parametrize("address", [PUBLIC, "2606:2800:220:1:248:1893:25c8:1946",
                                     "64:ff9b::5db8:d822", "::ffff:93.184.216.34",
                                     "2002:5db8:d822::1"])
def test_public_addresses_pass(address) -> None:
    assert is_public_address(address)


def test_the_refusal_does_not_name_the_addresses(net) -> None:
    with pytest.raises(ProtocolError) as e:
        fetch_video_url("https://evil.example.com/v.mp4", max_bytes=10,
                        resolve=_dns({"evil.example.com": ["10.0.0.7"]}))
    assert "10.0.0.7" not in json.dumps(e.value.to_envelope())


def test_max_bytes_defaults_and_must_be_positive(net) -> None:
    routes, _ = net
    dns = _dns({"cdn.example.com": [PUBLIC]})
    routes[("cdn.example.com", "/v.mp4")] = FakeResponse(body=b"abc")
    assert fetch_video_url("https://cdn.example.com/v.mp4", resolve=dns) == b"abc"
    routes[("cdn.example.com", "/v.mp4")] = FakeResponse(body=b"abc")
    assert fetch_video_url("https://cdn.example.com/v.mp4", max_bytes=None, resolve=dns) == b"abc"
    assert fetch_mod.DEFAULT_MAX_VIDEO_BYTES == 100 * 1024 * 1024
    for bad in (0, -1, 1.5, "10", True):
        with pytest.raises(ValueError):
            fetch_video_url("https://cdn.example.com/v.mp4", max_bytes=bad, resolve=dns)


def test_one_private_answer_among_public_ones_is_refused(net) -> None:
    with pytest.raises(ProtocolError):
        fetch_video_url("https://mixed.example.com/v.mp4", max_bytes=10,
                        resolve=_dns({"mixed.example.com": [PUBLIC, "10.1.2.3"]}))


def test_every_redirect_is_checked(net) -> None:
    routes, log = net
    routes[("cdn.example.com", "/v.mp4")] = FakeResponse(
        302, headers={"Location": "https://internal.example.com/secret"})
    with pytest.raises(ProtocolError) as e:
        fetch_video_url("https://cdn.example.com/v.mp4", max_bytes=10,
                        resolve=_dns({"cdn.example.com": [PUBLIC],
                                      "internal.example.com": ["10.0.0.9"]}))
    assert e.value.code == "invalid_request" and len(log) == 1


def test_redirect_to_http_is_refused(net) -> None:
    routes, _ = net
    routes[("cdn.example.com", "/v.mp4")] = FakeResponse(
        301, headers={"Location": "http://cdn.example.com/v.mp4"})
    with pytest.raises(ProtocolError):
        fetch_video_url("https://cdn.example.com/v.mp4", max_bytes=10,
                        resolve=_dns({"cdn.example.com": [PUBLIC]}))


def test_relative_redirects_and_the_hop_limit(net) -> None:
    routes, log = net
    routes[("cdn.example.com", "/a")] = FakeResponse(302, headers={"Location": "/b"})
    routes[("cdn.example.com", "/b")] = FakeResponse(302, headers={"Location": "/a"})
    with pytest.raises(ProtocolError) as e:
        fetch_video_url("https://cdn.example.com/a", max_bytes=10, max_redirects=3,
                        resolve=_dns({"cdn.example.com": [PUBLIC]}))
    assert "redirected more than 3" in e.value.message and len(log) == 4


def test_byte_cap_while_streaming_and_by_content_length(net) -> None:
    routes, _ = net
    routes[("cdn.example.com", "/big")] = FakeResponse(body=b"x" * 200_000)
    routes[("cdn.example.com", "/said")] = FakeResponse(body=b"", headers={"Content-Length": "999"})
    dns = _dns({"cdn.example.com": [PUBLIC]})
    for path in ("/big", "/said"):
        with pytest.raises(ProtocolError) as e:
            fetch_video_url("https://cdn.example.com" + path, max_bytes=100, resolve=dns)
        assert e.value.code == "payload_too_large"


def test_http_errors(net) -> None:
    routes, _ = net
    routes[("cdn.example.com", "/gone")] = FakeResponse(404)
    with pytest.raises(ProtocolError) as e:
        fetch_video_url("https://cdn.example.com/gone", max_bytes=10,
                        resolve=_dns({"cdn.example.com": [PUBLIC]}))
    assert e.value.code == "invalid_request" and e.value.details["status"] == 404


@pytest.mark.parametrize("url", [
    "http://cdn.example.com/v.mp4", "https://user:pw@cdn.example.com/v.mp4",
    "https://cdn.example.com:99999/v.mp4", "https://cdn.example.com/a b.mp4",
    "file:///etc/passwd", "https:///v.mp4",
])
def test_bad_urls_are_refused_before_any_lookup(url) -> None:
    def resolve(host, port):
        raise AssertionError("looked up")
    with pytest.raises(ProtocolError) as e:
        fetch_video_url(url, max_bytes=10, resolve=resolve)
    assert e.value.code == "invalid_request"


def test_deadline() -> None:
    with pytest.raises(ProtocolError) as e:
        fetch_video_url("https://cdn.example.com/v.mp4", max_bytes=10, timeout=0,
                        resolve=_dns({"cdn.example.com": [PUBLIC]}))
    assert e.value.code == "timeout"


# ---- the hard deadline, against a real (plain-TCP) server that drips ----------

class DripServer:
    """Answers one connection by sending ``head`` then ``body`` one byte
    every ``gap`` seconds."""

    def __init__(self, head: bytes, body: bytes = b"", gap: float = 0.4, drip_head=True):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.head, self.body, self.gap, self.drip_head = head, body, gap, drip_head
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        conn, _ = self.sock.accept()
        try:
            conn.recv(65536)
            if not self.drip_head:
                conn.sendall(self.head)
            for b in (self.head if self.drip_head else b"") + self.body:
                conn.sendall(bytes([b]))
                time.sleep(self.gap)
        except OSError:
            pass
        finally:
            conn.close()
            self.sock.close()


def _via(server: DripServer, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_connect",
                        lambda host, addr, port, timeout: http.client.HTTPConnection(
                            "127.0.0.1", server.port, timeout=timeout))


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_a_dripping_server_is_cut_off_at_the_deadline(monkeypatch, phase) -> None:
    head = b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n"
    server = (DripServer(head, gap=0.2) if phase == "headers"
              else DripServer(head, b"x" * 1000, gap=0.2, drip_head=False))
    _via(server, monkeypatch)
    start = time.monotonic()
    with pytest.raises(ProtocolError) as e:
        fetch_video_url("https://cdn.example.com/v.mp4", max_bytes=10_000, timeout=1.0,
                        resolve=_dns({"cdn.example.com": [PUBLIC]}))
    elapsed = time.monotonic() - start
    assert e.value.code == "timeout"
    assert elapsed < 1.0 + 0.5, f"held {elapsed:.1f}s for a 1s timeout"


def test_a_prompt_server_is_not_cut_off(monkeypatch) -> None:
    server = DripServer(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc", gap=0, drip_head=False)
    _via(server, monkeypatch)
    assert fetch_video_url("https://cdn.example.com/v.mp4", max_bytes=10, timeout=2.0,
                           resolve=_dns({"cdn.example.com": [PUBLIC]})) == b"abc"
