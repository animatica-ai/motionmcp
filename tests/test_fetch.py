# SPDX-License-Identifier: Apache-2.0
"""motionmcp.fetch_video_url: fetching a video_reference's URL without SSRF.
Standard library only (runs in the client-only job). The network is faked:
``resolve`` and the connection factory are replaced."""

from __future__ import annotations

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

    def read(self, n):
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
])
def test_non_public_addresses_are_refused(net, address) -> None:
    assert not is_public_address(address)
    with pytest.raises(ProtocolError) as e:
        fetch_video_url("https://evil.example.com/v.mp4", max_bytes=10,
                        resolve=_dns({"evil.example.com": [address]}))
    assert e.value.code == "invalid_request"
    assert net[1] == [], "never connected"


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
