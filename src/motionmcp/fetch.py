# SPDX-License-Identifier: Apache-2.0
"""Fetch a ``video_reference``'s ``video.url`` the safe way (MMCP 1.2).

A server that fetches a URL a client sent is an SSRF target: the URL can name
the server's own loopback, its cloud metadata endpoint, or a host on its
private network, directly or through a redirect or a DNS name that resolves
there. :func:`fetch_video_url` is the SDK's answer, standard library only:

- ``https`` only, with the URL rule the request schema applies;
- every host is resolved and **every** address it resolves to must be
  public (no loopback, private, link-local -- which covers the
  169.254.169.254 metadata address --, carrier-grade NAT, multicast,
  reserved or unspecified addresses); the connection then goes to the
  address that was checked, so a second DNS answer cannot swap it;
- redirects are followed by hand, at most ``max_redirects``, and each hop
  is checked the same way;
- the body is streamed and cut off past ``max_bytes`` (413
  ``payload_too_large``), and the whole fetch has a deadline.

Failures are :class:`~motionmcp.errors.ProtocolError`\\ s a backbone can let
propagate: ``invalid_request`` for a URL that can't or mustn't be fetched,
``payload_too_large`` for one over the cap, ``timeout`` for the deadline.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
from typing import Callable, Optional
from urllib.parse import urljoin, urlsplit

from ._urls import check_video_url
from .errors import ProtocolError

__all__ = ["fetch_video_url", "is_public_address"]

_REDIRECTS = (301, 302, 303, 307, 308)
_CHUNK = 64 * 1024


def is_public_address(address: str) -> bool:
    """True if ``address`` (IPv4 or IPv6) is a globally routable unicast
    address a server may fetch from."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _resolve(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise ProtocolError("invalid_request", f"video.url host {host!r} does not resolve: {exc}",
                            details={"host": host}) from None
    return list(dict.fromkeys(info[4][0] for info in infos))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """An HTTPS connection to one checked address, verifying the certificate
    (and sending SNI) for the URL's host name."""

    def __init__(self, host: str, address: str, port: int, timeout: float,
                 context: ssl.SSLContext):
        super().__init__(host, port, timeout=timeout, context=context)
        self._address = address

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _default_connect(host: str, address: str, port: int, timeout: float) -> http.client.HTTPConnection:
    return _PinnedHTTPSConnection(host, address, port, timeout, ssl.create_default_context())


# Replaceable for tests: (host, checked address, port, timeout) -> connection.
_connect: Callable[[str, str, int, float], http.client.HTTPConnection] = _default_connect


def fetch_video_url(url: str, *, max_bytes: int, timeout: float = 30.0,
                    max_redirects: int = 3, user_agent: str = "motionmcp-fetch",
                    resolve: Optional[Callable[[str, int], list[str]]] = None) -> bytes:
    """Download ``url`` (a ``video_reference``'s ``video.url``) and return its
    bytes, refusing anything a server must not fetch. See the module doc.

    ``max_bytes`` is required: pass the model's ``limits.max_video_bytes``.
    ``timeout`` bounds the whole fetch, redirects included.
    """
    resolve = resolve or _resolve
    deadline = time.monotonic() + timeout
    for _hop in range(max_redirects + 1):
        try:
            check_video_url(url)
        except ValueError as exc:
            raise ProtocolError("invalid_request", str(exc), details={"url": url}) from None
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port or 443
        addresses = resolve(host, port)
        blocked = [a for a in addresses if not is_public_address(a)]
        if not addresses or blocked:
            raise ProtocolError(
                "invalid_request",
                f"video.url host {host!r} resolves to an address the server won't fetch",
                details={"host": host, "blocked": blocked})
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        conn = _connect(host, addresses[0], port, _left(deadline))
        try:
            conn.request("GET", path, headers={"Host": parts.netloc, "User-Agent": user_agent,
                                               "Accept": "video/*"})
            resp = conn.getresponse()
            if resp.status in _REDIRECTS:
                location = resp.getheader("Location")
                if not location:
                    raise ProtocolError("invalid_request", "video.url redirect without Location")
                url = urljoin(url, location)
                continue
            if resp.status != 200:
                raise ProtocolError("invalid_request",
                                    f"video.url answered HTTP {resp.status}",
                                    details={"status": resp.status})
            declared = resp.getheader("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise _too_large(int(declared), max_bytes)
            return _read_capped(resp, conn, max_bytes, deadline)
        except (OSError, http.client.HTTPException) as exc:
            if isinstance(exc, socket.timeout) or time.monotonic() >= deadline:
                raise ProtocolError("timeout", "fetching video.url timed out") from None
            raise ProtocolError("invalid_request", f"could not fetch video.url: {exc}") from None
        finally:
            conn.close()
    raise ProtocolError("invalid_request", f"video.url redirected more than {max_redirects} times")


def _left(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise ProtocolError("timeout", "fetching video.url timed out")
    return left


def _read_capped(resp, conn, max_bytes: int, deadline: float) -> bytes:
    chunks, total = [], 0
    while True:
        if conn.sock is not None:
            conn.sock.settimeout(_left(deadline))
        chunk = resp.read(_CHUNK)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            raise _too_large(None, max_bytes)
        chunks.append(chunk)


def _too_large(size: Optional[int], cap: int) -> ProtocolError:
    return ProtocolError(
        "payload_too_large",
        f"video.url is {'more than ' + str(cap) if size is None else size} bytes; "
        f"max_video_bytes is {cap}",
        details={"max_video_bytes": cap, **({} if size is None else {"video_bytes": size})},
    )
