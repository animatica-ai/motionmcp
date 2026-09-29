# SPDX-License-Identifier: Apache-2.0
"""The one rule for a ``video_reference`` URL, shared by the request schema,
the client's ``video_source`` and ``fetch_video_url``. Standard library only
(the client imports it without the [server] extras)."""

from __future__ import annotations

from urllib.parse import urlsplit

MAX_URL_LENGTH = 4096


def check_video_url(url: str) -> str:
    """Return ``url`` if it is an acceptable video URL, else raise ValueError.

    ``https`` (any case) with a host; no user info (``user:pw@``), no
    whitespace or control characters, a valid port if one is given, at most
    :data:`MAX_URL_LENGTH` characters. Where the host points is not checked
    here: that needs a DNS lookup, and is ``fetch_video_url``'s job.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("video.url must be a non-empty string")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"video.url is longer than {MAX_URL_LENGTH} characters")
    if any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in url):
        raise ValueError("video.url must not contain whitespace or control characters")
    parts = urlsplit(url)
    if parts.scheme.lower() != "https":
        raise ValueError("video.url must be an https URL")
    if "@" in parts.netloc:
        raise ValueError("video.url must not carry user info (user:password@)")
    if not parts.hostname:
        raise ValueError("video.url must have a host")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("video.url has an invalid port") from None
    if port is not None and not 0 < port < 65536:
        raise ValueError("video.url has an invalid port")
    return url
