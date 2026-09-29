# SPDX-License-Identifier: Apache-2.0
"""Build the ``video`` of a ``video_reference`` segment (MMCP 1.2).

Standard library only, like the rest of the client: this reads a file and
base64-encodes it; it never decodes the video.
"""

from __future__ import annotations

import base64
import os

from .._urls import check_video_url

__all__ = ["video_source"]

# The containers a video_reference may carry inline, by file extension.
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}


def video_source(path=None, *, url=None, media_type=None, max_bytes=None):
    """Return a ``video_reference`` segment's ``video`` object.

    ``video_source(url="https://...")`` gives ``{"url": ...}`` for the server
    to fetch. ``video_source("clip.mp4")`` reads the file and gives
    ``{"data": <base64>, "media_type": ...}``, the media type taken from the
    extension (``.mp4`` / ``.m4v``, ``.mov``, ``.webm``) unless
    ``media_type`` is passed. The URL must pass the server's rule (https,
    a host, no ``user:pw@``, no whitespace, a valid port).

    Pass the model's ``limits.max_video_bytes`` as ``max_bytes`` to refuse a
    file over it before reading it (``ValueError``): the SDK server refuses
    inline video over it (413 ``payload_too_large``), and the body must still
    fit ``limits.max_request_bytes`` (base64 is a third bigger than the file).
    The encoding is standard, padded base64 on one line, as the protocol
    requires.
    """
    if (path is None) == (url is None):
        raise ValueError("pass exactly one of a file path or url=")
    if url is not None:
        if media_type is not None:
            raise ValueError("media_type goes with a file, not a url")
        return {"url": check_video_url(str(url))}
    if media_type is None:
        ext = os.path.splitext(str(path))[1].lower()
        media_type = _MEDIA_TYPES.get(ext)
        if media_type is None:
            raise ValueError(
                "cannot tell the media type of %r; pass media_type= (one of %s)"
                % (str(path), sorted(set(_MEDIA_TYPES.values()))))
    elif media_type not in _MEDIA_TYPES.values():
        raise ValueError("media_type must be one of %s" % sorted(set(_MEDIA_TYPES.values())))
    if max_bytes is not None and os.path.getsize(path) > max_bytes:
        raise ValueError("%r is %d bytes; max_bytes is %d"
                         % (str(path), os.path.getsize(path), max_bytes))
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        raise ValueError("%r is empty" % str(path))
    return {"data": base64.b64encode(data).decode("ascii"), "media_type": media_type}
