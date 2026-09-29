# SPDX-License-Identifier: Apache-2.0
"""The client's HTTP path, end to end against a real socket (``fake_server``).

Covers ``get_capabilities``, ``generate`` and ``probe_server`` on the wire:
successful ``200`` answers and the headers that go out with them (the
``headers=`` identity contract; ``Authorization`` from ``access_token``
always wins), error envelopes mapped to ``MmcpError``, the ``202`` plus
polling path, GLB (``model/gltf-binary``) answers, and probe statuses.
Assertions are on ``MmcpError.code`` / ``.details``, never on message text.
"""

from __future__ import annotations

import json

import pytest
from gltf_doc import build_glb, build_gltf

from motionmcp.client.gltf_parser import parse_gltf
from motionmcp.client.http import (
    MmcpError,
    clear_capabilities_cache,
    generate,
    get_capabilities,
    poll_job,
    probe_server,
)


class TestCapabilitiesSmoke:
    """Fixture smoke test: one request goes out, the programmed body comes back."""

    def test_get_capabilities_hits_the_fake_server_once(self, mmcp_server):
        body = {"models": [{"id": "kimodo-soma-rp"}]}
        mmcp_server.enqueue(200, headers={"Content-Type": "application/json"},
                            body=json.dumps(body))

        result = get_capabilities(mmcp_server.url, use_cache=False)

        assert result == body
        assert len(mmcp_server.requests) == 1
        assert mmcp_server.requests[0].method == "GET"
        assert mmcp_server.requests[0].path == "/capabilities"


class TestGenerateSuccess:
    def test_200_gltf_json_returns_the_document_and_headers_on_the_wire(self, mmcp_server):
        doc = build_gltf()
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf+json"},
                            body=json.dumps(doc))

        result = generate(
            mmcp_server.url, {"prompt": "walk"}, timeout=5, access_token="tok123",
            headers={"Authorization": "Bearer should-not-win", "X-App": "test/1.0"},
        )

        assert result == doc
        assert len(mmcp_server.requests) == 1
        req = mmcp_server.requests[0]
        assert req.method == "POST"
        assert req.path == "/generate"
        assert req.header("Content-Type") == "application/json"
        # Accept lists several types (GLB too) — contains, not equals
        assert "model/gltf+json" in req.header("Accept")
        assert req.header("X-App") == "test/1.0"
        # access_token-derived Authorization wins over the same key in headers=
        assert req.header("Authorization") == "Bearer tok123"


class TestGenerateGlb:
    def test_200_gltf_binary_returns_a_json_serialisable_document(self, mmcp_server):
        doc = build_gltf()
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf-binary"},
                            body=build_glb(doc))

        result = generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        json.dumps(result)  # no bytes anywhere in the document
        assert parse_gltf(result)["joint_names"] == parse_gltf(doc)["joint_names"]

    def test_content_type_parameters_do_not_matter(self, mmcp_server):
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf-binary; charset=binary"},
                            body=build_glb(build_gltf()))

        result = generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert isinstance(result, dict)

    def test_bad_magic_is_bad_response(self, mmcp_server):
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf-binary"},
                            body=b"nope" + build_glb(build_gltf())[4:])

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "bad_response"

    def test_accept_offers_gltf_binary_at_lower_quality(self, mmcp_server):
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf+json"},
                            body=json.dumps(build_gltf()))

        generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert "model/gltf-binary;q=0.9" in mmcp_server.requests[0].header("Accept")

    def test_202_then_200_gltf_binary_returns_the_document(self, mmcp_server):
        doc = build_gltf()
        mmcp_server.enqueue(202, headers={"Location": "/generate/jobs/1", "Retry-After": "0"})
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf-binary"},
                            body=build_glb(doc))

        result = generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        json.dumps(result)
        assert parse_gltf(result)["joint_names"] == parse_gltf(doc)["joint_names"]

    def test_bad_glb_while_polling_is_bad_response(self, mmcp_server):
        mmcp_server.enqueue(202, headers={"Location": "/generate/jobs/1", "Retry-After": "0"})
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf-binary"},
                            body=b"nope" + build_glb(build_gltf())[4:])

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "bad_response"

class TestGenerateErrors:
    def test_400_envelope_maps_to_mmcperror_code_and_details(self, mmcp_server):
        envelope = {"error": {"code": "unknown_joint", "message": "Joint 'Foo' not found",
                              "details": {"joint": "Foo"}}}
        mmcp_server.enqueue(400, headers={"Content-Type": "application/json"},
                            body=json.dumps(envelope))

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        err = exc_info.value
        assert err.code == "unknown_joint"
        assert str(err) == envelope["error"]["message"]
        assert err.details["joint"] == "Foo"
        assert err.details["status"] == 400   # added, never overwriting the server's

    def test_500_without_json_body_is_http_error(self, mmcp_server):
        mmcp_server.enqueue(500, headers={"Content-Type": "text/plain"}, body="boom")

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "http_error"
        assert exc_info.value.details["status"] == 500

    def test_401_is_auth_required(self, mmcp_server):
        mmcp_server.enqueue(401)

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "auth_required"

    def test_200_html_is_bad_response(self, mmcp_server):
        mmcp_server.enqueue(200, headers={"Content-Type": "text/html"}, body="<html></html>")

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "bad_response"

    def test_200_empty_body_is_bad_response(self, mmcp_server):
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf+json"}, body=b"")

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "bad_response"


class TestGenerateAsyncPolling:
    def test_202_then_200_polls_and_returns_the_document(self, mmcp_server):
        doc = build_gltf()
        mmcp_server.enqueue(202, headers={"Location": "/generate/jobs/1", "Retry-After": "0"})
        mmcp_server.enqueue(202, headers={"Retry-After": "0"})
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf+json"},
                            body=json.dumps(doc))

        result = generate(
            mmcp_server.url, {"prompt": "walk"}, timeout=5, access_token="tok123",
            headers={"X-App": "test/1.0"},
        )

        assert result == doc
        assert len(mmcp_server.requests) == 3
        start_req, poll1, poll2 = mmcp_server.requests
        assert start_req.path == "/generate"
        assert start_req.header("X-App") == "test/1.0"
        assert start_req.header("Authorization") == "Bearer tok123"
        # identity headers ride on the request that STARTS a generation and
        # on nothing else; Authorization rides on every request.
        for poll in (poll1, poll2):
            assert poll.method == "GET"
            assert poll.path == "/generate/jobs/1"
            assert poll.header("X-App") is None
            assert poll.header("Authorization") == "Bearer tok123"

    def test_202_without_location_is_bad_response(self, mmcp_server):
        mmcp_server.enqueue(202, headers={"Retry-After": "0"})

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "bad_response"

    def test_polling_forever_times_out(self, mmcp_server):
        mmcp_server.enqueue(202, headers={"Location": "/generate/jobs/1", "Retry-After": "0"})
        # client sleeps max(retry_after, 0.5)s between polls; queue enough
        # 202s to outlast timeout=1.5 without the fake server's empty-queue
        # 500 fallback kicking in and masking the timeout.
        for _ in range(6):
            mmcp_server.enqueue(202, headers={"Retry-After": "0"})

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=1.5)

        assert exc_info.value.code == "timeout"


class TestGenerateTransport:
    def test_nothing_listening_is_connection_failed(self):
        with pytest.raises(MmcpError) as exc_info:
            generate("http://127.0.0.1:9", {"prompt": "walk"}, timeout=2)

        assert exc_info.value.code == "connection_failed"


class TestProbeServer:
    def test_200_is_online(self, mmcp_server):
        clear_capabilities_cache()
        caps = {"models": [{"id": "kimodo-soma-rp"}]}
        mmcp_server.enqueue(200, headers={"Content-Type": "application/json"},
                            body=json.dumps(caps))

        result = probe_server(mmcp_server.url, timeout=5)

        assert result.ok is True
        assert result.status == "online"
        assert result.capabilities == caps

    def test_401_is_auth_required(self, mmcp_server):
        clear_capabilities_cache()
        mmcp_server.enqueue(401)

        result = probe_server(mmcp_server.url, timeout=5)

        assert result.ok is False
        assert result.status == "auth_required"

    def test_503_is_http_error(self, mmcp_server):
        clear_capabilities_cache()
        mmcp_server.enqueue(503)

        result = probe_server(mmcp_server.url, timeout=5)

        assert result.ok is False
        assert result.status == "http_error"

    def test_html_is_bad_response(self, mmcp_server):
        clear_capabilities_cache()
        mmcp_server.enqueue(200, headers={"Content-Type": "text/html"}, body="<html></html>")

        result = probe_server(mmcp_server.url, timeout=5)

        assert result.ok is False
        assert result.status == "bad_response"

    def test_enveloped_error_is_http_error_not_unreachable(self, mmcp_server):
        # A 403 with the server's own envelope code was reachable: the status
        # must say "http_error", not "unreachable", and keep the server's message.
        clear_capabilities_cache()
        mmcp_server.enqueue(
            403,
            headers={"Content-Type": "application/json"},
            body=json.dumps({"error": {"code": "forbidden", "message": "no entitlement",
                                       "details": {}}}),
        )

        result = probe_server(mmcp_server.url, timeout=5)

        assert result.ok is False
        assert result.status == "http_error"
        assert "no entitlement" in result.message


class TestHttpErrorMapping:
    """One HTTPError -> MmcpError mapping for /capabilities, /generate and polling."""

    def test_capabilities_envelope_carries_the_server_message(self, mmcp_server):
        envelope = {"error": {"code": "forbidden", "message": "Plan does not include this model"}}
        mmcp_server.enqueue(403, headers={"Content-Type": "application/json"},
                            body=json.dumps(envelope))

        with pytest.raises(MmcpError) as exc_info:
            get_capabilities(mmcp_server.url, use_cache=False)

        err = exc_info.value
        assert err.code == "forbidden"
        assert str(err) == envelope["error"]["message"]
        assert err.details["status"] == 403

    def test_401_envelope_is_auth_required_with_the_server_message(self, mmcp_server):
        envelope = {"error": {"code": "unauthorized", "message": "Token expired"}}
        mmcp_server.enqueue(401, headers={"Content-Type": "application/json"},
                            body=json.dumps(envelope))

        with pytest.raises(MmcpError) as exc_info:
            get_capabilities(mmcp_server.url, use_cache=False)

        err = exc_info.value
        assert err.code == "auth_required"
        assert str(err) == envelope["error"]["message"]
        assert err.details["status"] == 401

    def test_retry_after_lands_in_details_as_float(self, mmcp_server):
        mmcp_server.enqueue(429, headers={"Retry-After": "7"})

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.details["retry_after"] == 7.0
        assert exc_info.value.details["status"] == 429

    def test_unparseable_retry_after_is_left_out(self, mmcp_server):
        mmcp_server.enqueue(503, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "http_error"
        assert "retry_after" not in exc_info.value.details

    def test_x_request_id_lands_in_details(self, mmcp_server):
        mmcp_server.enqueue(500, headers={"X-Request-ID": "req-abc"})

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.details["request_id"] == "req-abc"

    def test_x_job_id_is_the_fallback_request_id(self, mmcp_server):
        mmcp_server.enqueue(500, headers={"X-Job-Id": "job-42"})

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.details["request_id"] == "job-42"

    def test_x_request_id_wins_over_x_job_id(self, mmcp_server):
        mmcp_server.enqueue(500, headers={"X-Request-ID": "req-abc", "X-Job-Id": "job-42"})

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.details["request_id"] == "req-abc"

    def test_envelope_while_polling_keeps_the_envelope_code(self, mmcp_server):
        envelope = {"error": {"code": "job_failed", "message": "Inference crashed"}}
        mmcp_server.enqueue(202, headers={"Location": "/generate/jobs/1", "Retry-After": "0"})
        mmcp_server.enqueue(500, headers={"Content-Type": "application/json"},
                            body=json.dumps(envelope))

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        err = exc_info.value
        assert err.code == "job_failed"
        assert str(err) == envelope["error"]["message"]
        assert err.details["status"] == 500

    def test_401_while_polling_is_auth_required(self, mmcp_server):
        mmcp_server.enqueue(202, headers={"Location": "/generate/jobs/1", "Retry-After": "0"})
        mmcp_server.enqueue(401)

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        assert exc_info.value.code == "auth_required"
        assert exc_info.value.details["status"] == 401

    def test_json_without_error_key_is_http_error(self, mmcp_server):
        mmcp_server.enqueue(422, headers={"Content-Type": "application/json"},
                            body=json.dumps({"detail": "field required"}))

        with pytest.raises(MmcpError) as exc_info:
            get_capabilities(mmcp_server.url, use_cache=False)

        assert exc_info.value.code == "http_error"
        assert exc_info.value.details["status"] == 422


class TestResumablePolling:
    """Errors name their phase; a failed poll resumes with ``poll_job``, no second POST."""

    def test_401_while_polling_names_the_poll_phase_and_location(self, mmcp_server):
        mmcp_server.enqueue(202, headers={"Location": "/generate/jobs/1", "Retry-After": "0"})
        mmcp_server.enqueue(401)

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5, access_token="old")

        err = exc_info.value
        assert err.code == "auth_required"
        assert err.details == {"status": 401, "phase": "poll", "location": "/generate/jobs/1"}

    def test_poll_job_resumes_the_same_job_without_a_post(self, mmcp_server):
        doc = build_gltf()
        mmcp_server.enqueue(202, headers={"Location": "/generate/jobs/1", "Retry-After": "0"})
        mmcp_server.enqueue(401)
        mmcp_server.enqueue(200, headers={"Content-Type": "model/gltf+json"},
                            body=json.dumps(doc))

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5, access_token="old")
        result = poll_job(mmcp_server.url, exc_info.value.details["location"],
                          retry_after=0, timeout=5, access_token="fresh")

        assert result == doc
        assert [r.method for r in mmcp_server.requests] == ["POST", "GET", "GET"]
        resume = mmcp_server.requests[-1]
        assert resume.path == "/generate/jobs/1"
        assert resume.header("Authorization") == "Bearer fresh"

    def test_absolute_location_is_polled_and_reported_as_a_path(self, mmcp_server):
        mmcp_server.enqueue(500)

        with pytest.raises(MmcpError) as exc_info:
            poll_job(mmcp_server.url, mmcp_server.url + "/generate/jobs/7",
                     retry_after=0, timeout=5)

        assert mmcp_server.requests[0].path == "/generate/jobs/7"
        assert exc_info.value.details["location"] == "/generate/jobs/7"

    def test_poll_timeout_names_the_poll_phase(self, mmcp_server):
        for _ in range(4):
            mmcp_server.enqueue(202, headers={"Retry-After": "0"})

        with pytest.raises(MmcpError) as exc_info:
            poll_job(mmcp_server.url, "/generate/jobs/1", retry_after=0, timeout=0.8)

        assert exc_info.value.code == "timeout"
        assert exc_info.value.details == {"phase": "poll", "location": "/generate/jobs/1"}

    def test_401_on_the_post_names_the_generate_phase(self, mmcp_server):
        mmcp_server.enqueue(401)

        with pytest.raises(MmcpError) as exc_info:
            generate(mmcp_server.url, {"prompt": "walk"}, timeout=5)

        err = exc_info.value
        assert err.code == "auth_required"
        assert err.details == {"status": 401, "phase": "generate"}
        assert len(mmcp_server.requests) == 1

    def test_unreachable_server_names_the_generate_phase(self):
        with pytest.raises(MmcpError) as exc_info:
            generate("http://127.0.0.1:9", {"prompt": "walk"}, timeout=2)

        assert exc_info.value.code == "connection_failed"
        assert exc_info.value.details["phase"] == "generate"
        assert "location" not in exc_info.value.details

    def test_poll_job_is_exported(self):
        from motionmcp import client
        from motionmcp.client import http

        assert "poll_job" in client.__all__
        assert "poll_job" in http.__all__
        assert client.poll_job is http.poll_job
