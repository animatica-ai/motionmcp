# SPDX-License-Identifier: Apache-2.0
"""pick_model: the model the caller asked for used to be ignored."""

from motionmcp.client.http import pick_model

CAPS = {"models": [
    {"id": "kimodo-soma-rp", "fps": 30.0},
    {"id": "ardy-core-rp", "fps": 20.0},
]}


class TestPickModel:
    def test_picks_the_requested_model(self):
        assert pick_model(CAPS, "ardy-core-rp")["fps"] == 20.0
        assert pick_model(CAPS, "kimodo-soma-rp")["fps"] == 30.0

    def test_match_is_case_and_space_insensitive(self):
        assert pick_model(CAPS, "  ARDY-Core-RP ")["id"] == "ardy-core-rp"

    def test_unknown_name_falls_back_to_first(self):
        # the same setting travels between servers; a model missing here
        # must not break generation
        assert pick_model(CAPS, "gpt-dancer")["id"] == "kimodo-soma-rp"

    def test_no_choice_keeps_the_previous_behaviour(self):
        for wanted in (None, "", "   "):
            assert pick_model(CAPS, wanted)["id"] == "kimodo-soma-rp"

    def test_no_models_returns_none(self):
        for caps in ({"models": []}, {}, None, {"models": None}):
            assert pick_model(caps, "ardy-core-rp") is None

    def test_entry_without_id_is_skipped_not_crashed(self):
        caps = {"models": [{"fps": 1.0}, {"id": "ardy-core-rp"}]}
        assert pick_model(caps, "ardy-core-rp")["id"] == "ardy-core-rp"


class TestRetargetState:
    """supports_retargeting (a model property) vs whether this install can."""

    def test_reads_the_health_field(self, monkeypatch):
        import io
        import json as _json

        from motionmcp.client import http as mmcp_client

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        for value, expected in (("ready", "ready"),
                                ("unavailable", "unavailable"),
                                ("something-else", "unknown"),
                                (None, "unknown")):
            payload = _json.dumps({"status": "ok", "retargeting": value}).encode()
            monkeypatch.setattr(mmcp_client.urllib.request, "urlopen",
                                lambda *a, payload=payload, **k: _Resp(payload))
            assert mmcp_client.retarget_state("http://x") == expected

    def test_unreachable_server_is_unknown_not_a_refusal(self):
        from motionmcp.client import http as mmcp_client
        # nothing listening: must not raise, and must not claim "unavailable"
        assert mmcp_client.retarget_state("http://127.0.0.1:9", timeout=0.3) == "unknown"


class TestCachedCapabilities:
    """The cache-only read must use the same key get_capabilities writes."""

    def test_returns_what_get_capabilities_stored(self, monkeypatch):
        import io
        import json as _json

        from motionmcp.client import http as mmcp_client

        class _Resp(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        payload = _json.dumps({"models": [{"id": "kimodo-soma-rp"},
                                          {"id": "ardy-core-rp"}]}).encode()
        mmcp_client.clear_capabilities_cache()
        monkeypatch.setattr(mmcp_client.urllib.request, "urlopen",
                            lambda *a, **k: _Resp(payload))
        mmcp_client.get_capabilities("http://x:8000")

        caps = mmcp_client.cached_capabilities("http://x:8000")
        assert caps is not None, "key mismatch: the cache read found nothing"
        assert [m["id"] for m in caps["models"]] == ["kimodo-soma-rp",
                                                    "ardy-core-rp"]
        # trailing slash must hit the same entry
        assert mmcp_client.cached_capabilities("http://x:8000/") is not None

    def test_cold_cache_is_none_not_a_network_call(self):
        from motionmcp.client import http as mmcp_client
        mmcp_client.clear_capabilities_cache()
        assert mmcp_client.cached_capabilities("http://127.0.0.1:9") is None

    @staticmethod
    def _serve_per_token(monkeypatch):
        """Fake urlopen answering with the model list of the Bearer token's
        account; returns the list of Authorization headers it was called with."""
        import io
        import json as _json

        from motionmcp.client import http as mmcp_client

        class _Resp(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        calls = []

        def _urlopen(req, *a, **k):
            auth = req.get_header("Authorization")
            calls.append(auth)
            model = {None: "local-rp", "Bearer tok-a": "model-a",
                     "Bearer tok-b": "model-b"}[auth]
            return _Resp(_json.dumps({"models": [{"id": model}]}).encode())

        mmcp_client.clear_capabilities_cache()
        monkeypatch.setattr(mmcp_client.urllib.request, "urlopen", _urlopen)
        return mmcp_client, calls

    def test_two_tokens_get_two_entries(self, monkeypatch):
        mmcp_client, calls = self._serve_per_token(monkeypatch)
        a = mmcp_client.get_capabilities("http://x:8000", access_token="tok-a")
        b = mmcp_client.get_capabilities("http://x:8000", access_token="tok-b")
        assert a["models"][0]["id"] == "model-a"
        assert b["models"][0]["id"] == "model-b"
        assert len(calls) == 2
        assert mmcp_client.cached_capabilities(
            "http://x:8000", access_token="tok-a")["models"][0]["id"] == "model-a"
        assert mmcp_client.cached_capabilities(
            "http://x:8000", access_token="tok-b")["models"][0]["id"] == "model-b"

    def test_same_token_hits_cache_without_network(self, monkeypatch):
        mmcp_client, calls = self._serve_per_token(monkeypatch)
        first = mmcp_client.get_capabilities("http://x:8000", access_token="tok-a")
        again = mmcp_client.get_capabilities("http://x:8000", access_token="tok-a")
        assert again is first
        assert calls == ["Bearer tok-a"]

    def test_no_token_and_token_are_separate_entries(self, monkeypatch):
        mmcp_client, calls = self._serve_per_token(monkeypatch)
        mmcp_client.get_capabilities("http://x:8000")
        assert mmcp_client.cached_capabilities(
            "http://x:8000", access_token="tok-a") is None
        mmcp_client.get_capabilities("http://x:8000", access_token="tok-a")
        assert mmcp_client.cached_capabilities(
            "http://x:8000")["models"][0]["id"] == "local-rp"
        assert mmcp_client.cached_capabilities(
            "http://x:8000", access_token="tok-a")["models"][0]["id"] == "model-a"
        assert len(calls) == 2

    def test_trailing_slash_hits_same_token_entry(self, monkeypatch):
        mmcp_client, calls = self._serve_per_token(monkeypatch)
        mmcp_client.get_capabilities("http://x:8000", access_token="tok-a")
        mmcp_client.get_capabilities("http://x:8000/", access_token="tok-a")
        assert len(calls) == 1
        assert mmcp_client.cached_capabilities(
            "http://x:8000/", access_token="tok-a")["models"][0]["id"] == "model-a"
