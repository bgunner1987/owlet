"""Tests for the local Owlet API compatibility layer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Self
from unittest.mock import AsyncMock

import aiohttp
import pytest
from pyowletapi.exceptions import OwletAuthenticationError, OwletConnectionError

from custom_components.owlet.api import (
    MAX_RETRY_AFTER_SECONDS,
    OwletAPI,
    OwletAPIAuthenticationError,
    OwletAPIRequestError,
    OwletLegacyRequestError,
)

_VALID_PROPERTIES = [
    {
        "property": {
            "name": "REAL_TIME_VITALS",
            "value": '{"hr":123,"ox":98}',
        }
    }
]


class FakeResponse:
    """Minimal aiohttp response context manager for deterministic tests."""

    def __init__(
        self,
        status: int,
        payload: Any = None,
        *,
        headers: dict[str, str] | None = None,
        json_error: BaseException | None = None,
        text: str | None = None,
    ) -> None:
        self.status = status
        self.payload = {} if payload is None else payload
        self.headers = headers or {}
        self.json_error = json_error
        self.response_text = text
        self.json_calls = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def json(self) -> Any:
        self.json_calls += 1
        if self.json_error:
            raise self.json_error
        return self.payload

    async def text(self) -> str:
        if self.response_text is not None:
            return self.response_text
        return json.dumps(self.payload)


class FakeSession:
    """Record calls and return a fixed sequence of responses or exceptions."""

    def __init__(self, *responses: FakeResponse | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def make_api(
    *responses: FakeResponse | BaseException,
    token: str | None = "fake-token",
    expiry: float | None = 9_999_999_999,
    refresh: str | None = "fake-refresh",
) -> tuple[OwletAPI, FakeSession, AsyncMock]:
    """Create an API with fake HTTP and sleep implementations."""
    session = FakeSession(*responses)
    sleep = AsyncMock()
    api = OwletAPI(
        "europe",
        token=token,
        expiry=expiry,
        refresh=refresh,
        session=session,
        sleep=sleep,
    )
    return api, session, sleep


def successful_refresh(api: OwletAPI) -> AsyncMock:
    """Return a refresh mock that updates all in-memory credentials."""

    async def _refresh() -> dict[str, Any]:
        api._update_tokens("new-token", 9_999_999_999, "new-refresh")
        return api.tokens

    return AsyncMock(side_effect=_refresh)


def paths(api: OwletAPI, session: FakeSession) -> list[str]:
    """Return recorded request URLs without the regional base URL."""
    return [call["url"].removeprefix(api._api_url) for call in session.calls]


def test_error_types_preserve_coordinator_classification() -> None:
    """Temporary requests stay connection errors; exhausted auth stays auth."""
    assert issubclass(OwletAPIRequestError, OwletConnectionError)
    assert not issubclass(OwletAPIRequestError, OwletAuthenticationError)
    assert issubclass(OwletAPIAuthenticationError, OwletAuthenticationError)


@pytest.mark.asyncio
async def test_valid_token_property_poll_has_no_authentication_probe() -> None:
    """A normal V3 poll sends APP_ACTIVE and properties, never /devices.json."""
    api, session, sleep = make_api(
        FakeResponse(204),
        FakeResponse(200, _VALID_PROPERTIES),
    )

    result = await api.get_properties("TEST-SERIAL")

    assert result["response"]["REAL_TIME_VITALS"]["value"]
    assert paths(api, session) == [
        "/dsns/TEST-SERIAL/properties/APP_ACTIVE/datapoints.json",
        "/dsns/TEST-SERIAL/properties.json",
    ]
    assert "/devices.json" not in paths(api, session)
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token", "expiry"),
    [("fake-token", 0), (None, 9_999_999_999), ("fake-token", None)],
)
async def test_missing_or_expired_token_refreshes_before_target_request(
    token: str | None, expiry: float | None
) -> None:
    """Missing or expired local auth refreshes without a /devices.json probe."""
    api, session, _sleep = make_api(
        FakeResponse(200, {"ok": True}), token=token, expiry=expiry
    )
    api.refresh_authentication = successful_refresh(api)

    assert await api._request("GET", "/target.json") == {"ok": True}

    api.refresh_authentication.assert_awaited_once()
    assert paths(api, session) == ["/target.json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404])
async def test_client_error_is_not_retried_or_refreshed(status: int) -> None:
    """Non-authentication client errors fail immediately with safe context."""
    api, session, sleep = make_api(FakeResponse(status, {"error": "request"}))
    api.refresh_authentication = AsyncMock()

    with pytest.raises(OwletAPIRequestError) as caught:
        await api._request("GET", "/target.json")

    assert caught.value.status == status
    assert len(session.calls) == 1
    sleep.assert_not_awaited()
    api.refresh_authentication.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failure_refreshes_and_replays_once(status: int) -> None:
    """An actual 401 or 403 causes one refresh and one successful replay."""
    api, session, sleep = make_api(
        FakeResponse(status),
        FakeResponse(200, {"ok": True}),
    )
    api.refresh_authentication = successful_refresh(api)

    assert await api._request("GET", "/target.json") == {"ok": True}

    api.refresh_authentication.assert_awaited_once()
    assert paths(api, session) == ["/target.json", "/target.json"]
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_401_after_refresh_is_authentication_error() -> None:
    """A rejected replay is not refreshed or retried indefinitely."""
    api, session, sleep = make_api(FakeResponse(401), FakeResponse(401))
    api.refresh_authentication = successful_refresh(api)

    with pytest.raises(OwletAPIAuthenticationError, match="after token refresh"):
        await api._request("GET", "/target.json")

    api.refresh_authentication.assert_awaited_once()
    assert len(session.calls) == 2
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_http_500_then_success_retries_without_refresh() -> None:
    """A short HTTP 500 interruption succeeds on its bounded retry."""
    api, session, sleep = make_api(
        FakeResponse(500, {"error": "temporary"}),
        FakeResponse(200, {"ok": True}),
    )
    api.refresh_authentication = AsyncMock()

    assert await api._request("GET", "/target.json") == {"ok": True}

    assert len(session.calls) == 2
    sleep.assert_awaited_once_with(1.0)
    api.refresh_authentication.assert_not_awaited()


@pytest.mark.asyncio
async def test_persistent_http_500_stops_after_three_attempts() -> None:
    """A persistent HTTP 500 fails after two short backoffs."""
    api, session, sleep = make_api(
        *[FakeResponse(500, {"error": "temporary"}) for _ in range(3)]
    )
    api.refresh_authentication = AsyncMock()

    with pytest.raises(OwletAPIRequestError) as caught:
        await api._request("GET", "/target.json")

    assert caught.value.status == 500
    assert len(session.calls) == 3
    assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0]
    api.refresh_authentication.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [502, 503, 504])
async def test_other_server_errors_are_retried(status: int) -> None:
    """Gateway and unavailable responses use the same bounded retry policy."""
    api, session, sleep = make_api(
        FakeResponse(status),
        FakeResponse(200, {"ok": True}),
    )
    api.refresh_authentication = AsyncMock()

    assert await api._request("GET", "/target.json") == {"ok": True}

    assert len(session.calls) == 2
    sleep.assert_awaited_once_with(1.0)
    api.refresh_authentication.assert_not_awaited()


@pytest.mark.asyncio
async def test_http_429_uses_valid_retry_after_without_refresh() -> None:
    """A valid Retry-After value controls the bounded retry delay."""
    api, session, sleep = make_api(
        FakeResponse(429, headers={"Retry-After": "3.5"}),
        FakeResponse(200, {"ok": True}),
    )
    api.refresh_authentication = AsyncMock()

    assert await api._request("GET", "/target.json") == {"ok": True}

    assert len(session.calls) == 2
    sleep.assert_awaited_once_with(3.5)
    api.refresh_authentication.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_after", "expected_delay"),
    [("invalid", 1.0), ("999999", MAX_RETRY_AFTER_SECONDS)],
)
async def test_http_429_invalid_or_large_retry_after_is_bounded(
    retry_after: str, expected_delay: float
) -> None:
    """Bad server delay values cannot block a Home Assistant poll for minutes."""
    api, _session, sleep = make_api(
        FakeResponse(429, headers={"Retry-After": retry_after}),
        FakeResponse(200, {"ok": True}),
    )

    assert await api._request("GET", "/target.json") == {"ok": True}
    sleep.assert_awaited_once_with(expected_delay)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [aiohttp.ClientConnectionError("private detail"), asyncio.TimeoutError()],
)
async def test_transport_failure_retries_without_leaking_detail(
    error: BaseException,
) -> None:
    """Network errors and timeouts retry once without exposing their text."""
    api, session, sleep = make_api(error, FakeResponse(200, {"ok": True}))

    assert await api._request("GET", "/target.json") == {"ok": True}

    assert len(session.calls) == 2
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_refresh_transport_failure_is_bounded_and_safe() -> None:
    """Token refresh transport failures retry and become connection errors."""
    api, _session, sleep = make_api(expiry=0)
    api.refresh_authentication = AsyncMock(
        side_effect=[
            aiohttp.ClientConnectionError("private detail"),
            asyncio.TimeoutError(),
            aiohttp.ClientConnectionError("private final detail"),
        ]
    )

    with pytest.raises(OwletAPIRequestError) as caught:
        await api.authenticate()

    assert caught.value.endpoint == "/authentication/refresh"
    assert "private" not in str(caught.value)
    assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_http_204_post_does_not_parse_json() -> None:
    """A successful bodyless POST is accepted without response.json()."""
    response = FakeResponse(204, json_error=AssertionError("must not be called"))
    api, _session, _sleep = make_api(response)

    assert await api._request("POST", "/command.json", {}) is None
    assert response.json_calls == 0


@pytest.mark.asyncio
async def test_all_successful_2xx_statuses_are_accepted() -> None:
    """Successful non-200 responses with JSON are accepted."""
    api, _session, _sleep = make_api(FakeResponse(202, {"accepted": True}))
    assert await api._request("POST", "/command.json", {}) == {"accepted": True}


@pytest.mark.asyncio
async def test_invalid_json_from_required_get_has_context() -> None:
    """A JSON-required GET fails clearly and is not retried."""
    api, session, sleep = make_api(
        FakeResponse(200, json_error=ValueError("private parser detail"))
    )

    with pytest.raises(OwletAPIRequestError) as caught:
        await api._request("GET", "/target.json")

    assert "invalid JSON response" in str(caught.value)
    assert "private parser detail" not in str(caught.value)
    assert len(session.calls) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_activate_guarantees_local_request_method() -> None:
    """activate() cannot fall back to pyowletapi.request()."""
    api, _session, _sleep = make_api()
    api._request = AsyncMock(return_value=None)

    await api.activate("TEST-SERIAL")

    api._request.assert_awaited_once_with(
        "POST",
        "/dsns/TEST-SERIAL/properties/APP_ACTIVE/datapoints.json",
        {"datapoint": {"metadata": {}, "value": 1}},
    )


@pytest.mark.asyncio
async def test_get_properties_guarantees_local_request_method() -> None:
    """get_properties() uses the local method for activation and retrieval."""
    api, _session, _sleep = make_api()
    api._request = AsyncMock(side_effect=[None, _VALID_PROPERTIES])

    await api.get_properties("TEST-SERIAL")

    assert [call.args[:2] for call in api._request.await_args_list] == [
        (
            "POST",
            "/dsns/TEST-SERIAL/properties/APP_ACTIVE/datapoints.json",
        ),
        ("GET", "/dsns/TEST-SERIAL/properties.json"),
    ]


@pytest.mark.asyncio
async def test_post_command_guarantees_local_request_method() -> None:
    """post_command() uses the local method for both HTTP requests."""
    api, _session, _sleep = make_api()
    api._request = AsyncMock(side_effect=[None, {"ok": True}])

    result = await api.post_command("TEST-SERIAL", "BASE_STATION_ON_CMD", {})

    assert result == {"ok": True}
    assert [call.args[:2] for call in api._request.await_args_list] == [
        (
            "POST",
            "/dsns/TEST-SERIAL/properties/APP_ACTIVE/datapoints.json",
        ),
        (
            "POST",
            "/dsns/TEST-SERIAL/properties/BASE_STATION_ON_CMD/datapoints.json",
        ),
    ]


@pytest.mark.asyncio
async def test_get_devices_and_version_check_use_local_requests() -> None:
    """Inherited discovery helpers dynamically dispatch into local methods."""
    api, _session, _sleep = make_api()
    devices = [{"device": {"dsn": "TEST-SERIAL"}}]
    api._request = AsyncMock(side_effect=[devices, None, _VALID_PROPERTIES])

    result = await api.get_devices([3])

    assert result["response"] == devices
    assert [call.args[:2] for call in api._request.await_args_list] == [
        ("GET", "/devices.json"),
        (
            "POST",
            "/dsns/TEST-SERIAL/properties/APP_ACTIVE/datapoints.json",
        ),
        ("GET", "/dsns/TEST-SERIAL/properties.json"),
    ]


@pytest.mark.asyncio
async def test_legacy_failure_has_safe_context_once_per_phase(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A legacy generic error is contextual, redacted, and warning-deduplicated."""
    api, _session, _sleep = make_api()
    legacy = OwletConnectionError(
        "Error sending request parent@example.com auth_token secret-value"
    )

    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            api._request = AsyncMock(side_effect=[None, legacy])
            with pytest.raises(OwletLegacyRequestError) as caught:
                await api.get_properties("SERIAL-123456")

        api._request = AsyncMock(side_effect=[None, _VALID_PROPERTIES])
        await api.get_properties("SERIAL-123456")

        api._request = AsyncMock(side_effect=[None, legacy])
        with pytest.raises(OwletLegacyRequestError):
            await api.get_properties("SERIAL-123456")

    message = str(caught.value)
    assert "Operation: get_properties" in message
    assert "custom_components.owlet.api.OwletAPI" in message
    assert "custom_components.owlet.api.OwletAPI._request" in message
    assert "<redacted-dsn>" in message
    assert "SERIAL-123456" not in message
    assert "parent@example.com" not in caplog.text
    assert "secret-value" not in caplog.text
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 2


@pytest.mark.asyncio
async def test_http_error_redacts_secrets_email_and_device_serial() -> None:
    """Exceptions never expose response credentials, email, or full DSNs."""
    api, _session, _sleep = make_api(
        FakeResponse(
            500,
            {
                "authorization": "auth_token response-secret",
                "refresh_token": "refresh-secret",
                "email": "parent@example.com",
            },
        )
    )

    with pytest.raises(OwletAPIRequestError) as caught:
        await api._send_once(
            "GET",
            "/dsns/SERIAL-123456/properties.json",
            None,
            expect_json=True,
        )

    message = str(caught.value)
    for secret in (
        "response-secret",
        "refresh-secret",
        "parent@example.com",
        "SERIAL-123456",
    ):
        assert secret not in message
    assert "<redacted-dsn>" in message
