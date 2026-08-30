"""Tests for the local Owlet API compatibility layer."""

import asyncio
import json
from unittest.mock import AsyncMock

import aiohttp
import pytest
from pyowletapi.exceptions import OwletAuthenticationError

from custom_components.owlet.api import (
    OwletAPI,
    OwletAPIAuthenticationError,
    OwletAPIRequestError,
)


class FakeResponse:
    def __init__(self, status: int, payload=None) -> None:
        self.status = status
        self.payload = payload if payload is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return self.payload

    async def text(self):
        return json.dumps(self.payload)


class FakeSession:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)

    def request(self, *_args, **_kwargs):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def make_api(*responses) -> OwletAPI:
    return OwletAPI(
        "europe",
        token="fake",
        expiry=9999999999,
        refresh="fake",
        session=FakeSession(*responses),
    )


@pytest.mark.asyncio
async def test_http_200_returns_json():
    api = make_api(FakeResponse(200, {"ok": True}))
    assert await api._send("GET", "/properties.json", None) == {"ok": True}


@pytest.mark.asyncio
async def test_valid_properties_response():
    properties = [
        {
            "property": {
                "name": "REAL_TIME_VITALS",
                "value": '{"hr":123,"ox":98}',
            }
        }
    ]
    api = make_api(FakeResponse(200, properties))
    assert await api._send("GET", "/dsns/TEST/properties.json", None) == properties


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404, 429, 500])
async def test_http_error_has_safe_request_context(status):
    api = make_api(
        FakeResponse(
            status,
            {
                "error": "failed",
                "refresh_token": "secret",
                "email": "parent@example.com",
            },
        )
    )
    with pytest.raises(OwletAPIRequestError) as caught:
        await api._send("POST", "/dsns/TEST/properties/APP_ACTIVE/datapoints.json", {})
    message = str(caught.value)
    assert (
        f"HTTP {status}" in message
        and "Method: POST" in message
        and "APP_ACTIVE" in message
    )
    assert "secret" not in message and "parent@example.com" not in message
    assert caught.value.retryable is (status in (429, 500))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_refreshes_and_retries_once(status):
    api = make_api(
        FakeResponse(200, []),
        FakeResponse(status),
        FakeResponse(200, {"ok": True}),
    )
    api.authenticate = AsyncMock(return_value={"api_token": "new"})
    assert await api._request("GET", "/properties.json", None) == {"ok": True}
    api.authenticate.assert_awaited_once()


@pytest.mark.asyncio
async def test_auth_error_after_refresh_requests_reauth():
    api = make_api(FakeResponse(401))
    api.authenticate = AsyncMock(return_value={"api_token": "new"})
    with pytest.raises(OwletAPIAuthenticationError):
        await api._refresh_and_retry("GET", "/properties.json", None)


@pytest.mark.asyncio
async def test_expired_token_refresh_failure_is_auth_error():
    api = make_api()
    api.authenticate = AsyncMock(side_effect=OwletAuthenticationError("expired"))
    with pytest.raises(OwletAuthenticationError, match="expired"):
        await api._refresh_and_retry("GET", "/properties.json", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [aiohttp.ClientConnectionError("detail"), asyncio.TimeoutError()]
)
async def test_network_error_is_temporary_connection_error(error):
    api = make_api(error)
    with pytest.raises(OwletAPIRequestError) as caught:
        await api._send("GET", "/properties.json", None)
    assert "network error" in str(caught.value) and "detail" not in str(caught.value)
