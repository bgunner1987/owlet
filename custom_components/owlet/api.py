"""Owlet API compatibility layer with safe diagnostics and auth recovery."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import aiohttp
from pyowletapi.api import OwletAPI as PyOwletAPI
from pyowletapi.exceptions import OwletAuthenticationError, OwletConnectionError

_SUCCESS = {200, 201}
_AUTH_FAILURE = {401, 403}
_REDACTED_KEYS = re.compile(
    r"authorization|cookie|email|password|refresh|secret|token", re.IGNORECASE
)
_TOKEN_VALUE = re.compile(r"(?i)(bearer|auth_token)\s+[^\s,;]+|eyJ[A-Za-z0-9_.-]{20,}")
_EMAIL_VALUE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


class OwletAPIRequestError(OwletConnectionError):
    """An HTTP or transport failure with safe request context."""

    def __init__(
        self,
        method: str,
        endpoint: str,
        *,
        status: int | None = None,
        response: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.method = method
        self.endpoint = endpoint
        self.status = status
        self.response = response
        self.reason = reason
        self.retryable = status == 429 or status is not None and status >= 500
        super().__init__(self._message())

    def _message(self) -> str:
        headline = (
            f"Owlet API request failed: HTTP {self.status}"
            if self.status is not None
            else "Owlet API request failed: network error"
        )
        parts = [headline, f"Method: {self.method}", f"Endpoint: {self.endpoint}"]
        if self.response:
            parts.append(f"Response: {self.response}")
        elif self.reason:
            parts.append(f"Reason: {self.reason}")
        return " | ".join(parts)


class OwletAPIAuthenticationError(OwletAuthenticationError):
    """Authentication failed after one controlled token refresh."""

    def __init__(self, method: str, endpoint: str, status: int) -> None:
        self.method = method
        self.endpoint = endpoint
        self.status = status
        super().__init__(
            f"Owlet API authentication failed after token refresh: HTTP {status}"
            f" | Method: {method} | Endpoint: {endpoint}"
        )


def _sanitise_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if _REDACTED_KEYS.search(str(key))
            else _sanitise_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitise_value(item) for item in value[:10]]
    if isinstance(value, str):
        value = _TOKEN_VALUE.sub("<redacted>", value)
        return _EMAIL_VALUE.sub("<redacted-email>", value)
    return value


async def _safe_response_summary(response: aiohttp.ClientResponse) -> str:
    """Return a bounded, redacted error response suitable for HA logs."""
    try:
        text = await response.text()
    except (UnicodeError, aiohttp.ClientError):
        return "<unreadable response>"
    if not text:
        return "<empty response>"
    try:
        summary = json.dumps(
            _sanitise_value(json.loads(text)), ensure_ascii=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        summary = str(_sanitise_value(text))
    return summary[:512] + ("..." if len(summary) > 512 else "")


class OwletAPI(PyOwletAPI):
    """Small, local hardening layer over the pinned PyPI API client."""

    async def _send(
        self, method: str, endpoint: str, data: dict[str, Any] | None
    ) -> Any:
        try:
            async with self.session.request(
                method, self._api_url + endpoint, headers=self.headers, json=data
            ) as response:
                if response.status not in _SUCCESS:
                    raise OwletAPIRequestError(
                        method,
                        endpoint,
                        status=response.status,
                        response=await _safe_response_summary(response),
                    )
                return await response.json()
        except OwletAPIRequestError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise OwletAPIRequestError(
                method, endpoint, reason=type(err).__name__
            ) from err

    async def _refresh_and_retry(
        self, method: str, endpoint: str, data: dict[str, Any] | None
    ) -> Any:
        self._auth_token = None
        await self.authenticate()
        try:
            return await self._send(method, endpoint, data)
        except OwletAPIRequestError as err:
            if err.status in _AUTH_FAILURE:
                raise OwletAPIAuthenticationError(method, endpoint, err.status) from err
            raise

    async def validate_authentication(self) -> Any:
        """Validate without treating rate limits/server failures as bad auth."""
        try:
            await self._send("GET", "/devices.json", None)
        except OwletAPIRequestError as err:
            if err.status in _AUTH_FAILURE:
                return await self._refresh_and_retry("GET", "/devices.json", None)
            raise
        return None

    async def _request(
        self, method: str, url: str, data: dict[str, Any] | None = None
    ) -> Any:
        """Send an API request, refreshing and retrying once on 401/403."""
        await self.validate_authentication()
        try:
            return await self._send(method, url, data)
        except OwletAPIRequestError as err:
            if err.status in _AUTH_FAILURE:
                return await self._refresh_and_retry(method, url, data)
            raise
