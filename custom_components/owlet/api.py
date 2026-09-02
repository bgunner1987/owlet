"""Owlet API compatibility layer with safe diagnostics and auth recovery."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import aiohttp
from pyowletapi.api import OwletAPI as PyOwletAPI
from pyowletapi.exceptions import OwletAuthenticationError, OwletConnectionError

_LOGGER = logging.getLogger(__name__)

_AUTH_FAILURE = {401, 403}
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRY_DELAYS = (1.0, 2.0)
MAX_RETRY_AFTER_SECONDS = 5.0

_REDACTED_KEYS = re.compile(
    r"authorization|cookie|email|password|refresh|secret|token", re.IGNORECASE
)
_TOKEN_VALUE = re.compile(r"(?i)(bearer|auth_token)\s+[^\s,;]+|eyJ[A-Za-z0-9_.-]{20,}")
_EMAIL_VALUE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_DSN_SEGMENT = re.compile(r"(?<=/dsns/)[^/?]+", re.IGNORECASE)


def _sanitise_endpoint(endpoint: str) -> str:
    """Remove device serial numbers from an endpoint used in diagnostics."""
    return _DSN_SEGMENT.sub("<redacted-dsn>", endpoint)


class OwletAPIRequestError(OwletConnectionError):
    """An HTTP, response, or transport failure with safe request context."""

    def __init__(
        self,
        method: str,
        endpoint: str,
        *,
        status: int | None = None,
        response: str | None = None,
        reason: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.method = method.upper()
        self.endpoint = _sanitise_endpoint(endpoint)
        self.status = status
        self.response = response
        self.reason = reason
        self.retry_after = retry_after
        self.retryable = status is None or status in _RETRYABLE_STATUS
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
        self.method = method.upper()
        self.endpoint = _sanitise_endpoint(endpoint)
        self.status = status
        super().__init__(
            f"Owlet API authentication failed after token refresh: HTTP {status}"
            f" | Method: {self.method} | Endpoint: {self.endpoint}"
        )


class OwletLegacyRequestError(OwletConnectionError):
    """An unexpected failure from the pinned client's legacy request path."""

    def __init__(
        self,
        operation: str,
        method: str,
        endpoint: str,
        api_class: str,
        request_implementation: str,
    ) -> None:
        self.operation = operation
        self.method = method.upper()
        self.endpoint = _sanitise_endpoint(endpoint)
        self.api_class = api_class
        self.request_implementation = request_implementation
        super().__init__(
            "Unexpected legacy Owlet request failure"
            f" | Operation: {operation}"
            f" | Method: {self.method}"
            f" | Endpoint: {self.endpoint}"
            f" | API class: {api_class}"
            f" | Request implementation: {request_implementation}"
        )


def _sanitise_value(value: Any) -> Any:
    """Remove credentials and personal data from a response value."""
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


def _retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    """Parse and bound a numeric Retry-After response header."""
    if not headers:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None
    if delay < 0:
        return None
    return min(delay, MAX_RETRY_AFTER_SECONDS)


class OwletAPI(PyOwletAPI):
    """Small, local hardening layer over the pinned PyPI API client."""

    def __init__(
        self,
        *args: Any,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise retry hooks and legacy-error phase tracking."""
        super().__init__(*args, **kwargs)
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or time.time
        self._legacy_failure_reported = False

    def _token_is_current(self) -> bool:
        """Return whether the locally stored token exists and has not expired."""
        try:
            expiry = float(self._expiry)
        except (TypeError, ValueError):
            return False
        return bool(self._auth_token) and expiry > self._clock()

    async def _refresh_authentication_with_retries(self) -> dict[str, Any] | None:
        """Retry short refresh transport failures without exposing their details."""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return await self.refresh_authentication()
            except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                if attempt == len(RETRY_DELAYS):
                    raise OwletAPIRequestError(
                        "POST",
                        "/authentication/refresh",
                        reason=type(err).__name__,
                    ) from err
                await self._sleep(RETRY_DELAYS[attempt])

        raise AssertionError("unreachable authentication retry state")

    async def authenticate(self) -> dict[str, Any] | None:
        """Authenticate from local expiry state without probing /devices.json."""
        if self._auth_token is None and self._refresh is None:
            if self._user is None or self._password is None:
                raise OwletAuthenticationError("Username or password not supplied")
            await self.password_verification()

        if self._token_is_current():
            return None

        tokens = await self._refresh_authentication_with_retries()
        if tokens:
            # pyowletapi clears this flag before returning refreshed tokens. Keep it
            # set until an integration response can persist the new credentials.
            self._tokens_changed = True
        return tokens

    async def _send_once(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None,
        *,
        expect_json: bool,
    ) -> Any:
        """Perform exactly one HTTP request and validate its response."""
        method = method.upper()
        try:
            async with self.session.request(
                method,
                self._api_url + endpoint,
                headers=self.headers,
                json=data,
            ) as response:
                if not 200 <= response.status < 300:
                    raise OwletAPIRequestError(
                        method,
                        endpoint,
                        status=response.status,
                        response=await _safe_response_summary(response),
                        retry_after=_retry_after_seconds(response.headers),
                    )

                if response.status == 204:
                    if expect_json:
                        raise OwletAPIRequestError(
                            method,
                            endpoint,
                            status=response.status,
                            reason="successful response contained no JSON body",
                        )
                    return None

                try:
                    return await response.json()
                except (
                    aiohttp.ContentTypeError,
                    json.JSONDecodeError,
                    TypeError,
                    UnicodeError,
                    ValueError,
                ) as err:
                    if not expect_json:
                        return None
                    raise OwletAPIRequestError(
                        method,
                        endpoint,
                        status=response.status,
                        reason="invalid JSON response",
                    ) from err
        except OwletAPIRequestError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise OwletAPIRequestError(
                method, endpoint, reason=type(err).__name__
            ) from err

    async def _request_with_retries(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None,
    ) -> Any:
        """Retry bounded temporary failures without refreshing authentication."""
        expect_json = method.upper() == "GET"
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return await self._send_once(
                    method, endpoint, data, expect_json=expect_json
                )
            except OwletAPIRequestError as err:
                if err.status in _AUTH_FAILURE:
                    raise
                if not err.retryable or attempt == len(RETRY_DELAYS):
                    raise
                delay = (
                    err.retry_after
                    if err.status == 429 and err.retry_after is not None
                    else RETRY_DELAYS[attempt]
                )
                await self._sleep(delay)

        raise AssertionError("unreachable retry state")

    async def _force_refresh(self) -> dict[str, Any] | None:
        """Invalidate the rejected token and obtain a replacement once."""
        self._auth_token = None
        self._expiry = None
        self.headers.pop("Authorization", None)
        return await self.authenticate()

    async def _refresh_and_retry(
        self, method: str, endpoint: str, data: dict[str, Any] | None
    ) -> Any:
        """Refresh once and replay the original request exactly once."""
        await self._force_refresh()
        try:
            return await self._request_with_retries(method, endpoint, data)
        except OwletAPIRequestError as err:
            if err.status in _AUTH_FAILURE:
                raise OwletAPIAuthenticationError(method, endpoint, err.status) from err
            raise

    async def _request(
        self, method: str, endpoint: str, data: dict[str, Any] | None = None
    ) -> Any:
        """Send directly, refreshing only after expiry or an actual 401/403."""
        await self.authenticate()
        try:
            return await self._request_with_retries(method, endpoint, data)
        except OwletAPIRequestError as err:
            if err.status in _AUTH_FAILURE:
                return await self._refresh_and_retry(method, endpoint, data)
            raise

    def _legacy_error(
        self, operation: str, method: str, endpoint: str
    ) -> OwletLegacyRequestError:
        """Build and log safe legacy context once per continuous failure phase."""
        api_class = f"{type(self).__module__}.{type(self).__qualname__}"
        request_class = type(self)._request
        request_implementation = (
            f"{request_class.__module__}.{request_class.__qualname__}"
        )
        error = OwletLegacyRequestError(
            operation,
            method,
            endpoint,
            api_class,
            request_implementation,
        )
        if not self._legacy_failure_reported:
            _LOGGER.warning("%s", error)
            self._legacy_failure_reported = True
        return error

    async def _operation_request(
        self,
        operation: str,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> Any:
        """Call the local request path and contextualise any legacy failure."""
        try:
            return await self._request(method, endpoint, data)
        except (OwletAPIRequestError, OwletAPIAuthenticationError):
            raise
        except OwletConnectionError as err:
            raise self._legacy_error(operation, method, endpoint) from err

    async def request(
        self, method: str, url: str, data: dict[str, Any] | None = None
    ) -> Any:
        """Route inherited pyowletapi callers through the local implementation."""
        result = await self._operation_request("request", method, url, data)
        self._legacy_failure_reported = False
        return result

    async def validate_authentication(self) -> None:
        """Explicitly validate credentials when setup really needs device data."""
        await self._operation_request("validate_authentication", "GET", "/devices.json")
        self._legacy_failure_reported = False

    async def _is_valid_version(self, dsn: str, versions: list[int]) -> bool:
        """Check a device version without consuming tokens needed by discovery."""
        properties = await self.get_properties(dsn)
        properties_item = properties["response"]
        if "REAL_TIME_VITALS" in properties_item:
            valid = 3 in versions
        elif "CHARGE_STATUS" in properties_item:
            valid = 2 in versions
        else:
            valid = False

        if "tokens" in properties:
            # This internal caller cannot persist credentials. Leave the refresh
            # pending so inherited get_devices() returns it to Home Assistant.
            self._tokens_changed = True
        return valid

    async def activate(
        self, device_serial: str, *, _reset_legacy_phase: bool = True
    ) -> None:
        """Set APP_ACTIVE through the local request path.

        The pinned client states that activation is required before reading
        properties, but it defines no safe lifetime. Keep one activation per poll
        until a reliable server contract supports a bounded cache.
        """
        endpoint = f"/dsns/{device_serial}/properties/APP_ACTIVE/datapoints.json"
        data = {"datapoint": {"metadata": {}, "value": 1}}
        await self._operation_request("activate", "POST", endpoint, data)
        if _reset_legacy_phase:
            self._legacy_failure_reported = False

    async def get_properties(self, device: str) -> dict[str, Any]:
        """Fetch and normalise properties using only the local request path."""
        await self.activate(device, _reset_legacy_phase=False)
        endpoint = f"/dsns/{device}/properties.json"
        raw_properties = await self._operation_request(
            "get_properties", "GET", endpoint
        )
        if not isinstance(raw_properties, list):
            raise OwletAPIRequestError(
                "GET",
                endpoint,
                status=200,
                reason="expected a JSON array of device properties",
            )

        properties: dict[str, Any] = {}
        try:
            for item in raw_properties:
                prop = item["property"]
                properties[prop["name"]] = prop
        except (KeyError, TypeError) as err:
            raise OwletAPIRequestError(
                "GET",
                endpoint,
                status=200,
                reason="invalid device properties JSON structure",
            ) from err

        response: dict[str, Any] = {"response": properties}
        if self._tokens_changed:
            response["tokens"] = self.tokens
            self._tokens_changed = False
        self._legacy_failure_reported = False
        return response

    async def post_command(
        self, device: str, command: str, data: dict[str, Any]
    ) -> Any:
        """Post a sock command using only the local request path."""
        await self.activate(device, _reset_legacy_phase=False)
        endpoint = f"/dsns/{device}/properties/{command}/datapoints.json"
        response = await self._operation_request("post_command", "POST", endpoint, data)
        self._legacy_failure_reported = False
        return response
