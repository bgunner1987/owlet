"""Tests for the local Owlet Sock compatibility layer."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from custom_components.owlet.sock import (
    INVALID_VITALS_WARNING_POLLS,
    VITALS_STALE_TIMEOUT_SECONDS,
    Sock,
)

_MISSING = object()
_VALID_VITALS = (
    '{"ox":98,"hr":123,"bat":80,"btt":600,"rsi":34,"st":35,'
    '"chg":0,"ss":1,"oxta":97,"bso":1}'
)
_VALID_REVISION = '{"rev":3}'


class FakeAPI:
    """Return a deterministic sequence of device property responses."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)

    async def get_properties(self, _serial: str) -> dict[str, Any]:
        return self.responses.pop(0)


def _properties_response(
    vitals: object = _VALID_VITALS,
    *,
    revision: object = _VALID_REVISION,
    low_battery: object = 0,
) -> dict[str, Any]:
    raw_properties: dict[str, Any] = {
        "LOW_BATT_ALRT": {"value": low_battery},
    }
    if revision is not _MISSING:
        raw_properties["oem_sock_version"] = {"value": revision}
    if vitals is not _MISSING:
        raw_properties["REAL_TIME_VITALS"] = {
            "value": vitals,
            "data_updated_at": "2026-09-01T10:00:00Z",
        }
    return {"response": raw_properties}


def _make_sock(*responses: dict[str, Any]) -> Sock:
    return Sock(FakeAPI(*responses), {"dsn": "TEST-SOCK"})


@pytest.mark.asyncio
async def test_valid_real_time_vitals() -> None:
    """Valid V3 vitals are normalised as before."""
    sock = _make_sock(_properties_response())

    result = await sock.update_properties()

    assert result["properties"]["heart_rate"] == 123.0
    assert result["properties"]["oxygen_saturation"] == 98.0
    assert result["properties"]["last_updated"] == "2026/09/01 10:00:00"


@pytest.mark.asyncio
async def test_null_real_time_vitals(caplog: pytest.LogCaptureFixture) -> None:
    """A single null poll is an expected temporary state, not an error."""
    sock = _make_sock(_properties_response(None, low_battery=1))

    with caplog.at_level(logging.DEBUG):
        result = await sock.update_properties()

    assert result["properties"]["heart_rate"] is None
    assert result["properties"]["low_battery_alert"] is True
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_empty_real_time_vitals() -> None:
    """An empty V3 value is ignored safely."""
    sock = _make_sock(_properties_response(""))

    result = await sock.update_properties()

    assert result["properties"]["heart_rate"] is None


@pytest.mark.asyncio
async def test_invalid_real_time_vitals_json() -> None:
    """Malformed JSON does not fail the entire property update."""
    sock = _make_sock(_properties_response("{invalid json"))

    result = await sock.update_properties()

    assert result["properties"]["oxygen_saturation"] is None


@pytest.mark.asyncio
async def test_unexpected_real_time_vitals_json_type() -> None:
    """A valid JSON value that is not an object is discarded."""
    sock = _make_sock(_properties_response("[]"))

    result = await sock.update_properties()

    assert result["properties"]["heart_rate"] is None


@pytest.mark.asyncio
async def test_valid_vitals_after_null_poll() -> None:
    """A valid poll after null data updates values normally."""
    sock = _make_sock(_properties_response(None), _properties_response())

    first = await sock.update_properties()
    second = await sock.update_properties()

    assert first["properties"]["heart_rate"] is None
    assert second["properties"]["heart_rate"] == 123.0


@pytest.mark.asyncio
async def test_multiple_null_polls_keep_recent_vitals() -> None:
    """Several short null polls retain the last valid measurements."""
    sock = _make_sock(
        _properties_response(),
        _properties_response(None),
        _properties_response(None),
    )
    now = [0.0]
    sock._clock = lambda: now[0]

    valid = await sock.update_properties()
    now[0] = 10
    first_null = await sock.update_properties()
    now[0] = 20
    second_null = await sock.update_properties()

    assert valid["properties"]["heart_rate"] == 123.0
    assert first_null["properties"]["heart_rate"] == 123.0
    assert second_null["properties"]["heart_rate"] == 123.0
    assert second_null["properties"]["last_updated"] == "2026/09/01 10:00:00"


@pytest.mark.asyncio
async def test_stale_timeout_marks_vitals_unknown() -> None:
    """Cached vitals become unknown after the bounded stale timeout."""
    sock = _make_sock(_properties_response(), _properties_response(None))
    now = [0.0]
    sock._clock = lambda: now[0]

    await sock.update_properties()
    now[0] = VITALS_STALE_TIMEOUT_SECONDS + 1
    stale = await sock.update_properties()

    assert stale["properties"]["heart_rate"] is None
    assert stale["properties"]["oxygen_saturation"] is None
    assert stale["properties"]["last_updated"] is None


@pytest.mark.asyncio
async def test_missing_real_time_vitals() -> None:
    """A missing V3 property still exposes stable unknown keys."""
    sock = _make_sock(_properties_response(_MISSING))

    result = await sock.update_properties()

    assert sock.version == 3
    assert "heart_rate" in result["properties"]
    assert result["properties"]["heart_rate"] is None


@pytest.mark.asyncio
async def test_null_sock_revision() -> None:
    """A null sock revision does not block otherwise valid vitals."""
    sock = _make_sock(_properties_response(revision=None))

    result = await sock.update_properties()

    assert sock.revision is None
    assert result["properties"]["heart_rate"] == 123.0


@pytest.mark.asyncio
async def test_missing_sock_revision_property() -> None:
    """A missing revision property does not block otherwise valid vitals."""
    sock = _make_sock(_properties_response(revision=_MISSING))

    result = await sock.update_properties()

    assert sock.revision is None
    assert result["properties"]["heart_rate"] == 123.0


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", ["", "{invalid json", "[]", "{}"])
async def test_invalid_sock_revision(revision: str) -> None:
    """Empty, malformed, non-object, and keyless revision data are ignored."""
    sock = _make_sock(_properties_response(revision=revision))

    result = await sock.update_properties()

    assert sock.revision is None
    assert result["properties"]["heart_rate"] == 123.0


@pytest.mark.asyncio
async def test_repeated_invalid_vitals_warn_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Persistent invalid data produces one warning without log spam."""
    responses = [
        _properties_response(None) for _ in range(INVALID_VITALS_WARNING_POLLS + 2)
    ]
    sock = _make_sock(*responses)

    with caplog.at_level(logging.WARNING):
        for _response in responses:
            await sock.update_properties()

    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
