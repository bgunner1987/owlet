"""Compatibility layer for resilient Owlet sock property handling."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from pyowletapi.const import PROPERTIES, VITALS_2, VITALS_3
from pyowletapi.sock import PropertiesDict
from pyowletapi.sock import Sock as PyOwletSock

_LOGGER = logging.getLogger(__name__)

VITALS_STALE_TIMEOUT_SECONDS = 60
INVALID_VITALS_WARNING_POLLS = 3

_JSON_INPUT_TYPES = (str, bytes, bytearray)
_VITAL_PROPERTY_KEYS = frozenset(
    description
    for property_type, mappings in VITALS_3.items()
    for description in mappings
    if property_type != "other"
) | {"last_updated"}
_VITAL_RAW_KEYS = frozenset(
    raw_key
    for property_type, mappings in VITALS_3.items()
    for raw_key in mappings.values()
    if property_type != "other"
)


def _decode_json_object(value: object) -> dict[str, Any] | None:
    """Decode a non-empty JSON object without exposing its contents."""
    if not isinstance(value, _JSON_INPUT_TYPES) or not value.strip():
        return None

    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return None

    return decoded if isinstance(decoded, dict) else None


class Sock(PyOwletSock):
    """Harden pyowletapi Sock against temporarily malformed device data."""

    def __init__(self, api: Any, data: dict[str, Any]) -> None:
        """Initialise the compatibility layer and its bounded vitals cache."""
        super().__init__(api, data)
        self._last_valid_vitals: dict[str, Any] | None = None
        self._last_valid_vitals_at: float | None = None
        self._invalid_vitals_polls = 0
        self._invalid_vitals_warning_emitted = False
        self._clock: Callable[[], float] = time.monotonic

    def _normalise_base_properties(self) -> dict[str, Any]:
        """Normalise non-vital properties independently from REAL_TIME_VITALS."""
        properties: dict[str, Any] = {}
        for converter, mappings in PROPERTIES.items():
            for description, raw_key in mappings.items():
                raw_property = self._raw_properties.get(raw_key)
                if not isinstance(raw_property, Mapping):
                    continue

                value = raw_property.get("value")
                if value is None:
                    continue

                try:
                    properties[description] = converter(value)
                except (TypeError, ValueError):
                    _LOGGER.debug("Ignoring invalid Owlet property %s", description)
        return properties

    def _normalise_v2_properties(self) -> dict[str, Any]:
        """Normalise V2 properties while ignoring malformed individual values."""
        properties: dict[str, Any] = {}
        for converter, mappings in VITALS_2.items():
            for description, raw_key in mappings.items():
                raw_property = self._raw_properties.get(raw_key)
                if not isinstance(raw_property, Mapping):
                    continue

                value = raw_property.get("value")
                if value is None:
                    continue

                try:
                    properties[description] = converter(value)
                except (TypeError, ValueError):
                    _LOGGER.debug("Ignoring invalid Owlet V2 property %s", description)
        return properties

    def _read_vitals(self) -> tuple[dict[str, Any] | None, str]:
        """Return decoded V3 vitals and a safe reason when unavailable."""
        raw_vitals = self._raw_properties.get("REAL_TIME_VITALS")
        if not isinstance(raw_vitals, Mapping):
            return None, "missing property"

        value = raw_vitals.get("value")
        if value is None:
            return None, "null value"
        if isinstance(value, _JSON_INPUT_TYPES) and not value.strip():
            return None, "empty value"

        vitals = _decode_json_object(value)
        if vitals is None:
            return None, "invalid JSON object"
        if not _VITAL_RAW_KEYS.intersection(vitals):
            return None, "no recognised measurements"
        return vitals, ""

    def _normalise_v3_properties(self, vitals: dict[str, Any]) -> dict[str, Any]:
        """Normalise each V3 vital without allowing one bad field to fail a poll."""
        properties = dict.fromkeys(_VITAL_PROPERTY_KEYS)

        for converter, mappings in VITALS_3.items():
            if converter == "other":
                continue

            for description, raw_key in mappings.items():
                value = vitals.get(raw_key)
                if value is None:
                    continue

                try:
                    properties[description] = (
                        value if description == "base_station_on" else converter(value)
                    )
                except (TypeError, ValueError):
                    _LOGGER.debug("Ignoring invalid Owlet vital field %s", description)

        raw_vitals = self._raw_properties.get("REAL_TIME_VITALS")
        if isinstance(raw_vitals, Mapping):
            updated_at = raw_vitals.get("data_updated_at")
            if isinstance(updated_at, str):
                try:
                    properties["last_updated"] = (
                        datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ")
                        .replace(tzinfo=UTC)
                        .strftime("%Y/%m/%d %H:%M:%S")
                    )
                except ValueError:
                    _LOGGER.debug("Ignoring invalid Owlet vitals timestamp")

        return properties

    def _record_invalid_vitals(self, reason: str) -> None:
        """Log transient failures at debug and one warning when they persist."""
        self._invalid_vitals_polls += 1
        if (
            self._invalid_vitals_polls >= INVALID_VITALS_WARNING_POLLS
            and not self._invalid_vitals_warning_emitted
        ):
            _LOGGER.warning(
                "Owlet real-time vitals have been invalid for %s consecutive polls; "
                "cached values expire after %s seconds",
                self._invalid_vitals_polls,
                VITALS_STALE_TIMEOUT_SECONDS,
            )
            self._invalid_vitals_warning_emitted = True
        elif self._invalid_vitals_polls < INVALID_VITALS_WARNING_POLLS:
            _LOGGER.debug(
                "Ignoring temporary Owlet real-time vitals update (%s; poll %s)",
                reason,
                self._invalid_vitals_polls,
            )

    def _cached_or_unavailable_vitals(self) -> dict[str, Any]:
        """Return recent cached vitals or explicit unknown values once stale."""
        properties = dict.fromkeys(_VITAL_PROPERTY_KEYS)
        if (
            self._last_valid_vitals is not None
            and self._last_valid_vitals_at is not None
            and self._clock() - self._last_valid_vitals_at
            <= VITALS_STALE_TIMEOUT_SECONDS
        ):
            properties.update(self._last_valid_vitals)
        return properties

    async def _normalise_properties(self) -> dict[str, Any]:
        """Normalise device data without failing the whole poll on bad vitals."""
        properties = self._normalise_base_properties()

        if self._version == 3:
            vitals, reason = self._read_vitals()
            if vitals is None:
                self._record_invalid_vitals(reason)
                properties.update(self._cached_or_unavailable_vitals())
            else:
                vital_properties = self._normalise_v3_properties(vitals)
                self._last_valid_vitals = vital_properties.copy()
                self._last_valid_vitals_at = self._clock()
                if self._invalid_vitals_polls:
                    _LOGGER.debug(
                        "Owlet real-time vitals recovered after %s invalid polls",
                        self._invalid_vitals_polls,
                    )
                self._invalid_vitals_polls = 0
                self._invalid_vitals_warning_emitted = False
                properties.update(vital_properties)

        elif self._version == 2:
            properties.update(self._normalise_v2_properties())

        return properties

    async def _check_version(self) -> None:
        """Detect the sock version while tolerating a missing V3 vitals property."""
        if "REAL_TIME_VITALS" in self._raw_properties:
            self._version = 3
        elif "CHARGE_STATUS" in self._raw_properties:
            self._version = 2
        elif self._version not in (2, 3):
            self._version = 3 if "oem_sock_version" in self._raw_properties else 0

    async def _check_revision(self) -> None:
        """Read the revision only when oem_sock_version is a valid JSON object."""
        raw_revision = self._raw_properties.get("oem_sock_version")
        if not isinstance(raw_revision, Mapping):
            _LOGGER.debug("Owlet sock revision property is missing")
            return

        revision_data = _decode_json_object(raw_revision.get("value"))
        if revision_data is None:
            _LOGGER.debug("Ignoring invalid Owlet sock revision data")
            return

        revision = revision_data.get("rev")
        if isinstance(revision, bool) or not isinstance(revision, int):
            _LOGGER.debug("Ignoring Owlet sock revision without an integer revision")
            return
        self._revision = revision

    async def update_properties(self) -> PropertiesDict:
        """Update properties while rechecking version and revision safely."""
        payload = await self._api.get_properties(self.serial)
        raw_properties = payload.get("response", {})
        self._raw_properties = (
            raw_properties if isinstance(raw_properties, dict) else {}
        )

        await self._check_version()
        if self._revision is None and self._version == 3:
            await self._check_revision()
        self._properties = await self._normalise_properties()

        response: PropertiesDict = {
            "raw_properties": self._raw_properties,
            "properties": self._properties,
        }
        if "tokens" in payload:
            response["tokens"] = payload["tokens"]
        return response
