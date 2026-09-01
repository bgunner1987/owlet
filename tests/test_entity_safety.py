"""Tests for safe entity access when Owlet properties are absent."""

from __future__ import annotations

from types import SimpleNamespace

from custom_components.owlet.binary_sensor import OwletAwakeSensor, OwletBinarySensor
from custom_components.owlet.sensor import OwletSensor, OwletSleepSensor
from custom_components.owlet.switch import OwletBaseSwitch


def _entity(key: str) -> SimpleNamespace:
    return SimpleNamespace(
        sock=SimpleNamespace(properties={}),
        entity_description=SimpleNamespace(key=key),
    )


def test_sensor_missing_property_is_unknown() -> None:
    """A missing sensor property returns unknown instead of raising KeyError."""
    assert OwletSensor.native_value.fget(_entity("heart_rate")) is None


def test_sleep_sensor_missing_property_is_unknown() -> None:
    """A missing sleep state returns unknown instead of raising KeyError."""
    assert OwletSleepSensor.native_value.fget(_entity("sleep_state")) is None


def test_binary_sensor_missing_property_is_unknown() -> None:
    """A missing binary property returns unknown instead of raising KeyError."""
    assert OwletBinarySensor.is_on.fget(_entity("charging")) is None


def test_awake_sensor_missing_property_is_unknown() -> None:
    """A missing sleep state does not make the awake sensor appear on."""
    assert OwletAwakeSensor.is_on.fget(_entity("sleep_state")) is None


def test_switch_missing_property_is_unknown() -> None:
    """A missing switch property returns unknown instead of raising KeyError."""
    assert OwletBaseSwitch.is_on.fget(_entity("base_station_on")) is None
