"""Sanity unit tests for `resolve_location` using a fake `spark` stub.

These tests exercise the pure logic of `resolve_location` (validation,
row-to-dict mapping, no-match handling) without a real SparkSession, which
is unavailable outside a Databricks runtime. Full integration testing
requires a real Spark session on Databricks.

Validates: Requirements 6.1, 6.5, 6.6, 6.7, 6.8
"""

import pytest

from src.agent.location_resolver import resolve_location


class _FakeRow:
    def __init__(self, city_name, latitude, longitude):
        self.city_name = city_name
        self.latitude = latitude
        self.longitude = longitude


class _FakeResultDataFrame:
    def __init__(self, rows):
        self._rows = rows

    def collect(self):
        return self._rows


class FakeGeocodeSpark:
    """Minimal fake SparkSession supporting `spark.sql(query, args=...)`
    for the geocode lookup query used by `resolve_location`."""

    def __init__(self, geocode_rows):
        # geocode_rows: list of (city_name, postal_code, latitude, longitude)
        self._geocode_rows = geocode_rows

    def sql(self, query, args=None):
        args = args or {}
        input_value = args.get("input")
        for city_name, postal_code, latitude, longitude in self._geocode_rows:
            if (
                city_name is not None
                and city_name.lower() == str(input_value).lower()
            ) or (postal_code is not None and postal_code == input_value):
                return _FakeResultDataFrame(
                    [_FakeRow(city_name, latitude, longitude)]
                )
        return _FakeResultDataFrame([])


def test_resolve_location_city_name_match():
    spark = FakeGeocodeSpark([("berlin", "10115", 52.52, 13.405)])
    result = resolve_location(spark, "Berlin")
    assert result == {"city_name": "berlin", "latitude": 52.52, "longitude": 13.405}


def test_resolve_location_postal_code_match():
    spark = FakeGeocodeSpark([("berlin", "10115", 52.52, 13.405)])
    result = resolve_location(spark, "10115")
    assert result == {"city_name": "berlin", "latitude": 52.52, "longitude": 13.405}


def test_resolve_location_no_match_returns_none():
    spark = FakeGeocodeSpark([("berlin", "10115", 52.52, 13.405)])
    result = resolve_location(spark, "Atlantis")
    assert result is None


def test_resolve_location_empty_input_raises_value_error():
    spark = FakeGeocodeSpark([])
    with pytest.raises(ValueError):
        resolve_location(spark, "")


def test_resolve_location_too_long_input_raises_value_error():
    spark = FakeGeocodeSpark([])
    with pytest.raises(ValueError):
        resolve_location(spark, "x" * 201)
