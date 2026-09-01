"""Unit tests for `resolve_location` (Task 12.2).

Exercises location resolution against a fake `spark` stub (no real
SparkSession is available outside a Databricks runtime):

1. Known city match (case-insensitive)
2. Known postal code match
3. Unknown location -> None
4. Empty string -> ValueError
5. 200-char string (boundary, valid) vs 201-char string (invalid)

Validates: Requirements 6.5, 6.6, 6.7
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


GEOCODE_ROWS = [("berlin", "10115", 52.52, 13.405)]


def test_resolve_location_known_city_match():
    """Known city name (case-insensitive) resolves to its coordinates."""
    spark = FakeGeocodeSpark(GEOCODE_ROWS)
    result = resolve_location(spark, "BeRlIn")
    assert result == {"city_name": "berlin", "latitude": 52.52, "longitude": 13.405}


def test_resolve_location_known_postal_code_match():
    """Known postal code resolves to the expected city/coordinates dict."""
    spark = FakeGeocodeSpark(GEOCODE_ROWS)
    result = resolve_location(spark, "10115")
    assert result == {"city_name": "berlin", "latitude": 52.52, "longitude": 13.405}


def test_resolve_location_unknown_location_returns_none():
    """No matching row -> resolve_location returns None."""
    spark = FakeGeocodeSpark(GEOCODE_ROWS)
    result = resolve_location(spark, "Atlantis")
    assert result is None


def test_resolve_location_empty_string_raises_value_error():
    """Empty input fails validation and raises ValueError."""
    spark = FakeGeocodeSpark(GEOCODE_ROWS)
    with pytest.raises(ValueError):
        resolve_location(spark, "")


def test_resolve_location_200_char_string_is_valid_no_match():
    """Exactly 200 characters is a valid length (boundary); with no
    matching row, this returns None rather than raising."""
    spark = FakeGeocodeSpark(GEOCODE_ROWS)
    result = resolve_location(spark, "x" * 200)
    assert result is None


def test_resolve_location_201_char_string_raises_value_error():
    """201 characters exceeds the max length and raises ValueError."""
    spark = FakeGeocodeSpark(GEOCODE_ROWS)
    with pytest.raises(ValueError):
        resolve_location(spark, "x" * 201)
