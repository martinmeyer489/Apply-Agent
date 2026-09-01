"""Integration test for real geocode lookup resolution against `ops.geocode_lookup`.

This is a **live integration test**: it exercises
`src.agent.location_resolver.resolve_location()` against a real Spark
session connected to Unity Catalog's `job_agent.ops.geocode_lookup` table
(loaded from `data/geocode_lookup/geocode_lookup.csv` by
`notebooks/00a_load_reference_data.py`, task 4.1). It cannot run in a plain
local Python environment (no `pyspark`, no SQL warehouse) and is
automatically skipped there.

When run against a live workspace, for each of a few known cities that are
verified (in this test module itself) to actually exist in the bundled
`data/geocode_lookup/geocode_lookup.csv`, it:

1. Queries `resolve_location(spark, city_name)`.
2. Asserts a match is returned (Req 6.5 — city name lookup).
3. Asserts the returned latitude/longitude match the CSV's row for that
   city within a small tolerance.

Requirements: 6.5
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.integration._databricks_env import SKIP_REASON, has_databricks_environment

pytestmark = pytest.mark.integration

GEOCODE_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "geocode_lookup" / "geocode_lookup.csv"
)
COORDINATE_TOLERANCE = 0.01

# City names as they appear in `geocode_lookup.csv`'s `city_name` column
# (some rows have numeric prefixes, e.g. "31:london" / "32:london", so we
# match by substring rather than requiring an exact CSV cell value; what
# matters for this test is the *lookup input* a user would type, e.g.
# "london", "new york", "berlin").
KNOWN_CITIES = ["berlin", "london", "new york"]


def _first_csv_row_for_city(city_name: str) -> dict:
    """Return the first `geocode_lookup.csv` row whose `city_name` contains `city_name`."""
    with open(GEOCODE_CSV_PATH, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if city_name in row["city_name"].lower():
                return row
    raise AssertionError(f"no geocode_lookup.csv row found for city '{city_name}'")


class TestGeocodeLookupCsvFixture:
    """Sanity-check the CSV fixture itself (runs locally, no Databricks needed)."""

    @pytest.mark.parametrize("city_name", KNOWN_CITIES)
    def test_known_city_exists_in_bundled_csv(self, city_name):
        row = _first_csv_row_for_city(city_name)
        assert float(row["latitude"])
        assert float(row["longitude"])


@pytest.mark.skipif(not has_databricks_environment(), reason=SKIP_REASON)
class TestGeocodeLookupIntegration:
    """Resolve known cities against the real `ops.geocode_lookup` Delta table."""

    @pytest.fixture(scope="class")
    def spark(self):
        from pyspark.sql import SparkSession

        return SparkSession.builder.appName("job-agent-geocode-test").getOrCreate()

    @pytest.mark.parametrize("city_name", KNOWN_CITIES)
    def test_resolve_known_city_matches_csv_coordinates(self, spark, city_name):
        """A live run of this test would:

        1. Call `resolve_location(spark, city_name)` against the real
           `ops.geocode_lookup` table (Req 6.5).
        2. Assert a match is returned (not `None`).
        3. Assert the returned latitude/longitude are within
           `COORDINATE_TOLERANCE` degrees of the corresponding row in the
           bundled `data/geocode_lookup/geocode_lookup.csv` fixture — the
           source of truth `notebooks/00a_load_reference_data.py` loads
           `ops.geocode_lookup` from.
        """
        from src.agent.location_resolver import resolve_location

        expected_row = _first_csv_row_for_city(city_name)
        expected_lat = float(expected_row["latitude"])
        expected_lng = float(expected_row["longitude"])

        resolved = resolve_location(spark, city_name)

        assert resolved is not None, f"expected a match for known city '{city_name}'"
        assert abs(resolved["latitude"] - expected_lat) <= COORDINATE_TOLERANCE
        assert abs(resolved["longitude"] - expected_lng) <= COORDINATE_TOLERANCE
