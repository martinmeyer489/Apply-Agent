"""User location resolution against the offline Geocode_Lookup dataset.

Resolves a user-submitted home location string (city name or postal code)
to latitude/longitude coordinates by querying `job_agent.ops.geocode_lookup`
via the SQL warehouse. No outbound network requests are issued.

Validates: Requirements 6.1, 6.5, 6.6, 6.7, 6.8
"""

from __future__ import annotations

from typing import Optional

from src.utils.input_validation import validate_location_input

GEOCODE_LOOKUP_TABLE = "job_agent.ops.geocode_lookup"

RESOLVE_LOCATION_QUERY = f"""
SELECT city_name, latitude, longitude
FROM {GEOCODE_LOOKUP_TABLE}
WHERE LOWER(city_name) = LOWER(:input) OR postal_code = :input
LIMIT 1
"""


def resolve_location(spark, input_text: str) -> Optional[dict]:
    """Resolve a home location input to a city name and coordinates.

    Validates `input_text` (1-200 characters, Requirement 6.6), then queries
    `job_agent.ops.geocode_lookup` via the SQL warehouse, matching on a
    case-insensitive city name or an exact postal code (Requirement 6.5).

    Args:
        spark: The active SparkSession (connected to the SQL warehouse).
        input_text: The user-submitted home location string (city name or
            postal code).

    Returns:
        A dict with keys `city_name`, `latitude`, `longitude` for the first
        matching row, or `None` if no entry matches (Requirement 6.7).

    Raises:
        ValueError: If `input_text` fails validation (empty or >200 chars),
            with the validation error message (Requirement 6.6).
    """
    is_valid, message = validate_location_input(input_text)
    if not is_valid:
        raise ValueError(message)

    result_df = spark.sql(RESOLVE_LOCATION_QUERY, args={"input": input_text})
    rows = result_df.collect()

    if not rows:
        return None

    row = rows[0]
    return {
        "city_name": row.city_name,
        "latitude": row.latitude,
        "longitude": row.longitude,
    }
