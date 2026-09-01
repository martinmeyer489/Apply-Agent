"""Haversine distance utility.

Computes the great-circle distance between two geographic coordinate pairs.
Used by the `compute_commute_distance` UC Function tool and the Matching
Agent's commute-radius filtering logic (Requirement 7.6).
"""

import math

EARTH_RADIUS_KM = 6371.0


def compute_commute_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle (haversine) distance between two coordinates.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        The distance in kilometres, rounded to 1 decimal place.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(EARTH_RADIUS_KM * c, 1)
