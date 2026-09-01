"""Property-based test for the commute-radius candidate filter.

Property 19: Commute Filter Soundness
Validates: Requirements 7.6

The real filtering logic lives inside the Matching Agent, which calls the
`job_agent.gold.compute_commute_distance` UC Function (a haversine distance
calculation) once per candidate returned by Vector Search and excludes any
candidate whose distance from the User_Profile home coordinates exceeds the
Commute_Radius. This test models that filter in pure Python, reusing the
already-implemented `compute_commute_distance` helper directly so the
distance math mirrors the UC Function exactly.
"""

from hypothesis import given, strategies as st

from src.utils.haversine import compute_commute_distance


def filter_by_commute_radius(
    candidates: list, home_lat: float, home_lon: float, radius_km: float
) -> list:
    """Filter candidates to those within `radius_km` of the home coordinates.

    Mirrors the Matching Agent's per-candidate `compute_commute_distance`
    call followed by exclusion of any candidate whose distance exceeds the
    Commute_Radius (Requirement 7.6).

    Args:
        candidates: list of dicts, each with `latitude` and `longitude`.
        home_lat: User_Profile home latitude.
        home_lon: User_Profile home longitude.
        radius_km: Commute_Radius in kilometres.

    Returns:
        The subset of `candidates` whose computed distance from
        (home_lat, home_lon) is <= radius_km.
    """
    kept = []
    for candidate in candidates:
        distance = compute_commute_distance(
            home_lat, home_lon, candidate["latitude"], candidate["longitude"]
        )
        if distance <= radius_km:
            kept.append(candidate)
    return kept


# --- Strategies -------------------------------------------------------------

_lat = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)
_lon = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)

_candidate = st.fixed_dictionaries(
    {
        "listing_id": st.text(min_size=1, max_size=10),
        "latitude": _lat,
        "longitude": _lon,
    }
)

_candidates = st.lists(_candidate, min_size=0, max_size=25)
_radius = st.floats(min_value=0.0, max_value=20000.0, allow_nan=False, allow_infinity=False)


@given(candidates=_candidates, home_lat=_lat, home_lon=_lon, radius_km=_radius)
def test_kept_candidates_are_within_radius(candidates, home_lat, home_lon, radius_km):
    """**Validates: Requirements 7.6**

    Every candidate present in the filtered result has a computed distance
    from the home coordinates that is <= the commute radius -- the filter
    never lets an out-of-range candidate through.
    """
    result = filter_by_commute_radius(candidates, home_lat, home_lon, radius_km)

    for candidate in result:
        distance = compute_commute_distance(
            home_lat, home_lon, candidate["latitude"], candidate["longitude"]
        )
        assert distance <= radius_km


@given(candidates=_candidates, home_lat=_lat, home_lon=_lon, radius_km=_radius)
def test_excluded_candidates_are_beyond_radius(candidates, home_lat, home_lon, radius_km):
    """**Validates: Requirements 7.6**

    Every candidate NOT present in the filtered result has a computed
    distance from the home coordinates that exceeds the commute radius --
    the filter never wrongly excludes an in-range candidate (soundness in
    both directions, i.e. no misclassification).
    """
    result = filter_by_commute_radius(candidates, home_lat, home_lon, radius_km)
    kept_ids = {id(c) for c in result}

    for candidate in candidates:
        if id(candidate) not in kept_ids:
            distance = compute_commute_distance(
                home_lat, home_lon, candidate["latitude"], candidate["longitude"]
            )
            assert distance > radius_km


@given(candidates=_candidates, home_lat=_lat, home_lon=_lon, radius_km=_radius)
def test_filter_never_grows_the_candidate_set(candidates, home_lat, home_lon, radius_km):
    """**Validates: Requirements 7.6**

    Sanity property: filtering can only remove candidates, never fabricate
    new ones, and always preserves relative order of survivors.
    """
    result = filter_by_commute_radius(candidates, home_lat, home_lon, radius_km)

    assert len(result) <= len(candidates)

    # result must be a subsequence (order-preserving subset) of candidates,
    # matched by identity since candidate dicts may compare equal.
    result_idx = 0
    for candidate in candidates:
        if result_idx < len(result) and candidate is result[result_idx]:
            result_idx += 1
    assert result_idx == len(result)
