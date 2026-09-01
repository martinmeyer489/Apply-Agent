"""Property-based tests for haversine distance computation.

**Validates: Requirements 7.6**
"""

from hypothesis import given, strategies as st

from utils.haversine import compute_commute_distance

lat_strategy = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)
lon_strategy = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)


@given(lat_strategy, lon_strategy, lat_strategy, lon_strategy)
def test_distance_is_non_negative(lat1: float, lon1: float, lat2: float, lon2: float) -> None:
    """Property 18 (part 1): distance is always non-negative.

    **Validates: Requirements 7.6**
    """
    distance = compute_commute_distance(lat1, lon1, lat2, lon2)
    assert distance >= 0.0


@given(lat_strategy, lon_strategy, lat_strategy, lon_strategy)
def test_distance_is_symmetric(lat1: float, lon1: float, lat2: float, lon2: float) -> None:
    """Property 18 (part 2): distance(a, b) == distance(b, a).

    **Validates: Requirements 7.6**
    """
    forward = compute_commute_distance(lat1, lon1, lat2, lon2)
    backward = compute_commute_distance(lat2, lon2, lat1, lon1)
    assert forward == backward


@given(lat_strategy, lon_strategy)
def test_distance_to_self_is_zero(lat1: float, lon1: float) -> None:
    """Property 18 (part 3): distance from a point to itself is 0.0.

    **Validates: Requirements 7.6**
    """
    distance = compute_commute_distance(lat1, lon1, lat1, lon1)
    assert distance == 0.0


@given(lat_strategy, lon_strategy, lat_strategy, lon_strategy)
def test_distance_is_rounded_to_one_decimal(lat1: float, lon1: float, lat2: float, lon2: float) -> None:
    """Property 18 (part 4): result is rounded to 1 decimal place.

    **Validates: Requirements 7.6**
    """
    distance = compute_commute_distance(lat1, lon1, lat2, lon2)
    assert distance == round(distance, 1)
