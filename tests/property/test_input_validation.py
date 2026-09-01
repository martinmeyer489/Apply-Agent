"""Property-based tests for location and commute radius input validation.

Property 17: Commute Radius and Location Input Validation
Validates: Requirements 6.3, 6.4, 6.6
"""

from hypothesis import given, strategies as st

from src.utils.input_validation import (
    MAX_COMMUTE_RADIUS_KM,
    MAX_LOCATION_LENGTH,
    MIN_COMMUTE_RADIUS_KM,
    MIN_LOCATION_LENGTH,
    validate_commute_radius,
    validate_location_input,
)


@given(text=st.text(max_size=400))
def test_validate_location_input_matches_specification(text):
    """**Validates: Requirements 6.6**

    For arbitrary text strings, validate_location_input returns valid=True
    iff 1 <= len(text) <= 200; otherwise it returns invalid with a
    non-empty message.
    """
    is_valid, message = validate_location_input(text)

    expected_valid = MIN_LOCATION_LENGTH <= len(text) <= MAX_LOCATION_LENGTH

    assert is_valid == expected_valid
    assert isinstance(message, str)
    assert len(message) > 0


@given(
    value=st.integers(min_value=-1_000_000, max_value=1_000_000)
    | st.none()
)
def test_validate_commute_radius_matches_specification(value):
    """**Validates: Requirements 6.3, 6.4**

    For arbitrary int values (or None), validate_commute_radius returns
    valid=True iff 1 <= value <= 200; otherwise it returns invalid with a
    non-empty message.
    """
    is_valid, message = validate_commute_radius(value)

    expected_valid = (
        value is not None
        and MIN_COMMUTE_RADIUS_KM <= value <= MAX_COMMUTE_RADIUS_KM
    )

    assert is_valid == expected_valid
    assert isinstance(message, str)
    assert len(message) > 0

    if not is_valid:
        assert "1" in message
        assert "200" in message
