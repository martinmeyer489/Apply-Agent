"""Property-based tests for the listing record validation utility.

Property 3: Required-Field Record Skip Validation
Validates: Requirements 2.6
"""

from hypothesis import given, strategies as st

from src.utils.validation import REQUIRED_FIELDS, validate_listing_record

# Values that always constitute a non-empty, present field.
non_empty_text = st.text(min_size=1).filter(lambda s: s.strip() != "")

# Values that represent an "empty" field per the validator's semantics
# (absent handled separately via missing keys).
empty_values = st.one_of(st.none(), st.just(""), st.just("   "))

# A dict of arbitrary extra, non-required keys/values that should not
# influence validation outcome.
extra_fields = st.dictionaries(
    keys=st.text(min_size=1, max_size=10).filter(lambda s: s not in REQUIRED_FIELDS),
    values=st.one_of(st.text(), st.integers(), st.none()),
    max_size=5,
)


@given(extra=extra_fields)
def test_all_required_fields_present_is_valid(extra):
    """**Validates: Requirements 2.6**

    A record with all three required fields present and non-empty is
    always valid, with an empty missing-fields list.
    """
    record = dict(extra)
    for field in REQUIRED_FIELDS:
        record[field] = f"value-for-{field}"

    is_valid, missing = validate_listing_record(record)

    assert is_valid is True
    assert missing == []


@given(
    missing_subset=st.lists(
        st.sampled_from(REQUIRED_FIELDS), min_size=1, max_size=len(REQUIRED_FIELDS), unique=True
    ),
    missing_style=st.sampled_from(["absent", "none", "empty", "whitespace"]),
    extra=extra_fields,
)
def test_missing_subset_is_invalid_with_exact_missing_list(missing_subset, missing_style, extra):
    """**Validates: Requirements 2.6**

    A record missing any non-empty subset of the required fields is
    invalid, and the returned missing-fields list exactly matches the
    set of missing fields (in REQUIRED_FIELDS order).
    """
    record = dict(extra)

    for field in REQUIRED_FIELDS:
        if field in missing_subset:
            if missing_style == "absent":
                continue  # do not set the key at all
            elif missing_style == "none":
                record[field] = None
            elif missing_style == "empty":
                record[field] = ""
            else:  # whitespace
                record[field] = "   "
        else:
            record[field] = f"value-for-{field}"

    is_valid, missing = validate_listing_record(record)

    expected_missing = [f for f in REQUIRED_FIELDS if f in missing_subset]

    assert is_valid is False
    assert missing == expected_missing


@given(
    record=st.dictionaries(
        keys=st.text(max_size=15),
        values=st.one_of(
            st.none(),
            st.text(max_size=20),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.lists(st.integers(), max_size=3),
        ),
        max_size=8,
    )
)
def test_never_raises_for_arbitrary_dict(record):
    """**Validates: Requirements 2.6**

    The function never raises for arbitrary dict inputs, regardless of
    which keys/values are present, and always returns a (bool, list) pair.
    """
    is_valid, missing = validate_listing_record(record)

    assert isinstance(is_valid, bool)
    assert isinstance(missing, list)
    assert all(field in REQUIRED_FIELDS for field in missing)


def test_empty_dict_is_invalid_with_all_missing():
    """An empty record is missing all required fields."""
    is_valid, missing = validate_listing_record({})

    assert is_valid is False
    assert missing == list(REQUIRED_FIELDS)
