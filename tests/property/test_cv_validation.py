"""Property-based tests for CV file validation.

Property 13: CV File Validation
Validates: Requirements 5.1, 5.5

Note: This file is also used by tasks 1.13 (Property 14), 1.15 (Property 15),
and 11.2 (Property 16), which append additional tests here.
"""

from hypothesis import given, strategies as st

from src.utils.input_validation import (
    MAX_CV_FILE_SIZE_BYTES,
    validate_cv_file,
)

# Formats that should be accepted, in various casings.
_VALID_FORMAT_VARIANTS = st.sampled_from(
    ["PDF", "pdf", "Pdf", "DOCX", "docx", "Docx", "DocX"]
)


@given(
    file_format=st.text(max_size=20),
    file_size=st.integers(min_value=-10_000_000, max_value=10_000_000),
)
def test_validate_cv_file_matches_specification(file_format, file_size):
    """**Validates: Requirements 5.1, 5.5**

    For arbitrary file_format strings and file_size ints, validate_cv_file
    returns valid=True iff the format is PDF or DOCX (case-insensitive) AND
    0 < file_size <= 5 MB; otherwise it returns invalid with a non-empty
    message.
    """
    is_valid, message = validate_cv_file(file_format, file_size)

    expected_format_ok = (
        file_format is not None
        and file_format.strip().upper() in ("PDF", "DOCX")
    )
    expected_size_ok = 0 < file_size <= MAX_CV_FILE_SIZE_BYTES
    expected_valid = expected_format_ok and expected_size_ok

    assert is_valid == expected_valid
    assert isinstance(message, str)
    assert len(message) > 0

    if not is_valid:
        # Error message must mention the accepted formats, the max size,
        # and the non-empty requirement.
        assert "PDF" in message
        assert "DOCX" in message
        assert "5 MB" in message


@given(
    file_format=_VALID_FORMAT_VARIANTS,
    file_size=st.integers(min_value=1, max_value=MAX_CV_FILE_SIZE_BYTES),
)
def test_validate_cv_file_accepts_valid_pdf_docx_case_insensitive(file_format, file_size):
    """**Validates: Requirements 5.1**

    Valid PDF/DOCX formats (in any casing) with a size in (0, 5MB] are
    always accepted.
    """
    is_valid, message = validate_cv_file(file_format, file_size)

    assert is_valid is True
    assert len(message) > 0


@given(
    file_format=_VALID_FORMAT_VARIANTS,
    file_size=st.integers(min_value=-10_000_000, max_value=0),
)
def test_validate_cv_file_rejects_non_positive_size(file_format, file_size):
    """**Validates: Requirements 5.5**

    Zero or negative file sizes are always rejected, even with a valid
    format.
    """
    is_valid, message = validate_cv_file(file_format, file_size)

    assert is_valid is False
    assert len(message) > 0


@given(
    file_format=_VALID_FORMAT_VARIANTS,
    file_size=st.integers(
        min_value=MAX_CV_FILE_SIZE_BYTES + 1, max_value=MAX_CV_FILE_SIZE_BYTES + 10_000_000
    ),
)
def test_validate_cv_file_rejects_oversized_files(file_format, file_size):
    """**Validates: Requirements 5.5**

    Files larger than 5 MB are always rejected, even with a valid format.
    """
    is_valid, message = validate_cv_file(file_format, file_size)

    assert is_valid is False
    assert len(message) > 0


# ---------------------------------------------------------------------------
# Property 14: CV Field Bound Enforcement
# Validates: Requirements 5.4
# ---------------------------------------------------------------------------

from src.utils.cv_bounds import (
    MAX_EDUCATION_HISTORY,
    MAX_JOB_TITLE_HISTORY,
    MAX_SKILLS,
    MAX_SUMMARY_CHARS,
    MAX_YEARS_OF_EXPERIENCE,
    MIN_YEARS_OF_EXPERIENCE,
    enforce_cv_bounds,
)

_skills_strategy = st.lists(st.text(max_size=50), max_size=100)
_education_strategy = st.lists(st.text(max_size=50), max_size=40)
_job_titles_strategy = st.lists(st.text(max_size=50), max_size=60)
_years_strategy = st.integers(min_value=-1000, max_value=1000)
_summary_strategy = st.text(max_size=5000)


@given(
    skills=_skills_strategy,
    years_of_experience=_years_strategy,
    education_history=_education_strategy,
    job_title_history=_job_titles_strategy,
    qualifications_summary=_summary_strategy,
)
def test_enforce_cv_bounds_satisfies_all_bounds(
    skills,
    years_of_experience,
    education_history,
    job_title_history,
    qualifications_summary,
):
    """**Validates: Requirements 5.4**

    For arbitrary parsed CV fields with arbitrary-length lists, arbitrary
    int years, and arbitrary-length strings, the enforced output always
    satisfies all persisted bounds:
      - skills <= 50 entries
      - years_of_experience clamped to [0, 99]
      - education_history <= 20 entries
      - job_title_history <= 30 entries
      - qualifications_summary <= 2000 characters
    """
    parsed_fields = {
        "skills": skills,
        "years_of_experience": years_of_experience,
        "education_history": education_history,
        "job_title_history": job_title_history,
        "qualifications_summary": qualifications_summary,
    }

    result = enforce_cv_bounds(parsed_fields)

    assert len(result["skills"]) <= MAX_SKILLS
    assert MIN_YEARS_OF_EXPERIENCE <= result["years_of_experience"] <= MAX_YEARS_OF_EXPERIENCE
    assert len(result["education_history"]) <= MAX_EDUCATION_HISTORY
    assert len(result["job_title_history"]) <= MAX_JOB_TITLE_HISTORY
    assert len(result["qualifications_summary"]) <= MAX_SUMMARY_CHARS

    # Truncation must preserve a prefix of the original data, not arbitrary
    # reordering or dropping of elements.
    assert result["skills"] == skills[:MAX_SKILLS]
    assert result["education_history"] == education_history[:MAX_EDUCATION_HISTORY]
    assert result["job_title_history"] == job_title_history[:MAX_JOB_TITLE_HISTORY]
    assert result["qualifications_summary"] == qualifications_summary[:MAX_SUMMARY_CHARS]


@given(data=st.dictionaries(st.text(max_size=20), st.text(max_size=20), max_size=5))
def test_enforce_cv_bounds_handles_missing_fields_gracefully(data):
    """**Validates: Requirements 5.4**

    Arbitrary dicts that do not necessarily contain any of the bounded
    fields must not raise, and the result must not introduce keys that
    were not present in the input.
    """
    result = enforce_cv_bounds(data)

    assert set(result.keys()) == set(data.keys())


# ---------------------------------------------------------------------------
# Property 15: Partial Parse Warning Completeness
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------

from src.utils.cv_warnings import CV_FIELDS, get_unresolved_fields

# Values that should count as "resolved" or "unresolved" per field type.
_list_or_str_field_values = st.one_of(
    st.none(),
    st.just([]),
    st.just(""),
    st.lists(st.text(max_size=10), min_size=1, max_size=5),
    st.text(min_size=1, max_size=10),
)

_years_field_values = st.one_of(
    st.none(),
    st.integers(min_value=0, max_value=99),
)

_parsed_dict_strategy = st.fixed_dictionaries(
    {
        "skills": _list_or_str_field_values,
        "years_of_experience": _years_field_values,
        "education_history": _list_or_str_field_values,
        "job_title_history": _list_or_str_field_values,
        "qualifications_summary": _list_or_str_field_values,
    }
)


def _is_unresolved(field, value):
    if field == "years_of_experience":
        return value is None
    if value is None:
        return True
    if isinstance(value, (list, str)) and len(value) == 0:
        return True
    return False


@given(parsed=_parsed_dict_strategy)
def test_get_unresolved_fields_matches_specification(parsed):
    """**Validates: Requirements 5.6**

    For arbitrary parsed dicts with the 5 known CV fields set to a mix of
    None, empty, and non-empty values, get_unresolved_fields returns
    exactly the set of fields that are unresolved per specification:
    missing/None or empty list/string is unresolved, except
    years_of_experience which is unresolved only when None (0 is valid).
    The returned list must only ever contain known CV field names.
    """
    result = get_unresolved_fields(parsed)

    expected_unresolved = {
        field for field in CV_FIELDS if _is_unresolved(field, parsed.get(field))
    }

    assert set(result) == expected_unresolved
    assert set(result).issubset(set(CV_FIELDS))
    # No duplicates.
    assert len(result) == len(set(result))


@given(parsed=st.dictionaries(st.text(max_size=20), st.text(max_size=20), max_size=5))
def test_get_unresolved_fields_handles_missing_keys(parsed):
    """**Validates: Requirements 5.6**

    Arbitrary dicts that may not contain any of the 5 known CV fields at
    all must be treated as fully unresolved for every missing field, and
    the function must not raise or return unknown field names.
    """
    result = get_unresolved_fields(parsed)

    expected_unresolved = {
        field for field in CV_FIELDS if _is_unresolved(field, parsed.get(field))
    }

    assert set(result) == expected_unresolved
    assert set(result).issubset(set(CV_FIELDS))


# ---------------------------------------------------------------------------
# Property 16: CV Re-Upload Replacement
# Validates: Requirements 5.9
# ---------------------------------------------------------------------------

# There is no dedicated "replace CV fields in session state" function yet
# (that logic lives in the Gradio app, task 16.1, not yet implemented). This
# property test models the replacement semantics that any correct
# implementation must satisfy via a small pure helper defined inline here.

_CV_FIELD_NAMES = {
    "skills",
    "years_of_experience",
    "education_history",
    "job_title_history",
    "qualifications_summary",
    "unresolved_fields",
}


def replace_cv_fields(session_profile: dict, new_cv_fields: dict) -> dict:
    """Return a new session profile with CV fields replaced by new_cv_fields.

    All keys in session_profile that are recognized CV fields are dropped,
    then overwritten with the values from new_cv_fields. Non-CV fields
    (e.g. home_latitude, home_longitude, commute_radius_km) are preserved
    unchanged. Keys in new_cv_fields that are not recognized CV field names
    are still applied (a fresh CV parse should fully determine the CV
    fields), but recognized non-CV keys are never touched by this function.
    """
    result = {
        key: value
        for key, value in session_profile.items()
        if key not in _CV_FIELD_NAMES
    }
    result.update(new_cv_fields)
    return result


# Non-CV fields that must never be affected by a CV re-upload.
_NON_CV_KEYS = st.sampled_from(
    ["home_latitude", "home_longitude", "commute_radius_km", "profile_id", "location_name"]
)

_cv_field_keys = st.sampled_from(sorted(_CV_FIELD_NAMES))

_field_value_strategy = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=20),
    st.lists(st.text(max_size=10), max_size=5),
)

_session_profile_strategy = st.dictionaries(
    keys=st.one_of(_cv_field_keys, _NON_CV_KEYS),
    values=_field_value_strategy,
    max_size=10,
)

_new_cv_fields_strategy = st.dictionaries(
    keys=_cv_field_keys,
    values=_field_value_strategy,
    max_size=6,
)


@given(
    session_profile=_session_profile_strategy,
    new_cv_fields=_new_cv_fields_strategy,
)
def test_replace_cv_fields_overwrites_cv_fields_and_preserves_others(
    session_profile, new_cv_fields
):
    """**Validates: Requirements 5.9**

    For arbitrary session profiles containing a mix of CV and non-CV
    fields, and arbitrary new_cv_fields, replace_cv_fields must:
      (a) overwrite every CV field key present in new_cv_fields with the
          new value, with no trace of the old value remaining,
      (b) preserve every non-CV field from the original profile unchanged,
      (c) be idempotent when applying the same new_cv_fields twice.
    """
    result = replace_cv_fields(session_profile, new_cv_fields)

    # (a) every key in new_cv_fields is present with the new value; old CV
    # field values (if different from the new one) do not survive.
    for key, value in new_cv_fields.items():
        assert result[key] == value

    old_cv_keys = {
        key for key in session_profile if key in _CV_FIELD_NAMES
    }
    for key in old_cv_keys:
        if key not in new_cv_fields:
            # A CV field from the old profile that wasn't provided by the
            # new parse must not survive the replacement at all, since a
            # fresh CV upload replaces the whole CV field set.
            assert key not in result

    # (b) non-CV fields preserved unchanged.
    non_cv_items = {
        key: value
        for key, value in session_profile.items()
        if key not in _CV_FIELD_NAMES
    }
    for key, value in non_cv_items.items():
        assert result[key] == value

    # (c) idempotence: applying the same replacement twice yields the same
    # result as applying it once.
    result_twice = replace_cv_fields(result, new_cv_fields)
    assert result_twice == result
