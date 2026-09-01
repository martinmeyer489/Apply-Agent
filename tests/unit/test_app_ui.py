"""Unit tests for the Gradio app (Task 16.7).

Exercises the pure/testable event-handler helpers in `src.app.main` without
requiring a live Databricks connection:

1. Auto-stop 503 detection (`_is_auto_stop_error`, `_auto_stop_restart_message`)
2. Session state preservation across failure paths in
   `handle_find_matches` / `handle_draft_application`
3. Completeness gate messaging for various combinations of missing fields

Validates: Requirements 10.4, 10.7, 10.8, 11.5
"""

from unittest.mock import patch

import pytest

from src.app.main import (
    INITIAL_USER_PROFILE,
    MISSING_FIELD_DESCRIPTIONS,
    _auto_stop_restart_message,
    _build_completeness_gate_message,
    _is_auto_stop_error,
    handle_draft_application,
    handle_find_matches,
)


# ---------------------------------------------------------------------------
# 1. Auto-stop 503 detection
# ---------------------------------------------------------------------------

class _FakeHTTPError(Exception):
    """A fake exception exposing a `status_code` attribute, like an SDK error."""

    def __init__(self, status_code, message="request failed"):
        super().__init__(message)
        self.status_code = status_code


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeResponseWrappedError(Exception):
    """A fake exception exposing `.response.status_code`, like `requests`."""

    def __init__(self, status_code, message="request failed"):
        super().__init__(message)
        self.response = _FakeResponse(status_code)


def test_is_auto_stop_error_true_for_status_code_attribute():
    assert _is_auto_stop_error(_FakeHTTPError(503)) is True


def test_is_auto_stop_error_true_for_response_status_code_attribute():
    assert _is_auto_stop_error(_FakeResponseWrappedError(503)) is True


def test_is_auto_stop_error_true_for_503_in_message():
    assert _is_auto_stop_error(RuntimeError("Service Unavailable: 503")) is True


def test_is_auto_stop_error_false_for_generic_exception():
    assert _is_auto_stop_error(RuntimeError("something went wrong")) is False


def test_is_auto_stop_error_false_for_500_status():
    assert _is_auto_stop_error(_FakeHTTPError(500)) is False
    assert _is_auto_stop_error(_FakeResponseWrappedError(500)) is False


def test_auto_stop_restart_message_content():
    message = _auto_stop_restart_message()
    assert "24 hours" in message
    assert "Restart" in message


# ---------------------------------------------------------------------------
# 2. Session state preservation
# ---------------------------------------------------------------------------

def _complete_profile():
    profile = dict(INITIAL_USER_PROFILE)
    profile.update(
        {
            "skills": ["python", "spark"],
            "home_latitude": 52.5200,
            "home_longitude": 13.4050,
            "commute_radius_km": 25,
        }
    )
    return profile


def test_handle_find_matches_preserves_profile_on_endpoint_failure():
    original_profile = _complete_profile()
    profile_copy = dict(original_profile)

    with patch(
        "src.app.main._invoke_matching_agent",
        return_value=(None, RuntimeError("boom")),
    ):
        # handle_find_matches does not return user_profile as an output
        # (it is not part of this handler's outputs), so the returned
        # results/messages should reflect failure while the caller's
        # profile dict itself must remain byte-for-byte unchanged.
        results_table, message, results_list, dropdown_update = handle_find_matches(
            profile_copy
        )

    assert profile_copy == original_profile
    assert results_table == {"headers": results_table["headers"], "data": []}
    assert results_list == []


def test_handle_find_matches_preserves_profile_on_incomplete_gate():
    incomplete_profile = dict(INITIAL_USER_PROFILE)
    profile_copy = dict(incomplete_profile)

    results_table, message, results_list, dropdown_update = handle_find_matches(
        profile_copy
    )

    assert profile_copy == incomplete_profile
    assert results_list == []
    assert "Complete these steps" in message


def test_handle_draft_application_preserves_profile_on_endpoint_failure():
    original_profile = _complete_profile()
    profile_copy = dict(original_profile)

    with patch(
        "src.app.main._invoke_draft_application",
        return_value=(None, RuntimeError("boom")),
    ):
        cover_letter = handle_draft_application("listing-1", profile_copy)

    assert profile_copy == original_profile
    assert cover_letter == ""


def test_handle_draft_application_preserves_profile_when_no_listing_selected():
    original_profile = _complete_profile()
    profile_copy = dict(original_profile)

    cover_letter = handle_draft_application(None, profile_copy)

    assert profile_copy == original_profile
    assert cover_letter == ""


# ---------------------------------------------------------------------------
# 3. Completeness gate with various missing fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "missing_fields",
    [
        ["skills"],
        ["home_coordinates"],
        ["commute_radius_km"],
        ["skills", "home_coordinates"],
        ["skills", "commute_radius_km"],
        ["home_coordinates", "commute_radius_km"],
        ["skills", "home_coordinates", "commute_radius_km"],
    ],
)
def test_build_completeness_gate_message_names_missing_steps(missing_fields):
    message = _build_completeness_gate_message(missing_fields)
    for field in missing_fields:
        assert MISSING_FIELD_DESCRIPTIONS[field] in message

    # Fields NOT missing should not have their descriptions present, unless
    # another missing field happens to share the same description text.
    present_descriptions = {MISSING_FIELD_DESCRIPTIONS[f] for f in missing_fields}
    for field, description in MISSING_FIELD_DESCRIPTIONS.items():
        if field not in missing_fields and description not in present_descriptions:
            assert description not in message


def test_handle_find_matches_gate_missing_skills_only():
    profile = dict(INITIAL_USER_PROFILE)
    profile["home_latitude"] = 52.52
    profile["home_longitude"] = 13.40
    profile["commute_radius_km"] = 30

    _, message, _, _ = handle_find_matches(profile)

    assert MISSING_FIELD_DESCRIPTIONS["skills"] in message
    assert MISSING_FIELD_DESCRIPTIONS["home_coordinates"] not in message
    assert MISSING_FIELD_DESCRIPTIONS["commute_radius_km"] not in message


def test_handle_find_matches_gate_missing_coordinates_only():
    profile = dict(INITIAL_USER_PROFILE)
    profile["skills"] = ["python"]
    profile["commute_radius_km"] = 30

    _, message, _, _ = handle_find_matches(profile)

    assert MISSING_FIELD_DESCRIPTIONS["home_coordinates"] in message
    assert MISSING_FIELD_DESCRIPTIONS["skills"] not in message


def test_handle_find_matches_gate_missing_radius_only():
    profile = dict(INITIAL_USER_PROFILE)
    profile["skills"] = ["python"]
    profile["home_latitude"] = 52.52
    profile["home_longitude"] = 13.40
    profile["commute_radius_km"] = None

    _, message, _, _ = handle_find_matches(profile)

    assert MISSING_FIELD_DESCRIPTIONS["commute_radius_km"] in message
    assert MISSING_FIELD_DESCRIPTIONS["skills"] not in message
    assert MISSING_FIELD_DESCRIPTIONS["home_coordinates"] not in message


def test_handle_find_matches_gate_all_missing():
    profile = dict(INITIAL_USER_PROFILE)
    # INITIAL_USER_PROFILE defaults commute_radius_km to 50 (the slider's
    # default value), so force it to None to exercise all three gates.
    profile["commute_radius_km"] = None

    _, message, _, _ = handle_find_matches(profile)

    for description in MISSING_FIELD_DESCRIPTIONS.values():
        assert description in message


def test_handle_find_matches_proceeds_when_profile_complete():
    profile = _complete_profile()

    with patch(
        "src.app.main._invoke_matching_agent",
        return_value=({"results": []}, None),
    ) as mock_invoke:
        _, message, results_list, _ = handle_find_matches(profile)

    mock_invoke.assert_called_once()
    assert results_list == []
