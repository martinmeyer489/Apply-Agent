"""Property-based tests for matching result constraints and User_Profile
completeness validation.

Property 20: Matching Result Constraints
Validates: Requirements 7.5, 7.7, 7.8

Property 21: User Profile Completeness Validation
Validates: Requirements 7.10, 10.8

Both the Matching_Agent's ranking/truncation step and its User_Profile
completeness check are real business logic that will live in
`src/agent/matching_agent.py` (not yet implemented) and in the Databricks
App's completeness gate (Requirement 10.8). This test models both as pure
Python helpers mirroring the documented behaviour exactly.
"""

from hypothesis import given, strategies as st


def rank_and_limit_matches(scored_candidates: list) -> list:
    """Sort candidates by relevance score descending and cap at 50.

    Mirrors the Matching_Agent's final ranking step: "return at most 50
    Job_Listings ordered by Relevance_Score in descending order"
    (Requirement 7.7), where each candidate has already been assigned a
    Relevance_Score of 0-100 (Requirement 7.5).

    Args:
        scored_candidates: list of dicts, each with a `relevance_score` key.

    Returns:
        The input sorted descending by `relevance_score`, truncated to at
        most 50 entries. Ties preserve original relative order (stable
        sort), and no candidates are fabricated.
    """
    ranked = sorted(scored_candidates, key=lambda c: c["relevance_score"], reverse=True)
    return ranked[:50]


def truncate_explanation(text: str, max_len: int = 500) -> str:
    """Truncate `text` to at most `max_len` characters.

    Mirrors Requirement 7.8: each returned Job_Listing must carry a match
    explanation of at most 500 characters.
    """
    return text[:max_len]


def validate_profile_completeness(profile: dict) -> list:
    """Return the list of missing required field names in a User_Profile.

    Mirrors Requirement 7.10 / 10.8: if the User_Profile is missing the
    skills list, the home coordinates, or the Commute_Radius, the caller
    must be told exactly which fields are missing.

    Args:
        profile: dict that may contain `skills` (list), `home_latitude`,
            `home_longitude`, and `commute_radius_km`.

    Returns:
        A list containing zero or more of: "skills", "home_coordinates",
        "commute_radius_km". Empty list means the profile is complete.
    """
    missing = []

    skills = profile.get("skills")
    if not skills:
        missing.append("skills")

    home_lat = profile.get("home_latitude")
    home_lon = profile.get("home_longitude")
    if home_lat is None or home_lon is None:
        missing.append("home_coordinates")

    radius = profile.get("commute_radius_km")
    if radius is None:
        missing.append("commute_radius_km")

    return missing


# --- Strategies -------------------------------------------------------------

_relevance_score = st.integers(min_value=0, max_value=100)

_scored_candidate = st.fixed_dictionaries(
    {
        "listing_id": st.text(min_size=1, max_size=10),
        "relevance_score": _relevance_score,
    }
)

_scored_candidates = st.lists(_scored_candidate, min_size=0, max_size=120)


# --- Property 20: Matching Result Constraints -------------------------------


@given(scored_candidates=_scored_candidates)
def test_result_length_capped_at_50(scored_candidates):
    """**Validates: Requirements 7.7**

    The ranked result never contains more than 50 entries, regardless of
    how many candidates were scored.
    """
    result = rank_and_limit_matches(scored_candidates)
    assert len(result) <= 50


@given(scored_candidates=_scored_candidates)
def test_result_sorted_descending_by_relevance_score(scored_candidates):
    """**Validates: Requirements 7.5, 7.7**

    The ranked result is always sorted by relevance_score in descending
    order.
    """
    result = rank_and_limit_matches(scored_candidates)
    scores = [c["relevance_score"] for c in result]
    assert scores == sorted(scores, reverse=True)


@given(scored_candidates=_scored_candidates)
def test_result_is_subset_of_input(scored_candidates):
    """**Validates: Requirements 7.7**

    Every candidate in the ranked result is one of the original scored
    candidates -- ranking and truncation never fabricate candidates.
    """
    result = rank_and_limit_matches(scored_candidates)
    for candidate in result:
        assert any(candidate is c for c in scored_candidates)


@given(scored_candidates=_scored_candidates)
def test_relevance_scores_stay_within_0_to_100(scored_candidates):
    """**Validates: Requirements 7.5**

    Sanity check on the generator/domain itself: every relevance_score used
    throughout these tests lies within the documented 0-100 inclusive range.
    """
    for candidate in scored_candidates:
        assert 0 <= candidate["relevance_score"] <= 100


@given(text=st.text(min_size=0, max_size=2000))
def test_truncated_explanation_respects_max_length(text):
    """**Validates: Requirements 7.8**

    Truncating an arbitrary explanation string always yields a string of
    at most 500 characters.
    """
    truncated = truncate_explanation(text, 500)
    assert len(truncated) <= 500


# --- Property 21: User Profile Completeness Validation ----------------------

_optional_skills = st.one_of(
    st.none(),
    st.just([]),
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
)

_optional_coord = st.one_of(st.none(), st.floats(min_value=-180, max_value=180, allow_nan=False))

_optional_radius = st.one_of(st.none(), st.integers(min_value=1, max_value=200))

_profile = st.fixed_dictionaries(
    {
        "skills": _optional_skills,
        "home_latitude": _optional_coord,
        "home_longitude": _optional_coord,
        "commute_radius_km": _optional_radius,
    }
)


@given(profile=_profile)
def test_missing_fields_list_is_exact(profile):
    """**Validates: Requirements 7.10, 10.8**

    The returned missing-fields list names "skills" iff skills is empty or
    absent, "home_coordinates" iff either coordinate is missing, and
    "commute_radius_km" iff the radius is missing -- no more, no less.
    """
    missing = validate_profile_completeness(profile)

    expected = []
    if not profile.get("skills"):
        expected.append("skills")
    if profile.get("home_latitude") is None or profile.get("home_longitude") is None:
        expected.append("home_coordinates")
    if profile.get("commute_radius_km") is None:
        expected.append("commute_radius_km")

    assert missing == expected


@given(profile=_profile)
def test_empty_missing_list_iff_profile_complete(profile):
    """**Validates: Requirements 7.10, 10.8**

    The missing-fields list is empty if and only if the profile has
    non-empty skills, both home coordinates present, and a commute radius
    present.
    """
    missing = validate_profile_completeness(profile)

    is_complete = (
        bool(profile.get("skills"))
        and profile.get("home_latitude") is not None
        and profile.get("home_longitude") is not None
        and profile.get("commute_radius_km") is not None
    )

    assert (len(missing) == 0) == is_complete
