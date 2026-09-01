"""Unit tests for the Matching Agent (Task 14.3).

Exercises `MatchingAgent.match` and `MatchingAgent.predict` with injected
fake tool callables (no real Databricks SDK / UC Function / Vector Search
calls are made):

1. Profile with all fields -> successful match
2. Profile missing skills -> error (and `search_listings` never called)
3. All candidates beyond commute radius -> empty result + suggestion
4. `predict` wraps `match` correctly (valid request / missing profile_id)

Validates: Requirements 7.5, 7.9, 7.10
"""

from src.agent.matching_agent import MatchingAgent


def _complete_profile():
    return {
        "skills": ["python", "spark"],
        "job_title_history": ["Data Engineer"],
        "qualifications_summary": "Experienced data engineer.",
        "home_latitude": 52.5200,
        "home_longitude": 13.4050,
        "commute_radius_km": 25,
    }


def _candidates():
    return [
        {
            "listing_id": "abc123",
            "job_title": "Senior Data Engineer",
            "company_name": "Acme Corp",
            "latitude": 52.5300,
            "longitude": 13.4100,
            "similarity_score": 0.87,
        },
        {
            "listing_id": "def456",
            "job_title": "ML Engineer",
            "company_name": "Beta Inc",
            "latitude": 52.5000,
            "longitude": 13.3900,
            "similarity_score": 0.42,
        },
    ]


class _CountingWrapper:
    """Wraps a callable and records how many times it was invoked."""

    def __init__(self, fn):
        self._fn = fn
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self._fn(*args, **kwargs)


# --------------------------------------------------------------------------
# 1. Profile with all fields -> successful match
# --------------------------------------------------------------------------


def test_match_with_complete_profile_returns_successful_results():
    profile = _complete_profile()
    fake_distance = 5.0

    tools = {
        "get_user_profile": lambda profile_id: profile,
        "search_listings": lambda query_text, max_results: _candidates(),
        "compute_commute_distance": lambda lat1, lon1, lat2, lon2: fake_distance,
    }

    agent = MatchingAgent(tools=tools)
    result = agent.match("profile-1")

    assert "error" not in result
    assert "results" in result
    assert len(result["results"]) == len(_candidates())

    for match in result["results"]:
        assert 0 <= match["relevance_score"] <= 100
        assert isinstance(match["explanation"], str)
        assert len(match["explanation"]) <= 500
        assert match["distance_km"] == round(fake_distance, 1)


def test_match_results_are_ranked_by_relevance_score_descending():
    profile = _complete_profile()

    tools = {
        "get_user_profile": lambda profile_id: profile,
        "search_listings": lambda query_text, max_results: _candidates(),
        "compute_commute_distance": lambda lat1, lon1, lat2, lon2: 5.0,
    }

    agent = MatchingAgent(tools=tools)
    result = agent.match("profile-1")

    scores = [match["relevance_score"] for match in result["results"]]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# 2. Profile missing skills -> error
# --------------------------------------------------------------------------


def test_match_with_missing_skills_returns_error_without_searching():
    profile = {
        "skills": [],
        "home_latitude": 52.5200,
        "home_longitude": 13.4050,
        "commute_radius_km": 25,
    }

    search_listings = _CountingWrapper(lambda query_text, max_results: _candidates())

    tools = {
        "get_user_profile": lambda profile_id: profile,
        "search_listings": search_listings,
        "compute_commute_distance": lambda lat1, lon1, lat2, lon2: 5.0,
    }

    agent = MatchingAgent(tools=tools)
    result = agent.match("profile-1")

    assert "error" in result
    assert "missing_fields" in result
    assert "skills" in result["missing_fields"]
    assert search_listings.call_count == 0


def test_match_with_absent_skills_field_returns_error():
    """A profile missing the `skills` key entirely (not just empty) is
    also treated as incomplete."""
    profile = {
        "home_latitude": 52.5200,
        "home_longitude": 13.4050,
        "commute_radius_km": 25,
    }

    tools = {
        "get_user_profile": lambda profile_id: profile,
        "search_listings": lambda query_text, max_results: _candidates(),
        "compute_commute_distance": lambda lat1, lon1, lat2, lon2: 5.0,
    }

    agent = MatchingAgent(tools=tools)
    result = agent.match("profile-1")

    assert "error" in result
    assert "skills" in result["missing_fields"]


# --------------------------------------------------------------------------
# 3. All candidates beyond radius -> empty result + suggestion
# --------------------------------------------------------------------------


def test_match_with_all_candidates_beyond_radius_returns_empty_with_message():
    profile = _complete_profile()

    tools = {
        "get_user_profile": lambda profile_id: profile,
        "search_listings": lambda query_text, max_results: _candidates(),
        "compute_commute_distance": lambda lat1, lon1, lat2, lon2: profile["commute_radius_km"] + 10,
    }

    agent = MatchingAgent(tools=tools)
    result = agent.match("profile-1")

    assert result["results"] == []
    assert "message" in result
    assert isinstance(result["message"], str)
    assert len(result["message"]) > 0
    assert str(profile["commute_radius_km"]) in result["message"] or "radius" in result["message"].lower()


# --------------------------------------------------------------------------
# 4. `predict` wraps `match` correctly
# --------------------------------------------------------------------------


def test_predict_with_valid_request_matches_direct_match_call():
    profile = _complete_profile()

    tools = {
        "get_user_profile": lambda profile_id: profile,
        "search_listings": lambda query_text, max_results: _candidates(),
        "compute_commute_distance": lambda lat1, lon1, lat2, lon2: 5.0,
    }

    agent = MatchingAgent(tools=tools)

    direct_result = agent.match("profile-1")
    predict_result = agent.predict({"profile_id": "profile-1"})

    assert predict_result == direct_result


def test_predict_with_missing_profile_id_returns_error_without_calling_tools():
    get_user_profile = _CountingWrapper(lambda profile_id: _complete_profile())
    search_listings = _CountingWrapper(lambda query_text, max_results: _candidates())
    compute_commute_distance = _CountingWrapper(lambda lat1, lon1, lat2, lon2: 5.0)

    tools = {
        "get_user_profile": get_user_profile,
        "search_listings": search_listings,
        "compute_commute_distance": compute_commute_distance,
    }

    agent = MatchingAgent(tools=tools)
    result = agent.predict({})

    assert "error" in result
    assert get_user_profile.call_count == 0
    assert search_listings.call_count == 0
    assert compute_commute_distance.call_count == 0
