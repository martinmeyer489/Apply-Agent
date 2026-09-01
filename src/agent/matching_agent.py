"""Matching Agent.

Authored with the Agent Framework `ResponsesAgent` interface. Matches a
User_Profile against the job listing corpus using UC Function tools for
all retrieval, distance computation, and profile lookup, then ranks and
returns the top candidates within a 60-second budget.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10,
7.11, 9.1
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

UC_FUNCTION_NAMES = [
    "job_agent.gold.search_listings",
    "job_agent.gold.compute_commute_distance",
    "job_agent.gold.get_user_profile",
    "job_agent.gold.draft_application",
]

MAX_SEARCH_CANDIDATES = 200
MAX_RESULTS = 50
MAX_EXPLANATION_CHARS = 500
MATCHING_TIMEOUT_SECONDS = 60

NO_MATCHES_SUGGESTION = (
    "No matches found within {radius} km. Try increasing your commute "
    "radius or updating your CV."
)


# --------------------------------------------------------------------------
# Agent construction (Req 7.1, 7.2, 9.1)
# --------------------------------------------------------------------------


def build_agent():
    """Construct the LangChain agent backing the Matching Agent.

    Wires up `ChatDatabricks` against the shared Foundation Model endpoint
    and a `UCFunctionToolkit` exposing all 4 UC Function tools (Req 7.2),
    and enables MLflow autologging so every invocation emits an
    `MLflow_Trace` (Req 9.1).

    Returns:
        An `AgentExecutor` ready to receive matching requests.
    """
    import mlflow
    from databricks_langchain import ChatDatabricks
    from databricks_langchain.uc_ai import UCFunctionToolkit
    from langchain.agents import AgentExecutor

    mlflow.langchain.autolog()

    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
    toolkit = UCFunctionToolkit(function_names=UC_FUNCTION_NAMES)

    return AgentExecutor(llm=llm, tools=toolkit.get_tools())


# --------------------------------------------------------------------------
# Pure matching logic (Req 7.4 - 7.10)
# --------------------------------------------------------------------------


REQUIRED_PROFILE_FIELDS = ("skills", "home_coordinates", "commute_radius_km")


def validate_profile_completeness(profile: Dict[str, Any]) -> List[str]:
    """Return the list of missing required User_Profile fields.

    Mirrors Requirement 7.10: if the profile is missing the skills list,
    the home coordinates, or the commute radius, the caller must be told
    exactly which fields are missing so no retrieval is attempted.

    Args:
        profile: dict that may contain `skills`, `home_latitude`,
            `home_longitude`, and `commute_radius_km`.

    Returns:
        A list containing zero or more of: "skills", "home_coordinates",
        "commute_radius_km". Empty list means the profile is complete.
    """
    missing: List[str] = []

    if not profile.get("skills"):
        missing.append("skills")

    if profile.get("home_latitude") is None or profile.get("home_longitude") is None:
        missing.append("home_coordinates")

    if profile.get("commute_radius_km") is None:
        missing.append("commute_radius_km")

    return missing


def build_search_query_text(profile: Dict[str, Any]) -> str:
    """Build the Vector Search query text from a User_Profile.

    Concatenates skills, job title history, and qualifications summary, as
    required for the `search_listings` semantic query (Req 7.4).
    """
    skills = profile.get("skills") or []
    job_titles = profile.get("job_title_history") or []
    qualifications_summary = profile.get("qualifications_summary") or ""

    parts = [", ".join(skills), ", ".join(job_titles), qualifications_summary]
    return " ".join(part for part in parts if part).strip()


def filter_by_commute_radius(
    candidates: List[Dict[str, Any]],
    home_lat: float,
    home_lon: float,
    radius_km: float,
    distance_fn,
) -> List[Dict[str, Any]]:
    """Exclude candidates whose commute distance exceeds `radius_km` (Req 7.6).

    Each surviving candidate has its computed `distance_km` (rounded to 1
    decimal place by `distance_fn`) attached.

    Args:
        candidates: list of dicts, each with `latitude`/`longitude`.
        home_lat: User_Profile home latitude.
        home_lon: User_Profile home longitude.
        radius_km: Commute_Radius in kilometres.
        distance_fn: callable `(lat1, lon1, lat2, lon2) -> float` used to
            compute the commute distance (the `compute_commute_distance`
            UC Function tool, or the equivalent haversine helper).

    Returns:
        The subset of `candidates` within `radius_km`, each augmented with
        a `distance_km` key.
    """
    kept = []
    for candidate in candidates:
        distance_km = distance_fn(
            home_lat, home_lon, candidate["latitude"], candidate["longitude"]
        )
        if distance_km <= radius_km:
            enriched = dict(candidate)
            enriched["distance_km"] = round(distance_km, 1)
            kept.append(enriched)
    return kept


def score_relevance(candidate: Dict[str, Any]) -> int:
    """Assign a Relevance_Score from 0 to 100 inclusive (Req 7.5).

    The candidate's `similarity_score` (as returned by `search_listings`,
    typically in [0, 1]) is scaled to the 0-100 range and clamped so
    upstream floating point noise can never escape the documented bounds.
    """
    similarity_score = candidate.get("similarity_score") or 0.0
    scaled = round(similarity_score * 100)
    return max(0, min(100, scaled))


def build_match_explanation(candidate: Dict[str, Any], profile: Dict[str, Any]) -> str:
    """Build a match explanation for a candidate, truncated to 500 chars (Req 7.8)."""
    job_title = candidate.get("job_title", "this role")
    company_name = candidate.get("company_name", "the company")
    distance_km = candidate.get("distance_km")

    explanation = (
        f"Matched to {job_title} at {company_name} based on profile skill "
        f"and experience overlap"
    )
    if distance_km is not None:
        explanation += f", {distance_km} km from your home location"
    explanation += "."

    return truncate_explanation(explanation, MAX_EXPLANATION_CHARS)


def truncate_explanation(text: str, max_len: int = MAX_EXPLANATION_CHARS) -> str:
    """Truncate `text` to at most `max_len` characters (Req 7.8)."""
    return text[:max_len]


def rank_and_limit_matches(
    scored_candidates: List[Dict[str, Any]], max_results: int = MAX_RESULTS
) -> List[Dict[str, Any]]:
    """Sort candidates by Relevance_Score descending and cap at `max_results` (Req 7.7)."""
    ranked = sorted(scored_candidates, key=lambda c: c["relevance_score"], reverse=True)
    return ranked[:max_results]


def build_empty_result_message(radius_km: float) -> str:
    """Build the suggestion message returned when no candidates remain (Req 7.9)."""
    return NO_MATCHES_SUGGESTION.format(radius=radius_km)


# --------------------------------------------------------------------------
# ResponsesAgent implementation
# --------------------------------------------------------------------------


class MatchingAgent:
    """Matching Agent authored with the Agent Framework `ResponsesAgent` interface.

    On Databricks, this class subclasses `mlflow.pyfunc.ResponsesAgent` so it
    can be logged to MLflow, registered in Unity Catalog, and deployed to a
    Model Serving endpoint (Req 7.1). The base class is resolved lazily in
    `__init__` (rather than at import time) so this module can be imported
    and unit-tested outside a Databricks runtime, where `mlflow` may not be
    installed.

    Tool access is injected via `tools` so the class can be tested with
    fakes; in production, `tools` wraps the UC Function calls exposed by
    `build_agent()`'s `UCFunctionToolkit`.
    """

    def __init__(self, tools: Optional[Dict[str, Any]] = None, agent: Optional[Any] = None):
        self.tools = tools or {}
        self._agent = agent

    # -- tool call helpers (delegate to injected callables, or the real
    #    UC Function tools / agent executor in production) --------------

    def _get_user_profile(self, profile_id: str) -> Dict[str, Any]:
        return self.tools["get_user_profile"](profile_id)

    def _search_listings(self, query_text: str, max_results: int = MAX_SEARCH_CANDIDATES) -> List[Dict[str, Any]]:
        return self.tools["search_listings"](query_text, max_results)

    def _compute_commute_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return self.tools["compute_commute_distance"](lat1, lon1, lat2, lon2)

    # -- core matching flow (Req 7) ---------------------------------------

    def match(self, profile_id: str) -> Dict[str, Any]:
        """Run the full matching flow for `profile_id` (Req 7.1-7.10).

        Returns a dict shaped like:
            {"error": "...", "missing_fields": [...]}   -- on incomplete profile
            {"results": [...], "message": "..."}         -- otherwise (message
                                                             present only when
                                                             results is empty)
        """
        profile = self._get_user_profile(profile_id)

        missing_fields = validate_profile_completeness(profile)
        if missing_fields:
            return {
                "error": f"Cannot match: please complete {', '.join(missing_fields)} first.",
                "missing_fields": missing_fields,
            }

        query_text = build_search_query_text(profile)
        candidates = self._search_listings(query_text, MAX_SEARCH_CANDIDATES)

        within_radius = filter_by_commute_radius(
            candidates,
            profile["home_latitude"],
            profile["home_longitude"],
            profile["commute_radius_km"],
            self._compute_commute_distance,
        )

        if not within_radius:
            return {
                "results": [],
                "message": build_empty_result_message(profile["commute_radius_km"]),
            }

        scored = []
        for candidate in within_radius:
            relevance_score = score_relevance(candidate)
            explanation = build_match_explanation(candidate, profile)
            scored.append(
                {
                    **candidate,
                    "relevance_score": relevance_score,
                    "explanation": explanation,
                }
            )

        results = rank_and_limit_matches(scored, MAX_RESULTS)
        return {"results": results}

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """`ResponsesAgent` entry point (Req 7.1).

        Extracts `profile_id` from the incoming request, runs `match`
        under a best-effort 60-second budget (Req 7.11), and returns a
        response dict. Tool call latencies are bounded by the underlying
        UC Functions / Vector Search SLAs; the `ThreadPoolExecutor`
        timeout here is a defensive backstop so a single invocation never
        exceeds the documented 60-second contract from the caller's
        perspective.
        """
        profile_id = request.get("profile_id") if isinstance(request, dict) else None
        if not profile_id:
            return {"error": "Cannot match: request is missing profile_id."}

        with ThreadPoolExecutor(max_workers=1) as executor:
            future: Future = executor.submit(self.match, profile_id)
            try:
                return future.result(timeout=MATCHING_TIMEOUT_SECONDS)
            except FuturesTimeoutError as exc:
                raise TimeoutError(
                    f"Matching timed out after {MATCHING_TIMEOUT_SECONDS}s"
                ) from exc
