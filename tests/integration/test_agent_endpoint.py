"""Integration test for invoking the deployed Matching Agent's endpoint.

This is a **live integration test**: it exercises the real Model Serving
endpoint (`job-agent-matching`, deployed by
`notebooks/07_register_deploy_agent.py`, task 14.2) via
`databricks.sdk.WorkspaceClient().serving_endpoints.query()`. It cannot run
in a plain local Python environment (no deployed endpoint, no workspace
credentials) and is automatically skipped there.

When run against a live workspace, it:

1. Invokes the endpoint with a test `profile_id` — one of the seeded
   evaluation dataset's `case_id`s (`ops.evaluation_dataset`, seeded by
   `notebooks/05a_seed_evaluation_dataset.py`, task 17.1) which is bridged
   into `gold.user_profiles` by `notebooks/05_run_evaluation.py`, so a
   resolvable profile is guaranteed to exist without this test needing to
   insert its own row.
2. Asserts the response has the shape `MatchingAgent.predict()` documents
   (Req 7 — either `{"results": [...], "message": "..."}` or
   `{"error": "...", "missing_fields": [...]}`).
3. Asserts the round trip completed within 60 seconds (Req 7.11), timed
   with `time.monotonic()` around the call.

Requirements: 7.3, 7.11 (referenced via 7.3 in task 20.2)
"""

from __future__ import annotations

import os
import time

import pytest

from tests.integration._databricks_env import SKIP_REASON, has_databricks_environment

MATCHING_AGENT_ENDPOINT_NAME = os.environ.get(
    "JOB_AGENT_MATCHING_ENDPOINT", "job-agent-matching"
)
TEST_PROFILE_ID = os.environ.get("JOB_AGENT_TEST_PROFILE_ID", "")
MATCHING_TIMEOUT_SECONDS = 60

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not has_databricks_environment(), reason=SKIP_REASON)
@pytest.mark.skipif(
    not TEST_PROFILE_ID,
    reason=(
        "Set JOB_AGENT_TEST_PROFILE_ID to a resolvable profile_id/case_id "
        "(e.g. a case_id already bridged into gold.user_profiles by "
        "notebooks/05_run_evaluation.py) before running this test live."
    ),
)
class TestMatchingAgentEndpoint:
    """Invoke the deployed Matching Agent endpoint with a test profile."""

    @pytest.fixture(scope="class")
    def workspace_client(self):
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient()

    def test_invoke_endpoint_returns_expected_shape_within_timing_budget(self, workspace_client):
        """A live run of this test would:

        1. Call `serving_endpoints.query(name="job-agent-matching",
           inputs={"profile_id": TEST_PROFILE_ID})` and time the call with
           `time.monotonic()`.
        2. Assert the elapsed time is <= 60 seconds (Req 7.11).
        3. Assert the response is shaped like either a successful match
           response (`results` list, optionally a `message` when empty)
           or an error response (`error` string + `missing_fields` list),
           matching the two branches of `MatchingAgent.predict()`/
           `MatchingAgent.match()` (Req 7.3, 7.5, 7.7-7.10).
        """
        start_time = time.monotonic()
        response = workspace_client.serving_endpoints.query(
            name=MATCHING_AGENT_ENDPOINT_NAME,
            inputs={"profile_id": TEST_PROFILE_ID},
        )
        elapsed_seconds = time.monotonic() - start_time

        assert elapsed_seconds <= MATCHING_TIMEOUT_SECONDS, (
            f"matching round trip took {elapsed_seconds:.1f}s, exceeding the "
            f"{MATCHING_TIMEOUT_SECONDS}s budget from Requirement 7.11"
        )

        payload = response if isinstance(response, dict) else dict(
            getattr(response, "predictions", response)
        )

        if "error" in payload:
            assert isinstance(payload["error"], str) and payload["error"]
            assert isinstance(payload.get("missing_fields"), list)
        else:
            assert "results" in payload
            assert isinstance(payload["results"], list)
            assert len(payload["results"]) <= 50, (
                "the agent must return at most 50 results (Req 7.7)"
            )
            for result in payload["results"]:
                assert 0 <= result["relevance_score"] <= 100
                assert len(result["explanation"]) <= 500
            if not payload["results"]:
                assert "message" in payload, (
                    "an empty result set must be paired with a suggestion "
                    "message (Req 7.9)"
                )
