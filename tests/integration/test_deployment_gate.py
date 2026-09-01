"""Integration test for the deployment gate against real `ops.evaluation_results`.

This is a **live integration test**: unlike the deployment gate's property
test (`tests/property/test_deployment_gate.py`, task 17.4), which exercises
`check_deployment_gate()` against a hand-rolled fake `spark`/DataFrame,
this test calls it against a real Spark session reading the real
`job_agent.ops.evaluation_results` Delta table. It cannot run in a plain
local Python environment (no `pyspark`, no Unity Catalog) and is
automatically skipped there.

When run against a live workspace, it attempts promotion for two model
versions:

(a) A version WITH evaluation results present (e.g. one already evaluated
    via `notebooks/05_run_evaluation.py`) -> asserts `blocked=False`.
(b) A version WITHOUT any evaluation results — a large sentinel integer
    version number guaranteed not to exist in `ops.evaluation_results` ->
    asserts `blocked=True` and that the reason names the missing
    evaluation.

Requirements: 9.6
"""

from __future__ import annotations

import os

import pytest

from tests.integration._databricks_env import SKIP_REASON, has_databricks_environment

CATALOG = os.environ.get("JOB_AGENT_TEST_CATALOG", "job_agent")
EVALUATED_MODEL_VERSION = os.environ.get("JOB_AGENT_TEST_MODEL_VERSION", "")

# Guaranteed not to exist as a real Unity Catalog registered model version.
UNEVALUATED_MODEL_VERSION_SENTINEL = 999_999_999

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not has_databricks_environment(), reason=SKIP_REASON)
class TestDeploymentGateIntegration:
    """Attempt promotion with and without recorded evaluation results."""

    @pytest.fixture(scope="class")
    def spark(self):
        from pyspark.sql import SparkSession

        return SparkSession.builder.appName("job-agent-deployment-gate-test").getOrCreate()

    def test_promotion_blocked_without_evaluation_results(self, spark):
        """A live run of this test would:

        Call `check_deployment_gate(UNEVALUATED_MODEL_VERSION_SENTINEL,
        spark=spark)` against the real `ops.evaluation_results` table and
        assert:
        - `blocked=True` (no row can exist for a sentinel version this
          large), and
        - the reason names the missing evaluation results (Req 9.6).
        """
        from src.utils.deployment_gate import check_deployment_gate

        blocked, reason = check_deployment_gate(
            UNEVALUATED_MODEL_VERSION_SENTINEL, spark=spark
        )

        assert blocked is True
        assert "No evaluation results" in reason
        assert str(UNEVALUATED_MODEL_VERSION_SENTINEL) in reason

    @pytest.mark.skipif(
        not EVALUATED_MODEL_VERSION,
        reason=(
            "Set JOB_AGENT_TEST_MODEL_VERSION to a registered "
            "job_agent.gold.matching_agent version that already has a row in "
            "ops.evaluation_results (e.g. after running "
            "notebooks/05_run_evaluation.py) before running this test live."
        ),
    )
    def test_promotion_allowed_with_evaluation_results(self, spark):
        """A live run of this test would:

        Call `check_deployment_gate(EVALUATED_MODEL_VERSION, spark=spark)`
        against the real `ops.evaluation_results` table (populated by
        `notebooks/05_run_evaluation.py`) and assert:
        - `blocked=False`, and
        - the reason cites the evaluated version and its scorer means
          (Req 9.5, 9.6).
        """
        from src.utils.deployment_gate import check_deployment_gate

        blocked, reason = check_deployment_gate(
            int(EVALUATED_MODEL_VERSION), spark=spark
        )

        assert blocked is False
        assert EVALUATED_MODEL_VERSION in reason
        assert "match_relevance_mean" in reason
        assert "groundedness_mean" in reason
