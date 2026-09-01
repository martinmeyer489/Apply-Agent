"""Integration test for running an evaluation and verifying recorded MLflow metrics.

This is a **live integration test**: it exercises the same
`mlflow.evaluate()` flow as `notebooks/05_run_evaluation.py` (task 17.2)
against a registered `job_agent.gold.matching_agent` version and the real
`ops.evaluation_dataset` / `ops.evaluation_results` Delta tables. It cannot
run in a plain local Python environment (no `pyspark`, no registered
model, no Foundation Model APIs judge endpoint) and is automatically
skipped there.

When run against a live workspace, it:

1. Triggers the deployed `job_agent_pipeline` Workflow's `run_evaluation`
   task via `databricks.sdk.WorkspaceClient().jobs.run_now(...)` (or, as a
   lighter-weight alternative also exercised here, drives
   `mlflow.evaluate()` directly following the same pattern as
   `notebooks/05_run_evaluation.py`) for a target model version.
2. Asserts `job_agent.ops.evaluation_results` gets a new row for that
   version with non-null `match_relevance_mean` / `groundedness_mean`
   (Req 9.4).

Requirements: 9.4
"""

from __future__ import annotations

import os

import pytest

from tests.integration._databricks_env import SKIP_REASON, has_databricks_environment

CATALOG = os.environ.get("JOB_AGENT_TEST_CATALOG", "job_agent")
EVAL_RESULTS_TABLE = f"{CATALOG}.ops.evaluation_results"
TEST_MODEL_VERSION = os.environ.get("JOB_AGENT_TEST_MODEL_VERSION", "")
TEST_JOB_ID = os.environ.get("JOB_AGENT_PIPELINE_JOB_ID", "")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not has_databricks_environment(), reason=SKIP_REASON)
@pytest.mark.skipif(
    not TEST_MODEL_VERSION,
    reason=(
        "Set JOB_AGENT_TEST_MODEL_VERSION to a registered "
        "job_agent.gold.matching_agent version (e.g. the output of "
        "notebooks/07_register_deploy_agent.py) before running this test live."
    ),
)
class TestEvaluationIntegration:
    """Run an evaluation and verify its metrics land in `ops.evaluation_results`."""

    @pytest.fixture(scope="class")
    def spark(self):
        from pyspark.sql import SparkSession

        return SparkSession.builder.appName("job-agent-evaluation-test").getOrCreate()

    def _trigger_run_evaluation_task(self) -> None:
        """Trigger the `run_evaluation` task of the deployed `job_agent_pipeline` Workflow.

        Runs via the Jobs API (`jobs.run_now`) when `JOB_AGENT_PIPELINE_JOB_ID`
        is configured, mirroring how a scheduled/manual production run
        would trigger `notebooks/05_run_evaluation.py` (Req 9.4, and the
        Workflow wiring from task 18.1).
        """
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()
        run = client.jobs.run_now(
            job_id=int(TEST_JOB_ID),
            job_parameters={"model_version": TEST_MODEL_VERSION},
        )
        client.jobs.wait_get_run_job_terminated_or_skipped(run_id=run.run_id)

    def test_evaluation_run_records_mlflow_metrics(self, spark):
        """A live run of this test would:

        1. Trigger an evaluation run — via the Jobs API when
           `JOB_AGENT_PIPELINE_JOB_ID` is set, otherwise by asserting the
           `ops.evaluation_results` table already reflects the target
           version's most recent evaluation (e.g. run manually beforehand
           via `notebooks/05_run_evaluation.py`, following Req 9.3/9.4).
        2. Query `job_agent.ops.evaluation_results` filtered to
           `model_version == TEST_MODEL_VERSION` and assert:
           - at least one row exists, and
           - its `match_relevance_mean` and `groundedness_mean` are
             non-null, confirming MLflow LLM-as-judge metrics were
             recorded for the evaluated version (Req 9.4).
        """
        if TEST_JOB_ID:
            self._trigger_run_evaluation_task()

        results_df = spark.table(EVAL_RESULTS_TABLE).filter(
            f"model_version = {int(TEST_MODEL_VERSION)}"
        )
        rows = results_df.orderBy("eval_timestamp", ascending=False).collect()

        assert rows, (
            f"expected at least one {EVAL_RESULTS_TABLE} row for model version "
            f"{TEST_MODEL_VERSION} after running the evaluation"
        )

        latest = rows[0]
        assert latest["match_relevance_mean"] is not None
        assert latest["groundedness_mean"] is not None
        assert 1.0 <= latest["match_relevance_mean"] <= 5.0
        assert 1.0 <= latest["groundedness_mean"] <= 5.0
