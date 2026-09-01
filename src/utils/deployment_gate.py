"""Deployment gate for the registered Matching Agent model.

Before a registered `job_agent.gold.matching_agent` version is promoted to
the Model Serving endpoint, this module checks whether that version has
recorded evaluation results in `job_agent.ops.evaluation_results`
(populated by `notebooks/05_run_evaluation.py`, task 17.2). If no results
exist for the version, promotion is blocked and the missing evaluation is
reported (Req 9 AC5-6).

Validates: Requirements 9.5, 9.6
"""

from __future__ import annotations

from typing import Optional, Tuple

EVALUATION_RESULTS_TABLE = "job_agent.ops.evaluation_results"


def check_deployment_gate(model_version: int, spark: Optional[object] = None) -> Tuple[bool, str]:
    """Check whether a registered model version may be promoted.

    Queries `job_agent.ops.evaluation_results` for rows matching
    `model_version`. If none exist, promotion is blocked. If at least one
    exists, promotion is allowed and the most recent scorer means are
    included in the reason for observability.

    Args:
        model_version: The Unity Catalog registered model version number
            to check.
        spark: Optional SparkSession to use for the query. If omitted,
            the active Databricks notebook session (or a local one) is
            resolved via `SparkSession.builder.getOrCreate()`. Exposed as
            a parameter primarily so tests can inject a fake spark.

    Returns:
        A `(blocked, reason)` tuple:
            - `blocked=True` and a reason naming the missing evaluation
              results when no rows exist for `model_version`.
            - `blocked=False` and a reason citing the latest scorer means
              when evaluation results exist and promotion is allowed.
    """
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

    results_df = spark.table(EVALUATION_RESULTS_TABLE)
    rows = (
        results_df.filter(results_df.model_version == model_version)
        .orderBy("eval_timestamp", ascending=False)
        .collect()
    )

    if not rows:
        return (
            True,
            f"No evaluation results found for model version {model_version}. "
            "Run notebooks/05_run_evaluation.py before promoting this version.",
        )

    latest = rows[0]
    return (
        False,
        f"Evaluation results found for model version {model_version} "
        f"(match_relevance_mean={latest['match_relevance_mean']}, "
        f"groundedness_mean={latest['groundedness_mean']}); promotion allowed.",
    )
