"""Shared skip-gate helper for live-Databricks integration tests.

Every `tests/integration/test_*.py` module gates its tests behind
`_has_databricks_environment()` so the suite is automatically skipped (not
failed/errored) in this repo's local dev/CI environment, which has no
`pyspark`, no Databricks SDK credentials, and no live workspace resources
(Vector Search endpoints, Model Serving endpoints, SQL warehouse, Foundation
Model APIs). When run *inside* a Databricks cluster/serverless job/notebook
(or with `DATABRICKS_HOST` configured for Databricks Connect), the gate
passes and the tests exercise the real resources they document.

Factored out here (rather than duplicated per-file, as
`tests/integration/test_pipeline_e2e.py` did before this module existed) so
all integration test modules share one implementation and one reason
string.
"""

from __future__ import annotations

import os


def has_databricks_environment() -> bool:
    """Detect whether this process is running against a live Databricks workspace.

    Checks (in order, all cheap and side-effect free):
    1. `DATABRICKS_HOST` (or `DATABRICKS_RUNTIME_VERSION`, set automatically
       inside any Databricks cluster/serverless runtime) is set — the
       standard signal that we are either running *inside* a Databricks
       compute environment or have workspace credentials configured.
    2. `pyspark` is importable *and* a real, already-active `SparkSession`
       can be obtained (Databricks notebooks/jobs inject a live session;
       a bare local `pyspark` install with no session would not indicate
       a genuine Databricks/Delta/Unity Catalog environment).

    Returns:
        True if a live Databricks workspace environment appears to be
        available, False otherwise (in which case this module's tests
        are skipped).
    """
    if os.environ.get("DATABRICKS_HOST") or os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return True

    try:
        import pyspark  # noqa: F401
        from pyspark.sql import SparkSession

        active_session = SparkSession.getActiveSession()
        return active_session is not None
    except Exception:  # noqa: BLE001 - any import/lookup failure means "not available"
        return False


SKIP_REASON = (
    "Requires a live Databricks workspace (set DATABRICKS_HOST, or run inside "
    "a Databricks cluster/serverless notebook/job with pyspark and an active "
    "SparkSession, plus the relevant deployed resource: Vector Search index, "
    "Model Serving endpoint, SQL warehouse, or Foundation Model APIs access)"
)
