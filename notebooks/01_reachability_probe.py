# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Reachability Probe
# MAGIC
# MAGIC Probes a configurable list of candidate job-board domains from
# MAGIC **serverless compute** (a notebook task, not a serverless UDF — UDFs
# MAGIC have no outbound internet access at all) to determine which domains are
# MAGIC reachable from Databricks Free Edition's trusted-domain allowlist.
# MAGIC
# MAGIC For each candidate domain, issues an HTTP `HEAD` request with a 30
# MAGIC second timeout and records the outcome (`reachable`/`blocked`), the
# MAGIC HTTP status code (if any), and the error message (if any). Results are
# MAGIC MERGEd into `job_agent.ops.reachability_report` on `domain`, followed by
# MAGIC a summary row carrying `reachable_count`/`blocked_count`.
# MAGIC
# MAGIC The `Ingestion_Pipeline` (`notebooks/02_ingest_listings.py`) reads this
# MAGIC report and only attempts collection from domains whose most recent
# MAGIC outcome is `reachable`.
# MAGIC
# MAGIC Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7

# COMMAND ----------

import sys
import os

# Allow `from src.models.schemas import ...` when this notebook is run from
# the Databricks Workspace (repo root is not automatically on sys.path
# there, unlike when running via `databricks bundle` sync or repos).
_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_REPO_ROOT = os.path.abspath(os.path.join(_NOTEBOOK_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.schemas import OPS_REACHABILITY_REPORT_SCHEMA  # noqa: E402

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets
# MAGIC
# MAGIC `candidate_domains` accepts a comma-separated list of domains (no
# MAGIC scheme, no path) and defaults to 5 major job-board domains, satisfying
# MAGIC the "at least 5 job-board source domains" requirement (Req 1 AC1).

# COMMAND ----------

DEFAULT_CANDIDATE_DOMAINS = "indeed.com,linkedin.com,glassdoor.com,monster.com,reed.co.uk"

dbutils.widgets.text("candidate_domains", DEFAULT_CANDIDATE_DOMAINS, "Candidate domains (comma-separated)")
dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("schema", "ops", "Schema name")
dbutils.widgets.text("table", "reachability_report", "Table name")
dbutils.widgets.text("timeout_seconds", "30", "Per-domain HEAD request timeout (seconds)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABLE = dbutils.widgets.get("table")
TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"
TIMEOUT_SECONDS = int(dbutils.widgets.get("timeout_seconds"))

CANDIDATE_DOMAINS = [
    domain.strip()
    for domain in dbutils.widgets.get("candidate_domains").split(",")
    if domain.strip()
]

if len(CANDIDATE_DOMAINS) < 5:
    raise ValueError(
        f"Expected at least 5 candidate domains, got {len(CANDIDATE_DOMAINS)}: "
        f"{CANDIDATE_DOMAINS}. Update the 'candidate_domains' widget."
    )

print(f"Target table:      {TABLE_NAME}")
print(f"Per-domain timeout: {TIMEOUT_SECONDS}s")
print(f"Candidate domains ({len(CANDIDATE_DOMAINS)}): {CANDIDATE_DOMAINS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe function
# MAGIC
# MAGIC Issues an HTTP `HEAD` request to `https://{domain}` with the configured
# MAGIC timeout. Any exception (timeout, DNS failure, connection error, TLS
# MAGIC error, etc.) is caught individually per domain so a failure never stops
# MAGIC the remaining probes from running (Req 1 AC4).

# COMMAND ----------

import requests
from datetime import datetime, timezone


def probe_domain(domain: str, timeout_seconds: int = TIMEOUT_SECONDS) -> dict:
    """Probe a single candidate domain via HTTP HEAD.

    Args:
        domain: Bare domain name (e.g. "indeed.com"), no scheme or path.
        timeout_seconds: Connection/read timeout in seconds.

    Returns:
        A dict with keys `domain`, `probe_timestamp`, `outcome`
        (`reachable`|`blocked`), `http_status_code` (int | None), and
        `error_message` (str | None).
    """
    probe_timestamp = datetime.now(timezone.utc)
    try:
        response = requests.head(f"https://{domain}", timeout=timeout_seconds)
        return {
            "domain": domain,
            "probe_timestamp": probe_timestamp,
            "outcome": "reachable",
            "http_status_code": response.status_code,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001 - any failure classifies as blocked
        return {
            "domain": domain,
            "probe_timestamp": probe_timestamp,
            "outcome": "blocked",
            "http_status_code": None,
            "error_message": str(exc),
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the probe
# MAGIC
# MAGIC Every candidate domain is attempted regardless of prior failures
# MAGIC (Req 1 AC4).

# COMMAND ----------

probe_results = [probe_domain(domain) for domain in CANDIDATE_DOMAINS]

for result in probe_results:
    status = result["http_status_code"]
    error = result["error_message"]
    print(f"  {result['domain']:<20} -> {result['outcome']:<10} status={status} error={error}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute summary counts
# MAGIC
# MAGIC `reachable_count + blocked_count` always equals the number of candidate
# MAGIC domains probed (Req 1 AC5).

# COMMAND ----------

reachable_count = sum(1 for r in probe_results if r["outcome"] == "reachable")
blocked_count = sum(1 for r in probe_results if r["outcome"] == "blocked")

print(f"Reachable: {reachable_count}")
print(f"Blocked:   {blocked_count}")
print(f"Total:     {reachable_count + blocked_count} (expected {len(CANDIDATE_DOMAINS)})")

assert reachable_count + blocked_count == len(CANDIDATE_DOMAINS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the rows to MERGE
# MAGIC
# MAGIC Per-domain detail rows carry `reachable_count`/`blocked_count` as
# MAGIC `None` (those fields are populated only on the summary row). The
# MAGIC summary row uses a sentinel domain value of `__summary__` and a null
# MAGIC `http_status_code`/`error_message`, and is timestamped with the same
# MAGIC probe run time as the detail rows.

# COMMAND ----------

SUMMARY_DOMAIN = "__summary__"
run_timestamp = datetime.now(timezone.utc)

detail_rows = [
    (
        r["domain"],
        r["probe_timestamp"],
        r["outcome"],
        r["http_status_code"],
        r["error_message"],
        None,
        None,
    )
    for r in probe_results
]

summary_row = (
    SUMMARY_DOMAIN,
    run_timestamp,
    "reachable" if reachable_count > 0 else "blocked",
    None,
    None,
    reachable_count,
    blocked_count,
)

all_rows = detail_rows + [summary_row]

report_df = spark.createDataFrame(all_rows, schema=OPS_REACHABILITY_REPORT_SCHEMA)
display(report_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MERGE into `ops.reachability_report`
# MAGIC
# MAGIC MERGE keyed on `domain` so re-running the probe updates each domain's
# MAGIC (and the summary's) most recent outcome in place rather than
# MAGIC accumulating duplicate history rows (Req 1 AC5, AC6).

# COMMAND ----------

from delta.tables import DeltaTable

if not spark.catalog.tableExists(TABLE_NAME):
    report_df.write.format("delta").saveAsTable(TABLE_NAME)
    print(f"Created and populated {TABLE_NAME}")
else:
    target = DeltaTable.forName(spark, TABLE_NAME)
    (
        target.alias("target")
        .merge(report_df.alias("source"), "target.domain = source.domain")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"Merged {report_df.count()} rows into {TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

result_df = spark.table(TABLE_NAME).filter(f"domain != '{SUMMARY_DOMAIN}'")
display(result_df.orderBy("domain"))

summary = {
    "candidate_domains": CANDIDATE_DOMAINS,
    "reachable_count": reachable_count,
    "blocked_count": blocked_count,
    "table": TABLE_NAME,
}
print(summary)
