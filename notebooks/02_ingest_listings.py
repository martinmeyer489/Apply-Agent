# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Ingestion Pipeline
# MAGIC
# MAGIC Fetches job listings from the **German Federal Employment Agency (Bundesagentur für Arbeit) Jobsuche API** 
# MAGIC (`https://jobsuche.api.bund.dev`) and populates `Job_Listing` records.
# MAGIC 
# MAGIC The API is publicly available with a fixed API key (`X-API-Key: jobboerse-jobsuche`).
# MAGIC If the API is unreachable, the pipeline falls back to the bundled dataset in
# MAGIC `job_agent.volumes.bundled_fallback`.
# MAGIC 
# MAGIC Records are validated, batched, and MERGE-upserted into `job_agent.bronze.job_listings` keyed on
# MAGIC `source_url`, preserving the original `listing_id` on updates.
# MAGIC
# MAGIC **Prerequisite**: `notebooks/00_setup_catalog.py` must have already run.
# MAGIC
# MAGIC API Reference: https://jobsuche.api.bund.dev (via bundesAPI)
# MAGIC
# MAGIC Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 11.8, 11.9

# COMMAND ----------

import sys
import os

# Allow `from src... import ...` when this notebook is run from the
# Databricks Workspace (repo root is not automatically on sys.path there,
# unlike when running via `databricks bundle` sync or repos).
_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_REPO_ROOT = os.path.abspath(os.path.join(_NOTEBOOK_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.schemas import (  # noqa: E402
    BRONZE_INGESTION_ERRORS_SCHEMA,
    BRONZE_JOB_LISTINGS_SCHEMA,
    GOLD_PIPELINE_SUMMARY_SCHEMA,
)
from src.utils.batching import batch_records  # noqa: E402
from src.utils.checkpoints import get_resume_batch, write_checkpoint  # noqa: E402
from src.utils.listing_id import derive_listing_id  # noqa: E402
from src.utils.validation import validate_listing_record  # noqa: E402

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("run_id", "", "Workflow run ID (blank = auto-generate)")
dbutils.widgets.text("batch_size", "500", "Max records per batch")
dbutils.widgets.text("request_timeout_seconds", "30", "Per-request timeout (seconds)")
dbutils.widgets.text(
    "local_fallback_csv_path",
    "../data/bundled_fallback/bundled_listings.csv",
    "Path to the local bundled_listings.csv (relative to this notebook)",
)
dbutils.widgets.text("job_search_query", "Data Engineer", "Job title search term")
dbutils.widgets.text("location_search", "Deutschland", "Location search term")
dbutils.widgets.text("max_pages", "5", "Maximum pages to fetch from API")

CATALOG = dbutils.widgets.get("catalog")
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
REQUEST_TIMEOUT_SECONDS = int(dbutils.widgets.get("request_timeout_seconds"))
LOCAL_FALLBACK_CSV_PATH = dbutils.widgets.get("local_fallback_csv_path")
JOB_SEARCH_QUERY = dbutils.widgets.get("job_search_query")
LOCATION_SEARCH = dbutils.widgets.get("location_search")
MAX_PAGES = int(dbutils.widgets.get("max_pages"))

import uuid  # noqa: E402

RUN_ID = dbutils.widgets.get("run_id").strip() or str(uuid.uuid4())
PIPELINE_NAME = "ingestion"

BRONZE_LISTINGS_TABLE = f"{CATALOG}.bronze.job_listings"
INGESTION_ERRORS_TABLE = f"{CATALOG}.bronze.ingestion_errors"
PIPELINE_SUMMARY_TABLE = f"{CATALOG}.gold.pipeline_summary"

FALLBACK_VOLUME_PATH = f"/Volumes/{CATALOG}/volumes/bundled_fallback"
FALLBACK_VOLUME_CSV_PATH = f"{FALLBACK_VOLUME_PATH}/bundled_listings.csv"

# Jobsuche API configuration
JOBSUCHE_BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"
JOBSUCHE_API_KEY = "jobboerse-jobsuche"  # Public client ID, no secret

print(f"Run ID:                 {RUN_ID}")
print(f"Bronze listings table:  {BRONZE_LISTINGS_TABLE}")
print(f"Ingestion errors table: {INGESTION_ERRORS_TABLE}")
print(f"Pipeline summary table: {PIPELINE_SUMMARY_TABLE}")
print(f"Batch size:             {BATCH_SIZE}")
print(f"Request timeout:        {REQUEST_TIMEOUT_SECONDS}s")
print(f"Job search query:       {JOB_SEARCH_QUERY}")
print(f"Location search:        {LOCATION_SEARCH}")
print(f"Max pages:              {MAX_PAGES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Determine if Jobsuche API is reachable (Req 2.1)
# MAGIC
# MAGIC Performs a lightweight health check against the Jobsuche API.
# MAGIC If the API is reachable, we fetch jobs live. Otherwise, we fall back
# MAGIC to the bundled dataset.

# COMMAND ----------

import requests


def check_api_reachable(timeout_seconds: int = 10) -> bool:
    """Check if the Jobsuche API is reachable via a simple query."""
    test_url = f"{JOBSUCHE_BASE_URL}/pc/v6/jobs?was=test&size=1"
    try:
        response = requests.get(
            test_url,
            headers={"X-API-Key": JOBSUCHE_API_KEY},
            timeout=timeout_seconds,
        )
        # Any 2xx or 4xx response means the API is reachable (401/403 just means no results)
        return response.status_code < 500
    except Exception:  # noqa: BLE001
        return False


api_reachable = check_api_reachable(REQUEST_TIMEOUT_SECONDS)
print(f"Jobsuche API reachable: {api_reachable}")

# COMMAND ----------

from datetime import datetime, timezone  # noqa: E402
import base64
import time

candidate_records: list[dict] = []
api_errors: list[dict] = []
ingestion_mode: str

if not api_reachable:
    ingestion_mode = "bundled_fallback"
    print("Jobsuche API unreachable — loading bundled fallback dataset.")
else:
    ingestion_mode = "api"
    print(f"Jobsuche API reachable — fetching jobs with query '{JOB_SEARCH_QUERY}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a — Bundled fallback loader (Req 2.9, 2.10)
# MAGIC
# MAGIC Uploads `data/bundled_fallback/bundled_listings.csv` to the
# MAGIC `job_agent.volumes.bundled_fallback` Volume if it is not already
# MAGIC present there (mirroring the upload pattern in
# MAGIC `notebooks/00a_load_reference_data.py`), then reads it back and tags
# MAGIC every row with `ingestion_mode = 'bundled_fallback'`.

# COMMAND ----------

import csv
import shutil


def load_bundled_fallback_records() -> list[dict]:
    """Load the bundled fallback job listings dataset.

    Uploads the repo-bundled CSV to the `bundled_fallback` Volume if it is
    not already there, then reads it back as a list of dicts. Every record
    is tagged with `ingestion_mode = 'bundled_fallback'` and
    `source_domain = 'bundled_fallback'` (Req 2.9).
    """
    os.makedirs(FALLBACK_VOLUME_PATH, exist_ok=True)

    if not os.path.exists(FALLBACK_VOLUME_CSV_PATH):
        resolved_local_path = os.path.abspath(
            os.path.join(_NOTEBOOK_DIR, LOCAL_FALLBACK_CSV_PATH)
            if not os.path.isabs(LOCAL_FALLBACK_CSV_PATH)
            else LOCAL_FALLBACK_CSV_PATH
        )
        if not os.path.exists(resolved_local_path):
            raise FileNotFoundError(
                f"Could not find bundled_listings.csv at '{resolved_local_path}'. "
                "Set the 'local_fallback_csv_path' widget to the correct location."
            )
        shutil.copyfile(resolved_local_path, FALLBACK_VOLUME_CSV_PATH)
        print(f"Uploaded {resolved_local_path} -> {FALLBACK_VOLUME_CSV_PATH}")
    else:
        print(f"Bundled fallback CSV already present at {FALLBACK_VOLUME_CSV_PATH}")

    records = []
    with open(FALLBACK_VOLUME_CSV_PATH, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            records.append(
                {
                    "listing_id": row["listing_id"],
                    "job_title": row["job_title"],
                    "company_name": row["company_name"],
                    "job_description": row.get("job_description"),
                    "location_text": row.get("location_text"),
                    "source_url": row["source_url"],
                    "ingestion_timestamp": row["ingestion_timestamp"],
                    "ingestion_mode": "bundled_fallback",
                    "source_domain": "bundled_fallback",
                }
            )
    return records


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b — Jobsuche API fetcher (Req 2.1, 2.4, 2.5, 2.7)
# MAGIC
# MAGIC Fetches job listings from the German Federal Employment Agency's Jobsuche API:
# MAGIC 1. Calls `GET /pc/v6/jobs` with search parameters (`was`, `wo`, pagination).
# MAGIC 2. For each job, calls `GET /pc/v4/jobdetails/{base64(refnr)}` to get full details.
# MAGIC 3. Rate-limits to respect API best practices.
# MAGIC
# MAGIC Any exception (network error, timeout, HTTP error) is caught, logged as an
# MAGIC `api_error`, and the pipeline falls back to bundled data.

# COMMAND ----------


def fetch_jobs_from_api(
    search_query: str,
    location: str,
    max_pages: int,
    timeout_seconds: int,
) -> list[dict]:
    """Fetch job listings from the Jobsuche API.

    Args:
        search_query: Job title search term (e.g. "Data Engineer").
        location: Location search term (e.g. "Berlin" or "Deutschland").
        max_pages: Maximum number of result pages to fetch.
        timeout_seconds: Request timeout in seconds.

    Returns:
        A list of job listing record dicts with keys: job_title, company_name,
        job_description, location_text, source_url.
    """
    records = []
    page = 1
    page_size = 50  # Maximum allowed by API

    while page <= max_pages:
        # Rate limit: be respectful to the public API
        time.sleep(1.0)

        # Search jobs endpoint
        search_url = f"{JOBSUCHE_BASE_URL}/pc/v6/jobs"
        params = {
            "was": search_query,
            "wo": location,
            "page": page,
            "size": page_size,
            "veroeffentlichtseit": 30,  # Jobs from last 30 days
        }

        try:
            response = requests.get(
                search_url,
                headers={"X-API-Key": JOBSUCHE_API_KEY},
                params=params,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            search_data = response.json()
        except Exception as exc:
            api_errors.append({
                "source_domain": "arbeitsagentur.de",
                "error_message": f"Search API failed on page {page}: {exc}",
                "error_type": "api_error",
            })
            print(f"  Page {page}: ERROR fetching search results: {exc}")
            break

        # Extract job refs from search response
        stellenangebote = search_data.get("stellenangebote", [])
        if not stellenangebote:
            print(f"  Page {page}: No more results")
            break

        print(f"  Page {page}: Found {len(stellenangebote)} job(s)")

        for job in stellenangebote:
            refnr = job.get("refnr")
            if not refnr:
                continue

            # Fetch full job details
            try:
                detail_url = f"{JOBSUCHE_BASE_URL}/pc/v4/jobdetails/{base64.b64encode(refnr.encode()).decode()}"
                detail_response = requests.get(
                    detail_url,
                    headers={"X-API-Key": JOBSUCHE_API_KEY},
                    timeout=timeout_seconds,
                )
                detail_response.raise_for_status()
                detail_data = detail_response.json()
            except Exception as exc:
                api_errors.append({
                    "source_domain": "arbeitsagentur.de",
                    "error_message": f"Details API failed for refnr {refnr}: {exc}",
                    "error_type": "api_error",
                })
                print(f"    Refnr {refnr}: ERROR fetching details, skipping")
                continue

            # Extract fields from detail response
            job_title = detail_data.get("stellenangebotsTitel") or detail_data.get("titel", "")
            company_name = detail_data.get("arbeitgeber", "")
            job_description = detail_data.get("stellenangebotsBeschreibung") or detail_data.get("stellenbeschreibung", "")

            # Extract location from arbeitsorte (can be multiple)
            arbeitsorte = detail_data.get("arbeitsorte", [])
            if arbeitsorte:
                primary_loc = arbeitsorte[0]
                location_parts = []
                if primary_loc.get("plz"):
                    location_parts.append(str(primary_loc["plz"]))
                if primary_loc.get("ort"):
                    location_parts.append(primary_loc["ort"])
                if primary_loc.get("region"):
                    location_parts.append(primary_loc["region"])
                location_text = ", ".join(location_parts) if location_parts else location
            else:
                location_text = location

            # Build a stable source URL (the BA job portal URL)
            source_url = f"https://www.arbeitsagentur.de/jobsuche/jobdetails/{refnr}"

            records.append({
                "job_title": job_title,
                "company_name": company_name,
                "job_description": job_description,
                "location_text": location_text,
                "source_url": source_url,
                # listing_id and timestamp added after validation
            })

        page += 1

    return records


# COMMAND ----------

if ingestion_mode == "bundled_fallback":
    candidate_records = load_bundled_fallback_records()
    print(f"Loaded {len(candidate_records)} bundled fallback records.")
else:
    # Fetch from Jobsuche API (tag records as 'api' ingestion mode)
    try:
        candidate_records = fetch_jobs_from_api(
            search_query=JOB_SEARCH_QUERY,
            location=LOCATION_SEARCH,
            max_pages=MAX_PAGES,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        print(f"Fetched {len(candidate_records)} job(s) from Jobsuche API")
        
        # Tag API records appropriately
        for record in candidate_records:
            record["ingestion_mode"] = "api"
            record["source_domain"] = "arbeitsagentur.de"
    except Exception as exc:
        # Any unhandled error triggers fallback
        api_errors.append({
            "source_domain": "arbeitsagentur.de",
            "error_message": f"API fetch failed: {exc}",
            "error_type": "api_error",
        })
        print(f"API fetch failed: {exc}")
        print("Falling back to bundled dataset...")
        ingestion_mode = "bundled_fallback"
        candidate_records = load_bundled_fallback_records()
        print(f"Loaded {len(candidate_records)} bundled fallback records.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Validate records (Req 2.6)
# MAGIC
# MAGIC Each candidate record is checked with `validate_listing_record` for the
# MAGIC presence of `job_title`, `company_name`, and `source_url`. Invalid
# MAGIC records are skipped and logged to `bronze.ingestion_errors` with the
# MAGIC names of the missing fields; valid records also get a deterministic
# MAGIC `listing_id` derived from their `source_url` (Req 2.2).

# COMMAND ----------

valid_records: list[dict] = []
skipped_record_errors: list[dict] = []

for record in candidate_records:
    is_valid, missing_fields = validate_listing_record(record)
    if not is_valid:
        skipped_record_errors.append(
            {
                "source_domain": record.get("source_domain", "arbeitsagentur.de"),
                "source_url": record.get("source_url"),
                "error_type": "missing_fields",
                "error_message": f"Missing required field(s): {', '.join(missing_fields)}",
                "missing_fields": missing_fields,
            }
        )
        continue

    record["listing_id"] = derive_listing_id(record["source_url"])
    record["ingestion_timestamp"] = datetime.now(timezone.utc)
    valid_records.append(record)

records_skipped = len(skipped_record_errors)
print(f"Valid records: {len(valid_records)} / {len(candidate_records)} (skipped: {records_skipped})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Log ingestion errors (Req 2.5, 2.6)
# MAGIC
# MAGIC API-level errors and record-level skips are combined and written to
# MAGIC `bronze.ingestion_errors`.

# COMMAND ----------

import uuid as _uuid  # noqa: E402

all_error_rows = []
for err in api_errors:
    all_error_rows.append(
        (
            str(_uuid.uuid4()),
            err.get("source_domain"),
            None,
            err.get("error_type", "api_error"),
            err["error_message"],
            None,
            datetime.now(timezone.utc),
        )
    )
for err in skipped_record_errors:
    all_error_rows.append(
        (
            str(_uuid.uuid4()),
            err.get("source_domain"),
            err.get("source_url"),
            err.get("error_type", "missing_fields"),
            err["error_message"],
            err.get("missing_fields"),
            datetime.now(timezone.utc),
        )
    )

source_errors = len(api_errors)

if all_error_rows:
    errors_df = spark.createDataFrame(all_error_rows, schema=BRONZE_INGESTION_ERRORS_SCHEMA)
    errors_df.write.format("delta").mode("append").saveAsTable(INGESTION_ERRORS_TABLE)
    print(f"Logged {len(all_error_rows)} error row(s) to {INGESTION_ERRORS_TABLE}")
else:
    print("No ingestion errors to log.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Batch, checkpoint, and MERGE upsert (Req 2.3, 2.8, 11.8, 11.9)
# MAGIC
# MAGIC Valid records are split into batches of at most `BATCH_SIZE` (default
# MAGIC 500). Processing resumes from `get_resume_batch` so a restarted run
# MAGIC skips already-completed batches. Each batch is MERGE-upserted into
# MAGIC `bronze.job_listings` keyed on `source_url`: on a match, all fields
# MAGIC except `listing_id` are updated (retaining the original identifier);
# MAGIC on no match, the full row is inserted. A checkpoint is written after
# MAGIC each successfully processed batch.

# COMMAND ----------

from delta.tables import DeltaTable  # noqa: E402
from pyspark.sql.functions import lit, to_timestamp  # noqa: E402

batches = batch_records(valid_records, batch_size=BATCH_SIZE)
resume_batch_index = get_resume_batch(spark, PIPELINE_NAME, RUN_ID)

print(f"Total batches: {len(batches)}; resuming from batch index {resume_batch_index}")

records_added = 0
records_updated = 0

if not spark.catalog.tableExists(BRONZE_LISTINGS_TABLE):
    empty_df = spark.createDataFrame([], schema=BRONZE_JOB_LISTINGS_SCHEMA)
    empty_df.write.format("delta").saveAsTable(BRONZE_LISTINGS_TABLE)

bronze_target = DeltaTable.forName(spark, BRONZE_LISTINGS_TABLE)

for batch_index, batch in enumerate(batches):
    if batch_index < resume_batch_index:
        print(f"Skipping already-completed batch {batch_index}")
        continue

    batch_rows = [
        (
            r["listing_id"],
            r["job_title"],
            r["company_name"],
            r.get("job_description"),
            r.get("location_text"),
            r["source_url"],
            r["ingestion_timestamp"],
            r["ingestion_mode"],
            "unenriched",
            r.get("source_domain") or "bundled_fallback",
        )
        for r in batch
    ]

    batch_df = spark.createDataFrame(batch_rows, schema=BRONZE_JOB_LISTINGS_SCHEMA)
    batch_df = batch_df.withColumn("ingestion_timestamp", to_timestamp("ingestion_timestamp"))

    batch_urls = [r["source_url"] for r in batch]
    existing_urls = {
        row["source_url"]
        for row in spark.table(BRONZE_LISTINGS_TABLE)
        .select("source_url")
        .filter(spark.table(BRONZE_LISTINGS_TABLE).source_url.isin(batch_urls))
        .collect()
    }

    batch_added = sum(1 for url in batch_urls if url not in existing_urls)
    batch_updated = len(batch) - batch_added

    (
        bronze_target.alias("target")
        .merge(batch_df.alias("source"), "target.source_url = source.source_url")
        .whenMatchedUpdate(
            set={
                "job_title": "source.job_title",
                "company_name": "source.company_name",
                "job_description": "source.job_description",
                "location_text": "source.location_text",
                "ingestion_timestamp": "source.ingestion_timestamp",
                "ingestion_mode": "source.ingestion_mode",
                "source_domain": "source.source_domain",
                # `listing_id` and `enrichment_state` are intentionally
                # omitted so the original values are retained on update
                # (Req 2.3).
            }
        )
        .whenNotMatchedInsertAll()
        .execute()
    )

    records_added += batch_added
    records_updated += batch_updated

    write_checkpoint(spark, PIPELINE_NAME, RUN_ID, batch_index)
    print(f"Batch {batch_index}: {len(batch)} record(s) merged ({batch_added} added, {batch_updated} updated)")

print(f"Records added: {records_added}, records updated: {records_updated}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Produce and persist the run summary (Req 2.11)

# COMMAND ----------

run_timestamp = datetime.now(timezone.utc)

summary = {
    "run_id": RUN_ID,
    "pipeline_name": PIPELINE_NAME,
    "ingestion_mode": ingestion_mode,
    "records_added": records_added,
    "records_updated": records_updated,
    "records_skipped": records_skipped,
    "source_errors": source_errors,
    "enriched_count": None,
    "partially_enriched_count": None,
    "failed_count": None,
    "unenriched_count": None,
    "run_timestamp": run_timestamp,
}

summary_row = [
    (
        summary["run_id"],
        summary["pipeline_name"],
        summary["ingestion_mode"],
        summary["records_added"],
        summary["records_updated"],
        summary["records_skipped"],
        summary["source_errors"],
        summary["enriched_count"],
        summary["partially_enriched_count"],
        summary["failed_count"],
        summary["unenriched_count"],
        summary["run_timestamp"],
    )
]

summary_df = spark.createDataFrame(summary_row, schema=GOLD_PIPELINE_SUMMARY_SCHEMA)

if not spark.catalog.tableExists(PIPELINE_SUMMARY_TABLE):
    summary_df.write.format("delta").saveAsTable(PIPELINE_SUMMARY_TABLE)
else:
    summary_df.write.format("delta").mode("append").saveAsTable(PIPELINE_SUMMARY_TABLE)

print(f"Wrote pipeline summary to {PIPELINE_SUMMARY_TABLE}")
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

total_processed = records_added + records_updated + records_skipped
print(f"Total candidate records: {len(candidate_records)}")
print(f"Total processed (added + updated + skipped): {total_processed}")
assert total_processed == len(candidate_records), (
    "records_added + records_updated + records_skipped must equal the "
    "total number of candidate records collected"
)

display(spark.table(BRONZE_LISTINGS_TABLE).limit(10))
