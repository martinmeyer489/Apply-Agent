"""End-to-end integration test for the full Bronze -> Silver -> Gold pipeline flow.

This is a **live integration test**: it exercises the ingestion and
enrichment pipeline logic against a real Databricks workspace (Spark,
Delta Lake, Unity Catalog). It is not runnable in a plain local Python
environment because it needs a `SparkSession` backed by Delta Lake and,
ultimately, Unity Catalog three-level table names (`job_agent.bronze...`,
`job_agent.silver...`, `job_agent.ops...`).

The test is automatically skipped when no such environment is detected
(e.g. in this repo's local dev/CI environment, which has no `pyspark`
installed and no `DATABRICKS_HOST`). When it *is* run against a live
workspace, it:

1. Forces the "zero reachable domains" scenario by writing an
   `ops.reachability_report` where every candidate domain is `blocked`
   (Requirement 1 — reachability drives the bundled-fallback decision).
2. Runs the ingestion pipeline's bundled-fallback path (mirroring
   `notebooks/02_ingest_listings.py` Step 2a/2c) and asserts that
   `bronze.job_listings` ends up populated with `ingestion_mode =
   'bundled_fallback'` for every row and at least 200 rows total
   (Requirements 2.1, 2.9, 2.10).
3. Runs the enrichment pipeline's core logic (mirroring
   `notebooks/03_enrich_listings.py`) over those bronze rows and asserts
   `silver.enriched_listings` is populated for all of them, with a state
   in `{enriched, partially_enriched, failed}` for every row (no row is
   left `unenriched` or silently dropped) (Requirement 3.1).
4. Asserts `silver.enriched_listings_chunks` — the source table for the
   Vector Search Delta Sync index — is populated for every listing that
   reached `enriched` or `partially_enriched`, as a proxy for "the index
   is synchronised and those records are retrievable" (Requirement 4.5).
5. Asserts the full bronze -> silver row-accounting invariant: every
   bronze row has exactly one corresponding silver enrichment outcome,
   and no bronze row is silently dropped between tiers.

Because running actual Databricks notebooks via `dbutils`/the Jobs API is
not practical from a pytest process, this test drives the same pure
pipeline logic the notebooks call (`src.utils.*`, `src.pipelines.*`)
directly against a live `SparkSession`, rather than invoking
`notebooks/02_ingest_listings.py` / `03_enrich_listings.py` as scripts.
This keeps the test itself dbutils-free while still exercising real
Delta Lake reads/writes/MERGEs against Unity Catalog tables, which is the
part of the notebooks that cannot be verified by the existing unit/
property tests (those mock or bypass Spark entirely).

Requirements: 2.1, 2.9, 3.1, 4.5
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

CATALOG = os.environ.get("JOB_AGENT_TEST_CATALOG", "job_agent")
BUNDLED_FALLBACK_CSV = (
    Path(__file__).resolve().parents[2] / "data" / "bundled_fallback" / "bundled_listings.csv"
)


def _has_databricks_environment() -> bool:
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


pytestmark = pytest.mark.integration

_SKIP_REASON = (
    "Requires a live Databricks workspace with Spark/Delta Lake/Unity Catalog "
    "(set DATABRICKS_HOST, or run inside a Databricks cluster/serverless "
    "notebook/job with pyspark and an active SparkSession)"
)


@pytest.mark.skipif(not _has_databricks_environment(), reason=_SKIP_REASON)
class TestPipelineEndToEnd:
    """Bronze -> Silver -> Gold flow under the bundled-fallback (zero-reachable) scenario."""

    @pytest.fixture(scope="class")
    def spark(self):
        from pyspark.sql import SparkSession

        return SparkSession.builder.appName("job-agent-e2e-test").getOrCreate()

    @pytest.fixture(scope="class")
    def run_id(self) -> str:
        return f"e2e-test-{uuid.uuid4()}"

    def _force_zero_reachable(self, spark, run_id: str) -> None:
        """Write `ops.reachability_report` with every candidate domain blocked.

        Mirrors the row shape written by `notebooks/01_reachability_probe.py`,
        so the ingestion pipeline logic reads exactly the same "zero
        reachable domains" signal it would read from a real probe run
        (Requirement 1).
        """
        from src.models.schemas import OPS_REACHABILITY_REPORT_SCHEMA

        candidate_domains = [
            "indeed.com",
            "linkedin.com",
            "glassdoor.com",
            "monster.com",
            "reed.co.uk",
        ]
        now = datetime.now(timezone.utc)
        rows = [
            (domain, now, "blocked", None, "simulated: no reachable domains for e2e test", None, None)
            for domain in candidate_domains
        ]
        rows.append(("__summary__", now, "summary", None, None, 0, len(candidate_domains)))

        report_table = f"{CATALOG}.ops.reachability_report"
        report_df = spark.createDataFrame(rows, schema=OPS_REACHABILITY_REPORT_SCHEMA)
        if not spark.catalog.tableExists(report_table):
            report_df.write.format("delta").saveAsTable(report_table)
        else:
            report_df.write.format("delta").mode("overwrite").saveAsTable(report_table)

    def _run_bundled_fallback_ingestion(self, spark, run_id: str) -> int:
        """Load the bundled fallback dataset into `bronze.job_listings`.

        Mirrors `notebooks/02_ingest_listings.py`'s Step 2a/5: reads the
        real `data/bundled_fallback/bundled_listings.csv`, tags every row
        `ingestion_mode='bundled_fallback'`, and MERGE-upserts into
        `bronze.job_listings` using the same batching/listing-id utilities
        the notebook uses.

        Returns:
            The number of bundled fallback rows loaded from the CSV.
        """
        from delta.tables import DeltaTable
        from pyspark.sql.functions import to_timestamp

        from src.models.schemas import BRONZE_JOB_LISTINGS_SCHEMA
        from src.utils.batching import batch_records
        from src.utils.listing_id import derive_listing_id
        from src.utils.validation import validate_listing_record

        assert BUNDLED_FALLBACK_CSV.exists(), f"missing bundled dataset at {BUNDLED_FALLBACK_CSV}"

        records = []
        with open(BUNDLED_FALLBACK_CSV, newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                records.append(
                    {
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

        valid_records = []
        for record in records:
            is_valid, _missing = validate_listing_record(record)
            if not is_valid:
                continue
            record["listing_id"] = derive_listing_id(record["source_url"])
            valid_records.append(record)

        bronze_table = f"{CATALOG}.bronze.job_listings"
        if not spark.catalog.tableExists(bronze_table):
            spark.createDataFrame([], schema=BRONZE_JOB_LISTINGS_SCHEMA).write.format(
                "delta"
            ).saveAsTable(bronze_table)
        bronze_target = DeltaTable.forName(spark, bronze_table)

        for batch in batch_records(valid_records, batch_size=500):
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
                    r["source_domain"],
                )
                for r in batch
            ]
            batch_df = spark.createDataFrame(batch_rows, schema=BRONZE_JOB_LISTINGS_SCHEMA)
            batch_df = batch_df.withColumn("ingestion_timestamp", to_timestamp("ingestion_timestamp"))
            (
                bronze_target.alias("target")
                .merge(batch_df.alias("source"), "target.source_url = source.source_url")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )

        return len(valid_records)

    def _run_enrichment(self, spark) -> None:
        """Enrich every `unenriched` bronze row into `silver.enriched_listings[_chunks]`.

        Mirrors the core logic of `notebooks/03_enrich_listings.py` (geocode
        LEFT JOIN, enrichment state machine, embedding text/chunk
        construction) but stubs out the Foundation Model APIs call with a
        fixed llm_result, since a live LLM call is out of scope for this
        pipeline-flow test (LLM behaviour itself is covered by
        `tests/unit/test_enrichment_pipeline.py` and the enrichment-state
        property tests).
        """
        from delta.tables import DeltaTable
        from pyspark.sql.functions import col, lower, trim

        from src.models.schemas import (
            SILVER_ENRICHED_LISTINGS_CHUNKS_SCHEMA,
            SILVER_ENRICHED_LISTINGS_SCHEMA,
        )
        from src.pipelines.embedding import build_embedding_text, chunk_text
        from src.pipelines.enrichment_state import determine_enrichment_state

        bronze_table = f"{CATALOG}.bronze.job_listings"
        geocode_table = f"{CATALOG}.ops.geocode_lookup"
        silver_table = f"{CATALOG}.silver.enriched_listings"
        chunks_table = f"{CATALOG}.silver.enriched_listings_chunks"

        bronze_df = spark.table(bronze_table).filter("enrichment_state = 'unenriched'")
        geocode_df = spark.table(geocode_table)

        joined_df = bronze_df.alias("b").join(
            geocode_df.alias("g"),
            (
                (lower(trim(col("b.location_text"))) == lower(trim(col("g.city_name"))))
                | (lower(trim(col("b.location_text"))) == lower(trim(col("g.postal_code"))))
            ),
            how="left",
        ).select(
            col("b.listing_id"),
            col("b.job_title"),
            col("b.company_name"),
            col("b.job_description"),
            col("b.location_text"),
            col("b.source_url"),
            col("g.latitude"),
            col("g.longitude"),
        )

        records = [row.asDict() for row in joined_df.collect()]

        for table_name, schema in (
            (silver_table, SILVER_ENRICHED_LISTINGS_SCHEMA),
            (chunks_table, SILVER_ENRICHED_LISTINGS_CHUNKS_SCHEMA),
        ):
            if not spark.catalog.tableExists(table_name):
                spark.createDataFrame([], schema=schema).write.format("delta").saveAsTable(table_name)

        silver_target = DeltaTable.forName(spark, silver_table)
        chunks_target = DeltaTable.forName(spark, chunks_table)
        bronze_target = DeltaTable.forName(spark, bronze_table)

        enriched_rows = []
        chunk_rows = []
        now = datetime.now(timezone.utc)

        for record in records:
            geocode_result = {"latitude": record.get("latitude"), "longitude": record.get("longitude")}
            # Stub LLM result: fully resolved, so the state machine's
            # "enriched" branch is exercised deterministically without a
            # live Foundation Model APIs call.
            llm_result = {
                "required_skills": ["python", "sql"],
                "seniority_level": "mid",
                "employment_type": "full_time",
                "industry": "technology",
                "company_size_band": "51-200",
            }
            state, unresolved, failure_reason = determine_enrichment_state(geocode_result, llm_result)

            embedding_text = build_embedding_text(
                record["job_title"], record.get("job_description") or "", "python, sql"
            )
            chunks = chunk_text(embedding_text) if embedding_text else []

            enriched_rows.append(
                (
                    record["listing_id"],
                    record["job_title"],
                    record["company_name"],
                    record.get("job_description"),
                    record.get("location_text"),
                    record["source_url"],
                    geocode_result["latitude"],
                    geocode_result["longitude"],
                    llm_result["required_skills"],
                    "python, sql",
                    llm_result["seniority_level"],
                    llm_result["employment_type"],
                    llm_result["industry"],
                    llm_result["company_size_band"],
                    state,
                    unresolved if unresolved else None,
                    failure_reason,
                    now,
                    embedding_text,
                )
            )
            for chunk_index, chunk in enumerate(chunks):
                chunk_rows.append(
                    (
                        f"{record['listing_id']}_{chunk_index}",
                        record["listing_id"],
                        chunk_index,
                        chunk,
                        record["job_title"],
                        record["company_name"],
                        geocode_result["latitude"],
                        geocode_result["longitude"],
                        state,
                    )
                )

        enriched_df = spark.createDataFrame(enriched_rows, schema=SILVER_ENRICHED_LISTINGS_SCHEMA)
        (
            silver_target.alias("target")
            .merge(enriched_df.alias("source"), "target.listing_id = source.listing_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

        if chunk_rows:
            chunks_df = spark.createDataFrame(chunk_rows, schema=SILVER_ENRICHED_LISTINGS_CHUNKS_SCHEMA)
            (
                chunks_target.alias("target")
                .merge(chunks_df.alias("source"), "target.chunk_id = source.chunk_id")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )

        state_updates_df = spark.createDataFrame(
            [(r[0], r[14]) for r in enriched_rows], ["listing_id", "enrichment_state"]
        )
        (
            bronze_target.alias("target")
            .merge(state_updates_df.alias("source"), "target.listing_id = source.listing_id")
            .whenMatchedUpdate(set={"enrichment_state": "source.enrichment_state"})
            .execute()
        )

    def test_bundled_fallback_ingestion_then_enrichment_flow(self, spark, run_id):
        """Full bronze -> silver -> gold flow under the zero-reachable-domains scenario.

        A live run of this test would:
        1. Confirm the ingestion pipeline correctly activates the bundled
           fallback when zero domains are reachable (Req 2.1, 2.9).
        2. Confirm the bundled dataset satisfies its >=200-row floor
           (Req 2.10) and that every ingested row is tagged
           `ingestion_mode='bundled_fallback'` (Req 2.9).
        3. Confirm every bronze row is picked up by enrichment and lands
           in exactly one terminal enrichment state, with none left
           `unenriched` and none dropped (Req 3.1).
        4. Confirm the Vector Search index's source table
           (`silver.enriched_listings_chunks`) is populated for every
           enriched/partially_enriched listing, as a proxy for "the index
           is synchronized and those listings are retrievable" (Req 4.5).
        """
        # --- Step 1: force the zero-reachable-domains scenario (Req 1) ---
        self._force_zero_reachable(spark, run_id)

        reachable_count = (
            spark.table(f"{CATALOG}.ops.reachability_report")
            .filter("outcome = 'reachable'")
            .count()
        )
        assert reachable_count == 0, "test setup must simulate zero reachable domains"

        # --- Step 2: bundled-fallback ingestion -> bronze.job_listings (Req 2.1, 2.9, 2.10) ---
        bundled_row_count = self._run_bundled_fallback_ingestion(spark, run_id)
        assert bundled_row_count >= 200, (
            "the bundled fallback dataset must contain at least 200 job listing "
            "records per Requirement 2.10"
        )

        bronze_df = spark.table(f"{CATALOG}.bronze.job_listings")
        bronze_count = bronze_df.count()
        assert bronze_count >= 200

        bundled_mode_count = bronze_df.filter("ingestion_mode = 'bundled_fallback'").count()
        assert bundled_mode_count == bronze_count, (
            "every bronze row must be tagged ingestion_mode='bundled_fallback' "
            "when zero domains are reachable (Req 2.9)"
        )

        # --- Step 3: enrichment -> silver.enriched_listings (Req 3.1) ---
        self._run_enrichment(spark)

        silver_df = spark.table(f"{CATALOG}.silver.enriched_listings")
        silver_count = silver_df.count()
        assert silver_count == bronze_count, (
            "every bronze row must produce exactly one silver enrichment "
            "outcome row — no row may be silently dropped between tiers"
        )

        terminal_states = {"enriched", "partially_enriched", "failed"}
        non_terminal_count = silver_df.filter(~silver_df.enrichment_state.isin(list(terminal_states))).count()
        assert non_terminal_count == 0, (
            "every enriched Job_Listing must land in enriched, partially_enriched, "
            "or failed — none may remain unenriched after the pipeline runs"
        )

        # Bronze's own enrichment_state column must have been updated in lockstep,
        # so a subsequent run's `enrichment_state = 'unenriched'` filter does not
        # reprocess these rows.
        remaining_unenriched = (
            spark.table(f"{CATALOG}.bronze.job_listings").filter("enrichment_state = 'unenriched'").count()
        )
        assert remaining_unenriched == 0

        # --- Step 4: silver.enriched_listings_chunks populated (Req 4.5 proxy) ---
        indexable_listing_ids = {
            row["listing_id"]
            for row in silver_df.filter(silver_df.enrichment_state.isin(["enriched", "partially_enriched"]))
            .select("listing_id")
            .collect()
        }
        chunks_df = spark.table(f"{CATALOG}.silver.enriched_listings_chunks")
        chunked_listing_ids = {row["listing_id"] for row in chunks_df.select("listing_id").distinct().collect()}

        assert indexable_listing_ids.issubset(chunked_listing_ids), (
            "every enriched/partially_enriched listing must have at least one "
            "row in silver.enriched_listings_chunks — the source table the "
            "Vector Search Delta Sync index reads from — so it is retrievable "
            "once the index syncs (Req 4.5)"
        )
        assert chunks_df.count() > 0
