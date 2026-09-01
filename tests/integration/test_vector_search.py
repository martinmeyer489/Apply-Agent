"""Integration test for the Vector Search index sync and query flow.

This is a **live integration test**: it exercises the real Mosaic AI
Vector Search endpoint/index created by `notebooks/04_sync_vector_index.py`
(task 8.1) — `job_agent_vs_endpoint` /
`job_agent.silver.enriched_listings_index` — via
`databricks.vector_search.client.VectorSearchClient`. It cannot run in a
plain local Python environment (no `databricks-vector-search` client
session, no deployed endpoint/index) and is automatically skipped there.

When run against a live workspace, it:

1. Triggers a sync of `job_agent.silver.enriched_listings_index` via
   `.sync()` (Req 4.5 — the index must be resynchronised as
   `silver.enriched_listings` is updated by the Enrichment_Pipeline).
2. Issues a `.similarity_search()` query against the index requesting the
   metadata columns the index is defined to store (`listing_id`,
   `job_title`, `company_name`, `latitude`, `longitude`,
   `enrichment_state` — Req 4.4) and asserts every returned row includes
   all of them.

Requirements: 4.5
"""

from __future__ import annotations

import os

import pytest

from tests.integration._databricks_env import SKIP_REASON, has_databricks_environment

VS_ENDPOINT_NAME = os.environ.get("JOB_AGENT_VS_ENDPOINT", "job_agent_vs_endpoint")
VS_INDEX_NAME = os.environ.get(
    "JOB_AGENT_VS_INDEX", "job_agent.silver.enriched_listings_index"
)

METADATA_COLUMNS = [
    "listing_id",
    "job_title",
    "company_name",
    "latitude",
    "longitude",
    "enrichment_state",
]

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not has_databricks_environment(), reason=SKIP_REASON)
class TestVectorSearchIndex:
    """Sync + query the Delta Sync Vector Search index over enriched listings."""

    @pytest.fixture(scope="class")
    def vs_client(self):
        from databricks.vector_search.client import VectorSearchClient

        return VectorSearchClient()

    @pytest.fixture(scope="class")
    def vs_index(self, vs_client):
        return vs_client.get_index(
            endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME
        )

    def test_sync_then_query_returns_metadata_columns(self, vs_index):
        """A live run of this test would:

        1. Trigger `.sync()` on the Delta Sync index so any pending
           `silver.enriched_listings_chunks` changes are applied
           (Req 4.5).
        2. Run a `.similarity_search()` query for a generic job-search
           phrase, requesting all 6 documented metadata columns
           (Req 4.4), and assert:
           - at least one result is returned (the bundled fallback
             dataset guarantees non-empty indexable content), and
           - every returned row includes all 6 metadata columns
             (Req 4.4/4.5 — the index must expose the columns the
             Matching Agent's `search_listings` UC Function relies on).
        """
        vs_index.sync()

        search_result = vs_index.similarity_search(
            query_text="data engineer python sql",
            columns=METADATA_COLUMNS,
            num_results=5,
        )

        result_rows = search_result["result"]["data_array"]
        returned_columns = [
            column["name"] for column in search_result["manifest"]["columns"]
        ]

        assert len(result_rows) > 0, "expected at least one indexed candidate row"
        for expected_column in METADATA_COLUMNS:
            assert expected_column in returned_columns, (
                f"expected metadata column '{expected_column}' to be present in "
                f"the similarity_search response manifest, got {returned_columns}"
            )

        # Every row must have a value (possibly null for lat/lng on a
        # partially_enriched listing) at each expected column's position.
        for row in result_rows:
            assert len(row) == len(returned_columns), (
                "each result row must have one value per returned column "
                "(including all requested metadata columns)"
            )
