# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Vector Search Setup and Sync
# MAGIC
# MAGIC Creates the Mosaic AI Vector Search endpoint `job_agent_vs_endpoint`
# MAGIC and the Delta Sync index `job_agent.silver.enriched_listings_index`
# MAGIC (TRIGGERED pipeline) over `silver.enriched_listings_chunks`, embedding
# MAGIC the `embedding_text` column via the Foundation Model APIs embedding
# MAGIC model `databricks-gte-large-en`. Both endpoint and index creation are
# MAGIC idempotent — if either already exists, creation is skipped and, for
# MAGIC the index, a sync is triggered instead so newly enriched records
# MAGIC become retrievable (Req 4 AC5).
# MAGIC
# MAGIC The retrievable metadata columns synced alongside the embedding are
# MAGIC `listing_id`, `job_title`, `company_name`, `latitude`, `longitude`, and
# MAGIC `enrichment_state` (Req 4 AC4).
# MAGIC
# MAGIC If endpoint/index creation or sync fails, the error is logged to
# MAGIC `ops.indexing_errors` with a timestamp, and the previously synchronised
# MAGIC index contents are retained — a Delta Sync index is a standing Unity
# MAGIC Catalog object whose most recently synced rows remain queryable until
# MAGIC a *subsequent* sync succeeds, so a failed sync attempt (or a failure to
# MAGIC even reach the sync call, e.g. because a create attempt raised first)
# MAGIC never deletes or replaces the existing index (Req 4 AC6).
# MAGIC
# MAGIC Note (Req 4 AC7): the Vector Search index itself, and the
# MAGIC `search_listings` UC Function that queries it (see
# MAGIC `notebooks/06_create_uc_functions.py`), are responsible for enforcing
# MAGIC the 200 max-results-per-query constraint. This notebook only creates
# MAGIC and synchronises the index — it issues no similarity-search queries,
# MAGIC so there is nothing to bound here.
# MAGIC
# MAGIC **Prerequisite**: `notebooks/00_setup_catalog.py` and
# MAGIC `notebooks/03_enrich_listings.py` must have already run so that
# MAGIC `silver.enriched_listings_chunks` exists and is populated.
# MAGIC
# MAGIC Requirements: 4.1, 4.2, 4.4, 4.5, 4.6, 4.7

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

from src.models.schemas import OPS_INDEXING_ERRORS_SCHEMA  # noqa: E402

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("vs_endpoint_name", "job_agent_vs_endpoint", "Vector Search endpoint name")
dbutils.widgets.text("index_name", "silver.enriched_listings_index", "Index name (schema.table, catalog prepended)")
dbutils.widgets.text("source_table", "silver.enriched_listings_chunks", "Source table (schema.table, catalog prepended)")
dbutils.widgets.text("embedding_model_endpoint", "databricks-gte-large-en", "Foundation Model APIs embedding endpoint")

CATALOG = dbutils.widgets.get("catalog")
VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name")
INDEX_NAME = f"{CATALOG}.{dbutils.widgets.get('index_name')}"
SOURCE_TABLE_NAME = f"{CATALOG}.{dbutils.widgets.get('source_table')}"
EMBEDDING_MODEL_ENDPOINT = dbutils.widgets.get("embedding_model_endpoint")

INDEXING_ERRORS_TABLE = f"{CATALOG}.ops.indexing_errors"

# Columns synced as retrievable metadata alongside the embedding (Req 4 AC4).
COLUMNS_TO_SYNC = [
    "listing_id",
    "job_title",
    "company_name",
    "latitude",
    "longitude",
    "enrichment_state",
]

# The primary key of the source (chunk-level) table, and the column whose
# text is embedded via the Foundation Model APIs embedding endpoint
# (Req 4 AC2 — job_title + job_description + required_skills concatenation,
# pre-built and pre-chunked into this column by notebooks/03_enrich_listings.py).
PRIMARY_KEY_COLUMN = "chunk_id"
EMBEDDING_SOURCE_COLUMN = "embedding_text"

print(f"Vector Search endpoint: {VS_ENDPOINT_NAME}")
print(f"Index name:             {INDEX_NAME}")
print(f"Source table:           {SOURCE_TABLE_NAME}")
print(f"Embedding model:        {EMBEDDING_MODEL_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Error logging helper (Req 4.6)
# MAGIC
# MAGIC Any failure during endpoint creation, index creation, or sync is
# MAGIC appended to `ops.indexing_errors` with a UUID `error_id` and a UTC
# MAGIC timestamp. Logging a failure never touches — and therefore never
# MAGIC drops or replaces — the Vector Search index itself, so previously
# MAGIC synchronised index contents are retained.

# COMMAND ----------

import uuid
from datetime import datetime, timezone


def log_indexing_error(error_message: str) -> None:
    """Append a failure record to `ops.indexing_errors` (Req 4 AC6)."""
    error_row = [(str(uuid.uuid4()), error_message, datetime.now(timezone.utc))]
    error_df = spark.createDataFrame(error_row, schema=OPS_INDEXING_ERRORS_SCHEMA)
    if not spark.catalog.tableExists(INDEXING_ERRORS_TABLE):
        error_df.write.format("delta").saveAsTable(INDEXING_ERRORS_TABLE)
    else:
        error_df.write.format("delta").mode("append").saveAsTable(INDEXING_ERRORS_TABLE)
    print(f"Logged indexing error to {INDEXING_ERRORS_TABLE}: {error_message}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Idempotent endpoint creation (Req 4.1)
# MAGIC
# MAGIC Checks whether `VS_ENDPOINT_NAME` already exists via `get_endpoint`
# MAGIC before creating it. Creation is skipped if the endpoint is already
# MAGIC present, so re-running this notebook never raises an "already exists"
# MAGIC error and never attempts to recreate a live endpoint.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()


def endpoint_exists(endpoint_name: str) -> bool:
    try:
        vsc.get_endpoint(endpoint_name)
        return True
    except Exception:  # noqa: BLE001 - any lookup failure means "not found"
        return False


try:
    if endpoint_exists(VS_ENDPOINT_NAME):
        print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists, skipping creation")
    else:
        vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
        print(f"Created endpoint '{VS_ENDPOINT_NAME}'")
except Exception as exc:  # noqa: BLE001 - log and re-raise so the task fails visibly
    log_indexing_error(f"Endpoint creation failed for '{VS_ENDPOINT_NAME}': {exc}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Idempotent index creation, or triggered sync (Req 4.1, 4.2, 4.4, 4.5, 4.6)
# MAGIC
# MAGIC Checks whether `INDEX_NAME` already exists via `get_index`:
# MAGIC - **Not found** → create the Delta Sync index (`pipeline_type="TRIGGERED"`)
# MAGIC   over `SOURCE_TABLE_NAME`, embedding `embedding_text` via
# MAGIC   `EMBEDDING_MODEL_ENDPOINT`, syncing `COLUMNS_TO_SYNC` as retrievable
# MAGIC   metadata. A newly created TRIGGERED index performs its first sync
# MAGIC   automatically.
# MAGIC - **Already exists** → this run represents a subsequent trigger
# MAGIC   (e.g. after a new enrichment run wrote new/updated chunk rows), so
# MAGIC   trigger a sync instead of attempting to recreate the index
# MAGIC   (Req 4 AC5).
# MAGIC
# MAGIC Any exception raised by either path is logged to `ops.indexing_errors`
# MAGIC (Req 4 AC6). Because neither the failed `create_delta_sync_index` call
# MAGIC nor a failed `.sync()` call deletes the index object, the previously
# MAGIC synchronised contents (if any) remain queryable.

# COMMAND ----------

def index_exists(endpoint_name: str, index_name: str) -> bool:
    try:
        vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        return True
    except Exception:  # noqa: BLE001 - any lookup failure means "not found"
        return False


# A Delta Sync index requires its source table to expose row-level changes.
# Enable Change Data Feed on the chunks table (idempotent) before creating
# the index, otherwise create_delta_sync_index fails with:
#   "... is not a valid Vector Search source. Please retry after enabling
#    change data feed (delta.enableChangeDataFeed = true) ..."
spark.sql(
    f"ALTER TABLE {SOURCE_TABLE_NAME} "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Ensured Change Data Feed is enabled on {SOURCE_TABLE_NAME}")


try:
    if index_exists(VS_ENDPOINT_NAME, INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists, triggering sync")
        vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=INDEX_NAME).sync()
        print(f"Sync triggered for '{INDEX_NAME}'")
    else:
        vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=INDEX_NAME,
            source_table_name=SOURCE_TABLE_NAME,
            pipeline_type="TRIGGERED",
            primary_key=PRIMARY_KEY_COLUMN,
            embedding_source_column=EMBEDDING_SOURCE_COLUMN,
            embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT,
            columns_to_sync=COLUMNS_TO_SYNC,
        )
        print(f"Created Delta Sync index '{INDEX_NAME}' over '{SOURCE_TABLE_NAME}'")
except Exception as exc:  # noqa: BLE001 - log; previous index contents (if any) are retained
    log_indexing_error(f"Index creation/sync failed for '{INDEX_NAME}': {exc}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

summary = {
    "vs_endpoint_name": VS_ENDPOINT_NAME,
    "index_name": INDEX_NAME,
    "source_table": SOURCE_TABLE_NAME,
    "embedding_model_endpoint": EMBEDDING_MODEL_ENDPOINT,
    "columns_to_sync": COLUMNS_TO_SYNC,
    "primary_key": PRIMARY_KEY_COLUMN,
    "embedding_source_column": EMBEDDING_SOURCE_COLUMN,
}
print(summary)

if spark.catalog.tableExists(INDEXING_ERRORS_TABLE):
    display(spark.table(INDEXING_ERRORS_TABLE).orderBy("error_timestamp", ascending=False).limit(10))
