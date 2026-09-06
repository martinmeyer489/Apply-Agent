# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Catalog and Table Setup
# MAGIC
# MAGIC Creates the `job_agent` Unity Catalog catalog, its schemas
# MAGIC (`bronze`, `silver`, `gold`, `ops`, `volumes`), the three managed
# MAGIC Volumes used for offline reference data and file uploads
# MAGIC (`bundled_fallback`, `geocode_lookup`, `cv_uploads`), and every Delta
# MAGIC table defined in `src/models/schemas.py` (`TABLE_SCHEMAS`).
# MAGIC
# MAGIC All operations are idempotent (`IF NOT EXISTS` semantics) and every
# MAGIC asset is addressed via its Unity Catalog three-level name
# MAGIC (`job_agent.<schema>.<object>`).
# MAGIC
# MAGIC Requirements: 11.2

# COMMAND ----------

import sys
import os

# Allow `from src.models.schemas import TABLE_SCHEMAS` when this notebook is
# run from the Databricks Workspace (repo root is not automatically on
# sys.path there, unlike when running via `databricks bundle` sync or repos).
_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_REPO_ROOT = os.path.abspath(os.path.join(_NOTEBOOK_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.schemas import TABLE_SCHEMAS  # noqa: E402

# COMMAND ----------

CATALOG_NAME = "job_agent"
SCHEMAS = ["bronze", "silver", "gold", "ops", "volumes"]
VOLUMES = ["bundled_fallback", "geocode_lookup", "cv_uploads"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create catalog

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create schemas
# MAGIC
# MAGIC `bronze`, `silver`, `gold`, and `ops` hold Delta tables. `volumes`
# MAGIC groups the managed Volumes per the catalog layout in the design
# MAGIC document.

# COMMAND ----------

for schema_name in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Volumes
# MAGIC
# MAGIC - `job_agent.volumes.bundled_fallback` — bundled job-listing CSV used
# MAGIC   when no source domain is reachable.
# MAGIC - `job_agent.volumes.geocode_lookup` — city/postal-code → lat/lng CSV
# MAGIC   used for offline geocoding.
# MAGIC - `job_agent.volumes.cv_uploads` — user-uploaded CV files (PDF/DOCX).

# COMMAND ----------

for volume_name in VOLUMES:
    spark.sql(
        f"CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.volumes.{volume_name}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Delta tables
# MAGIC
# MAGIC Iterates over `TABLE_SCHEMAS` (keyed by `"<schema>.<table>"`, e.g.
# MAGIC `"bronze.job_listings"`) from `src/models/schemas.py` and creates each
# MAGIC table addressed by its full three-level name
# MAGIC (`job_agent.<schema>.<table>`) if it does not already exist. Creating
# MAGIC from an empty typed DataFrame preserves the exact PySpark schema
# MAGIC (including nested `ARRAY<STRING>` columns) without hand-written DDL.

# COMMAND ----------

for table_key, table_schema in TABLE_SCHEMAS.items():
    full_table_name = f"{CATALOG_NAME}.{table_key}"
    if not spark.catalog.tableExists(full_table_name):
        empty_df = spark.createDataFrame([], schema=table_schema)
        empty_df.write.format("delta").saveAsTable(full_table_name)
        print(f"Created table {full_table_name}")
    else:
        print(f"Table {full_table_name} already exists, skipping")

# The Vector Search Delta Sync index is built over
# `silver.enriched_listings_chunks`, which requires row-level change tracking
# on its source table. Enable Change Data Feed (idempotent) so notebook 04's
# create_delta_sync_index call succeeds.
spark.sql(
    f"ALTER TABLE {CATALOG_NAME}.silver.enriched_listings_chunks "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Enabled Change Data Feed on {CATALOG_NAME}.silver.enriched_listings_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

created_tables = [f"{CATALOG_NAME}.{table_key}" for table_key in TABLE_SCHEMAS]
print(f"Catalog: {CATALOG_NAME}")
print(f"Schemas: {[f'{CATALOG_NAME}.{s}' for s in SCHEMAS]}")
print(f"Volumes: {[f'{CATALOG_NAME}.volumes.{v}' for v in VOLUMES]}")
print(f"Tables ({len(created_tables)}):")
for t in created_tables:
    print(f"  - {t}")
