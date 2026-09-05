# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Health Check (Deprecated)
# MAGIC
# MAGIC **This notebook is deprecated.** The ingestion pipeline (notebook 02) now 
# MAGIC performs its own health check against the Jobsuche API and falls back to 
# MAGIC bundled data if the API is unreachable.
# MAGIC
# MAGIC This notebook is kept as a placeholder to maintain workflow task IDs.
# MAGIC The workflow no longer runs this notebook.
# MAGIC
# MAGIC Requirements: (deprecated - functionality moved to 02_ingest_listings.py)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")

CATALOG = dbutils.widgets.get("catalog")

print(f"Catalog: {CATALOG}")
print("This notebook is deprecated. The ingestion pipeline handles API reachability internally.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## No-op
# MAGIC
# MAGIC This cell exists only to make this a valid Databricks notebook.

# COMMAND ----------

print("Deprecated reachability probe - no action needed.")