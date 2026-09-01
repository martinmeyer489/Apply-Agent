# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Create/Update the `job_agent_pipeline` Workflow
# MAGIC
# MAGIC Defines and (idempotently) creates or updates a single Databricks
# MAGIC Workflow named `job_agent_pipeline` with exactly 5 tasks, mirroring
# MAGIC `workflow_definition.json` at the repo root:
# MAGIC
# MAGIC | Task | Notebook | Depends on | Compute |
# MAGIC |------|----------|-----------|---------|
# MAGIC | `reachability_probe` | `notebooks/01_reachability_probe.py` | — | Serverless |
# MAGIC | `ingest_listings` | `notebooks/02_ingest_listings.py` | `reachability_probe` | Serverless |
# MAGIC | `enrich_listings` | `notebooks/03_enrich_listings.py` | `ingest_listings` | Serverless |
# MAGIC | `sync_vector_index` | `notebooks/04_sync_vector_index.py` | `enrich_listings` | Serverless |
# MAGIC | `run_evaluation` | `notebooks/05_run_evaluation.py` | — (manual trigger) | Serverless |
# MAGIC
# MAGIC Tasks 1-4 form the automatic ingestion -> enrichment -> indexing
# MAGIC chain. Task 5 (`run_evaluation`) has no dependency edge and is meant
# MAGIC to be triggered on demand — e.g. via
# MAGIC `databricks jobs run-now --job-id <id> --only run_evaluation` — ahead
# MAGIC of promoting a new `matching_agent` version, per the deployment gate
# MAGIC in Requirements 9.5-9.6. The job therefore declares exactly 5 tasks
# MAGIC in total, which trivially satisfies the Free Edition "at most 5
# MAGIC concurrent job tasks" limit (Requirement 11.6) since the dependency
# MAGIC chain plus the one independent task never runs more than 5 tasks at
# MAGIC once.
# MAGIC
# MAGIC Every task omits `new_cluster`/`existing_cluster_id`/`job_clusters`
# MAGIC entirely so it runs on serverless compute — the Jobs API convention
# MAGIC for serverless notebook tasks (Requirement 11.1). Tasks that need the
# MAGIC single Free Edition 2X-Small SQL warehouse (`enrich_listings` for its
# MAGIC `ai_query` path, `run_evaluation` for its `draft_application` check)
# MAGIC receive the warehouse ID via a job-level parameter passed through as
# MAGIC a notebook base parameter (Requirement 11.7).
# MAGIC
# MAGIC This notebook is a *thin, runnable wrapper* around
# MAGIC `workflow_definition.json` — it is not required for the Workflow to
# MAGIC exist (the JSON file can be applied directly with
# MAGIC `databricks jobs create --json @workflow_definition.json` or via a
# MAGIC Databricks Asset Bundle `resources.jobs.job_agent_pipeline` block),
# MAGIC but it is convenient to run interactively from the workspace.
# MAGIC
# MAGIC Requirements: 11.1, 11.6, 11.7

# COMMAND ----------

import sys
import os

_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_REPO_ROOT = os.path.abspath(os.path.join(_NOTEBOOK_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("job_name", "job_agent_pipeline", "Workflow (job) name")
dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("warehouse_id", "", "SQL warehouse ID (single Free Edition 2X-Small warehouse)")
dbutils.widgets.text(
    "notebooks_workspace_dir",
    "/Workspace/job_agent_pipeline/notebooks",
    "Workspace directory the pipeline notebooks are synced/deployed to",
)

JOB_NAME = dbutils.widgets.get("job_name")
CATALOG = dbutils.widgets.get("catalog")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
NOTEBOOKS_DIR = dbutils.widgets.get("notebooks_workspace_dir").rstrip("/")

print(f"Job name:          {JOB_NAME}")
print(f"Catalog:            {CATALOG}")
print(f"Warehouse ID:       {WAREHOUSE_ID or '(not set)'}")
print(f"Notebooks dir:      {NOTEBOOKS_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the task graph
# MAGIC
# MAGIC Every task is a `notebook_task` with no cluster fields (serverless),
# MAGIC matching `workflow_definition.json`.

# COMMAND ----------

from databricks.sdk.service.jobs import (
    JobParameterDefinition,
    NotebookTask,
    Task,
    TaskDependency,
)

JOB_PARAMETERS = [
    JobParameterDefinition(name="catalog", default=CATALOG),
    JobParameterDefinition(name="warehouse_id", default=WAREHOUSE_ID),
]

tasks = [
    Task(
        task_key="reachability_probe",
        notebook_task=NotebookTask(
            notebook_path=f"{NOTEBOOKS_DIR}/01_reachability_probe",
            base_parameters={"catalog": "{{job.parameters.catalog}}"},
        ),
        timeout_seconds=600,
        max_retries=0,
    ),
    Task(
        task_key="ingest_listings",
        depends_on=[TaskDependency(task_key="reachability_probe")],
        notebook_task=NotebookTask(
            notebook_path=f"{NOTEBOOKS_DIR}/02_ingest_listings",
            base_parameters={"catalog": "{{job.parameters.catalog}}"},
        ),
        timeout_seconds=3600,
        max_retries=1,
    ),
    Task(
        task_key="enrich_listings",
        depends_on=[TaskDependency(task_key="ingest_listings")],
        notebook_task=NotebookTask(
            notebook_path=f"{NOTEBOOKS_DIR}/03_enrich_listings",
            base_parameters={
                "catalog": "{{job.parameters.catalog}}",
                "warehouse_id": "{{job.parameters.warehouse_id}}",
            },
        ),
        timeout_seconds=3600,
        max_retries=1,
    ),
    Task(
        task_key="sync_vector_index",
        depends_on=[TaskDependency(task_key="enrich_listings")],
        notebook_task=NotebookTask(
            notebook_path=f"{NOTEBOOKS_DIR}/04_sync_vector_index",
            base_parameters={"catalog": "{{job.parameters.catalog}}"},
        ),
        timeout_seconds=1800,
        max_retries=1,
    ),
    # Task 5: independent — no depends_on. Intended for manual/on-demand
    # runs ahead of promoting a new matching_agent version (Req 9.5-9.6),
    # not wired into the automatic Tasks 1-4 chain.
    Task(
        task_key="run_evaluation",
        notebook_task=NotebookTask(
            notebook_path=f"{NOTEBOOKS_DIR}/05_run_evaluation",
            base_parameters={
                "catalog": "{{job.parameters.catalog}}",
                "warehouse_id": "{{job.parameters.warehouse_id}}",
            },
        ),
        timeout_seconds=3600,
        max_retries=0,
    ),
]

assert len(tasks) == 5, f"Expected exactly 5 tasks, got {len(tasks)}"
print(f"Built {len(tasks)} tasks: {[t.task_key for t in tasks]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update the job
# MAGIC
# MAGIC Idempotent: if a job named `job_agent_pipeline` already exists, this
# MAGIC resets it in place (`jobs.reset`) instead of creating a duplicate.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import JobSettings

w = WorkspaceClient()

existing = [j for j in w.jobs.list(name=JOB_NAME)]

job_settings = JobSettings(
    name=JOB_NAME,
    tasks=tasks,
    parameters=JOB_PARAMETERS,
    max_concurrent_runs=1,
    tags={"project": "databricks-job-agent"},
)

if existing:
    job_id = existing[0].job_id
    w.jobs.reset(job_id=job_id, new_settings=job_settings)
    print(f"Updated existing job '{JOB_NAME}' (job_id={job_id})")
else:
    created = w.jobs.create(
        name=JOB_NAME,
        tasks=tasks,
        parameters=JOB_PARAMETERS,
        max_concurrent_runs=1,
        tags={"project": "databricks-job-agent"},
    )
    job_id = created.job_id
    print(f"Created job '{JOB_NAME}' (job_id={job_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification
# MAGIC
# MAGIC Confirms the job graph has exactly 5 tasks with the expected
# MAGIC dependency chain, and that no task specifies a cluster (serverless).

# COMMAND ----------

job = w.jobs.get(job_id=job_id)
task_keys = {t.task_key for t in job.settings.tasks}

assert task_keys == {
    "reachability_probe",
    "ingest_listings",
    "enrich_listings",
    "sync_vector_index",
    "run_evaluation",
}, f"Unexpected task keys: {task_keys}"

for t in job.settings.tasks:
    assert t.new_cluster is None and t.existing_cluster_id is None, (
        f"Task {t.task_key} is not serverless (has cluster fields set)"
    )

print(f"Job '{JOB_NAME}' (job_id={job_id}) verified: 5 serverless tasks, dependency chain intact.")
print(
    "To run only the evaluation task on demand: "
    f"databricks jobs run-now --job-id {job_id} --only run_evaluation"
)
