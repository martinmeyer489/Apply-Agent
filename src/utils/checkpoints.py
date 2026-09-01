"""Batch checkpoint read/write utility for pipeline resume support.

Pipelines (ingestion, enrichment) record their progress in
`job_agent.ops.batch_checkpoints` after each successfully processed batch.
On restart, a pipeline resumes from `last_successful_batch + 1` rather than
re-processing already-completed batches.

Validates: Requirements 11.8, 11.9
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

CHECKPOINT_TABLE = "job_agent.ops.batch_checkpoints"


def read_checkpoint(spark, pipeline_name: str, run_id: str) -> Optional[int]:
    """Read the last successful batch index for a pipeline run.

    Queries `job_agent.ops.batch_checkpoints` for rows matching
    `pipeline_name` and `run_id`, and returns the `last_successful_batch`
    of the most recent checkpoint (ordered by `checkpoint_timestamp` desc).

    Args:
        spark: The active SparkSession.
        pipeline_name: Name of the pipeline (e.g. "ingestion", "enrichment").
        run_id: The Workflow run ID (or equivalent) scoping the checkpoint.

    Returns:
        The `last_successful_batch` value of the most recent checkpoint
        row, or `None` if no checkpoint exists for this pipeline/run.
    """
    checkpoints_df = spark.table(CHECKPOINT_TABLE)
    rows = (
        checkpoints_df.filter(
            (checkpoints_df.pipeline_name == pipeline_name)
            & (checkpoints_df.run_id == run_id)
        )
        .orderBy("checkpoint_timestamp", ascending=False)
        .limit(1)
        .collect()
    )

    if not rows:
        return None

    return rows[0]["last_successful_batch"]


def write_checkpoint(spark, pipeline_name: str, run_id: str, batch_index: int) -> None:
    """Write a checkpoint recording the last successfully processed batch.

    Appends a new row to `job_agent.ops.batch_checkpoints` with the given
    `pipeline_name`, `run_id`, `batch_index` (as `last_successful_batch`),
    and the current UTC timestamp.

    Args:
        spark: The active SparkSession.
        pipeline_name: Name of the pipeline (e.g. "ingestion", "enrichment").
        run_id: The Workflow run ID (or equivalent) scoping the checkpoint.
        batch_index: The 0-based index of the batch that just completed
            successfully.
    """
    checkpoint_row = [
        (pipeline_name, run_id, batch_index, datetime.now(timezone.utc))
    ]
    columns = ["pipeline_name", "run_id", "last_successful_batch", "checkpoint_timestamp"]

    checkpoint_df = spark.createDataFrame(checkpoint_row, columns)
    checkpoint_df.write.format("delta").mode("append").saveAsTable(CHECKPOINT_TABLE)


def get_resume_batch(spark, pipeline_name: str, run_id: str) -> int:
    """Determine the batch index a pipeline run should resume from.

    Args:
        spark: The active SparkSession.
        pipeline_name: Name of the pipeline (e.g. "ingestion", "enrichment").
        run_id: The Workflow run ID (or equivalent) scoping the checkpoint.

    Returns:
        `last_successful_batch + 1` if a checkpoint exists, otherwise `0`.
    """
    last_successful_batch = read_checkpoint(spark, pipeline_name, run_id)

    if last_successful_batch is None:
        return 0

    return last_successful_batch + 1
