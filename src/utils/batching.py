"""Batching utility for splitting records into fixed-size chunks.

Used by the Ingestion_Pipeline and Enrichment_Pipeline to process records
in batches of at most `batch_size` records per batch (Requirements 2.8, 3.9, 11.8).
"""

from typing import Any, List, Sequence


def batch_records(records: Sequence[Any], batch_size: int = 500) -> List[List[Any]]:
    """Split a sequence of records into a list of batches.

    Args:
        records: The full sequence of records to split into batches.
        batch_size: The maximum number of records per batch. Must be a
            positive integer.

    Returns:
        A list of batches, where each batch is a list containing at most
        `batch_size` records. Returns an empty list if `records` is empty.

    Raises:
        ValueError: If `batch_size` is not a positive integer.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    return [
        list(records[i : i + batch_size])
        for i in range(0, len(records), batch_size)
    ]
