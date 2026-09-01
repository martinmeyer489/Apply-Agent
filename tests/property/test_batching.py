"""Property-based tests for the batching utility.

Property 6: Pipeline Batch Size Invariant
Validates: Requirements 2.8, 3.9, 11.8
"""

from hypothesis import given, strategies as st

from src.utils.batching import batch_records


@given(
    records=st.lists(st.integers(), max_size=2000),
    batch_size=st.integers(min_value=1, max_value=1000),
)
def test_batch_size_invariant(records, batch_size):
    """**Validates: Requirements 2.8, 3.9, 11.8**

    For arbitrary lists of records and batch sizes:
    - every batch has length <= batch_size
    - concatenating all batches in order reproduces the original list exactly
    - no records are lost or duplicated
    """
    batches = batch_records(records, batch_size=batch_size)

    # Every batch respects the size cap.
    for batch in batches:
        assert len(batch) <= batch_size

    # Concatenating batches in order reproduces the original list exactly
    # (implies no records lost, duplicated, or reordered).
    flattened = [item for batch in batches for item in batch]
    assert flattened == list(records)


@given(records=st.lists(st.integers(), max_size=2000))
def test_batch_size_invariant_default_batch_size(records):
    """Same invariant using the default batch_size=500."""
    batches = batch_records(records)

    for batch in batches:
        assert len(batch) <= 500

    flattened = [item for batch in batches for item in batch]
    assert flattened == list(records)
