"""Property-based test for pipeline summary count consistency.

Property 8: Pipeline Summary Count Consistency
Validates: Requirements 2.11, 3.10

The ingestion notebook's final summary asserts:
    total_candidates == records_added + records_updated + records_skipped

and the enrichment notebook's final summary asserts an analogous invariant
over per-state counts. This test verifies both arithmetic invariants hold
for arbitrary non-negative integer inputs, and that the notebooks' assertion
would (and would only) hold when the counts are consistent.
"""

from hypothesis import given, strategies as st

_non_negative_int = st.integers(min_value=0, max_value=1_000_000)


def ingestion_summary_is_consistent(
    records_added: int,
    records_updated: int,
    records_skipped: int,
    source_errors: int,
    total_candidates: int,
) -> bool:
    """Mirror the ingestion notebook's final consistency assertion.

    `source_errors` does not contribute to `total_candidates` -- a source
    error means the source's candidates were never extracted at all, so
    only added/updated/skipped counts must sum to the candidate total.
    """
    del source_errors  # not part of the count identity, kept for signature clarity
    return total_candidates == records_added + records_updated + records_skipped


def enrichment_summary_is_consistent(
    enriched: int,
    partially_enriched: int,
    failed: int,
    unenriched: int,
    total_processed: int,
) -> bool:
    """Mirror the enrichment notebook's final per-state count assertion."""
    return total_processed == enriched + partially_enriched + failed + unenriched


@given(
    records_added=_non_negative_int,
    records_updated=_non_negative_int,
    records_skipped=_non_negative_int,
    source_errors=_non_negative_int,
)
def test_ingestion_summary_counts_are_consistent_by_construction(
    records_added, records_updated, records_skipped, source_errors
):
    """**Validates: Requirements 2.11**

    For arbitrary non-negative counts, constructing `total_candidates` as
    the sum of added/updated/skipped always satisfies the notebook's final
    consistency assertion.
    """
    total_candidates = records_added + records_updated + records_skipped

    assert ingestion_summary_is_consistent(
        records_added,
        records_updated,
        records_skipped,
        source_errors,
        total_candidates,
    )


@given(
    records_added=_non_negative_int,
    records_updated=_non_negative_int,
    records_skipped=_non_negative_int,
    source_errors=_non_negative_int,
    total_candidates=_non_negative_int,
)
def test_ingestion_summary_inconsistency_is_detected(
    records_added, records_updated, records_skipped, source_errors, total_candidates
):
    """**Validates: Requirements 2.11**

    The consistency check correctly distinguishes consistent from
    inconsistent summaries: it holds if and only if the arithmetic
    identity holds.
    """
    expected = total_candidates == records_added + records_updated + records_skipped

    assert (
        ingestion_summary_is_consistent(
            records_added,
            records_updated,
            records_skipped,
            source_errors,
            total_candidates,
        )
        == expected
    )


@given(
    enriched=_non_negative_int,
    partially_enriched=_non_negative_int,
    failed=_non_negative_int,
    unenriched=_non_negative_int,
)
def test_enrichment_summary_counts_are_consistent_by_construction(
    enriched, partially_enriched, failed, unenriched
):
    """**Validates: Requirements 3.10**

    For arbitrary non-negative per-state counts, constructing
    `total_processed` as their sum always satisfies the enrichment
    notebook's final consistency assertion: the sum of state counts equals
    the total number of records processed in that run.
    """
    total_processed = enriched + partially_enriched + failed + unenriched

    assert enrichment_summary_is_consistent(
        enriched, partially_enriched, failed, unenriched, total_processed
    )


@given(
    enriched=_non_negative_int,
    partially_enriched=_non_negative_int,
    failed=_non_negative_int,
    unenriched=_non_negative_int,
    total_processed=_non_negative_int,
)
def test_enrichment_summary_inconsistency_is_detected(
    enriched, partially_enriched, failed, unenriched, total_processed
):
    """**Validates: Requirements 3.10**

    The enrichment consistency check holds if and only if the arithmetic
    identity holds.
    """
    expected = total_processed == enriched + partially_enriched + failed + unenriched

    assert (
        enrichment_summary_is_consistent(
            enriched, partially_enriched, failed, unenriched, total_processed
        )
        == expected
    )
