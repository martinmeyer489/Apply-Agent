"""Property-based test for ingestion error resilience.

Property 7: Ingestion Error Resilience
Validates: Requirements 2.5

The real ingestion notebook processes a list of source domains in a loop,
wrapping each domain's fetch/extract logic in a try/except that logs the
error to `bronze.ingestion_errors` and continues to the next domain on
failure. This test models that per-domain try/except/continue pattern in
pure Python.
"""

from hypothesis import given, strategies as st


class SourceProcessingError(Exception):
    """Raised by a fake per-source processing function to simulate a
    source-level failure (e.g. network error, parse error)."""


def process_sources(sources: list, process_fn) -> tuple:
    """Process each source in order, continuing past failures.

    Mirrors the notebook pattern:

        for source in sources:
            try:
                process_fn(source)
                processed.append(source)
            except Exception as exc:
                errors.append((source, exc))
                continue

    Args:
        sources: list of source identifiers to process.
        process_fn: callable invoked once per source; may raise.

    Returns:
        (processed, errors) where `processed` is the list of sources that
        completed successfully (in original order) and `errors` is the list
        of (source, exception) pairs for sources whose processing raised.
    """
    processed = []
    errors = []
    for source in sources:
        try:
            process_fn(source)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            errors.append((source, exc))
            continue
        processed.append(source)
    return processed, errors


@given(
    fail_flags=st.lists(st.booleans(), min_size=0, max_size=30),
)
def test_non_failing_sources_are_still_processed(fail_flags):
    """**Validates: Requirements 2.5**

    All non-failing sources are processed successfully even when other
    sources in the same list raise exceptions.
    """
    sources = list(range(len(fail_flags)))

    def process_fn(source):
        if fail_flags[source]:
            raise SourceProcessingError(f"source {source} failed")

    processed, errors = process_sources(sources, process_fn)

    expected_processed = [s for s in sources if not fail_flags[s]]
    assert processed == expected_processed


@given(
    fail_flags=st.lists(st.booleans(), min_size=0, max_size=30),
)
def test_error_count_matches_number_of_failures(fail_flags):
    """**Validates: Requirements 2.5**

    The number of logged errors exactly equals the number of sources whose
    process_fn raised.
    """
    sources = list(range(len(fail_flags)))

    def process_fn(source):
        if fail_flags[source]:
            raise SourceProcessingError(f"source {source} failed")

    _, errors = process_sources(sources, process_fn)

    expected_error_count = sum(1 for flag in fail_flags if flag)
    assert len(errors) == expected_error_count
    assert {source for source, _ in errors} == {
        s for s in sources if fail_flags[s]
    }


@given(
    fail_flags=st.lists(st.booleans(), min_size=0, max_size=30),
)
def test_processing_order_unaffected_by_failure_position(fail_flags):
    """**Validates: Requirements 2.5**

    Processing continues in the original list order regardless of where
    failures occur -- the relative order of successfully processed sources
    always matches their relative order in the input list.
    """
    sources = list(range(len(fail_flags)))

    def process_fn(source):
        if fail_flags[source]:
            raise SourceProcessingError(f"source {source} failed")

    processed, errors = process_sources(sources, process_fn)

    # processed is a strictly increasing subsequence of `sources`.
    assert processed == sorted(processed)

    # every source appears in exactly one of processed/errors.
    error_sources = {source for source, _ in errors}
    assert set(processed) | error_sources == set(sources)
    assert set(processed).isdisjoint(error_sources)
