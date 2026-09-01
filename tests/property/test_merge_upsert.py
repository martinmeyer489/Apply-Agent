"""Property-based test for ingestion MERGE upsert idempotence.

Property 2: Ingestion MERGE Upsert Idempotence
Validates: Requirements 2.3

The real ingestion pipeline performs a Delta `MERGE ... ON source_url`
upsert into `bronze.job_listings`, retaining the original `listing_id` on
update and inserting a freshly derived `listing_id` on insert. Since Spark
and Delta Lake are not available in this environment, this test models the
MERGE as a pure-Python simulation operating on an in-memory dict keyed by
`source_url`.
"""

from copy import deepcopy

from hypothesis import given, strategies as st

from src.utils.listing_id import derive_listing_id


def simulate_merge(existing: dict, incoming: list) -> dict:
    """Simulate a `MERGE INTO bronze.job_listings ... ON source_url` upsert.

    For each incoming record:
    - if its `source_url` already exists in `existing`, update all fields
      EXCEPT `listing_id`, which is retained from the pre-merge state.
    - otherwise, insert the record with a freshly derived `listing_id`.

    Args:
        existing: mapping of source_url -> record (as it exists in
            `bronze.job_listings` before the merge).
        incoming: list of incoming records (each a dict with at least a
            `source_url` key) to upsert.

    Returns:
        A new mapping of source_url -> record reflecting the post-merge
        state. `existing` is not mutated.
    """
    state = deepcopy(existing)
    for record in incoming:
        source_url = record["source_url"]
        if source_url in state:
            original_listing_id = state[source_url]["listing_id"]
            updated = deepcopy(record)
            updated["listing_id"] = original_listing_id
            state[source_url] = updated
        else:
            inserted = deepcopy(record)
            inserted["listing_id"] = derive_listing_id(source_url)
            state[source_url] = inserted
    return state


# --- Strategies -------------------------------------------------------------

_source_urls = st.sampled_from(
    [f"https://example.com/jobs/{i}" for i in range(6)]
)

_record_strategy = st.fixed_dictionaries(
    {
        "source_url": _source_urls,
        "job_title": st.text(min_size=0, max_size=30),
        "company_name": st.text(min_size=0, max_size=30),
        "job_description": st.text(min_size=0, max_size=50),
        "location_text": st.text(min_size=0, max_size=30),
    }
)

_incoming_batch = st.lists(_record_strategy, min_size=0, max_size=15)


def _dedupe_by_source_url(records):
    """Collapse a list of records to at most one per source_url, keeping the
    last occurrence, mirroring how a single MERGE statement would apply a
    batch that (in practice) has already been deduplicated on the merge key.
    """
    deduped = {}
    for record in records:
        deduped[record["source_url"]] = record
    return list(deduped.values())


@given(incoming=_incoming_batch)
def test_merge_is_idempotent(incoming):
    """**Validates: Requirements 2.3**

    Applying the same incoming batch twice yields the same final state as
    applying it once (idempotence of the MERGE-on-source_url upsert).
    """
    incoming = _dedupe_by_source_url(incoming)

    once = simulate_merge({}, incoming)
    twice = simulate_merge(once, incoming)

    assert once == twice


@given(
    existing_records=st.lists(_record_strategy, min_size=0, max_size=8),
    incoming=_incoming_batch,
)
def test_merge_retains_listing_id_on_update(existing_records, incoming):
    """**Validates: Requirements 2.3**

    For any record whose source_url already existed before the merge, the
    listing_id after the merge always equals the listing_id before the
    merge -- it never changes on update.
    """
    existing_records = _dedupe_by_source_url(existing_records)
    existing = {}
    for record in existing_records:
        source_url = record["source_url"]
        existing[source_url] = {
            **record,
            "listing_id": derive_listing_id(source_url),
        }

    incoming = _dedupe_by_source_url(incoming)

    pre_merge_listing_ids = {
        source_url: rec["listing_id"] for source_url, rec in existing.items()
    }

    result = simulate_merge(existing, incoming)

    for source_url, pre_listing_id in pre_merge_listing_ids.items():
        assert result[source_url]["listing_id"] == pre_listing_id


@given(incoming=_incoming_batch)
def test_merge_applied_twice_matches_union_of_source_urls(incoming):
    """**Validates: Requirements 2.3**

    Sanity property: the resulting state after (possibly repeated) merges
    always has exactly one entry per distinct source_url present in the
    incoming batch (no duplication, no loss).
    """
    incoming = _dedupe_by_source_url(incoming)

    once = simulate_merge({}, incoming)
    twice = simulate_merge(once, incoming)

    expected_keys = {record["source_url"] for record in incoming}
    assert set(once.keys()) == expected_keys
    assert set(twice.keys()) == expected_keys
