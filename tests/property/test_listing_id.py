"""Property-based tests for listing ID derivation.

**Validates: Requirements 2.2**
"""

import re

from hypothesis import given, strategies as st

from utils.listing_id import derive_listing_id

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


@given(st.text())
def test_derive_listing_id_is_deterministic(source_url: str) -> None:
    """Property 1 (part 1): calling derive_listing_id twice with the same
    URL always yields the same ID.

    **Validates: Requirements 2.2**
    """
    first = derive_listing_id(source_url)
    second = derive_listing_id(source_url)
    assert first == second


@given(st.text())
def test_derive_listing_id_output_format(source_url: str) -> None:
    """Property 1 (part 2): output is always a 64-character lowercase hex
    string.

    **Validates: Requirements 2.2**
    """
    listing_id = derive_listing_id(source_url)
    assert isinstance(listing_id, str)
    assert len(listing_id) == 64
    assert HEX64_RE.match(listing_id) is not None


@given(st.text(), st.text())
def test_derive_listing_id_collision_resistance(url_a: str, url_b: str) -> None:
    """Property 1 (part 3): distinct source URLs yield distinct listing IDs
    (collision resistance, sampled).

    **Validates: Requirements 2.2**
    """
    if url_a == url_b:
        return
    assert derive_listing_id(url_a) != derive_listing_id(url_b)
