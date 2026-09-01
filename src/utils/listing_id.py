"""Listing ID derivation utility.

Provides a deterministic, collision-resistant identifier for job listings
based on their source URL. Used across the ingestion pipeline to key
`bronze.job_listings` records.
"""

import hashlib


def derive_listing_id(source_url: str) -> str:
    """Derive a deterministic, stable listing identifier from the source URL.

    Args:
        source_url: The canonical URL of the job listing.

    Returns:
        A 64-character lowercase hex string (SHA-256 digest) of the UTF-8
        encoded source URL.
    """
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()
