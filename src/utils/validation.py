"""Record validation utility for ingestion.

Checks that a collected job listing record contains the required fields
before it is written to the Bronze tier. Records missing required fields
are skipped by the ingestion pipeline and logged with the names of the
missing fields.
"""

REQUIRED_FIELDS = ("job_title", "company_name", "source_url")


def validate_listing_record(record: dict) -> tuple[bool, list[str]]:
    """Validate that a listing record has all required fields populated.

    A field is considered missing if it is absent from the record, is
    ``None``, or is an empty/whitespace-only string.

    Args:
        record: The candidate job listing record, typically a dict with
            keys such as ``job_title``, ``company_name``, ``job_description``,
            ``location_text``, and ``source_url``.

    Returns:
        A tuple ``(is_valid, missing_fields)`` where ``is_valid`` is
        ``True`` only if all required fields are present and non-empty,
        and ``missing_fields`` is the list of required field names (in
        the order defined by ``REQUIRED_FIELDS``) that are missing or
        empty.
    """
    missing_fields: list[str] = []

    for field in REQUIRED_FIELDS:
        value = record.get(field) if record else None
        if value is None:
            missing_fields.append(field)
        elif isinstance(value, str) and value.strip() == "":
            missing_fields.append(field)

    return (len(missing_fields) == 0, missing_fields)
