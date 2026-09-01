"""Partial CV parse warning utilities.

Used to determine which CV_Parser output fields were not resolved so the
Databricks_App can display a warning naming them (Requirement 5.6).
"""

from typing import List

# The fields extracted by the CV_Parser (Requirement 5.3).
CV_FIELDS = (
    "skills",
    "years_of_experience",
    "education_history",
    "job_title_history",
    "qualifications_summary",
)


def get_unresolved_fields(parsed: dict) -> List[str]:
    """Return the names of CV fields that were not resolved.

    A field counts as unresolved if it is missing from `parsed`, is `None`,
    or is an empty list/string. The exception is `years_of_experience`,
    where `0` is a valid resolved value and only `None` (or a missing key)
    counts as unresolved.

    Args:
        parsed: The dict of parsed CV fields, as produced by the CV_Parser.

    Returns:
        A list of field names, in the order defined by `CV_FIELDS`, that
        were not resolved.
    """
    unresolved = []

    for field in CV_FIELDS:
        value = parsed.get(field)

        if field == "years_of_experience":
            if value is None:
                unresolved.append(field)
            continue

        if value is None:
            unresolved.append(field)
        elif isinstance(value, (list, str)) and len(value) == 0:
            unresolved.append(field)

    return unresolved
