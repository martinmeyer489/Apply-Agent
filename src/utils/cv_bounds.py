"""CV field bound enforcement utilities.

Used by the CV_Parser to truncate/clamp parsed CV fields to the persisted
bounds required by the system (Requirement 5.4).
"""

from typing import Any, Dict

MAX_SKILLS = 50
MIN_YEARS_OF_EXPERIENCE = 0
MAX_YEARS_OF_EXPERIENCE = 99
MAX_EDUCATION_HISTORY = 20
MAX_JOB_TITLE_HISTORY = 30
MAX_SUMMARY_CHARS = 2000


def enforce_cv_bounds(parsed_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce persisted field bounds on parsed CV fields.

    Truncates/clamps the following fields (Requirement 5.4):
      - skills: at most 50 entries
      - years_of_experience: clamped to [0, 99]
      - education_history: at most 20 entries
      - job_title_history: at most 30 entries
      - qualifications_summary: at most 2000 characters

    Fields that are missing from `parsed_fields` are left absent from the
    result (handled gracefully, no KeyError raised). Fields present but with
    an unexpected type (e.g. a string where a list is expected) are passed
    through unchanged, since bound enforcement only applies to the expected
    type for that field.

    Args:
        parsed_fields: The raw parsed CV fields, potentially missing keys or
            containing values that exceed the persisted bounds.

    Returns:
        A new dict containing only the bound-enforced values for the keys
        present in `parsed_fields`. The input dict is not mutated.
    """
    result: Dict[str, Any] = dict(parsed_fields) if parsed_fields else {}

    if "skills" in result:
        skills = result["skills"]
        if isinstance(skills, list):
            result["skills"] = skills[:MAX_SKILLS]

    if "years_of_experience" in result:
        years = result["years_of_experience"]
        if isinstance(years, (int, float)) and not isinstance(years, bool):
            clamped = max(MIN_YEARS_OF_EXPERIENCE, min(MAX_YEARS_OF_EXPERIENCE, years))
            result["years_of_experience"] = int(clamped)

    if "education_history" in result:
        education = result["education_history"]
        if isinstance(education, list):
            result["education_history"] = education[:MAX_EDUCATION_HISTORY]

    if "job_title_history" in result:
        job_titles = result["job_title_history"]
        if isinstance(job_titles, list):
            result["job_title_history"] = job_titles[:MAX_JOB_TITLE_HISTORY]

    if "qualifications_summary" in result:
        summary = result["qualifications_summary"]
        if isinstance(summary, str):
            result["qualifications_summary"] = summary[:MAX_SUMMARY_CHARS]

    return result
