"""Input validation utilities for CV uploads, location text, and commute radius.

Used by the Databricks_App to validate user-submitted inputs before they are
passed further into the pipeline (Requirements 5.1, 5.5, 6.3, 6.4, 6.6).
"""

from typing import Tuple

ACCEPTED_CV_FORMATS = ("PDF", "DOCX")
MAX_CV_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

MIN_LOCATION_LENGTH = 1
MAX_LOCATION_LENGTH = 200

MIN_COMMUTE_RADIUS_KM = 1
MAX_COMMUTE_RADIUS_KM = 200


def validate_cv_file(file_format: str, file_size: int) -> Tuple[bool, str]:
    """Validate an uploaded CV file's format and size.

    Accepts PDF or DOCX formats (case-insensitive) with a size greater than
    0 bytes and at most 5 MB (Requirements 5.1, 5.5).

    Args:
        file_format: The file format/extension, e.g. "PDF" or "DOCX".
        file_size: The file size in bytes.

    Returns:
        A tuple of (is_valid, message). When invalid, the message states the
        accepted formats PDF and DOCX, the maximum size of 5 MB, and the
        requirement that the file be non-empty.
    """
    error_message = (
        "Invalid CV file: only PDF and DOCX formats are accepted, the file "
        "must be non-empty, and the maximum size is 5 MB."
    )

    if file_format is None or file_format.strip().upper() not in ACCEPTED_CV_FORMATS:
        return False, error_message

    if file_size <= 0:
        return False, error_message

    if file_size > MAX_CV_FILE_SIZE_BYTES:
        return False, error_message

    return True, "CV file is valid."


def validate_location_input(text: str) -> Tuple[bool, str]:
    """Validate a home location input string.

    Accepts non-empty strings of at most 200 characters (Requirement 6.6).

    Args:
        text: The submitted home location text.

    Returns:
        A tuple of (is_valid, message). When invalid, the message states
        that a home location of 1 to 200 characters is required.
    """
    error_message = "A home location of 1 to 200 characters is required."

    if text is None:
        return False, error_message

    length = len(text)
    if length < MIN_LOCATION_LENGTH or length > MAX_LOCATION_LENGTH:
        return False, error_message

    return True, "Location input is valid."


def validate_commute_radius(value: int) -> Tuple[bool, str]:
    """Validate a commute radius value expressed in kilometres.

    Accepts integer values from 1 to 200 kilometres inclusive
    (Requirements 6.3, 6.4).

    Args:
        value: The submitted commute radius in kilometres.

    Returns:
        A tuple of (is_valid, message). When invalid, the message states the
        accepted range of 1 to 200 kilometres.
    """
    error_message = "Commute radius must be between 1 and 200 kilometres."

    if value is None:
        return False, error_message

    if value < MIN_COMMUTE_RADIUS_KM or value > MAX_COMMUTE_RADIUS_KM:
        return False, error_message

    return True, "Commute radius is valid."
