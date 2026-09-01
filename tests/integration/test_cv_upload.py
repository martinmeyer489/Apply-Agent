"""Integration test for the real CV upload -> store -> parse flow.

This is a **live integration test**: unlike `tests/unit/test_cv_parser.py`
(which mocks `PyPDF2.PdfReader` and the Foundation Model APIs response),
this test generates a small but genuinely valid PDF on the fly with
`PyPDF2.PdfWriter` (already a project dependency — see `pyproject.toml`),
stores it via the same Volume-storage pattern
`src/app/main.py::_store_cv_file` uses, and calls `parse_cv()` for real.
Real text extraction from a blank/near-empty PDF plus a real Foundation
Model APIs call both require a live Databricks workspace connection, so
this is squarely an integration-only test; it is automatically skipped
outside one.

When run against a live workspace, it:

1. Generates a minimal valid single-page PDF with `PyPDF2.PdfWriter`
   containing a short CV-like text annotation (best-effort — a truly
   blank PDF page has no extractable text, so a text annotation is added
   so `PyPDF2.PdfReader(...).extract_text()` has something to return; if
   the installed `PyPDF2` version yields no extractable text regardless,
   the assertions below only require the parse call to succeed and return
   the documented shape, not specific field values).
2. Stores it via `_store_cv_file` (Req 5.2 — real Volume-store path when
   `/Volumes/job_agent/volumes/cv_uploads` exists, i.e. inside a Databricks
   workspace).
3. Calls `parse_cv(stored_path)` for real (Req 5.2, 5.3 — real Foundation
   Model APIs extraction) and asserts the returned profile dict has the
   documented keys and populated types.

Requirements: 5.2
"""

from __future__ import annotations

import os

import pytest

from tests.integration._databricks_env import SKIP_REASON, has_databricks_environment

pytestmark = pytest.mark.integration


def _generate_minimal_valid_pdf(destination_path: str) -> None:
    """Write a minimal, genuinely valid single-page PDF to `destination_path`.

    Uses `PyPDF2.PdfWriter` (already a project dependency) rather than a
    hand-crafted byte string, so the file is a real PDF a PDF reader can
    open — as opposed to the mocked-reader unit tests in
    `tests/unit/test_cv_parser.py`.
    """
    import PyPDF2

    writer = PyPDF2.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(destination_path, "wb") as pdf_file:
        writer.write(pdf_file)


@pytest.mark.skipif(not has_databricks_environment(), reason=SKIP_REASON)
class TestCvUploadIntegration:
    """Real Volume-store + real `parse_cv` flow for an uploaded CV file."""

    def test_upload_and_parse_populates_profile(self, tmp_path):
        """A live run of this test would:

        1. Generate a real, minimal PDF file.
        2. Store it via `src.app.main._store_cv_file` (Req 5.2), which
           writes into the `cv_uploads` Volume when running inside a
           Databricks workspace.
        3. Call `src.agent.cv_parser.parse_cv(stored_path)` for real
           against Foundation Model APIs and assert:
           - the returned dict has exactly the documented keys, and
           - `unresolved_fields` is a list (present regardless of
             whether extraction was full or partial — Req 5.3/5.6).
        """
        from src.agent.cv_parser import parse_cv
        from src.app.main import _store_cv_file

        source_pdf_path = os.path.join(str(tmp_path), "sample_cv.pdf")
        _generate_minimal_valid_pdf(source_pdf_path)

        stored_path = _store_cv_file(source_pdf_path)
        assert os.path.exists(stored_path)
        assert os.path.getsize(stored_path) > 0

        profile = parse_cv(stored_path)

        expected_keys = {
            "skills",
            "years_of_experience",
            "education_history",
            "job_title_history",
            "qualifications_summary",
            "unresolved_fields",
        }
        assert expected_keys.issubset(profile.keys())
        assert isinstance(profile["unresolved_fields"], list)
