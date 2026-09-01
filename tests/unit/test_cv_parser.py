"""Unit tests for the CV parser module.

Covers PDF/DOCX extraction, the 5 MB size boundary, empty-file and
non-PDF/DOCX rejection, and partial field extraction warnings.

Requirements: 5.1, 5.3, 5.5, 5.6
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.cv_parser import parse_cv


def _write_temp_file(suffix: str, content: bytes) -> str:
    """Create a temp file with the given suffix/content and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with open(fd, "wb") as f:
        f.write(content)
    return path


def _fake_llm_response(payload: dict):
    """Build a fake `serving_endpoints.query` response object."""
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


FULL_LLM_PAYLOAD = {
    "skills": ["Python", "SQL"],
    "years_of_experience": 5,
    "education_history": [
        {"institution": "MIT", "degree": "BSc", "field": "CS", "year": 2015}
    ],
    "job_title_history": [
        {"title": "Engineer", "company": "Acme", "start_year": 2015, "end_year": 2020}
    ],
    "qualifications_summary": "Experienced software engineer.",
}


class TestPdfExtraction:
    def test_sample_pdf_extraction(self):
        path = _write_temp_file(".pdf", b"%PDF-1.4 dummy content")
        try:
            fake_page = MagicMock()
            fake_page.extract_text.return_value = "John Doe CV text"
            fake_reader = MagicMock()
            fake_reader.pages = [fake_page]

            with patch("PyPDF2.PdfReader", return_value=fake_reader), patch(
                "databricks.sdk.WorkspaceClient"
            ) as mock_client_cls:
                mock_client = MagicMock()
                mock_client.serving_endpoints.query.return_value = _fake_llm_response(
                    FULL_LLM_PAYLOAD
                )
                mock_client_cls.return_value = mock_client

                result = parse_cv(path)

            assert result["skills"] == ["Python", "SQL"]
            assert result["years_of_experience"] == 5
            assert result["education_history"] == FULL_LLM_PAYLOAD["education_history"]
            assert result["job_title_history"] == FULL_LLM_PAYLOAD["job_title_history"]
            assert result["qualifications_summary"] == "Experienced software engineer."
            assert result["unresolved_fields"] == []
        finally:
            Path(path).unlink(missing_ok=True)


class TestDocxExtraction:
    def test_sample_docx_extraction(self):
        path = _write_temp_file(".docx", b"PK dummy docx bytes")
        try:
            fake_paragraph = MagicMock()
            fake_paragraph.text = "Jane Doe CV text"
            fake_document = MagicMock()
            fake_document.paragraphs = [fake_paragraph]

            with patch("docx.Document", return_value=fake_document), patch(
                "databricks.sdk.WorkspaceClient"
            ) as mock_client_cls:
                mock_client = MagicMock()
                mock_client.serving_endpoints.query.return_value = _fake_llm_response(
                    FULL_LLM_PAYLOAD
                )
                mock_client_cls.return_value = mock_client

                result = parse_cv(path)

            assert result["skills"] == ["Python", "SQL"]
            assert result["years_of_experience"] == 5
            assert result["unresolved_fields"] == []
        finally:
            Path(path).unlink(missing_ok=True)


class TestFiveMegabyteBoundary:
    def test_file_exactly_at_5mb_is_accepted(self):
        five_mb = 5 * 1024 * 1024
        path = _write_temp_file(".pdf", b"%PDF-1.4" + b"\x00" * (five_mb - 8))
        try:
            assert Path(path).stat().st_size == five_mb

            fake_page = MagicMock()
            fake_page.extract_text.return_value = "text"
            fake_reader = MagicMock()
            fake_reader.pages = [fake_page]

            with patch("PyPDF2.PdfReader", return_value=fake_reader), patch(
                "databricks.sdk.WorkspaceClient"
            ) as mock_client_cls:
                mock_client = MagicMock()
                mock_client.serving_endpoints.query.return_value = _fake_llm_response(
                    FULL_LLM_PAYLOAD
                )
                mock_client_cls.return_value = mock_client

                # Must not raise ValueError for size.
                result = parse_cv(path)

            assert result["unresolved_fields"] == []
        finally:
            Path(path).unlink(missing_ok=True)


class TestEmptyFileRejection:
    def test_empty_file_raises_value_error(self):
        path = _write_temp_file(".pdf", b"")
        try:
            with pytest.raises(ValueError):
                parse_cv(path)
        finally:
            Path(path).unlink(missing_ok=True)


class TestNonPdfDocxRejection:
    def test_txt_file_raises_value_error(self):
        path = _write_temp_file(".txt", b"just some plain text")
        try:
            with pytest.raises(ValueError):
                parse_cv(path)
        finally:
            Path(path).unlink(missing_ok=True)


class TestPartialFieldExtractionWarning:
    def test_missing_fields_reported_as_unresolved(self):
        path = _write_temp_file(".pdf", b"%PDF-1.4 dummy content")
        try:
            partial_payload = {
                "skills": ["Python"],
                "years_of_experience": 3,
            }

            fake_page = MagicMock()
            fake_page.extract_text.return_value = "text"
            fake_reader = MagicMock()
            fake_reader.pages = [fake_page]

            with patch("PyPDF2.PdfReader", return_value=fake_reader), patch(
                "databricks.sdk.WorkspaceClient"
            ) as mock_client_cls:
                mock_client = MagicMock()
                mock_client.serving_endpoints.query.return_value = _fake_llm_response(
                    partial_payload
                )
                mock_client_cls.return_value = mock_client

                result = parse_cv(path)

            assert result["skills"] == ["Python"]
            assert result["years_of_experience"] == 3
            assert sorted(result["unresolved_fields"]) == sorted(
                [
                    "education_history",
                    "job_title_history",
                    "qualifications_summary",
                ]
            )
        finally:
            Path(path).unlink(missing_ok=True)
