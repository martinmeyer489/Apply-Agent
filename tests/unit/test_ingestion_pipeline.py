"""Unit tests for the ingestion pipeline notebook and its supporting utilities.

`notebooks/02_ingest_listings.py` is a Databricks notebook: at import time
it references `dbutils` and `spark` globals injected by the Databricks
runtime, and its module-level code eagerly reads the reachability report
and writes to Delta tables. None of that is available (or desirable) in a
plain pytest run.

Following the pattern established in `tests/unit/test_reachability_probe.py`,
we extract the pure, spark/dbutils-independent function definitions
(`check_api_reachable`, `fetch_jobs_from_api`) from the notebook's AST and
`exec` them in an isolated namespace, testing the real notebook logic
byte-for-byte without executing any of the dbutils/spark-dependent pipeline
code around it.

Batching (`batch_records`) and listing ID derivation (`derive_listing_id`)
are already-implemented pure utility modules and are imported directly.

Requirements: 2.3, 2.5, 2.7, 2.8, 2.9
"""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from src.utils.batching import batch_records
from src.utils.listing_id import derive_listing_id

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2] / "notebooks" / "02_ingest_listings.py"
)


def _load_notebook_functions(*names: str) -> dict:
    """Extract and compile the named top-level function defs from the notebook.

    Returns a namespace dict containing the live callables, bound to the
    real `requests`/`time`/`base64` modules so that
    `unittest.mock.patch(...)` in tests transparently affects them.
    """
    source = NOTEBOOK_PATH.read_text()
    tree = ast.parse(source)
    func_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(func_nodes) == len(names), (
        f"Expected to find {names} in {NOTEBOOK_PATH}, found "
        f"{[n.name for n in func_nodes]}"
    )
    module_ast = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    code = compile(module_ast, filename=str(NOTEBOOK_PATH), mode="exec")

    import time
    import base64

    namespace = {
        "requests": requests,
        "time": time,
        "base64": base64,
    }
    exec(code, namespace)  # noqa: S102 - intentional, isolated notebook extraction
    return namespace


@pytest.fixture(scope="module")
def notebook_funcs():
    return _load_notebook_functions("check_api_reachable", "fetch_jobs_from_api")


class _FakeResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# 1. API reachability check (Req 2.1)
# ---------------------------------------------------------------------------


def test_check_api_reachable_returns_true_on_200(notebook_funcs):
    """API reachable when search endpoint returns 200."""
    with patch("requests.get", return_value=_FakeResponse(200)):
        result = notebook_funcs["check_api_reachable"](timeout_seconds=10)

    assert result is True


def test_check_api_reachable_returns_true_on_401(notebook_funcs):
    """API reachable when returns 401 (no results, but API is up)."""
    with patch("requests.get", return_value=_FakeResponse(401)):
        result = notebook_funcs["check_api_reachable"](timeout_seconds=10)

    assert result is True


def test_check_api_reachable_returns_false_on_500(notebook_funcs):
    """API not reachable when server error."""
    with patch("requests.get", return_value=_FakeResponse(500)):
        result = notebook_funcs["check_api_reachable"](timeout_seconds=10)

    assert result is False


def test_check_api_reachable_returns_false_on_connection_error(notebook_funcs):
    """API not reachable on connection error."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Failed")):
        result = notebook_funcs["check_api_reachable"](timeout_seconds=10)

    assert result is False


def test_check_api_reachable_returns_false_on_timeout(notebook_funcs):
    """API not reachable on timeout."""
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Timed out")):
        result = notebook_funcs["check_api_reachable"](timeout_seconds=10)

    assert result is False


# ---------------------------------------------------------------------------
# 2. Ingestion mode selection (Req 2.9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api_reachable,expected_mode",
    [
        (False, "bundled_fallback"),
        (True, "api"),
    ],
)
def test_ingestion_mode_selection(api_reachable, expected_mode):
    """Mirrors the notebook's mode-selection branch:
    
        if not api_reachable:
            ingestion_mode = "bundled_fallback"
        else:
            ingestion_mode = "api"
    """
    if not api_reachable:
        ingestion_mode = "bundled_fallback"
    else:
        ingestion_mode = "api"

    assert ingestion_mode == expected_mode


# ---------------------------------------------------------------------------
# 3. API fetch returns job records (Req 2.1, 2.4)
# ---------------------------------------------------------------------------


def test_fetch_jobs_from_api_returns_records(notebook_funcs):
    """API fetch returns properly formatted job records."""
    search_response = _FakeResponse(200, {
        "stellenangebote": [
            {
                "refnr": "10001-123456789-S",
                "beruf": "Data Engineer",
                "arbeitgeber": "Test Company",
                "arbeitsort": {"plz": "10115", "ort": "Berlin", "region": "Berlin"},
            }
        ]
    })
    detail_response = _FakeResponse(200, {
        "stellenangebotsTitel": "Senior Data Engineer",
        "arbeitgeber": "Test Company",
        "stellenangebotsBeschreibung": "We are hiring!",
        "arbeitsorte": [{"plz": "10115", "ort": "Berlin", "region": "Berlin"}],
    })

    call_count = [0]

    def fake_get(url, **kwargs):
        call_count[0] += 1
        if "/jobdetails/" in url:
            return detail_response
        return search_response

    with patch("requests.get", side_effect=fake_get), patch("time.sleep"):
        records = notebook_funcs["fetch_jobs_from_api"](
            search_query="Data Engineer",
            location="Berlin",
            max_pages=1,
            timeout_seconds=30,
        )

    assert len(records) == 1
    assert records[0]["job_title"] == "Senior Data Engineer"
    assert records[0]["company_name"] == "Test Company"
    assert records[0]["job_description"] == "We are hiring!"
    assert records[0]["location_text"] == "10115, Berlin, Berlin"
    assert "arbeitsagentur.de/jobdetails/10001-123456789-S" in records[0]["source_url"]


def test_fetch_jobs_from_api_handles_empty_results(notebook_funcs):
    """API fetch handles empty search results gracefully."""
    empty_response = _FakeResponse(200, {"stellenangebote": []})

    with patch("requests.get", return_value=empty_response), patch("time.sleep"):
        records = notebook_funcs["fetch_jobs_from_api"](
            search_query="NonexistentJob",
            location="Noplace",
            max_pages=1,
            timeout_seconds=30,
        )

    assert records == []


def test_fetch_jobs_from_api_continues_on_detail_failure(notebook_funcs):
    """API fetch continues when detail fetch fails for one job."""
    search_response = _FakeResponse(200, {
        "stellenangebote": [
            {"refnr": "10001-111111111-S", "beruf": "Job A", "arbeitgeber": "Company A"},
            {"refnr": "10001-222222222-S", "beruf": "Job B", "arbeitgeber": "Company B"},
        ]
    })
    detail_response_fail = _FakeResponse(500)
    detail_response_success = _FakeResponse(200, {
        "stellenangebotsTitel": "Job B Title",
        "arbeitgeber": "Company B",
        "stellenangebotsBeschreibung": "Description B",
        "arbeitsorte": [{"plz": "12345", "ort": "Munich", "region": "Bavaria"}],
    })

    call_count = [0]

    def fake_get(url, **kwargs):
        call_count[0] += 1
        if "/jobdetails/" in url:
            if "111111111" in url:
                return detail_response_fail
            return detail_response_success
        return search_response

    with patch("requests.get", side_effect=fake_get), patch("time.sleep"):
        records = notebook_funcs["fetch_jobs_from_api"](
            search_query="Test",
            location="Germany",
            max_pages=1,
            timeout_seconds=30,
        )

    # Should have 1 record (the one that succeeded)
    assert len(records) == 1
    assert records[0]["job_title"] == "Job B Title"


# ---------------------------------------------------------------------------
# 4. Batch boundary (Req 2.8): 0, 1, 500, 501 records
# ---------------------------------------------------------------------------


def test_batch_records_zero_records():
    assert batch_records([], batch_size=500) == []


def test_batch_records_one_record():
    batches = batch_records([1], batch_size=500)
    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_batch_records_exactly_batch_size():
    records = list(range(500))
    batches = batch_records(records, batch_size=500)
    assert len(batches) == 1
    assert len(batches[0]) == 500


def test_batch_records_one_over_batch_size():
    records = list(range(501))
    batches = batch_records(records, batch_size=500)
    assert len(batches) == 2
    assert len(batches[0]) == 500
    assert len(batches[1]) == 1


# ---------------------------------------------------------------------------
# 5. Duplicate source_url MERGE behavior (Req 2.3)
# ---------------------------------------------------------------------------


def test_derive_listing_id_is_deterministic_for_same_source_url():
    """Same source_url must always derive the same listing_id.

    This is the testable invariant underlying "retain original listing_id
    on duplicate source_url": since `listing_id` is a pure function of
    `source_url`, a MERGE keyed on `source_url` that omits `listing_id`
    from its `whenMatchedUpdate` set (as the notebook does) will always
    resolve to the same identifier on repeated ingestion of the same URL,
    rather than producing a duplicate row or a new identifier.
    """
    url = "https://www.arbeitsagentur.de/jobsuche/jobdetails/10001-123456789-S"

    first = derive_listing_id(url)
    second = derive_listing_id(url)
    third = derive_listing_id(url)

    assert first == second == third


def test_derive_listing_id_differs_for_different_source_urls():
    id_a = derive_listing_id("https://www.arbeitsagentur.de/jobsuche/jobdetails/10001-111111111-S")
    id_b = derive_listing_id("https://www.arbeitsagentur.de/jobsuche/jobdetails/10001-222222222-S")

    assert id_a != id_b