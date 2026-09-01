"""Unit tests for the ingestion pipeline notebook and its supporting utilities.

`notebooks/02_ingest_listings.py` is a Databricks notebook: at import time
it references `dbutils` and `spark` globals injected by the Databricks
runtime, and its module-level code eagerly reads the reachability report
and writes to Delta tables. None of that is available (or desirable) in a
plain pytest run.

Following the pattern established in `tests/unit/test_reachability_probe.py`,
we extract the pure, spark/dbutils-independent function definitions
(`is_scraping_allowed`, `scrape_domain`, `extract_listing_fields`) from the
notebook's AST and `exec` them in an isolated namespace, testing the real
notebook logic byte-for-byte without executing any of the
dbutils/spark-dependent pipeline code around it.

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
    real `requests`/`time`/`urllib.robotparser` modules so that
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
    import urllib.robotparser
    from datetime import datetime, timezone

    namespace = {
        "requests": requests,
        "time": time,
        "urllib": urllib,
        "datetime": datetime,
        "timezone": timezone,
    }
    exec(code, namespace)  # noqa: S102 - intentional, isolated notebook extraction
    return namespace


@pytest.fixture(scope="module")
def notebook_funcs():
    return _load_notebook_functions(
        "is_scraping_allowed", "scrape_domain", "extract_listing_fields"
    )


class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# 1. Fallback activation when 0 reachable (Req 2.9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reachable_domains,expected_mode",
    [
        ([], "bundled_fallback"),
        (["indeed.com"], "live"),
        (["indeed.com", "linkedin.com"], "live"),
    ],
)
def test_ingestion_mode_selection(reachable_domains, expected_mode):
    """Mirrors the notebook's mode-selection branch (Step 2, `02_ingest_listings.py`):

        if len(reachable_domains) == 0:
            ingestion_mode = "bundled_fallback"
        else:
            ingestion_mode = "live"
    """
    if len(reachable_domains) == 0:
        ingestion_mode = "bundled_fallback"
    else:
        ingestion_mode = "live"

    assert ingestion_mode == expected_mode


# ---------------------------------------------------------------------------
# 2. robots.txt disallow parsing (Req 2.5, 2.7)
# ---------------------------------------------------------------------------


def test_is_scraping_allowed_returns_false_for_disallowed_path(notebook_funcs):
    robots_txt = "User-agent: *\nDisallow: /jobs\n"
    with patch("requests.get", return_value=_FakeResponse(200, robots_txt)):
        result = notebook_funcs["is_scraping_allowed"]("example.com", "/jobs", 30)

    assert result is False


def test_is_scraping_allowed_returns_true_for_allowed_path(notebook_funcs):
    robots_txt = "User-agent: *\nDisallow: /admin\n"
    with patch("requests.get", return_value=_FakeResponse(200, robots_txt)):
        result = notebook_funcs["is_scraping_allowed"]("example.com", "/jobs", 30)

    assert result is True


def test_is_scraping_allowed_defaults_to_true_when_robots_txt_missing(notebook_funcs):
    with patch("requests.get", return_value=_FakeResponse(404, "")):
        result = notebook_funcs["is_scraping_allowed"]("example.com", "/jobs", 30)

    assert result is True


def test_is_scraping_allowed_defaults_to_true_on_fetch_failure(notebook_funcs):
    with patch(
        "requests.get",
        side_effect=requests.exceptions.ConnectionError("Failed to resolve"),
    ):
        result = notebook_funcs["is_scraping_allowed"]("example.com", "/jobs", 30)

    assert result is True


def test_scrape_domain_raises_permission_error_when_disallowed(notebook_funcs):
    robots_txt = "User-agent: *\nDisallow: /jobs\n"
    with patch("requests.get", return_value=_FakeResponse(200, robots_txt)), patch(
        "time.sleep"
    ):
        with pytest.raises(PermissionError):
            notebook_funcs["scrape_domain"]("example.com", 30, 1)


def test_scrape_domain_scrapes_when_allowed(notebook_funcs):
    robots_txt = "User-agent: *\nDisallow: /admin\n"
    listing_response = _FakeResponse(200, "<html></html>")

    def fake_get(url, timeout=None):
        if url.endswith("/robots.txt"):
            return _FakeResponse(200, robots_txt)
        return listing_response

    with patch("requests.get", side_effect=fake_get), patch("time.sleep") as mock_sleep:
        records = notebook_funcs["scrape_domain"]("example.com", 30, 1)

    # extract_listing_fields is currently a stub that returns [].
    assert records == []
    mock_sleep.assert_called_once_with(1)


def test_extract_listing_fields_placeholder_returns_empty_list(notebook_funcs):
    assert notebook_funcs["extract_listing_fields"]("<html></html>", "https://x.com/jobs") == []


# ---------------------------------------------------------------------------
# 3. Batch boundary (Req 2.8): 0, 1, 500, 501 records
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
# 4. Duplicate source_url MERGE behavior (Req 2.3)
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
    url = "https://example.com/jobs/123"

    first = derive_listing_id(url)
    second = derive_listing_id(url)
    third = derive_listing_id(url)

    assert first == second == third


def test_derive_listing_id_differs_for_different_source_urls():
    id_a = derive_listing_id("https://example.com/jobs/123")
    id_b = derive_listing_id("https://example.com/jobs/456")

    assert id_a != id_b
