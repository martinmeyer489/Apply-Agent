"""Unit tests for the reachability probe notebook.

`notebooks/01_reachability_probe.py` is a Databricks notebook: at import
time it references `dbutils` and `spark` globals injected by the Databricks
runtime, and its module-level code eagerly probes every candidate domain
and writes to a Delta table. None of that is available (or desirable) in a
plain pytest run.

Rather than stubbing `dbutils`/`spark` to make the whole notebook
importable, we extract just the `probe_domain` function's AST from the
notebook source and `exec` it in an isolated namespace that provides the
handful of names it actually needs (`requests`, `datetime`, `timezone`, and
a `TIMEOUT_SECONDS` default). This tests the real probe logic byte-for-byte
as it exists in the notebook, without executing any of the
dbutils/spark-dependent pipeline code around it.

Requirements: 1.2, 1.3, 1.4
"""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2] / "notebooks" / "01_reachability_probe.py"
)


def _load_probe_domain():
    """Extract and compile just the `probe_domain` function from the notebook.

    Returns the live `probe_domain` callable, bound to the real `requests`
    module so that `unittest.mock.patch("requests.head", ...)` in tests
    transparently affects it.
    """
    source = NOTEBOOK_PATH.read_text()
    tree = ast.parse(source)
    func_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "probe_domain"
    )
    module_ast = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    code = compile(module_ast, filename=str(NOTEBOOK_PATH), mode="exec")

    from datetime import datetime, timezone

    namespace = {
        "requests": requests,
        "datetime": datetime,
        "timezone": timezone,
        "TIMEOUT_SECONDS": 30,
    }
    exec(code, namespace)  # noqa: S102 - intentional, isolated notebook extraction
    return namespace["probe_domain"]


@pytest.fixture(scope="module")
def probe_domain():
    return _load_probe_domain()


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_probe_domain_timeout_is_blocked(probe_domain):
    with patch("requests.head", side_effect=requests.exceptions.Timeout("Read timed out")):
        result = probe_domain("indeed.com")

    assert result["domain"] == "indeed.com"
    assert result["outcome"] == "blocked"
    assert result["http_status_code"] is None
    assert result["error_message"] is not None
    assert "timed out" in result["error_message"].lower()


def test_probe_domain_http_200_is_reachable(probe_domain):
    with patch("requests.head", return_value=_FakeResponse(200)):
        result = probe_domain("linkedin.com")

    assert result["domain"] == "linkedin.com"
    assert result["outcome"] == "reachable"
    assert result["http_status_code"] == 200
    assert result["error_message"] is None


def test_probe_domain_dns_error_is_blocked(probe_domain):
    with patch(
        "requests.head",
        side_effect=requests.exceptions.ConnectionError("Failed to resolve 'glassdoor.com'"),
    ):
        result = probe_domain("glassdoor.com")

    assert result["domain"] == "glassdoor.com"
    assert result["outcome"] == "blocked"
    assert result["http_status_code"] is None
    assert result["error_message"] is not None


def test_all_domains_blocked_summary(probe_domain):
    domains = ["indeed.com", "linkedin.com", "glassdoor.com", "monster.com", "reed.co.uk"]
    with patch(
        "requests.head",
        side_effect=requests.exceptions.ConnectionError("blocked by allowlist"),
    ):
        results = [probe_domain(domain) for domain in domains]

    reachable_count = sum(1 for r in results if r["outcome"] == "reachable")
    blocked_count = sum(1 for r in results if r["outcome"] == "blocked")

    assert reachable_count == 0
    assert blocked_count == len(domains)
    assert reachable_count + blocked_count == len(domains)
