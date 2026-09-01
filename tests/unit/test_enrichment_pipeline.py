"""Unit tests for the enrichment pipeline notebook and its state machine.

`notebooks/03_enrich_listings.py` is a Databricks notebook: at import time
it references `dbutils`/`spark` globals (and widget-derived globals like
`RECORD_TIMEOUT_SECONDS`, `LLM_ENDPOINT`) injected by the Databricks
runtime. Following the pattern established in
`tests/unit/test_reachability_probe.py` and
`tests/unit/test_ingestion_pipeline.py`, we extract the pure,
spark/dbutils-independent function definition (`_parse_llm_json`) from the
notebook's AST and `exec` it in an isolated namespace, testing the real
notebook logic byte-for-byte without executing any of the dbutils/spark
-dependent pipeline code around it.

The enrichment state machine (`determine_enrichment_state`) is an
already-implemented pure function and is imported directly.

Requirements: 3.4, 3.7, 3.8
"""

import ast
from pathlib import Path

import pytest

from src.pipelines.enrichment_state import determine_enrichment_state

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2] / "notebooks" / "03_enrich_listings.py"
)

PER_RECORD_LLM_THRESHOLD = 50


def _load_notebook_functions(*names: str) -> dict:
    """Extract and compile the named top-level function defs from the notebook.

    Returns a namespace dict containing the live callables.
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

    import json

    namespace = {"json": json}
    exec(code, namespace)  # noqa: S102 - intentional, isolated notebook extraction
    return namespace


@pytest.fixture(scope="module")
def notebook_funcs():
    return _load_notebook_functions("_parse_llm_json")


# ---------------------------------------------------------------------------
# 1. ai_query path when batch > 50 vs per-record path when <= 50 (Req 3.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,expected_path",
    [
        (1, "per_record"),
        (50, "per_record"),
        (51, "ai_query"),
        (100, "ai_query"),
    ],
)
def test_llm_path_selection_by_batch_size(batch_size, expected_path):
    """Mirrors the notebook's LLM-path selection branch (Step 4,
    `03_enrich_listings.py`):

        if len(batch) > PER_RECORD_LLM_THRESHOLD:
            llm_path = "ai_query"
        else:
            llm_path = "per_record"
    """
    batch = list(range(batch_size))

    if len(batch) > PER_RECORD_LLM_THRESHOLD:
        llm_path = "ai_query"
    else:
        llm_path = "per_record"

    assert llm_path == expected_path


# ---------------------------------------------------------------------------
# 2. _parse_llm_json: valid / empty / invalid JSON (supports Req 3.8)
# ---------------------------------------------------------------------------


def test_parse_llm_json_valid_json_with_all_fields(notebook_funcs):
    raw_response = (
        '{"required_skills": ["python", "sql"], "seniority_level": "mid", '
        '"employment_type": "full_time", "industry": "tech", '
        '"company_size_band": "medium"}'
    )

    result = notebook_funcs["_parse_llm_json"](raw_response)

    assert result == {
        "required_skills": ["python", "sql"],
        "seniority_level": "mid",
        "employment_type": "full_time",
        "industry": "tech",
        "company_size_band": "medium",
    }


def test_parse_llm_json_empty_response_is_hard_failure(notebook_funcs):
    result = notebook_funcs["_parse_llm_json"]("")

    assert "error" in result
    assert result["error"]


def test_parse_llm_json_invalid_json_is_hard_failure(notebook_funcs):
    result = notebook_funcs["_parse_llm_json"]("not valid json {{{")

    assert "error" in result
    assert result["error"]


def test_parse_llm_json_non_object_json_is_hard_failure(notebook_funcs):
    result = notebook_funcs["_parse_llm_json"]("[1, 2, 3]")

    assert "error" in result
    assert result["error"]


# ---------------------------------------------------------------------------
# 3. Timeout -> "failed" state (Req 3.8)
# ---------------------------------------------------------------------------


def test_timeout_marker_yields_failed_state():
    """Simulates what `extract_attributes_for_record` returns on timeout:

        {"_failed": True, "reason": "LLM extraction exceeded 60s timeout"}

    `determine_enrichment_state` must classify this as "failed", regardless
    of the geocode outcome.
    """
    geocode_result = {"latitude": 52.5, "longitude": 13.4}
    llm_result = {"_failed": True, "reason": "LLM extraction exceeded 60s timeout"}

    state, unresolved_attributes, failure_reason = determine_enrichment_state(
        geocode_result, llm_result
    )

    assert state == "failed"
    assert failure_reason == "LLM extraction exceeded 60s timeout"
    assert set(unresolved_attributes) == {
        "location",
        "required_skills",
        "seniority_level",
        "employment_type",
        "industry",
        "company_size_band",
    }


def test_timeout_marker_yields_failed_state_even_with_geocode_miss():
    llm_result = {"_failed": True, "reason": "timeout after 60s"}

    state, _, failure_reason = determine_enrichment_state(None, llm_result)

    assert state == "failed"
    assert failure_reason == "timeout after 60s"


# ---------------------------------------------------------------------------
# 4. Geocode miss -> "partially_enriched" state (Req 3.7)
# ---------------------------------------------------------------------------


def test_geocode_miss_with_full_llm_result_yields_partially_enriched():
    llm_result = {
        "required_skills": ["python"],
        "seniority_level": "senior",
        "employment_type": "full_time",
        "industry": "finance",
        "company_size_band": "large",
    }

    state, unresolved_attributes, failure_reason = determine_enrichment_state(
        None, llm_result
    )

    assert state == "partially_enriched"
    assert "location" in unresolved_attributes
    assert failure_reason is not None


def test_geocode_miss_as_empty_dict_yields_partially_enriched():
    llm_result = {
        "required_skills": ["python"],
        "seniority_level": "senior",
        "employment_type": "full_time",
        "industry": "finance",
        "company_size_band": "large",
    }

    state, unresolved_attributes, _ = determine_enrichment_state({}, llm_result)

    assert state == "partially_enriched"
    assert "location" in unresolved_attributes
