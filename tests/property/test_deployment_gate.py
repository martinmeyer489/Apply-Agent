"""Property-based test for deployment gate enforcement.

Property 24: Deployment Gate Enforcement
Validates: Requirements 9.6
"""

from hypothesis import given, strategies as st

from src.utils.deployment_gate import check_deployment_gate, EVALUATION_RESULTS_TABLE


class _FakeRow(dict):
    """Dict subclass supporting attribute-style and __getitem__ access,
    mimicking a PySpark Row for the columns we care about."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _FakeFilterExpr:
    """A predicate produced by comparing a `_FakeColumn` to a value."""

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def matches(self, row):
        return row.get(self.name) == self.value


class _FakeColumn:
    def __init__(self, name):
        self._name = name

    def __eq__(self, other):
        return _FakeFilterExpr(self._name, other)


class _FakeDataFrame:
    """Minimal fake DataFrame supporting the chain used by
    `check_deployment_gate`: `.filter(...).orderBy(...).collect()`, plus
    attribute access for building filter predicates
    (`results_df.model_version`)."""

    def __init__(self, rows):
        self._rows = list(rows)

    def __getattr__(self, item):
        return _FakeColumn(item)

    def filter(self, predicate):
        return _FakeDataFrame([r for r in self._rows if predicate.matches(r)])

    def orderBy(self, column_name, ascending=True):
        rows = sorted(
            self._rows,
            key=lambda r: r.get(column_name),
            reverse=not ascending,
        )
        return _FakeDataFrame(rows)

    def collect(self):
        return [_FakeRow(r) for r in self._rows]


class FakeGateSpark:
    """Fake `spark` implementing just enough of the SparkSession API for
    `check_deployment_gate` to query an in-memory list of evaluation
    result rows."""

    def __init__(self, rows):
        self._rows = list(rows)

    def table(self, name):
        assert name == EVALUATION_RESULTS_TABLE
        return _FakeDataFrame(self._rows)


# A strategy for a single evaluation_results row.
_eval_row_strategy = st.fixed_dictionaries(
    {
        "model_version": st.integers(min_value=1, max_value=20),
        "match_relevance_mean": st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        "groundedness_mean": st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        "eval_timestamp": st.integers(min_value=0, max_value=1_000_000),
    }
)


@given(
    rows=st.lists(_eval_row_strategy, min_size=0, max_size=10),
    queried_version=st.integers(min_value=1, max_value=20),
)
def test_deployment_gate_enforcement(rows, queried_version):
    """**Validates: Requirements 9.6**

    For an arbitrary set of `evaluation_results` rows and an arbitrary
    queried model version:
    - `blocked=True` if and only if no row has `model_version ==
      queried_version`.
    - `blocked=False` if and only if at least one row does.
    - The reason string is always non-empty.
    - When blocked, the reason mentions "No evaluation results".
    - When allowed, the reason mentions the version number and the
      scorer means.
    """
    spark = FakeGateSpark(rows)

    blocked, reason = check_deployment_gate(queried_version, spark=spark)

    matching_rows = [r for r in rows if r["model_version"] == queried_version]
    has_results = len(matching_rows) > 0

    assert blocked == (not has_results)
    assert reason
    assert isinstance(reason, str)

    if blocked:
        assert "No evaluation results" in reason
    else:
        assert str(queried_version) in reason
        assert "match_relevance_mean" in reason
        assert "groundedness_mean" in reason
