"""Property-based test for batch checkpoint resume correctness.

Property 25: Batch Checkpoint Resume Correctness
Validates: Requirements 11.9
"""

from datetime import datetime, timezone

from hypothesis import given, strategies as st

from src.utils.checkpoints import get_resume_batch, write_checkpoint

CHECKPOINT_TABLE = "job_agent.ops.batch_checkpoints"


class _FakeRow(dict):
    """Dict subclass supporting attribute-style and __getitem__ access,
    mimicking a PySpark Row for the columns we care about."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _FakeFilterExpr:
    """A predicate produced by comparing a `_FakeColumn` to a value.
    Supports `&` composition, matching the `(a == b) & (c == d)` pattern."""

    def __init__(self, *clauses):
        # clauses is a flat list of (name, value) tuples
        self.clauses = []
        for clause in clauses:
            if isinstance(clause, tuple):
                self.clauses.append(clause)
            else:
                raise TypeError(f"Unexpected clause: {clause!r}")

    def __and__(self, other):
        return _FakeFilterExpr(*(self.clauses + other.clauses))

    def matches(self, row):
        return all(row.get(name) == value for name, value in self.clauses)


# _FakeColumn.__eq__ needs to produce a (name, value) tuple wrapped by
# _FakeFilterExpr; adjust construction accordingly.
def _column_eq(name, value):
    return _FakeFilterExpr((name, value))


class _FakeColumnFixed:
    def __init__(self, name):
        self._name = name

    def __eq__(self, other):
        return _column_eq(self._name, other)


class _FakeDataFrame:
    """Minimal fake DataFrame supporting the chain used by
    `read_checkpoint`: `.filter(...).orderBy(...).limit(...).collect()`,
    plus attribute access for building filter predicates (`df.pipeline_name`).
    """

    def __init__(self, rows):
        self._rows = list(rows)

    def __getattr__(self, item):
        # Support `checkpoints_df.pipeline_name == pipeline_name` style access.
        return _FakeColumnFixed(item)

    def filter(self, predicate):
        return _FakeDataFrame([r for r in self._rows if predicate.matches(r)])

    def orderBy(self, column_name, ascending=True):
        rows = sorted(
            self._rows,
            key=lambda r: r.get(column_name),
            reverse=not ascending,
        )
        return _FakeDataFrame(rows)

    def limit(self, n):
        return _FakeDataFrame(self._rows[:n])

    def collect(self):
        return [_FakeRow(r) for r in self._rows]


class _FakeWriter:
    def __init__(self, store, rows):
        self._store = store
        self._rows = rows

    def format(self, _fmt):
        return self

    def mode(self, _mode):
        return self

    def saveAsTable(self, table_name):
        self._store.tables.setdefault(table_name, []).extend(self._rows)


class _FakeCreatedDataFrame:
    def __init__(self, store, rows, columns):
        self._store = store
        self._rows = [dict(zip(columns, row)) for row in rows]

    @property
    def write(self):
        return _FakeWriter(self._store, self._rows)


class FakeCheckpointStore:
    """Fake `spark` implementing just enough of the SparkSession API for
    `read_checkpoint` and `write_checkpoint` to operate against an
    in-memory list of checkpoint rows.

    Rows are ordered by insertion order, which serves as a proxy for
    `checkpoint_timestamp` ordering since writes happen strictly in
    sequence within a single test run (with monotonically increasing
    synthetic timestamps to make ordering unambiguous).
    """

    def __init__(self):
        self.tables = {}
        self._clock = 0

    def table(self, name):
        return _FakeDataFrame(self.tables.get(name, []))

    def createDataFrame(self, rows, columns):
        # Replace the caller-supplied timestamp with a monotonically
        # increasing synthetic one so ordering by checkpoint_timestamp is
        # deterministic regardless of real-clock resolution.
        self._clock += 1
        fixed_rows = []
        for row in rows:
            row = list(row)
            if "checkpoint_timestamp" in columns:
                ts_index = columns.index("checkpoint_timestamp")
                row[ts_index] = self._clock
            fixed_rows.append(tuple(row))
        return _FakeCreatedDataFrame(self, fixed_rows, columns)


@given(batch_indices=st.lists(st.integers(min_value=0, max_value=10_000), min_size=0, max_size=20))
def test_checkpoint_resume_correctness(batch_indices):
    """**Validates: Requirements 11.9**

    For an arbitrary sequence of batch indices written in order via
    `write_checkpoint` for a fixed (pipeline_name, run_id):
    - before any writes, `get_resume_batch` returns 0
    - after each write, `get_resume_batch` equals the most recently
      written `batch_index + 1`
    """
    spark = FakeCheckpointStore()
    pipeline_name = "enrichment"
    run_id = "run-abc-123"

    # No checkpoint written yet -> resume from 0.
    assert get_resume_batch(spark, pipeline_name, run_id) == 0

    last_written = None
    for batch_index in batch_indices:
        write_checkpoint(spark, pipeline_name, run_id, batch_index)
        last_written = batch_index
        assert get_resume_batch(spark, pipeline_name, run_id) == last_written + 1


@given(
    run_a=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
    run_b=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
    batch_a=st.integers(min_value=0, max_value=1000),
    batch_b=st.integers(min_value=0, max_value=1000),
)
def test_checkpoint_resume_isolated_per_run(run_a, run_b, batch_a, batch_b):
    """**Validates: Requirements 11.9**

    Checkpoints for different `run_id`s (or `pipeline_name`s) do not
    interfere with one another: writing a checkpoint for one run does not
    change the resume batch for a distinct run.
    """
    if run_a == run_b:
        return  # only interested in distinct runs for this property

    spark = FakeCheckpointStore()
    pipeline_name = "ingestion"

    write_checkpoint(spark, pipeline_name, run_a, batch_a)

    # run_b has no checkpoint yet, regardless of what happened to run_a.
    assert get_resume_batch(spark, pipeline_name, run_b) == 0

    write_checkpoint(spark, pipeline_name, run_b, batch_b)

    assert get_resume_batch(spark, pipeline_name, run_a) == batch_a + 1
    assert get_resume_batch(spark, pipeline_name, run_b) == batch_b + 1
