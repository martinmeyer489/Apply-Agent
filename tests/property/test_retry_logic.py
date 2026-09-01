"""Property-based tests for the retry-with-backoff decorator.

Property 26: Foundation Model API Retry Logic
Validates: Requirements 11.10
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.retry import retry_with_backoff  # noqa: E402


class RetryableError(Exception):
    """Simulated API error carrying a retryable status code."""

    def __init__(self, status_code: int, message: str = "error"):
        super().__init__(message)
        self.status_code = status_code


class NonRetryableError(Exception):
    """Simulated API error carrying a non-retryable status code."""

    def __init__(self, status_code: int, message: str = "error"):
        super().__init__(message)
        self.status_code = status_code


def _make_sleep_recorder():
    """Returns a sleep_fn stub and the list it records delays into."""
    delays: list[float] = []

    def sleep_fn(seconds: float) -> None:
        delays.append(seconds)

    return sleep_fn, delays


# --- Property 1: always-retryable-error function is retried exactly
# max_attempts times then raises. -------------------------------------------


@given(
    max_attempts=st.integers(min_value=0, max_value=5),
    backoff_base=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=50)
def test_always_retryable_error_retried_exactly_max_attempts_then_raises(
    max_attempts, backoff_base
):
    sleep_fn, delays = _make_sleep_recorder()
    call_count = {"n": 0}

    @retry_with_backoff(
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        retryable_codes=(429,),
        sleep_fn=sleep_fn,
    )
    def always_fails():
        call_count["n"] += 1
        raise RetryableError(429)

    try:
        always_fails()
        raised = False
    except RetryableError:
        raised = True

    assert raised, "expected the retryable error to eventually propagate"
    # Initial call + max_attempts retries.
    assert call_count["n"] == max_attempts + 1
    # One sleep per retry attempt (not after the final failed attempt's
    # non-retry, since attempts are exhausted at that point).
    assert len(delays) == max_attempts


# --- Property 2: non-retryable error is never retried. ----------------------


@given(
    max_attempts=st.integers(min_value=0, max_value=5),
    status_code=st.integers(min_value=400, max_value=599).filter(
        lambda c: c != 429
    ),
)
@settings(max_examples=50)
def test_non_retryable_error_never_retried(max_attempts, status_code):
    sleep_fn, delays = _make_sleep_recorder()
    call_count = {"n": 0}

    @retry_with_backoff(
        max_attempts=max_attempts,
        backoff_base=2,
        retryable_codes=(429,),
        sleep_fn=sleep_fn,
    )
    def fails_non_retryable():
        call_count["n"] += 1
        raise NonRetryableError(status_code)

    try:
        fails_non_retryable()
        raised = False
    except NonRetryableError:
        raised = True

    assert raised
    # Only the initial call is made, no retries.
    assert call_count["n"] == 1
    assert delays == []


# --- Property 3: delays follow backoff_base ** (attempt) exponential growth.


@given(
    max_attempts=st.integers(min_value=1, max_value=5),
    backoff_base=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=50)
def test_delays_follow_exponential_backoff(max_attempts, backoff_base):
    sleep_fn, delays = _make_sleep_recorder()

    @retry_with_backoff(
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        retryable_codes=(429,),
        sleep_fn=sleep_fn,
    )
    def always_fails():
        raise RetryableError(429)

    try:
        always_fails()
    except RetryableError:
        pass

    expected_delays = [
        backoff_base ** (attempt + 1) for attempt in range(max_attempts)
    ]
    assert delays == expected_delays


# --- Property 4: a function that eventually succeeds within max_attempts
# returns the successful result. ---------------------------------------------


@given(
    max_attempts=st.integers(min_value=1, max_value=5),
    success_call_index=st.integers(min_value=0, max_value=4),
    result=st.integers(),
)
@settings(max_examples=50)
def test_eventual_success_within_max_attempts_returns_result(
    max_attempts, success_call_index, result
):
    # Only exercise cases where success happens within the allowed attempts
    # (initial call is attempt 0; up to max_attempts retries follow).
    if success_call_index > max_attempts:
        return

    sleep_fn, delays = _make_sleep_recorder()
    call_count = {"n": 0}

    @retry_with_backoff(
        max_attempts=max_attempts,
        backoff_base=2,
        retryable_codes=(429,),
        sleep_fn=sleep_fn,
    )
    def eventually_succeeds():
        call_index = call_count["n"]
        call_count["n"] += 1
        if call_index < success_call_index:
            raise RetryableError(429)
        return result

    outcome = eventually_succeeds()

    assert outcome == result
    assert call_count["n"] == success_call_index + 1
    assert len(delays) == success_call_index
