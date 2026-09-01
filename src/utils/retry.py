"""Retry decorator with exponential backoff for rate-limited API calls.

Wraps Foundation Model API calls (and other retryable operations) so that
responses rejected with a retryable status code (e.g. HTTP 429) are retried
with exponential backoff rather than failing immediately.

Validates: Requirements 11.10
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class RetryAttempt:
    """Record of a single retry attempt, for observability."""

    attempt: int
    delay_seconds: float
    status_code: int | None
    error_message: str | None


@dataclass
class RetryLog:
    """Accumulated observability data for a single decorated call."""

    attempts: list[RetryAttempt] = field(default_factory=list)
    succeeded: bool = False
    total_delay_seconds: float = 0.0

    def record(self, attempt: RetryAttempt) -> None:
        self.attempts.append(attempt)
        self.total_delay_seconds += attempt.delay_seconds


def _extract_status_code(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception.

    Supports the common conventions used by `requests`, the Databricks SDK,
    and plain exceptions that set a `status_code` or `response.status_code`
    attribute.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code

    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)

    return None


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_base: int = 2,
    retryable_codes: tuple[int, ...] = (429,),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory that retries a function on retryable errors.

    On each retryable failure, waits `backoff_base ** (attempt + 1)` seconds
    before retrying (i.e. 2s, 4s, 8s for the default `backoff_base=2` and
    `max_attempts=3`). Retry attempts and delays are recorded on the wrapped
    function via the `retry_log` attribute for observability, and the log of
    the most recent call is also attached to `wrapper.last_retry_log`.

    Args:
        max_attempts: Maximum number of retry attempts (not counting the
            initial call).
        backoff_base: Base for the exponential backoff delay calculation.
        retryable_codes: HTTP status codes that should trigger a retry.
        sleep_fn: Sleep function to use between retries (overridable for
            testing so tests do not need to wait in real time).

    Returns:
        A decorator that wraps the target function with retry behaviour.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            log = RetryLog()
            wrapper.last_retry_log = log  # type: ignore[attr-defined]

            for attempt in range(max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    log.succeeded = True
                    return result
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    status_code = _extract_status_code(exc)
                    is_retryable = status_code in retryable_codes
                    has_attempts_left = attempt < max_attempts

                    if is_retryable and has_attempts_left:
                        delay = backoff_base ** (attempt + 1)
                        log.record(
                            RetryAttempt(
                                attempt=attempt + 1,
                                delay_seconds=delay,
                                status_code=status_code,
                                error_message=str(exc),
                            )
                        )
                        sleep_fn(delay)
                        continue

                    # Not retryable, or attempts exhausted: propagate.
                    log.record(
                        RetryAttempt(
                            attempt=attempt + 1,
                            delay_seconds=0.0,
                            status_code=status_code,
                            error_message=str(exc),
                        )
                    )
                    raise

            # Unreachable: loop always returns or raises.
            raise RuntimeError("retry_with_backoff: exhausted loop unexpectedly")

        wrapper.last_retry_log = None  # type: ignore[attr-defined]
        return wrapper

    return decorator
