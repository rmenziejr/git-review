"""LangGraph middleware utilities for git-review.

This module provides a collection of **middleware** decorators/wrappers that
can be applied to any LangGraph node function.  Each wrapper adds cross-cutting
behaviour (logging, token counting, retry logic) without polluting the core
node business logic.

Design
------
All middleware is implemented as a *higher-order function* that accepts a node
callable and returns a new callable with the same signature::

    wrapped_node = SomeMiddleware(...)(original_node)

Multiple middleware can be stacked via :func:`apply_middleware`::

    node = apply_middleware(
        my_node,
        LoggingMiddleware(logger=my_logger),
        TokenCountingMiddleware(counter=my_counter),
        RetryMiddleware(max_attempts=3),
    )

Available middleware
--------------------
:class:`LoggingMiddleware`
    Logs the node name, elapsed time, and whether the call succeeded.

:class:`TokenCountingMiddleware`
    Estimates token usage for the ``summary_text`` field produced by a node
    and records it in a shared :class:`TokenUsageCounter`.

:class:`RetryMiddleware`
    Wraps a node with exponential-backoff retry logic so transient errors
    (e.g. rate limits, network blips) are handled automatically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type alias: a LangGraph node is any callable that accepts a state dict and
# returns a (partial) state dict.
NodeFunc = Callable[[Any], Any]


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------

class LoggingMiddleware:
    """Middleware that logs entry, exit, and elapsed time for each node call.

    Parameters
    ----------
    node_logger:
        The :class:`logging.Logger` to write to.  Defaults to the module
        logger when *None*.
    level:
        Logging level for success messages (default: ``logging.DEBUG``).

    Example
    -------
    ::

        import logging
        from git_review.langgraph_middleware import LoggingMiddleware

        validate_node = LoggingMiddleware()(original_validate_node)
    """

    def __init__(
        self,
        node_logger: logging.Logger | None = None,
        level: int = logging.DEBUG,
    ) -> None:
        self._logger = node_logger or logger
        self._level = level

    def __call__(self, node_func: NodeFunc) -> NodeFunc:
        """Return a wrapped version of *node_func* with logging."""
        func_name = getattr(node_func, "__name__", repr(node_func))
        log = self._logger
        level = self._level

        def wrapper(state: Any) -> Any:
            log.log(level, "[%s] entering node", func_name)
            start = time.perf_counter()
            try:
                result = node_func(state)
                elapsed = time.perf_counter() - start
                log.log(level, "[%s] completed in %.3fs", func_name, elapsed)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                log.error(
                    "[%s] failed after %.3fs: %s",
                    func_name,
                    elapsed,
                    exc,
                    exc_info=True,
                )
                raise

        wrapper.__name__ = func_name
        return wrapper


# ---------------------------------------------------------------------------
# TokenCountingMiddleware
# ---------------------------------------------------------------------------

@dataclass
class TokenUsageCounter:
    """Accumulates approximate token usage recorded by :class:`TokenCountingMiddleware`.

    Attributes
    ----------
    total_tokens:
        Running total of estimated tokens across all wrapped node calls.
    by_node:
        Per-node token totals, keyed by node function name.
    """

    total_tokens: int = 0
    by_node: dict[str, int] = field(default_factory=dict)

    def record(self, node_name: str, tokens: int) -> None:
        """Add *tokens* to the total and to the *node_name* bucket."""
        self.total_tokens += tokens
        self.by_node[node_name] = self.by_node.get(node_name, 0) + tokens


class TokenCountingMiddleware:
    """Middleware that estimates token usage for the text produced by a node.

    The estimation uses a simple heuristic: **1 token ≈ 4 characters**.  This
    is intentionally approximate; replace the implementation with a proper
    tokeniser (e.g. ``tiktoken``) if accuracy matters.

    The middleware inspects the ``summary_text`` field of the node's return
    value.  Nodes that do not produce ``summary_text`` contribute 0 tokens.

    Parameters
    ----------
    counter:
        A :class:`TokenUsageCounter` instance shared across wrapped nodes.
        If *None*, a new counter is created and stored on ``self.counter``.

    Example
    -------
    ::

        from git_review.langgraph_middleware import TokenCountingMiddleware, TokenUsageCounter

        counter = TokenUsageCounter()
        summarize = TokenCountingMiddleware(counter=counter)(original_summarize)
        # … run graph …
        print(f"Total tokens used: {counter.total_tokens}")
    """

    #: Approximate characters per token.
    CHARS_PER_TOKEN: int = 4

    def __init__(self, counter: TokenUsageCounter | None = None) -> None:
        self.counter: TokenUsageCounter = counter or TokenUsageCounter()

    def __call__(self, node_func: NodeFunc) -> NodeFunc:
        """Return a wrapped version of *node_func* that counts tokens."""
        func_name = getattr(node_func, "__name__", repr(node_func))
        counter = self.counter
        chars_per_token = self.CHARS_PER_TOKEN

        def wrapper(state: Any) -> Any:
            result = node_func(state)
            text = (result or {}).get("summary_text", "") if isinstance(result, dict) else ""
            tokens = max(0, len(text) // chars_per_token)
            counter.record(func_name, tokens)
            logger.debug(
                "[%s] token usage (approx): %d tokens from %d chars",
                func_name,
                tokens,
                len(text),
            )
            return result

        wrapper.__name__ = func_name
        return wrapper


# ---------------------------------------------------------------------------
# RetryMiddleware
# ---------------------------------------------------------------------------

class RetryMiddleware:
    """Middleware that retries a failing node with exponential backoff.

    The node is retried up to *max_attempts* times.  The wait between
    attempts starts at *initial_delay* seconds and doubles with each attempt
    up to *max_delay* seconds.

    Parameters
    ----------
    max_attempts:
        Maximum number of total attempts (including the first).
    initial_delay:
        Seconds to wait before the first retry.
    max_delay:
        Maximum wait time between retries.
    exceptions:
        Tuple of exception types to catch and retry.  All other exceptions
        propagate immediately.

    Example
    -------
    ::

        from git_review.langgraph_middleware import RetryMiddleware

        summarize = RetryMiddleware(max_attempts=3)(original_summarize)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._max_attempts = max_attempts
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._exceptions = exceptions

    def __call__(self, node_func: NodeFunc) -> NodeFunc:
        """Return a wrapped version of *node_func* with retry logic."""
        func_name = getattr(node_func, "__name__", repr(node_func))
        max_attempts = self._max_attempts
        initial_delay = self._initial_delay
        max_delay = self._max_delay
        exceptions = self._exceptions

        def wrapper(state: Any) -> Any:
            delay = initial_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return node_func(state)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "[%s] all %d attempts failed; re-raising: %s",
                            func_name,
                            max_attempts,
                            exc,
                        )
                        raise
                    logger.warning(
                        "[%s] attempt %d/%d failed (%s); retrying in %.1fs",
                        func_name,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
            raise last_exc  # type: ignore[misc]  # unreachable but satisfies mypy

        wrapper.__name__ = func_name
        return wrapper


# ---------------------------------------------------------------------------
# apply_middleware helper
# ---------------------------------------------------------------------------

def apply_middleware(node_func: NodeFunc, *middleware: Any) -> NodeFunc:
    """Apply multiple middleware wrappers to *node_func* in order.

    Middleware are applied from **left to right**: the first middleware in
    the list becomes the outermost wrapper.

    Parameters
    ----------
    node_func:
        The node callable to wrap.
    *middleware:
        One or more middleware instances (e.g. :class:`LoggingMiddleware`,
        :class:`TokenCountingMiddleware`, :class:`RetryMiddleware`).

    Returns
    -------
    NodeFunc
        The wrapped node callable.

    Example
    -------
    ::

        from git_review.langgraph_middleware import (
            apply_middleware,
            LoggingMiddleware,
            RetryMiddleware,
            TokenCountingMiddleware,
            TokenUsageCounter,
        )

        counter = TokenUsageCounter()
        wrapped = apply_middleware(
            summarize_node,
            LoggingMiddleware(),
            TokenCountingMiddleware(counter=counter),
            RetryMiddleware(max_attempts=3),
        )
    """
    wrapped = node_func
    # Apply middleware in reverse order so the first item in `middleware` ends
    # up as the *outermost* wrapper (i.e. the first to be entered and the last
    # to be exited), matching the documented "left to right" behaviour.
    for mw in reversed(middleware):
        wrapped = mw(wrapped)
    return wrapped
