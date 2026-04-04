"""LangGraph-based review pipeline for git-review.

This module implements the same summarisation workflow as
:class:`~git_review.llm_client.LLMClient` but as a LangGraph
:class:`~langgraph.graph.StateGraph`.  Using a graph gives you:

- **Explicit state management** – every step reads from and writes to a
  shared :class:`ReviewState` typed dict.
- **Checkpointing / persistence** – pass a *checkpointer* (e.g.
  ``MemorySaver``) so the graph can be resumed or inspected at any node.
- **Conditional branching** – the ``refine`` node is only reached when the
  initial summary is too short (< :data:`MIN_SUMMARY_CHARS` characters).
- **Easy extensibility** – add new nodes (human-in-the-loop approval,
  Slack notification, …) without touching existing logic.

Quick start
-----------
::

    from git_review.langgraph_pipeline import build_review_graph
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_review_graph(
        openai_api_key="sk-...",
        checkpointer=MemorySaver(),
    )
    result = graph.invoke(
        {"summary": review_summary},
        config={"configurable": {"thread_id": "run-1"}},
    )
    print(result["summary_text"])
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .llm_client import LLMClient, _build_user_message, _DEFAULT_SYSTEM_PROMPT
from .models import ReviewSummary

logger = logging.getLogger(__name__)

try:
    from typing import TypedDict

    from langgraph.graph import END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver  # noqa: F401 – re-exported for callers
except ImportError as _lg_exc:  # pragma: no cover
    raise ImportError(
        "The 'langgraph' package is required for the pipeline. "
        "Install it with:  pip install 'git-review[langgraph]'"
    ) from _lg_exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: If the initial summary is shorter than this many characters, the ``refine``
#: node will attempt to generate a longer, more detailed summary.
MIN_SUMMARY_CHARS: int = 200

#: Refine prompt appended to the system prompt when the initial summary is
#: too short.
_REFINE_INSTRUCTION = (
    "\n\nThe previous summary was too brief. "
    "Please produce a more detailed and comprehensive summary of at least "
    f"{MIN_SUMMARY_CHARS} characters, following all the original guidelines."
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ReviewState(TypedDict, total=False):
    """Mutable state passed between nodes in the review graph.

    Fields
    ------
    summary:
        The :class:`~git_review.models.ReviewSummary` to process.  Populated
        by the caller before invoking the graph.
    validation_errors:
        List of human-readable validation messages produced by
        ``validate_data``.  Empty list means no problems were found.
    summary_text:
        The LLM-generated summary text.  Populated by ``summarize`` and
        optionally updated by ``refine``.
    needs_refinement:
        Set to *True* by ``summarize`` when the generated text is too short.
    """

    summary: ReviewSummary
    validation_errors: list[str]
    summary_text: str
    needs_refinement: bool


# ---------------------------------------------------------------------------
# Node builders (public API)
# ---------------------------------------------------------------------------

def make_validate_node() -> Any:
    """Return a node function that validates the :class:`ReviewSummary`."""

    def validate_data(state: ReviewState) -> ReviewState:
        """Validate the ReviewSummary in *state* and collect any warnings."""
        review: ReviewSummary = state["summary"]
        errors: list[str] = []

        if not review.commits and not review.issues and not review.pull_requests:
            errors.append(
                "No commits, issues, or pull requests found in the given time window."
            )

        if review.since >= review.until:
            errors.append(
                f"Invalid date window: since={review.since.date()} is not before "
                f"until={review.until.date()}."
            )

        for err in errors:
            logger.warning("validate_data: %s", err)

        return {"validation_errors": errors}  # type: ignore[return-value]

    return validate_data


def make_summarize_node(llm_client: LLMClient) -> Any:
    """Return a node function that calls the LLM to generate a summary."""

    def summarize(state: ReviewState) -> ReviewState:
        """Generate a summary using *llm_client* and detect if refinement is needed."""
        review: ReviewSummary = state["summary"]

        logger.debug("summarize: generating summary for %s/%s", review.owner, review.repo)
        text = llm_client.summarise(review)

        needs_refinement = len(text) < MIN_SUMMARY_CHARS
        if needs_refinement:
            logger.debug(
                "summarize: summary too short (%d chars), will refine", len(text)
            )

        return {  # type: ignore[return-value]
            "summary_text": text,
            "needs_refinement": needs_refinement,
        }

    return summarize


def make_refine_node(llm_client: LLMClient, system_prompt: str) -> Any:
    """Return a node function that refines a too-short summary.

    The node re-invokes the LLM with an augmented system prompt that
    explicitly instructs it to produce a longer answer.
    """

    def refine(state: ReviewState) -> ReviewState:
        """Re-invoke the LLM with an instruction to produce a longer summary."""
        review: ReviewSummary = state["summary"]
        logger.debug("refine: requesting longer summary for %s/%s", review.owner, review.repo)

        augmented_prompt = system_prompt + _REFINE_INSTRUCTION
        user_message = _build_user_message(review)

        text = llm_client.invoke_with_prompt(augmented_prompt, user_message)
        review.summary_text = text
        logger.debug("refine: new summary length = %d chars", len(text))

        return {"summary_text": text, "needs_refinement": False}  # type: ignore[return-value]

    return refine


# Private aliases kept for backwards compatibility with any existing call-sites.
_make_validate_node = make_validate_node
_make_summarize_node = make_summarize_node
_make_refine_node = make_refine_node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_summarize(state: ReviewState) -> str:
    """Return ``"refine"`` when the summary needs refinement, ``END`` otherwise."""
    return "refine" if state.get("needs_refinement") else END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_review_graph(
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    checkpointer: Any = None,
) -> Any:
    """Build and compile the review :class:`~langgraph.graph.StateGraph`.

    Parameters
    ----------
    openai_api_key:
        OpenAI (or compatible) API key.  Falls back to the
        ``OPENAI_API_KEY`` environment variable when *None*.
    model:
        LLM model identifier, e.g. ``"gpt-4o"``.
    base_url:
        Custom OpenAI-compatible base URL (Ollama, Azure, Groq …).
    system_prompt:
        Optional Jinja2 template string to override the default system
        prompt.  Passed directly to :class:`~git_review.llm_client.LLMClient`.
    checkpointer:
        LangGraph checkpointer instance (e.g. ``MemorySaver()``) that
        enables state persistence across invocations.  When *None* the
        graph runs without checkpointing.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready for ``graph.invoke(…)``.

    Example
    -------
    ::

        from git_review.langgraph_pipeline import build_review_graph
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_review_graph(
            openai_api_key="sk-...",
            checkpointer=MemorySaver(),
        )
        result = graph.invoke(
            {"summary": review_summary},
            config={"configurable": {"thread_id": "my-run"}},
        )
        print(result["summary_text"])
    """
    llm_client = LLMClient(
        api_key=openai_api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
    )
    resolved_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    # -- Build nodes -------------------------------------------------------
    validate_node = make_validate_node()
    summarize_node = make_summarize_node(llm_client)
    refine_node = make_refine_node(llm_client, resolved_prompt)

    # -- Assemble graph ----------------------------------------------------
    builder: StateGraph = StateGraph(ReviewState)

    builder.add_node("validate_data", validate_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("refine", refine_node)

    builder.set_entry_point("validate_data")
    builder.add_edge("validate_data", "summarize")
    builder.add_conditional_edges(
        "summarize",
        _route_after_summarize,
        {"refine": "refine", END: END},
    )
    builder.add_edge("refine", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(**compile_kwargs)
