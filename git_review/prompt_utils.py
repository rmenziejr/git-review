"""Utilities for loading, validating, and rendering Jinja2 prompt templates.

Every component in git-review that calls an LLM accepts an optional
``system_prompt`` string that is treated as a Jinja2 template.  This module
provides the shared helpers used to load, validate, and render those
templates.

Validation rules
----------------
Before any LLM call is made, :func:`validate_prompt_template` checks that:

1. The template string is syntactically valid Jinja2.
2. Every variable referenced in the template (``{{ var }}``) is present in
   the set of *available* context variables for that component.  Referencing
   an unknown variable is an error because Jinja2 would silently produce an
   empty string, which is almost certainly unintended.

Available context variables per component
------------------------------------------
:class:`~git_review.llm_client.LLMClient`:
    ``n``           – formatted summary, e.g. ``"5 commits, 3 issues, 2 PRs"``
    ``n_commits``   – integer commit count
    ``n_issues``    – integer issue count
    ``n_prs``       – integer pull-request count

:class:`~git_review.commit_message_generator.CommitMessageGenerator`:
    *(no template variables – the prompt is passed as static text)*

:class:`~git_review.issue_factory.IssueFactory`:
    *(no template variables – the prompt is passed as static text)*
"""

from __future__ import annotations

import jinja2
from jinja2 import meta


def load_prompt_file(path: str) -> str:
    """Read and return the contents of *path* as a UTF-8 string.

    Parameters
    ----------
    path:
        Filesystem path to the Jinja2 template file.

    Raises
    ------
    OSError
        If the file cannot be read.
    """
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def validate_prompt_template(
    template_str: str,
    available_vars: set[str],
    *,
    label: str = "prompt template",
) -> None:
    """Validate *template_str* as a Jinja2 template.

    Checks that the template is syntactically valid **and** that every
    variable it references is present in *available_vars*.

    Parameters
    ----------
    template_str:
        The Jinja2 template source to validate.
    available_vars:
        Variables that will be provided when the template is rendered.
        Any ``{{ var }}`` reference that is *not* in this set is reported
        as an error.
    label:
        Human-readable name used in error messages (e.g. the filename).

    Raises
    ------
    ValueError
        If the template is syntactically invalid or references a variable
        that is not in *available_vars*.
    """
    env = jinja2.Environment()
    try:
        ast = env.parse(template_str)
    except jinja2.TemplateSyntaxError as exc:
        raise ValueError(f"Invalid Jinja2 syntax in {label}: {exc}") from exc

    referenced = meta.find_undeclared_variables(ast)
    unknown = referenced - available_vars
    if unknown:
        available_str = (
            ", ".join(sorted(available_vars)) if available_vars else "(none)"
        )
        raise ValueError(
            f"{label} references unknown variable(s): "
            f"{', '.join(sorted(unknown))}. "
            f"Available variables are: {available_str}."
        )


def render_prompt(template_str: str, **kwargs: object) -> str:
    """Render *template_str* as a Jinja2 template using *kwargs* as context.

    Parameters
    ----------
    template_str:
        The Jinja2 template source.
    **kwargs:
        Template context variables.

    Returns
    -------
    str
        The rendered string.
    """
    env = jinja2.Environment()
    return env.from_string(template_str).render(**kwargs)
