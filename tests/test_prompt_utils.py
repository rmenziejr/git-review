"""Tests for git_review.prompt_utils."""

from __future__ import annotations

import tempfile
import os

import pytest

from git_review.prompt_utils import (
    load_prompt_file,
    render_prompt,
    validate_prompt_template,
)


# ---------------------------------------------------------------------------
# load_prompt_file
# ---------------------------------------------------------------------------

def test_load_prompt_file_reads_content() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False, encoding="utf-8") as fh:
        fh.write("Hello {{ name }}!")
        path = fh.name
    try:
        result = load_prompt_file(path)
        assert result == "Hello {{ name }}!"
    finally:
        os.unlink(path)


def test_load_prompt_file_raises_on_missing_file() -> None:
    with pytest.raises(OSError):
        load_prompt_file("/nonexistent/path/to/prompt.j2")


# ---------------------------------------------------------------------------
# validate_prompt_template
# ---------------------------------------------------------------------------

def test_validate_accepts_valid_template_with_known_vars() -> None:
    validate_prompt_template("Hello {{ name }}!", {"name"})  # should not raise


def test_validate_accepts_template_with_no_vars() -> None:
    validate_prompt_template("Static prompt text.", set())  # should not raise


def test_validate_accepts_subset_of_available_vars() -> None:
    # Template only uses some of the available vars – that is fine
    validate_prompt_template("{{ n }} commits total.", {"n", "n_commits", "n_issues"})


def test_validate_raises_on_unknown_variable() -> None:
    with pytest.raises(ValueError, match="unknown variable"):
        validate_prompt_template("{{ bogus }}", set())


def test_validate_raises_on_multiple_unknown_variables() -> None:
    with pytest.raises(ValueError, match="unknown variable"):
        validate_prompt_template("{{ a }} and {{ b }}", {"a"})


def test_validate_raises_on_syntax_error() -> None:
    with pytest.raises(ValueError, match="Invalid Jinja2 syntax"):
        validate_prompt_template("{% for %}", set())


def test_validate_includes_available_vars_in_error_message() -> None:
    with pytest.raises(ValueError, match="n_commits"):
        validate_prompt_template("{{ unknown }}", {"n_commits"})


def test_validate_uses_label_in_error_message() -> None:
    with pytest.raises(ValueError, match="my-prompt.j2"):
        validate_prompt_template("{{ x }}", set(), label="my-prompt.j2")


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------

def test_render_prompt_substitutes_variables() -> None:
    result = render_prompt("{{ n }} total items.", n="5 commits")
    assert result == "5 commits total items."


def test_render_prompt_with_no_variables() -> None:
    result = render_prompt("Static text.")
    assert result == "Static text."


def test_render_prompt_with_multiple_variables() -> None:
    result = render_prompt(
        "{{ n_commits }} commits, {{ n_issues }} issues",
        n_commits=3,
        n_issues=7,
    )
    assert result == "3 commits, 7 issues"


def test_render_prompt_leaves_unused_vars_alone() -> None:
    # Extra context vars that template doesn't use should be silently ignored
    result = render_prompt("Hello!", n_commits=5, n_issues=2)
    assert result == "Hello!"
