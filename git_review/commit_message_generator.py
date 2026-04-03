"""Commit message generator for git-review.

Uses the LLM to write a conventional commit message based on the staged
git diff in the current working directory.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

from .prompt_utils import validate_prompt_template

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

# No template variables are rendered into the commit-message system prompt.
_PROMPT_VARS: set[str] = set()

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert software engineer writing Git commit messages.
Given a git diff, produce a single commit message following the
Conventional Commits specification (https://www.conventionalcommits.org/).

Rules:
- First line: <type>(<optional scope>): <short description>  (max 72 chars)
  Valid types: feat, fix, docs, style, refactor, perf, test, chore, build, ci
- Leave a blank line after the first line.
- Optionally add a body with more detail (wrap at 72 chars).
- Be concise and factual. Do not invent changes not shown in the diff.
- Output ONLY the commit message text, no extra commentary.
"""


def get_git_diff(repo_path: str = ".") -> str:
    """Return the staged git diff, or the unstaged diff if nothing is staged.

    Parameters
    ----------
    repo_path:
        Path to the git repository root.  Defaults to the current directory.

    Returns
    -------
    str
        The diff text, or an empty string when there are no changes.

    Raises
    ------
    RuntimeError
        If the ``git`` command is not available or the directory is not a git
        repository.
    """
    try:
        staged = subprocess.run(
            ["git", "diff", "--staged"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        if staged.stdout.strip():
            return staged.stdout

        unstaged = subprocess.run(
            ["git", "diff"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return unstaged.stdout
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git executable not found. Please install Git."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git command failed: {exc.stderr.strip()}"
        ) from exc


class CommitMessageGenerator:
    """Generate a commit message from a git diff using an LLM.

    Parameters
    ----------
    api_key:
        OpenAI (or compatible) API key.  Falls back to the
        ``OPENAI_API_KEY`` environment variable when *None*.
    model:
        Model identifier, e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``.
    base_url:
        Custom endpoint for non-OpenAI providers (Ollama, Azure, Groq …).
    system_prompt:
        Optional Jinja2 template string to replace the built-in system
        prompt.  The commit-message prompt has no template variables, so
        any ``{{ var }}`` expression will raise a :exc:`ValueError` at
        construction time.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        thinking_mode: Optional[str] = False,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required. "
                "Install it with:  pip install openai"
            )

        if system_prompt is not None:
            validate_prompt_template(
                system_prompt,
                _PROMPT_VARS,
                label="system_prompt",
            )
        self.thinking_mode = thinking_mode
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def generate(self, diff: str) -> str:
        """Return a commit message string for *diff*.

        Parameters
        ----------
        diff:
            The output of ``git diff --staged`` (or ``git diff``).
        """
        logger.debug(
            "Sending diff (%d chars) to %s for commit message generation",
            len(diff),
            self._model,
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": diff},
            ],
            extra_body={"chat_template_kwargs":{"enable_thinking":self.thinking_mode}}
        )
        return (response.choices[0].message.content or "").strip()
