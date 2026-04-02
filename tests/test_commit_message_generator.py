"""Tests for git_review.commit_message_generator."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from git_review.commit_message_generator import CommitMessageGenerator, get_git_diff


# ---------------------------------------------------------------------------
# get_git_diff tests
# ---------------------------------------------------------------------------

SAMPLE_DIFF = """\
diff --git a/foo.py b/foo.py
index 1234567..abcdefg 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
-    pass
+    print("hello")
+
"""


def _make_run(stdout: str, returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    result.stderr = ""
    return result


def test_get_git_diff_returns_staged_when_present() -> None:
    with patch("subprocess.run", return_value=_make_run(SAMPLE_DIFF)) as mock_run:
        diff = get_git_diff()
    assert diff == SAMPLE_DIFF
    # Only one subprocess.run call should be made (staged diff)
    assert mock_run.call_count == 1
    args = mock_run.call_args[0][0]
    assert "--staged" in args


def test_get_git_diff_falls_back_to_unstaged_when_staged_empty() -> None:
    staged_empty = _make_run("")
    unstaged = _make_run(SAMPLE_DIFF)

    with patch("subprocess.run", side_effect=[staged_empty, unstaged]) as mock_run:
        diff = get_git_diff()

    assert diff == SAMPLE_DIFF
    assert mock_run.call_count == 2


def test_get_git_diff_raises_runtime_error_when_git_missing() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        with pytest.raises(RuntimeError, match="git executable not found"):
            get_git_diff()


def test_get_git_diff_raises_runtime_error_on_subprocess_failure() -> None:
    exc = subprocess.CalledProcessError(128, "git", stderr="not a git repository")
    with patch("subprocess.run", side_effect=exc):
        with pytest.raises(RuntimeError, match="git command failed"):
            get_git_diff()


def test_get_git_diff_passes_repo_path() -> None:
    with patch("subprocess.run", return_value=_make_run(SAMPLE_DIFF)) as mock_run:
        get_git_diff("/some/path")
    assert mock_run.call_args[1]["cwd"] == "/some/path"


# ---------------------------------------------------------------------------
# CommitMessageGenerator tests
# ---------------------------------------------------------------------------

def _make_openai_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_generate_returns_commit_message() -> None:
    expected = "feat(foo): add hello implementation"

    mock_openai_cls = MagicMock()
    mock_openai_cls.return_value.chat.completions.create.return_value = (
        _make_openai_response(expected)
    )

    with patch("git_review.commit_message_generator.OpenAI", mock_openai_cls):
        generator = CommitMessageGenerator(api_key="sk-fake")
        result = generator.generate(SAMPLE_DIFF)

    assert result == expected


def test_generate_strips_whitespace() -> None:
    raw = "  feat: add feature  \n\n"

    mock_openai_cls = MagicMock()
    mock_openai_cls.return_value.chat.completions.create.return_value = (
        _make_openai_response(raw)
    )

    with patch("git_review.commit_message_generator.OpenAI", mock_openai_cls):
        generator = CommitMessageGenerator(api_key="sk-fake")
        result = generator.generate(SAMPLE_DIFF)

    assert result == raw.strip()


def test_generate_passes_model_and_base_url() -> None:
    mock_openai_cls = MagicMock()
    mock_openai_cls.return_value.chat.completions.create.return_value = (
        _make_openai_response("fix: something")
    )

    with patch("git_review.commit_message_generator.OpenAI", mock_openai_cls):
        generator = CommitMessageGenerator(
            api_key="sk-fake", model="gpt-4o", base_url="http://localhost/v1"
        )
        generator.generate(SAMPLE_DIFF)

    mock_openai_cls.assert_called_once_with(
        api_key="sk-fake", base_url="http://localhost/v1"
    )
    call_kwargs = mock_openai_cls.return_value.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
