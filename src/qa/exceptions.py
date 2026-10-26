"""Exception hierarchy for the Q&A agent.

Mirrors the style of ``src.memory.exceptions``: every error this package
raises to its callers is a :class:`QAError` (or a subclass), so UI code can
catch one type. Underlying :class:`~src.memory.exceptions.HindsightError`
failures are wrapped rather than leaked, keeping their cause on
``__cause__``.
"""

from __future__ import annotations

from typing import Optional


class QAError(Exception):
    """Base class for every error raised by the Q&A agent."""

    code: str = "qa_error"

    def __init__(self, message: str, *, detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.detail:
            return f"{self.message} ({self.detail})"
        return self.message


class RetrievalError(QAError):
    """The memory store could not be queried."""

    code = "retrieval_error"


class EmptyKnowledgeBaseError(QAError):
    """No project/memory in the store matched the question at all.

    This is not a failure of the agent — it is a correct, explicit "I don't
    know" outcome that must never be silently papered over with a guessed
    answer.
    """

    code = "empty_knowledge_base"


class AnswerNotGroundedError(QAError):
    """A candidate answer could not be tied back to any retrieved source.

    Raised internally by the groundedness check and turned into a refusal
    answer rather than propagated to the caller as a hard failure.
    """

    code = "not_grounded"
