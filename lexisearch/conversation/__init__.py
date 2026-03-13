"""Multi-turn conversation memory for LexiSearch RAG pipelines.

This package provides the building blocks for maintaining conversation state
across multiple user-assistant turns, enabling follow-up question handling
and context-aware retrieval.

Components:

- :class:`~lexisearch.conversation.history.Role` — user / assistant / system role enum.
- :class:`~lexisearch.conversation.history.Message` — a single conversation message.
- :class:`~lexisearch.conversation.history.ConversationHistory` — ordered message log.
- :class:`~lexisearch.conversation.memory.ConversationMemory` — memory with sliding-window
  and token-budget compaction.
- :class:`~lexisearch.conversation.contextualizer.QueryContextualizer` — reformulates
  follow-up queries into self-contained retrieval queries.

Example:
    >>> from lexisearch.conversation import ConversationMemory, QueryContextualizer
    >>> mem = ConversationMemory(max_turns=5)
    >>> mem.add_user("What is dense retrieval?")
    Message(role='user', ...)
    >>> mem.add_assistant("Dense retrieval encodes queries and documents as vectors...")
    Message(role='assistant', ...)
    >>> ctx = QueryContextualizer()
    >>> ctx.standalone_query("Tell me more about that.", mem.history)
    'Tell me more about that. (in the context of: What is dense retrieval?)'
"""

from __future__ import annotations

from lexisearch.conversation.contextualizer import QueryContextualizer
from lexisearch.conversation.history import ConversationHistory, Message, Role
from lexisearch.conversation.memory import ConversationMemory

__all__ = [
    "ConversationHistory",
    "ConversationMemory",
    "Message",
    "QueryContextualizer",
    "Role",
]
