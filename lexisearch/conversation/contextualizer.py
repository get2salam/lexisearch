"""Query contextualisation using conversation history.

:class:`QueryContextualizer` reformulates follow-up queries so that
retrieval systems — which have no inherent notion of conversation state —
receive self-contained, fully-specified query strings.
"""

from __future__ import annotations

from typing import ClassVar

from lexisearch.conversation.history import ConversationHistory, Role


class QueryContextualizer:
    """Reformulates follow-up queries using recent conversation history.

    In multi-turn RAG sessions, users often issue elliptical or anaphoric
    queries that reference earlier context ("Expand on the third point",
    "What did you mean by that?").  This class provides:

    - :meth:`is_followup` — heuristic detection of follow-up queries.
    - :meth:`contextualize` — builds a context-enriched query by prepending
      recent exchange summaries.
    - :meth:`standalone_query` — converts a follow-up into a self-contained
      query suitable for stateless retrieval.

    Args:
        context_window: Number of recent turn *pairs* to include when
            building context. Defaults to 3.

    Example:
        >>> ctx = QueryContextualizer(context_window=2)
        >>> history.add_user("Tell me about transformer architectures.")
        >>> history.add_assistant("Transformers use self-attention...")
        >>> ctx.is_followup("What about the encoder stack?")
        False
        >>> ctx.standalone_query("What about the encoder stack?", history)
        'What about the encoder stack? (in the context of: Tell me about transformer archi...)'
    """

    #: Lowercase phrases that signal a follow-up query
    FOLLOWUP_PATTERNS: ClassVar[list[str]] = [
        "what about",
        "tell me more",
        "expand on",
        "elaborate",
        "the previous",
        "that point",
        "you mentioned",
        "as you said",
        "regarding that",
        "more about that",
        "what else",
        "continue",
        "and also",
        "furthermore",
        "in addition",
        "besides that",
        "going back",
        "as mentioned",
    ]

    def __init__(self, context_window: int = 3) -> None:
        """Initialise the contextualizer.

        Args:
            context_window: Number of recent turn pairs to draw context from.
        """
        self.context_window = context_window

    def is_followup(self, query: str) -> bool:
        """Detect whether a query is likely a follow-up to prior context.

        Uses a keyword/phrase heuristic — no ML model required.

        Args:
            query: The user's current query string.

        Returns:
            ``True`` if the query appears to reference prior context.
        """
        q_lower = query.lower()
        return any(p in q_lower for p in self.FOLLOWUP_PATTERNS)

    def contextualize(self, query: str, history: ConversationHistory) -> str:
        """Build a context-enriched query by prepending recent history.

        The returned string places recent exchanges above the current query
        so that an LLM receiving it understands the conversational background.
        If the history is empty the query is returned unchanged.

        Args:
            query: The current user query.
            history: The full conversation history.

        Returns:
            A string combining recent history and the current query.
        """
        if not history.messages:
            return query

        non_system = [m for m in history.messages if m.role != Role.SYSTEM]
        recent = non_system[-(self.context_window * 2):]
        if not recent:
            return query

        parts: list[str] = []
        for msg in recent:
            if msg.role == Role.USER:
                parts.append(f"Previous question: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                snippet = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                parts.append(f"Previous answer: {snippet}")

        context_block = "\n".join(parts)
        return f"{context_block}\n\nCurrent question: {query}"

    def standalone_query(self, query: str, history: ConversationHistory) -> str:
        """Convert a follow-up query to a self-contained standalone query.

        Useful when the downstream retriever processes queries without
        access to conversation state.  If the query is not detected as a
        follow-up, or the history is empty, it is returned as-is.

        The heuristic appends a parenthetical context hint derived from the
        topic of the most recent user turn.

        Args:
            query: The follow-up query to reformulate.
            history: Conversation history providing contextual topics.

        Returns:
            A standalone query string with embedded context hint if needed.
        """
        if not history.messages:
            return query

        if not self.is_followup(query):
            return query

        last_user = history.last_user_message
        if not last_user:
            return query

        # Use first 60 chars of the last user query as a topic hint
        hint = last_user.content[:60].strip()
        if hint and hint.lower() not in query.lower():
            return f"{query} (in the context of: {hint})"

        return query
