"""Conversation memory with configurable retention policies.

:class:`ConversationMemory` wraps :class:`ConversationHistory` and enforces
sliding-window (max turns) and token-budget constraints so that the context
window passed to an LLM never exceeds configured limits.
"""

from __future__ import annotations

from lexisearch.conversation.history import ConversationHistory, Message, Role


class ConversationMemory:
    """Manages conversation context with automatic compaction.

    Supported retention policies:

    - **Max turns** — keeps at most ``max_turns`` user-assistant pairs.
    - **Max tokens** — trims oldest non-system messages until the total
      estimated token count fits within the budget.
    - **System preservation** — system messages are always kept when
      ``preserve_system=True`` (default).

    Example:
        >>> mem = ConversationMemory(max_turns=3)
        >>> mem.add_user("What is RAG?")
        Message(role='user', ...)
        >>> mem.add_assistant("RAG stands for Retrieval-Augmented Generation.")
        Message(role='assistant', ...)
        >>> len(mem)
        2
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 4000,
        preserve_system: bool = True,
    ) -> None:
        """Initialise ConversationMemory.

        Args:
            max_turns: Maximum number of user-assistant pairs to retain.
            max_tokens: Maximum total estimated token count across all
                retained messages.
            preserve_system: Always keep system messages regardless of
                compaction policies.
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.preserve_system = preserve_system
        self._history = ConversationHistory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> ConversationHistory:
        """Access the underlying :class:`ConversationHistory`.

        Returns:
            The wrapped history object.
        """
        return self._history

    def add_user(self, content: str, **metadata: object) -> Message:
        """Append a user message and compact if needed.

        Args:
            content: User message text.
            **metadata: Extra metadata forwarded to the message.

        Returns:
            The created :class:`Message`.
        """
        msg = self._history.add_user(content, **metadata)
        self._compact()
        return msg

    def add_assistant(self, content: str, **metadata: object) -> Message:
        """Append an assistant message and compact if needed.

        Args:
            content: Assistant response text.
            **metadata: Extra metadata forwarded to the message.

        Returns:
            The created :class:`Message`.
        """
        msg = self._history.add_assistant(content, **metadata)
        self._compact()
        return msg

    def add_system(self, content: str, **metadata: object) -> Message:
        """Prepend (or append) a system message without compaction.

        System messages are exempt from compaction when
        ``preserve_system=True``.

        Args:
            content: System instruction text.
            **metadata: Extra metadata.

        Returns:
            The created :class:`Message`.
        """
        return self._history.add_system(content, **metadata)

    def get_context(self, max_turns: int | None = None) -> list[Message]:
        """Return messages suitable for use as an LLM context window.

        Args:
            max_turns: Optionally further limit to the last *max_turns* pairs.
                Overrides the instance-level ``max_turns`` for this call.

        Returns:
            Ordered list of :class:`Message` objects.
        """
        msgs = self._history.messages
        turns = max_turns if max_turns is not None else self.max_turns

        system = [m for m in msgs if m.role == Role.SYSTEM] if self.preserve_system else []
        non_system = [m for m in msgs if m.role != Role.SYSTEM]

        # Limit to turns x 2 (each turn = 1 user + 1 assistant)
        recent = non_system[-(turns * 2) :]
        return system + recent

    def format_context(self, max_turns: int | None = None) -> str:
        """Format context messages as a plain-text string.

        Produces lines like ``USER: ...`` and ``ASSISTANT: ...`` that can
        be injected into a prompt template.

        Args:
            max_turns: Optional per-call turn limit.

        Returns:
            Multi-line string with role-prefixed messages.
        """
        return "\n".join(
            f"{msg.role.value.upper()}: {msg.content}" for msg in self.get_context(max_turns)
        )

    def clear(self) -> None:
        """Remove all messages from memory."""
        self._history.clear()

    def __len__(self) -> int:
        """Return the total number of retained messages."""
        return len(self._history)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"ConversationMemory("
            f"messages={len(self)}, max_turns={self.max_turns}, "
            f"max_tokens={self.max_tokens})"
        )

    # ------------------------------------------------------------------
    # Internal compaction
    # ------------------------------------------------------------------

    def _compact(self) -> None:
        """Apply retention policies to the message list in-place."""
        msgs = self._history.messages

        system: list[Message] = []
        non_system: list[Message] = []
        for m in msgs:
            (system if self.preserve_system and m.role == Role.SYSTEM else non_system).append(m)

        # --- Sliding window (max_turns x 2 non-system messages) ---
        limit = self.max_turns * 2
        if len(non_system) > limit:
            non_system = non_system[-limit:]

        # --- Token budget ---
        combined = system + non_system
        while sum(m.token_estimate for m in combined) > self.max_tokens and non_system:
            non_system.pop(0)
            combined = system + non_system

        self._history.messages = combined
