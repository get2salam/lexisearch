"""Conversation history data models.

Provides :class:`Message`, :class:`Role`, and :class:`ConversationHistory`
for tracking multi-turn exchanges in RAG-powered chat applications.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Role(Enum):
    """Speaker role in a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """A single message within a conversation.

    Attributes:
        role: Who produced the message (user, assistant, or system).
        content: The text content of the message.
        id: Unique identifier for the message.
        timestamp: UTC time the message was created.
        metadata: Arbitrary key-value metadata (e.g., token counts, sources).
    """

    role: Role
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Estimate the token count for this message (chars / 4).

        Returns:
            A rough token count, always >= 1.
        """
        return max(1, len(self.content) // 4)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return f"Message(role={self.role.value!r}, content={preview!r})"


@dataclass
class ConversationHistory:
    """An ordered sequence of conversation messages.

    Provides helpers for appending messages and querying the history
    by role or recency.

    Attributes:
        id: Unique conversation identifier.
        messages: Ordered list of messages.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add(self, role: Role, content: str, **metadata: Any) -> Message:
        """Append a new message and return it.

        Args:
            role: Speaker role.
            content: Message text.
            **metadata: Extra metadata key-value pairs.

        Returns:
            The created :class:`Message`.
        """
        msg = Message(role=role, content=content, metadata=dict(metadata))
        self.messages.append(msg)
        return msg

    def add_user(self, content: str, **metadata: Any) -> Message:
        """Append a user message.

        Args:
            content: User message text.
            **metadata: Extra metadata.

        Returns:
            The created :class:`Message`.
        """
        return self.add(Role.USER, content, **metadata)

    def add_assistant(self, content: str, **metadata: Any) -> Message:
        """Append an assistant message.

        Args:
            content: Assistant message text.
            **metadata: Extra metadata.

        Returns:
            The created :class:`Message`.
        """
        return self.add(Role.ASSISTANT, content, **metadata)

    def add_system(self, content: str, **metadata: Any) -> Message:
        """Append a system message.

        Args:
            content: System prompt or instruction text.
            **metadata: Extra metadata.

        Returns:
            The created :class:`Message`.
        """
        return self.add(Role.SYSTEM, content, **metadata)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def last_user_message(self) -> Message | None:
        """Return the most recent user message, or None if none exists.

        Returns:
            The last user :class:`Message` or ``None``.
        """
        for msg in reversed(self.messages):
            if msg.role == Role.USER:
                return msg
        return None

    @property
    def last_assistant_message(self) -> Message | None:
        """Return the most recent assistant message, or None if none exists.

        Returns:
            The last assistant :class:`Message` or ``None``.
        """
        for msg in reversed(self.messages):
            if msg.role == Role.ASSISTANT:
                return msg
        return None

    @property
    def total_tokens(self) -> int:
        """Estimate total token count across all messages.

        Returns:
            Sum of per-message token estimates.
        """
        return sum(m.token_estimate for m in self.messages)

    @property
    def turn_count(self) -> int:
        """Count the number of complete user-assistant turn pairs.

        Returns:
            Number of (user, assistant) pairs.
        """
        user_count = sum(1 for m in self.messages if m.role == Role.USER)
        assistant_count = sum(1 for m in self.messages if m.role == Role.ASSISTANT)
        return min(user_count, assistant_count)

    def messages_by_role(self, role: Role) -> list[Message]:
        """Return all messages with the given role.

        Args:
            role: The :class:`Role` to filter by.

        Returns:
            Filtered list of messages.
        """
        return [m for m in self.messages if m.role == role]

    def recent(self, n: int) -> list[Message]:
        """Return the last *n* messages.

        Args:
            n: Number of messages to return.

        Returns:
            The most recent up-to-n messages.
        """
        return self.messages[-n:] if n > 0 else []

    def clear(self) -> None:
        """Remove all messages from the history."""
        self.messages.clear()

    def __len__(self) -> int:
        """Return total message count."""
        return len(self.messages)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"ConversationHistory(id={self.id!r}, messages={len(self.messages)})"
