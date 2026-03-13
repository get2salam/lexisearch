"""Tests for the lexisearch.conversation package."""

from __future__ import annotations

from lexisearch.conversation import (
    ConversationHistory,
    ConversationMemory,
    Message,
    QueryContextualizer,
    Role,
)

# ---------------------------------------------------------------------------
# ConversationHistory tests
# ---------------------------------------------------------------------------


class TestConversationHistory:
    def test_add_user(self) -> None:
        h = ConversationHistory()
        msg = h.add_user("Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert len(h) == 1

    def test_add_assistant(self) -> None:
        h = ConversationHistory()
        msg = h.add_assistant("Hi there")
        assert msg.role == Role.ASSISTANT

    def test_add_system(self) -> None:
        h = ConversationHistory()
        msg = h.add_system("You are a helpful assistant.")
        assert msg.role == Role.SYSTEM

    def test_message_order(self) -> None:
        h = ConversationHistory()
        h.add_user("Q1")
        h.add_assistant("A1")
        h.add_user("Q2")
        assert h.messages[0].content == "Q1"
        assert h.messages[1].content == "A1"
        assert h.messages[2].content == "Q2"

    def test_last_user_message(self) -> None:
        h = ConversationHistory()
        h.add_user("First")
        h.add_assistant("Response")
        h.add_user("Second")
        assert h.last_user_message is not None
        assert h.last_user_message.content == "Second"

    def test_last_assistant_message(self) -> None:
        h = ConversationHistory()
        h.add_assistant("First")
        h.add_user("Q")
        h.add_assistant("Second")
        assert h.last_assistant_message is not None
        assert h.last_assistant_message.content == "Second"

    def test_last_user_message_none_when_empty(self) -> None:
        assert ConversationHistory().last_user_message is None

    def test_total_tokens_positive(self) -> None:
        h = ConversationHistory()
        h.add_user("What is retrieval-augmented generation?")
        assert h.total_tokens > 0

    def test_turn_count(self) -> None:
        h = ConversationHistory()
        h.add_user("Q1")
        h.add_assistant("A1")
        h.add_user("Q2")
        h.add_assistant("A2")
        assert h.turn_count == 2

    def test_messages_by_role(self) -> None:
        h = ConversationHistory()
        h.add_user("U1")
        h.add_assistant("A1")
        h.add_user("U2")
        users = h.messages_by_role(Role.USER)
        assert len(users) == 2
        assert all(m.role == Role.USER for m in users)

    def test_recent(self) -> None:
        h = ConversationHistory()
        for i in range(5):
            h.add_user(f"Q{i}")
        recent = h.recent(3)
        assert len(recent) == 3
        assert recent[0].content == "Q2"

    def test_clear(self) -> None:
        h = ConversationHistory()
        h.add_user("Hello")
        h.clear()
        assert len(h) == 0

    def test_unique_ids(self) -> None:
        h1 = ConversationHistory()
        h2 = ConversationHistory()
        assert h1.id != h2.id

    def test_message_metadata(self) -> None:
        h = ConversationHistory()
        msg = h.add_user("Query", source="web", confidence=0.9)
        assert msg.metadata["source"] == "web"


# ---------------------------------------------------------------------------
# ConversationMemory tests
# ---------------------------------------------------------------------------


class TestConversationMemory:
    def test_basic_add(self) -> None:
        mem = ConversationMemory()
        mem.add_user("Hello")
        mem.add_assistant("Hi")
        assert len(mem) == 2

    def test_max_turns_sliding_window(self) -> None:
        mem = ConversationMemory(max_turns=2)
        # Add 4 turns (8 messages)
        for i in range(4):
            mem.add_user(f"Q{i}")
            mem.add_assistant(f"A{i}")
        # Should keep only last 2 turns (4 non-system messages)
        non_system = [m for m in mem.history.messages if m.role != Role.SYSTEM]
        assert len(non_system) <= 4

    def test_system_messages_preserved(self) -> None:
        mem = ConversationMemory(max_turns=1, preserve_system=True)
        mem.add_system("You are helpful.")
        for i in range(5):
            mem.add_user(f"Q{i}")
            mem.add_assistant(f"A{i}")
        sys_msgs = [m for m in mem.history.messages if m.role == Role.SYSTEM]
        assert len(sys_msgs) == 1

    def test_system_messages_dropped_when_disabled(self) -> None:
        mem = ConversationMemory(max_turns=1, preserve_system=False)
        mem.add_system("Instruction")
        mem.add_user("Q1")
        mem.add_assistant("A1")
        mem.add_user("Q2")
        mem.add_assistant("A2")
        sys_msgs = [m for m in mem.history.messages if m.role == Role.SYSTEM]
        assert len(sys_msgs) == 0

    def test_token_budget_respected(self) -> None:
        # Each message ~25 chars → ~6 tokens estimate; budget=50 keeps ~8 messages
        mem = ConversationMemory(max_turns=100, max_tokens=50)
        for i in range(20):
            mem.add_user(f"Q{i} short query")
            mem.add_assistant(f"A{i} short reply")
        assert mem.history.total_tokens <= 50

    def test_get_context_returns_messages(self) -> None:
        mem = ConversationMemory()
        mem.add_user("Q1")
        mem.add_assistant("A1")
        ctx = mem.get_context()
        assert len(ctx) == 2

    def test_get_context_respects_max_turns_override(self) -> None:
        mem = ConversationMemory(max_turns=10)
        for i in range(5):
            mem.add_user(f"Q{i}")
            mem.add_assistant(f"A{i}")
        ctx = mem.get_context(max_turns=2)
        non_system = [m for m in ctx if m.role != Role.SYSTEM]
        assert len(non_system) <= 4

    def test_format_context_contains_roles(self) -> None:
        mem = ConversationMemory()
        mem.add_user("What is RAG?")
        mem.add_assistant("RAG is retrieval-augmented generation.")
        text = mem.format_context()
        assert "USER:" in text
        assert "ASSISTANT:" in text

    def test_clear(self) -> None:
        mem = ConversationMemory()
        mem.add_user("hi")
        mem.clear()
        assert len(mem) == 0

    def test_repr(self) -> None:
        mem = ConversationMemory(max_turns=5, max_tokens=1000)
        r = repr(mem)
        assert "max_turns=5" in r
        assert "max_tokens=1000" in r

    def test_add_system(self) -> None:
        mem = ConversationMemory()
        msg = mem.add_system("Be concise.")
        assert msg.role == Role.SYSTEM


# ---------------------------------------------------------------------------
# QueryContextualizer tests
# ---------------------------------------------------------------------------


class TestQueryContextualizer:
    def _make_history(self, turns: list[tuple[str, str]]) -> ConversationHistory:
        h = ConversationHistory()
        for user, assistant in turns:
            h.add_user(user)
            h.add_assistant(assistant)
        return h

    def test_is_followup_detected(self) -> None:
        ctx = QueryContextualizer()
        assert ctx.is_followup("tell me more about that")
        assert ctx.is_followup("expand on the second point")
        assert ctx.is_followup("what about the other approach?")

    def test_is_followup_not_detected(self) -> None:
        ctx = QueryContextualizer()
        assert not ctx.is_followup("What is the capital of France?")
        assert not ctx.is_followup("Explain transformer models from scratch.")

    def test_contextualize_empty_history_returns_query(self) -> None:
        ctx = QueryContextualizer()
        h = ConversationHistory()
        result = ctx.contextualize("Any question?", h)
        assert result == "Any question?"

    def test_contextualize_includes_history(self) -> None:
        ctx = QueryContextualizer()
        h = self._make_history([("What is dense retrieval?", "It uses vectors.")])
        result = ctx.contextualize("Tell me more.", h)
        assert "Previous question:" in result
        assert "Current question:" in result
        assert "Tell me more." in result

    def test_contextualize_truncates_long_assistant_msg(self) -> None:
        ctx = QueryContextualizer()
        long_answer = "x" * 500
        h = self._make_history([("Short question?", long_answer)])
        result = ctx.contextualize("Follow up?", h)
        assert "..." in result  # truncation indicator

    def test_standalone_query_non_followup_unchanged(self) -> None:
        ctx = QueryContextualizer()
        h = self._make_history([("Q?", "A.")])
        result = ctx.standalone_query("What is the Earth's diameter?", h)
        assert result == "What is the Earth's diameter?"

    def test_standalone_query_followup_adds_hint(self) -> None:
        ctx = QueryContextualizer()
        h = self._make_history([("What are embedding models?", "They encode text as vectors.")])
        result = ctx.standalone_query("Tell me more about that.", h)
        assert "in the context of" in result
        assert "What are embedding models?" in result or "What are embedding" in result

    def test_standalone_query_empty_history_unchanged(self) -> None:
        ctx = QueryContextualizer()
        h = ConversationHistory()
        result = ctx.standalone_query("Tell me more.", h)
        assert result == "Tell me more."

    def test_context_window_limits_history(self) -> None:
        ctx = QueryContextualizer(context_window=1)
        h = self._make_history(
            [
                ("Q1", "A1"),
                ("Q2", "A2"),
                ("Q3", "A3"),
            ]
        )
        result = ctx.contextualize("Follow up?", h)
        # With window=1 we should see Q3/A3 but NOT Q1/A1
        assert "Q1" not in result
        assert "Q3" in result

    def test_custom_followup_patterns_via_subclass(self) -> None:
        from typing import ClassVar

        class CustomCtx(QueryContextualizer):
            FOLLOWUP_PATTERNS: ClassVar[list[str]] = ["please elaborate"]

        ctx = CustomCtx()
        assert ctx.is_followup("please elaborate on that")
        assert not ctx.is_followup("tell me more")  # not in custom list

    def test_message_repr(self) -> None:
        msg = Message(role=Role.USER, content="Hello world, this is a test message here")
        assert "user" in repr(msg)

    def test_history_repr(self) -> None:
        h = ConversationHistory()
        h.add_user("x")
        assert "messages=1" in repr(h)
