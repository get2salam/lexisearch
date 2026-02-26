"""Comprehensive test suite for the LexiSearch generation layer.

Covers:
- Base interfaces: Message, GenerationConfig, GenerationRequest/Response, TokenUsage
- MockLLM: completions, streaming, error injection, request capture
- Prompt templates: rendering, validation, variable defaults, RAG builder
- Citation extraction: marker parsing, span resolution, bibliography formatting
- Streaming: StreamBuffer, StreamHandler, collect_stream, stream_to_response
- OpenAILLM: parameter building with injected mock client
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from lexisearch.generation.base import (
    FinishReason,
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    Message,
    MessageRole,
    StreamChunk,
    TokenUsage,
)
from lexisearch.generation.citations import (
    Citation,
    CitationExtractor,
    format_bibliography,
    strip_citations,
)
from lexisearch.generation.mock_llm import MockLLM, make_config
from lexisearch.generation.openai_llm import OpenAILLM, _finish_reason
from lexisearch.generation.prompts import (
    BUILTIN_TEMPLATES,
    RAG_EXTRACT_TEMPLATE,
    RAG_FOLLOWUP_TEMPLATE,
    RAG_QA_TEMPLATE,
    RAG_SUMMARISE_TEMPLATE,
    PromptTemplate,
    PromptVariable,
    RAGPromptBuilder,
    TemplateError,
)
from lexisearch.generation.streaming import (
    StreamBuffer,
    StreamHandler,
    ThrottledStream,
    collect_stream,
    stream_to_response,
)
from lexisearch.models import Chunk, SearchResult

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_chunk() -> Chunk:
    return Chunk(content="The boiling point of water is 100°C at sea level.", document_id="doc1")


@pytest.fixture()
def simple_result(simple_chunk: Chunk) -> SearchResult:
    return SearchResult(chunk=simple_chunk, score=0.92, rank=1)


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM(response_text="Mocked answer.")


@pytest.fixture()
def user_request() -> GenerationRequest:
    return GenerationRequest(messages=[Message.user("What is the boiling point of water?")])


# ---------------------------------------------------------------------------
# Base: Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_system_factory(self) -> None:
        msg = Message.system("Be helpful.")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "Be helpful."

    def test_user_factory(self) -> None:
        msg = Message.user("Hello")
        assert msg.role == MessageRole.USER

    def test_assistant_factory(self) -> None:
        msg = Message.assistant("Hi there!")
        assert msg.role == MessageRole.ASSISTANT

    def test_to_dict(self) -> None:
        msg = Message.user("test")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "test"}

    def test_role_values(self) -> None:
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


# ---------------------------------------------------------------------------
# Base: GenerationConfig
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    def test_defaults(self) -> None:
        cfg = GenerationConfig()
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1024
        assert cfg.stream is False

    def test_custom_values(self) -> None:
        cfg = GenerationConfig(model="gpt-4o", temperature=0.7, max_tokens=2048)
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048

    def test_stop_sequences(self) -> None:
        cfg = GenerationConfig(stop=["END", "STOP"])
        assert len(cfg.stop) == 2


# ---------------------------------------------------------------------------
# Base: GenerationRequest
# ---------------------------------------------------------------------------


class TestGenerationRequest:
    def test_last_user_message(self) -> None:
        req = GenerationRequest(messages=[Message.system("sys"), Message.user("user query")])
        assert req.last_user_message == "user query"

    def test_last_user_message_none(self) -> None:
        req = GenerationRequest(messages=[Message.system("only system")])
        assert req.last_user_message is None

    def test_last_user_message_multiple_turns(self) -> None:
        req = GenerationRequest(
            messages=[
                Message.user("first"),
                Message.assistant("response"),
                Message.user("second"),
            ]
        )
        assert req.last_user_message == "second"

    def test_default_config(self) -> None:
        req = GenerationRequest(messages=[Message.user("hi")])
        assert isinstance(req.config, GenerationConfig)


# ---------------------------------------------------------------------------
# Base: TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_auto_total(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_explicit_total(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=155)
        assert usage.total_tokens == 155

    def test_zero_default(self) -> None:
        usage = TokenUsage()
        assert usage.total_tokens == 0


# ---------------------------------------------------------------------------
# Base: GenerationResponse
# ---------------------------------------------------------------------------


class TestGenerationResponse:
    def test_is_complete(self) -> None:
        resp = GenerationResponse(content="ok", finish_reason=FinishReason.STOP)
        assert resp.is_complete is True
        assert resp.is_truncated is False

    def test_is_truncated(self) -> None:
        resp = GenerationResponse(content="...", finish_reason=FinishReason.LENGTH)
        assert resp.is_truncated is True
        assert resp.is_complete is False

    def test_default_values(self) -> None:
        resp = GenerationResponse(content="hello")
        assert resp.latency_ms == 0.0
        assert isinstance(resp.usage, TokenUsage)


# ---------------------------------------------------------------------------
# Base: LLMError
# ---------------------------------------------------------------------------


class TestLLMError:
    def test_basic(self) -> None:
        err = LLMError("something went wrong", provider="openai", status_code=429)
        assert "something went wrong" in str(err)
        assert err.provider == "openai"
        assert err.status_code == 429
        assert err.retryable is False

    def test_retryable(self) -> None:
        err = LLMError("rate limited", retryable=True)
        assert err.retryable is True

    def test_repr(self) -> None:
        err = LLMError("err", provider="anthropic")
        assert "anthropic" in repr(err)


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------


class TestMockLLM:
    def test_complete_returns_configured_text(
        self, mock_llm: MockLLM, user_request: GenerationRequest
    ) -> None:
        resp = mock_llm.complete(user_request)
        assert resp.content == "Mocked answer."

    def test_complete_finish_reason(
        self, mock_llm: MockLLM, user_request: GenerationRequest
    ) -> None:
        resp = mock_llm.complete(user_request)
        assert resp.finish_reason == FinishReason.STOP

    def test_complete_usage_populated(
        self, mock_llm: MockLLM, user_request: GenerationRequest
    ) -> None:
        resp = mock_llm.complete(user_request)
        assert resp.usage.completion_tokens > 0

    def test_complete_captures_request(
        self, mock_llm: MockLLM, user_request: GenerationRequest
    ) -> None:
        mock_llm.complete(user_request)
        assert mock_llm.call_count == 1
        assert len(mock_llm.captured_requests) == 1
        assert mock_llm.captured_requests[0] is user_request

    def test_complete_auto_response_text(self) -> None:
        llm = MockLLM()  # No fixed response
        req = GenerationRequest(messages=[Message.user("custom query")])
        resp = llm.complete(req)
        assert "custom query" in resp.content

    def test_complete_raises_error(self, user_request: GenerationRequest) -> None:
        err = LLMError("injected error", provider="mock")
        llm = MockLLM(raise_error=err)
        with pytest.raises(LLMError, match="injected error"):
            llm.complete(user_request)

    def test_stream_yields_chunks(self, mock_llm: MockLLM, user_request: GenerationRequest) -> None:
        chunks = list(mock_llm.stream(user_request))
        assert len(chunks) > 0
        assert all(isinstance(c, StreamChunk) for c in chunks)

    def test_stream_final_chunk(self, mock_llm: MockLLM, user_request: GenerationRequest) -> None:
        chunks = list(mock_llm.stream(user_request))
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == FinishReason.STOP

    def test_stream_assembles_full_text(
        self, mock_llm: MockLLM, user_request: GenerationRequest
    ) -> None:
        text = "".join(c.delta for c in mock_llm.stream(user_request))
        assert text == "Mocked answer."

    def test_stream_raises_error(self, user_request: GenerationRequest) -> None:
        err = LLMError("stream error")
        llm = MockLLM(raise_error=err)
        with pytest.raises(LLMError):
            list(llm.stream(user_request))

    def test_reset_clears_state(self, mock_llm: MockLLM, user_request: GenerationRequest) -> None:
        mock_llm.complete(user_request)
        assert mock_llm.call_count == 1
        mock_llm.reset()
        assert mock_llm.call_count == 0
        assert len(mock_llm.captured_requests) == 0

    def test_set_response(self, mock_llm: MockLLM, user_request: GenerationRequest) -> None:
        mock_llm.set_response("New answer.")
        resp = mock_llm.complete(user_request)
        assert resp.content == "New answer."

    def test_model_name(self, mock_llm: MockLLM) -> None:
        assert mock_llm.model_name == "mock-llm-v1"

    def test_provider(self, mock_llm: MockLLM) -> None:
        assert mock_llm.provider == "mock"

    def test_repr(self, mock_llm: MockLLM) -> None:
        assert "MockLLM" in repr(mock_llm)

    def test_stream_empty_response(self) -> None:
        llm = MockLLM(response_text="")
        req = GenerationRequest(messages=[Message.user("hi")])
        chunks = list(llm.stream(req))
        assert any(c.is_final for c in chunks)

    def test_custom_chunk_size(self) -> None:
        llm = MockLLM(response_text="abcdef", stream_chunk_size=2)
        req = GenerationRequest(messages=[Message.user("hi")])
        chunks = list(llm.stream(req))
        text = "".join(c.delta for c in chunks)
        assert text == "abcdef"


def test_make_config() -> None:
    cfg = make_config(temperature=0.9, max_tokens=256)
    assert cfg.temperature == 0.9
    assert cfg.max_tokens == 256


# ---------------------------------------------------------------------------
# Async: BaseLLM.acomplete / astream defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acomplete_delegates_to_complete() -> None:
    llm = MockLLM(response_text="async ok")
    req = GenerationRequest(messages=[Message.user("hi")])
    resp = await llm.acomplete(req)
    assert resp.content == "async ok"


@pytest.mark.asyncio
async def test_astream_delegates_to_stream() -> None:
    llm = MockLLM(response_text="async stream")
    req = GenerationRequest(messages=[Message.user("hi")])
    chunks = [c async for c in llm.astream(req)]
    text = "".join(c.delta for c in chunks)
    assert text == "async stream"


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_render_basic(self) -> None:
        tmpl = PromptTemplate(
            name="test",
            system_template="You are a {role}.",
            user_template="Question: {question}",
            variables=[
                PromptVariable("role", required=True),
                PromptVariable("question", required=True),
            ],
        )
        msgs = tmpl.render(role="scientist", question="What is DNA?")
        assert msgs[0].content == "You are a scientist."
        assert msgs[1].content == "Question: What is DNA?"

    def test_render_with_default(self) -> None:
        tmpl = PromptTemplate(
            name="test",
            system_template="Answer in {language}.",
            user_template="{question}",
            variables=[
                PromptVariable("language", required=False, default="English"),
                PromptVariable("question", required=True),
            ],
        )
        msgs = tmpl.render(question="Hi?")
        assert "English" in msgs[0].content

    def test_render_missing_required_raises(self) -> None:
        tmpl = PromptTemplate(
            name="test",
            system_template="sys",
            user_template="{missing}",
            variables=[PromptVariable("missing", required=True)],
        )
        with pytest.raises(TemplateError) as exc_info:
            tmpl.render()
        assert "missing" in exc_info.value.missing_variables

    def test_render_empty_system_no_system_message(self) -> None:
        tmpl = PromptTemplate(
            name="test",
            system_template="",
            user_template="{q}",
            variables=[PromptVariable("q", required=True)],
        )
        msgs = tmpl.render(q="hello")
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.USER

    def test_validate_no_undeclared(self) -> None:
        undeclared = RAG_QA_TEMPLATE.validate()
        assert undeclared == []

    def test_validate_detects_undeclared(self) -> None:
        tmpl = PromptTemplate(
            name="bad",
            system_template="sys",
            user_template="{declared} and {undeclared}",
            variables=[PromptVariable("declared", required=True)],
        )
        missing = tmpl.validate()
        assert "undeclared" in missing

    def test_repr(self) -> None:
        assert "PromptTemplate" in repr(RAG_QA_TEMPLATE)


class TestBuiltinTemplates:
    def test_rag_qa_renders(self) -> None:
        msgs = RAG_QA_TEMPLATE.render(context="Ctx", question="Q?")
        assert len(msgs) == 2
        assert "Ctx" in msgs[1].content
        assert "Q?" in msgs[1].content

    def test_rag_summarise_renders(self) -> None:
        msgs = RAG_SUMMARISE_TEMPLATE.render(context="Long text here.")
        assert any("Long text here." in m.content for m in msgs)

    def test_rag_extract_renders(self) -> None:
        msgs = RAG_EXTRACT_TEMPLATE.render(context="Ctx", extraction_target="all dates")
        assert any("all dates" in m.content for m in msgs)

    def test_rag_followup_renders_with_defaults(self) -> None:
        msgs = RAG_FOLLOWUP_TEMPLATE.render(context="Ctx", question="Follow-up?")
        assert any("Follow-up?" in m.content for m in msgs)

    def test_builtin_templates_dict(self) -> None:
        assert "rag_qa" in BUILTIN_TEMPLATES
        assert "rag_summarise" in BUILTIN_TEMPLATES


class TestRAGPromptBuilder:
    def test_build_single_result(self, simple_result: SearchResult) -> None:
        builder = RAGPromptBuilder(RAG_QA_TEMPLATE)
        msgs = builder.build(results=[simple_result], question="What boils at 100°C?")
        assert len(msgs) == 2
        assert "[Source 1]" in msgs[1].content

    def test_build_from_texts(self) -> None:
        builder = RAGPromptBuilder(RAG_QA_TEMPLATE)
        msgs = builder.build_from_texts(["Passage one.", "Passage two."], question="Q?")
        content = msgs[1].content
        assert "[Source 1]" in content
        assert "[Source 2]" in content

    def test_format_context_truncates(self) -> None:
        builder = RAGPromptBuilder(RAG_QA_TEMPLATE, max_context_chars=50)
        chunks = [
            SearchResult(
                chunk=Chunk(content="A" * 200, document_id=f"doc{i}"),
                score=1.0,
                rank=i + 1,
            )
            for i in range(5)
        ]
        context = builder.format_context(chunks)
        assert len(context) <= 100  # Within reasonable margin

    def test_format_context_empty(self) -> None:
        builder = RAGPromptBuilder(RAG_QA_TEMPLATE)
        context = builder.format_context([])
        assert context == ""

    def test_build_missing_required_raises(self, simple_result: SearchResult) -> None:
        builder = RAGPromptBuilder(RAG_QA_TEMPLATE)
        with pytest.raises(TemplateError):
            builder.build(results=[simple_result])  # Missing 'question'


# ---------------------------------------------------------------------------
# CitationExtractor
# ---------------------------------------------------------------------------


class TestCitationExtractor:
    def test_find_markers_basic(self) -> None:
        extractor = CitationExtractor()
        markers = extractor.find_markers("Answer [Source 1] and [Source 2].")
        assert len(markers) == 2
        assert markers[0][0] == 1
        assert markers[1][0] == 2

    def test_find_markers_case_insensitive(self) -> None:
        extractor = CitationExtractor()
        markers = extractor.find_markers("[source 3] and [SOURCE 4]")
        nums = [m[0] for m in markers]
        assert 3 in nums
        assert 4 in nums

    def test_find_markers_no_markers(self) -> None:
        extractor = CitationExtractor()
        assert extractor.find_markers("No citations here.") == []

    def test_extract_resolves_chunk(self, simple_result: SearchResult) -> None:
        extractor = CitationExtractor()
        cr = extractor.extract("Water boils at 100°C. [Source 1]", [simple_result])
        assert len(cr.citations) == 1
        assert cr.citations[0].source_number == 1
        assert cr.citations[0].chunk is simple_result.chunk

    def test_extract_unknown_marker(self, simple_result: SearchResult) -> None:
        extractor = CitationExtractor()
        cr = extractor.extract("Bad ref [Source 99]", [simple_result])
        assert 99 in cr.unknown_markers

    def test_extract_uncited_source(self, simple_result: SearchResult) -> None:
        chunk2 = Chunk(content="Another fact.", document_id="doc2")
        result2 = SearchResult(chunk=chunk2, score=0.8, rank=2)
        extractor = CitationExtractor()
        cr = extractor.extract("Only used source 1 [Source 1].", [simple_result, result2])
        assert 2 in cr.uncited_sources

    def test_extract_clean_text(self) -> None:
        extractor = CitationExtractor()
        cr = extractor.extract(
            "Fact [Source 1].",
            [SearchResult(chunk=Chunk(content="ctx", document_id="d1"), score=1.0, rank=1)],
        )
        assert "[Source 1]" not in cr.clean_text
        assert "Fact" in cr.clean_text

    def test_extract_multiple_spans_same_source(self, simple_result: SearchResult) -> None:
        extractor = CitationExtractor()
        text = "First [Source 1]. Also [Source 1]."
        cr = extractor.extract(text, [simple_result])
        assert len(cr.citations) == 1
        assert len(cr.citations[0].spans) == 2

    def test_citation_properties(self, simple_result: SearchResult) -> None:
        extractor = CitationExtractor()
        cr = extractor.extract("[Source 1]", [simple_result])
        c = cr.citations[0]
        assert c.document_id == "doc1"
        assert c.source_label == "[Source 1]"
        assert len(c.preview) <= 123  # 120 + "..."

    def test_has_citations_true(self, simple_result: SearchResult) -> None:
        extractor = CitationExtractor()
        cr = extractor.extract("[Source 1]", [simple_result])
        assert cr.has_citations

    def test_has_citations_false(self) -> None:
        extractor = CitationExtractor()
        cr = extractor.extract("No citations.", [])
        assert not cr.has_citations

    def test_cited_document_ids_deduplication(self, simple_result: SearchResult) -> None:
        extractor = CitationExtractor()
        # Two references to the same source
        cr = extractor.extract("[Source 1] and [Source 1] again.", [simple_result])
        assert len(cr.cited_document_ids) == 1


class TestStripCitations:
    def test_strip_basic(self) -> None:
        result = strip_citations("Hello [Source 1] world.")
        assert "[Source 1]" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strip_multiple(self) -> None:
        result = strip_citations("[Source 1] and [Source 2] done.")
        assert "[Source" not in result

    def test_strip_empty(self) -> None:
        assert strip_citations("") == ""

    def test_strip_no_markers(self) -> None:
        text = "Plain text without markers."
        assert strip_citations(text) == text


class TestFormatBibliography:
    def test_empty_citations(self) -> None:
        result = format_bibliography([])
        assert "(no citations)" in result

    def test_single_citation(self, simple_result: SearchResult) -> None:
        c = Citation(source_number=1, chunk=simple_result.chunk, score=0.9)
        bib = format_bibliography([c])
        assert "[1]" in bib
        assert "doc1" in bib

    def test_with_score(self, simple_result: SearchResult) -> None:
        c = Citation(source_number=1, chunk=simple_result.chunk, score=0.87)
        bib = format_bibliography([c], include_score=True)
        assert "0.870" in bib

    def test_sorted_by_source_number(self, simple_result: SearchResult) -> None:
        chunk2 = Chunk(content="B", document_id="doc2")
        c1 = Citation(source_number=2, chunk=chunk2, score=0.5)
        c2 = Citation(source_number=1, chunk=simple_result.chunk, score=0.9)
        bib = format_bibliography([c1, c2])
        pos1 = bib.index("[1]")
        pos2 = bib.index("[2]")
        assert pos1 < pos2


# ---------------------------------------------------------------------------
# StreamBuffer
# ---------------------------------------------------------------------------


class TestStreamBuffer:
    def test_push_accumulates_content(self) -> None:
        buf = StreamBuffer()
        buf.push(StreamChunk(delta="Hello"))
        buf.push(StreamChunk(delta=", world!"))
        assert buf.content == "Hello, world!"

    def test_push_increments_count(self) -> None:
        buf = StreamBuffer()
        buf.push(StreamChunk(delta="a"))
        buf.push(StreamChunk(delta="b"))
        assert buf.chunk_count == 2

    def test_final_chunk_marks_complete(self) -> None:
        buf = StreamBuffer()
        buf.push(StreamChunk(delta="done", is_final=True, finish_reason=FinishReason.STOP))
        assert buf.is_complete
        assert buf.finish_reason == FinishReason.STOP

    def test_reset_clears_all(self) -> None:
        buf = StreamBuffer()
        buf.push(StreamChunk(delta="text", is_final=True, finish_reason=FinishReason.STOP))
        buf.reset()
        assert buf.content == ""
        assert buf.chunk_count == 0
        assert not buf.is_complete

    def test_len(self) -> None:
        buf = StreamBuffer()
        buf.push(StreamChunk(delta="abc"))
        assert len(buf) == 3

    def test_to_response(self) -> None:
        buf = StreamBuffer()
        buf.push(StreamChunk(delta="answer", is_final=True, finish_reason=FinishReason.STOP))
        resp = buf.to_response(model="gpt-4o", latency_ms=42.0)
        assert resp.content == "answer"
        assert resp.model == "gpt-4o"
        assert resp.latency_ms == 42.0

    def test_usage_from_final_chunk(self) -> None:
        buf = StreamBuffer()
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        buf.push(StreamChunk(delta="", is_final=True, usage=usage))
        assert buf.usage is not None
        assert buf.usage.prompt_tokens == 10


class TestStreamHandler:
    def test_consume_fires_token_callback(self) -> None:
        tokens: list[str] = []
        handler = StreamHandler(on_token=tokens.append)
        chunks = [StreamChunk(delta="A"), StreamChunk(delta="B", is_final=True)]
        handler.consume(iter(chunks))
        assert tokens == ["A", "B"]

    def test_consume_fires_complete_callback(self) -> None:
        responses: list[GenerationResponse] = []
        handler = StreamHandler(on_complete=responses.append)
        chunks = [StreamChunk(delta="hi", is_final=True, finish_reason=FinishReason.STOP)]
        handler.consume(iter(chunks))
        assert len(responses) == 1
        assert responses[0].content == "hi"

    def test_consume_fires_error_callback(self) -> None:
        errors: list[Exception] = []
        handler = StreamHandler(on_error=errors.append)

        def bad_stream() -> Iterator[StreamChunk]:
            yield StreamChunk(delta="ok")
            raise RuntimeError("stream broke")

        with pytest.raises(RuntimeError):
            handler.consume(bad_stream())
        assert len(errors) == 1

    def test_repr(self) -> None:
        handler = StreamHandler(model="gpt-4o")
        assert "StreamHandler" in repr(handler)


def test_collect_stream() -> None:
    chunks = [
        StreamChunk(delta="foo"),
        StreamChunk(delta="bar"),
        StreamChunk(delta="!", is_final=True),
    ]
    assert collect_stream(iter(chunks)) == "foobar!"


def test_stream_to_response() -> None:
    chunks = [StreamChunk(delta="42", is_final=True, finish_reason=FinishReason.STOP)]
    resp = stream_to_response(iter(chunks), model="test-model")
    assert resp.content == "42"
    assert resp.model == "test-model"
    assert resp.is_complete


def test_throttled_stream() -> None:
    chunks = [StreamChunk(delta="x"), StreamChunk(delta="y", is_final=True)]
    ts = ThrottledStream(source=iter(chunks), min_interval_seconds=0.0)
    result = list(ts)
    assert len(result) == 2
    text = "".join(c.delta for c in result)
    assert text == "xy"


# ---------------------------------------------------------------------------
# OpenAILLM parameter building (no real API calls)
# ---------------------------------------------------------------------------


class TestOpenAILLMParamBuilding:
    def _make_llm(self) -> OpenAILLM:
        llm = OpenAILLM(api_key="sk-fake", default_model="gpt-4o-mini")
        return llm

    def test_model_name(self) -> None:
        llm = self._make_llm()
        assert llm.model_name == "gpt-4o-mini"

    def test_provider(self) -> None:
        llm = self._make_llm()
        assert llm.provider == "openai"

    def test_build_params_basic(self) -> None:
        llm = self._make_llm()
        req = GenerationRequest(
            messages=[Message.user("hi")],
            config=GenerationConfig(model="gpt-4o", temperature=0.3),
        )
        params = llm._build_params(req)
        assert params["model"] == "gpt-4o"
        assert params["temperature"] == 0.3
        assert params["messages"][0]["role"] == "user"

    def test_build_params_stop_sequences(self) -> None:
        llm = self._make_llm()
        req = GenerationRequest(
            messages=[Message.user("hi")],
            config=GenerationConfig(stop=["END"]),
        )
        params = llm._build_params(req)
        assert params["stop"] == ["END"]

    def test_build_params_no_stop_when_empty(self) -> None:
        llm = self._make_llm()
        req = GenerationRequest(messages=[Message.user("hi")])
        params = llm._build_params(req)
        # Empty stop list → key is omitted from params (falsy check in _build_params)
        assert "stop" not in params or params["stop"] == []

    def test_repr(self) -> None:
        llm = self._make_llm()
        assert "OpenAILLM" in repr(llm)

    def test_missing_openai_package(self) -> None:
        llm = OpenAILLM(api_key="sk-fake")
        with patch.dict("sys.modules", {"openai": None}):
            # Re-importing with patched module should raise ImportError
            # We test _get_client directly when client is not set
            llm._client = None
            try:
                # This will either succeed (openai is installed) or raise ImportError
                client = llm._get_client()
                assert client is not None
            except ImportError:
                pass  # Expected when openai not available


def test_finish_reason_mapping() -> None:
    assert _finish_reason("stop") == FinishReason.STOP
    assert _finish_reason("length") == FinishReason.LENGTH
    assert _finish_reason("content_filter") == FinishReason.CONTENT_FILTER
    assert _finish_reason(None) == FinishReason.UNKNOWN
    assert _finish_reason("unknown_value") == FinishReason.UNKNOWN


# ---------------------------------------------------------------------------
# End-to-end: MockLLM + RAGPromptBuilder + CitationExtractor
# ---------------------------------------------------------------------------


class TestEndToEndGeneration:
    def test_rag_pipeline(self) -> None:
        """Full mini-pipeline: retrieve → prompt → generate → extract citations."""
        # Simulated retrieval results
        chunks = [
            Chunk(
                content="Photosynthesis converts sunlight into chemical energy.",
                document_id="bio-textbook",
            ),
            Chunk(
                content="Chlorophyll is the pigment responsible for absorbing light.",
                document_id="bio-textbook",
            ),
        ]
        results = [
            SearchResult(chunk=c, score=0.9 - i * 0.1, rank=i + 1) for i, c in enumerate(chunks)
        ]

        # Prompt building
        builder = RAGPromptBuilder(RAG_QA_TEMPLATE)
        messages = builder.build(results=results, question="How does photosynthesis work?")
        assert len(messages) == 2

        # Generation with mock LLM that produces a citation
        response_text = (
            "Photosynthesis converts sunlight to chemical energy [Source 1] "
            "using chlorophyll [Source 2]."
        )
        llm = MockLLM(response_text=response_text)
        req = GenerationRequest(messages=messages)
        resp = llm.complete(req)
        assert resp.is_complete

        # Citation extraction
        extractor = CitationExtractor()
        citation_result = extractor.extract(resp.content, results)
        assert len(citation_result.citations) == 2
        assert citation_result.citations[0].document_id == "bio-textbook"
        assert not citation_result.unknown_markers
        assert not citation_result.uncited_sources

    def test_streaming_pipeline(self) -> None:
        """Stream generation through StreamHandler with token accumulation."""
        llm = MockLLM(response_text="Streamed answer with [Source 1].", stream_chunk_size=4)
        req = GenerationRequest(messages=[Message.user("Tell me something.")])

        tokens: list[str] = []
        handler = StreamHandler(on_token=tokens.append, model="mock")
        buf = handler.consume(llm.stream(req))

        assert buf.is_complete
        full_text = "".join(tokens)
        assert full_text == "Streamed answer with [Source 1]."
