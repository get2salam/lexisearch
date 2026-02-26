"""LexiSearch generation layer.

This sub-package provides LLM integration, prompt templating, citation
extraction, and streaming utilities for the RAG generation step.

Quick start::

    from lexisearch.generation import (
        MockLLM,
        RAGPromptBuilder,
        CitationExtractor,
        GenerationRequest,
        Message,
        stream_to_response,
    )

    llm = MockLLM(response_text="The answer is 42. [Source 1]")
    builder = RAGPromptBuilder()
    extractor = CitationExtractor()

Public API
----------
LLM base & errors
~~~~~~~~~~~~~~~~~
.. autosummary::
   :nosignatures:

   BaseLLM
   LLMError
   GenerationRequest
   GenerationResponse
   GenerationConfig
   Message
   MessageRole
   FinishReason
   StreamChunk
   TokenUsage

Concrete LLM adapters
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :nosignatures:

   MockLLM
   OpenAILLM

Prompt templates
~~~~~~~~~~~~~~~~
.. autosummary::
   :nosignatures:

   PromptTemplate
   PromptVariable
   RAGPromptBuilder
   RAG_QA_TEMPLATE
   RAG_SUMMARISE_TEMPLATE
   RAG_EXTRACT_TEMPLATE
   RAG_FOLLOWUP_TEMPLATE
   BUILTIN_TEMPLATES

Citations
~~~~~~~~~
.. autosummary::
   :nosignatures:

   Citation
   CitationResult
   CitationExtractor
   strip_citations
   format_bibliography

Streaming
~~~~~~~~~
.. autosummary::
   :nosignatures:

   StreamBuffer
   StreamHandler
   collect_stream
   stream_to_response
   ThrottledStream
"""

from lexisearch.generation.base import (
    BaseLLM,
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
    CitationResult,
    format_bibliography,
    strip_citations,
)
from lexisearch.generation.mock_llm import MockLLM, make_config
from lexisearch.generation.openai_llm import OpenAILLM
from lexisearch.generation.prompts import (
    BUILTIN_TEMPLATES,
    RAG_EXTRACT_TEMPLATE,
    RAG_FOLLOWUP_TEMPLATE,
    RAG_QA_TEMPLATE,
    RAG_SUMMARISE_TEMPLATE,
    PromptStyle,
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

__all__ = [
    # Prompts
    "BUILTIN_TEMPLATES",
    "RAG_EXTRACT_TEMPLATE",
    "RAG_FOLLOWUP_TEMPLATE",
    "RAG_QA_TEMPLATE",
    "RAG_SUMMARISE_TEMPLATE",
    # Base
    "BaseLLM",
    # Citations
    "Citation",
    "CitationExtractor",
    "CitationResult",
    "FinishReason",
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResponse",
    "LLMError",
    "Message",
    "MessageRole",
    # Adapters
    "MockLLM",
    "OpenAILLM",
    "PromptStyle",
    "PromptTemplate",
    "PromptVariable",
    "RAGPromptBuilder",
    # Streaming
    "StreamBuffer",
    "StreamChunk",
    "StreamHandler",
    "TemplateError",
    "ThrottledStream",
    "TokenUsage",
    "collect_stream",
    "format_bibliography",
    "make_config",
    "stream_to_response",
    "strip_citations",
]
