"""Prompt template system for the LexiSearch RAG pipeline.

Provides a composable, type-safe prompt template framework tailored for
retrieval-augmented generation.  Templates are rendered by substituting
named variables, with optional Jinja2-style conditionals for advanced use.

Key components
--------------
* :class:`PromptVariable` — declares a named slot with type and default.
* :class:`PromptTemplate` — a re-usable template with variable rendering.
* :class:`RAGPromptBuilder` — high-level builder for standard RAG prompts.
* Built-in templates for QA, summarisation, and extraction tasks.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lexisearch.generation.base import Message, MessageRole
from lexisearch.models import SearchResult


class TemplateError(ValueError):
    """Raised when a prompt template cannot be rendered.

    Attributes:
        template_name: Name of the template that failed.
        missing_variables: Variable names that were not supplied.
    """

    def __init__(
        self,
        message: str,
        template_name: str = "",
        missing_variables: list[str] | None = None,
    ) -> None:
        """Initialise the template error.

        Args:
            message: Human-readable error description.
            template_name: Name of the template that failed.
            missing_variables: Names of variables not supplied.
        """
        super().__init__(message)
        self.template_name = template_name
        self.missing_variables = missing_variables or []


class PromptStyle(str, Enum):
    """High-level style that controls default wording of system prompts.

    Attributes:
        PRECISE: Strict factual answers, no speculation.
        CONVERSATIONAL: Friendly, approachable tone.
        ACADEMIC: Formal, citation-heavy style.
        CONCISE: Short bullet-point answers.
    """

    PRECISE = "precise"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    CONCISE = "concise"


@dataclass
class PromptVariable:
    """Declares a named variable slot in a prompt template.

    Attributes:
        name: The variable placeholder name (without braces).
        description: Human-readable description for documentation.
        required: If True, rendering fails when the variable is absent.
        default: Value used when the variable is absent and not required.
    """

    name: str
    description: str = ""
    required: bool = True
    default: str = ""

    @property
    def placeholder(self) -> str:
        """The ``{name}`` placeholder string as it appears in templates.

        Returns:
            Brace-wrapped placeholder, e.g. ``"{context}"``.
        """
        return f"{{{self.name}}}"


@dataclass
class PromptTemplate:
    r"""A re-usable prompt template with named variable substitution.

    Templates use ``{variable_name}`` placeholders.  Variable definitions
    control which are required and what defaults to apply.

    Args:
        name: Unique template identifier.
        system_template: System message template string.
        user_template: User message template string.
        variables: Declared variables for validation and documentation.
        description: Human-readable description of the template's purpose.

    Examples:
        >>> tmpl = PromptTemplate(
        ...     name="qa",
        ...     system_template="You are a helpful assistant.",
        ...     user_template="Context: {context}\\n\\nQuestion: {question}",
        ...     variables=[
        ...         PromptVariable("context", required=True),
        ...         PromptVariable("question", required=True),
        ...     ],
        ... )
        >>> msgs = tmpl.render(context="Cats are mammals.", question="What are cats?")
        >>> msgs[1].content
        'Context: Cats are mammals.\\n\\nQuestion: What are cats?'
    """

    name: str
    system_template: str
    user_template: str
    variables: list[PromptVariable] = field(default_factory=list)
    description: str = ""

    # Regex matching {variable_name} placeholders
    _PLACEHOLDER_RE: re.Pattern[str] = re.compile(r"\{(\w+)\}")

    def _variable_map(self) -> dict[str, PromptVariable]:
        """Build a lookup from variable name → PromptVariable.

        Returns:
            Dict mapping name strings to their variable definitions.
        """
        return {v.name: v for v in self.variables}

    def render(self, **kwargs: Any) -> list[Message]:
        """Render the template with the given variable values.

        Args:
            **kwargs: Variable values keyed by placeholder name.

        Returns:
            A list of :class:`Message` objects ready for an LLM request.

        Raises:
            TemplateError: If a required variable is missing.
        """
        self._variable_map()
        values: dict[str, str] = {}
        missing: list[str] = []

        for var in self.variables:
            if var.name in kwargs:
                values[var.name] = str(kwargs[var.name])
            elif var.required:
                missing.append(var.name)
            else:
                values[var.name] = var.default

        if missing:
            raise TemplateError(
                f"Template '{self.name}' is missing required variables: {missing}",
                template_name=self.name,
                missing_variables=missing,
            )

        # Also pass through any extra kwargs not declared as variables
        for k, v in kwargs.items():
            if k not in values:
                values[k] = str(v)

        system_content = self.system_template.format(**values)
        user_content = self.user_template.format(**values)

        messages: list[Message] = []
        if system_content.strip():
            messages.append(Message(role=MessageRole.SYSTEM, content=system_content))
        messages.append(Message(role=MessageRole.USER, content=user_content))
        return messages

    def validate(self) -> list[str]:
        """Check that all placeholders in templates have variable definitions.

        Returns:
            List of undeclared placeholder names, empty if template is valid.
        """
        declared = {v.name for v in self.variables}
        system_vars = set(self._PLACEHOLDER_RE.findall(self.system_template))
        user_vars = set(self._PLACEHOLDER_RE.findall(self.user_template))
        all_vars = system_vars | user_vars
        return sorted(all_vars - declared)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"PromptTemplate(name={self.name!r}, variables={[v.name for v in self.variables]})"


# ---------------------------------------------------------------------------
# Built-in RAG templates
# ---------------------------------------------------------------------------

#: Standard question-answering template for RAG.
RAG_QA_TEMPLATE = PromptTemplate(
    name="rag_qa",
    description="Standard RAG question-answering with source context.",
    system_template=textwrap.dedent("""\
        You are a knowledgeable assistant that answers questions accurately and concisely.
        Answer ONLY based on the provided context. If the context does not contain enough
        information to answer the question, say "I don't have enough information to answer
        this based on the provided context." Do not speculate beyond what the context supports.
        Always cite the source(s) you used by referring to [Source N] markers.
    """).strip(),
    user_template=textwrap.dedent("""\
        Context:
        {context}

        Question: {question}

        Answer:
    """).strip(),
    variables=[
        PromptVariable("context", description="Retrieved context passages.", required=True),
        PromptVariable("question", description="User's question.", required=True),
    ],
)

#: Summarisation template for condensing retrieved passages.
RAG_SUMMARISE_TEMPLATE = PromptTemplate(
    name="rag_summarise",
    description="Summarise retrieved context passages into a concise answer.",
    system_template=textwrap.dedent("""\
        You are an expert summariser. Produce a concise, accurate summary of the provided
        passages. Preserve key facts, figures, and attributions. Output plain prose unless
        bullet points materially improve clarity.
    """).strip(),
    user_template=textwrap.dedent("""\
        Passages to summarise:
        {context}

        Topic or focus (optional): {topic}

        Summary:
    """).strip(),
    variables=[
        PromptVariable("context", description="Retrieved passages to summarise.", required=True),
        PromptVariable("topic", description="Optional focus topic.", required=False, default=""),
    ],
)

#: Extraction template for pulling structured information from context.
RAG_EXTRACT_TEMPLATE = PromptTemplate(
    name="rag_extract",
    description="Extract structured information from retrieved context.",
    system_template=textwrap.dedent("""\
        You are a precise information extraction engine. Extract only the information
        explicitly requested. Do not infer or add information not present in the context.
        Format your output as requested, defaulting to a numbered list if no format is specified.
    """).strip(),
    user_template=textwrap.dedent("""\
        Context:
        {context}

        Extract: {extraction_target}

        Output format: {output_format}

        Extracted information:
    """).strip(),
    variables=[
        PromptVariable("context", description="Source context.", required=True),
        PromptVariable(
            "extraction_target", description="What to extract (e.g., 'all dates').", required=True
        ),
        PromptVariable(
            "output_format",
            description="Desired output format.",
            required=False,
            default="numbered list",
        ),
    ],
)

#: Conversational follow-up template for multi-turn RAG.
RAG_FOLLOWUP_TEMPLATE = PromptTemplate(
    name="rag_followup",
    description="Multi-turn RAG conversation with history awareness.",
    system_template=textwrap.dedent("""\
        You are a helpful assistant engaged in a conversation. Use the retrieved context
        and conversation history to give accurate, coherent answers. Reference prior turns
        when relevant. Cite sources using [Source N] markers.
    """).strip(),
    user_template=textwrap.dedent("""\
        Conversation history:
        {history}

        Retrieved context:
        {context}

        Current question: {question}

        Answer:
    """).strip(),
    variables=[
        PromptVariable(
            "history",
            description="Prior conversation turns.",
            required=False,
            default="(no prior history)",
        ),
        PromptVariable("context", description="Retrieved context passages.", required=True),
        PromptVariable("question", description="Current user question.", required=True),
    ],
)


# Registry of built-in templates
BUILTIN_TEMPLATES: dict[str, PromptTemplate] = {
    tmpl.name: tmpl
    for tmpl in [
        RAG_QA_TEMPLATE,
        RAG_SUMMARISE_TEMPLATE,
        RAG_EXTRACT_TEMPLATE,
        RAG_FOLLOWUP_TEMPLATE,
    ]
}


class RAGPromptBuilder:
    """High-level builder that formats retrieved results into prompt messages.

    This class handles the common task of converting :class:`SearchResult`
    objects into a numbered context block and rendering it into a prompt.

    Args:
        template: The :class:`PromptTemplate` to use.
        max_context_chars: Maximum total characters for the context block.
        source_prefix: Prefix format for each context source label.

    Examples:
        >>> from lexisearch.models import SearchResult, Chunk
        >>> builder = RAGPromptBuilder(RAG_QA_TEMPLATE)
        >>> chunk = Chunk(content="The sky is blue.", document_id="doc1")
        >>> result = SearchResult(chunk=chunk, score=0.9, rank=1)
        >>> msgs = builder.build(results=[result], question="What colour is the sky?")
        >>> len(msgs)
        2
    """

    def __init__(
        self,
        template: PromptTemplate = RAG_QA_TEMPLATE,
        max_context_chars: int = 8000,
        source_prefix: str = "[Source {n}]",
    ) -> None:
        """Initialise the RAG prompt builder.

        Args:
            template: The :class:`PromptTemplate` to render.
            max_context_chars: Maximum characters for the context block.
            source_prefix: Label prefix for each context source.
                Use ``{n}`` as the source-number placeholder.
        """
        self.template = template
        self.max_context_chars = max_context_chars
        self.source_prefix = source_prefix

    def format_context(self, results: list[SearchResult]) -> str:
        """Format a list of search results into a numbered context block.

        Results are truncated to stay within :attr:`max_context_chars`.

        Args:
            results: Ordered list of search results (most relevant first).

        Returns:
            A formatted multi-line string with labelled passages.
        """
        parts: list[str] = []
        total_chars = 0

        for i, result in enumerate(results, start=1):
            label = self.source_prefix.format(n=i)
            passage = result.chunk.content.strip()
            entry = f"{label}\n{passage}"

            if total_chars + len(entry) > self.max_context_chars:
                remaining = self.max_context_chars - total_chars
                if remaining > 100:  # Only add if there's meaningful space
                    entry = f"{label}\n{passage[: remaining - len(label) - 5]}..."
                    parts.append(entry)
                break

            parts.append(entry)
            total_chars += len(entry) + 2  # +2 for the blank line separator

        return "\n\n".join(parts)

    def build(self, results: list[SearchResult], **kwargs: Any) -> list[Message]:
        """Build prompt messages from search results and template variables.

        Args:
            results: Retrieved search results to use as context.
            **kwargs: Additional template variables (e.g., ``question``).

        Returns:
            A list of :class:`Message` objects ready for an LLM request.

        Raises:
            TemplateError: If required template variables are missing.
        """
        context = self.format_context(results)
        return self.template.render(context=context, **kwargs)

    def build_from_texts(self, texts: list[str], **kwargs: Any) -> list[Message]:
        """Build prompt messages from raw text passages.

        Convenience method when :class:`SearchResult` objects are not
        available — wraps each text in a minimal :class:`Chunk`.

        Args:
            texts: List of text passages to use as context.
            **kwargs: Additional template variables.

        Returns:
            A list of :class:`Message` objects.
        """
        from lexisearch.models import Chunk

        results = [
            SearchResult(
                chunk=Chunk(content=text, document_id=f"text_{i}"),
                score=1.0,
                rank=i + 1,
            )
            for i, text in enumerate(texts)
        ]
        return self.build(results=results, **kwargs)
