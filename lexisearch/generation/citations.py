"""Citation extraction and attribution for RAG-generated responses.

After an LLM generates an answer that references ``[Source N]`` markers,
this module parses those markers, maps them back to the original retrieved
chunks, and produces structured :class:`Citation` objects for downstream
rendering (footnotes, hover-cards, audit trails, etc.).

Components
----------
* :class:`Citation` — a single source attribution with span info.
* :class:`CitationResult` — the annotated response with all citations.
* :class:`CitationExtractor` — parses and resolves ``[Source N]`` markers.
* :func:`strip_citations` — removes markers from text for clean display.
* :func:`format_bibliography` — renders a numbered reference list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lexisearch.models import Chunk, SearchResult


@dataclass
class Citation:
    """A single source citation extracted from generated text.

    Attributes:
        source_number: The ``N`` in ``[Source N]`` (1-indexed).
        chunk: The :class:`Chunk` the citation points to.
        score: Retrieval relevance score of the cited chunk.
        spans: Character spans ``(start, end)`` in the response text where
            this citation marker appears.
        metadata: Extra data (e.g., page number, section heading).
    """

    source_number: int
    chunk: Chunk
    score: float = 0.0
    spans: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        """The document ID of the cited chunk.

        Returns:
            The ``document_id`` of the underlying chunk.
        """
        return self.chunk.document_id

    @property
    def source_label(self) -> str:
        """The ``[Source N]`` label string.

        Returns:
            Formatted label, e.g. ``"[Source 3]"``.
        """
        return f"[Source {self.source_number}]"

    @property
    def preview(self) -> str:
        """First 120 characters of the cited passage.

        Returns:
            Truncated content preview.
        """
        content = self.chunk.content.strip()
        return content[:120] + "..." if len(content) > 120 else content

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"Citation(source={self.source_number}, "
            f"doc={self.document_id!r}, score={self.score:.4f})"
        )


@dataclass
class CitationResult:
    """The result of citation extraction from a generated response.

    Attributes:
        response_text: The original generated text (markers intact).
        clean_text: The text with citation markers stripped.
        citations: Resolved citation objects in source-number order.
        uncited_sources: Source numbers that were provided but not cited.
        unknown_markers: Marker numbers that appear in text but have no
            matching source (e.g., ``[Source 99]`` with only 3 sources).
    """

    response_text: str
    clean_text: str
    citations: list[Citation] = field(default_factory=list)
    uncited_sources: list[int] = field(default_factory=list)
    unknown_markers: list[int] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        """True if the response contains at least one resolved citation.

        Returns:
            Whether any citations were found and resolved.
        """
        return bool(self.citations)

    @property
    def cited_document_ids(self) -> list[str]:
        """Unique document IDs referenced in citations.

        Returns:
            Deduplicated list of document IDs, in citation order.
        """
        seen: set[str] = set()
        ids: list[str] = []
        for c in self.citations:
            if c.document_id not in seen:
                seen.add(c.document_id)
                ids.append(c.document_id)
        return ids

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"CitationResult(citations={len(self.citations)}, "
            f"unknown={len(self.unknown_markers)}, "
            f"uncited={len(self.uncited_sources)})"
        )


class CitationExtractor:
    """Parses ``[Source N]`` markers and resolves them to chunks.

    The extractor finds all occurrences of ``[Source N]`` (case-insensitive,
    flexible whitespace) in the response text and maps each unique ``N`` to
    the corresponding :class:`SearchResult` in the provided results list.

    Args:
        marker_pattern: Regex pattern for citation markers.  Must contain
            a named group ``num`` capturing the source number.

    Examples:
        >>> from lexisearch.models import SearchResult, Chunk
        >>> extractor = CitationExtractor()
        >>> chunk = Chunk(content="Water boils at 100°C at sea level.", document_id="doc1")
        >>> results = [SearchResult(chunk=chunk, score=0.95, rank=1)]
        >>> cr = extractor.extract("Water boils at 100°C [Source 1].", results)
        >>> cr.citations[0].source_number
        1
        >>> cr.has_citations
        True
    """

    #: Default pattern matching ``[Source N]`` with flexible whitespace.
    DEFAULT_PATTERN = re.compile(
        r"\[(?:Source|source|SOURCE)\s*(?P<num>\d+)\]",
        re.IGNORECASE,
    )

    def __init__(self, marker_pattern: re.Pattern[str] | None = None) -> None:
        """Initialise the extractor with an optional custom pattern.

        Args:
            marker_pattern: Custom compiled regex.  Must include a named
                group ``num`` capturing the source number.  Defaults to
                :attr:`DEFAULT_PATTERN`.
        """
        self._pattern = marker_pattern or self.DEFAULT_PATTERN

    def find_markers(self, text: str) -> list[tuple[int, int, int]]:
        """Find all citation markers in *text*.

        Args:
            text: The generated response text.

        Returns:
            List of ``(source_number, start_char, end_char)`` tuples,
            one per marker occurrence (duplicates included).
        """
        matches: list[tuple[int, int, int]] = []
        for m in self._pattern.finditer(text):
            num = int(m.group("num"))
            matches.append((num, m.start(), m.end()))
        return matches

    def extract(self, response_text: str, results: list[SearchResult]) -> CitationResult:
        """Extract and resolve citations from a generated response.

        Args:
            response_text: The generated text containing ``[Source N]`` markers.
            results: The ranked search results that were passed to the LLM as
                context.  ``results[0]`` corresponds to ``[Source 1]``, etc.

        Returns:
            A :class:`CitationResult` with resolved citations and metadata.
        """
        markers = self.find_markers(response_text)

        # Group spans by source number
        spans_by_num: dict[int, list[tuple[int, int]]] = {}
        for num, start, end in markers:
            spans_by_num.setdefault(num, []).append((start, end))

        cited_nums = set(spans_by_num.keys())
        provided_nums = set(range(1, len(results) + 1))

        citations: list[Citation] = []
        unknown_markers: list[int] = []

        for num in sorted(cited_nums):
            if 1 <= num <= len(results):
                sr = results[num - 1]
                citations.append(
                    Citation(
                        source_number=num,
                        chunk=sr.chunk,
                        score=sr.score,
                        spans=spans_by_num[num],
                    )
                )
            else:
                unknown_markers.append(num)

        uncited_sources = sorted(provided_nums - cited_nums)
        clean_text = strip_citations(response_text, self._pattern)

        return CitationResult(
            response_text=response_text,
            clean_text=clean_text,
            citations=citations,
            uncited_sources=uncited_sources,
            unknown_markers=unknown_markers,
        )

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"CitationExtractor(pattern={self._pattern.pattern!r})"


def strip_citations(text: str, pattern: re.Pattern[str] | None = None) -> str:
    """Remove all ``[Source N]`` markers from text.

    Args:
        text: Input text with citation markers.
        pattern: Custom regex pattern.  Defaults to
            :attr:`CitationExtractor.DEFAULT_PATTERN`.

    Returns:
        Text with markers removed and redundant whitespace collapsed.

    Examples:
        >>> strip_citations("The sky is blue [Source 1].")
        'The sky is blue.'
    """
    _pattern = pattern or CitationExtractor.DEFAULT_PATTERN
    cleaned = _pattern.sub("", text)
    # Collapse multiple spaces while preserving newlines
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def format_bibliography(
    citations: list[Citation],
    include_preview: bool = True,
    include_score: bool = False,
) -> str:
    """Render a numbered bibliography from citation objects.

    Args:
        citations: List of :class:`Citation` objects to format.
        include_preview: Whether to append a text preview for each source.
        include_score: Whether to append the retrieval score.

    Returns:
        A multi-line string bibliography.

    Examples:
        >>> # With two citations, produces:
        >>> # [1] doc_id: "The quick brown fox..."  (score: 0.92)
        >>> # [2] other_doc: "Lazy dogs are..."
    """
    if not citations:
        return "(no citations)"

    lines: list[str] = []
    for c in sorted(citations, key=lambda x: x.source_number):
        parts = [f"[{c.source_number}] {c.document_id}"]
        if include_preview:
            parts.append(f'  "{c.preview}"')
        if include_score:
            parts.append(f"  (relevance: {c.score:.3f})")
        lines.append("\n".join(parts))

    return "\n\n".join(lines)
