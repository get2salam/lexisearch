"""Entity and relation extraction from text.

Provides a rule-based ``EntityExtractor`` that requires no external NLP
models.  For production use, swap in the spaCy or HuggingFace backend by
implementing ``BaseEntityExtractor``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from lexisearch.knowledge_graph.graph import Entity, Relation

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseEntityExtractor(ABC):
    """Abstract base for entity/relation extractors."""

    @abstractmethod
    def extract(
        self,
        text: str,
        *,
        doc_id: str = "",
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from *text*.

        Parameters
        ----------
        text:
            Input document or chunk text.
        doc_id:
            Optional document identifier to attach to extracted items.

        Returns:
            Tuple of (entities, relations).
        """


# ---------------------------------------------------------------------------
# Extraction result
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Structured result from an extraction pass."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    raw_spans: list[dict[str, Any]] = field(default_factory=list)
    """Raw span dicts for debugging: ``{text, start, end, type}``."""


# ---------------------------------------------------------------------------
# Pattern-based rule extractor
# ---------------------------------------------------------------------------


class EntityExtractor(BaseEntityExtractor):
    """Rule-based entity extractor using regex and heuristics.

    Recognises three broad entity classes without requiring an NLP model:
    - **TECHNOLOGY** — known tech terms, libraries, tools, and acronyms
    - **CONCEPT** — domain terminology extracted via capitalisation heuristics
    - **MEASURE** — numeric measurements with units

    Relations are inferred from co-occurrence patterns in the same sentence
    using a small set of predicate templates.

    Parameters
    ----------
    min_entity_length:
        Minimum character length for extracted entities.
    max_entity_length:
        Maximum character length (prevents matching entire sentences).
    extract_relations:
        If ``True`` (default), also extract co-occurrence relations.
    """

    # Known tech keywords (case-insensitive matching)
    _TECH_TERMS = frozenset(
        {
            "faiss",
            "chromadb",
            "qdrant",
            "weaviate",
            "pinecone",
            "openai",
            "huggingface",
            "langchain",
            "llamaindex",
            "transformer",
            "bert",
            "gpt",
            "llm",
            "rag",
            "vector",
            "embedding",
            "retrieval",
            "attention",
            "encoder",
            "decoder",
            "tokeniser",
            "tokenizer",
            "pytorch",
            "tensorflow",
            "jax",
            "numpy",
            "scipy",
            "elasticsearch",
            "opensearch",
            "solr",
            "lucene",
            "bm25",
            "tfidf",
            "cosine",
            "euclidean",
            "docker",
            "kubernetes",
            "fastapi",
            "flask",
            "django",
            "postgresql",
            "redis",
            "mongodb",
            "neo4j",
            "github",
            "git",
            "python",
            "rust",
            "golang",
        }
    )

    # Predicate templates: (subject_pattern, predicate, object_pattern)
    _RELATION_PATTERNS: ClassVar[list[tuple[str, str, str]]] = [
        (r"(\w+) is (?:a|an) (\w+)", "IS_A", ""),
        (r"(\w+) (?:is )?developed by (\w+)", "DEVELOPED_BY", ""),
        (r"(\w+) (?:is )?built on (\w+)", "BUILT_ON", ""),
        (r"(\w+) uses? (\w+)", "USES", ""),
        (r"(\w+) extends? (\w+)", "EXTENDS", ""),
        (r"(\w+) (?:is )?part of (\w+)", "PART_OF", ""),
        (r"(\w+) integrates? (?:with )?(\w+)", "INTEGRATES_WITH", ""),
        (r"(\w+) and (\w+) are (?:related|similar)", "RELATED_TO", ""),
    ]

    def __init__(
        self,
        min_entity_length: int = 2,
        max_entity_length: int = 50,
        extract_relations: bool = True,
    ) -> None:
        """Initialise the rule-based entity extractor."""
        self.min_entity_length = min_entity_length
        self.max_entity_length = max_entity_length
        self.extract_relations = extract_relations

        # Pre-compile relation patterns
        self._compiled_relations = [
            (re.compile(pat, re.IGNORECASE), pred) for pat, pred, _ in self._RELATION_PATTERNS
        ]

        # Acronym pattern: 2-8 uppercase letters optionally ending in 's'
        self._acronym_re = re.compile(r"\b([A-Z]{2,8}s?)\b")

        # Capitalised multi-word term: "Retrieval Augmented Generation"
        self._cap_term_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")

        # Numeric measure: "128 dimensions", "0.9 precision"
        self._measure_re = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*(dimensions?|layers?|heads?|tokens?|bytes?|ms|seconds?)\b",
            re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        *,
        doc_id: str = "",
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from *text*.

        Returns:
            Tuple of (entities, relations).
        """
        result = self.extract_full(text, doc_id=doc_id)
        return result.entities, result.relations

    def extract_full(self, text: str, *, doc_id: str = "") -> ExtractionResult:
        """Full extraction returning an ``ExtractionResult`` with raw spans."""
        entities: dict[str, Entity] = {}
        raw_spans: list[dict[str, Any]] = []
        doc_id_set = {doc_id} if doc_id else set()

        # --- Technology terms ---
        for match in re.finditer(r"\b(\w+)\b", text, re.IGNORECASE):
            word = match.group(1)
            if word.lower() in self._TECH_TERMS:
                self._add_entity(entities, word, "TECHNOLOGY", doc_id_set, raw_spans, match.start())

        # --- Acronyms ---
        for match in self._acronym_re.finditer(text):
            word = match.group(1)
            if self.min_entity_length <= len(word) <= self.max_entity_length:
                self._add_entity(entities, word, "ACRONYM", doc_id_set, raw_spans, match.start())

        # --- Capitalised multi-word terms ---
        for match in self._cap_term_re.finditer(text):
            term = match.group(1)
            if self.min_entity_length <= len(term) <= self.max_entity_length:
                # Skip if it's just the first word of a sentence
                if text[: match.start()].rstrip().endswith((".", "!", "?", "\n", "")):
                    continue
                self._add_entity(entities, term, "CONCEPT", doc_id_set, raw_spans, match.start())

        # --- Measurements ---
        for match in self._measure_re.finditer(text):
            term = match.group(0)
            self._add_entity(entities, term, "MEASURE", doc_id_set, raw_spans, match.start())

        # --- Relations ---
        relations: list[Relation] = []
        if self.extract_relations and len(entities) >= 2:
            relations = self._extract_relations(text, entities, doc_id_set)

        return ExtractionResult(
            entities=list(entities.values()),
            relations=relations,
            raw_spans=raw_spans,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_entity(
        store: dict[str, Entity],
        name: str,
        etype: str,
        doc_ids: set[str],
        raw_spans: list[dict[str, Any]],
        start: int,
    ) -> None:
        """Add or merge an entity into *store*."""
        key = name.lower()
        if key in store:
            store[key].doc_ids |= doc_ids
        else:
            store[key] = Entity(name=name, entity_type=etype, doc_ids=set(doc_ids))
        raw_spans.append({"text": name, "start": start, "type": etype})

    def _extract_relations(
        self,
        text: str,
        entities: dict[str, Entity],
        doc_ids: set[str],
    ) -> list[Relation]:
        """Extract relations using sentence-level pattern matching."""
        entity_names = set(entities)
        relations: list[Relation] = []

        # Split into sentences for local co-occurrence
        sentences = re.split(r"[.!?\n]+", text)
        for sentence in sentences:
            sent_lower = sentence.lower()
            # Check which entities appear in this sentence
            present = [n for n in entity_names if n in sent_lower]
            if len(present) < 2:
                continue

            # Try explicit predicate patterns
            for pattern, predicate in self._compiled_relations:
                for match in pattern.finditer(sentence):
                    subj_text = match.group(1).lower()
                    obj_text = match.group(2).lower()
                    # Only create relation if both are known entities
                    if subj_text in entity_names and obj_text in entity_names:
                        rel = Relation(
                            subject=entities[subj_text].name,
                            predicate=predicate,
                            obj=entities[obj_text].name,
                            weight=1.0,
                            doc_ids=set(doc_ids),
                        )
                        relations.append(rel)

            # Co-occurrence relation for all pairs in the same sentence
            for i, e1 in enumerate(present):
                for e2 in present[i + 1 :]:
                    relations.append(
                        Relation(
                            subject=entities[e1].name,
                            predicate="CO_OCCURS_WITH",
                            obj=entities[e2].name,
                            weight=0.5,
                            doc_ids=set(doc_ids),
                        )
                    )

        return relations
