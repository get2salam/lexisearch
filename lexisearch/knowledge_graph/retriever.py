"""Graph-augmented retrieval using the knowledge graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lexisearch.knowledge_graph.extractor import EntityExtractor

if TYPE_CHECKING:
    from lexisearch.knowledge_graph.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGConfig:
    """Configuration for the GraphRAG retriever."""

    max_hops: int = 2
    """Maximum graph traversal depth from query entities."""

    max_graph_terms: int = 10
    """Maximum number of graph-expanded terms to add to the query."""

    base_top_k: int = 5
    """Number of results to fetch from the base retriever."""

    graph_weight: float = 0.3
    """Weight assigned to graph-expanded results (vs. vector results)."""

    include_relation_predicates: bool = True
    """Whether to include relation predicates in the expanded query context."""


@dataclass
class GraphRAGResult:
    """Result from a GraphRAG retrieval pass."""

    query: str
    expanded_query: str
    base_results: list[Any] = field(default_factory=list)
    graph_entities: list[str] = field(default_factory=list)
    graph_relations: list[str] = field(default_factory=list)
    combined_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphRAGRetriever:
    """Knowledge-graph-augmented retrieval.

    Enriches standard vector retrieval with structured graph traversal:

    1. **Entity extraction** — extract named entities from the query.
    2. **Graph expansion** — look up each entity in the knowledge graph and
       find neighbours within ``max_hops``.
    3. **Query augmentation** — append expanded terms to the query before
       vector retrieval.
    4. **Context enrichment** — include entity descriptions and relations in
       the generation context.

    Parameters
    ----------
    knowledge_graph:
        A pre-populated ``KnowledgeGraph`` instance.
    base_retriever:
        Callable ``(query: str, top_k: int) -> list[Any]`` — any retriever.
    extractor:
        Optional ``EntityExtractor`` for query entity extraction.
        Defaults to the rule-based ``EntityExtractor``.
    config:
        Configuration object.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        base_retriever: Any,
        extractor: EntityExtractor | None = None,
        config: GraphRAGConfig | None = None,
    ) -> None:
        """Initialise the GraphRAG retriever."""
        self.kg = knowledge_graph
        self.base_retriever = base_retriever
        self.extractor = extractor or EntityExtractor()
        self.config = config or GraphRAGConfig()

    def retrieve(self, query: str, *, top_k: int | None = None) -> GraphRAGResult:
        """Run a graph-augmented retrieval pass.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of results to return.

        Returns:
        -------
        GraphRAGResult
            Contains base results, graph entities, and expanded context.
        """
        k = top_k or self.config.base_top_k

        # Step 1: extract entities from the query
        query_entities, _ = self.extractor.extract(query)
        entity_names = [e.name for e in query_entities]
        logger.debug("GraphRAG query entities: %s", entity_names)

        # Step 2: graph traversal
        graph_entity_names: list[str] = []
        graph_relation_strings: list[str] = []
        for entity_name in entity_names:
            if self.kg.has_entity(entity_name):
                neighbours = self.kg.neighbours(entity_name, max_hops=self.config.max_hops)
                for n in neighbours[: self.config.max_graph_terms]:
                    if n.name not in graph_entity_names:
                        graph_entity_names.append(n.name)

                if self.config.include_relation_predicates:
                    relations = self.kg.get_relations(subject=entity_name)
                    for rel in relations[:5]:
                        graph_relation_strings.append(f"{rel.subject} {rel.predicate} {rel.obj}")

        # Step 3: build expanded query
        expansion_terms = graph_entity_names[: self.config.max_graph_terms]
        expanded_query = f"{query} {' '.join(expansion_terms)}" if expansion_terms else query

        # Step 4: call base retriever
        try:
            base_results = self.base_retriever(expanded_query, k)
        except Exception:
            logger.warning("Base retriever failed, falling back to original query")
            base_results = self.base_retriever(query, k)

        # Step 5: build combined context string
        context_parts: list[str] = []
        if graph_entity_names:
            context_parts.append(f"Related entities: {', '.join(graph_entity_names)}")
        if graph_relation_strings and self.config.include_relation_predicates:
            context_parts.append("Relations: " + "; ".join(graph_relation_strings[:5]))
        combined_context = "\n".join(context_parts)

        return GraphRAGResult(
            query=query,
            expanded_query=expanded_query,
            base_results=base_results,
            graph_entities=graph_entity_names,
            graph_relations=graph_relation_strings,
            combined_context=combined_context,
            metadata={
                "query_entities": entity_names,
                "num_graph_entities": len(graph_entity_names),
                "num_relations": len(graph_relation_strings),
            },
        )

    def index_document(self, text: str, *, doc_id: str = "") -> tuple[int, int]:
        """Extract entities/relations from *text* and add them to the graph.

        Parameters
        ----------
        text:
            Document text to index into the knowledge graph.
        doc_id:
            Document identifier attached to extracted items.

        Returns:
        -------
        tuple[int, int]
            ``(num_entities_added, num_relations_added)``
        """
        entities, relations = self.extractor.extract(text, doc_id=doc_id)
        before_e = self.kg.num_entities
        before_r = self.kg.num_relations
        self.kg.add_entities(entities)
        self.kg.add_relations(relations)
        added_e = self.kg.num_entities - before_e
        added_r = self.kg.num_relations - before_r
        logger.debug("Indexed doc %r: +%d entities, +%d relations", doc_id, added_e, added_r)
        return added_e, added_r
