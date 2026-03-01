"""LexiSearch Knowledge Graph module.

Provides an in-memory knowledge graph that augments dense retrieval with
structured entity/relation traversal.  Documents are parsed for named entities
and co-occurrence relations; at query time the graph is used to expand queries
and surface semantically related chunks that pure embedding search may miss.

Core concepts
-------------
``Entity``
    A named concept extracted from a document chunk (person, organisation,
    technology, concept, etc.).

``Relation``
    A directed edge between two entities: ``(subject, predicate, object)``.

``KnowledgeGraph``
    Container for entities and relations.  Supports addition, lookup, graph
    traversal (BFS/DFS), and serialisation.

``EntityExtractor``
    Extracts entities from text using rule-based patterns (no NLP model
    required by default; pluggable NLP backend supported).

``GraphRAGRetriever``
    Extends vanilla vector retrieval with graph-based query expansion.  For
    each retrieved chunk, its entities are looked up in the graph and
    neighbours are fetched to widen the context window.

Quick start::

    from lexisearch.knowledge_graph import KnowledgeGraph, EntityExtractor

    kg = KnowledgeGraph()
    extractor = EntityExtractor()

    text = \"\"\"FAISS is a library by Facebook AI Research for efficient
    similarity search and clustering of dense vectors.\"\"\"

    entities, relations = extractor.extract(text)
    kg.add_entities(entities)
    kg.add_relations(relations)

    neighbours = kg.neighbours("FAISS", max_hops=2)
"""

from __future__ import annotations

from lexisearch.knowledge_graph.extractor import EntityExtractor
from lexisearch.knowledge_graph.graph import Entity, KnowledgeGraph, Relation
from lexisearch.knowledge_graph.retriever import GraphRAGRetriever

__all__ = [
    "Entity",
    "EntityExtractor",
    "GraphRAGRetriever",
    "KnowledgeGraph",
    "Relation",
]
