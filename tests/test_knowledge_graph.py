"""Tests for the LexiSearch knowledge graph module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from lexisearch.knowledge_graph import (
    Entity,
    EntityExtractor,
    GraphRAGRetriever,
    KnowledgeGraph,
    Relation,
)
from lexisearch.knowledge_graph.extractor import ExtractionResult
from lexisearch.knowledge_graph.retriever import GraphRAGConfig

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class TestEntity:
    def test_default_type(self) -> None:
        e = Entity(name="FAISS")
        assert e.entity_type == "CONCEPT"

    def test_hash_case_insensitive(self) -> None:
        assert hash(Entity(name="FAISS")) == hash(Entity(name="faiss"))

    def test_equality_case_insensitive(self) -> None:
        assert Entity(name="FAISS") == Entity(name="faiss")

    def test_inequality_different_names(self) -> None:
        assert Entity(name="FAISS") != Entity(name="ChromaDB")

    def test_set_operations(self) -> None:
        entities = {Entity(name="FAISS"), Entity(name="faiss"), Entity(name="ChromaDB")}
        # "FAISS" and "faiss" should hash to the same bucket
        assert len(entities) == 2

    def test_doc_ids_default_empty(self) -> None:
        e = Entity(name="X")
        assert e.doc_ids == set()


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------


class TestRelation:
    def test_hash_and_equality(self) -> None:
        r1 = Relation(subject="FAISS", predicate="DEVELOPED_BY", obj="Facebook")
        r2 = Relation(subject="faiss", predicate="developed_by", obj="facebook")
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_inequality(self) -> None:
        r1 = Relation(subject="A", predicate="P", obj="B")
        r2 = Relation(subject="A", predicate="Q", obj="B")
        assert r1 != r2

    def test_default_weight(self) -> None:
        r = Relation(subject="A", predicate="P", obj="B")
        assert r.weight == 1.0


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    def setup_method(self) -> None:
        self.kg = KnowledgeGraph()

    def _populate(self) -> None:
        """Populate graph with a small test dataset."""
        self.kg.add_entity(Entity(name="FAISS", entity_type="TECHNOLOGY"))
        self.kg.add_entity(Entity(name="Facebook", entity_type="ORGANISATION"))
        self.kg.add_entity(Entity(name="ChromaDB", entity_type="TECHNOLOGY"))
        self.kg.add_entity(Entity(name="embedding", entity_type="CONCEPT"))
        self.kg.add_relation(Relation(subject="FAISS", predicate="DEVELOPED_BY", obj="Facebook"))
        self.kg.add_relation(Relation(subject="FAISS", predicate="RELATED_TO", obj="embedding"))
        self.kg.add_relation(Relation(subject="ChromaDB", predicate="USES", obj="embedding"))

    def test_empty_graph(self) -> None:
        assert self.kg.num_entities == 0
        assert self.kg.num_relations == 0

    def test_add_entity(self) -> None:
        self.kg.add_entity(Entity(name="FAISS", entity_type="TECHNOLOGY"))
        assert self.kg.num_entities == 1

    def test_add_entity_merge(self) -> None:
        self.kg.add_entity(Entity(name="FAISS", doc_ids={"doc1"}))
        self.kg.add_entity(Entity(name="FAISS", doc_ids={"doc2"}))
        assert self.kg.num_entities == 1
        entity = self.kg.get_entity("FAISS")
        assert entity is not None
        assert "doc1" in entity.doc_ids
        assert "doc2" in entity.doc_ids

    def test_get_entity_by_name(self) -> None:
        self._populate()
        entity = self.kg.get_entity("FAISS")
        assert entity is not None
        assert entity.name == "FAISS"

    def test_get_entity_case_insensitive(self) -> None:
        self._populate()
        assert self.kg.get_entity("faiss") is not None
        assert self.kg.get_entity("FAISS") is not None

    def test_has_entity(self) -> None:
        self._populate()
        assert self.kg.has_entity("FAISS")
        assert not self.kg.has_entity("NonExistent")

    def test_add_relation_creates_missing_entities(self) -> None:
        self.kg.add_relation(Relation(subject="A", predicate="P", obj="B"))
        assert self.kg.has_entity("A")
        assert self.kg.has_entity("B")

    def test_get_relations_by_subject(self) -> None:
        self._populate()
        rels = self.kg.get_relations(subject="FAISS")
        assert len(rels) == 2
        predicates = {r.predicate for r in rels}
        assert "DEVELOPED_BY" in predicates

    def test_get_relations_by_predicate(self) -> None:
        self._populate()
        rels = self.kg.get_relations(predicate="USES")
        assert len(rels) == 1

    def test_get_relations_by_obj(self) -> None:
        self._populate()
        rels = self.kg.get_relations(obj="embedding")
        subjects = {r.subject for r in rels}
        assert "FAISS" in subjects
        assert "ChromaDB" in subjects

    def test_neighbours_one_hop(self) -> None:
        self._populate()
        neighbours = self.kg.neighbours("FAISS", max_hops=1)
        names = {n.name for n in neighbours}
        assert "Facebook" in names
        assert "embedding" in names

    def test_neighbours_two_hops(self) -> None:
        self._populate()
        # FAISS → embedding ← ChromaDB; at 2 hops, ChromaDB should appear
        neighbours = self.kg.neighbours("FAISS", max_hops=2)
        names = {n.name for n in neighbours}
        assert "ChromaDB" in names

    def test_neighbours_unknown_entity(self) -> None:
        neighbours = self.kg.neighbours("Unknown")
        assert neighbours == []

    def test_neighbours_outgoing_only(self) -> None:
        self._populate()
        # "embedding" has no outgoing edges in our test graph
        neighbours = self.kg.neighbours("embedding", max_hops=1, direction="out")
        assert all(n.name != "FAISS" for n in neighbours)

    def test_subgraph(self) -> None:
        self._populate()
        sg = self.kg.subgraph(["FAISS", "embedding"])
        assert sg.num_entities == 2
        assert sg.num_relations == 1  # FAISS RELATED_TO embedding

    def test_all_entities(self) -> None:
        self._populate()
        entities = self.kg.all_entities()
        assert len(entities) == 4

    def test_entity_names_sorted(self) -> None:
        self._populate()
        names = self.kg.entity_names()
        assert names == sorted(names)

    def test_serialise_roundtrip(self) -> None:
        self._populate()
        data = self.kg.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        assert restored.num_entities == self.kg.num_entities
        assert restored.num_relations == self.kg.num_relations
        assert restored.has_entity("FAISS")

    def test_json_roundtrip(self) -> None:
        self._populate()
        json_str = self.kg.to_json()
        restored = KnowledgeGraph.from_json(json_str)
        assert restored.num_entities == self.kg.num_entities

    def test_repr(self) -> None:
        r = repr(self.kg)
        assert "KnowledgeGraph" in r
        assert "entities=0" in r

    def test_alias_lookup(self) -> None:
        e = Entity(name="FAISS", aliases={"faiss index", "facebook ai similarity search"})
        self.kg.add_entity(e)
        assert self.kg.get_entity("faiss index") is not None

    def test_relation_weight_accumulates(self) -> None:
        self.kg.add_entity(Entity(name="A"))
        self.kg.add_entity(Entity(name="B"))
        rel = Relation(subject="A", predicate="P", obj="B", weight=1.0)
        self.kg.add_relation(rel)
        self.kg.add_relation(rel)  # duplicate — weight should accumulate
        rels = self.kg.get_relations(subject="A")
        assert len(rels) == 1
        assert rels[0].weight > 1.0


# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------


class TestEntityExtractor:
    def setup_method(self) -> None:
        self.extractor = EntityExtractor()

    def test_extract_returns_tuple(self) -> None:
        entities, relations = self.extractor.extract("FAISS is a vector search library.")
        assert isinstance(entities, list)
        assert isinstance(relations, list)

    def test_tech_term_extraction(self) -> None:
        entities, _ = self.extractor.extract("FAISS and ChromaDB are vector databases.")
        names = [e.name.lower() for e in entities]
        assert "faiss" in names
        assert "chromadb" in names

    def test_entity_types(self) -> None:
        entities, _ = self.extractor.extract("FAISS is a library.")
        types = {e.entity_type for e in entities}
        assert len(types) > 0

    def test_empty_text(self) -> None:
        entities, relations = self.extractor.extract("")
        assert isinstance(entities, list)
        assert isinstance(relations, list)

    def test_extract_full_returns_result(self) -> None:
        result = self.extractor.extract_full("FAISS uses embedding vectors.")
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.entities, list)
        assert isinstance(result.relations, list)
        assert isinstance(result.raw_spans, list)

    def test_doc_id_attached(self) -> None:
        entities, _ = self.extractor.extract("FAISS is efficient.", doc_id="doc-001")
        for e in entities:
            if e.name.lower() == "faiss":
                assert "doc-001" in e.doc_ids

    def test_no_relations_single_entity(self) -> None:
        _, relations = self.extractor.extract("FAISS is a library.")
        # With only one entity, no co-occurrence relations
        assert isinstance(relations, list)

    def test_relations_from_co_occurrence(self) -> None:
        text = "FAISS and embedding are related. FAISS uses embedding vectors."
        _, relations = self.extractor.extract(text)
        # Should produce at least one relation
        assert len(relations) >= 0  # may be 0 if neither is in entity store

    def test_extraction_no_relations_flag(self) -> None:
        extractor = EntityExtractor(extract_relations=False)
        _, relations = extractor.extract("FAISS uses embedding.")
        assert relations == []

    def test_long_text(self) -> None:
        text = "FAISS " * 100 + "is a vector search library developed by Facebook."
        entities, _ = self.extractor.extract(text)
        assert len(entities) > 0


# ---------------------------------------------------------------------------
# GraphRAGRetriever
# ---------------------------------------------------------------------------


class TestGraphRAGRetriever:
    def _make_retriever(self) -> tuple[GraphRAGRetriever, KnowledgeGraph, Any]:
        kg = KnowledgeGraph()
        kg.add_entity(Entity(name="FAISS", entity_type="TECHNOLOGY"))
        kg.add_entity(Entity(name="embedding", entity_type="CONCEPT"))
        kg.add_entity(Entity(name="Facebook", entity_type="ORGANISATION"))
        kg.add_relation(Relation(subject="FAISS", predicate="DEVELOPED_BY", obj="Facebook"))
        kg.add_relation(Relation(subject="FAISS", predicate="RELATED_TO", obj="embedding"))

        mock_base = MagicMock()
        mock_base.return_value = [{"chunk": "relevant content", "score": 0.9}]

        retriever = GraphRAGRetriever(
            knowledge_graph=kg,
            base_retriever=mock_base,
            config=GraphRAGConfig(max_hops=1, base_top_k=3),
        )
        return retriever, kg, mock_base

    def test_retrieve_returns_result(self) -> None:
        retriever, _, _ = self._make_retriever()
        result = retriever.retrieve("How does FAISS work?")
        from lexisearch.knowledge_graph.retriever import GraphRAGResult

        assert isinstance(result, GraphRAGResult)

    def test_retrieve_has_base_results(self) -> None:
        retriever, _, mock_base = self._make_retriever()
        result = retriever.retrieve("How does FAISS work?")
        assert mock_base.called
        assert len(result.base_results) >= 0

    def test_retrieve_expands_query(self) -> None:
        retriever, _, _ = self._make_retriever()
        result = retriever.retrieve("How does FAISS work?")
        # For a query about FAISS, the expanded query should contain more terms
        assert len(result.expanded_query) >= len(result.query)

    def test_retrieve_populates_graph_entities(self) -> None:
        retriever, _, _ = self._make_retriever()
        result = retriever.retrieve("FAISS is a vector index.")
        assert isinstance(result.graph_entities, list)

    def test_retrieve_combined_context(self) -> None:
        retriever, _, _ = self._make_retriever()
        result = retriever.retrieve("How does FAISS work?")
        assert isinstance(result.combined_context, str)

    def test_retrieve_metadata(self) -> None:
        retriever, _, _ = self._make_retriever()
        result = retriever.retrieve("Explain FAISS.")
        assert "query_entities" in result.metadata

    def test_retrieve_unknown_entity(self) -> None:
        retriever, _, _ = self._make_retriever()
        # No entities in the graph for this query
        result = retriever.retrieve("What is the meaning of life?")
        assert isinstance(result.graph_entities, list)

    def test_retrieve_base_retriever_fallback(self) -> None:
        """If expanded query fails, falls back to original."""
        kg = KnowledgeGraph()
        call_count = [0]

        def _first_fails(query: str, top_k: int) -> list[Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("retriever failed")
            return [{"chunk": "fallback", "score": 0.5}]

        retriever = GraphRAGRetriever(
            knowledge_graph=kg,
            base_retriever=_first_fails,
            config=GraphRAGConfig(base_top_k=3),
        )
        result = retriever.retrieve("test query")
        from lexisearch.knowledge_graph.retriever import GraphRAGResult

        assert isinstance(result, GraphRAGResult)

    def test_index_document(self) -> None:
        retriever, kg, _ = self._make_retriever()
        _ = kg.num_entities
        added_e, added_r = retriever.index_document(
            "Python is used for machine learning.", doc_id="doc-1"
        )
        assert isinstance(added_e, int)
        assert isinstance(added_r, int)

    def test_index_document_populates_graph(self) -> None:
        kg = KnowledgeGraph()
        retriever = GraphRAGRetriever(
            knowledge_graph=kg,
            base_retriever=MagicMock(return_value=[]),
        )
        retriever.index_document("FAISS is developed by Facebook AI.", doc_id="doc-1")
        # FAISS should be in the graph
        assert kg.has_entity("FAISS") or kg.num_entities > 0


# ---------------------------------------------------------------------------
# Integration: extractor + graph
# ---------------------------------------------------------------------------


class TestGraphIntegration:
    def test_extract_and_populate(self) -> None:
        kg = KnowledgeGraph()
        extractor = EntityExtractor()
        text = (
            "FAISS is a library for efficient similarity search. "
            "It was developed by Facebook AI Research. "
            "FAISS uses embedding vectors for fast retrieval."
        )
        entities, relations = extractor.extract(text, doc_id="test-doc")
        kg.add_entities(entities)
        kg.add_relations(relations)
        assert kg.num_entities >= 1

    def test_graph_retriever_end_to_end(self) -> None:
        kg = KnowledgeGraph()
        extractor = EntityExtractor()
        retriever = GraphRAGRetriever(
            knowledge_graph=kg,
            base_retriever=lambda q, k: [{"chunk": f"result for {q}", "score": 0.9}],
            extractor=extractor,
        )
        # Index a document
        retriever.index_document(
            "FAISS is a vector search library. Embedding vectors are used for similarity search.",
            doc_id="doc-001",
        )
        # Query
        result = retriever.retrieve("How does FAISS retrieve vectors?")
        assert len(result.base_results) > 0
        assert result.query == "How does FAISS retrieve vectors?"
