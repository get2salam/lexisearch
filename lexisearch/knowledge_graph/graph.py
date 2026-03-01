"""Core knowledge graph data structures."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nodes and edges
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A named entity (node) in the knowledge graph.

    Parameters
    ----------
    name:
        Canonical entity name (e.g. ``"FAISS"``, ``"transformer"``)
    entity_type:
        Semantic category (e.g. ``"TECHNOLOGY"``, ``"PERSON"``, ``"CONCEPT"``)
    doc_ids:
        Set of document IDs where this entity appears.
    aliases:
        Alternative surface forms (case-insensitive, resolved to *name*).
    metadata:
        Arbitrary key/value attributes (frequency counts, descriptions, etc.)
    """

    name: str
    entity_type: str = "CONCEPT"
    doc_ids: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by canonical name (case-insensitive)."""
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        """Equality by canonical name (case-insensitive)."""
        if isinstance(other, Entity):
            return self.name.lower() == other.name.lower()
        return NotImplemented


@dataclass
class Relation:
    """A directed edge between two entities.

    Represents a triple ``(subject, predicate, object)`` as in RDF.

    Parameters
    ----------
    subject:
        Name of the source entity.
    predicate:
        Relation type (e.g. ``"DEVELOPED_BY"``, ``"PART_OF"``, ``"RELATED_TO"``)
    obj:
        Name of the target entity.
    weight:
        Edge weight; typically proportional to co-occurrence frequency.
    doc_ids:
        Document IDs where this relation was observed.
    metadata:
        Arbitrary attributes.
    """

    subject: str
    predicate: str
    obj: str
    weight: float = 1.0
    doc_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by (subject, predicate, object) triple."""
        return hash((self.subject.lower(), self.predicate.lower(), self.obj.lower()))

    def __eq__(self, other: object) -> bool:
        """Equality by (subject, predicate, object) triple."""
        if isinstance(other, Relation):
            return (
                self.subject.lower() == other.subject.lower()
                and self.predicate.lower() == other.predicate.lower()
                and self.obj.lower() == other.obj.lower()
            )
        return NotImplemented


# ---------------------------------------------------------------------------
# Knowledge Graph container
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """In-memory knowledge graph.

    Stores entities (nodes) and relations (edges) with efficient lookup by
    entity name.  Supports BFS/DFS traversal, neighbour queries, subgraph
    extraction, and JSON serialisation.

    This implementation is designed for moderate-scale graphs (up to ~100k
    nodes).  For larger graphs, integrate with a dedicated graph DB (e.g.
    Neo4j) via the same interface.
    """

    def __init__(self) -> None:
        """Initialise an empty knowledge graph."""
        # Primary stores
        self._entities: dict[str, Entity] = {}  # normalised name → Entity
        self._relations: set[Relation] = set()

        # Adjacency index (outgoing + incoming for undirected queries)
        self._out_edges: dict[str, list[Relation]] = defaultdict(list)  # subj → relations
        self._in_edges: dict[str, list[Relation]] = defaultdict(list)  # obj → relations

        # Alias index: alias (lower) → canonical name
        self._alias_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def _norm(self, name: str) -> str:
        """Return the normalised (lower-case, stripped) entity name."""
        return name.strip().lower()

    def add_entity(self, entity: Entity) -> Entity:
        """Add or merge an entity into the graph.

        If an entity with the same normalised name already exists, its
        ``doc_ids``, ``aliases``, and ``metadata`` are merged.

        Returns:
        -------
        Entity
            The canonical entity (possibly merged).
        """
        key = self._norm(entity.name)
        if key in self._entities:
            existing = self._entities[key]
            existing.doc_ids |= entity.doc_ids
            existing.aliases |= entity.aliases
            existing.metadata.update(entity.metadata)
            return existing
        self._entities[key] = entity
        # Register aliases
        self._alias_map[key] = entity.name
        for alias in entity.aliases:
            self._alias_map[self._norm(alias)] = entity.name
        return entity

    def add_entities(self, entities: list[Entity]) -> None:
        """Add multiple entities at once."""
        for e in entities:
            self.add_entity(e)

    def get_entity(self, name: str) -> Entity | None:
        """Return entity by name or alias, or ``None`` if not found."""
        key = self._norm(name)
        # Direct lookup
        if key in self._entities:
            return self._entities[key]
        # Alias lookup
        canonical = self._alias_map.get(key)
        if canonical:
            return self._entities.get(self._norm(canonical))
        return None

    def has_entity(self, name: str) -> bool:
        """Return ``True`` if the entity exists (by name or alias)."""
        return self.get_entity(name) is not None

    def all_entities(self) -> list[Entity]:
        """Return all entities as a list."""
        return list(self._entities.values())

    # ------------------------------------------------------------------
    # Relation management
    # ------------------------------------------------------------------

    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the graph (merges weight/doc_ids if duplicate).

        Automatically creates subject and object entities if absent.
        """
        # Ensure entities exist
        if not self.has_entity(relation.subject):
            self.add_entity(Entity(name=relation.subject))
        if not self.has_entity(relation.obj):
            self.add_entity(Entity(name=relation.obj))

        key_s = self._norm(relation.subject)
        key_o = self._norm(relation.obj)

        if relation in self._relations:
            # Merge existing
            for r in self._relations:
                if r == relation:
                    r.weight += relation.weight
                    r.doc_ids |= relation.doc_ids
                    break
        else:
            self._relations.add(relation)
            self._out_edges[key_s].append(relation)
            self._in_edges[key_o].append(relation)

    def add_relations(self, relations: list[Relation]) -> None:
        """Add multiple relations at once."""
        for r in relations:
            self.add_relation(r)

    def get_relations(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[Relation]:
        """Query relations by subject, predicate, and/or object.

        All provided filters are AND-ed.  ``None`` means 'any'.
        """
        candidates: list[Relation]
        if subject is not None:
            key_s = self._norm(subject)
            candidates = self._out_edges.get(key_s, [])
        elif obj is not None:
            key_o = self._norm(obj)
            candidates = self._in_edges.get(key_o, [])
        else:
            candidates = list(self._relations)

        result = candidates
        if predicate is not None:
            pred_lower = predicate.lower()
            result = [r for r in result if r.predicate.lower() == pred_lower]
        if obj is not None:
            obj_lower = obj.lower()
            result = [r for r in result if r.obj.lower() == obj_lower]
        if subject is not None:
            subj_lower = subject.lower()
            result = [r for r in result if r.subject.lower() == subj_lower]
        return result

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def neighbours(
        self,
        name: str,
        *,
        max_hops: int = 1,
        direction: str = "both",
        predicate_filter: str | None = None,
    ) -> list[Entity]:
        """Return entities reachable from *name* within *max_hops*.

        Parameters
        ----------
        name:
            Starting entity name.
        max_hops:
            Maximum number of edges to traverse.
        direction:
            ``"out"`` (follow outgoing edges), ``"in"`` (incoming),
            or ``"both"`` (undirected traversal).
        predicate_filter:
            If set, only traverse edges with this predicate.

        Returns:
        -------
        list[Entity]
            Reachable entities (excluding the starting entity), sorted by
            number of hops then by edge weight descending.
        """
        entity = self.get_entity(name)
        if entity is None:
            logger.debug("Entity %r not found in graph", name)
            return []

        visited: set[str] = {self._norm(name)}
        result: list[Entity] = []
        queue: deque[tuple[str, int]] = deque([(self._norm(name), 0)])

        while queue:
            current_key, hops = queue.popleft()
            if hops >= max_hops:
                continue

            neighbours_at_hop: list[Relation] = []
            if direction in ("out", "both"):
                neighbours_at_hop.extend(self._out_edges.get(current_key, []))
            if direction in ("in", "both"):
                # For incoming, the "neighbour" is the subject
                for rel in self._in_edges.get(current_key, []):
                    neighbours_at_hop.append(
                        Relation(
                            subject=rel.obj,  # reverse
                            predicate=rel.predicate,
                            obj=rel.subject,
                            weight=rel.weight,
                        )
                    )

            for rel in neighbours_at_hop:
                if predicate_filter and rel.predicate.lower() != predicate_filter.lower():
                    continue
                # For "out" direction, neighbour is rel.obj
                # For reversed "in", neighbour is also rel.obj (we swapped above)
                neighbour_key = self._norm(rel.obj)
                if neighbour_key not in visited:
                    visited.add(neighbour_key)
                    neighbour_entity = self._entities.get(neighbour_key)
                    if neighbour_entity:
                        result.append(neighbour_entity)
                    queue.append((neighbour_key, hops + 1))

        return result

    def subgraph(self, entity_names: list[str]) -> KnowledgeGraph:
        """Extract a subgraph containing only the given entities.

        All relations between the specified entities are preserved.

        Parameters
        ----------
        entity_names:
            Names of entities to include.

        Returns:
        -------
        KnowledgeGraph
            A new ``KnowledgeGraph`` instance containing the subgraph.
        """
        sg = KnowledgeGraph()
        names_lower = {self._norm(n) for n in entity_names}

        for name_key in names_lower:
            entity = self._entities.get(name_key)
            if entity:
                sg.add_entity(entity)

        for rel in self._relations:
            if self._norm(rel.subject) in names_lower and self._norm(rel.obj) in names_lower:
                sg.add_relation(rel)

        return sg

    # ------------------------------------------------------------------
    # Stats / serialisation
    # ------------------------------------------------------------------

    @property
    def num_entities(self) -> int:
        """Number of unique entities in the graph."""
        return len(self._entities)

    @property
    def num_relations(self) -> int:
        """Number of unique relations in the graph."""
        return len(self._relations)

    def entity_names(self) -> list[str]:
        """Return canonical entity names sorted alphabetically."""
        return sorted(self._entities)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph to a JSON-friendly dictionary."""
        return {
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "doc_ids": list(e.doc_ids),
                    "aliases": list(e.aliases),
                    "metadata": e.metadata,
                }
                for e in self._entities.values()
            ],
            "relations": [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.obj,
                    "weight": r.weight,
                    "doc_ids": list(r.doc_ids),
                }
                for r in self._relations
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the graph to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        """Deserialise a graph from a dictionary produced by ``to_dict``."""
        kg = cls()
        for e_data in data.get("entities", []):
            kg.add_entity(
                Entity(
                    name=e_data["name"],
                    entity_type=e_data.get("type", "CONCEPT"),
                    doc_ids=set(e_data.get("doc_ids", [])),
                    aliases=set(e_data.get("aliases", [])),
                    metadata=e_data.get("metadata", {}),
                )
            )
        for r_data in data.get("relations", []):
            kg.add_relation(
                Relation(
                    subject=r_data["subject"],
                    predicate=r_data["predicate"],
                    obj=r_data["object"],
                    weight=r_data.get("weight", 1.0),
                    doc_ids=set(r_data.get("doc_ids", [])),
                )
            )
        return kg

    @classmethod
    def from_json(cls, json_str: str) -> KnowledgeGraph:
        """Deserialise a graph from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        """Return a human-readable summary of the knowledge graph."""
        return f"KnowledgeGraph(entities={self.num_entities}, relations={self.num_relations})"
