# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Orchestration Context Builders."""

import re
from enum import Enum

from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.language_model.protocol.base import EmbeddingModel
from graphrag.query.input.retrieval.entities import (
    get_entity_by_id,
    get_entity_by_key,
    get_entity_by_name,
)
from graphrag.vector_stores.base import BaseVectorStore


class EntityVectorStoreKey(str, Enum):
    """Keys used as ids in the entity embedding vectorstores."""

    ID = "id"
    TITLE = "title"

    @staticmethod
    def from_string(value: str) -> "EntityVectorStoreKey":
        """Convert string to EntityVectorStoreKey."""
        if value == "id":
            return EntityVectorStoreKey.ID
        if value == "title":
            return EntityVectorStoreKey.TITLE

        msg = f"Invalid EntityVectorStoreKey: {value}"
        raise ValueError(msg)


def map_query_to_entities(
    query: str,
    text_embedding_vectorstore: BaseVectorStore,
    text_embedder: EmbeddingModel,
    all_entities_dict: dict[str, Entity],
    embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    include_entity_names: list[str] | None = None,
    exclude_entity_names: list[str] | None = None,
    k: int = 10,
    oversample_scaler: int = 2,
) -> list[Entity]:
    """Extract entities that match a given query using semantic similarity of text embeddings of query and entity descriptions."""
    if include_entity_names is None:
        include_entity_names = []
    if exclude_entity_names is None:
        exclude_entity_names = []
    all_entities = list(all_entities_dict.values())
    matched_entities = []
    if query != "":
        print("=== RUNNING CUSTOM DrugBank ENTITY EXTRACTION ===")
        # Extract all DrugBank IDs from the query (format: DB followed by 5 digits)
        drugbank_pattern = r"DB\d{5}"
        drugbank_ids = re.findall(drugbank_pattern, query)
        print("DRUGBANK IDS FOUND:", drugbank_ids)

        # For each DrugBank ID, find the entity with matching title
        for drugbank_id in drugbank_ids:
            # Search through all entities for matching title
            for entity in all_entities:
                if entity.title == drugbank_id:
                    matched_entities.append(entity)
                    break  # Only add the first match for each DrugBank ID
    else:
        all_entities.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
        matched_entities = all_entities[:k]

    # filter out excluded entities
    if exclude_entity_names:
        matched_entities = [
            entity
            for entity in matched_entities
            if entity.title not in exclude_entity_names
        ]

    # add entities in the include_entity list
    included_entities = []
    for entity_name in include_entity_names:
        included_entities.extend(get_entity_by_name(all_entities, entity_name))
    return included_entities + matched_entities


def find_nearest_neighbors_by_entity_rank(
    entity_name: str,
    all_entities: list[Entity],
    all_relationships: list[Relationship],
    exclude_entity_names: list[str] | None = None,
    k: int | None = 10,
) -> list[Entity]:
    """Retrieve entities that have direct connections with the target entity, sorted by entity rank."""
    if exclude_entity_names is None:
        exclude_entity_names = []
    entity_relationships = [
        rel
        for rel in all_relationships
        if rel.source == entity_name or rel.target == entity_name
    ]
    source_entity_names = {rel.source for rel in entity_relationships}
    target_entity_names = {rel.target for rel in entity_relationships}
    related_entity_names = (source_entity_names.union(target_entity_names)).difference(
        set(exclude_entity_names)
    )
    top_relations = [
        entity for entity in all_entities if entity.title in related_entity_names
    ]
    top_relations.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
    if k:
        return top_relations[:k]
    return top_relations
