# module8_neo4j_loader.py
"""
Module 8: Nap OntoKG vao Neo4j
Input :
  ./data/kg/entity_index.pkl        (entity metadata + surface forms - tu Module 5)
  ./data/kge/entity_embeddings.pt   (embedding cuoi cung Nx768 - tu Module 7)
  ./data/kge/entity_to_idx.json     (URI -> index - tu Module 7)
  ./data/module4_triples.jsonl      (triples - tu Module 4)

Ket qua: Neo4j database chua toan bo OntoKG, san sang truy van.

Mo hinh du lieu:
  (:Entity:Person   {uri, surface, wikidata_id, uri_source, embedding})
  (:Entity:Location {uri, surface, embedding})
  (:Article {article_id, uri})
  (:Topic   {name})
  (:Article)-[:HAS_ENTITY]->(:Entity)
  (:Entity)-[:OCCURS_AT {confidence}]->(:Entity)

Dependencies: pip install neo4j torch numpy
"""
from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from neo4j import GraphDatabase

LOGGER = logging.getLogger("module8_neo4j_loader")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)

LABEL_TO_NEO4J = {
    "PER":   "Person",
    "ORG":   "Organization",
    "LOC":   "Location",
    "EVENT": "Event",
    "TIME":  "TimeRef",
    "MISC":  "Concept",
}

RELATION_TO_NEO4J = {
    "occursAt":       "OCCURS_AT",
    "occursOn":       "OCCURS_ON",
    "organizedBy":    "ORGANIZED_BY",
    "participatesIn": "PARTICIPATES_IN",
    "locatedIn":      "LOCATED_IN",
    "manages":        "MANAGES",
    "hasPart":        "HAS_PART",
    "causedBy":       "CAUSED_BY",
    "relatedTo":      "RELATED_TO",
    "belongsTo":      "BELONGS_TO",
    "hasEntity":      "HAS_ENTITY",
}

ARTICLE_PREFIX = "http://transmtl.vn/article/"
TOPIC_PREFIX   = "http://transmtl.vn/onto/topic/"
BATCH_SIZE = 1000


class Neo4jLoader:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        LOGGER.info("Connected to Neo4j at %s", uri)

    def close(self):
        self.driver.close()

    def _run(self, query: str, params: Optional[Dict] = None):
        with self.driver.session(database=self.database) as session:
            return session.run(query, params or {}).consume()

    def _run_batched(self, query: str, rows: List[Dict], batch: int = BATCH_SIZE):
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch):
                session.run(query, {"rows": rows[i:i + batch]})

    def setup_schema(self, embedding_dim: int = 768):
        LOGGER.info("Setting up constraints and vector index...")
        self._run("CREATE CONSTRAINT entity_uri IF NOT EXISTS "
                  "FOR (e:Entity) REQUIRE e.uri IS UNIQUE")
        self._run("CREATE CONSTRAINT article_id IF NOT EXISTS "
                  "FOR (a:Article) REQUIRE a.article_id IS UNIQUE")
        self._run("CREATE CONSTRAINT topic_name IF NOT EXISTS "
                  "FOR (t:Topic) REQUIRE t.name IS UNIQUE")
        try:
            self._run(
                f"CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
                f"FOR (e:Entity) ON (e.embedding) "
                f"OPTIONS {{indexConfig: {{"
                f"`vector.dimensions`: {embedding_dim}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            )
            LOGGER.info("Vector index created (dim=%d)", embedding_dim)
        except Exception as e:
            LOGGER.warning("Vector index khong tao duoc (can Neo4j 5.11+): %s", e)

    def load_entities(self, entity_index_pkl, embeddings_pt, entity_to_idx_json):
        LOGGER.info("Loading entity metadata from %s", entity_index_pkl)
        with open(entity_index_pkl, "rb") as f:
            entity_data = pickle.load(f)

        emb_matrix = None
        uri_to_idx = {}
        if HAS_TORCH and Path(embeddings_pt).exists():
            emb_matrix = torch.load(embeddings_pt, map_location="cpu").numpy()
            with open(entity_to_idx_json, "r", encoding="utf-8") as f:
                uri_to_idx = json.load(f)
            LOGGER.info("Loaded embeddings: %s", emb_matrix.shape)

        by_label: Dict[str, List[Dict]] = defaultdict(list)
        for uri, item in entity_data.items():
            info  = item.get("info", {})
            label = info.get("label", "MISC")
            neo4j_label = LABEL_TO_NEO4J.get(label, "Concept")
            surfaces = info.get("surface_forms", [])
            surface  = surfaces[0] if surfaces else ""

            embedding = None
            if emb_matrix is not None and uri in uri_to_idx:
                embedding = emb_matrix[uri_to_idx[uri]].tolist()
            elif item.get("embedding") is not None:
                embedding = item["embedding"]

            by_label[neo4j_label].append({
                "uri":         uri,
                "surface":     surface,
                "wikidata_id": info.get("wikidata_id"),
                "uri_source":  info.get("uri_source", "new"),
                "occurrence":  info.get("occurrence_count", 0),
                "embedding":   embedding,
            })

        total = 0
        for neo4j_label, rows in by_label.items():
            query = (
                f"UNWIND $rows AS row "
                f"MERGE (e:Entity {{uri: row.uri}}) "
                f"SET e:{neo4j_label}, "
                f"    e.surface = row.surface, "
                f"    e.wikidata_id = row.wikidata_id, "
                f"    e.uri_source = row.uri_source, "
                f"    e.occurrence = row.occurrence, "
                f"    e.embedding = row.embedding"
            )
            self._run_batched(query, rows)
            total += len(rows)
            LOGGER.info("Loaded %d entities as :%s", len(rows), neo4j_label)
        LOGGER.info("Total entities loaded: %d", total)

    def load_triples(self, triples_jsonl):
        LOGGER.info("Loading triples from %s", triples_jsonl)
        by_relation: Dict[str, List[Dict]] = defaultdict(list)
        article_rows: List[Dict] = []
        topic_set: set = set()

        with open(triples_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                head = t.get("head_uri", "")
                rel  = t.get("relation", "")
                tail = t.get("tail_uri", "")
                if not (head and rel and tail):
                    continue
                if rel == "rdf:type":
                    continue

                if head.startswith(ARTICLE_PREFIX):
                    article_rows.append({
                        "article_id": head[len(ARTICLE_PREFIX):], "uri": head
                    })
                if tail.startswith(TOPIC_PREFIX):
                    topic_set.add(tail[len(TOPIC_PREFIX):])

                neo4j_rel = RELATION_TO_NEO4J.get(rel)
                if not neo4j_rel:
                    continue
                by_relation[neo4j_rel].append({
                    "head": head, "tail": tail,
                    "confidence": t.get("confidence", 1.0),
                    "method": t.get("method", ""),
                })

        if article_rows:
            seen = {r["article_id"]: r for r in article_rows}
            self._run_batched(
                "UNWIND $rows AS row "
                "MERGE (a:Article {article_id: row.article_id}) SET a.uri = row.uri",
                list(seen.values()),
            )
            LOGGER.info("Loaded %d Article nodes", len(seen))

        if topic_set:
            self._run_batched(
                "UNWIND $rows AS row MERGE (t:Topic {name: row.name})",
                [{"name": t} for t in topic_set],
            )
            LOGGER.info("Loaded %d Topic nodes", len(topic_set))

        for neo4j_rel, rows in by_relation.items():
            if neo4j_rel == "HAS_ENTITY":
                query = ("UNWIND $rows AS row "
                         "MATCH (a:Article {uri: row.head}) "
                         "MATCH (e:Entity {uri: row.tail}) "
                         "MERGE (a)-[:HAS_ENTITY]->(e)")
            elif neo4j_rel == "BELONGS_TO":
                query = ("UNWIND $rows AS row "
                         "MATCH (h {uri: row.head}) "
                         "MATCH (t:Topic {name: replace(row.tail, '" + TOPIC_PREFIX + "', '')}) "
                         "MERGE (h)-[:BELONGS_TO]->(t)")
            else:
                query = (f"UNWIND $rows AS row "
                         f"MATCH (h:Entity {{uri: row.head}}) "
                         f"MATCH (t:Entity {{uri: row.tail}}) "
                         f"MERGE (h)-[r:{neo4j_rel}]->(t) "
                         f"SET r.confidence = row.confidence, r.method = row.method")
            self._run_batched(query, rows)
            LOGGER.info("Loaded %d :%s relationships", len(rows), neo4j_rel)

    def print_stats(self):
        with self.driver.session(database=self.database) as session:
            n_e = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            n_a = session.run("MATCH (a:Article) RETURN count(a) AS c").single()["c"]
            n_r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print("\n" + "=" * 50)
        print("  NEO4J LOAD COMPLETE")
        print("=" * 50)
        print(f"  Entities      : {n_e:,}")
        print(f"  Articles      : {n_a:,}")
        print(f"  Relationships : {n_r:,}")
        print("=" * 50)

    def load_all(self, entity_index_pkl="./data/kg/entity_index.pkl",
                 embeddings_pt="./data/kge/entity_embeddings.pt",
                 entity_to_idx_json="./data/kge/entity_to_idx.json",
                 triples_jsonl="./data/module4_triples.jsonl",
                 embedding_dim=768, reset=False):
        if reset:
            LOGGER.warning("Xoa toan bo du lieu cu trong Neo4j...")
            self._run("MATCH (n) DETACH DELETE n")
        self.setup_schema(embedding_dim)
        self.load_entities(entity_index_pkl, embeddings_pt, entity_to_idx_json)
        self.load_triples(triples_jsonl)
        self.print_stats()


if __name__ == "__main__":
    import os
    DATA = os.environ.get("OKG_DATA_DIR", "./data")
    loader = Neo4jLoader(
        uri      = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user     = os.environ.get("NEO4J_USER", "neo4j"),
        password = os.environ.get("NEO4J_PASSWORD", "password"),
        database = os.environ.get("NEO4J_DATABASE", "neo4j"),
    )
    try:
        loader.load_all(
            entity_index_pkl   = os.path.join(DATA, "kg", "entity_index.pkl"),
            embeddings_pt      = os.path.join(DATA, "kge", "entity_embeddings.pt"),
            entity_to_idx_json = os.path.join(DATA, "kge", "entity_to_idx.json"),
            triples_jsonl      = os.path.join(DATA, "module4_triples.jsonl"),
            embedding_dim      = 768,
            reset              = True,
        )
    finally:
        loader.close()
