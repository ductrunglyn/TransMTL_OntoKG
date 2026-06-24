# module9_neo4j_retrieval.py
"""
Module 9: Truy van subgraph tu Neo4j de ket hop voi TransMTL
Dung trong CA training va inference cua TransMTL.

Ba chuc nang chinh:
  1. get_article_subgraph(article_id):
       Lay subgraph cua 1 bai bao (entities + relations giua chung)
       -> tra ve dinh dang cho GNN encoder (node features + edge_index + edge_type)

  2. link_entity_by_vector(embedding, top_k):
       Tim entity tuong dong nhat bang VECTOR INDEX cua Neo4j
       -> dung cho entity linking (thay brute-force cosine o Module 3 Tang 3)

  3. get_entity_neighbors(uris, hops):
       Lay k-hop neighborhood cho entity moi (ho tro inductive embedding)

Dependencies: pip install neo4j torch numpy
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from neo4j import GraphDatabase

LOGGER = logging.getLogger("module9_neo4j_retrieval")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)

# Cac relation type ngu nghia (bo metadata khi xay subgraph cho GNN)
SEMANTIC_RELATIONS = [
    "OCCURS_AT", "OCCURS_ON", "ORGANIZED_BY", "PARTICIPATES_IN",
    "LOCATED_IN", "MANAGES", "HAS_PART", "CAUSED_BY", "RELATED_TO",
]
# Anh xa relation type -> id (cho R-GCN edge_type)
RELATION_TO_ID = {rel: i for i, rel in enumerate(SEMANTIC_RELATIONS)}


class Neo4jRetriever:
    def __init__(self, uri: str, user: str, password: str,
                 database: str = "neo4j", embedding_dim: int = 768):
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.dim      = embedding_dim
        LOGGER.info("Connected to Neo4j at %s", uri)

    def close(self):
        self.driver.close()

    # ──────────────────────────────────────────────────────
    # 1. Subgraph cua mot bai bao -> dinh dang cho GNN
    # ──────────────────────────────────────────────────────
    def get_article_subgraph(self, article_id: str) -> Dict[str, Any]:
        """
        Tra ve subgraph cua bai bao duoi dang san sang cho PyTorch Geometric:
          {
            "uris":        [uri_0, uri_1, ...],         # thu tu node
            "node_feat":   np.ndarray (N, dim),         # embedding moi node
            "edge_index":  np.ndarray (2, E),           # [head_idx; tail_idx]
            "edge_type":   np.ndarray (E,),             # id loai quan he
            "labels":      [label_0, ...],              # nhan ontology moi node
          }
        """
        rel_pattern = "|".join(SEMANTIC_RELATIONS)

        query = f"""
        MATCH (a:Article {{article_id: $aid}})-[:HAS_ENTITY]->(e:Entity)
        WITH collect(DISTINCT e) AS ents
        UNWIND ents AS e
        OPTIONAL MATCH (e)-[r:{rel_pattern}]->(e2:Entity)
        WHERE e2 IN ents
        RETURN
          [x IN ents | {{uri: x.uri, emb: x.embedding,
                         label: head(labels(x))}}] AS nodes,
          collect(CASE WHEN r IS NOT NULL
                  THEN {{head: e.uri, tail: e2.uri, type: type(r)}}
                  ELSE NULL END) AS rels
        """
        with self.driver.session(database=self.database) as session:
            rec = session.run(query, {"aid": article_id}).single()

        if rec is None or not rec["nodes"]:
            return self._empty_subgraph()

        # Node list (dedup theo uri)
        seen = {}
        for n in rec["nodes"]:
            if n["uri"] not in seen:
                seen[n["uri"]] = n
        nodes = list(seen.values())

        uris   = [n["uri"] for n in nodes]
        labels = [n["label"] for n in nodes]
        uri_to_idx = {u: i for i, u in enumerate(uris)}

        # Node features
        feat = np.zeros((len(nodes), self.dim), dtype=np.float32)
        for i, n in enumerate(nodes):
            if n["emb"] is not None:
                feat[i] = np.array(n["emb"], dtype=np.float32)

        # Edges
        heads, tails, etypes = [], [], []
        for r in rec["rels"]:
            if r is None:
                continue
            h, t = r["head"], r["tail"]
            if h in uri_to_idx and t in uri_to_idx:
                heads.append(uri_to_idx[h])
                tails.append(uri_to_idx[t])
                etypes.append(RELATION_TO_ID.get(r["type"], 0))

        edge_index = (np.array([heads, tails], dtype=np.int64)
                      if heads else np.zeros((2, 0), dtype=np.int64))
        edge_type  = np.array(etypes, dtype=np.int64) if etypes else np.zeros((0,), dtype=np.int64)

        return {
            "uris": uris, "node_feat": feat,
            "edge_index": edge_index, "edge_type": edge_type,
            "labels": labels,
        }

    def _empty_subgraph(self) -> Dict[str, Any]:
        return {
            "uris": [], "node_feat": np.zeros((0, self.dim), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_type": np.zeros((0,), dtype=np.int64), "labels": [],
        }

    def subgraph_to_torch(self, sg: Dict[str, Any]):
        """Chuyen subgraph numpy -> tensor cho PyTorch Geometric."""
        if not HAS_TORCH:
            raise RuntimeError("Can torch: pip install torch")
        return {
            "x":          torch.tensor(sg["node_feat"], dtype=torch.float32),
            "edge_index": torch.tensor(sg["edge_index"], dtype=torch.long),
            "edge_type":  torch.tensor(sg["edge_type"], dtype=torch.long),
            "uris":       sg["uris"],
            "labels":     sg["labels"],
        }

    # ──────────────────────────────────────────────────────
    # 2. Entity linking bang VECTOR INDEX
    # ──────────────────────────────────────────────────────
    def link_entity_by_vector(
        self, embedding: List[float], top_k: int = 5,
        min_score: float = 0.92,
    ) -> List[Dict[str, Any]]:
        """
        Tim entity tuong dong nhat bang vector index cua Neo4j.
        Thay the brute-force cosine o Module 3 Tang 3 (nhanh hon nhieu).
        Tra ve [{uri, surface, score}, ...] da loc theo min_score.
        """
        query = """
        CALL db.index.vector.queryNodes('entity_embedding', $k, $vec)
        YIELD node, score
        RETURN node.uri AS uri, node.surface AS surface, score
        ORDER BY score DESC
        """
        with self.driver.session(database=self.database) as session:
            results = session.run(query, {"k": top_k, "vec": embedding}).data()
        return [r for r in results if r["score"] >= min_score]

    # ──────────────────────────────────────────────────────
    # 3. K-hop neighborhood cho entity moi (inductive)
    # ──────────────────────────────────────────────────────
    def get_entity_neighbors(
        self, uris: List[str], hops: int = 1,
    ) -> Dict[str, Any]:
        """
        Lay k-hop neighborhood cua cac entity da biet.
        Dung khi entity moi noi voi node da biet -> GNN tinh inductive embedding.
        """
        rel_pattern = "|".join(SEMANTIC_RELATIONS)
        query = f"""
        MATCH (e:Entity) WHERE e.uri IN $uris
        MATCH path = (e)-[:{rel_pattern}*1..{hops}]-(nb:Entity)
        WITH collect(DISTINCT e) + collect(DISTINCT nb) AS allnodes
        UNWIND allnodes AS n
        WITH DISTINCT n
        OPTIONAL MATCH (n)-[r:{rel_pattern}]->(m:Entity)
        RETURN
          collect(DISTINCT {{uri: n.uri, emb: n.embedding,
                            label: head(labels(n))}}) AS nodes,
          collect(DISTINCT CASE WHEN r IS NOT NULL
                  THEN {{head: n.uri, tail: m.uri, type: type(r)}}
                  ELSE NULL END) AS rels
        """
        with self.driver.session(database=self.database) as session:
            rec = session.run(query, {"uris": uris}).single()
        if rec is None:
            return self._empty_subgraph()

        # Tai su dung logic build subgraph
        nodes = [n for n in rec["nodes"] if n["uri"]]
        uris_list  = [n["uri"] for n in nodes]
        labels     = [n["label"] for n in nodes]
        uri_to_idx = {u: i for i, u in enumerate(uris_list)}

        feat = np.zeros((len(nodes), self.dim), dtype=np.float32)
        for i, n in enumerate(nodes):
            if n["emb"] is not None:
                feat[i] = np.array(n["emb"], dtype=np.float32)

        heads, tails, etypes = [], [], []
        for r in rec["rels"]:
            if r is None:
                continue
            if r["head"] in uri_to_idx and r["tail"] in uri_to_idx:
                heads.append(uri_to_idx[r["head"]])
                tails.append(uri_to_idx[r["tail"]])
                etypes.append(RELATION_TO_ID.get(r["type"], 0))

        edge_index = (np.array([heads, tails], dtype=np.int64)
                      if heads else np.zeros((2, 0), dtype=np.int64))
        edge_type  = np.array(etypes, dtype=np.int64) if etypes else np.zeros((0,), dtype=np.int64)

        return {
            "uris": uris_list, "node_feat": feat,
            "edge_index": edge_index, "edge_type": edge_type,
            "labels": labels,
        }


# ──────────────────────────────────────────────────────────
# Vi du su dung trong vong lap training/inference cua TransMTL
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = Neo4jRetriever(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",   # ← đổi từ "your_password" thành "password"
        database="neo4j",
        embedding_dim=768,
    )
    try:
        # --- Vi du 1: lay subgraph cua mot bai bao ---
        sg = retriever.get_article_subgraph("article_000000")
        print("Subgraph article_000000:")
        print(f"  So node : {len(sg['uris'])}")
        print(f"  So edge : {sg['edge_index'].shape[1]}")
        print(f"  Labels  : {set(sg['labels'])}")

        # Chuyen sang tensor cho GNN encoder cua TransMTL
        if HAS_TORCH and len(sg["uris"]) > 0:
            batch = retriever.subgraph_to_torch(sg)
            print(f"  x shape         : {tuple(batch['x'].shape)}")
            print(f"  edge_index shape: {tuple(batch['edge_index'].shape)}")
            # batch nay dua thang vao R-GCN/GAT encoder cua TransMTL

        # --- Vi du 2: entity linking bang vector index ---
        # vec = phobert_encode("Ha Noi")  # 768-dim
        # matches = retriever.link_entity_by_vector(vec, top_k=5, min_score=0.92)
        # print("Vector matches:", matches)

    finally:
        retriever.close()
