# module9_local_retrieval.py
"""
Backend TRUY VAN SUBGRAPH KHONG CAN NEO4J / DOCKER.

Doc thang cac artifact da sinh tren dia (Module 4 + Module 7) va dung
chi muc trong bo nho, cung cap dung giao dien nhu Neo4jRetriever:
    get_article_subgraph(article_id) -> {uris, node_feat, edge_index, edge_type, labels}
    subgraph_to_torch(sg)            -> {x, edge_index, edge_type, uris, labels}
    close()

Nho vay TransMTL co the train/test voi OntoKG ma KHONG can dung Neo4j.

Input:
  entity_emb_path   : data/kge/entity_embeddings.pt   (N x 768, tu Module 7)
  entity_idx_path   : data/kge/entity_to_idx.json     (URI -> index)
  triples_jsonl     : data/module4_triples.jsonl      (tu Module 4)

Dependencies: torch, numpy  (KHONG can neo4j)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

LOGGER = logging.getLogger("module9_local_retrieval")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)

ARTICLE_PREFIX = "http://transmtl.vn/article/"

# Cac quan he ngu nghia (ten camelCase nhu trong module4_triples.jsonl).
# Thu tu nay TRUNG voi SEMANTIC_RELATIONS cua module9_neo4j_retrieval.py
# de id quan he nhat quan giua hai backend.
SEMANTIC_RELATIONS = [
    "occursAt", "occursOn", "organizedBy", "participatesIn",
    "locatedIn", "manages", "hasPart", "causedBy", "relatedTo",
]
RELATION_TO_ID = {rel: i for i, rel in enumerate(SEMANTIC_RELATIONS)}
HAS_ENTITY_REL = "hasEntity"


class LocalKGRetriever:
    """Truy van subgraph theo article_id tu file, khong dung database."""

    def __init__(self, entity_emb_path: str, entity_idx_path: str,
                 triples_jsonl: str, embedding_dim: int = 768):
        if not HAS_TORCH:
            raise RuntimeError("Can torch: pip install torch")
        self.dim = embedding_dim

        # ── 1) Embedding matrix + URI -> index ──────────────────
        emb_path, idx_path = Path(entity_emb_path), Path(entity_idx_path)
        if not emb_path.exists():
            raise FileNotFoundError(f"Khong thay entity_embeddings.pt: {emb_path}")
        if not idx_path.exists():
            raise FileNotFoundError(f"Khong thay entity_to_idx.json: {idx_path}")
        mat = torch.load(str(emb_path), map_location="cpu")
        self.emb = mat.numpy().astype(np.float32) if hasattr(mat, "numpy") else np.asarray(mat, np.float32)
        with open(idx_path, "r", encoding="utf-8") as f:
            self.uri_to_idx: Dict[str, int] = json.load(f)
        LOGGER.info("Local KG: nap embedding %s, %d URI", self.emb.shape, len(self.uri_to_idx))

        # ── 2) Mot lan quet triples: article->entities & canh ngu nghia ──
        tri_path = Path(triples_jsonl)
        if not tri_path.exists():
            raise FileNotFoundError(f"Khong thay module4_triples.jsonl: {tri_path}")
        self.article_entities: Dict[str, List[str]] = defaultdict(list)
        self.edges_by_head: Dict[str, List[tuple]] = defaultdict(list)  # head -> [(tail, rel_id)]
        n_he = n_sem = 0
        with open(tri_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                h, rel, tl = t.get("head_uri", ""), t.get("relation", ""), t.get("tail_uri", "")
                if not (h and rel and tl):
                    continue
                if rel == HAS_ENTITY_REL and h.startswith(ARTICLE_PREFIX):
                    self.article_entities[h[len(ARTICLE_PREFIX):]].append(tl)
                    n_he += 1
                elif rel in RELATION_TO_ID:
                    self.edges_by_head[h].append((tl, RELATION_TO_ID[rel]))
                    n_sem += 1
        LOGGER.info("Local KG: %d bai, %d canh hasEntity, %d canh ngu nghia",
                    len(self.article_entities), n_he, n_sem)

    def close(self):
        pass

    # ──────────────────────────────────────────────────────────
    def _empty_subgraph(self) -> Dict[str, Any]:
        return {
            "uris": [], "node_feat": np.zeros((0, self.dim), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_type": np.zeros((0,), dtype=np.int64), "labels": [],
        }

    def get_article_subgraph(self, article_id: str) -> Dict[str, Any]:
        """Tra ve subgraph 1 bai bao, dinh dang giong Neo4jRetriever."""
        ents = self.article_entities.get(str(article_id))
        if not ents:
            return self._empty_subgraph()

        # Node: entity duy nhat cua bai, giu thu tu xuat hien
        seen, uris = set(), []
        for u in ents:
            if u not in seen:
                seen.add(u); uris.append(u)
        uri_to_local = {u: i for i, u in enumerate(uris)}

        feat = np.zeros((len(uris), self.dim), dtype=np.float32)
        for i, u in enumerate(uris):
            j = self.uri_to_idx.get(u)
            if j is not None and 0 <= j < self.emb.shape[0]:
                feat[i] = self.emb[j]

        # Canh: quan he ngu nghia giua cac node trong bai (dedup)
        heads, tails, etypes, edge_seen = [], [], [], set()
        for u in uris:
            for tl, rid in self.edges_by_head.get(u, ()):  # noqa
                if tl in uri_to_local:
                    key = (uri_to_local[u], uri_to_local[tl], rid)
                    if key in edge_seen:
                        continue
                    edge_seen.add(key)
                    heads.append(key[0]); tails.append(key[1]); etypes.append(rid)

        edge_index = (np.array([heads, tails], dtype=np.int64)
                      if heads else np.zeros((2, 0), dtype=np.int64))
        edge_type = np.array(etypes, dtype=np.int64) if etypes else np.zeros((0,), dtype=np.int64)
        return {
            "uris": uris, "node_feat": feat,
            "edge_index": edge_index, "edge_type": edge_type,
            "labels": [""] * len(uris),   # model khong dung labels
        }

    def subgraph_to_torch(self, sg: Dict[str, Any]):
        if not HAS_TORCH:
            raise RuntimeError("Can torch: pip install torch")
        return {
            "x":          torch.tensor(sg["node_feat"], dtype=torch.float32),
            "edge_index": torch.tensor(sg["edge_index"], dtype=torch.long),
            "edge_type":  torch.tensor(sg["edge_type"], dtype=torch.long),
            "uris":       sg["uris"],
            "labels":     sg["labels"],
        }
