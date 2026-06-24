# module5_kg_construction.py
"""
Module 5: Knowledge Graph Construction
Input : ./data/module3_entity_linked.jsonl   (entities + URIs)
        ./data/module4_triples.jsonl         (tất cả triple)

Output:
  ./data/kg/kg_global.ttl          RDF Turtle — lưu trữ chính thức
  ./data/kg/kg_networkx.pkl        NetworkX MultiDiGraph — tính toán graph
  ./data/kg/entity_index.pkl       dict: uri → {embedding, info, stats}
  ./data/kg/pykeen_triples.tsv     TSV 3 cột: head \t relation \t tail  (cho Module 7)
  ./data/kg/kg_stats.json          Thống kê tổng quan

Luồng xử lý:
  Pass 1 → Đọc module3_entity_linked.jsonl, xây entity_index
  Pass 2 → Đọc module4_triples.jsonl, xây RDF graph + NetworkX + TSV
  Pass 3 → Tính thống kê, lưu tất cả
"""
from __future__ import annotations

import json
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# rdflib
try:
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    LOGGER_TEMP = logging.getLogger("module5")
    LOGGER_TEMP.warning("rdflib không được cài — RDF output bị tắt. pip install rdflib")

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
LOGGER = logging.getLogger("module5_kg_construction")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)

# ──────────────────────────────────────────────────────────
# Namespace
# ──────────────────────────────────────────────────────────
BASE_ONTO    = "http://transmtl.vn/onto/"
BASE_ENT     = "http://transmtl.vn/ent/"
BASE_ARTICLE = "http://transmtl.vn/article/"
BASE_TOPIC   = "http://transmtl.vn/onto/topic/"
BASE_WD      = "http://www.wikidata.org/entity/"

# ──────────────────────────────────────────────────────────
# Quan hệ được bỏ qua khi export TSV cho KGE
# (metadata triple không mang thông tin quan hệ ngữ nghĩa)
# ──────────────────────────────────────────────────────────
SKIP_RELATIONS_FOR_KGE = {"rdf:type", "hasEntity"}


# ──────────────────────────────────────────────────────────
# Entity Index
# ──────────────────────────────────────────────────────────
class EntityIndex:
    """
    Lưu trữ thông tin từng entity:
      - uri → embedding (numpy)
      - uri → metadata (surface, label, uri_source, wikidata_id, count)
      - uri → surface_forms (set)

    Khi entity xuất hiện nhiều lần, embedding được cập nhật bằng
    running average (tránh lưu lại toàn bộ embedding).
    """

    def __init__(self):
        self._embeddings: Dict[str, np.ndarray] = {}
        self._counts:     Dict[str, int]         = {}
        self._info:       Dict[str, Dict[str, Any]] = {}
        self._surfaces:   Dict[str, set]          = {}

    # ── Cập nhật entity ─────────────────────────────────────
    def update(self, entity: Dict[str, Any]):
        uri = entity.get("uri")
        if not uri:
            return

        emb_list = entity.get("embedding")
        surface  = entity.get("surface") or entity.get("concept", "")

        # Running average embedding
        if emb_list is not None:
            emb = np.array(emb_list, dtype=np.float32)
            if uri in self._embeddings:
                n = self._counts.get(uri, 1)
                self._embeddings[uri] = (self._embeddings[uri] * n + emb) / (n + 1)
            else:
                self._embeddings[uri] = emb
            self._counts[uri] = self._counts.get(uri, 0) + 1

        # Metadata (lấy lần đầu tiên gặp)
        if uri not in self._info:
            self._info[uri] = {
                "uri":        uri,
                "label":      entity.get("label", "MISC"),
                "label_name": entity.get("label_name", ""),
                "uri_source": entity.get("uri_source", "new"),
                "wikidata_id": entity.get("wikidata_id"),
                "needs_review": entity.get("needs_review", False),
            }
        if uri not in self._surfaces:
            self._surfaces[uri] = set()
        if surface:
            self._surfaces[uri].add(surface)

    # ── Tra cứu ─────────────────────────────────────────────
    def get_embedding(self, uri: str) -> Optional[np.ndarray]:
        return self._embeddings.get(uri)

    def get_info(self, uri: str) -> Dict[str, Any]:
        info = self._info.get(uri, {})
        return {
            **info,
            "surface_forms": list(self._surfaces.get(uri, set())),
            "occurrence_count": self._counts.get(uri, 0),
        }

    @property
    def all_uris(self) -> List[str]:
        return list(self._info.keys())

    @property
    def n_entities(self) -> int:
        return len(self._info)

    # ── Lưu / tải ────────────────────────────────────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for uri in self._info:
            serializable[uri] = {
                "info":      self.get_info(uri),
                "embedding": self._embeddings[uri].tolist()
                             if uri in self._embeddings else None,
            }
        with open(path, "wb") as f:
            pickle.dump(serializable, f)
        LOGGER.info("EntityIndex saved: %d entities → %s", self.n_entities, path)

    @classmethod
    def load(cls, path: str) -> "EntityIndex":
        idx = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        for uri, item in data.items():
            info = item.get("info", {})
            idx._info[uri]    = info
            idx._surfaces[uri] = set(info.get("surface_forms", []))
            emb = item.get("embedding")
            if emb is not None:
                idx._embeddings[uri] = np.array(emb, dtype=np.float32)
        LOGGER.info("EntityIndex loaded: %d entities from %s", idx.n_entities, path)
        return idx


# ──────────────────────────────────────────────────────────
# RDF Builder
# ──────────────────────────────────────────────────────────
class RDFBuilder:
    """Xây dựng RDF graph bằng rdflib."""

    def __init__(self):
        if not HAS_RDFLIB:
            self.graph = None
            return

        self.graph = Graph()
        # Khai báo namespace
        self.ONTO    = Namespace(BASE_ONTO)
        self.ENT     = Namespace(BASE_ENT)
        self.ARTICLE = Namespace(BASE_ARTICLE)
        self.TOPIC   = Namespace(BASE_TOPIC)
        self.WD      = Namespace(BASE_WD)

        self.graph.bind("onto",    self.ONTO)
        self.graph.bind("ent",     self.ENT)
        self.graph.bind("article", self.ARTICLE)
        self.graph.bind("topic",   self.TOPIC)
        self.graph.bind("wd",      self.WD)
        self.graph.bind("owl",     OWL)
        self.graph.bind("rdf",     RDF)
        self.graph.bind("rdfs",    RDFS)

    def _to_uref(self, uri: str) -> Optional[URIRef]:
        """Chuyển URI string → rdflib URIRef."""
        if not uri or not uri.startswith("http"):
            return None
        try:
            return URIRef(uri)
        except Exception:
            return None

    def _relation_to_uref(self, relation: str) -> Optional[URIRef]:
        """Chuyển relation name → URIRef."""
        if not HAS_RDFLIB:
            return None
        if relation == "rdf:type":
            return RDF.type
        if ":" in relation:
            # Đã là prefixed URI
            return URIRef(relation)
        return URIRef(f"{BASE_ONTO}{relation}")

    def add_triple(self, head_uri: str, relation: str, tail_uri: str):
        if not HAS_RDFLIB or self.graph is None:
            return
        s = self._to_uref(head_uri)
        p = self._relation_to_uref(relation)
        o = self._to_uref(tail_uri)
        if s and p and o:
            self.graph.add((s, p, o))

    def add_label(self, uri: str, label_text: str, lang: str = "vi"):
        if not HAS_RDFLIB or self.graph is None:
            return
        s = self._to_uref(uri)
        if s and label_text:
            self.graph.add((s, RDFS.label, Literal(label_text, lang=lang)))

    def save(self, path: str):
        if not HAS_RDFLIB or self.graph is None:
            LOGGER.warning("rdflib không khả dụng — bỏ qua lưu RDF.")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=path, format="turtle")
        LOGGER.info("RDF saved: %d triples → %s", len(self.graph), path)

    @property
    def n_triples(self) -> int:
        if self.graph is None:
            return 0
        return len(self.graph)


# ──────────────────────────────────────────────────────────
# Module 5 — KG Builder
# ──────────────────────────────────────────────────────────
class KGBuilder:

    def __init__(self):
        self.entity_index  = EntityIndex()
        self.rdf_builder   = RDFBuilder()
        self.nx_graph      = nx.MultiDiGraph()  # cho phép nhiều cạnh giữa cùng 2 node
        self._triple_stats = Counter()           # relation → count

    # ── Pass 1: Xây entity index từ module3 ─────────────────
    def _build_entity_index(self, module3_jsonl: str):
        LOGGER.info("Pass 1: Building entity index from %s", module3_jsonl)
        n_articles = 0
        with open(module3_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    for ent in record.get("ner_entities", []):
                        self.entity_index.update(ent)
                    for con in record.get("concept_mentions", []):
                        # Concept cũng là entity trong KG
                        con_as_ent = {
                            **con,
                            "surface":    con.get("concept", ""),
                            "label":      "MISC",
                            "label_name": "Concept",
                        }
                        self.entity_index.update(con_as_ent)
                    n_articles += 1
                except Exception as e:
                    LOGGER.debug("Skip line in entity index build: %s", e)

        # Thêm label rdfs:label vào RDF
        for uri in self.entity_index.all_uris:
            info = self.entity_index.get_info(uri)
            surfaces = info.get("surface_forms", [])
            if surfaces:
                self.rdf_builder.add_label(uri, surfaces[0])

        LOGGER.info("Pass 1 done: %d entities from %d articles",
                    self.entity_index.n_entities, n_articles)

    # ── Pass 2: Đọc triple, nạp vào RDF + NetworkX + TSV ───
    def _build_graph_from_triples(
        self, triples_jsonl: str, tsv_writer
    ) -> int:
        LOGGER.info("Pass 2: Loading triples from %s", triples_jsonl)
        n_triples = 0

        with open(triples_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    head_uri = t.get("head_uri", "")
                    relation = t.get("relation", "")
                    tail_uri = t.get("tail_uri", "")

                    if not (head_uri and relation and tail_uri):
                        continue

                    # RDF
                    self.rdf_builder.add_triple(head_uri, relation, tail_uri)

                    # NetworkX
                    self.nx_graph.add_edge(
                        head_uri, tail_uri,
                        relation=relation,
                        confidence=t.get("confidence", 1.0),
                        method=t.get("method", "unknown"),
                        article_id=t.get("article_id", ""),
                    )
                    if not self.nx_graph.nodes[head_uri].get("label"):
                        self.nx_graph.nodes[head_uri]["label"]     = t.get("head_surface", "")
                        self.nx_graph.nodes[head_uri]["uri_source"] = self.entity_index.get_info(head_uri).get("uri_source", "")
                    if not self.nx_graph.nodes[tail_uri].get("label"):
                        self.nx_graph.nodes[tail_uri]["label"]     = t.get("tail_surface", "")

                    # TSV cho KGE (bỏ metadata triple)
                    if relation not in SKIP_RELATIONS_FOR_KGE:
                        tsv_writer.write(f"{head_uri}\t{relation}\t{tail_uri}\n")

                    self._triple_stats[relation] += 1
                    n_triples += 1

                except Exception as e:
                    LOGGER.debug("Skip triple line: %s", e)

        LOGGER.info("Pass 2 done: %d triples loaded", n_triples)
        return n_triples

    # ── Lưu tất cả output ────────────────────────────────────
    def _save_all(
        self,
        output_dir: str,
        n_triples: int,
    ):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. RDF Turtle
        self.rdf_builder.save(str(out / "kg_global.ttl"))

        # 2. NetworkX
        nx_path = out / "kg_networkx.pkl"
        with open(nx_path, "wb") as f:
            pickle.dump(self.nx_graph, f)
        LOGGER.info("NetworkX graph saved: %d nodes, %d edges → %s",
                    self.nx_graph.number_of_nodes(),
                    self.nx_graph.number_of_edges(),
                    nx_path)

        # 3. Entity index
        self.entity_index.save(str(out / "entity_index.pkl"))

        # 4. Thống kê
        wikidata_count = sum(
            1 for uri in self.entity_index.all_uris
            if self.entity_index.get_info(uri).get("uri_source") == "wikidata"
        )
        internal_count = sum(
            1 for uri in self.entity_index.all_uris
            if self.entity_index.get_info(uri).get("uri_source") == "internal"
        )
        new_count = sum(
            1 for uri in self.entity_index.all_uris
            if self.entity_index.get_info(uri).get("uri_source") == "new"
        )

        stats = {
            "n_entities":        self.entity_index.n_entities,
            "n_triples_total":   n_triples,
            "n_triples_rdf":     self.rdf_builder.n_triples,
            "n_nodes_nx":        self.nx_graph.number_of_nodes(),
            "n_edges_nx":        self.nx_graph.number_of_edges(),
            "entity_uri_source": {
                "wikidata": wikidata_count,
                "internal": internal_count,
                "new":      new_count,
            },
            "relation_counts": dict(self._triple_stats.most_common()),
        }
        stats_path = out / "kg_stats.json"
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("KG stats saved → %s", stats_path)
        self._print_summary(stats)

    @staticmethod
    def _print_summary(stats: Dict[str, Any]):
        print("\n" + "=" * 55)
        print("KG CONSTRUCTION SUMMARY")
        print("=" * 55)
        print(f"  Entities (nodes)    : {stats['n_entities']:,}")
        print(f"  Triples (edges)     : {stats['n_triples_total']:,}")
        print(f"  RDF triples         : {stats['n_triples_rdf']:,}")
        print(f"  NetworkX nodes      : {stats['n_nodes_nx']:,}")
        print(f"  NetworkX edges      : {stats['n_edges_nx']:,}")
        print()
        print("  Entity URI source:")
        for src, cnt in stats["entity_uri_source"].items():
            pct = cnt / max(stats["n_entities"], 1) * 100
            print(f"    {src:<12}: {cnt:,}  ({pct:.1f}%)")
        print()
        print("  Top 10 relations:")
        for rel, cnt in list(stats["relation_counts"].items())[:10]:
            print(f"    {rel:<25}: {cnt:,}")
        print("=" * 55)

    # ── Entry point ──────────────────────────────────────────
    def build(
        self,
        module3_jsonl:  str,
        module4_triples_jsonl: str,
        output_dir:     str = "./data/kg",
    ):
        tsv_path = Path(output_dir) / "pykeen_triples.tsv"
        tsv_path.parent.mkdir(parents=True, exist_ok=True)

        # Pass 1
        self._build_entity_index(module3_jsonl)

        # Pass 2 + stream TSV
        with open(tsv_path, "w", encoding="utf-8") as tsv_writer:
            tsv_writer.write("head\trelation\ttail\n")   # header
            n_triples = self._build_graph_from_triples(
                module4_triples_jsonl, tsv_writer
            )

        LOGGER.info("PyKEEN TSV saved → %s", tsv_path)

        # Lưu tất cả
        self._save_all(output_dir, n_triples)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    DATA = os.environ.get("OKG_DATA_DIR", "./data")
    builder = KGBuilder()
    builder.build(
        module3_jsonl=os.path.join(DATA, "module3_entity_linked.jsonl"),
        module4_triples_jsonl=os.path.join(DATA, "module4_triples.jsonl"),
        output_dir=os.path.join(DATA, "kg"),
    )