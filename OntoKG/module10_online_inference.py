# module10_online_inference.py
"""
Module 10: Bổ trợ tri thức ĐỘNG tại thời điểm test cho TransMTL.

Mục tiêu: hỗ trợ HAI KỊCH BẢN đánh giá mà người dùng yêu cầu, với cùng một
checkpoint TransMTL + cùng một OntoKG đã huấn luyện trên train+val:

  • OFFLINE (KG KHÔNG được cập nhật từ test):
      - Trích entity từ văn bản test -> LIÊN KẾT vào KG train+val (CHỈ ĐỌC).
      - Dựng subgraph chỉ gồm các entity ĐÃ CÓ trong KG + quan hệ ĐÃ CÓ trong KG
        giữa chúng. KHÔNG thêm entity/quan hệ mới.
      - Đo "tri thức tĩnh có sẵn hỗ trợ được bao nhiêu".

  • ONLINE (KG được cập nhật từ chính văn bản đầu vào):
      - Như trên, NHƯNG bổ sung thêm: entity MỚI (chưa có trong KG) được thêm vào
        subgraph với embedding PhoBERT (inductive), và các quan hệ MỚI trích từ
        văn bản test được thêm vào subgraph để bổ trợ TransMTL.
      - Đo "cập nhật tri thức động hỗ trợ được bao nhiêu".

Thiết kế TÁI DÙNG các module sẵn có (không viết lại NER/linking/relation):
  Module 2 (NER + concept)  -> Module 3 (entity linking) -> Module 4 (relation)
  Module 7 (entity_embeddings.pt) qua LocalKGRetriever cho feature/quan hệ KG.

LƯU Ý: module này nạp mô hình nặng (PhoBERT NER + embedding). Chạy trên máy có
GPU. Chưa được chạy thử trong môi trường sinh code — cần kiểm thử trên máy bạn.

Dependencies: torch, transformers, underthesea (giống Module 2/3/4).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from OntoKG.module9_local_retrieval import (
    LocalKGRetriever, RELATION_TO_ID, SEMANTIC_RELATIONS,
)

LOGGER = logging.getLogger("module10_online_inference")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(_h)

# NER model mặc định (khớp Module 2 __main__).
DEFAULT_NER_MODEL = "NlpHUST/ner-vietnamese-electra-base"
DEFAULT_EMB_MODEL = "vinai/phobert-base-v2"


def _rebuild_registry_emb_index(registry) -> None:
    """EntityRegistry.load() KHÔNG dựng lại chỉ mục vector (_emb_buf) -> tầng 3
    (embedding match) sẽ không hoạt động. Hàm này dựng lại từ registry.embeddings."""
    try:
        from OntoKG.module3_entity_linking import EMB_MATCH_LABELS
    except Exception:
        EMB_MATCH_LABELS = {"PER", "ORG", "LOC"}
    n = 0
    for uri, emb in getattr(registry, "embeddings", {}).items():
        info = registry.uri_info.get(uri, {})
        label = info.get("label", "MISC")
        if label in EMB_MATCH_LABELS:
            try:
                registry._emb_add(uri, label, np.asarray(emb, dtype=np.float32))
                n += 1
            except Exception:
                pass
    LOGGER.info("Rebuilt registry embedding index: %d vectors", n)


class OnlineKGAugmenter:
    """Trích + liên kết entity/quan hệ từ văn bản test rồi dựng subgraph cho TransMTL.

    mode = 'offline' (chỉ đọc KG) hoặc 'online' (bổ sung entity/quan hệ mới).
    """

    def __init__(
        self,
        data_dir: str,
        entity_emb_path: str,
        entity_idx_path: str,
        device: str = "cuda",
        ner_model_name: str = DEFAULT_NER_MODEL,
        emb_model_name: str = DEFAULT_EMB_MODEL,
        use_wikidata: bool = False,          # tắt mặc định: nhanh + tất định khi test
        use_phonlp: bool = False,
        similarity_threshold: float = 0.92,
        embedding_dim: int = 768,
    ):
        if not HAS_TORCH:
            raise RuntimeError("Cần torch: pip install torch")
        self.embedding_dim = embedding_dim
        data_dir = str(data_dir)

        # ── 1) KG tĩnh (train+val): embedding + uri->idx + quan hệ ngữ nghĩa ──
        triples_jsonl = str(Path(entity_emb_path).resolve().parent.parent / "module4_triples.jsonl")
        self.kg = LocalKGRetriever(
            entity_emb_path=entity_emb_path, entity_idx_path=entity_idx_path,
            triples_jsonl=triples_jsonl, embedding_dim=embedding_dim,
        )
        self.kg_emb = self.kg.emb               # (N, 768) đã train (Module 7)
        self.kg_uri_to_idx = self.kg.uri_to_idx
        self.edges_by_head = self.kg.edges_by_head

        # ── 2) Module 2: NER + concept ────────────────────────
        from OntoKG.module2_ner_concept import Module2NerConceptExtractor
        ontology_concepts = self._load_ontology_concepts(data_dir)
        self.m2 = Module2NerConceptExtractor(
            ontology_concepts=ontology_concepts,
            use_underthesea_ner=True,
            use_phobert_ner=True,
            phobert_model_name=ner_model_name,
            use_phonlp=use_phonlp,
            concept_embedding_model_name=emb_model_name,
        )

        # ── 3) Module 3: entity linking, NẠP registry train+val (chỉ đọc) ──
        from OntoKG.module3_entity_linking import Module3EntityLinker, EntityRegistry
        self.m3 = Module3EntityLinker(
            embedding_model=emb_model_name, device=device,
            wikidata_cache_path=os.path.join(data_dir, "wikidata_cache.json"),
            similarity_threshold=similarity_threshold,
            use_wikidata=use_wikidata, use_embedding_matching=True,
        )
        reg_path = os.path.join(data_dir, "entity_registry.pkl")
        if os.path.exists(reg_path):
            self.m3.registry = EntityRegistry.load(reg_path)
            _rebuild_registry_emb_index(self.m3.registry)
        else:
            LOGGER.warning("Không thấy %s — linking sẽ tạo URI mới, ít khớp KG.", reg_path)

        # ── 4) Module 4: relation extraction ───────────────────
        from OntoKG.module4_relation_extraction import Module4RelationExtractor
        self.m4 = Module4RelationExtractor()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_ontology_concepts(data_dir: str) -> List[Dict[str, Any]]:
        path = Path(data_dir) / "ontology" / "ontology_v1.1.json"
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get("concepts", []) or data.get("classes", []) or []
            if isinstance(data, list):
                return data
        except Exception as e:
            LOGGER.warning("Không đọc được ontology concepts: %s", e)
        return []

    # ------------------------------------------------------------------ #
    #  Trích + liên kết 1 văn bản -> (entities đã link, triples)          #
    # ------------------------------------------------------------------ #
    def process_text(self, text: str, title: str = "", topic_list: Optional[List[str]] = None) -> Dict[str, Any]:
        record = {
            "article_id": "__online__",
            "title": title or "",
            "summary": "",
            "content": text or "",
            "full_text": text or "",
            "topic_list": topic_list or [],
            "cleaned_keywords": [],
        }
        r2 = self.m2.process_record(record)        # + ner_entities, concept_mentions
        r3 = self.m3.process_record(r2)            # + uri, embedding cho mỗi entity
        _, triples = self.m4.process_record(r3)    # quan hệ (Triple)
        return {"linked": r3, "triples": triples}

    # ------------------------------------------------------------------ #
    #  Dựng subgraph cho 1 văn bản theo mode                              #
    # ------------------------------------------------------------------ #
    def build_subgraph(self, processed: Dict[str, Any], mode: str = "offline") -> Optional[Dict[str, Any]]:
        linked = processed["linked"]
        triples = processed["triples"]

        # Gom entity (NER + concept), dedupe theo URI, giữ embedding PhoBERT nếu có.
        ent_emb: Dict[str, Optional[np.ndarray]] = {}
        order: List[str] = []
        for ent in list(linked.get("ner_entities", [])) + list(linked.get("concept_mentions", [])):
            uri = ent.get("uri")
            if not uri:
                continue
            if uri not in ent_emb:
                emb = ent.get("embedding")
                ent_emb[uri] = (np.asarray(emb, dtype=np.float32)
                                if emb is not None else None)
                order.append(uri)

        known = {u for u in order if u in self.kg_uri_to_idx}

        if mode == "offline":
            node_uris = [u for u in order if u in known]      # chỉ entity đã có trong KG
        else:  # online
            node_uris = list(order)                            # known + new

        if not node_uris:
            return None

        uri_to_local = {u: i for i, u in enumerate(node_uris)}

        # Node features (768): known -> embedding KGE đã train; new -> PhoBERT; else 0.
        feat = np.zeros((len(node_uris), self.embedding_dim), dtype=np.float32)
        for u, i in uri_to_local.items():
            if u in self.kg_uri_to_idx:
                j = self.kg_uri_to_idx[u]
                if 0 <= j < self.kg_emb.shape[0]:
                    feat[i] = self.kg_emb[j]
            elif ent_emb.get(u) is not None and ent_emb[u].shape[0] == self.embedding_dim:
                feat[i] = ent_emb[u]

        heads, tails, etypes, seen = [], [], [], set()

        def _add_edge(h_uri, t_uri, rid):
            if h_uri in uri_to_local and t_uri in uri_to_local:
                key = (uri_to_local[h_uri], uri_to_local[t_uri], rid)
                if key not in seen:
                    seen.add(key)
                    heads.append(key[0]); tails.append(key[1]); etypes.append(rid)

        # (a) Quan hệ ĐÃ CÓ trong KG giữa các node known (cả offline & online).
        for u in node_uris:
            if u in known:
                for tl, rid in self.edges_by_head.get(u, ()):  # noqa
                    _add_edge(u, tl, rid)

        # (b) ONLINE: thêm quan hệ MỚI trích từ chính văn bản test.
        if mode == "online":
            for tr in triples:
                rel = getattr(tr, "relation", None) if not isinstance(tr, dict) else tr.get("relation")
                h_uri = getattr(tr, "head_uri", None) if not isinstance(tr, dict) else tr.get("head_uri")
                t_uri = getattr(tr, "tail_uri", None) if not isinstance(tr, dict) else tr.get("tail_uri")
                if rel in RELATION_TO_ID:
                    _add_edge(h_uri, t_uri, RELATION_TO_ID[rel])

        edge_index = (np.array([heads, tails], dtype=np.int64)
                      if heads else np.zeros((2, 0), dtype=np.int64))
        edge_type = np.array(etypes, dtype=np.int64) if etypes else np.zeros((0,), dtype=np.int64)
        return {
            "uris": node_uris, "node_feat": feat,
            "edge_index": edge_index, "edge_type": edge_type,
            "labels": [""] * len(node_uris),
        }

    def subgraph_to_torch(self, sg: Dict[str, Any]):
        return self.kg.subgraph_to_torch(sg)

    # ------------------------------------------------------------------ #
    #  Dựng kg_batch cho cả batch văn bản (dùng trong tester)            #
    # ------------------------------------------------------------------ #
    def build_kg_batch(self, texts: List[str], mode: str = "offline") -> List[Optional[Dict[str, Any]]]:
        kg_batch: List[Optional[Dict[str, Any]]] = []
        for txt in texts:
            try:
                processed = self.process_text(txt or "")
                sg = self.build_subgraph(processed, mode=mode)
                kg_batch.append(self.subgraph_to_torch(sg) if sg is not None else None)
            except Exception as e:
                LOGGER.debug("Online augment failed for 1 doc: %s", e)
                kg_batch.append(None)
        return kg_batch

    def close(self):
        try:
            self.kg.close()
        except Exception:
            pass
