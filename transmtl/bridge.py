# ontokg_data_bridge.py
"""
Cau noi giua dataset TransMTL va OntoKG (Neo4j).
Mat xich con thieu de tich hop: tu article_id -> truy van Neo4j -> kg_batch.

Cach dung trong transmtl/train.py / transmtl/tester.py:

    from transmtl.bridge import OntoKGBridge

    bridge = OntoKGBridge(
        uri="bolt://localhost:7687", user="neo4j", password="password",
        d_model=300, enabled=use_ontokg,
    )

    # Trong vong lap batch (dataset da tra them article_ids):
    kg_batch = bridge.build_kg_batch(article_ids)   # None neu enabled=False
    out = model(inp=src, tar=tgt, labels=labels, task="both",
                training=True, kg_batch=kg_batch)

    # Cuoi chuong trinh:
    bridge.close()
"""
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class OntoKGBridge:
    """Truy van subgraph tu Neo4j cho ca batch, tra ve kg_batch cho TransMTL."""

    def __init__(self, uri, user, password, d_model=300,
                 database="neo4j", enabled=True,
                 backend="local", entity_emb_path=None, entity_idx_path=None):
        """backend = 'local' (doc file, KHONG can Neo4j/Docker) hoac 'neo4j'."""
        self.enabled = enabled
        self.d_model = d_model
        self.retriever = None
        if not enabled:
            return

        if backend == "local":
            # KHONG can Neo4j: doc thang artifact tren dia (Module 4 + 7).
            from OntoKG.module9_local_retrieval import LocalKGRetriever
            if not entity_emb_path or not entity_idx_path:
                raise ValueError(
                    "backend='local' can entity_emb_path + entity_idx_path "
                    "(vd data/kge/entity_embeddings.pt, data/kge/entity_to_idx.json)")
            # module4_triples.jsonl nam o thu muc data goc = cha cua kge/
            data_dir = Path(entity_emb_path).resolve().parent.parent
            triples_jsonl = data_dir / "module4_triples.jsonl"
            self.retriever = LocalKGRetriever(
                entity_emb_path=entity_emb_path, entity_idx_path=entity_idx_path,
                triples_jsonl=str(triples_jsonl), embedding_dim=768,
            )
        else:
            # Import o day de backend local khong can goi neo4j
            from OntoKG.module9_neo4j_retrieval import Neo4jRetriever
            # LUU Y: entity embedding trong Neo4j la 768 chieu (tu Module 7),
            # GraphEncoder se chieu 768 -> d_model. Nen retriever giu dim=768.
            self.retriever = Neo4jRetriever(
                uri=uri, user=user, password=password,
                database=database, embedding_dim=768,
            )

    def build_kg_batch(self, article_ids: List[str]) -> Optional[List[Optional[Dict[str, Any]]]]:
        """
        article_ids: list article_id cua ca batch.
        Tra ve list subgraph (moi sample 1 phan tu) hoac None neu OntoKG tat.
        """
        if not self.enabled or self.retriever is None:
            return None

        kg_batch = []
        for aid in article_ids:
            try:
                sg = self.retriever.get_article_subgraph(aid)
                if sg["uris"]:
                    kg_batch.append(self.retriever.subgraph_to_torch(sg))
                else:
                    kg_batch.append(None)
            except Exception:
                kg_batch.append(None)
        return kg_batch

    def close(self):
        if self.retriever is not None:
            self.retriever.close()