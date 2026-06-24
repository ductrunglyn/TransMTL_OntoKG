# module7_kge_training.py
"""
Module 7: Knowledge Graph Embedding Training → entity_embeddings.pt (N × 768)
Input : ./data/kg/pykeen_triples.tsv     (output Module 5)
        ./data/kg/entity_index.pkl       (output Module 5)

Output:
  ./data/kge/entity_embeddings.pt      Tensor (N × 768) — ĐẦU VÀO TRANSMTL
  ./data/kge/entity_to_idx.json        URI → integer index (để lookup)
  ./data/kge/idx_to_uri.json           integer index → URI (ngược lại)
  ./data/kge/kge_model/                PyKEEN model checkpoint
  ./data/kge/training_results.json     Metrics: MRR, Hits@10

Kiến trúc:
  - Train TransE với embedding_dim=256 (hiệu quả hơn so với 768)
  - Project lên 768 chiều bằng linear layer đã train
  - Kết hợp với PhoBERT surface embedding (50% KGE + 50% PhoBERT)
  → entity_embeddings.pt shape (N, 768)

Tại sao không train TransE dim=768 trực tiếp?
  - TransE dim=768 với N=50K entities: bộ nhớ 50K×768×4B ≈ 150MB (OK)
  - Nhưng training TransE với dim=768 hội tụ chậm hơn và không ổn định
  - Tốt hơn: train dim=256 (nhanh, ổn định) → project lên 768
  - Projection W: (256 → 768) học được trong quá trình train

Nếu muốn dim=768 trực tiếp:
  Đặt KGE_EMBEDDING_DIM = 768 và PROJECTION_DIM = None

Dependencies: pip install pykeen torch
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pykeen.pipeline import pipeline as pykeen_pipeline
    from pykeen.triples import TriplesFactory
    HAS_PYKEEN = True
except ImportError:
    HAS_PYKEEN = False

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
LOGGER = logging.getLogger("module7_kge_training")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)

# ──────────────────────────────────────────────────────────
# Cấu hình
# ──────────────────────────────────────────────────────────
KGE_EMBEDDING_DIM = 256   # dimension train TransE (hiệu quả)
PROJECTION_DIM    = 768   # dimension đầu ra (khớp PhoBERT)
TARGET_DIM        = 768   # dimension cuối cùng của entity_embeddings.pt

KGE_MODEL         = "TransE"   # hoặc "RotatE" (chính xác hơn, chậm hơn)
KGE_EPOCHS        = 200
KGE_BATCH_SIZE    = 1024
KGE_LR            = 0.001
KGE_MARGIN        = 1.0
SPLIT_RATIOS      = [0.85, 0.075, 0.075]   # train/val/test
RANDOM_SEED       = 42

# Ngưỡng minimum triples để train (nếu ít hơn → không đủ dữ liệu)
MIN_TRIPLES = 100


# ──────────────────────────────────────────────────────────
# Tiện ích load entity index
# ──────────────────────────────────────────────────────────
def load_entity_index(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_phobert_embeddings_from_index(
    entity_data: Dict[str, Any],
    uris: List[str],
    dim: int = 768,
) -> np.ndarray:
    """
    Lấy PhoBERT surface embedding từ entity_index cho các URI.
    Trả về ma trận (N, dim). URI không có embedding → zero vector.
    """
    embs = np.zeros((len(uris), dim), dtype=np.float32)
    for i, uri in enumerate(uris):
        item = entity_data.get(uri)
        if item is None:
            continue
        emb = item.get("embedding")
        if emb is None:
            continue
        arr = np.array(emb, dtype=np.float32)
        if arr.shape[0] == dim:
            embs[i] = arr
    return embs


# ──────────────────────────────────────────────────────────
# Projection Layer (256 → 768)
# ──────────────────────────────────────────────────────────
class LinearProjection(nn.Module):
    """
    Ánh xạ tuyến tính kge_dim → target_dim.
    Được khởi tạo bằng SVD của PhoBERT embeddings để warm-start tốt.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        # Khởi tạo chuẩn Xavier
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def train_projection(
    kge_embs: np.ndarray,     # (N, 256) — TransE embeddings
    phobert_embs: np.ndarray, # (N, 768) — PhoBERT surface embeddings
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> LinearProjection:
    """
    Train Linear Projection: KGE_DIM → PHOBERT_DIM
    Mục tiêu: project KGE embedding gần với PhoBERT embedding nhất có thể.
    Chỉ train trên entity CÓ cả 2 loại embedding (không zero vector).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch cần được cài: pip install torch")

    # Lọc entity có cả 2 embedding
    phobert_norms = np.linalg.norm(phobert_embs, axis=1)
    valid_mask    = phobert_norms > 1e-6
    n_valid       = valid_mask.sum()

    LOGGER.info("Training projection on %d / %d entities with PhoBERT embeddings",
                n_valid, len(kge_embs))

    if n_valid < 10:
        LOGGER.warning(
            "Quá ít entity có PhoBERT embedding (%d). "
            "Projection sẽ không chính xác. Kiểm tra entity_index.pkl.",
            n_valid,
        )

    proj = LinearProjection(kge_embs.shape[1], phobert_embs.shape[1]).to(device)
    opt  = torch.optim.Adam(proj.parameters(), lr=lr)

    if n_valid > 0:
        X = torch.tensor(kge_embs[valid_mask], dtype=torch.float32, device=device)
        Y = torch.tensor(phobert_embs[valid_mask], dtype=torch.float32, device=device)
        # L2 normalize targets
        Y = nn.functional.normalize(Y, dim=1)

        for epoch in range(n_epochs):
            proj.train()
            opt.zero_grad()
            out  = proj(X)
            out  = nn.functional.normalize(out, dim=1)
            # Cosine embedding loss (tối thiểu hoá khoảng cách)
            loss = (1 - (out * Y).sum(dim=1)).mean()
            loss.backward()
            opt.step()

            if (epoch + 1) % 20 == 0:
                LOGGER.info("Projection epoch %d/%d — loss=%.4f", epoch + 1, n_epochs, loss.item())

    return proj


# ──────────────────────────────────────────────────────────
# Kết hợp KGE + PhoBERT
# ──────────────────────────────────────────────────────────
def combine_embeddings(
    kge_projected: np.ndarray,    # (N, 768) KGE sau projection
    phobert_embs: np.ndarray,     # (N, 768) PhoBERT surface
    alpha: float = 0.5,           # trọng số KGE (1-alpha = PhoBERT)
) -> np.ndarray:
    """
    Kết hợp có trọng số:
      final = alpha * KGE_projected + (1 - alpha) * PhoBERT_surface

    Với entity không có PhoBERT embedding (zero vector), dùng 100% KGE.
    Với entity không có KGE embedding (zero vector), dùng 100% PhoBERT.
    """
    phobert_norms = np.linalg.norm(phobert_embs, axis=1, keepdims=True)
    kge_norms     = np.linalg.norm(kge_projected, axis=1, keepdims=True)

    has_phobert = (phobert_norms > 1e-6).flatten()
    has_kge     = (kge_norms     > 1e-6).flatten()

    # Normalize cả hai trước khi kết hợp
    phobert_norm = phobert_embs / np.where(phobert_norms > 1e-6, phobert_norms, 1.0)
    kge_norm     = kge_projected / np.where(kge_norms     > 1e-6, kge_norms,     1.0)

    combined = np.zeros_like(kge_projected)

    both_mask  = has_kge & has_phobert
    kge_only   = has_kge & ~has_phobert
    bert_only  = ~has_kge & has_phobert

    combined[both_mask]  = alpha * kge_norm[both_mask] + (1 - alpha) * phobert_norm[both_mask]
    combined[kge_only]   = kge_norm[kge_only]
    combined[bert_only]  = phobert_norm[bert_only]

    # L2 normalize final
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.where(norms > 1e-6, norms, 1.0)

    LOGGER.info(
        "Combined: both=%d  kge_only=%d  phobert_only=%d  zero=%d",
        both_mask.sum(), kge_only.sum(), bert_only.sum(),
        (~has_kge & ~has_phobert).sum(),
    )
    return combined.astype(np.float32)


# ──────────────────────────────────────────────────────────
# Module 7 — Main
# ──────────────────────────────────────────────────────────
class Module7KGETrainer:

    def __init__(
        self,
        kge_model:      str   = KGE_MODEL,
        kge_dim:        int   = KGE_EMBEDDING_DIM,
        target_dim:     int   = TARGET_DIM,
        n_epochs:       int   = KGE_EPOCHS,
        batch_size:     int   = KGE_BATCH_SIZE,
        lr:             float = KGE_LR,
        margin:         float = KGE_MARGIN,
        combine_alpha:  float = 0.5,
        device:         str   = "cuda",
        random_seed:    int   = RANDOM_SEED,
    ):
        self.kge_model     = kge_model
        self.kge_dim       = kge_dim
        self.target_dim    = target_dim
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.margin        = margin
        self.combine_alpha = combine_alpha
        self.device        = device if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self.random_seed   = random_seed

    # ── Step 1: Train TransE ─────────────────────────────
    def _train_kge(
        self,
        triples_tsv: str,
        checkpoint_dir: str,
    ) -> Tuple[Any, TriplesFactory]:
        """
        Train KGE với PyKEEN pipeline.
        Trả về (result, full_triples_factory).
        """
        if not HAS_PYKEEN:
            raise RuntimeError("PyKEEN cần được cài: pip install pykeen")

        LOGGER.info("Loading triples from %s", triples_tsv)
        full_tf = TriplesFactory.from_path(
            triples_tsv,
            create_inverse_triples=False,
        )
        n_triples  = full_tf.num_triples
        n_entities = full_tf.num_entities
        LOGGER.info("Triples: %d  |  Entities: %d  |  Relations: %d",
                    n_triples, n_entities, full_tf.num_relations)

        if n_triples < MIN_TRIPLES:
            raise ValueError(
                f"Chỉ có {n_triples} triple — quá ít để train KGE. "
                f"Cần ít nhất {MIN_TRIPLES}."
            )

        train_tf, val_tf, test_tf = full_tf.split(
            SPLIT_RATIOS, random_state=self.random_seed
        )
        LOGGER.info("Split: train=%d  val=%d  test=%d",
                    train_tf.num_triples, val_tf.num_triples, test_tf.num_triples)

        LOGGER.info("Training %s (dim=%d, epochs=%d) on %s ...",
                    self.kge_model, self.kge_dim, self.n_epochs, self.device)

        # ĐoạN MỚI — tương thích mọi version PyKEEN
        result = pykeen_pipeline(
            model=self.kge_model,
            training=train_tf,
            validation=val_tf,
            testing=test_tf,
            model_kwargs={"embedding_dim": self.kge_dim},
            training_kwargs={
                "num_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "use_tqdm_batch": False,
            },
            optimizer="Adam",
            optimizer_kwargs={"lr": self.lr},
            loss="MarginRankingLoss",
            loss_kwargs={"margin": self.margin},
            negative_sampler="basic",
            stopper="early",
            stopper_kwargs={
                "patience": 10,
                "frequency": 5,          # kiểm tra mỗi 5 epoch
                "relative_delta": 0.001, # dùng relative_delta thay vì delta
            },
            evaluator_kwargs={"filtered": True},
            device=self.device,
            random_seed=self.random_seed,
            result_tracker=None,
        )

        # Lưu checkpoint
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        result.save_to_directory(checkpoint_dir)
        LOGGER.info("KGE checkpoint saved → %s", checkpoint_dir)

        return result, full_tf

    # ── Step 2: Extract & project embedding ──────────────
    def _extract_and_project(
        self,
        result: Any,
        full_tf: TriplesFactory,
        entity_index_pkl: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Trích xuất TransE embedding → project lên target_dim → kết hợp PhoBERT.
        Trả về (combined_matrix, ordered_uris).
        """
        # Lấy danh sách URI theo thứ tự index của PyKEEN
        idx_to_uri: Dict[int, str] = {
            v: k for k, v in full_tf.entity_to_id.items()
        }
        n = full_tf.num_entities
        ordered_uris = [idx_to_uri[i] for i in range(n)]

        # TransE entity embeddings (N, kge_dim)
        kge_embs = (
            result.model.entity_representations[0](indices=None)
            .detach()
            .cpu()
            .numpy()
        )
        LOGGER.info("KGE embedding matrix: %s", kge_embs.shape)

        # Load PhoBERT surface embeddings từ entity_index
        entity_data  = load_entity_index(entity_index_pkl)
        phobert_embs = get_phobert_embeddings_from_index(
            entity_data, ordered_uris, dim=self.target_dim
        )
        LOGGER.info("PhoBERT embeddings: %s (zero rows = no embedding)",
                    phobert_embs.shape)

        # Train projection: kge_dim → target_dim
        if self.kge_dim != self.target_dim:
            LOGGER.info("Training projection %d → %d ...", self.kge_dim, self.target_dim)
            proj = train_projection(
                kge_embs, phobert_embs,
                n_epochs=300, lr=1e-3, device=self.device,
            )
            proj.eval()
            with torch.no_grad():
                kge_tensor   = torch.tensor(kge_embs, dtype=torch.float32,
                                            device=self.device)
                kge_projected = proj(kge_tensor).cpu().numpy()
        else:
            # Không cần project
            kge_projected = kge_embs

        # Kết hợp KGE + PhoBERT
        combined = combine_embeddings(
            kge_projected, phobert_embs, alpha=self.combine_alpha
        )
        LOGGER.info("Final combined embedding: %s", combined.shape)
        return combined, ordered_uris

    # ── Step 3: Lưu output ───────────────────────────────
    def _save_outputs(
        self,
        combined: np.ndarray,
        ordered_uris: List[str],
        result: Any,
        output_dir: str,
    ):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. entity_embeddings.pt — đây là đầu ra chính
        emb_tensor = torch.tensor(combined, dtype=torch.float32)
        torch.save(emb_tensor, out / "entity_embeddings.pt")
        LOGGER.info("entity_embeddings.pt saved: shape=%s → %s",
                    tuple(emb_tensor.shape), out / "entity_embeddings.pt")

        # 2. Mapping URI ↔ index
        uri_to_idx = {uri: i for i, uri in enumerate(ordered_uris)}
        idx_to_uri = {i: uri for i, uri in enumerate(ordered_uris)}

        (out / "entity_to_idx.json").write_text(
            json.dumps(uri_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "idx_to_uri.json").write_text(
            json.dumps({str(k): v for k, v in idx_to_uri.items()},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("Mappings saved: entity_to_idx.json + idx_to_uri.json")

        # 3. Training results
        metrics = {}
        try:
            metrics = {
                "mrr_test":     result.get_metric("mean_reciprocal_rank"),
                "hits_at_1":    result.get_metric("hits_at_1"),
                "hits_at_10":   result.get_metric("hits_at_10"),
            }
        except Exception:
            metrics = {"note": "Metrics không khả dụng từ PyKEEN result"}

        training_info = {
            "model":        self.kge_model,
            "kge_dim":      self.kge_dim,
            "target_dim":   self.target_dim,
            "n_epochs":     self.n_epochs,
            "combine_alpha": self.combine_alpha,
            "n_entities":   len(ordered_uris),
            "embedding_shape": list(emb_tensor.shape),
            "metrics":      metrics,
        }
        (out / "training_results.json").write_text(
            json.dumps(training_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("Training results → %s", out / "training_results.json")

        self._print_summary(training_info, metrics)

    @staticmethod
    def _print_summary(info: Dict[str, Any], metrics: Dict[str, Any]):
        print("\n" + "=" * 55)
        print("MODULE 7 — KGE TRAINING COMPLETE")
        print("=" * 55)
        print(f"  Model              : {info['model']}")
        print(f"  KGE dimension      : {info['kge_dim']}")
        print(f"  Output dimension   : {info['target_dim']}")
        print(f"  N entities         : {info['n_entities']:,}")
        print(f"  Embedding shape    : {info['embedding_shape']}")
        print(f"  Combine alpha      : {info['combine_alpha']} (KGE) / "
              f"{1 - info['combine_alpha']} (PhoBERT)")
        print()
        print("  KGE Metrics (test set):")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:<20}: {v:.4f}")
            else:
                print(f"    {k:<20}: {v}")
        print()
        print("  Output files:")
        print("    entity_embeddings.pt  ← ĐẦU VÀO TRANSMTL")
        print("    entity_to_idx.json    ← URI → index (dùng khi lookup)")
        print("    idx_to_uri.json       ← index → URI")
        print("    kge_model/            ← PyKEEN checkpoint")
        print("=" * 55)

    # ── Entry point ──────────────────────────────────────
    def train(
        self,
        triples_tsv:      str,
        entity_index_pkl: str,
        output_dir:       str = "./data/kge",
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch cần được cài: pip install torch")
        if not HAS_PYKEEN:
            raise RuntimeError("PyKEEN cần được cài: pip install pykeen")

        checkpoint_dir = str(Path(output_dir) / "kge_model")

        # Step 1: Train KGE
        result, full_tf = self._train_kge(triples_tsv, checkpoint_dir)

        # Step 2: Extract + project + combine
        combined, ordered_uris = self._extract_and_project(
            result, full_tf, entity_index_pkl
        )

        # Step 3: Save
        self._save_outputs(combined, ordered_uris, result, output_dir)

        return combined, ordered_uris


# ──────────────────────────────────────────────────────────
# Hàm tiện ích dùng trong TransMTL (inference)
# ──────────────────────────────────────────────────────────
def load_entity_embeddings(
    embeddings_pt: str,
    entity_to_idx_json: str,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Load entity_embeddings.pt và mapping URI → index.
    Dùng trong TransMTL khi cần tra cứu embedding cho entity.

    Ví dụ:
        emb_matrix, uri_to_idx = load_entity_embeddings(
            "./data/kge/entity_embeddings.pt",
            "./data/kge/entity_to_idx.json",
        )
        # Tra cứu embedding cho URI
        idx = uri_to_idx.get("http://www.wikidata.org/entity/Q1748")
        if idx is not None:
            ha_noi_emb = emb_matrix[idx]  # tensor (768,)
    """
    emb = torch.load(embeddings_pt, map_location="cpu")
    with open(entity_to_idx_json, "r", encoding="utf-8") as f:
        uri_to_idx = json.load(f)
    LOGGER.info("Loaded entity_embeddings: shape=%s, n_uris=%d",
                tuple(emb.shape), len(uri_to_idx))
    return emb, uri_to_idx


def get_entity_embedding(
    uri: str,
    emb_matrix: torch.Tensor,
    uri_to_idx: Dict[str, int],
    fallback_dim: int = 768,
) -> torch.Tensor:
    """
    Tra cứu embedding cho một URI.
    Nếu URI không có trong index → trả về zero vector (entity chưa biết).

    Trong TransMTL, nếu gặp zero vector, model sẽ dùng PhoBERT embedding
    của surface form thay thế (xử lý ở gated fusion).
    """
    idx = uri_to_idx.get(uri)
    if idx is not None:
        return emb_matrix[idx]
    return torch.zeros(fallback_dim)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    DATA = os.environ.get("OKG_DATA_DIR", "./data")
    device = os.environ.get("OKG_DEVICE", "cuda")
    trainer = Module7KGETrainer(
        kge_model="TransE",
        kge_dim=256,          # train efficient
        target_dim=768,       # output khớp PhoBERT
        n_epochs=200,
        batch_size=1024,
        lr=0.001,
        margin=1.0,
        combine_alpha=0.5,    # 50% KGE + 50% PhoBERT surface
        device=device,
    )

    trainer.train(
        triples_tsv=os.path.join(DATA, "kg", "pykeen_triples.tsv"),
        entity_index_pkl=os.path.join(DATA, "kg", "entity_index.pkl"),
        output_dir=os.path.join(DATA, "kge"),
    )

    # Verify output
    emb_matrix, uri_to_idx = load_entity_embeddings(
        os.path.join(DATA, "kge", "entity_embeddings.pt"),
        os.path.join(DATA, "kge", "entity_to_idx.json"),
    )
    print(f"\nVerify: entity_embeddings.pt shape = {tuple(emb_matrix.shape)}")
    print(f"        Số URI trong index           = {len(uri_to_idx):,}")

    # Test tra cứu một entity (Wikidata Hà Nội)
    ha_noi_uri = "http://www.wikidata.org/entity/Q1748"
    ha_noi_emb = get_entity_embedding(ha_noi_uri, emb_matrix, uri_to_idx)
    if ha_noi_emb.norm() > 0:
        print(f"        Hà Nội embedding norm        = {ha_noi_emb.norm().item():.4f}")
    else:
        print("        Hà Nội chưa có trong KG (zero vector)")