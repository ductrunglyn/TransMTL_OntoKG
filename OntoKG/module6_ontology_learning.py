# module6_ontology_learning.py
"""
Module 6: Ontology Learning từ dữ liệu
Input : ./data/kg/entity_index.pkl  (output của Module 5)
Output: ./data/ontology/ontology_v1.1.json   — cấu trúc lớp học được
        ./data/ontology/cluster_report.txt   — báo cáo thủ công
        ./data/ontology/cluster_matrix.npy   — UMAP coords (để visualize)

Quy trình:
  1. Load entity embeddings từ entity_index.pkl
  2. Lọc entity có đủ dữ liệu (occurrence ≥ min_count)
  3. UMAP: giảm 768 → 50 chiều (giữ cấu trúc cụm)
  4. HDBSCAN: phát hiện số cụm tự nhiên (không cần chỉ định trước)
  5. Phân tích từng cụm: label distribution + surface form đại diện
  6. Đặt tên lớp tự động từ distribution + gợi ý cho review thủ công
  7. Xuất ontology JSON + báo cáo

Ghi chú quan trọng:
  - Module này KHÔNG thay đổi entity_embeddings trong Module 7
  - Đầu ra là cấu trúc lớp để cập nhật Ontology (OWL) thủ công hoặc bán tự động
  - Kết quả tốt nhất khi min_count ≥ 5 (lọc bỏ entity chỉ xuất hiện 1–2 lần)

Dependencies: pip install umap-learn hdbscan scikit-learn numpy
"""
from __future__ import annotations

import json
import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# UMAP + HDBSCAN (optional — graceful fallback nếu chưa cài)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
LOGGER = logging.getLogger("module6_ontology_learning")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)


def _loud_warn(title: str, *lines: str) -> None:
    """In cảnh báo nhiều dòng, có khung, để KHÔNG bị lẫn vào log INFO.

    Dùng khi module phải HẠ CẤP chất lượng (vd thiếu UMAP/HDBSCAN) — để người
    chạy nhìn thấy ngay thay vì âm thầm cho ra kết quả kém tin cậy.
    """
    bar = "!" * 72
    LOGGER.warning(bar)
    LOGGER.warning("!! %s", title)
    for ln in lines:
        LOGGER.warning("!! %s", ln)
    LOGGER.warning(bar)

# ──────────────────────────────────────────────────────────
# Ánh xạ NER label → tên lớp ontology ban đầu (seed)
# ──────────────────────────────────────────────────────────
SEED_CLASS_MAP = {
    "PER":   "Person",
    "ORG":   "Organization",
    "LOC":   "Location",
    "TIME":  "TimeRef",
    "EVENT": "Event",
    "MISC":  "Concept",
}

# Tên lớp cha dựa trên label chiếm đa số trong cụm
DOMINANT_LABEL_TO_PARENT = {
    "PER":   "Person",
    "ORG":   "Organization",
    "LOC":   "Location",
    "TIME":  "TimeRef",
    "EVENT": "Event",
    "MISC":  "Concept",
}


# ──────────────────────────────────────────────────────────
# Dataclass đại diện một lớp học được
# ──────────────────────────────────────────────────────────
@dataclass
class LearnedClass:
    cluster_id:          int
    suggested_name:      str          # tên gợi ý tự động
    parent_class:        str          # lớp cha trong ontology
    dominant_label:      str          # NER label chiếm đa số
    label_distribution:  Dict[str, int]
    n_entities:          int
    representative_surfaces: List[str]  # top-10 surface forms đại diện
    centroid:            Optional[List[float]]  # centroid embedding (768 dim)
    uri_source_distribution: Dict[str, int]
    is_new_class:        bool         # True nếu không khớp lớp seed nào
    confidence:          float        # mức độ thuần nhất của cụm (0-1)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Không serialize centroid vào JSON (quá lớn) — lưu riêng nếu cần
        d["centroid"] = None
        return d


# ──────────────────────────────────────────────────────────
# Tiện ích
# ──────────────────────────────────────────────────────────
def load_entity_index(path: str) -> Dict[str, Any]:
    """Load entity_index.pkl từ Module 5."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_embedding_matrix(
    entity_data: Dict[str, Any],
    min_count: int = 3,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Lọc entity có embedding và occurrence ≥ min_count.
    Trả về:
      - matrix: (N_filtered, 768)
      - uris:   list URI tương ứng
      - infos:  list metadata tương ứng
    """
    uris, embeddings, infos = [], [], []

    for uri, item in entity_data.items():
        emb = item.get("embedding")
        info = item.get("info", {})
        count = info.get("occurrence_count", 0)

        if emb is None:
            continue
        if count < min_count:
            continue

        emb_arr = np.array(emb, dtype=np.float32)
        if emb_arr.shape[0] != 768:
            continue
        # L2 normalize
        norm = np.linalg.norm(emb_arr)
        if norm < 1e-9:
            continue

        uris.append(uri)
        embeddings.append(emb_arr / norm)
        infos.append(info)

    if not embeddings:
        return np.empty((0, 768)), [], []

    return np.stack(embeddings, axis=0), uris, infos


def reduce_with_umap(
    matrix: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    random_state: int = 42,
    strict: bool = False,
) -> np.ndarray:
    """UMAP: 768 → n_components chiều. Fallback về PCA nếu UMAP chưa cài.

    strict=True → báo lỗi thay vì hạ cấp âm thầm (để không cho ra kết quả kém).
    """
    if HAS_UMAP:
        LOGGER.info("Running UMAP: %s → %d dims", matrix.shape, n_components)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
            low_memory=True,
        )
        return reducer.fit_transform(matrix)

    if strict:
        raise RuntimeError(
            "UMAP chưa được cài nhưng đang ở chế độ STRICT. "
            "Cài 'pip install umap-learn' hoặc bỏ OKG_ONTOLOGY_STRICT."
        )

    if HAS_SKLEARN:
        _loud_warn(
            "UMAP CHƯA CÀI — đang dùng PCA thay thế (chất lượng giảm).",
            "PCA tuyến tính không giữ được cấu trúc cụm phi tuyến như UMAP.",
            "Khắc phục: pip install umap-learn  rồi chạy lại module 6.",
        )
        pca = PCA(n_components=min(n_components, matrix.shape[1]),
                  random_state=random_state)
        return pca.fit_transform(matrix)

    _loud_warn(
        "Cả UMAP lẫn scikit-learn đều KHÔNG có — dùng embedding gốc 768 chiều.",
        "Clustering trên không gian gốc rất kém. Cài: pip install umap-learn scikit-learn",
    )
    return matrix


def cluster_with_hdbscan(
    reduced: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    strict: bool = False,
) -> np.ndarray:
    """
    HDBSCAN clustering. Trả về array nhãn (-1 = noise).
    Fallback về K-Means nếu HDBSCAN chưa cài.

    strict=True → báo lỗi thay vì hạ cấp âm thầm sang KMeans.
    """
    if HAS_HDBSCAN:
        LOGGER.info("Running HDBSCAN on shape %s", reduced.shape)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int((labels == -1).sum())
        LOGGER.info("HDBSCAN: %d clusters, %d noise points", n_clusters, n_noise)
        return labels

    if strict:
        raise RuntimeError(
            "HDBSCAN chưa được cài nhưng đang ở chế độ STRICT. "
            "Cài 'pip install hdbscan' hoặc bỏ OKG_ONTOLOGY_STRICT."
        )

    if HAS_SKLEARN:
        from sklearn.cluster import KMeans
        k = min(20, reduced.shape[0] // 10)
        k = max(k, 2)
        _loud_warn(
            f"HDBSCAN CHƯA CÀI — đang dùng KMeans(k={k}) thay thế (kết quả KHÔNG đáng tin).",
            f"KMeans bị ép đúng {k} cụm bất kể cấu trúc thật, không có khái niệm nhiễu.",
            "Số cụm 'tự nhiên' và cờ *NEW* sẽ là artifact, không phản ánh lớp ontology.",
            "Khắc phục: pip install hdbscan  rồi chạy lại module 6.",
        )
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return km.fit_predict(reduced)

    raise RuntimeError("Cần ít nhất hdbscan hoặc scikit-learn: pip install hdbscan scikit-learn")


# ──────────────────────────────────────────────────────────
# Phân tích từng cụm
# ──────────────────────────────────────────────────────────
def _cluster_confidence(label_dist: Dict[str, int]) -> float:
    """
    Đo độ thuần nhất: tỷ lệ nhãn chiếm đa số.
    1.0 = tất cả entity có cùng label. 0.0 = hoàn toàn hỗn hợp.
    """
    total = sum(label_dist.values())
    if total == 0:
        return 0.0
    return max(label_dist.values()) / total


def _suggest_class_name(
    dominant_label: str,
    top_surfaces: List[str],
    cluster_id: int,
) -> str:
    """
    Gợi ý tên lớp dựa trên nhãn và surface forms.
    Tên được tạo tự động — cần review thủ công.
    """
    base = DOMINANT_LABEL_TO_PARENT.get(dominant_label, "NewsEntity")

    # Phân tích pattern từ top surfaces
    surfaces_lower = [s.lower() for s in top_surfaces[:5]]

    # Một số heuristic đơn giản
    patterns = {
        "GovernmentBody": ["bộ ", "sở ", "ủy ban", "ubnd", "chính phủ", "hội đồng"],
        "MediaOrganization": ["báo ", "đài ", "vnexpress", "vtv", "vov"],
        "EducationalInstitution": ["trường ", "đại học", "học viện", "viện"],
        "Province": ["tỉnh", "thành phố", "tp.", "tp "],
        "Country": ["việt nam", "trung quốc", "mỹ", "nhật", "hàn quốc"],
        "Person_Official": ["chủ tịch", "thủ tướng", "bộ trưởng", "giám đốc"],
    }

    for subclass, keywords in patterns.items():
        if any(kw in " ".join(surfaces_lower) for kw in keywords):
            return subclass

    return f"{base}_Cluster{cluster_id}"


def analyze_clusters(
    labels: np.ndarray,
    uris: List[str],
    infos: List[Dict[str, Any]],
    original_embeddings: np.ndarray,
) -> List[LearnedClass]:
    """Phân tích từng cụm HDBSCAN và tạo LearnedClass."""
    cluster_ids = sorted(set(labels))
    learned_classes: List[LearnedClass] = []

    for cid in cluster_ids:
        if cid == -1:
            continue   # noise

        mask = labels == cid
        cluster_uris  = [uris[i]  for i in range(len(uris))  if mask[i]]
        cluster_infos = [infos[i] for i in range(len(infos)) if mask[i]]
        cluster_embs  = original_embeddings[mask]

        n = len(cluster_uris)
        if n < 3:
            continue

        # Label distribution
        label_dist: Dict[str, int] = Counter(
            info.get("label", "MISC") for info in cluster_infos
        )
        dominant_label = label_dist.most_common(1)[0][0]

        # URI source distribution
        source_dist: Dict[str, int] = Counter(
            info.get("uri_source", "new") for info in cluster_infos
        )

        # Surface forms đại diện (ưu tiên entity xuất hiện nhiều lần)
        surface_count: List[Tuple[str, int]] = []
        for info in cluster_infos:
            surfaces = info.get("surface_forms", [])
            count    = info.get("occurrence_count", 1)
            if surfaces:
                surface_count.append((surfaces[0], count))
        surface_count.sort(key=lambda x: x[1], reverse=True)
        top_surfaces = [s for s, _ in surface_count[:10]]

        # Centroid
        centroid = cluster_embs.mean(axis=0)

        # Confidence
        conf = _cluster_confidence(label_dist)

        # Tên gợi ý
        suggested_name = _suggest_class_name(dominant_label, top_surfaces, cid)

        # Có phải lớp mới không (không khớp các lớp seed)?
        seed_names = set(SEED_CLASS_MAP.values())
        parent_class = DOMINANT_LABEL_TO_PARENT.get(dominant_label, "NewsEntity")
        is_new = conf < 0.70  # cụm hỗn hợp nhiều label → có thể là lớp mới

        learned_classes.append(LearnedClass(
            cluster_id=int(cid),
            suggested_name=suggested_name,
            parent_class=parent_class,
            dominant_label=dominant_label,
            label_distribution=dict(label_dist),
            n_entities=n,
            representative_surfaces=top_surfaces,
            centroid=centroid.tolist(),
            uri_source_distribution=dict(source_dist),
            is_new_class=is_new,
            confidence=round(conf, 4),
        ))

    # Sắp xếp: cụm lớn và thuần nhất lên trước
    learned_classes.sort(key=lambda x: (x.n_entities, x.confidence), reverse=True)
    return learned_classes


# ──────────────────────────────────────────────────────────
# Xuất báo cáo
# ──────────────────────────────────────────────────────────
def write_cluster_report(
    learned_classes: List[LearnedClass],
    noise_count: int,
    total_entities: int,
    path: str,
):
    lines = [
        "=" * 70,
        "ONTOLOGY LEARNING REPORT — MODULE 6",
        "=" * 70,
        f"Tổng entity đưa vào clustering : {total_entities:,}",
        f"Noise (không thuộc cụm nào)     : {noise_count:,}",
        f"Số cụm phát hiện                : {len(learned_classes)}",
        "",
        "Ghi chú: 'suggested_name' chỉ là GỢI Ý — cần review thủ công!",
        "=" * 70,
        "",
    ]

    for i, lc in enumerate(learned_classes, start=1):
        new_flag = " ← POTENTIAL NEW CLASS" if lc.is_new_class else ""
        lines += [
            f"Cluster #{lc.cluster_id:3d}  [{lc.suggested_name}]{new_flag}",
            f"  Parent class     : {lc.parent_class}",
            f"  N entities       : {lc.n_entities:,}",
            f"  Dominant label   : {lc.dominant_label}  (confidence={lc.confidence:.2%})",
            f"  Label breakdown  : {dict(sorted(lc.label_distribution.items(), key=lambda x: -x[1]))}",
            f"  URI sources      : {lc.uri_source_distribution}",
            f"  Top surfaces     : {lc.representative_surfaces[:5]}",
            "",
        ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Cluster report → %s", path)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
class Module6OntologyLearner:

    def __init__(
        self,
        min_entity_count:  int   = 5,
        umap_components:   int   = 50,
        umap_neighbors:    int   = 15,
        hdbscan_min_size:  int   = 15,
        hdbscan_min_samp:  int   = 5,
        random_state:      int   = 42,
        strict:            bool  = False,
    ):
        self.min_entity_count = min_entity_count
        self.umap_components  = umap_components
        self.umap_neighbors   = umap_neighbors
        self.hdbscan_min_size = hdbscan_min_size
        self.hdbscan_min_samp = hdbscan_min_samp
        self.random_state     = random_state
        self.strict           = strict

    def run(
        self,
        entity_index_pkl:   str,
        output_ontology_json: str,
        output_report_txt:  str,
        output_umap_npy:    str,
    ):
        # ── 1. Load ────────────────────────────────────────
        LOGGER.info("Loading entity index from %s", entity_index_pkl)
        entity_data = load_entity_index(entity_index_pkl)
        LOGGER.info("Total entries in index: %d", len(entity_data))

        # ── 2. Build embedding matrix ─────────────────────
        matrix, uris, infos = build_embedding_matrix(
            entity_data, min_count=self.min_entity_count
        )
        if matrix.shape[0] < 10:
            LOGGER.error(
                "Chỉ có %d entity đủ tiêu chuẩn — không đủ để cluster. "
                "Giảm min_entity_count hoặc kiểm tra entity_index.pkl.",
                matrix.shape[0],
            )
            return
        LOGGER.info("Embedding matrix: %s (sau khi lọc min_count=%d)",
                    matrix.shape, self.min_entity_count)

        # ── 3. UMAP ────────────────────────────────────────
        reduced = reduce_with_umap(
            matrix,
            n_components=self.umap_components,
            n_neighbors=self.umap_neighbors,
            random_state=self.random_state,
            strict=self.strict,
        )

        # Tạo thư mục trước khi lưu (np.save không tự tạo)
        Path(output_umap_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_umap_npy, reduced)
        LOGGER.info("UMAP coords saved → %s", output_umap_npy)

        # ── 4. HDBSCAN ─────────────────────────────────────
        labels = cluster_with_hdbscan(
            reduced,
            min_cluster_size=self.hdbscan_min_size,
            min_samples=self.hdbscan_min_samp,
            strict=self.strict,
        )
        noise_count = int((labels == -1).sum())

        # ── 5. Phân tích cụm ──────────────────────────────
        learned_classes = analyze_clusters(labels, uris, infos, matrix)
        LOGGER.info("Analyzed %d clusters", len(learned_classes))

        # ── 6. Xuất JSON ──────────────────────────────────
        output = {
            "version": "v1.1",
            "source": "module6_hdbscan",
            "params": {
                "min_entity_count":  self.min_entity_count,
                "umap_components":   self.umap_components,
                "hdbscan_min_size":  self.hdbscan_min_size,
            },
            "summary": {
                "total_entities_clustered": int(matrix.shape[0]),
                "noise_entities":           noise_count,
                "n_clusters":               len(learned_classes),
            },
            "learned_classes": [lc.to_dict() for lc in learned_classes],
        }
        Path(output_ontology_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_ontology_json).write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("Ontology JSON → %s", output_ontology_json)

        # ── 7. Báo cáo ────────────────────────────────────
        write_cluster_report(
            learned_classes, noise_count, int(matrix.shape[0]),
            output_report_txt,
        )

        self._print_summary(learned_classes, noise_count, matrix.shape[0])

    @staticmethod
    def _print_summary(
        learned_classes: List[LearnedClass],
        noise_count: int,
        total: int,
    ):
        print("\n" + "=" * 55)
        print("MODULE 6 SUMMARY")
        print("=" * 55)
        print(f"  Total entities clustered  : {total:,}")
        print(f"  Noise (không vào cụm nào) : {noise_count:,}")
        print(f"  Clusters found            : {len(learned_classes)}")
        new_classes = [lc for lc in learned_classes if lc.is_new_class]
        print(f"  Potential new classes     : {len(new_classes)}")
        print()
        print("  Top 10 clusters:")
        for lc in learned_classes[:10]:
            flag = " *NEW*" if lc.is_new_class else ""
            print(f"    [{lc.cluster_id:3d}] {lc.suggested_name:<30} "
                  f"n={lc.n_entities:,}  conf={lc.confidence:.2%}{flag}")
        print("=" * 55)
        print()
        print("ACTION REQUIRED: Đọc cluster_report.txt và review các lớp được gợi ý,")
        print("đặc biệt các lớp có nhãn *NEW* — đây là lớp tiềm năng cần thêm vào ontology.")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    def _int_env(key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, default))
        except (TypeError, ValueError):
            LOGGER.warning("ENV %s không phải số — dùng mặc định %d", key, default)
            return default

    def _bool_env(key: str, default: bool = False) -> bool:
        return os.environ.get(key, str(default)).strip().lower() \
            not in ("0", "false", "no", "off", "")

    DATA = os.environ.get("OKG_DATA_DIR", "./data")

    # Tham số có thể chỉnh qua biến môi trường (không phải sửa code mỗi lần):
    #   OKG_MIN_ENTITY_COUNT  (mặc định 5)  — lọc entity xuất hiện ít lần
    #   OKG_UMAP_COMPONENTS / OKG_UMAP_NEIGHBORS
    #   OKG_HDBSCAN_MIN_SIZE / OKG_HDBSCAN_MIN_SAMP
    #   OKG_ONTOLOGY_STRICT=1 — báo lỗi nếu thiếu UMAP/HDBSCAN thay vì hạ cấp
    learner = Module6OntologyLearner(
        min_entity_count=_int_env("OKG_MIN_ENTITY_COUNT", 5),
        umap_components=_int_env("OKG_UMAP_COMPONENTS", 50),
        umap_neighbors=_int_env("OKG_UMAP_NEIGHBORS", 15),
        hdbscan_min_size=_int_env("OKG_HDBSCAN_MIN_SIZE", 15),
        hdbscan_min_samp=_int_env("OKG_HDBSCAN_MIN_SAMP", 5),
        strict=_bool_env("OKG_ONTOLOGY_STRICT", False),
    )
    learner.run(
        entity_index_pkl=os.path.join(DATA, "kg", "entity_index.pkl"),
        output_ontology_json=os.path.join(DATA, "ontology", "ontology_v1.1.json"),
        output_report_txt=os.path.join(DATA, "ontology", "cluster_report.txt"),
        output_umap_npy=os.path.join(DATA, "ontology", "cluster_matrix.npy"),
    )