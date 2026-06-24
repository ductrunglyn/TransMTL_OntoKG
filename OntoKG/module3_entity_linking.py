# module3_entity_linking.py
"""
Module 3: Entity Linking
Input : ./data/module2_ner_concept.jsonl  (output của Module 2)
Output: ./data/module3_entity_linked.jsonl  (entities đã có URI)
        ./data/entity_registry.pkl          (registry URI → entity info)
        ./data/wikidata_cache.json          (cache kết quả Wikidata)

Quy trình mỗi entity:
  1. Tra bảng Wikidata cache (không gọi API nếu đã biết)
  2. Gọi Wikidata wbsearchentities API (chỉ cho LOC / ORG / PER)
  3. Khớp với entity đã có trong registry bằng norm_surface
  4. Khớp bằng PhoBERT cosine similarity (ngưỡng 0.92)
  5. Tạo URI nội bộ mới nếu không tìm được
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:
    torch = None
    AutoModel = None
    AutoTokenizer = None

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
LOGGER = logging.getLogger("module3_entity_linking")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)

# ──────────────────────────────────────────────────────────
# Hằng số
# ──────────────────────────────────────────────────────────
BASE_URI     = "http://transmtl.vn/ent"
TOPIC_URI    = "http://transmtl.vn/onto/topic"
ONTO_URI     = "http://transmtl.vn/onto"
WD_URI       = "http://www.wikidata.org/entity"

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE    = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# Labels cần thử Wikidata (TIME / MISC / EVENT → bỏ qua)
WIKIDATA_ENABLED_LABELS = {"LOC", "PER", "ORG"}

# Chỉ so khớp embedding (Tầng 3) cho nhãn proper-noun. KHÔNG so khớp concept
# (MISC) vì concept rất nhiều, hiếm khi trùng, và là nguyên nhân chính gây
# bùng nổ số entity + làm chậm O(N^2).
EMB_MATCH_LABELS = {"LOC", "PER", "ORG"}

# Hints từ khoá để xác nhận type qua description Wikidata
WIKIDATA_TYPE_HINTS: Dict[str, List[str]] = {
    "LOC": [
        # Tiếng Việt
        "thành phố", "tỉnh", "quốc gia", "địa danh", "huyện", "xã",
        "thị xã", "thị trấn",
        # Tiếng Anh (Wikidata thường trả về mô tả tiếng Anh)
        "city", "province", "country", "district", "municipality",
        "region", "commune", "capital", "town", "village", "island",
        "territory", "prefecture", "state", "county", "ward",
        "city in", "province in", "district in", "capital of",
    ],
    "PER": [
        # Tiếng Việt
        "người", "nhà", "chính khách", "diễn viên",
        # Tiếng Anh
        "human", "person", "politician", "actor", "scientist",
        "athlete", "official", "leader", "director", "minister",
        "president", "general", "businessman", "singer", "writer",
        "journalist", "researcher", "professor", "engineer",
    ],
    "ORG": [
        # Tiếng Việt
        "tổ chức", "công ty", "bộ", "viện", "cơ quan", "tập đoàn",
        # Tiếng Anh
        "organization", "company", "ministry", "university",
        "institution", "agency", "corporation", "association",
        "committee", "department", "bureau", "school", "hospital",
        "foundation", "bank", "enterprise", "group", "authority",
    ],
}

# ──────────────────────────────────────────────────────────
# Tiện ích chuẩn hoá
# ──────────────────────────────────────────────────────────
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_key(text: str) -> str:
    """Lowercase, bỏ dấu câu đầu/cuối, collapse spaces — dùng làm khoá registry."""
    text = normalize_text(text).lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def make_uri(surface: str) -> str:
    """Tạo URI nội bộ từ surface form (md5 8 ký tự)."""
    h = hashlib.md5(normalize_key(surface).encode()).hexdigest()[:8]
    return f"{BASE_URI}/{h}"


def load_alias_map(path: Optional[str]) -> Dict[str, str]:
    """(2) Đọc từ điển alias JSON, trả về map normalize_key(alias) -> dạng_chuẩn.

    Hỗ trợ CẢ HAI kiểu JSON:
      • {"Hà Nội": ["HN", "TP Hà Nội", "Thủ đô"]}   (canonical -> list alias)
      • {"HN": "Hà Nội", "TP.HCM": "Thành phố Hồ Chí Minh"}  (alias -> canonical)
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        LOGGER.warning("Không thấy file alias %s — bỏ qua từ điển alias.", path)
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        LOGGER.warning("Lỗi đọc file alias %s: %s", path, e)
        return {}

    amap: Dict[str, str] = {}
    for key, val in raw.items():
        if isinstance(val, list):                 # canonical -> [alias, ...]
            canonical = str(key)
            amap[normalize_key(canonical)] = canonical
            for alias in val:
                if alias:
                    amap[normalize_key(alias)] = canonical
        elif val:                                 # alias -> canonical
            amap[normalize_key(key)] = str(val)
    return amap


def segment_for_phobert(text: str) -> str:
    """Word-segment text với underthesea (PhoBERT cần underscore)."""
    text = normalize_text(text)
    if not text:
        return text
    try:
        from underthesea import word_tokenize
        out = word_tokenize(text, format="text")
        return WHITESPACE_RE.sub(" ", normalize_text(out)).strip()
    except Exception:
        return text


# ──────────────────────────────────────────────────────────
# Wikidata Linker (với file cache)
# ──────────────────────────────────────────────────────────
class WikidataLinker:
    """
    Gọi Wikidata wbsearchentities API với:
      - File cache để tránh gọi lại
      - Rate limiting (0.25s/request)
      - Type validation qua description
    """

    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    REQUEST_INTERVAL = 0.25
    HEADERS = {
        "User-Agent": "TransMTL-OntoKG/1.0 (Vietnamese news research; mailto:ductrunghoang26@gmail.com)"
    }

    def __init__(self, cache_path: str = "./data/wikidata_cache.json"):
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Optional[str]] = {}   # norm_key → wikidata_id or None
        self._last_request = 0.0
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                LOGGER.info("Loaded Wikidata cache: %d entries", len(self._cache))
            except Exception as e:
                LOGGER.warning("Failed to load Wikidata cache: %s", e)

    def save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request = time.time()

    def _search(self, query: str, lang: str = "vi") -> List[Dict[str, Any]]:
        self._rate_limit()
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": lang,
            "type": "item",
            "format": "json",
            "limit": 5,
        }
        try:
            resp = requests.get(
                self.WIKIDATA_API, params=params, timeout=8,
                headers=self.HEADERS,   # ← THÊM DÒNG NÀY
            )
            resp.raise_for_status()
            return resp.json().get("search", [])
        except Exception as e:
            LOGGER.debug("Wikidata API error for '%s': %s", query, e)
            return []

    def _type_matches(self, description: str, label: str) -> bool:
        desc_lower = description.lower()
        hints = WIKIDATA_TYPE_HINTS.get(label, [])
        return any(h in desc_lower for h in hints)

    def lookup(self, surface: str, label: str) -> Optional[str]:
        """
        Trả về Wikidata Q-ID (VD: 'Q1748') nếu tìm thấy, None nếu không.
        Kết quả được cache vào file.
        """
        if label not in WIKIDATA_ENABLED_LABELS:
            return None

        cache_key = f"{label}::{normalize_key(surface)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        results = self._search(surface)
        wikidata_id = None

        for item in results:
            desc = item.get("description", "")
            if not desc:
                # Không có description → chấp nhận LOC (ít false positive hơn)
                if label == "LOC":
                    wikidata_id = item["id"]
                    break
            elif self._type_matches(desc, label):
                wikidata_id = item["id"]
                break

        self._cache[cache_key] = wikidata_id
        return wikidata_id

    def get_aliases(self, wikidata_id: str, langs=("vi", "en")) -> List[str]:
        """(1) Lấy label + danh sách 'also known as' của một Q-ID (cache lại).

        Dùng để đăng ký mọi biến thể tên (vd Q1748 -> 'Hà Nội', 'Hanoi', 'Thủ đô
        Hà Nội'...) về cùng một URI, giúp các mention sau khớp ngay ở Tầng 1.
        """
        if not wikidata_id:
            return []
        cache_key = f"ALIASES::{wikidata_id}"
        if cache_key in self._cache:
            return self._cache[cache_key] or []

        self._rate_limit()
        params = {
            "action": "wbgetentities",
            "ids": wikidata_id,
            "props": "labels|aliases",
            "languages": "|".join(langs),
            "format": "json",
        }
        surfaces: List[str] = []
        try:
            resp = requests.get(
                self.WIKIDATA_API, params=params, timeout=8, headers=self.HEADERS,
            )
            resp.raise_for_status()
            ent = resp.json().get("entities", {}).get(wikidata_id, {})
            for lang in langs:
                lab = ent.get("labels", {}).get(lang, {}).get("value")
                if lab:
                    surfaces.append(lab)
                for al in ent.get("aliases", {}).get(lang, []):
                    if al.get("value"):
                        surfaces.append(al["value"])
        except Exception as e:
            LOGGER.debug("Wikidata aliases error for %s: %s", wikidata_id, e)

        seen, out = set(), []
        for s in surfaces:
            s = normalize_text(s)
            k = normalize_key(s)
            if s and k not in seen:
                seen.add(k)
                out.append(s)
        self._cache[cache_key] = out
        return out


# ──────────────────────────────────────────────────────────
# PhoBERT Embedding Encoder
# ──────────────────────────────────────────────────────────
class EmbeddingEncoder:
    """
    Encode text thành vector 768 chiều bằng PhoBERT.
    Dùng mean pooling trên token embeddings.
    Có in-memory cache để tránh encode lại cùng 1 text.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self._cache: Dict[str, np.ndarray] = {}

        if torch is None or AutoTokenizer is None:
            LOGGER.warning("torch/transformers không có — entity embedding bị tắt.")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            LOGGER.info("Loaded embedding model: %s on %s", model_name, device)
        except Exception as e:
            LOGGER.warning("Failed to load embedding model: %s", e)

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def encode(self, text: str) -> Optional[np.ndarray]:
        if not self.available:
            return None
        key = normalize_key(text)
        if key in self._cache:
            return self._cache[key]

        seg = segment_for_phobert(text)
        if not seg:
            return None
        try:
            inputs = self.tokenizer(
                seg, return_tensors="pt", truncation=True, max_length=64, padding=True
            )
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            arr = emb[0].cpu().numpy()
            self._cache[key] = arr
            return arr
        except Exception as e:
            LOGGER.debug("Encode failed for '%s': %s", text, e)
            return None

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ──────────────────────────────────────────────────────────
# Entity Registry
# ──────────────────────────────────────────────────────────
class EntityRegistry:
    """
    Lưu toàn bộ entity đã biết.
    Hỗ trợ:
      - Lookup nhanh theo norm_surface (O(1))
      - Lookup theo cosine similarity của embedding
      - Persist ra file pkl
    """

    def __init__(self):
        # norm_key → uri (chỉ lưu URI tốt nhất)
        self.surface_to_uri: Dict[str, str] = {}
        # uri → metadata
        self.uri_info: Dict[str, Dict[str, Any]] = {}
        # uri → numpy embedding
        self.embeddings: Dict[str, np.ndarray] = {}

        # ── Chỉ mục embedding ĐÃ CHUẨN HOÁ để so khớp vector hoá (nhanh) ──
        # Dùng buffer cấp phát sẵn (nhân đôi khi đầy) -> thêm O(1) khấu hao,
        # tìm kiếm O(số entity cùng nhãn) bằng 1 phép nhân ma trận BLAS.
        self._emb_buf: Optional[np.ndarray] = None   # (cap, D) float32, đã chuẩn hoá
        self._emb_cap: int = 0
        self._emb_n: int = 0
        self._emb_uris: List[str] = []               # song song với hàng buffer
        self._label_idx: Dict[str, List[int]] = {}   # label -> list chỉ số hàng

    # ── Lookup ──────────────────────────────────────────────
    def get_uri_by_surface(self, norm_key: str) -> Optional[str]:
        return self.surface_to_uri.get(norm_key)

    def _emb_add(self, uri: str, label: str, emb: np.ndarray):
        """Thêm 1 embedding (đã chuẩn hoá) vào chỉ mục để so khớp nhanh."""
        v = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(v)
        if n < 1e-9:
            return
        v = v / n
        D = v.shape[0]
        if self._emb_buf is None:
            self._emb_cap = 1024
            self._emb_buf = np.zeros((self._emb_cap, D), dtype=np.float32)
        if self._emb_n >= self._emb_cap:
            self._emb_cap *= 2
            new_buf = np.zeros((self._emb_cap, D), dtype=np.float32)
            new_buf[: self._emb_n] = self._emb_buf[: self._emb_n]
            self._emb_buf = new_buf
        row = self._emb_n
        self._emb_buf[row] = v
        self._emb_uris.append(uri)
        self._label_idx.setdefault(label, []).append(row)
        self._emb_n += 1

    def get_uri_by_embedding(
        self, emb: np.ndarray, threshold: float = 0.92, label: Optional[str] = None
    ) -> Optional[str]:
        """So khớp cosine VECTOR HOÁ. Nếu có label -> chỉ so trong các entity
        cùng nhãn (nhanh hơn nhiều). Trả về URI tốt nhất nếu >= threshold."""
        if self._emb_buf is None or self._emb_n == 0:
            return None
        q = np.asarray(emb, dtype=np.float32)
        nq = np.linalg.norm(q)
        if nq < 1e-9:
            return None
        q = q / nq

        if label is not None:
            rows = self._label_idx.get(label)
            if not rows:
                return None
            sub = self._emb_buf[rows]                # (k, D), embeddings đã chuẩn hoá
            sims = sub @ q                           # cosine vì cả hai đã chuẩn hoá
            j = int(np.argmax(sims))
            return self._emb_uris[rows[j]] if sims[j] >= threshold else None

        mat = self._emb_buf[: self._emb_n]
        sims = mat @ q
        j = int(np.argmax(sims))
        return self._emb_uris[j] if sims[j] >= threshold else None

    # ── Register ─────────────────────────────────────────────
    def register(
        self,
        norm_key: str,
        uri: str,
        surface: str,
        label: str,
        uri_source: str,
        wikidata_id: Optional[str] = None,
        emb: Optional[np.ndarray] = None,
    ):
        """Đăng ký entity mới hoặc cập nhật thống kê entity đã có."""
        self.surface_to_uri[norm_key] = uri

        if uri not in self.uri_info:
            self.uri_info[uri] = {
                "uri": uri,
                "label": label,
                "canonical_surface": surface,
                "surface_forms": set(),
                "wikidata_id": wikidata_id,
                "uri_source": uri_source,
                "count": 0,
            }
        info = self.uri_info[uri]
        info["surface_forms"].add(surface)
        info["count"] += 1

        if emb is not None and uri not in self.embeddings:
            self.embeddings[uri] = emb
            # Chỉ đưa vào chỉ mục so khớp cho nhãn proper-noun (bỏ qua concept)
            # để buffer nhỏ gọn và phép nhân ma trận nhanh.
            if label in EMB_MATCH_LABELS:
                self._emb_add(uri, label, emb)

    # ── Persist ─────────────────────────────────────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Chuyển set → list để có thể pickle
        serializable = {}
        for uri, info in self.uri_info.items():
            serializable[uri] = {**info, "surface_forms": list(info["surface_forms"])}
        with open(path, "wb") as f:
            pickle.dump(
                {"uri_info": serializable, "surface_to_uri": self.surface_to_uri,
                 "embeddings": self.embeddings},
                f,
            )
        LOGGER.info("Registry saved: %d entities → %s", len(self.uri_info), path)

    @classmethod
    def load(cls, path: str) -> "EntityRegistry":
        reg = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        reg.surface_to_uri = data.get("surface_to_uri", {})
        reg.embeddings = data.get("embeddings", {})
        for uri, info in data.get("uri_info", {}).items():
            info["surface_forms"] = set(info.get("surface_forms", []))
            reg.uri_info[uri] = info
        LOGGER.info("Registry loaded: %d entities from %s", len(reg.uri_info), path)
        return reg


# ──────────────────────────────────────────────────────────
# Module 3 — Main Linker
# ──────────────────────────────────────────────────────────
class Module3EntityLinker:
    """
    Xử lý từng article từ Module 2:
      - Gán URI cho mỗi entity trong ner_entities
      - Gán URI cho mỗi concept trong concept_mentions
      - Thêm URI cho article và topic
    """

    def __init__(
        self,
        embedding_model: str = "vinai/phobert-base-v2",
        device: str = "cpu",
        wikidata_cache_path: str = "./data/wikidata_cache.json",
        similarity_threshold: float = 0.92,
        use_wikidata: bool = True,
        use_embedding_matching: bool = True,
        alias_dict_path: Optional[str] = None,        # (2) từ điển alias thủ công
        ingest_wikidata_aliases: bool = True,         # (1) nạp alias từ Wikidata
    ):
        self.registry = EntityRegistry()
        self.wikidata = WikidataLinker(cache_path=wikidata_cache_path) if use_wikidata else None
        self.encoder = EmbeddingEncoder(model_name=embedding_model, device=device)
        self.sim_threshold = similarity_threshold
        self.use_wikidata = use_wikidata
        self.use_embedding = use_embedding_matching and self.encoder.available

        # (2) Mặc định nạp aliases.json đặt cạnh module nếu không truyền đường dẫn.
        if alias_dict_path is None:
            _default = Path(__file__).with_name("aliases.json")
            alias_dict_path = str(_default) if _default.exists() else None
        self.alias_map = load_alias_map(alias_dict_path)
        if self.alias_map:
            LOGGER.info("Đã nạp %d alias thủ công từ %s", len(self.alias_map), alias_dict_path)

        # (1) Nạp alias Wikidata: chỉ ingest 1 lần / mỗi Q-ID.
        self.ingest_wd_aliases = ingest_wikidata_aliases
        self._aliases_ingested: set = set()

    # ── (1) Đăng ký alias Wikidata về cùng URI ──────────────
    def _ingest_wikidata_aliases(self, uri: str, wikidata_id: str, label: str):
        if not (self.ingest_wd_aliases and self.use_wikidata and self.wikidata):
            return
        if uri in self._aliases_ingested:
            return
        self._aliases_ingested.add(uri)
        for alias in self.wikidata.get_aliases(wikidata_id):
            k = normalize_key(alias)
            # Chỉ thêm nếu khoá chưa trỏ tới URI nào (tránh đè liên kết đã có).
            if k and k not in self.registry.surface_to_uri:
                self.registry.register(k, uri, alias, label, "wikidata_alias",
                                       wikidata_id, None)

    # ── Link một entity ─────────────────────────────────────
    def _link_one(
        self,
        surface: str,
        label: str,
        needs_review_labels: set = frozenset({"PER", "ORG"}),
    ) -> Dict[str, Any]:
        """
        Trả về dict: {uri, uri_source, wikidata_id, needs_review, embedding}
        """
        # (2) Chuẩn hoá alias thủ công TRƯỚC khi linking:
        #     "HN"/"TP Hà Nội"/"Thủ đô" -> "Hà Nội" rồi mới đi qua 4 tầng.
        if self.alias_map:
            canonical = self.alias_map.get(normalize_key(surface))
            if canonical:
                surface = canonical

        norm_key = normalize_key(surface)
        emb = self.encoder.encode(surface) if self.use_embedding else None

        # ── Tầng 1: surface match trong registry ────────────
        uri = self.registry.get_uri_by_surface(norm_key)
        if uri:
            info = self.registry.uri_info[uri]
            self.registry.register(norm_key, uri, surface, label, info["uri_source"],
                                   info.get("wikidata_id"), emb)
            return {
                "uri": uri,
                "uri_source": info["uri_source"],
                "wikidata_id": info.get("wikidata_id"),
                "needs_review": False,
                "embedding": emb.tolist() if emb is not None else None,
            }

        # ── Tầng 2: Wikidata lookup ──────────────────────────
        wikidata_id = None
        if self.use_wikidata and self.wikidata and label in WIKIDATA_ENABLED_LABELS:
            wikidata_id = self.wikidata.lookup(surface, label)

        if wikidata_id:
            uri = f"{WD_URI}/{wikidata_id}"
            self.registry.register(norm_key, uri, surface, label, "wikidata", wikidata_id, emb)
            # (1) Đăng ký mọi alias Wikidata về cùng URI này (chỉ 1 lần / Q-ID).
            self._ingest_wikidata_aliases(uri, wikidata_id, label)
            return {
                "uri": uri,
                "uri_source": "wikidata",
                "wikidata_id": wikidata_id,
                "needs_review": False,
                "embedding": emb.tolist() if emb is not None else None,
            }

        # ── Tầng 3: Embedding similarity (chỉ cho proper-noun, cùng nhãn) ──
        if emb is not None and self.use_embedding and label in EMB_MATCH_LABELS:
            matched_uri = self.registry.get_uri_by_embedding(
                emb, self.sim_threshold, label=label
            )
            if matched_uri:
                info = self.registry.uri_info[matched_uri]
                self.registry.register(norm_key, matched_uri, surface, label,
                                       info["uri_source"], info.get("wikidata_id"), emb)
                return {
                    "uri": matched_uri,
                    "uri_source": info["uri_source"],
                    "wikidata_id": info.get("wikidata_id"),
                    "needs_review": False,
                    "embedding": emb.tolist() if emb is not None else None,
                }

        # ── Tầng 4: Tạo URI nội bộ mới ──────────────────────
        uri = make_uri(surface)
        needs_review = label in needs_review_labels
        self.registry.register(norm_key, uri, surface, label, "new", None, emb)
        return {
            "uri": uri,
            "uri_source": "new",
            "wikidata_id": None,
            "needs_review": needs_review,
            "embedding": emb.tolist() if emb is not None else None,
        }

    # ── Xử lý một record ────────────────────────────────────
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        article_id = record.get("article_id", "")
        topic_list = record.get("topic_list", [])

        # Linked NER entities
        linked_entities: List[Dict[str, Any]] = []
        for ent in record.get("ner_entities", []):
            link = self._link_one(ent["surface"], ent["label"])
            linked_entities.append({**ent, **link})

        # Linked concepts (tất cả dùng label "MISC" → Concept)
        linked_concepts: List[Dict[str, Any]] = []
        for con in record.get("concept_mentions", []):
            link = self._link_one(con["concept"], "MISC")
            linked_concepts.append({**con, **link})

        # URI cho article và topic
        article_uri = f"http://transmtl.vn/article/{article_id}"
        topic_uris  = [f"{TOPIC_URI}/{t}" for t in topic_list]

        out = dict(record)
        out["ner_entities"]    = linked_entities
        out["concept_mentions"] = linked_concepts
        out["article_uri"]     = article_uri
        out["topic_uris"]      = topic_uris

        return out

    # ── Xử lý batch từ file ─────────────────────────────────
    def link_all(
        self,
        input_jsonl: str,
        output_jsonl: str,
        registry_pkl: str,
        wikidata_cache_json: str,
        save_every: int = 500,
    ):
        input_path  = Path(input_jsonl)
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        errors, processed = 0, 0

        with input_path.open("r", encoding="utf-8") as fin, \
             output_path.open("w", encoding="utf-8") as fout:

            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    out    = self.process_record(record)
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    processed += 1
                except Exception as e:
                    LOGGER.exception("Failed at line %d: %s", line_no, e)
                    errors += 1

                if processed % save_every == 0:
                    self.registry.save(registry_pkl)
                    if self.wikidata:
                        self.wikidata.save_cache()
                    LOGGER.info("Progress: %d processed, %d errors", processed, errors)

        self.registry.save(registry_pkl)
        if self.wikidata:
            self.wikidata.save_cache()
        LOGGER.info("Done. %d processed, %d errors → %s", processed, errors, output_jsonl)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    DATA = os.environ.get("OKG_DATA_DIR", "./data")
    device = os.environ.get("OKG_DEVICE", "cuda")
    linker = Module3EntityLinker(
        embedding_model="vinai/phobert-base-v2",
        device=device,
        wikidata_cache_path=os.path.join(DATA, "wikidata_cache.json"),
        similarity_threshold=0.92,
        use_wikidata=True,
        # Đặt OKG_EMB_MATCH=0 để TẮT hẳn so khớp embedding (nhanh nhất, chỉ dựa
        # vào surface + alias + Wikidata).
        use_embedding_matching=os.environ.get("OKG_EMB_MATCH", "1").lower()
                                not in ("0", "false", "no", "off"),
        # (2) đường dẫn từ điển alias (mặc định lấy OntoKG/aliases.json cạnh module).
        alias_dict_path=os.environ.get("OKG_ALIAS_JSON", None),
        # (1) nạp alias Wikidata (đặt OKG_WD_ALIASES=0 để tắt nếu muốn nhanh hơn).
        ingest_wikidata_aliases=os.environ.get("OKG_WD_ALIASES", "1").lower()
                                 not in ("0", "false", "no", "off"),
    )

    linker.link_all(
        input_jsonl=os.path.join(DATA, "module2_ner_concept.jsonl"),
        output_jsonl=os.path.join(DATA, "module3_entity_linked.jsonl"),
        registry_pkl=os.path.join(DATA, "entity_registry.pkl"),
        wikidata_cache_json=os.path.join(DATA, "wikidata_cache.json"),
        save_every=500,
    )