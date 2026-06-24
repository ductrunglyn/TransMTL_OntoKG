from __future__ import annotations

import ast
import html
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# =========================================================
# optional torch / transformers
# =========================================================
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer, pipeline
except Exception:
    pipeline = None
    AutoModel = None
    AutoTokenizer = None

# =========================================================
# underthesea
# =========================================================
try:
    from underthesea import ner as uts_ner
except Exception:
    uts_ner = None

try:
    from underthesea import sent_tokenize as uts_sent_tokenize
except Exception:
    uts_sent_tokenize = None

try:
    from underthesea import word_tokenize as uts_word_tokenize
except Exception:
    uts_word_tokenize = None

# =========================================================
# PhoNLP
# =========================================================
try:
    import phonlp
except Exception:
    phonlp = None


# =========================================================
# Logging
# =========================================================
LOGGER = logging.getLogger("module2_ner_concept")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    LOGGER.addHandler(stream_handler)


# =========================================================
# Regex
# =========================================================
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")

# =========================================================
# Ontology label map
# =========================================================
NER_LABEL_MAP = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "TIME": "TimeRef",
    "EVENT": "Event",
    "MISC": "Concept",
}

VALID_NER_LABELS = {"PER", "ORG", "LOC", "TIME", "EVENT", "MISC"}

# Chuẩn hóa label của các model NER khác nhau về bộ nhãn chung
PHOBERT_LABEL_NORM = {
    "PERSON": "PER",
    "PEOPLE": "PER",
    "PER": "PER",
    "ORGANIZATION": "ORG",
    "ORG": "ORG",
    "LOCATION": "LOC",
    "LOC": "LOC",
    "GPE": "LOC",
    "DATE": "TIME",
    "TIME": "TIME",
    "EVENT": "EVENT",
    "PRODUCT": "MISC",
    "MISC": "MISC",
    "WORK_OF_ART": "MISC",
}

# =========================================================
# Required fields from module 1 output
# =========================================================
REQUIRED_INPUT_FIELDS = [
    "article_id",
    "title",
    "summary",
    "content",
    "publish_time",
    "topic_list",
    "cleaned_keywords",
    "full_text",
    "full_text_tokens",
    "full_text_sentences",
    "full_text_pos",
]

# =========================================================
# Stopwords
# =========================================================
STOPWORDS_CONCEPT = {
    "của", "và", "hoặc", "nhưng", "thì", "là", "các", "những", "một", "như",
    "ở", "tại", "với", "theo", "trong", "ngoài", "đến", "từ", "cho", "để",
    "về", "được", "bị", "này", "kia", "đó",
}


# =========================================================
# Utils
# =========================================================
def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_key(text: str) -> str:
    text = normalize_text(text)
    text = text.lower()
    text = PUNCT_EDGE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_ner_label(label: str) -> str:
    label = normalize_text(label).upper()
    label = label.replace("B-", "").replace("I-", "")
    return PHOBERT_LABEL_NORM.get(label, label)


def dedupe_preserve(items: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        x = normalize_text(x)
        if not x:
            continue
        key = normalize_key(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def parse_list_like(value: Any) -> List[str]:
    """
    Parse list-like content from JSONL fields.
    Expected use here:
      - cleaned_keywords from module 1
      - topic_list from module 1
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [normalize_text(x) for x in value if normalize_text(x)]

    if isinstance(value, tuple):
        return [normalize_text(x) for x in value if normalize_text(x)]

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [normalize_text(x) for x in parsed if normalize_text(x)]
            if isinstance(parsed, tuple):
                return [normalize_text(x) for x in parsed if normalize_text(x)]
        except Exception:
            pass

        parts = re.split(r"\s*[,;|]\s*", value)
        return [normalize_text(x) for x in parts if normalize_text(x)]

    return []


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    if uts_sent_tokenize is not None:
        try:
            sents = uts_sent_tokenize(text)
            if isinstance(sents, str):
                sents = [sents]
            return [normalize_text(x) for x in sents if normalize_text(x)]
        except Exception:
            pass

    chunks = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [normalize_text(x) for x in chunks if normalize_text(x)]


def segment_text_for_phobert(text: str) -> str:
    """
    PhoBERT/NER backend cần văn bản đã word-segment với dấu underscore.
    """
    text = normalize_text(text)
    if not text:
        return ""

    if uts_word_tokenize is not None:
        try:
            out = uts_word_tokenize(text, format="text")
        except TypeError:
            out = uts_word_tokenize(text)
        except Exception:
            out = None

        if out is not None:
            if isinstance(out, list):
                out = " ".join(normalize_text(tok) for tok in out if normalize_text(tok))
            else:
                out = normalize_text(out)
            return WHITESPACE_RE.sub(" ", out).strip()

    return WHITESPACE_RE.sub(" ", text).strip()


def pos_is_nounlike(pos_tag: str) -> bool:
    pos_tag = normalize_text(pos_tag)
    return pos_tag.startswith("N") or pos_tag.startswith("Np") or pos_tag.startswith("Nc")


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a = set(normalize_key(text_a).split())
    b = set(normalize_key(text_b).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# =========================================================
# Dataclasses
# =========================================================
@dataclass
class NERMention:
    surface: str
    norm_surface: str
    label: str
    label_name: str
    confidence: float
    source: str
    sentence_index: int
    votes: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class ConceptMention:
    concept: str
    norm_concept: str
    score: float
    source: str
    evidence: str
    linked_concept: Optional[str]
    similarity: Optional[float]
    action: str

    def to_dict(self):
        return asdict(self)


# =========================================================
# PhoNLP wrapper
# =========================================================
class PhoNLPWrapper:
    def __init__(self, save_dir: str = "phonlp_model"):
        self.model = None
        if phonlp is None:
            LOGGER.warning("PhoNLP not installed.")
            return

        try:
            self.model = phonlp.load(save_dir=save_dir)
            LOGGER.info("Loaded PhoNLP from %s", save_dir)
        except Exception as e:
            LOGGER.warning("Failed loading PhoNLP: %s", e)

    def predict_ner(self, sentence: str):
        if self.model is None:
            return None
        try:
            return self.model.annotate(text=sentence)
        except Exception as e:
            LOGGER.warning("PhoNLP annotate failed: %s", e)
            return None


# =========================================================
# Main extractor
# =========================================================
class Module2NerConceptExtractor:
    def __init__(
        self,
        ontology_concepts: Optional[List[Dict[str, Any]]] = None,
        use_underthesea_ner: bool = True,
        use_phobert_ner: bool = True,
        phobert_model_name: Optional[str] = None,
        use_phonlp: bool = True,
        phonlp_save_dir: str = "phonlp_model",
        concept_embedding_model_name: Optional[str] = "vinai/phobert-base-v2",
        concept_link_threshold: float = 0.85,
        concept_review_threshold: float = 0.65,
        concept_min_score: float = 0.50,   # tăng từ 0.35 → 0.50
        max_concept_tokens: int = 6,
        max_concepts: int = 20,            # thêm tham số mới
    ):
        self.ontology_concepts = ontology_concepts or []

        self.concept_link_threshold = concept_link_threshold
        self.concept_review_threshold = concept_review_threshold
        self.concept_min_score = concept_min_score
        self.max_concept_tokens = max_concept_tokens
        self.max_concepts = max_concepts

        # underthesea NER
        self.use_underthesea_ner = use_underthesea_ner and uts_ner is not None

        # Thiết bị cho các model transformer. Mặc định chạy GPU nếu có
        # (trước đây pipeline/embed model chạy CPU -> RẤT chậm).
        import os as _os
        _dev = str(_os.environ.get("OKG_DEVICE", "cuda"))
        if torch is not None and torch.cuda.is_available() and "cuda" in _dev:
            self._torch_device = torch.device("cuda")
            self._hf_device = 0
            self._hf_dtype = torch.float16
        else:
            self._torch_device = torch.device("cpu") if torch is not None else None
            self._hf_device = -1
            self._hf_dtype = None

        # PhoBERT / HF NER
        self.use_phobert_ner = bool(use_phobert_ner and pipeline is not None and phobert_model_name)
        self.phobert_ner = None
        if self.use_phobert_ner:
            try:
                _pipe_kwargs = dict(
                    task="token-classification",
                    model=phobert_model_name,
                    aggregation_strategy="simple",
                    device=self._hf_device,
                )
                if self._hf_dtype is not None:
                    _pipe_kwargs["torch_dtype"] = self._hf_dtype
                self.phobert_ner = pipeline(**_pipe_kwargs)
                LOGGER.info("Loaded NER model: %s (device=%s)", phobert_model_name, self._hf_device)
            except Exception as e:
                self.phobert_ner = None
                LOGGER.warning("Failed loading NER model %s: %s", phobert_model_name, e)

        self._ner_needs_segmentation = self._model_needs_segmentation(phobert_model_name)
        LOGGER.info(
            "NER model segmentation: %s (model=%s)",
            self._ner_needs_segmentation,
            phobert_model_name,
        )

        # PhoNLP
        self.use_phonlp = use_phonlp
        self.phonlp = PhoNLPWrapper(save_dir=phonlp_save_dir) if use_phonlp else None

        # Concept embedding encoder
        self.concept_embedding_model_name = concept_embedding_model_name
        self.embed_tokenizer = None
        self.embed_model = None
        self._embed_cache: Dict[str, Any] = {}
        self._ontology_embed_cache: Dict[str, Any] = {}

        if (
            concept_embedding_model_name
            and AutoTokenizer is not None
            and AutoModel is not None
            and torch is not None
        ):
            try:
                self.embed_tokenizer = AutoTokenizer.from_pretrained(
                    concept_embedding_model_name,
                    use_fast=False,
                )
                self.embed_model = AutoModel.from_pretrained(concept_embedding_model_name)
                self.embed_model.eval()
                if getattr(self, "_torch_device", None) is not None:
                    self.embed_model.to(self._torch_device)   # GPU nếu có
                LOGGER.info("Loaded concept embedding model: %s (device=%s)",
                            concept_embedding_model_name, getattr(self, "_torch_device", "cpu"))
            except Exception as e:
                self.embed_tokenizer = None
                self.embed_model = None
                LOGGER.warning("Failed loading concept embedding model %s: %s", concept_embedding_model_name, e)

    # =====================================================
    # Embedding helpers
    # =====================================================
    def _embed_text(self, text: str):
        if self.embed_model is None or self.embed_tokenizer is None or torch is None:
            return None

        text = normalize_text(text)
        if not text:
            return None

        key = normalize_key(text)
        if key in self._embed_cache:
            return self._embed_cache[key]

        seg = segment_text_for_phobert(text)
        if not seg:
            return None

        try:
            inputs = self.embed_tokenizer(
                seg,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            if getattr(self, "_torch_device", None) is not None:
                inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.embed_model(**inputs)

            last_hidden = outputs.last_hidden_state  # [1, seq, hidden]
            mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            emb = (summed / denom)[0].detach().cpu()

            self._embed_cache[key] = emb
            return emb
        except Exception as e:
            LOGGER.debug("Embedding failed for text=%s: %s", text, e)
            return None

    @staticmethod
    def _cosine_from_tensors(vec_a, vec_b) -> float:
        if vec_a is None or vec_b is None or torch is None:
            return 0.0
        try:
            return float(torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0).item())
        except Exception:
            return 0.0

    # =====================================================
    # Validation
    # =====================================================
    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        missing = []
        for field in REQUIRED_INPUT_FIELDS:
            if field not in record:
                missing.append(field)
        return missing

    # =====================================================
    # underthesea NER
    # =====================================================
    def _extract_underthesea_ner(self, sentence: str, sentence_index: int) -> List[NERMention]:
        out = []
        if not self.use_underthesea_ner:
            return out

        try:
            preds = uts_ner(sentence)
        except Exception:
            return out

        if not isinstance(preds, list):
            return out

        current_tokens = []
        current_label  = None   # lưu label đã normalize, ví dụ "LOC"

        def flush():
            nonlocal current_tokens, current_label
            if not current_tokens or not current_label:
                current_tokens = []
                current_label  = None
                return
            surface = normalize_text(" ".join(current_tokens))
            if surface and current_label in VALID_NER_LABELS:
                out.append(NERMention(
                    surface=surface,
                    norm_surface=normalize_key(surface),
                    label=current_label,
                    label_name=NER_LABEL_MAP.get(current_label, current_label),
                    confidence=0.90,
                    source="underthesea",
                    sentence_index=sentence_index,
                    votes={current_label: 1.0},
                ))
            current_tokens = []
            current_label  = None

        for item in preds:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue

            word    = normalize_text(item[0])
            raw_tag = normalize_text(item[3])   # "B-LOC", "I-LOC", "O" — giữ nguyên B-/I-

            if not word:
                continue

            if raw_tag.startswith("B-"):
                flush()
                bio_label     = raw_tag[2:]                         # "LOC"
                current_label = normalize_ner_label(bio_label)     # qua PHOBERT_LABEL_NORM
                if current_label in VALID_NER_LABELS:
                    current_tokens = [word]
                else:
                    current_label = None   # nhãn không hợp lệ, bỏ qua

            elif raw_tag.startswith("I-"):
                bio_label  = raw_tag[2:]
                ner_label  = normalize_ner_label(bio_label)
                if current_label is not None and current_label == ner_label:
                    current_tokens.append(word)
                else:
                    flush()   # label không khớp → kết thúc entity hiện tại

            else:
                flush()   # "O" hoặc tag khác

        flush()
        return out

    # =====================================================
    # PhoBERT NER
    # =====================================================
    def _extract_phobert_ner(
        self,
        sentence: str,
        sentence_index: int,
        segmented_sentence: Optional[str] = None,
    ) -> List[NERMention]:
        out = []
        if self.phobert_ner is None:
            return out

        # Chỉ segment nếu model thực sự cần (PhoBERT); ELECTRA dùng plain text
        if self._ner_needs_segmentation:
            input_text = segmented_sentence or segment_text_for_phobert(sentence)
        else:
            input_text = sentence   # plain text cho NlpHUST ELECTRA và các model khác

        try:
            preds = self.phobert_ner(input_text)
        except Exception as e:
            LOGGER.warning("NER model failed: %s", e)
            return out

        if not isinstance(preds, list):
            return out

        for p in preds:
            if not isinstance(p, dict):
                continue

            # Sửa A2: replace underscore TRƯỚC, normalize (gộp spaces) SAU
            raw_word = p.get("word", "")
            surface  = normalize_text(raw_word.replace("_", " "))
            label    = normalize_ner_label(p.get("entity_group", ""))
            score    = float(p.get("score", 0.0))

            if label not in VALID_NER_LABELS or not surface:
                continue

            out.append(NERMention(
                surface=surface,
                norm_surface=normalize_key(surface),
                label=label,
                label_name=NER_LABEL_MAP.get(label, label),
                confidence=score,
                source="phobert",
                sentence_index=sentence_index,
                votes={label: score},
            ))
        return out
    
    # Thêm method vào class:
    @staticmethod
    def _model_needs_segmentation(model_name: Optional[str]) -> bool:
        """
        Chỉ PhoBERT-based models (VinAI) mới cần word-segmented input (có underscore).
        ELECTRA, BERT, XLM-R và các model khác dùng plain text.
        """
        if not model_name:
            return False
        name = model_name.lower()
        return "phobert" in name or ("vinai" in name)

    # =====================================================
    # PhoNLP NER
    # =====================================================
    def _extract_phonlp_ner(self, sentence: str, sentence_index: int) -> List[NERMention]:
        out = []
        if self.phonlp is None or self.phonlp.model is None:
            return out

        try:
            preds = self.phonlp.model.annotate(text=sentence)
            if not isinstance(preds, dict):
                return out

            tokens_per_sent = preds.get("tokens", [])
            ner_per_sent    = preds.get("ner",    [])

            if not isinstance(tokens_per_sent, list) or not isinstance(ner_per_sent, list):
                return out

            current_tokens = []
            current_label  = None

            def flush():
                nonlocal current_tokens, current_label
                if not current_tokens or not current_label:
                    current_tokens = []
                    current_label  = None
                    return
                surface = normalize_text(" ".join(current_tokens))
                if surface and current_label in VALID_NER_LABELS:
                    out.append(NERMention(
                        surface=surface,
                        norm_surface=normalize_key(surface),
                        label=current_label,
                        label_name=NER_LABEL_MAP.get(current_label, current_label),
                        confidence=0.92,
                        source="phonlp",
                        sentence_index=sentence_index,
                        votes={current_label: 1.0},
                    ))
                current_tokens = []
                current_label  = None

            for tokens_sent, ner_sent in zip(tokens_per_sent, ner_per_sent):
                if not isinstance(tokens_sent, list) or not isinstance(ner_sent, list):
                    continue

                for word, raw_tag in zip(tokens_sent, ner_sent):
                    word    = normalize_text(word)
                    raw_tag = normalize_text(raw_tag)   # "B-PER", "I-PER", "O"

                    if not word:
                        continue

                    if raw_tag.startswith("B-"):
                        flush()
                        bio_label     = raw_tag[2:]
                        current_label = normalize_ner_label(bio_label)
                        if current_label in VALID_NER_LABELS:
                            current_tokens = [word]
                        else:
                            current_label = None

                    elif raw_tag.startswith("I-"):
                        bio_label = raw_tag[2:]
                        ner_label = normalize_ner_label(bio_label)
                        if current_label is not None and current_label == ner_label:
                            current_tokens.append(word)
                        else:
                            flush()

                    else:
                        flush()

            flush()

        except Exception as e:
            LOGGER.warning("PhoNLP NER failed: %s", e)

        return out

    # =====================================================
    # Ensemble merge
    # =====================================================
    def _merge_ner_mentions(self, mentions: List[NERMention]) -> List[Dict[str, Any]]:
        grouped = defaultdict(list)
        for m in mentions:
            grouped[m.norm_surface].append(m)

        final_mentions = []

        for norm_surface, group in grouped.items():
            label_scores = defaultdict(float)
            for m in group:
                label_scores[m.label] += m.confidence

            best_label = max(label_scores.items(), key=lambda x: x[1])[0]

            best_surface = sorted(
                group,
                key=lambda x: (x.confidence, len(x.surface)),
                reverse=True,
            )[0].surface

            avg_conf = sum(x.confidence for x in group) / len(group)
            sources = sorted(set(x.source for x in group))

            final_mentions.append(
                {
                    "surface": best_surface,
                    "norm_surface": norm_surface,
                    "label": best_label,
                    "label_name": NER_LABEL_MAP.get(best_label, best_label),
                    "confidence": round(avg_conf, 4),
                    "sources": sources,
                    "votes": dict(label_scores),
                }
            )

        final_mentions = sorted(
            final_mentions,
            key=lambda x: (x["confidence"], len(x["surface"])),
            reverse=True,
        )

        final_mentions = [
            m for m in final_mentions
            if not self._is_false_positive(m)
        ]
        return final_mentions

    # Thêm method vào class:
    @staticmethod
    def _is_false_positive(entity: Dict[str, Any]) -> bool:
        """
        Lọc false positive rõ ràng từ underthesea.
        Không filter quá mạnh — chỉ loại các trường hợp chắc chắn sai.
        """
        surface = entity.get("surface", "")
        label   = entity.get("label", "")
        sources = entity.get("sources", [])

        # Chỉ lọc nếu từ DUY NHẤT underthesea (chưa được xác nhận bởi nguồn khác)
        if sources != ["underthesea"]:
            return False

        # Các pattern hay bị nhận sai là LOC
        if label == "LOC":
            surface_lower = surface.lower()
            # "lớp X" (lớp học) không phải Location
            if re.match(r"^lớp\s+\d+", surface_lower):
                return True
            # Số đơn thuần
            if re.match(r"^\d+$", surface.strip()):
                return True
            # Quá ngắn (1 ký tự)
            if len(surface.strip()) <= 1:
                return True

        return False

    # =====================================================
    # Concept extraction
    # =====================================================
    def _extract_candidate_phrases(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        IMPORTANT:
        - Only uses cleaned_keywords from module 1
        - Uses full_text_pos from module 1
        - Does NOT use keywords_list / keywords / removed_keywords
        """
        title = normalize_text(record.get("title", ""))
        cleaned_keywords = parse_list_like(record.get("cleaned_keywords", []))
        full_pos = record.get("full_text_pos", [])
        ner_entities = record.get("ner_entities", [])

        entity_norms = {
            normalize_key(e.get("surface", ""))
            for e in ner_entities
            if isinstance(e, dict) and e.get("surface")
        }

        candidates: List[Dict[str, Any]] = []

        # 1) cleaned_keywords ONLY
        for kw in cleaned_keywords:
            norm_kw = normalize_key(kw)
            if not norm_kw:
                continue
            if norm_kw in entity_norms:
                continue
            candidates.append(
                {
                    "phrase": kw,
                    "source": "cleaned_keywords",
                    "evidence": "cleaned_keywords",
                }
            )

        # 2) POS noun phrases from full_text_pos
        if isinstance(full_pos, list) and full_pos:
            current: List[Tuple[str, str]] = []

            def flush_np():
                nonlocal current
                if not current:
                    return

                toks = [normalize_text(x[0]) for x in current if normalize_text(x[0])]
                if 1 <= len(toks) <= self.max_concept_tokens:
                    phrase = normalize_text(" ".join(toks))
                    norm_phrase = normalize_key(phrase)
                    if norm_phrase and norm_phrase not in entity_norms:
                        candidates.append(
                            {
                                "phrase": phrase,
                                "source": "pos_np",
                                "evidence": "full_text_pos",
                            }
                        )
                current = []

            for item in full_pos:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue

                tok = normalize_text(item[0])
                pos = normalize_text(item[1])

                if not tok:
                    continue

                if tok.lower() in STOPWORDS_CONCEPT:
                    flush_np()
                    continue

                if pos_is_nounlike(pos):
                    current.append((tok, pos))
                    if len(current) >= self.max_concept_tokens:
                        flush_np()
                else:
                    flush_np()

            flush_np()

        # 3) title n-grams
        title_tokens = [x for x in title.split() if x]
        for n in range(2, min(5, len(title_tokens)) + 1):
            for i in range(0, len(title_tokens) - n + 1):
                phrase = " ".join(title_tokens[i: i + n])
                norm_phrase = normalize_key(phrase)
                if norm_phrase and norm_phrase not in entity_norms:
                    candidates.append(
                        {
                            "phrase": phrase,
                            "source": "title_ngram",
                            "evidence": "title",
                        }
                    )

        # dedupe
        out = []
        seen = set()
        for c in candidates:
            phrase = normalize_text(c["phrase"])
            norm_phrase = normalize_key(phrase)
            if not norm_phrase:
                continue
            if norm_phrase not in seen:
                seen.add(norm_phrase)
                c["phrase"] = phrase
                out.append(c)

        return out

    def _score_concept(self, phrase: str, record: Dict[str, Any]) -> float:
        phrase_l = phrase.lower()
        full_text = normalize_text(record.get("full_text", "")).lower()
        title = normalize_text(record.get("title", "")).lower()
        summary = normalize_text(record.get("summary", "")).lower()
        cleaned_keywords = [normalize_text(x).lower() for x in parse_list_like(record.get("cleaned_keywords", []))]

        count = full_text.count(phrase_l)
        tf_score = min(count / 3.0, 1.0)

        title_boost = 1.0 if phrase_l in title else 0.0
        summary_boost = 0.5 if phrase_l in summary else 0.0
        keyword_boost = 1.0 if phrase_l in cleaned_keywords else 0.0
        length_boost = min(len(phrase.split()) / 4.0, 1.0)

        score = (
            0.30 * tf_score
            + 0.20 * length_boost
            + 0.20 * title_boost
            + 0.10 * summary_boost
            + 0.20 * keyword_boost
        )

        return round(max(0.0, min(score, 1.0)), 4)

    def _link_concept(self, phrase: str) -> Dict[str, Any]:
        """
        Ưu tiên cosine similarity trên embedding nếu encoder được load.
        Fallback về Jaccard nếu không có embedding model.
        """
        phrase_emb = self._embed_text(phrase)
        if phrase_emb is not None:
            best_label = None
            best_sim = 0.0

            for item in self.ontology_concepts:
                label = normalize_text(item.get("label", ""))
                aliases = item.get("aliases", []) or item.get("alias", []) or []
                all_names = [label] + parse_list_like(aliases)

                for name in all_names:
                    name = normalize_text(name)
                    if not name:
                        continue

                    key = normalize_key(name)
                    if key in self._ontology_embed_cache:
                        name_emb = self._ontology_embed_cache[key]
                    else:
                        name_emb = self._embed_text(name)
                        if name_emb is not None:
                            self._ontology_embed_cache[key] = name_emb

                    sim = self._cosine_from_tensors(phrase_emb, name_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_label = label

            if best_sim >= self.concept_link_threshold:
                action = "link"
            elif best_sim >= self.concept_review_threshold:
                action = "review"
            else:
                action = "new"

            return {
                "linked_concept": best_label,
                "similarity": round(best_sim, 4),
                "action": action,
            }

        # Fallback lexical overlap
        phrase_norm = normalize_key(phrase)
        best_label = None
        best_sim = 0.0

        for item in self.ontology_concepts:
            label = normalize_text(item.get("label", ""))
            aliases = item.get("aliases", []) or item.get("alias", []) or []
            all_names = [label] + parse_list_like(aliases)

            for name in all_names:
                sim = jaccard_similarity(phrase_norm, name)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

        if best_sim >= self.concept_link_threshold:
            action = "link"
        elif best_sim >= self.concept_review_threshold:
            action = "review"
        else:
            action = "new"

        return {
            "linked_concept": best_label,
            "similarity": round(best_sim, 4),
            "action": action,
        }

    def _extract_concepts(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = self._extract_candidate_phrases(record)
        out = []

        for c in candidates:
            phrase = c["phrase"]
            score  = self._score_concept(phrase, record)
            if score < self.concept_min_score:
                continue

            link_info = self._link_concept(phrase)
            out.append(ConceptMention(
                concept=phrase,
                norm_concept=normalize_key(phrase),
                score=score,
                source=c["source"],
                evidence=c["evidence"],
                linked_concept=link_info["linked_concept"],
                similarity=link_info["similarity"],
                action=link_info["action"],
            ).to_dict())

        out = sorted(out, key=lambda x: x["score"], reverse=True)
        out = out[:self.max_concepts]   # ← THÊM GIỚI HẠN
        return out

    # =====================================================
    # Process single record
    # =====================================================
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        title = normalize_text(record.get("title", ""))
        summary = normalize_text(record.get("summary", ""))
        content = normalize_text(record.get("content", ""))
        full_text = normalize_text(record.get("full_text", ""))
        full_text_segmented = normalize_text(record.get("full_text_segmented", ""))

        if not full_text_segmented and full_text:
            full_text_segmented = segment_text_for_phobert(full_text)

        sentences = record.get("full_text_sentences", [])
        if not isinstance(sentences, list) or not sentences:
            sentences = split_sentences(full_text)

        # =====================================================
        # NER extraction from 3 backends
        # =====================================================
        mentions: List[NERMention] = []

        for i, sent in enumerate(sentences):
            sent_segmented = segment_text_for_phobert(sent)
            mentions.extend(self._extract_underthesea_ner(sent, i))
            mentions.extend(self._extract_phobert_ner(sent, i, segmented_sentence=sent_segmented))
            mentions.extend(self._extract_phonlp_ner(sent, i))

        ner_entities = self._merge_ner_mentions(mentions)

        # =====================================================
        # Concept extraction
        # =====================================================
        enriched_record = dict(record)
        enriched_record["ner_entities"] = ner_entities
        concept_mentions = self._extract_concepts(enriched_record)

        return {
            "article_id": record.get("article_id", ""),
            "title": title,
            "summary": summary,
            "content": content,
            "publish_time": record.get("publish_time", None),
            "topic_list": record.get("topic_list", []),
            "cleaned_keywords": parse_list_like(record.get("cleaned_keywords", [])),
            "full_text": full_text,
            "full_text_segmented": full_text_segmented,
            "full_text_tokens": record.get("full_text_tokens", []),
            "full_text_sentences": sentences,
            "full_text_pos": record.get("full_text_pos", []),
            "ner_entities": ner_entities,
            "concept_mentions": concept_mentions,
            "entity_surface_forms": dedupe_preserve([x["surface"] for x in ner_entities]),
            "concept_surface_forms": dedupe_preserve([x["concept"] for x in concept_mentions]),
            "stats": {
                "n_entities": len(ner_entities),
                "n_concepts": len(concept_mentions),
            },
        }

    # =====================================================
    # Batch process
    # =====================================================
    def load_and_process_jsonl(
        self,
        input_jsonl: Union[str, Path],
        output_jsonl: Union[str, Path],
        error_log_jsonl: Union[str, Path],
    ) -> pd.DataFrame:
        input_path = Path(input_jsonl)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        records = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except Exception as e:
                    LOGGER.warning("Invalid JSON at line %d: %s", line_no, e)

        outputs = []
        errors = []

        for idx, rec in enumerate(records):
            missing = self.validate_record(rec)
            if missing:
                errors.append(
                    {
                        "row_index": idx,
                        "article_id": rec.get("article_id", ""),
                        "error": f"Missing required fields: {missing}",
                    }
                )
                continue

            try:
                out = self.process_record(rec)
                outputs.append(out)
            except Exception as e:
                LOGGER.exception("Failed processing row %d", idx)
                errors.append(
                    {
                        "row_index": idx,
                        "article_id": rec.get("article_id", ""),
                        "error": str(e),
                    }
                )

        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        error_path = Path(error_log_jsonl)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

        LOGGER.info("Processed %d articles", len(outputs))
        if errors:
            LOGGER.warning("Encountered %d errors", len(errors))

        return pd.DataFrame(outputs)


# =========================================================
# Load ontology concepts
# =========================================================
def load_ontology_concepts(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import os
    DATA = os.environ.get("OKG_DATA_DIR", "./data")
    input_jsonl = os.path.join(DATA, "preprocessed_articles.jsonl")
    output_jsonl = os.path.join(DATA, "module2_ner_concept.jsonl")
    error_log_jsonl = os.path.join(DATA, "module2_errors.jsonl")

    ontology_concepts_path = None
    ontology_concepts = []
    if ontology_concepts_path:
        ontology_concepts = load_ontology_concepts(ontology_concepts_path)

    # Chọn một trong hai:
    # phobert_model_name = "NlpHUST/ner-vietnamese-electra-base"
    phobert_model_name = "NlpHUST/ner-vietnamese-electra-base"
    # phobert_model_name = "vilm/phobert-base-v2-ner"

    extractor = Module2NerConceptExtractor(
        ontology_concepts=ontology_concepts,
        use_underthesea_ner=True,
        use_phobert_ner=True,
        phobert_model_name=phobert_model_name,
        use_phonlp=True,
        phonlp_save_dir="phonlp_model",
        concept_embedding_model_name="vinai/phobert-base-v2",
        concept_link_threshold=0.85,
        concept_review_threshold=0.65,
        concept_min_score=0.50,    # tăng từ 0.35
        max_concept_tokens=6,
        max_concepts=20,           # thêm mới
    )

    df_out = extractor.load_and_process_jsonl(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        error_log_jsonl=error_log_jsonl,
    )

    print(f"Processed {len(df_out)} articles -> {output_jsonl}")
