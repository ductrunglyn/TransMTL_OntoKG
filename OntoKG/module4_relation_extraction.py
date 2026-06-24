# module4_relation_extraction.py
"""
Module 4: Relation Extraction
Input : ./data/module3_entity_linked.jsonl  (output của Module 3)
Output: ./data/module4_triples.jsonl        (tất cả triple đã trích xuất)
        ./data/module4_enriched.jsonl       (record gốc + trường 'triples')

Hai loại triple được tạo ra:
  A. Metadata triple (tự động, không cần NLP):
       (article_uri, rdf:type, Article)
       (article_uri, belongsTo, topic_uri)
       (entity_uri, rdf:type, OntologyClass)
       (article_uri, hasEntity, entity_uri)

  B. Relation triple (từ NLP pattern matching):
       (head_uri, relation, tail_uri)
     Được trích xuất bằng cách:
       1. Tìm các entity xuất hiện trong cùng một câu
       2. Kiểm tra text giữa chúng có khớp pattern không
       3. Validate loại head/tail theo ontology rule
"""
from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
LOGGER = logging.getLogger("module4_relation_extraction")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)

# ──────────────────────────────────────────────────────────
# Namespace
# ──────────────────────────────────────────────────────────
ONTO_URI    = "http://transmtl.vn/onto"
ARTICLE_URI = "http://transmtl.vn/article"
TOPIC_URI   = "http://transmtl.vn/onto/topic"

RDF_TYPE    = "rdf:type"

# ──────────────────────────────────────────────────────────
# Ánh xạ NER label → ontology class URI
# ──────────────────────────────────────────────────────────
LABEL_TO_CLASS_URI: Dict[str, str] = {
    "PER":   f"{ONTO_URI}/Person",
    "ORG":   f"{ONTO_URI}/Organization",
    "LOC":   f"{ONTO_URI}/Location",
    "TIME":  f"{ONTO_URI}/TimeRef",
    "EVENT": f"{ONTO_URI}/Event",
    "MISC":  f"{ONTO_URI}/Concept",
}

# ──────────────────────────────────────────────────────────
# Ràng buộc quan hệ: {relation: [(head_labels), (tail_labels)]}
# None = bất kỳ label nào đều hợp lệ
# ──────────────────────────────────────────────────────────
RELATION_CONSTRAINTS: Dict[str, Tuple[Optional[Set[str]], Optional[Set[str]]]] = {
    "occursAt":       ({"EVENT", "ORG"},          {"LOC"}),
    "occursOn":       ({"EVENT"},                  {"TIME"}),
    "organizedBy":    ({"EVENT"},                  {"ORG"}),
    "participatesIn": ({"PER"},                    {"EVENT"}),
    "locatedIn":      ({"LOC"},                    {"LOC"}),
    "hasPart":        (None,                       None),
    "relatedTo":      (None,                       None),
    "isA":            (None,                       None),
    "causedBy":       ({"EVENT"},                  {"EVENT", "ORG", "PER"}),
    "manages":        ({"PER"},                    {"ORG"}),
}

# ──────────────────────────────────────────────────────────
# Pattern quy tắc tiếng Việt
# Mỗi rule: {keywords, relation, head_types, tail_types, head_before_tail}
#   head_before_tail=True  → head xuất hiện trước keyword, tail sau
#   head_before_tail=False → tail xuất hiện trước keyword, head sau
# ──────────────────────────────────────────────────────────
RELATION_RULES: List[Dict[str, Any]] = [
    # ── occursAt ──────────────────────────────────────────
    {
        "keywords": [
            r"diễn ra (?:tại|ở|ở tại)",
            r"tổ chức (?:tại|ở)",
            r"xảy ra (?:tại|ở)",
            r"tiến hành (?:tại|ở)",
            r"được tổ chức (?:tại|ở)",
        ],
        "relation": "occursAt",
        "head_types": {"EVENT", "ORG"},
        "tail_types": {"LOC"},
        "head_before_tail": True,
    },
    # ── occursOn ──────────────────────────────────────────
    {
        "keywords": [
            r"diễn ra (?:vào|ngày|lúc)",
            r"tổ chức (?:vào|ngày)",
            r"(?:vào|ngày) \d",
            r"(?:tháng|năm) \d",
        ],
        "relation": "occursOn",
        "head_types": {"EVENT"},
        "tail_types": {"TIME"},
        "head_before_tail": True,
    },
    # ── organizedBy ───────────────────────────────────────
    {
        "keywords": [
            r"do .{2,30} tổ chức",
            r"được .{2,30} tổ chức",
            r"(?:phối hợp|chủ trì) tổ chức",
        ],
        "relation": "organizedBy",
        "head_types": {"EVENT"},
        "tail_types": {"ORG"},
        "head_before_tail": True,
    },
    # ── participatesIn ────────────────────────────────────
    {
        "keywords": [
            r"tham gia",
            r"tham dự",
            r"có mặt (?:tại|trong)",
            r"dự (?:hội nghị|hội thảo|lễ|cuộc)",
        ],
        "relation": "participatesIn",
        "head_types": {"PER"},
        "tail_types": {"EVENT"},
        "head_before_tail": True,
    },
    # ── locatedIn ─────────────────────────────────────────
    {
        "keywords": [
            r"thuộc (?:tỉnh|thành phố|huyện|quận|thị xã)",
            r"nằm (?:ở|tại|trong)",
            r"ở (?:tỉnh|thành phố|huyện)",
        ],
        "relation": "locatedIn",
        "head_types": {"LOC", "ORG"},
        "tail_types": {"LOC"},
        "head_before_tail": True,
    },
    # ── manages ───────────────────────────────────────────
    {
        "keywords": [
            r"(?:chủ tịch|giám đốc|tổng giám đốc|trưởng|phó|bộ trưởng|thủ tướng) .{0,20}của",
            r"lãnh đạo",
            r"đứng đầu",
        ],
        "relation": "manages",
        "head_types": {"PER"},
        "tail_types": {"ORG"},
        "head_before_tail": True,
    },
    # ── hasPart ───────────────────────────────────────────
    {
        "keywords": [
            r"bao gồm",
            r"gồm có",
            r"gồm các",
            r"có các môn",
        ],
        "relation": "hasPart",
        "head_types": None,
        "tail_types": None,
        "head_before_tail": True,
    },
    # ── causedBy ──────────────────────────────────────────
    {
        "keywords": [
            r"do .{2,30} gây ra",
            r"nguyên nhân (?:là|do)",
            r"(?:vì|bởi vì|do)",
        ],
        "relation": "causedBy",
        "head_types": {"EVENT"},
        "tail_types": {"EVENT", "ORG", "PER"},
        "head_before_tail": True,
    },
    # ── relatedTo (chung) ─────────────────────────────────
    {
        "keywords": [
            r"liên quan (?:đến|tới)",
            r"có liên quan",
        ],
        "relation": "relatedTo",
        "head_types": None,
        "tail_types": None,
        "head_before_tail": True,
    },
]

# Compile patterns một lần
for rule in RELATION_RULES:
    rule["_compiled"] = [re.compile(kw, re.IGNORECASE | re.UNICODE)
                         for kw in rule["keywords"]]


# ──────────────────────────────────────────────────────────
# Dataclass Triple
# ──────────────────────────────────────────────────────────
@dataclass
class Triple:
    head_uri:     str
    relation:     str
    tail_uri:     str
    head_surface: str
    tail_surface: str
    confidence:   float
    method:       str    # "metadata" | "rule"
    article_id:   str
    sentence_idx: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────
# Tiện ích
# ──────────────────────────────────────────────────────────
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE    = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def validate_triple(
    head_label: str,
    relation: str,
    tail_label: str,
) -> bool:
    """Kiểm tra (head_label, relation, tail_label) có hợp lệ theo ontology không."""
    constraint = RELATION_CONSTRAINTS.get(relation)
    if constraint is None:
        return False
    valid_heads, valid_tails = constraint
    if valid_heads is not None and head_label not in valid_heads:
        return False
    if valid_tails is not None and tail_label not in valid_tails:
        return False
    return True


def entity_positions_in_sentence(
    sentence: str,
    entities: List[Dict[str, Any]],
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    Tìm vị trí (start, end) của từng entity trong câu.
    Trả về list (start, end, entity_dict) sắp xếp theo start.
    """
    sentence_lower = sentence.lower()
    positions = []
    for ent in entities:
        surface = normalize_text(ent.get("surface", ""))
        if not surface:
            continue
        idx = sentence_lower.find(surface.lower())
        if idx >= 0:
            positions.append((idx, idx + len(surface), ent))
    positions.sort(key=lambda x: x[0])
    return positions


# ──────────────────────────────────────────────────────────
# Module 4 — Main Extractor
# ──────────────────────────────────────────────────────────
class Module4RelationExtractor:

    def __init__(self, min_confidence: float = 0.70):
        self.min_confidence = min_confidence

    # ── A. Metadata triples ────────────────────────────────
    def _make_metadata_triples(
        self,
        record: Dict[str, Any],
        article_id: str,
        article_uri: str,
    ) -> List[Triple]:
        triples: List[Triple] = []

        # (article, rdf:type, Article)
        triples.append(Triple(
            head_uri=article_uri,
            relation=RDF_TYPE,
            tail_uri=f"{ONTO_URI}/Article",
            head_surface=article_id,
            tail_surface="Article",
            confidence=1.0,
            method="metadata",
            article_id=article_id,
        ))

        # (article, belongsTo, topic)
        for topic_uri in record.get("topic_uris", []):
            triples.append(Triple(
                head_uri=article_uri,
                relation="belongsTo",
                tail_uri=topic_uri,
                head_surface=article_id,
                tail_surface=topic_uri.split("/")[-1],
                confidence=1.0,
                method="metadata",
                article_id=article_id,
            ))

        # Cho mỗi entity trong bài
        all_entities = list(record.get("ner_entities", []))
        for ent in all_entities:
            uri = ent.get("uri")
            if not uri:
                continue
            label    = ent.get("label", "MISC")
            surface  = ent.get("surface", "")
            class_uri = LABEL_TO_CLASS_URI.get(label, f"{ONTO_URI}/NewsEntity")

            # (entity, rdf:type, OntologyClass)
            triples.append(Triple(
                head_uri=uri,
                relation=RDF_TYPE,
                tail_uri=class_uri,
                head_surface=surface,
                tail_surface=label,
                confidence=1.0,
                method="metadata",
                article_id=article_id,
            ))

            # (article, hasEntity, entity)
            triples.append(Triple(
                head_uri=article_uri,
                relation="hasEntity",
                tail_uri=uri,
                head_surface=article_id,
                tail_surface=surface,
                confidence=1.0,
                method="metadata",
                article_id=article_id,
            ))

            # (entity, belongsTo, topic) — nếu entity có topic hint từ label_name
            for topic_uri in record.get("topic_uris", []):
                triples.append(Triple(
                    head_uri=uri,
                    relation="belongsTo",
                    tail_uri=topic_uri,
                    head_surface=surface,
                    tail_surface=topic_uri.split("/")[-1],
                    confidence=0.80,
                    method="metadata",
                    article_id=article_id,
                ))

        return triples

    # ── B. Relation triples từ NLP patterns ────────────────
    def _extract_relation_triples(
        self,
        record: Dict[str, Any],
        article_id: str,
    ) -> List[Triple]:
        triples: List[Triple] = []
        entities = record.get("ner_entities", [])
        # Chỉ lấy entity có URI
        entities = [e for e in entities if e.get("uri")]

        sentences = record.get("full_text_sentences", [])

        for sent_idx, sentence in enumerate(sentences):
            sentence = normalize_text(sentence)
            if not sentence:
                continue

            # Tìm entity có mặt trong câu này
            positions = entity_positions_in_sentence(sentence, entities)
            if len(positions) < 2:
                continue   # cần ít nhất 2 entity để tạo quan hệ

            # Duyệt qua các cặp entity liền kề / trong cùng câu
            for i in range(len(positions)):
                for j in range(i + 1, min(i + 4, len(positions))):
                    s1, e1, ent1 = positions[i]
                    s2, e2, ent2 = positions[j]

                    # Text nằm giữa hai entity (bridge text)
                    bridge = sentence[e1:s2].strip() if s2 > e1 else ""
                    # Text toàn câu cũng dùng để match pattern rộng hơn
                    context = sentence

                    for rule in RELATION_RULES:
                        matched = False
                        for pat in rule["_compiled"]:
                            if pat.search(bridge) or pat.search(context):
                                matched = True
                                break
                        if not matched:
                            continue

                        # Xác định head/tail theo hướng của rule
                        if rule["head_before_tail"]:
                            head, tail = ent1, ent2
                        else:
                            head, tail = ent2, ent1

                        head_label = head.get("label", "MISC")
                        tail_label = tail.get("label", "MISC")
                        relation   = rule["relation"]

                        if not validate_triple(head_label, relation, tail_label):
                            continue

                        # Tránh self-loop
                        if head["uri"] == tail["uri"]:
                            continue

                        conf = min(head.get("confidence", 0.9),
                                   tail.get("confidence", 0.9))
                        if conf < self.min_confidence:
                            continue

                        triples.append(Triple(
                            head_uri=head["uri"],
                            relation=relation,
                            tail_uri=tail["uri"],
                            head_surface=head.get("surface", ""),
                            tail_surface=tail.get("surface", ""),
                            confidence=round(conf, 4),
                            method="rule",
                            article_id=article_id,
                            sentence_idx=sent_idx,
                        ))

        # Dedup (giữ triple duy nhất theo (head, relation, tail))
        seen: Set[Tuple[str, str, str]] = set()
        unique: List[Triple] = []
        for t in triples:
            key = (t.head_uri, t.relation, t.tail_uri)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    # ── Xử lý một record ────────────────────────────────────
    def process_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Triple]]:
        article_id  = record.get("article_id", "")
        article_uri = record.get("article_uri", f"http://transmtl.vn/article/{article_id}")

        metadata_triples = self._make_metadata_triples(record, article_id, article_uri)
        relation_triples = self._extract_relation_triples(record, article_id)

        all_triples = metadata_triples + relation_triples

        enriched = dict(record)
        enriched["triples"] = [t.to_dict() for t in all_triples]
        enriched["stats"]["n_metadata_triples"] = len(metadata_triples)
        enriched["stats"]["n_relation_triples"] = len(relation_triples)

        return enriched, all_triples

    # ── Xử lý batch ─────────────────────────────────────────
    def extract_all(
        self,
        input_jsonl: str,
        output_triples_jsonl: str,
        output_enriched_jsonl: str,
    ):
        input_path    = Path(input_jsonl)
        triples_path  = Path(output_triples_jsonl)
        enriched_path = Path(output_enriched_jsonl)
        triples_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_path.parent.mkdir(parents=True, exist_ok=True)

        total_metadata = total_relation = processed = errors = 0

        with input_path.open("r", encoding="utf-8") as fin, \
             triples_path.open("w", encoding="utf-8") as ft, \
             enriched_path.open("w", encoding="utf-8") as fe:

            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record            = json.loads(line)
                    enriched, triples = self.process_record(record)

                    fe.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    for t in triples:
                        ft.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

                    total_metadata += enriched["stats"].get("n_metadata_triples", 0)
                    total_relation += enriched["stats"].get("n_relation_triples", 0)
                    processed += 1
                except Exception as e:
                    LOGGER.exception("Failed at line %d: %s", line_no, e)
                    errors += 1

                if processed % 1000 == 0:
                    LOGGER.info(
                        "Progress: %d articles | meta=%d rel=%d triples",
                        processed, total_metadata, total_relation,
                    )

        LOGGER.info(
            "Done: %d articles, %d metadata triples, %d relation triples, %d errors",
            processed, total_metadata, total_relation, errors,
        )
        LOGGER.info("Triples → %s", output_triples_jsonl)
        LOGGER.info("Enriched → %s", output_enriched_jsonl)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    DATA = os.environ.get("OKG_DATA_DIR", "./data")
    extractor = Module4RelationExtractor(min_confidence=0.70)
    extractor.extract_all(
        input_jsonl=os.path.join(DATA, "module3_entity_linked.jsonl"),
        output_triples_jsonl=os.path.join(DATA, "module4_triples.jsonl"),
        output_enriched_jsonl=os.path.join(DATA, "module4_enriched.jsonl"),
    )