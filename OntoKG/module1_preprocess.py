from __future__ import annotations

import ast
import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

try:
    from underthesea import text_normalize as uts_text_normalize
except Exception:
    uts_text_normalize = None

try:
    from underthesea import word_tokenize as uts_word_tokenize
except Exception:
    uts_word_tokenize = None

try:
    from underthesea import pos_tag as uts_pos_tag
except Exception:
    uts_pos_tag = None

try:
    from underthesea import sent_tokenize as uts_sent_tokenize
except Exception:
    uts_sent_tokenize = None


# =========================
# Logging
# =========================
LOGGER = logging.getLogger("module1_preprocess")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    LOGGER.addHandler(stream_handler)


# =========================
# Regex
# =========================
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
QUOTE_RE = re.compile(r'^[\'"“”‘’]+|[\'"“”‘’]+$')
BRACKET_RE = re.compile(r"^[\[\(\{]\s*|\s*[\]\)\}]$")


REQUIRED_COLUMNS = [
    "title",
    "summary",
    "content",
    "publish_time",
    "topic",
    "cleaned_keywords",
]

OPTIONAL_COLUMNS = [
    "article_id",
]


# =========================
# Helper functions
# =========================
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def normalize_text_field(value: Any) -> str:
    """
    Chuẩn hóa một trường text:
    - xử lý missing
    - unescape HTML
    - chuẩn hóa Unicode NFC
    - xóa control chars
    - bỏ HTML tags
    - gộp khoảng trắng
    """
    if _is_missing(value):
        return ""

    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)

    if uts_text_normalize is not None:
        try:
            text = uts_text_normalize(text)
        except Exception:
            pass

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_keyword_item(value: Any) -> str:
    """
    Chuẩn hóa keyword phrase nhưng không ép lower-case để giữ nguyên tên riêng.
    """
    text = normalize_text_field(value)
    if not text:
        return ""
    text = QUOTE_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = text.strip(" ,;:|/\\")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_topic_item(value: Any) -> str:
    """
    Chuẩn hóa topic thành token ổn định.
    Ví dụ:
      "con nguoi" -> "con_nguoi"
      "Giáo dục"  -> "giáo_dục"
    """
    text = normalize_text_field(value)
    if not text:
        return ""
    text = QUOTE_RE.sub("", text)
    text = text.strip().lower()
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = WHITESPACE_RE.sub("_", text)
    return text


def parse_list_like(value: Any) -> List[str]:
    """
    Parse các dạng:
    - list/tuple/set
    - string kiểu "['a', 'b']"
    - string ngăn cách bằng dấu phẩy/chấm phẩy
    - scalar
    """
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        parsed = None
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None

        if isinstance(parsed, (list, tuple, set)):
            items = list(parsed)
        elif parsed is not None and not isinstance(parsed, (dict, bytes)):
            items = [parsed]
        else:
            items = re.split(r"\s*[,;|]\s*", text)
    else:
        items = [value]

    cleaned = []
    for item in items:
        s = normalize_text_field(item)
        if s:
            cleaned.append(s)
    return cleaned


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def split_sentences(text: str) -> List[str]:
    text = normalize_text_field(text)
    if not text:
        return []

    if uts_sent_tokenize is not None:
        try:
            sents = uts_sent_tokenize(text)
            if isinstance(sents, str):
                sents = [sents]
            out = [normalize_text_field(s) for s in sents if normalize_text_field(s)]
            if out:
                return out
        except Exception:
            pass

    chunks = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [normalize_text_field(c) for c in chunks if normalize_text_field(c)]


def segment_vietnamese(text: str) -> str:
    """
    Trả về chuỗi đã tách từ kiểu underthesea, giữ dấu gạch dưới.
    Dùng cho PhoBERT / module 2.
    """
    text = normalize_text_field(text)
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
                out = " ".join(normalize_text_field(tok) for tok in out if normalize_text_field(tok))
            else:
                out = normalize_text_field(out)
            return WHITESPACE_RE.sub(" ", out).strip()

    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize_vietnamese(text: str) -> List[str]:
    text = normalize_text_field(text)
    if not text:
        return []

    segmented = segment_vietnamese(text)
    if segmented:
        return [tok for tok in segmented.split(" ") if tok]

    return [tok for tok in text.split(" ") if tok]


def pos_tag_vietnamese(text: str) -> List[Tuple[str, str]]:
    text = normalize_text_field(text)
    if not text or uts_pos_tag is None:
        return []

    try:
        tagged = uts_pos_tag(text)
    except Exception:
        return []

    normalized = []
    for item in tagged:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            token = normalize_text_field(item[0])
            tag = normalize_text_field(item[1])
            if token:
                normalized.append((token, tag))
        elif isinstance(item, dict):
            token = normalize_text_field(item.get("word") or item.get("token") or "")
            tag = normalize_text_field(item.get("pos") or item.get("tag") or "")
            if token:
                normalized.append((token, tag))
    return normalized


def parse_publish_time(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None

    text = normalize_text_field(value)
    if not text:
        return None

    candidates = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]

    for fmt in candidates:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and "%H" not in fmt:
                return dt.date().isoformat()
            return dt.isoformat(sep=" ")
        except Exception:
            continue

    return text


def parse_cleaned_keywords(value: Any) -> List[str]:
    """
    Chỉ dùng cleaned_keywords.
    """
    items = parse_list_like(value)
    normalized = []
    seen = set()

    for kw in items:
        item = normalize_keyword_item(kw)
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(item)

    return normalized


# =========================
# Data model
# =========================
@dataclass
class ProcessedArticle:
    article_id: str

    title_raw: str
    summary_raw: str
    content_raw: str
    publish_time_raw: Any
    topic_raw: Any
    cleaned_keywords_raw: Any

    title: str = ""
    summary: str = ""
    content: str = ""
    full_text: str = ""
    full_text_segmented: str = ""
    publish_time: Optional[str] = None

    topic_list: List[str] = field(default_factory=list)
    cleaned_keywords: List[str] = field(default_factory=list)

    title_sentences: List[str] = field(default_factory=list)
    summary_sentences: List[str] = field(default_factory=list)
    content_sentences: List[str] = field(default_factory=list)
    full_text_sentences: List[str] = field(default_factory=list)

    title_tokens: List[str] = field(default_factory=list)
    summary_tokens: List[str] = field(default_factory=list)
    content_tokens: List[str] = field(default_factory=list)
    full_text_tokens: List[str] = field(default_factory=list)

    title_pos: List[Tuple[str, str]] = field(default_factory=list)
    summary_pos: List[Tuple[str, str]] = field(default_factory=list)
    content_pos: List[Tuple[str, str]] = field(default_factory=list)
    full_text_pos: List[Tuple[str, str]] = field(default_factory=list)

    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# Preprocessor
# =========================
class Module1Preprocessor:
    """
    Module 1: preprocessing only.
    """

    def __init__(
        self,
        use_pos: bool = True,
        include_summary_in_full_text: bool = True,
    ):
        self.use_pos = use_pos
        self.include_summary_in_full_text = include_summary_in_full_text

    def validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        return missing

    def _combine_full_text(self, title: str, summary: str, content: str) -> str:
        parts = [title]
        if self.include_summary_in_full_text and summary:
            parts.append(summary)
        if content:
            parts.append(content)
        return normalize_text_field(" ".join([p for p in parts if p]))

    def preprocess_row(self, row: Dict[str, Any], article_id: Optional[str] = None) -> ProcessedArticle:
        title = normalize_text_field(row.get("title", ""))
        summary = normalize_text_field(row.get("summary", ""))
        content = normalize_text_field(row.get("content", ""))
        publish_time = parse_publish_time(row.get("publish_time"))

        topic_raw = row.get("topic", "")
        cleaned_keywords_raw = row.get("cleaned_keywords", "")

        if article_id is None:
            article_id = str(row.get("article_id", "")).strip() or ""

        topic_list = parse_list_like(topic_raw)
        topic_list = [normalize_topic_item(t) for t in topic_list if normalize_topic_item(t)]
        topic_list = dedupe_preserve_order(topic_list)

        cleaned_keywords = parse_cleaned_keywords(cleaned_keywords_raw)

        full_text = self._combine_full_text(title, summary, content)
        full_text_segmented = segment_vietnamese(full_text)

        title_sentences = split_sentences(title)
        summary_sentences = split_sentences(summary)
        content_sentences = split_sentences(content)
        full_text_sentences = split_sentences(full_text)

        title_tokens = tokenize_vietnamese(title)
        summary_tokens = tokenize_vietnamese(summary)
        content_tokens = tokenize_vietnamese(content)
        full_text_tokens = tokenize_vietnamese(full_text)

        # Chỉ full_text_pos được Module 2 sử dụng (trích noun phrase). Bỏ POS cho
        # title/summary/content để tránh chạy pos_tag 4 lần -> nhanh ~4x.
        title_pos = []
        summary_pos = []
        content_pos = []
        full_text_pos = pos_tag_vietnamese(full_text) if self.use_pos else []

        stats = {
            "n_title_sentences": len(title_sentences),
            "n_summary_sentences": len(summary_sentences),
            "n_content_sentences": len(content_sentences),
            "n_full_text_sentences": len(full_text_sentences),
            "n_title_tokens": len(title_tokens),
            "n_summary_tokens": len(summary_tokens),
            "n_content_tokens": len(content_tokens),
            "n_full_text_tokens": len(full_text_tokens),
            "n_cleaned_keywords": len(cleaned_keywords),
            "n_topics": len(topic_list),
        }

        return ProcessedArticle(
            article_id=article_id,

            title_raw=title,
            summary_raw=summary,
            content_raw=content,
            publish_time_raw=row.get("publish_time"),
            topic_raw=topic_raw,
            cleaned_keywords_raw=cleaned_keywords_raw,

            title=title,
            summary=summary,
            content=content,
            full_text=full_text,
            full_text_segmented=full_text_segmented,
            publish_time=publish_time,

            topic_list=topic_list,
            cleaned_keywords=cleaned_keywords,

            title_sentences=title_sentences,
            summary_sentences=summary_sentences,
            content_sentences=content_sentences,
            full_text_sentences=full_text_sentences,

            title_tokens=title_tokens,
            summary_tokens=summary_tokens,
            content_tokens=content_tokens,
            full_text_tokens=full_text_tokens,

            title_pos=title_pos,
            summary_pos=summary_pos,
            content_pos=content_pos,
            full_text_pos=full_text_pos,

            stats=stats,
        )

    def preprocess_dataframe(self, df: pd.DataFrame, log_errors: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        records = []
        errors = []

        for idx, row in df.iterrows():
            try:
                article_id = str(row.get("article_id", "")).strip() or f"article_{idx:06d}"
                processed = self.preprocess_row(row.to_dict(), article_id=article_id)
                records.append(processed.to_dict())
            except Exception as e:
                err = {
                    "row_index": int(idx),
                    "error": str(e),
                    "row_data": row.to_dict(),
                }
                errors.append(err)
                if log_errors:
                    LOGGER.exception("Failed to preprocess row %s", idx)

        return pd.DataFrame.from_records(records), errors

    def load_and_preprocess_csv(
        self,
        input_csv: Union[str, Path],
        output_jsonl: Optional[Union[str, Path]] = None,
        error_log_jsonl: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        input_path = Path(input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path, engine="python", dtype=str, keep_default_na=False)

        missing_cols = self.validate_dataframe(df)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required columns are: {REQUIRED_COLUMNS}"
            )

        out_df, errors = self.preprocess_dataframe(df, log_errors=True)

        if output_jsonl is not None:
            out_path = Path(output_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for rec in out_df.to_dict(orient="records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if error_log_jsonl is not None:
            err_path = Path(error_log_jsonl)
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with err_path.open("w", encoding="utf-8") as f:
                for err in errors:
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")

        LOGGER.info("Processed %d articles", len(out_df))
        if errors:
            LOGGER.warning("Encountered %d errored rows", len(errors))

        return out_df


def preprocess_records(
    records: List[Dict[str, Any]],
    use_pos: bool = True,
    include_summary_in_full_text: bool = True,
) -> List[Dict[str, Any]]:
    pre = Module1Preprocessor(
        use_pos=use_pos,
        include_summary_in_full_text=include_summary_in_full_text,
    )
    output = []
    for idx, row in enumerate(records):
        article_id = str(row.get("article_id", "")).strip() or f"article_{idx:06d}"
        output.append(pre.preprocess_row(row, article_id=article_id).to_dict())
    return output


if __name__ == "__main__":
    import os
    no_pos = False
    include_summary_in_full_text = True
    DATA = os.environ.get("OKG_DATA_DIR", "./data")
    # Khi chạy qua main.py, biến OKG_INPUT_CSV được set tự động = data_split/trainval.csv
    # nên KHÔNG cần sửa path ở đây. Giá trị dưới chỉ là mặc định khi chạy module lẻ.
    input_csv = os.environ.get("OKG_INPUT_CSV", "./data_split/trainval.csv")
    output_jsonl = os.path.join(DATA, "preprocessed_articles.jsonl")
    error_log_jsonl = os.path.join(DATA, "preprocess_errors.jsonl")

    pre = Module1Preprocessor(
        use_pos=not no_pos,
        include_summary_in_full_text=include_summary_in_full_text,
    )

    df_out = pre.load_and_preprocess_csv(
        input_csv=input_csv,
        output_jsonl=output_jsonl,
        error_log_jsonl=error_log_jsonl,
    )
    print(f"Processed {len(df_out)} articles -> {output_jsonl}")