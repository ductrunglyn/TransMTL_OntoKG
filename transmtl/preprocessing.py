# train.py
import ast
import logging
import os
import numpy as np
import torch
import random
import torch.nn as nn
import unicodedata
import hashlib
import re
from typing import Tuple, List, Dict, Optional
import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer

from conf import LABELS

try:
    import fasttext
except ImportError:
    try:
        import fasttext_wheel as fasttext
    except ImportError:
        print("⚠️ Lỗi import fasttext. Vui lòng cài đặt.")

logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)

# --- 0. HÀM CỐ ĐỊNH SEED ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Đã cố định Seed = {seed}")

# -------------------------
# Utils: normalization + corpus creation + tokenizer training + checksum
# -------------------------
def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    return unicodedata.normalize("NFC", s)

def parse_keywords_field(value) -> List[str]:
    """
    Parse trường `cleaned_keywords` từ CSV gốc một cách thống nhất với
    OntoKG/module1_preprocess.parse_list_like. Hỗ trợ:
      - list/tuple/set Python
      - chuỗi kiểu "['a', 'b']"  (ast.literal_eval)
      - chuỗi ngăn cách bằng dấu , ; |
    Trả về list[str] đã strip, bỏ rỗng.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
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
        # scalar (vd float nan từ pandas) -> bỏ qua nếu là nan
        try:
            if value != value:  # NaN check
                return []
        except Exception:
            pass
        items = [value]
    out = []
    for it in items:
        s = str(it).strip()
        if s:
            out.append(s)
    return out


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_corpus_from_csv(csv_file, out_corpus_path, text_col="content", summary_col="summary", keywords_col="cleaned_keywords"):
    df = pd.read_csv(csv_file)
    if keywords_col not in df.columns:
        raise ValueError(
            f"CSV '{csv_file}' thiếu cột bắt buộc '{keywords_col}'. "
            f"Các cột hiện có: {list(df.columns)}. "
            f"Pipeline yêu cầu cột 'cleaned_keywords' (KHÔNG fallback sang 'keywords')."
        )
    os.makedirs(os.path.dirname(out_corpus_path) or ".", exist_ok=True)
    with open(out_corpus_path, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            c = normalize_text(row.get(text_col, ""))
            if c:
                fout.write(c + "\n")
            s = normalize_text(row.get(summary_col, ""))
            if s:
                fout.write(s + "\n")
            for kw in parse_keywords_field(row.get(keywords_col, "")):
                kwn = normalize_text(kw)
                if kwn:
                    fout.write(kwn + "\n")
    logger.info(f"✅ Built corpus at {out_corpus_path}")

def train_bpe_tokenizer(corpus_path, out_dir, vocab_size, min_frequency):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>", "<sep>"]
    tokenizer.train(files=[corpus_path],
                    vocab_size=vocab_size,
                    min_frequency=min_frequency,
                    special_tokens=special_tokens)
    tokenizer.save_model(out_dir)
    logger.info(f"✅ Trained BPE tokenizer and saved to {out_dir} (vocab.json + merges.txt)")

def ensure_tokenizer_for_csv(csv_file, vocab_size, min_frequency, tokenizer_dir="vocab_subword", corpus_name="corpus.txt"):
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.txt")
    corpus_path = os.path.join(tokenizer_dir, corpus_name)
    sha_path = os.path.join(tokenizer_dir, "corpus.sha256")

    # If tokenizer exists and checksum present, compare without rebuilding corpus.
    if os.path.exists(vocab_path) and os.path.exists(merges_path) and os.path.exists(sha_path):
        build_corpus_from_csv(csv_file, corpus_path)
        new_sha = compute_file_sha256(corpus_path)
        try:
            with open(sha_path, "r", encoding="utf-8") as f:
                old_sha = f.read().strip()
        except Exception:
            old_sha = None

        if old_sha == new_sha:
            logger.info("Tokenizer exists and corpus checksum unchanged -> skipping tokenizer training.")
            return tokenizer_dir
        else:
            logger.info("Tokenizer exists but corpus changed -> re-training tokenizer.")
            train_bpe_tokenizer(corpus_path, tokenizer_dir, vocab_size=vocab_size, min_frequency=min_frequency)
            with open(sha_path, "w", encoding="utf-8") as f:
                f.write(compute_file_sha256(corpus_path))
            return tokenizer_dir

    # If tokenizer files missing (or sha missing), create corpus and train
    logger.info(f"Tokenizer missing or incomplete in {tokenizer_dir}. Building corpus and training tokenizer...")
    build_corpus_from_csv(csv_file, corpus_path)
    train_bpe_tokenizer(corpus_path, tokenizer_dir, vocab_size=vocab_size, min_frequency=min_frequency)
    # write checksum
    try:
        sha = compute_file_sha256(corpus_path)
        with open(sha_path, "w", encoding="utf-8") as f:
            f.write(sha)
    except Exception:
        logger.warning("Could not write corpus checksum file.")
    return tokenizer_dir

# -------------------------
# Helper: get words spans from text (whitespace-based)
# -------------------------
def get_word_spans(text: str) -> List[Tuple[str, int, int]]:
    spans = []
    for m in re.finditer(r'\S+', text):
        spans.append((m.group(0), m.start(), m.end()))
    return spans

# -------------------------
# Normalize keyword for matching (strip surrounding punctuation, lower)
# -------------------------
def normalize_keyword_for_matching(kw: str) -> str:
    if kw is None:
        return ""
    s = unicodedata.normalize("NFC", str(kw)).strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.lower()
    # strip surrounding punctuation/quotes that often are not part of the phrase
    s = s.strip(" \"'“”‘’(),.:;!?—-[]{}<>")
    return s

def find_token_span_for_keyword(raw_text: str, token_offsets: List[Tuple[int, int]], keyword: str) -> List[Tuple[int, int]]:
    out_spans = []
    if not keyword:
        return out_spans
    text_proc = raw_text.lower()
    kw_proc = keyword.lower()
    start = 0
    while True:
        idx = text_proc.find(kw_proc, start)
        if idx == -1:
            break
        end_char = idx + len(kw_proc)
        tok_start = None
        tok_end = None
        for ti, (ts, te) in enumerate(token_offsets):
            # skip tokens entirely before span
            if te <= idx:
                continue
            # tokens entirely after span -> stop scanning
            if ts >= end_char:
                break
            # intersection exists
            if tok_start is None:
                tok_start = ti
            tok_end = ti
        if tok_start is not None:
            out_spans.append((tok_start, tok_end))
        start = idx + 1
    return out_spans

# -------------------------
# Convert subword BIOES -> word BIOES (for evaluation)
# -------------------------
def subword_labels_to_word_labels(subword_labels: List[int], token_to_word_map: List[int], num_words: int, labels_map: Dict[str,int] = LABELS) -> List[int]:
    O = labels_map["O"]; B = labels_map["B"]; I = labels_map["I"]; E = labels_map["E"]; S = labels_map["S"]
    word_labels = [O] * num_words
    n = len(subword_labels)
    i = 0
    while i < n:
        tag = int(subword_labels[i])
        if tag == O:
            i += 1
            continue
        # if single-token S
        if tag == S:
            w = token_to_word_map[i]
            if 0 <= w < num_words:
                word_labels[w] = S
            i += 1
            continue
        # if B: find end (E) or consume until a token labeled E is found
        if tag == B:
            j = i
            while j < n and int(subword_labels[j]) != E:
                j += 1
            if j >= n:
                # no explicit E found; treat span until last non-O token
                j = i
                while j + 1 < n and int(subword_labels[j + 1]) in (I,):
                    j += 1
            # token indices i..j inclusive form one keyword span
            w_start = token_to_word_map[i]
            w_end = token_to_word_map[j]
            if w_start == w_end:
                # entire keyword falls in one word -> S
                if 0 <= w_start < num_words:
                    word_labels[w_start] = S
            else:
                # multiple words
                if 0 <= w_start < num_words:
                    word_labels[w_start] = B
                for w in range(w_start + 1, w_end):
                    if 0 <= w < num_words:
                        word_labels[w] = I
                if 0 <= w_end < num_words:
                    word_labels[w_end] = E
            i = j + 1
            continue
        # Unexpected tag (I/E) at this point: try to skip
        i += 1
    return word_labels

# --- Helper Functions ---
def load_fasttext_bin_embeddings(word2idx, bin_path, d_model, pad_idx=0):
    """
    Tao ma tran embedding tu FastText .bin. (DA BO module tu dong nghia.)
    """

    if not os.path.exists(bin_path):
        print(f"LOI: Khong tim thay file vector tai '{bin_path}'")
        return None

    print(f"Dang load FastText model tu {bin_path}...")
    fasttext.FastText.eprint = lambda x: None
    ft_model = fasttext.load_model(bin_path)

    vec_dim = ft_model.get_dimension()
    if vec_dim != d_model:
        print(f"CANH BAO: Dimension model ({vec_dim}) KHAC d_model ({d_model}).")

    print("Dang tao ma tran embedding co so...")
    ft_vocab = set(ft_model.get_words())
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, d_model), dtype='float32')

    exact_cnt = compound_cnt = generated_cnt = 0

    for word, i in word2idx.items():
        if i == pad_idx:
            continue
        # Uu tien 1: Chinh xac
        if word in ft_vocab:
            embedding_matrix[i] = ft_model.get_word_vector(word)
            exact_cnt += 1
            continue
        # Uu tien 2: Tu ghep (dau _)
        if '_' in word:
            subwords = word.split('_')
            valid_vecs = []
            for sw in subwords:
                if sw in ft_vocab:
                    valid_vecs.append(ft_model.get_word_vector(sw))
                elif sw.lower() in ft_vocab:
                    valid_vecs.append(ft_model.get_word_vector(sw.lower()))
            if len(valid_vecs) == len(subwords):
                embedding_matrix[i] = np.mean(valid_vecs, axis=0)
                compound_cnt += 1
                continue
        # Uu tien 3: Generated
        embedding_matrix[i] = ft_model.get_word_vector(word)
        generated_cnt += 1

    del ft_model
    del ft_vocab

    total = exact_cnt + compound_cnt + generated_cnt
    print(f"Da tao vector: {total}/{vocab_size}")
    print(f"  Exact: {exact_cnt} | Compound: {compound_cnt} | Generated: {generated_cnt}")

    return embedding_matrix

def ids_to_text(ids: List[int], idx2word: List[str], pad_idx: int, cls_idx: Optional[int] = None, sep_idx: Optional[int] = None) -> str:
    words = []
    for i in ids:
        if i == pad_idx: continue
        if cls_idx is not None and i == cls_idx: continue
        if sep_idx is not None and i == sep_idx: continue
        if 0 <= int(i) < len(idx2word):
            w = idx2word[int(i)]
        else:
            w = "<unk>"
        words.append(w)
    return " ".join(words).strip()

def convert_tags_to_keyphrases(tag_seq: List[int], token_words: List[str]) -> List[str]:
    kws = []
    cur = []
    for tag, word in zip(tag_seq, token_words):
        if tag == 0: 
            if cur: kws.append(" ".join(cur)); cur = []
        elif tag == 4:
            if cur: kws.append(" ".join(cur)); cur = []
            kws.append(word)
        elif tag == 1:
            if cur: kws.append(" ".join(cur))
            cur = [word]
        elif tag == 2:
            if cur: cur.append(word)
            else: cur = [word] 
        elif tag == 3:
            if cur: cur.append(word); kws.append(" ".join(cur)); cur = []
            else: kws.append(word); cur = []
    if cur: kws.append(" ".join(cur))

    out = []
    for k in kws:
        kk = " ".join(k.strip().split()).lower()
        if kk: out.append(kk)
    seen = set(); final = []
    for x in out:
        if x not in seen: final.append(x); seen.add(x)
    return final

# Lưu ý: evaluate_keyphrase_lists() đã được gỡ khỏi đây (bản trùng, không dùng).
# Dùng bản chính thức trong transmtl/evaluation.py.

