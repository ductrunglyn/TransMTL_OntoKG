# data_utils.py  (đã thêm article_id để truy vấn OntoKG/Neo4j)
import os
import logging
import re
from typing import Tuple, List, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import ByteLevelBPETokenizer
from conf import IGNORE_INDEX, LABELS, LEN_IN, LEN_OUT, PAD_IDX
from .preprocessing import *
logger = logging.getLogger("data_utils")
logger.setLevel(logging.INFO)
# -------------------------
# Dataset: dùng BPE tokenizer đã train; lưu token->word map & word_texts để chuyển nhãn
# -------------------------
class MultiTaskDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer_dir: str = "vocab_subword", max_len_in: int = LEN_IN, max_len_out: int = LEN_OUT,
                 add_special_tokens: bool = True, allow_overlapping: bool = False, vocab_size: int = 30000, min_frequency: int = 2):
        super().__init__()
        # ensure tokenizer exists (train if needed)
        ensure_tokenizer_for_csv(csv_file, vocab_size=vocab_size, min_frequency=min_frequency, tokenizer_dir=tokenizer_dir)
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        merges_path = os.path.join(tokenizer_dir, "merges.txt")
        if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
            raise FileNotFoundError(f"Tokenizer files not found in '{tokenizer_dir}'. Ensure tokenizer exists or call ensure_tokenizer_for_csv() first.")
        tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
        self.tokenizer = tokenizer
        df = pd.read_csv(csv_file)
        text_col = "content"
        summary_col = "summary"
        keywords_col = "keywords"

        # NEW: article_id để truy vấn subgraph trong Neo4j (OntoKG).
        # split_dataset.py đã tạo sẵn cột này; nếu thiếu thì sinh theo chỉ số hàng.
        if "article_id" in df.columns:
            self.article_ids: List[str] = [str(x) for x in df["article_id"].tolist()]
        else:
            self.article_ids = [f"row_{i:06d}" for i in range(len(df))]

        texts_tokens: List[List[str]] = []
        summaries_tokens: List[List[str]] = []
        keywords_tokenized: List[List[List[str]]] = []
        # maps per sample for evaluation
        self.raw_texts: List[str] = []
        self.token_to_word_maps: List[List[int]] = []
        self.word_texts_list: List[List[str]] = []
        # NEW: store token offsets (char start,end) for each subword token
        self.token_offsets_list: List[List[Tuple[int, int]]] = []
        logger.info("Tokenizing dataset texts with BPE tokenizer...")
        for _, row in df.iterrows():
            txt = normalize_text(row.get(text_col, ""))
            sumry = normalize_text(row.get(summary_col, ""))
            raw_kws = str(row.get(keywords_col, ""))
            enc_txt = tokenizer.encode(txt)
            t_tokens = enc_txt.tokens
            t_offsets = enc_txt.offsets  # list of (start, end)
            if not t_tokens:
                t_tokens = [tokenizer.token_to_str(tokenizer.token_to_id("<unk>"))] if tokenizer.token_to_id("<unk>") is not None else ["<unk>"]
                t_offsets = [(0, len(txt))]
            # --- IMPORTANT: truncate tokens and offsets *together* to keep them aligned ---
            if len(t_tokens) > max_len_in:
                t_tokens = t_tokens[:max_len_in]
                t_offsets = t_offsets[:max_len_in]
            texts_tokens.append(t_tokens)
            self.raw_texts.append(txt)
            self.token_offsets_list.append(t_offsets)
            # compute word spans and token->word map
            words_spans = get_word_spans(txt)
            word_texts = [w for (w, s, e) in words_spans]
            self.word_texts_list.append(word_texts)
            token_to_word = []
            for (ts, te) in t_offsets:
                # find word that contains ts (or overlaps)
                w_idx = None
                for idx_w, (_, ws, we) in enumerate(words_spans):
                    if ts >= ws and te <= we:
                        w_idx = idx_w
                        break
                    # fallback if token spans overlap partially
                    if (ts < we and te > ws):
                        w_idx = idx_w
                        break
                if w_idx is None:
                    # fallback to nearest word by start position
                    nearest = None
                    min_dist = None
                    for idx_w, (_, ws, we) in enumerate(words_spans):
                        d = abs(ts - ws)
                        if min_dist is None or d < min_dist:
                            min_dist = d
                            nearest = idx_w
                    w_idx = nearest if nearest is not None else 0
                token_to_word.append(int(w_idx))
            self.token_to_word_maps.append(token_to_word)
            # summary tokens
            enc_sum = tokenizer.encode(sumry)
            s_tokens = enc_sum.tokens
            summaries_tokens.append(s_tokens)
            # keywords tokenized to subword tokens (for convenience - not used for matching)
            kw_list = []
            if raw_kws:
                for kw in raw_kws.split(","):
                    kw = kw.strip()
                    if not kw:
                        continue
                    enc_kw = tokenizer.encode(normalize_text(kw))
                    kw_tokens = enc_kw.tokens
                    if kw_tokens:
                        kw_list.append(kw_tokens)
            keywords_tokenized.append(kw_list)
        # vocab map token->id
        try:
            vocab_map = tokenizer.get_vocab()
        except Exception:
            vocab_map = {}
            cur = 0
            for toks in texts_tokens + summaries_tokens:
                for t in toks:
                    if t not in vocab_map:
                        vocab_map[t] = cur
                        cur += 1
        max_id = max(vocab_map.values()) if vocab_map else -1
        idx2word = [None] * (max_id + 1)
        for tok, idx in vocab_map.items():
            if 0 <= idx < len(idx2word):
                idx2word[idx] = tok
        for i in range(len(idx2word)):
            if idx2word[i] is None:
                idx2word[i] = f"<unk_{i}>"
        # store attrs
        self.emb_matrix = None
        self.word2idx = vocab_map
        self.idx2word = idx2word
        self.vocab_size = len(idx2word)
        pad_id = tokenizer.token_to_id("<pad>")
        unk_id = tokenizer.token_to_id("<unk>")
        sos_id = tokenizer.token_to_id("<sos>")
        eos_id = tokenizer.token_to_id("<eos>")
        self.pad_idx = int(pad_id) if pad_id is not None else PAD_IDX
        self.unk_idx = int(unk_id) if unk_id is not None else 1
        self.cls_idx = int(sos_id) if sos_id is not None else None
        self.sep_idx = int(eos_id) if eos_id is not None else None
        self.add_special_tokens = add_special_tokens
        self.allow_overlapping = allow_overlapping
        logger.info(f"✅ Tokenizer vocab_size={self.vocab_size}; pad={self.pad_idx} unk={self.unk_idx}")
        # convert texts -> ids and labels (subword-level BIOES)
        self.inputs: List[List[int]] = []
        self.attention_masks: List[List[int]] = []
        self.labels: List[List[int]] = []
        self.summaries_ids: List[List[int]] = []
        self.keyword_maps: List[List[str]] = []
        for i, t_tokens in enumerate(texts_tokens):
            ids = [ self.word2idx.get(t, self.unk_idx) for t in t_tokens ]
            if not ids:
                ids = [self.unk_idx]
            if len(ids) > max_len_in:
                ids = ids[:max_len_in]
                t_tokens = t_tokens[:max_len_in]
            attn = [1] * len(ids)
            # Use char-span based matching to create subword-level BIOES labels
            kw_list = keywords_tokenized[i]
            token_offsets = self.token_offsets_list[i]
            raw_txt = self.raw_texts[i]
            labels_tok, kw_spans = self._keywords_to_bioes_subword(t_tokens, token_offsets, raw_txt, kw_list, LABELS, allow_overlapping=self.allow_overlapping)
            if len(labels_tok) > max_len_in:
                labels_tok = labels_tok[:max_len_in]
            s_toks = summaries_tokens[i]
            s_ids = [ self.word2idx.get(t, self.unk_idx) for t in s_toks ]
            if add_special_tokens:
                s_out = []
                if self.cls_idx is not None:
                    s_out.append(self.cls_idx)
                s_out.extend(s_ids)
                if self.sep_idx is not None:
                    s_out.append(self.sep_idx)
                s_ids = s_out
            if not s_ids:
                s_ids = [self.unk_idx]
            if len(s_ids) > max_len_out:
                s_ids = s_ids[:max_len_out]
            self.inputs.append(ids)
            self.attention_masks.append(attn)
            self.labels.append(labels_tok)
            self.summaries_ids.append(s_ids)
            self.keyword_maps.append(kw_spans)
    def _keywords_to_bioes_subword(self, tokens: List[str], token_offsets: List[Tuple[int, int]], raw_text: str,
                                   keywords: List[List[str]], labels2idx: Dict[str,int], allow_overlapping: bool = False):
        n = len(tokens)
        O = labels2idx["O"]; B = labels2idx["B"]; I = labels2idx["I"]; E = labels2idx["E"]; S = labels2idx["S"]
        labels = [O] * n
        keyword_spans = []
        # Build keyword strings (approx) from token lists if available
        raw_kws = []
        for kw in keywords:
            if isinstance(kw, list):
                raw_kws.append(" ".join(kw))
            else:
                raw_kws.append(str(kw))
        for kw in raw_kws:
            kw_norm = normalize_keyword_for_matching(kw)
            if not kw_norm:
                keyword_spans.append("")
                continue
            spans = find_token_span_for_keyword(raw_text, token_offsets, kw_norm)
            if not spans:
                kw_loose = re.sub(r'[^\w\s]', ' ', kw_norm)
                kw_loose = re.sub(r'\s+', ' ', kw_loose).strip()
                if kw_loose and kw_loose != kw_norm:
                    spans = find_token_span_for_keyword(raw_text, token_offsets, kw_loose)
            if not spans:
                keyword_spans.append("")
                continue
            # take first match
            t0, t1 = spans[0]
            # SAFETY: if t0 is outside current truncated tokens, skip this match
            if t0 >= n:
                # cannot label because tokens truncated -- skip match
                keyword_spans.append("")
                continue
            # clip t1 to last available token index
            if t1 >= n:
                t1 = n - 1
                if t1 < t0:
                    keyword_spans.append("")
                    continue
            L = t1 - t0 + 1
            if L == 1:
                labels[t0] = S
            else:
                labels[t0] = B
                for j in range(t0 + 1, t1):
                    if 0 <= j < n:
                        labels[j] = I
                labels[t1] = E
            keyword_spans.append(" ".join([str(k + 1) for k in range(t0, t1 + 1)]))
        return labels, keyword_spans
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        # return tensors + raw text + maps + article_id (cho truy vấn OntoKG)
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.summaries_ids[idx], dtype=torch.long),
            torch.tensor(self.attention_masks[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.raw_texts[idx],
            self.token_to_word_maps[idx],
            self.word_texts_list[idx],
            self.article_ids[idx],          # NEW (phần tử thứ 8)
        )
# Collator: dynamic padding by batch + pass-through lists of strings/lists
class CollateCPU:
    def __init__(self, pad_idx: int, vocab_size: int, ignore_idx: int = IGNORE_INDEX):
        self.pad_idx = int(pad_idx)
        self.vocab_size = int(vocab_size)
        self.ignore_idx = ignore_idx
    def __call__(self, batch):
        # each item: src, s_ids, attn, labels, raw_text, token_to_word_map, word_texts, article_id
        src_batch, s_batch, attn_batch, lbl_batch, raw_texts, token_maps, word_texts, article_ids = zip(*batch)
        src = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        s_ids = pad_sequence(s_batch, batch_first=True, padding_value=self.pad_idx)
        attn = pad_sequence(attn_batch, batch_first=True, padding_value=0)
        labels = pad_sequence(lbl_batch, batch_first=True, padding_value=self.ignore_idx)
        # sanitize out-of-range
        try:
            bad_mask = (src < 0) | (src >= self.vocab_size)
            if bad_mask.any():
                src = src.masked_fill(bad_mask, self.pad_idx)
                attn = attn.masked_fill(bad_mask, 0)
        except Exception:
            pass
        # token_maps and word_texts are lists of lists; raw_texts/article_ids là list strings
        return src, s_ids, attn, labels, list(raw_texts), list(token_maps), list(word_texts), list(article_ids)
# -------------------------
# get_loaders: đảm bảo tokenizer được tạo từ CSV trước khi tạo dataset
# -------------------------
def get_loaders(data_path, len_in, len_out, num_workers, batch_size, val_ratio=0.2, test_ratio=0.2,
                seed=42, tokenizer_dir: str = "vocab_subword", vocab_size: int = 40000,
                min_frequency: int = 2, min_freq: int = None):
    # backward-compatible alias: nếu caller truyền min_freq (từ train.py cũ), dùng nó
    if min_freq is not None:
        min_frequency = int(min_freq)
    else:
        min_frequency = int(min_frequency)
    # ensure tokenizer exists (created from same CSV)
    ensure_tokenizer_for_csv(data_path, vocab_size=vocab_size, min_frequency=min_frequency, tokenizer_dir=tokenizer_dir)
    ds = MultiTaskDataset(data_path, tokenizer_dir=tokenizer_dir, max_len_in=len_in, max_len_out=len_out,
                          add_special_tokens=True, allow_overlapping=False, vocab_size=vocab_size, min_frequency=min_frequency)
    n = len(ds)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough samples to split train/val/test with given ratios.")
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(ds, [n_train, n_val, n_test], generator=generator)
    pad_idx = ds.pad_idx
    vocab_size = ds.vocab_size
    word2idx = ds.word2idx
    idx2word = ds.idx2word
    emb_matrix = ds.emb_matrix
    collate_fn = CollateCPU(pad_idx=pad_idx, vocab_size=vocab_size)
    using_cuda = torch.cuda.is_available()
    pin_memory = True if using_cuda else False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, vocab_size, pad_idx, word2idx, idx2word, emb_matrix, ds