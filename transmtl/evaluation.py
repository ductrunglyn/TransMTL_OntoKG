# evaluation.py
import numpy as np
from typing import List
from rouge_score import rouge_scorer
import logging
logger = logging.getLogger(__name__)

def normalize_phrase(phrase: str) -> str:
    p = phrase.replace("_", " ").lower().strip()
    p = " ".join(p.split())
    return p

def unique_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def token_labels_to_keyphrases(decoded_seq: List[int], token_words: List[str]):
    """
    Convert numeric BIOES sequence + token words -> list of keyphrase strings
    decoded_seq: list[int] tags (0..4)
    token_words: list[str] words aligned
    """
    L = min(len(decoded_seq), len(token_words))
    kws = []
    i = 0
    cur = []
    while i < L:
        tag = int(decoded_seq[i])
        if tag == 4:  # S
            if cur:
                kws.append(" ".join(cur)); cur=[]
            kws.append(token_words[i])
            i += 1
            continue
        if tag == 1:  # B
            if cur:
                kws.append(" ".join(cur)); cur=[]
            cur = [token_words[i]]
            j = i+1
            while j < L and decoded_seq[j] in (2,3):
                cur.append(token_words[j])
                if decoded_seq[j] == 3:
                    break
                j += 1
            kws.append(" ".join(cur))
            cur = []
            i = j+1
            continue
        # O or I/E unexpected handling
        i += 1
    kws = [normalize_phrase(k) for k in kws if k.strip()]
    return unique_preserve_order(kws)

def evaluate_keyphrase_lists(predictions: List[List[str]], groundtruth: List[List[str]]):
    assert len(predictions) == len(groundtruth)
    total = len(predictions)
    pred_count = 0
    true_count = 0
    match = 0
    for p, g in zip(predictions, groundtruth):
        ps = set([normalize_phrase(x) for x in p])
        gs = set([normalize_phrase(x) for x in g])
        pred_count += len(ps)
        true_count += len(gs)
        match += len(ps & gs)
    precision = match / pred_count if pred_count > 0 else 0.0
    recall = match / true_count if true_count > 0 else 0.0
    f1 = (2*precision*recall / (precision+recall)) if (precision+recall) > 0 else 0.0
    return {"Precision": precision, "Recall": recall, "F1-score": f1}

def evaluate_summaries(preds: List[str], refs: List[str]):
    """
    Compute mean ROUGE1,2,3,L (f-measure) across lists of strings.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rouge3","rougeL"], use_stemmer=True)
    r1, r2, r3, rl = [], [], [], []
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        r3.append(scores["rouge3"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "ROUGE1": float(np.mean(r1)) if r1 else 0.0,
        "ROUGE2": float(np.mean(r2)) if r2 else 0.0,
        "ROUGE3": float(np.mean(r3)) if r3 else 0.0,
        "ROUGEL": float(np.mean(rl)) if rl else 0.0,
    }
