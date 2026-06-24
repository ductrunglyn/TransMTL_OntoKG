# testing_v2.py
import logging
import time
import numpy as np
import torch
from rouge_score import rouge_scorer

from .data import get_loaders, get_loader_for_csv, subword_labels_to_word_labels
# FIX: đổi TransMTL_v2 → TransMTL
from .model import TransformerMTL
from .preprocessing import load_fasttext_bin_embeddings, seed_everything, ids_to_text, convert_tags_to_keyphrases
# FIX: đổi utils_v2 → utils  +  thêm compute_summary_loss_from_logprobs
from .losses import (
    compute_summary_loss_from_logits,
    compute_summary_loss_from_logprobs,   # FIX 4: cần cho copy mechanism
    subword_labels_to_word_labels_fallback,
    load_checkpoint_state,
    ensure_decoder_sos,
)
from .bridge import OntoKGBridge   # OntoKG: truy vấn Neo4j khi test

seed_everything(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test")


# ============================================================
#  FIX 2: Decode ids → text dùng tokenizer.decode() (BPE-aware)
# ============================================================
def _decode_ids_to_text(ids, tokenizer, idx2word, pad_idx, cls_idx, sep_idx):
    """
    Ưu tiên tokenizer.decode() để xử lý đúng byte-level BPE prefix (Ġ).
    Tránh lỗi: "ng" + "ày" → "ng ày" thay vì "ngày".
    Fallback về ids_to_text() khi tokenizer không có sẵn.
    """
    filtered = [
        int(i) for i in ids
        if i != pad_idx
        and (cls_idx is None or i != cls_idx)
        and (sep_idx is None or i != sep_idx)
        and 0 <= int(i) < len(idx2word)
    ]
    if not filtered:
        return ""
    if tokenizer is not None:
        try:
            return tokenizer.decode(filtered).strip()
        except Exception:
            pass
    # fallback
    return ids_to_text(filtered, idx2word, pad_idx, cls_idx, sep_idx)


# ============================================================
#  FIX 4: Chọn loss function phù hợp (CE vs NLL)
# ============================================================
def _compute_summary_loss(out, tgt_sum, pad_idx, ignore_index, label_smoothing, device):
    """
    Dùng NLLLoss khi copy mechanism bật (summary_log_probs có sẵn).
    Dùng CrossEntropyLoss khi chỉ có summary_logits.
    """
    log_probs = out.get("summary_log_probs")
    logits    = out.get("summary_logits")

    if log_probs is not None:
        pred_lp = (log_probs[:, :-1, :] if log_probs.size(1) >= 2 else log_probs).contiguous()
        gold    = tgt_sum[:, 1: 1 + pred_lp.size(1)].to(pred_lp.device)
        return compute_summary_loss_from_logprobs(pred_lp, gold, pad_idx, ignore_index)

    if logits is not None:
        pred_lg = (logits[:, :-1, :] if logits.size(1) >= 2 else logits).contiguous()
        gold    = tgt_sum[:, 1: 1 + pred_lg.size(1)].to(pred_lg.device)
        return compute_summary_loss_from_logits(pred_lg, gold, pad_idx, ignore_index, label_smoothing)

    return torch.tensor(0.0, device=device)


# ============================================================
#  run_test
# ============================================================
def run_test(path_weight, data_path, len_in, len_out, num_workers, batch_size, d_model, pad_idx,
             pretrained_vec_path, num_layers, num_heads, dff, dropout, freeze_embeddings, mmoe_num_experts,
             mmoe_expert_hidden, mmoe_gate_hidden, mmoe_dropout, mmoe_use_residual, mmoe_gate_temperature,
             mmoe_residual_scale, ignore_index, device, use_mmoe, size_vocab,
             # ── OntoKG params (khớp với train_v2.train_model) ──
             use_ontokg=False, entity_emb_path=None, entity_idx_path=None,
             neo4j_uri=None, neo4j_pass=None, ontokg_backend="local",
             tokenizer_csv=None):

    # FIX 2: lấy tokenizer từ dataset object (ds ở vị trí cuối)
    if tokenizer_csv is not None:
        # CHẾ ĐỘ PRE-SPLIT: đánh giá trên TOÀN BỘ data_path (=test.csv), KHÔNG
        # re-split; vocab cố định trên tokenizer_csv (=trainval.csv) để khớp
        # checkpoint đã train.
        print(f"[data] Pre-split test: eval trên toàn bộ {data_path} | "
              f"tokenizer corpus={tokenizer_csv}")
        test_loader, vocab, pad_idx, word2idx, idx2word, ds = get_loader_for_csv(
            data_path, tokenizer_csv, len_in, len_out, num_workers, batch_size,
            vocab_size=size_vocab, min_freq=3, shuffle=False,
        )
    else:
        # CHẾ ĐỘ CŨ (tương thích ngược): re-split data_path 60/20/20, lấy test nội bộ.
        _, _, test_loader, vocab, pad_idx, word2idx, idx2word, _, ds = get_loaders(
            data_path, len_in, len_out, num_workers, batch_size,
            val_ratio=0.2, test_ratio=0.2, seed=42, min_freq=3, vocab_size=size_vocab
        )
    tokenizer = getattr(ds, "tokenizer", None)
    if tokenizer is None:
        logger.warning("Tokenizer not found in dataset — ROUGE sẽ dùng fallback ids_to_text.")

    emb_matrix = None
    if pretrained_vec_path is not None:
        emb_matrix = load_fasttext_bin_embeddings(
            word2idx, pretrained_vec_path, d_model, pad_idx
        )

    # FIX 4 + FIX 6: thêm use_copy=True; bật OntoKG để khớp checkpoint khi train có KG
    model = TransformerMTL(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
        max_len_in=len_in, max_len_out=len_out, dropout=dropout,
        emb_matrix=emb_matrix, word2idx=word2idx, idx2word=idx2word, num_key_labels=5,
        freeze_embeddings=freeze_embeddings,
        mmoe_num_experts=mmoe_num_experts,
        mmoe_expert_hidden=mmoe_expert_hidden,
        mmoe_gate_hidden=mmoe_gate_hidden,
        mmoe_dropout=mmoe_dropout,
        mmoe_use_residual=mmoe_use_residual,
        mmoe_gate_temperature=mmoe_gate_temperature,
        mmoe_residual_scale=mmoe_residual_scale,
        use_mmoe=use_mmoe,
        use_copy=True,    # FIX 4: phải khớp với lúc train
        use_ontokg=use_ontokg, kg_in_dim=768, kg_num_relations=9,
    )
    model = model.to(device)

    state = load_checkpoint_state(path_weight, device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # OntoKG: khởi tạo bridge (backend 'local' đọc file, hoặc 'neo4j')
    bridge = OntoKGBridge(
        uri=neo4j_uri or "bolt://localhost:7687", user="neo4j",
        password=neo4j_pass or "password", d_model=d_model, enabled=use_ontokg,
        backend=ontokg_backend, entity_emb_path=entity_emb_path,
        entity_idx_path=entity_idx_path,
    )

    scorer       = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge1_list  = []; rouge2_list = []; rougel_list = []
    all_pred_kws = []; all_gold_kws = []
    total_sum_loss = 0.0; total_key_loss = 0.0; n_samples = 0

    logger.info("Running evaluation on test set...")
    start = time.time()

    with torch.no_grad():
        for batch in test_loader:
            article_ids = None
            if len(batch) == 8:
                src, tgt_sum, attn, labels, raw_texts, token_maps, word_texts, article_ids = batch
            elif len(batch) == 7:
                src, tgt_sum, attn, labels, raw_texts, token_maps, word_texts = batch
            else:
                src, tgt_sum, attn, labels = batch[:4]
                raw_texts = None; token_maps = None; word_texts = None

            src    = src.to(device); tgt_sum = tgt_sum.to(device); labels = labels.to(device)
            B_cur  = src.size(0)

            # OntoKG: subgraph cho cả batch (None nếu tắt hoặc thiếu article_ids)
            kg_batch = (bridge.build_kg_batch(article_ids)
                        if (bridge is not None and article_ids is not None) else None)

            # Đảm bảo target có SOS để tính loss chuẩn như lúc train
            tgt_sum_sos = ensure_decoder_sos(tgt_sum, model, device)
            out = model(inp=src, tar=tgt_sum_sos, labels=labels, task="both",
                        training=False, kg_batch=kg_batch)
            key_nll = out.get("key_nll", 0.0)

            # FIX 3 + FIX 5: Beam search (auto-regressive, có copy mechanism)
            # Fallback đúng: dùng greedy_decode_batch thay vì teacher-forced argmax
            gen_ids_list = None
            try:
                gen_ids_list = model.beam_search_generate_batch(
                    src, model.max_len_out, beam_size=3, len_penalty=0.6, n_gram_block=3,
                    kg_batch=kg_batch
                )
            except Exception as e:
                logger.debug(f"Beam search failed → fallback greedy: {e}")
                try:
                    # FIX 5: dùng greedy auto-regressive (không phải teacher-forced argmax)
                    gen_ids_list = model.greedy_decode_batch(src, max_len=model.max_len_out,
                                                             kg_batch=kg_batch)
                except Exception as e2:
                    logger.debug(f"Greedy decode also failed: {e2}")
                    gen_ids_list = [[] for _ in range(B_cur)]

            key_decoded = out.get("key_decoded")
            if key_decoded is not None and isinstance(key_decoded, torch.Tensor):
                key_decoded = key_decoded.cpu().numpy()

            tgt_np = tgt_sum_sos.cpu().numpy()

            for i in range(B_cur):
                if word_texts is not None:
                    token_words = word_texts[i]
                else:
                    token_ids   = src.cpu().numpy()[i].tolist()
                    token_words = [
                        idx2word[int(idx)] if 0 <= int(idx) < len(idx2word) else "<unk>"
                        for idx in token_ids if int(idx) != pad_idx
                    ]
                num_words = len(token_words)

                subword_len = len(token_maps[i]) if token_maps is not None else src.size(1)

                # FIX 2: reference text dùng tokenizer.decode()
                ref_ids  = tgt_np[i].tolist()
                ref_text = _decode_ids_to_text(
                    ref_ids, tokenizer, idx2word, pad_idx, model.cls_idx, model.sep_idx
                )

                # FIX 2: predicted text dùng tokenizer.decode()
                gen_seq  = gen_ids_list[i] if i < len(gen_ids_list) else []
                pred_ids = list(gen_seq) if isinstance(gen_seq, (list, tuple, np.ndarray)) else []
                pred_text = _decode_ids_to_text(
                    pred_ids, tokenizer, idx2word, pad_idx, model.cls_idx, model.sep_idx
                )

                # ROUGE
                try:
                    score = scorer.score(ref_text, pred_text)
                    rouge1_list.append(score["rouge1"].fmeasure)
                    rouge2_list.append(score["rouge2"].fmeasure)
                    rougel_list.append(score["rougeL"].fmeasure)
                except Exception as e:
                    logger.debug(f"ROUGE scoring failed sample {i}: {e}")

                # KEY: map subword → word-level
                if key_decoded is not None:
                    sub_pred = [int(x) for x in key_decoded[i][:subword_len]]
                else:
                    sub_pred = []
                label_row = labels.cpu().numpy()[i].tolist()
                sub_gold  = [int(x) for x in label_row[:subword_len]]

                if token_maps is not None and num_words > 0:
                    token_map = token_maps[i]
                    if not token_map:
                        token_map = list(range(subword_len))
                    try:
                        if callable(subword_labels_to_word_labels):
                            word_pred_labels = subword_labels_to_word_labels(
                                sub_pred, token_map, num_words
                            ) if sub_pred else [0] * num_words
                            word_gold_labels = subword_labels_to_word_labels(
                                sub_gold, token_map, num_words
                            )
                        else:
                            raise Exception("not callable")
                    except Exception:
                        word_pred_labels = subword_labels_to_word_labels_fallback(
                            sub_pred, token_map, num_words
                        ) if sub_pred else [0] * num_words
                        word_gold_labels = subword_labels_to_word_labels_fallback(
                            sub_gold, token_map, num_words
                        )
                else:
                    word_pred_labels = sub_pred[:num_words] if sub_pred else []
                    word_gold_labels = sub_gold[:num_words]

                all_pred_kws.append(
                    convert_tags_to_keyphrases(word_pred_labels, token_words) if word_pred_labels else []
                )
                all_gold_kws.append(
                    convert_tags_to_keyphrases(word_gold_labels, token_words) if word_gold_labels else []
                )
                n_samples += 1

            # FIX 4: tính summary loss dùng _compute_summary_loss (hỗ trợ copy)
            batch_sum_loss = float(
                _compute_summary_loss(
                    out, tgt_sum_sos, pad_idx, ignore_index,
                    label_smoothing=0.0, device=device
                ).item()
            )
            batch_key_loss = float(key_nll.item()) if isinstance(key_nll, torch.Tensor) else float(key_nll)
            total_sum_loss += batch_sum_loss * B_cur
            total_key_loss += batch_key_loss * B_cur

    bridge.close()
    elapsed = time.time() - start
    logger.info(f"Evaluation finished in {elapsed:.1f}s on {n_samples} samples.")

    rouge1 = float(np.mean(rouge1_list)) if rouge1_list else 0.0
    rouge2 = float(np.mean(rouge2_list)) if rouge2_list else 0.0
    rougel = float(np.mean(rougel_list)) if rougel_list else 0.0

    from .evaluation import evaluate_keyphrase_lists
    key_metrics  = evaluate_keyphrase_lists(all_pred_kws, all_gold_kws)
    avg_sum_loss = total_sum_loss / n_samples if n_samples > 0 else 0.0
    avg_key_loss = total_key_loss / n_samples if n_samples > 0 else 0.0

    logger.info("=== FINAL TEST RESULTS ===")
    logger.info(f"Mode: {'MMoE' if use_mmoe else 'Direct'} | Copy: ON")
    logger.info(f"Avg Summary Loss: {avg_sum_loss:.4f}")
    logger.info(f"Avg Key Loss:     {avg_key_loss:.4f}")
    logger.info(f"ROUGE-1: {rouge1:.4f}")
    logger.info(f"ROUGE-2: {rouge2:.4f}")
    logger.info(f"ROUGE-L: {rougel:.4f}")
    logger.info(
        f"Keyphrase F1: {key_metrics['F1-score']:.4f} "
        f"(P: {key_metrics['Precision']:.4f}, R: {key_metrics['Recall']:.4f})"
    )

    result_txt_path = path_weight.replace(".pt", "_test_results.txt")
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== FINAL TEST RESULTS ({'MMoE' if use_mmoe else 'Direct'} | Copy: ON) ===\n")
        f.write(f"Avg Summary Loss: {avg_sum_loss:.4f}\n")
        f.write(f"Avg Key Loss:     {avg_key_loss:.4f}\n")
        f.write(f"ROUGE-1: {rouge1:.4f}\n")
        f.write(f"ROUGE-2: {rouge2:.4f}\n")
        f.write(f"ROUGE-L: {rougel:.4f}\n")
        f.write(
            f"Keyphrase F1: {key_metrics['F1-score']:.4f} "
            f"(P: {key_metrics['Precision']:.4f}, R: {key_metrics['Recall']:.4f})\n"
        )

    return {
        "avg_sum_loss": avg_sum_loss,
        "avg_key_loss": avg_key_loss,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougel,
        "key_metrics": key_metrics,
        "n_samples": n_samples,
    }