# train_v2.py  (đã tích hợp OntoKG + bỏ synonym)
import math
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from rouge_score import rouge_scorer
from .data import get_loaders
from .model import TransformerMTL
from .preprocessing import load_fasttext_bin_embeddings, seed_everything, convert_tags_to_keyphrases
from .evaluation import evaluate_keyphrase_lists
from .bridge import OntoKGBridge   # NEW: cầu nối OntoKG/Neo4j
from .losses import (
    compute_summary_loss_from_logits,
    compute_summary_loss_from_logprobs,
    compute_key_loss_from_raw,
    compute_entropy_regularizer,
    combine_task_losses,
    ensure_decoder_sos,
    subword_labels_to_word_labels_fallback,
    load_checkpoint_state,
)
logger = logging.getLogger("train_v2")
logger.setLevel(logging.INFO)
# ============================================================
#  Cấu hình weight floor
# ============================================================
MIN_SUM_WEIGHT  = 0.25
SUM_LOGIT_FLOOR = math.log(MIN_SUM_WEIGHT / (1.0 - MIN_SUM_WEIGHT))   # ≈ -1.099
def _param_count(params):
    return sum(p.numel() for p in params if p is not None)
# ============================================================
#  Decode ids → text dùng tokenizer.decode() (BPE-aware)
# ============================================================
def _decode_ids_to_text(ids, tokenizer, idx2word, pad_idx, cls_idx, sep_idx):
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
    return " ".join(idx2word[i] for i in filtered).strip()
# ============================================================
#  Chọn loss function phù hợp với output của model
# ============================================================
def _compute_summary_loss(out, tgt_sum, pad_idx, ignore_index, label_smoothing, device):
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
#  Unpack batch (hỗ trợ 8-tuple có article_ids)
# ============================================================
def _unpack_batch(batch):
    if len(batch) == 8:
        src, tgt_sum, attn, labels, raw_texts, token_maps, word_texts, article_ids = batch
    elif len(batch) == 7:
        src, tgt_sum, attn, labels, raw_texts, token_maps, word_texts = batch
        article_ids = None
    else:
        src, tgt_sum, attn, labels = batch[:4]
        raw_texts = token_maps = word_texts = article_ids = None
    return src, tgt_sum, attn, labels, raw_texts, token_maps, word_texts, article_ids
# ============================================================
#  train_one_epoch
# ============================================================
def train_one_epoch(
    model, train_loader, optimizer_model, optimizer_w, device,
    pad_idx, ignore_index, label_smoothing, entropy_lambda, clip_norm,
    weight_update_per_batch=True,
    update_weights_enabled=True,
    loss_norm_alpha=0.99,
    avg_sum=1.0,
    avg_key=1.0,
    weight_softmax_temp=1.0,
    use_pcgrad=True,
    bridge=None,                 # NEW: OntoKGBridge (None = baseline)
):
    model.train()
    sum_batch_loss = 0.0
    n_examples     = 0
    params = [p for p in model.parameters() if p.requires_grad and p is not model.weight_logits]
    eps    = 1e-8
    for batch in train_loader:
        src, tgt_sum, attn, labels, raw_texts, token_maps, word_texts, article_ids = _unpack_batch(batch)
        src     = src.to(device)
        tgt_sum = tgt_sum.to(device)
        labels  = labels.to(device)
        tgt_sum = ensure_decoder_sos(tgt_sum, model, device)

        # NEW: tạo kg_batch từ OntoKG (None nếu tắt hoặc không có article_ids)
        kg_batch = bridge.build_kg_batch(article_ids) if (bridge is not None and article_ids is not None) else None

        out     = model(inp=src, tar=tgt_sum, labels=labels, task="both", training=True, kg_batch=kg_batch)
        # tự động chọn CE hoặc NLL tuỳ copy mechanism
        L_sum       = _compute_summary_loss(out, tgt_sum, pad_idx, ignore_index, label_smoothing, device)
        L_key       = compute_key_loss_from_raw(out.get("key_nll", None), device)
        ent_contrib = compute_entropy_regularizer(out.get("mmoe_gate_probs"), entropy_lambda)
        s_val   = float(L_sum.detach().cpu().item()) if isinstance(L_sum, torch.Tensor) else float(L_sum)
        k_val   = float(L_key.detach().cpu().item()) if isinstance(L_key, torch.Tensor) else float(L_key)
        avg_sum = loss_norm_alpha * avg_sum + (1.0 - loss_norm_alpha) * s_val
        avg_key = loss_norm_alpha * avg_key + (1.0 - loss_norm_alpha) * k_val
        norm_L_sum = L_sum / (avg_sum + eps)
        norm_L_key = L_key / (avg_key + eps)
        if use_pcgrad:
            optimizer_model.zero_grad()
            if L_sum.requires_grad:
                L_sum.backward(retain_graph=True)
            grads_sum = [p.grad.clone() if p.grad is not None else None for p in params]
            optimizer_model.zero_grad()
            if L_key.requires_grad:
                L_key.backward(retain_graph=True)
            grads_key = [p.grad.clone() if p.grad is not None else None for p in params]
            optimizer_model.zero_grad()
            task_grads = [grads_sum, grads_key]
            T          = len(task_grads)
            for i in range(T):
                gi = task_grads[i]
                for j in range(T):
                    if i == j: continue
                    gj = task_grads[j]
                    for k, (gik, gjk) in enumerate(zip(gi, gj)):
                        if gik is None or gjk is None: continue
                        dot = (gik * gjk).sum()
                        if dot < 0:
                            gj_norm2 = (gjk * gjk).sum()
                            if gj_norm2.item() > 0:
                                gi[k] = gik - (dot / (gj_norm2 + 1e-12)) * gjk
            combined = []
            for k in range(len(params)):
                g_total = None
                for t in range(T):
                    gt = task_grads[t][k]
                    if gt is None: continue
                    g_total = gt.clone() if g_total is None else g_total + gt
                combined.append(None if g_total is None else g_total / float(T))
            for p, g in zip(params, combined):
                p.grad = None if g is None else g.to(p.device)
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm)
            optimizer_model.step()
            if update_weights_enabled and weight_update_per_batch:
                optimizer_w.zero_grad()
                w_probs = torch.softmax(model.weight_logits / float(weight_softmax_temp), dim=0)
                (w_probs[0] * norm_L_sum.detach() + w_probs[1] * norm_L_key.detach()).backward()
                optimizer_w.step()
                with torch.no_grad():
                    if (model.weight_logits[0] - model.weight_logits[1]).item() < SUM_LOGIT_FLOOR:
                        model.weight_logits[0] = model.weight_logits[1] + SUM_LOGIT_FLOOR
                w_after = torch.softmax(model.weight_logits.detach() / float(weight_softmax_temp), dim=0)
                recorded_loss = float(w_after[0].item() * s_val + w_after[1].item() * k_val)
            else:
                w_curr = torch.softmax(model.weight_logits.detach() / float(weight_softmax_temp), dim=0)
                recorded_loss = float(w_curr[0].item() * s_val + w_curr[1].item() * k_val)
        else:
            optimizer_model.zero_grad(); optimizer_w.zero_grad()
            loss_for_model, _, _ = combine_task_losses(L_sum, L_key, model.weight_logits)
            if ent_contrib != 0.0:
                loss_for_model = loss_for_model - ent_contrib
            loss_for_model.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer_model.step()
            if update_weights_enabled and weight_update_per_batch:
                optimizer_w.zero_grad()
                w_probs = torch.softmax(model.weight_logits / float(weight_softmax_temp), dim=0)
                (w_probs[0] * norm_L_sum.detach() + w_probs[1] * norm_L_key.detach()).backward()
                optimizer_w.step()
                with torch.no_grad():
                    if (model.weight_logits[0] - model.weight_logits[1]).item() < SUM_LOGIT_FLOOR:
                        model.weight_logits[0] = model.weight_logits[1] + SUM_LOGIT_FLOOR
                w_after = torch.softmax(model.weight_logits.detach() / float(weight_softmax_temp), dim=0)
                recorded_loss = float(w_after[0].item() * s_val + w_after[1].item() * k_val)
            else:
                w_curr = torch.softmax(model.weight_logits.detach() / float(weight_softmax_temp), dim=0)
                recorded_loss = float(w_curr[0].item() * s_val + w_curr[1].item() * k_val)
        sum_batch_loss += recorded_loss * src.size(0)
        n_examples     += src.size(0)
    return sum_batch_loss / n_examples if n_examples > 0 else 0.0, avg_sum, avg_key
# ============================================================
#  validate
# ============================================================
def validate(
    model, val_loader, device, pad_idx, ignore_index, label_smoothing,
    idx2word, tokenizer=None, use_fallback_mapper=False, bridge=None,   # NEW: bridge
):
    model.eval()
    val_sum_loss  = 0.0
    val_key_loss  = 0.0
    val_n         = 0
    rouge1_scores = []
    all_pred_kws  = []
    all_gold_kws  = []
    scorer        = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    try:
        from .data import subword_labels_to_word_labels
    except Exception:
        subword_labels_to_word_labels = None
    use_fallback = (subword_labels_to_word_labels is None) or use_fallback_mapper
    with torch.no_grad():
        for batch in val_loader:
            src, tgt_sum, attn, labels, raw_texts, token_maps, word_texts, article_ids = _unpack_batch(batch)
            src     = src.to(device)
            tgt_sum = tgt_sum.to(device)
            labels  = labels.to(device)
            tgt_sum_sos = ensure_decoder_sos(tgt_sum, model, device)

            # NEW: kg_batch cho validation
            kg_batch = bridge.build_kg_batch(article_ids) if (bridge is not None and article_ids is not None) else None

            # Forward teacher-forced → chỉ để tính LOSS
            out = model(inp=src, tar=tgt_sum_sos, labels=labels, task="both", training=False, kg_batch=kg_batch)
            loss_s = _compute_summary_loss(out, tgt_sum_sos, pad_idx, ignore_index, label_smoothing, device)
            key_nll = out.get("key_nll", torch.tensor(0.0, device=device))
            loss_k  = key_nll if isinstance(key_nll, torch.Tensor) else torch.tensor(float(key_nll), device=device)
            # Auto-regressive greedy generation để tính ROUGE (truyền kg_batch)
            gen_ids_list = model.greedy_decode_batch(src, max_len=model.max_len_out, kg_batch=kg_batch)
            key_decoded = out.get("key_decoded", None)
            key_dec_np  = key_decoded.cpu().numpy() if key_decoded is not None else None
            B      = src.size(0)
            tgt_np = tgt_sum_sos.cpu().numpy()
            for i in range(B):
                token_words = word_texts[i] if word_texts is not None else [
                    idx2word[int(idx)] if 0 <= int(idx) < len(idx2word) else "<unk>"
                    for idx in src.cpu().numpy()[i].tolist() if int(idx) != pad_idx
                ]
                num_words = len(token_words)
                ref_text = _decode_ids_to_text(
                    tgt_np[i].tolist(), tokenizer, idx2word, pad_idx, model.cls_idx, model.sep_idx
                )
                pred_text = _decode_ids_to_text(
                    gen_ids_list[i], tokenizer, idx2word, pad_idx, model.cls_idx, model.sep_idx
                )
                try:
                    score = scorer.score(ref_text, pred_text)
                    rouge1_scores.append(score["rouge1"].fmeasure)
                except Exception:
                    pass
                token_map = token_maps[i] if token_maps is not None else None
                label_row = labels.cpu().numpy()[i].tolist()
                sub_len   = min(len(token_map), len(label_row)) if token_map is not None else len(label_row)
                sub_pred  = [int(x) for x in key_dec_np[i][:sub_len]] if key_dec_np is not None else []
                sub_gold  = [int(x if x != ignore_index else 0) for x in label_row[:sub_len]]
                if token_map is not None and num_words > 0:
                    mapper = (
                        subword_labels_to_word_labels_fallback if use_fallback
                        else subword_labels_to_word_labels
                    )
                    word_pred_labels = mapper(sub_pred, token_map, num_words) if sub_pred else [0] * num_words
                    word_gold_labels = mapper(sub_gold, token_map, num_words)
                else:
                    word_pred_labels = sub_pred[:num_words] if sub_pred else []
                    word_gold_labels = sub_gold[:num_words]
                all_pred_kws.append(
                    convert_tags_to_keyphrases(word_pred_labels, token_words) if word_pred_labels else []
                )
                all_gold_kws.append(
                    convert_tags_to_keyphrases(word_gold_labels, token_words) if word_gold_labels else []
                )
                val_n += 1
            val_sum_loss += float(loss_s.item()) * B
            val_key_loss += float(loss_k.item()) * B
    val_sum_loss /= val_n if val_n > 0 else 1
    val_key_loss /= val_n if val_n > 0 else 1
    r1  = float(np.mean(rouge1_scores)) if rouge1_scores else 0.0
    key_res = evaluate_keyphrase_lists(all_pred_kws, all_gold_kws)
    kf1 = key_res["F1-score"]
    val_score = 2.0 * r1 + kf1
    return {
        "val_sum_loss": val_sum_loss,
        "val_key_loss": val_key_loss,
        "rouge1": r1,
        "key_f1": kf1,
        "val_score": val_score,
        "all_pred_kws": all_pred_kws,
        "all_gold_kws": all_gold_kws,
    }
# ============================================================
#  train_model
# ============================================================
def train_model(
    data_path, save_score_path, pad_idx, label_smoothing, pretrained_vec_path,
    num_layers, d_model, num_heads, dff, len_in, len_out, dropout, freeze_embeddings,
    mmoe_num_experts, mmoe_expert_hidden, mmoe_gate_hidden, mmoe_dropout,
    mmoe_use_residual, mmoe_gate_temperature, mmoe_residual_scale, lr, weight_decay,
    num_epochs, warmup_mmoe, entropy_lambda, clip_norm, ignore_index,
    num_workers, batch_size, use_mmoe, device, size_vocab,
    # ── OntoKG params (đã bỏ use_synonym, synonym_path) ──
    use_ontokg=False, entity_emb_path=None, entity_idx_path=None,
    neo4j_uri=None, neo4j_pass=None, ontokg_backend="local",
):
    summary_lr_mult     = 2.0
    crf_lr_mult         = 1.0
    weight_lr_mult      = 5.0
    weight_update_per_batch = True
    loss_norm_alpha     = 0.99
    weight_softmax_temp = 1.0
    patience            = 5
    seed_everything(42)
    print(f"Using device: {device}")
    train_loader, val_loader, _, vocab, pad_idx, word2idx, idx2word, _, ds = get_loaders(
        data_path, len_in, len_out, num_workers, batch_size,
        val_ratio=0.2, test_ratio=0.2, seed=42, min_freq=3, vocab_size=size_vocab,
    )
    tokenizer = getattr(ds, "tokenizer", None)
    if tokenizer is None:
        logger.warning("Dataset không có .tokenizer — ROUGE sẽ dùng fallback ids_to_text.")
    # đã bỏ use_synonym/synonym_path
    emb_matrix = load_fasttext_bin_embeddings(word2idx, pretrained_vec_path, d_model, pad_idx)
    print("Initializing TransformerMTL...")
    model = TransformerMTL(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
        max_len_in=len_in, max_len_out=len_out, dropout=dropout,
        emb_matrix=emb_matrix, word2idx=word2idx, idx2word=idx2word,
        num_key_labels=5, freeze_embeddings=freeze_embeddings,
        mmoe_num_experts=mmoe_num_experts, mmoe_expert_hidden=mmoe_expert_hidden,
        mmoe_gate_hidden=mmoe_gate_hidden, pad_idx=pad_idx,
        mmoe_dropout=mmoe_dropout, mmoe_use_residual=mmoe_use_residual,
        mmoe_gate_temperature=mmoe_gate_temperature,
        mmoe_residual_scale=mmoe_residual_scale, use_mmoe=use_mmoe,
        use_copy=True,
        use_ontokg=use_ontokg, kg_in_dim=768, kg_num_relations=9,   # NEW
    )
    model = model.to(device)

    # NEW: khởi tạo OntoKGBridge (truy vấn Neo4j lúc train/validate)
    bridge = OntoKGBridge(
        uri=neo4j_uri or "bolt://localhost:7687", user="neo4j",
        password=neo4j_pass or "password", d_model=d_model, enabled=use_ontokg,
        backend=ontokg_backend, entity_emb_path=entity_emb_path,
        entity_idx_path=entity_idx_path,
    )

    weight_logits = nn.Parameter(torch.zeros(2, device=device))
    model.register_parameter("weight_logits", weight_logits)
    with torch.no_grad():
        model.weight_logits.data = torch.tensor([0.2, 0.0], device=device)
    crf_params       = list(model.crf_decoder.parameters()) + list(model.final_key_proj.parameters())
    crf_param_ids    = {id(p) for p in crf_params}
    weight_logits_id = id(weight_logits)
    summary_keywords = ["summary", "sum", "decoder", "dec", "final", "out_proj", "generator", "decode"]
    key_keywords     = ["key", "crf", "final_key", "key_proj"]
    copy_keywords    = ["copy_gate"]
    summary_params = []
    other_params   = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        pid   = id(p)
        lname = name.lower()
        if pid in crf_param_ids or pid == weight_logits_id: continue
        if any(k in lname for k in key_keywords): continue
        if any(k in lname for k in summary_keywords) or any(k in lname for k in copy_keywords):
            summary_params.append(p)
        else:
            other_params.append(p)
    if not summary_params:
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if "decoder" in name.lower() and id(p) not in crf_param_ids and id(p) != weight_logits_id:
                summary_params.append(p)
    print("Param group sizes: summary=%d, shared=%d, crf=%d" % (
        _param_count(summary_params), _param_count(other_params), _param_count(crf_params)
    ))
    optimizer_model = AdamW(
        [
            {"params": summary_params, "lr": lr * summary_lr_mult},
            {"params": other_params,   "lr": lr},
            {"params": crf_params,     "lr": lr * crf_lr_mult},
        ],
        weight_decay=weight_decay,
    )
    optimizer_w = AdamW(
        [{"params": [weight_logits], "lr": lr * weight_lr_mult, "weight_decay": 0.0}],
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_model, mode="min", factor=0.5, patience=3
    )
    avg_sum            = 1.0
    avg_key            = 1.0
    best_val_score     = -1.0
    no_improve_counter = 0
    use_fallback_mapper = False
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        if getattr(model, "mmoe", None) is not None:
            frozen = epoch <= warmup_mmoe
            for p in model.mmoe.parameters():
                p.requires_grad = not frozen
        train_loss, avg_sum, avg_key = train_one_epoch(
            model, train_loader, optimizer_model, optimizer_w, device,
            pad_idx, ignore_index, label_smoothing, entropy_lambda, clip_norm,
            weight_update_per_batch=weight_update_per_batch,
            update_weights_enabled=True,
            loss_norm_alpha=loss_norm_alpha,
            avg_sum=avg_sum,
            avg_key=avg_key,
            weight_softmax_temp=weight_softmax_temp,
            use_pcgrad=True,
            bridge=bridge,                  # NEW
        )
        val_res = validate(
            model, val_loader, device, pad_idx, ignore_index, label_smoothing,
            idx2word, tokenizer=tokenizer, use_fallback_mapper=use_fallback_mapper,
            bridge=bridge,                  # NEW
        )
        scheduler.step(val_res["val_sum_loss"])
        w_print = torch.softmax(
            model.weight_logits.detach() / float(weight_softmax_temp), dim=0
        ).cpu().numpy()
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val SumLoss {val_res['val_sum_loss']:.4f} | "
            f"Val KeyLoss {val_res['val_key_loss']:.4f}"
        )
        print(
            f"Metrics: ROUGE1 {val_res['rouge1']:.4f} | "
            f"Key F1 {val_res['key_f1']:.4f} | "
            f"Val Score {val_res['val_score']:.4f} | "
            f"time {(time.time() - t0):.1f}s"
        )
        print(f"Auto-balancing weights: Sum ~ {w_print[0]:.4f}, Key ~ {w_print[1]:.4f}")
        print(f"Running avg norm: avg_sum={avg_sum:.6f}, avg_key={avg_key:.6f}")
        current_val_score = val_res["val_score"]
        if current_val_score > best_val_score:
            best_val_score     = current_val_score
            no_improve_counter = 0
            torch.save(model.state_dict(), save_score_path)
            print(f"✅ Saved BEST SCORE model: {current_val_score:.4f} → {save_score_path}")
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                with torch.no_grad():
                    w = torch.softmax(model.weight_logits, dim=0)
                    if w[0].item() < MIN_SUM_WEIGHT:
                        model.weight_logits.data[0] = (
                            model.weight_logits.data[1] + SUM_LOGIT_FLOOR
                        )
                        print(
                            f"[INFO] No improvement + sum weight thấp "
                            f"({w[0].item():.3f}) → nudge logits về sum."
                        )
                    else:
                        print(f"[INFO] No improvement in {patience} epochs.")
                no_improve_counter = 0
    bridge.close()           # NEW: đóng kết nối Neo4j
    return best_val_score