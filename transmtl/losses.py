# utils.py
import os
import torch
import torch.nn as nn

# ------------------------------------------------------------------ #
#  Loss / Utility Functions                                           #
# ------------------------------------------------------------------ #

def compute_summary_loss_from_logits(logits, targets, pad_idx, ignore_index, label_smoothing):
    """
    CrossEntropy loss cho summary task (raw logits, không có copy mechanism).
    logits:  (B, T_pred, V)
    targets: (B, T_target) — đã align bởi caller (không có SOS ở đầu).
    Trả về: scalar tensor (mean per non-ignored token).
    """
    device = logits.device
    B, T_pred, V = logits.size()

    if targets.size(1) < T_pred:
        pad_more = torch.full(
            (targets.size(0), T_pred - targets.size(1)),
            pad_idx, dtype=targets.dtype, device=device,
        )
        targets = torch.cat([targets.to(device), pad_more], dim=1)
    elif targets.size(1) > T_pred:
        targets = targets[:, :T_pred].to(device)

    tgt = targets.clone().to(device)
    tgt[tgt == pad_idx] = ignore_index

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=ignore_index, reduction="sum", label_smoothing=label_smoothing
    )
    loss = loss_fn(logits.reshape(-1, V), tgt.reshape(-1))
    non_ignored = int((tgt.reshape(-1) != ignore_index).sum().item())
    if non_ignored == 0:
        return loss * 0.0
    return loss / non_ignored


def compute_summary_loss_from_logprobs(log_probs, targets, pad_idx, ignore_index):
    """
    NLLLoss cho summary task khi dùng copy mechanism.
    log_probs: (B, T_pred, V) — đã là log(final_distribution), KHÔNG phải raw logits.
    targets:   (B, T_target) — đã align bởi caller.
    Trả về: scalar tensor (mean per non-ignored token).

    Lưu ý: KHÔNG có label_smoothing vì log_probs đã là phân phối kết hợp;
    áp smoothing lên log_probs sẽ sai về mặt toán học.
    """
    device = log_probs.device
    B, T_pred, V = log_probs.size()

    if targets.size(1) < T_pred:
        pad_more = torch.full(
            (targets.size(0), T_pred - targets.size(1)),
            pad_idx, dtype=targets.dtype, device=device,
        )
        targets = torch.cat([targets.to(device), pad_more], dim=1)
    elif targets.size(1) > T_pred:
        targets = targets[:, :T_pred].to(device)

    tgt = targets.clone().to(device)
    tgt[tgt == pad_idx] = ignore_index

    loss_fn  = nn.NLLLoss(ignore_index=ignore_index, reduction="sum")
    loss     = loss_fn(log_probs.reshape(-1, V), tgt.reshape(-1))
    non_ignored = int((tgt.reshape(-1) != ignore_index).sum().item())
    if non_ignored == 0:
        return loss * 0.0
    return loss / non_ignored


def compute_key_loss_from_raw(key_raw, device):
    """
    Normalize key-task loss. Accepts None, float, hoặc tensor (scalar / vector).
    Trả về tensor >= 0 on device.
    """
    if key_raw is None:
        return torch.tensor(0.0, device=device)

    if isinstance(key_raw, torch.Tensor):
        key_nll_tensor = key_raw.to(device)
        if key_nll_tensor.numel() > 1:
            key_nll_tensor = key_nll_tensor.mean()
    else:
        key_nll_tensor = torch.tensor(float(key_raw), device=device)

    if key_nll_tensor.item() < 0:
        print(f"WARNING: Negative key_nll ({key_nll_tensor.item():.6f}) → converting to positive.")
        key_nll_tensor = (-key_nll_tensor).clamp(min=0.0)

    return key_nll_tensor.clamp(min=0.0, max=1e6)


def compute_entropy_regularizer(gate_probs, entropy_lambda, max_ent_contrib=0.5):
    """
    Entropy regularization từ gate_probs (list of tensors per task).
    Trả về value (scalar) để trừ khỏi total loss.
    Nếu gate_probs is None hoặc entropy_lambda <= 0 → 0.0.
    """
    if gate_probs is None or entropy_lambda <= 0.0:
        return 0.0

    ent = 0.0
    for g in gate_probs:
        p = g.clamp(min=1e-12)
        ent += -(p * p.log()).sum(dim=1).mean()
    ent = ent / max(len(gate_probs), 1)
    ent_contrib = entropy_lambda * ent
    if isinstance(ent_contrib, torch.Tensor):
        ent_contrib = torch.clamp(ent_contrib, max=max_ent_contrib)
    else:
        ent_contrib = min(ent_contrib, max_ent_contrib)
    return ent_contrib


def combine_task_losses(loss_sum, key_nll_tensor, weight_logits, temp=1.0):
    """
    Combine hai losses dùng softmax weights từ weight_logits.
    Trả về (combined_loss, (w_sum, w_key), weights_tensor).
    """
    weights = torch.softmax(weight_logits / float(temp), dim=0)
    w_sum   = weights[0]
    w_key   = weights[1]
    loss    = w_sum * loss_sum + w_key * key_nll_tensor
    return loss, (w_sum, w_key), weights


def ensure_decoder_sos(tgt_sum, model, device):
    """Đảm bảo tgt_sum có SOS (cls_idx) ở đầu. Trả về tensor on device."""
    if not hasattr(model, "cls_idx") or model.cls_idx is None:
        return tgt_sum.to(device)
    tgt_sum = tgt_sum.to(device)
    if tgt_sum.size(1) == 0 or (tgt_sum[:, 0] != model.cls_idx).any():
        B   = tgt_sum.size(0)
        sos = torch.full((B, 1), model.cls_idx, dtype=tgt_sum.dtype, device=device)
        tgt_sum = torch.cat([sos, tgt_sum], dim=1)
        if tgt_sum.size(1) > model.max_len_out:
            tgt_sum = tgt_sum[:, :model.max_len_out]
    return tgt_sum


def subword_labels_to_word_labels_fallback(sub_labels, token_map, n_words):
    """Fallback mapper: ưu tiên label theo priority dict."""
    word_labels = [0] * n_words
    priority    = {4: 5, 1: 4, 3: 3, 2: 2, 0: 1}
    for sw_idx, sw_label in enumerate(sub_labels):
        if sw_idx >= len(token_map):
            break
        widx = token_map[sw_idx]
        if widx is None or widx < 0 or widx >= n_words:
            continue
        cur = word_labels[widx]
        if priority.get(int(sw_label), 0) > priority.get(int(cur), 0):
            word_labels[widx] = int(sw_label)
    return word_labels


def load_checkpoint_state(path_weight: str, device: torch.device) -> dict:
    """Load checkpoint và strip 'module.' prefix nếu có (DataParallel)."""
    if not os.path.exists(path_weight):
        raise FileNotFoundError(f"Checkpoint not found: {path_weight}")
    state     = torch.load(path_weight, map_location=device)
    new_state = {}
    for k, v in state.items():
        new_state[k[7:] if k.startswith("module.") else k] = v
    return new_state