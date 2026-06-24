# TransMTL.py  (đã tích hợp OntoKG + bỏ synonym)
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

# OntoKG components (additive — kg_batch=None thì chạy như baseline)
from .fusion import GraphEncoder, GatedFusion, encode_kg_batch

# ----------------- Utilities -----------------
def create_masks(inp, tar, pad_idx=0, device=None):
    if device is None:
        device = inp.device
    enc_padding_mask = (inp == pad_idx).to(device).unsqueeze(1).unsqueeze(2)
    dec_padding_mask = enc_padding_mask.clone()
    seq_len_out = tar.size(1)
    look_ahead = torch.triu(torch.ones((seq_len_out, seq_len_out), device=device), diagonal=1).bool()
    look_ahead = look_ahead.unsqueeze(0).unsqueeze(1)
    tar_padding_mask = (tar == pad_idx).to(device).unsqueeze(1).unsqueeze(2)
    look_ahead_mask = look_ahead | (tar_padding_mask.expand(-1, -1, seq_len_out, seq_len_out))
    return enc_padding_mask, look_ahead_mask, dec_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ----------------- Transformer Blocks -----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.split_heads(self.Wq(q))
        K = self.split_heads(self.Wk(k))
        V = self.split_heads(self.Wv(v))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out(out), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, training=True, mask=None):
        attn_out, _ = self.mha(x, x, x, mask)
        attn_out = self.drop1(attn_out) if training else attn_out
        out1 = self.ln1(x + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.drop2(ffn_out) if training else ffn_out
        return self.ln2(out1 + ffn_out)


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, max_pos, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, training=True, mask=None):
        out = x
        for layer in self.layers:
            out = layer(out, training, mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, training=True, look_ahead_mask=None, padding_mask=None):
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.drop1(attn1) if training else attn1
        out1 = self.ln1(x + attn1)

        attn2, cross_attn_w = self.mha2(out1, enc_out, enc_out, padding_mask)
        attn2 = self.drop2(attn2) if training else attn2
        out2 = self.ln2(out1 + attn2)

        ffn_out = self.ffn(out2)
        ffn_out = self.drop3(ffn_out) if training else ffn_out
        out3 = self.ln3(out2 + ffn_out)

        cross_attn_avg = cross_attn_w.mean(dim=1)
        return out3, cross_attn_avg


class Decoder_Sum(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, max_len_out, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, training=True, look_ahead_mask=None, padding_mask=None):
        out = x
        last_cross_attn = None
        for layer in self.layers:
            out, cross_attn = layer(out, enc_out, training, look_ahead_mask, padding_mask)
            last_cross_attn = cross_attn
        out = self.ln_final(out)
        return out, last_cross_attn


# ----------------- Copy Gate -----------------
class CopyGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model * 3, 1, bias=True)

    def forward(self, dec_hidden, context, input_emb):
        gate_input = torch.cat([dec_hidden, context, input_emb], dim=-1)
        return torch.sigmoid(self.linear(gate_input))


# ----------------- MMoE -----------------
class MMoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, expert_hidden=None, num_tasks=2,
                 gate_hidden=None, dropout=0.0, use_residual=True,
                 gate_temperature=1.0, residual_scale=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.gate_temperature = gate_temperature
        self.use_residual = use_residual
        self.residual_scale = residual_scale

        if expert_hidden is None: expert_hidden = input_dim * 4
        if gate_hidden is None:   gate_hidden   = max(input_dim // 4, 32)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(expert_hidden, input_dim),
            ) for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, gate_hidden), nn.ReLU(),
                nn.Linear(gate_hidden, num_experts),
            ) for _ in range(num_tasks)
        ])
        self.output_ln = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_tasks)])

    def forward(self, x, mask=None, return_gates=False):
        if mask is not None:
            maskf = mask.float().unsqueeze(-1)
            pooled = (x * maskf).sum(dim=1) / maskf.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)

        expert_stack = torch.cat([e(x).unsqueeze(2) for e in self.experts], dim=2)

        task_outputs = []
        gate_probs_list = []
        for t in range(self.num_tasks):
            gate_logits = self.gates[t](pooled) / self.gate_temperature
            gate_probs = F.softmax(gate_logits, dim=-1)
            gate_probs_list.append(gate_probs)
            gw = gate_probs.unsqueeze(1).unsqueeze(-1)
            weighted = (expert_stack * gw).sum(dim=2)
            if self.use_residual:
                weighted = weighted + self.residual_scale * x
            task_outputs.append(self.output_ln[t](weighted))

        if return_gates:
            return task_outputs, gate_probs_list
        return task_outputs


# ----------------- Main Model -----------------
class TransformerMTL(nn.Module):
    def __init__(
        self,
        num_layers=4, d_model=512, num_heads=8, dff=2048,
        max_len_in=1024, max_len_out=128, dropout=0.1,
        emb_matrix=None, word2idx=None, idx2word=None,
        freeze_embeddings=False, num_key_labels=5,
        mmoe_num_experts=4, mmoe_expert_hidden=None,
        mmoe_gate_hidden=None, pad_idx=0,
        mmoe_dropout=0.0, mmoe_use_residual=True,
        mmoe_gate_temperature=1.0, mmoe_residual_scale=0.1,
        use_mmoe=True,
        use_copy=True,
        use_ontokg=False,        # OntoKG: bật tích hợp đồ thị tri thức
        kg_in_dim=768,           # chiều embedding entity (từ Module 7)
        kg_num_relations=9,      # số loại quan hệ ngữ nghĩa
    ):
        super().__init__()
        if word2idx is None or idx2word is None:
            raise ValueError("TransformerMTL yêu cầu word2idx và idx2word.")

        self.word2idx   = word2idx
        self.idx2word   = idx2word
        self.vocab_size = len(idx2word)
        self.pad_idx    = int(self.word2idx.get("<pad>", pad_idx))
        self.cls_idx    = self.word2idx.get("<sos>", None)
        self.sep_idx    = self.word2idx.get("<eos>", None)
        self.max_len_out = max_len_out
        self.use_mmoe   = use_mmoe
        self.use_copy   = use_copy
        self.d_model    = d_model

        # 1. Embeddings
        self.embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_idx)
        if emb_matrix is not None:
            if isinstance(emb_matrix, np.ndarray):
                emb_tensor = torch.from_numpy(emb_matrix).float()
            else:
                emb_tensor = emb_matrix
            if emb_tensor.size(1) != d_model:
                raise ValueError(f"emb_matrix dim ({emb_tensor.size(1)}) != d_model ({d_model})")
            self.embedding.weight.data.copy_(emb_tensor)
            print(f"Model: Loaded embedding matrix with shape {emb_tensor.shape}")
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            print("Model: Embeddings are frozen.")

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=max_len_in)

        # 2. Shared Encoder
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, max_len_in, dropout)

        # 2b. OntoKG components (additive)
        self.use_ontokg = use_ontokg
        if use_ontokg:
            self.graph_encoder = GraphEncoder(
                in_dim=kg_in_dim, d_model=d_model,
                num_relations=kg_num_relations, dropout=dropout,
            )
            self.gated_fusion = GatedFusion(d_model, num_heads=num_heads, dropout=dropout)
            print("Model: OntoKG fusion ENABLED.")
        else:
            self.graph_encoder = None
            self.gated_fusion  = None

        # 3. MMoE
        if self.use_mmoe:
            self.mmoe = MMoE(
                input_dim=d_model, num_experts=mmoe_num_experts,
                expert_hidden=mmoe_expert_hidden, num_tasks=2,
                gate_hidden=mmoe_gate_hidden, dropout=mmoe_dropout,
                use_residual=mmoe_use_residual,
                gate_temperature=mmoe_gate_temperature,
                residual_scale=mmoe_residual_scale,
            )
        else:
            self.mmoe = None
            print("Model: MMoE DISABLED — direct encoder-decoder.")

        # 4. Summary decoder
        self.decoder_sum    = Decoder_Sum(num_layers, d_model, num_heads, dff, max_len_out, dropout)
        self.final_sum_proj = nn.Linear(d_model, self.vocab_size)

        # Copy Gate
        if self.use_copy:
            self.copy_gate = CopyGate(d_model)
            print("Model: Copy mechanism ENABLED.")
        else:
            self.copy_gate = None

        # 5. Keyphrase head (CRF)
        self.final_key_proj = nn.Linear(d_model, num_key_labels)
        self.crf_decoder    = CRF(num_key_labels, batch_first=True)

        # Xavier init
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                if id(p) == id(self.embedding.weight) and emb_matrix is not None:
                    continue
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------ #
    #  OntoKG fusion helper: H_tok + gated(cross_attn(H_tok, E_kg))      #
    # ------------------------------------------------------------------ #
    def _apply_ontokg_fusion(self, enc_out_shared, kg_batch, device):
        if self.use_ontokg and self.gated_fusion is not None and kg_batch is not None:
            E_kg, kg_mask = encode_kg_batch(
                self.graph_encoder, kg_batch, self.d_model, device
            )
            return self.gated_fusion(enc_out_shared, E_kg, kg_mask)
        return enc_out_shared

    # ------------------------------------------------------------------ #
    #  Helper nội bộ: tính copy-augmented log-probs                      #
    # ------------------------------------------------------------------ #
    def _apply_copy(self, raw_logits, dec_out, cross_attn, tar_emb_pe, enc_out_sum, src):
        device = raw_logits.device
        B, T_dec, _ = raw_logits.size()

        context = torch.bmm(cross_attn, enc_out_sum)

        tar_trunc = tar_emb_pe[:, :T_dec, :]
        p_gen     = self.copy_gate(dec_out, context, tar_trunc)

        vocab_probs = torch.softmax(raw_logits, dim=-1)

        copy_dist   = torch.zeros(B, T_dec, self.vocab_size, device=device)
        src_safe    = src.clamp(0, self.vocab_size - 1)
        src_expand  = src_safe.unsqueeze(1).expand(-1, T_dec, -1)
        copy_dist.scatter_add_(2, src_expand, cross_attn)

        final_probs = p_gen * vocab_probs + (1.0 - p_gen) * copy_dist
        return torch.log(final_probs.clamp(min=1e-9))

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, inp, tar=None, labels=None, task="both", training=True, kg_batch=None):
        device           = inp.device
        enc_padding_mask = (inp == self.pad_idx).to(device).unsqueeze(1).unsqueeze(2)

        # 1. Encode
        x = self.embedding(inp) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        enc_out_shared = self.encoder(x, training=training, mask=enc_padding_mask)

        # 1b. OntoKG fusion (chỉ khi bật + có kg_batch)
        enc_out_shared = self._apply_ontokg_fusion(enc_out_shared, kg_batch, device)

        # 2. MMoE routing
        gate_probs = None
        if self.use_mmoe:
            mask_tokens            = (inp != self.pad_idx)
            mmoe_outs, gate_probs  = self.mmoe(enc_out_shared, mask=mask_tokens, return_gates=True)
            enc_out_sum = mmoe_outs[0]
            enc_out_key = mmoe_outs[1]
        else:
            enc_out_sum = enc_out_shared
            enc_out_key = enc_out_shared

        out = {"mmoe_gate_probs": gate_probs}

        # 3. Summary task
        if task in ["sum", "both"]:
            if tar is None:
                out["enc_out_sum"] = enc_out_sum
            else:
                _, look_ahead_mask, _ = create_masks(inp, tar, self.pad_idx, device)
                tar_emb    = self.embedding(tar) * math.sqrt(self.d_model)
                tar_emb_pe = self.pos_encoding(tar_emb)

                dec_out, cross_attn = self.decoder_sum(
                    tar_emb_pe, enc_out_sum,
                    training=training,
                    look_ahead_mask=look_ahead_mask,
                    padding_mask=enc_padding_mask,
                )
                raw_logits = self.final_sum_proj(dec_out)
                out["summary_logits"] = raw_logits

                if self.use_copy and cross_attn is not None:
                    out["summary_log_probs"] = self._apply_copy(
                        raw_logits, dec_out, cross_attn, tar_emb_pe, enc_out_sum, inp
                    )

        # 4. Keyword task (CRF)
        if task in ["key", "both"]:
            key_emissions          = self.final_key_proj(enc_out_key)
            out["key_emissions"]   = key_emissions
            mask_crf               = (inp != self.pad_idx).bool().to(device)
            mask_crf[:, 0]         = True

            if labels is not None:
                if labels.size(1) > key_emissions.size(1):
                    labels = labels[:, :key_emissions.size(1)]
                labels_safe            = labels.clone()
                labels_safe[labels_safe < 0] = 0
                log_likelihood         = self.crf_decoder(
                    key_emissions, labels_safe, mask=mask_crf, reduction="token_mean"
                )
                out["key_nll"] = -log_likelihood

            if not training:
                decoded_lists = self.crf_decoder.decode(key_emissions, mask=mask_crf)
                max_l         = key_emissions.size(1)
                padded        = np.zeros((len(decoded_lists), max_l), dtype=int)
                for i, seq in enumerate(decoded_lists):
                    if len(seq) > 0:
                        padded[i, :len(seq)] = seq
                out["key_decoded"] = torch.tensor(padded, device=device)

        return out

    # ------------------------------------------------------------------ #
    #  Greedy auto-regressive decode                                      #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def greedy_decode_batch(self, inp, max_len=None, kg_batch=None):
        if max_len is None:
            max_len = self.max_len_out
        self.eval()
        device = inp.device
        B      = inp.size(0)

        enc_padding_mask = (inp == self.pad_idx).to(device).unsqueeze(1).unsqueeze(2)
        x = self.embedding(inp) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        enc_out_shared = self.encoder(x, training=False, mask=enc_padding_mask)

        # OntoKG fusion
        enc_out_shared = self._apply_ontokg_fusion(enc_out_shared, kg_batch, device)

        if self.use_mmoe:
            mask_tokens = (inp != self.pad_idx)
            mmoe_outs   = self.mmoe(enc_out_shared, mask=mask_tokens, return_gates=False)
            enc_out_sum = mmoe_outs[0]
        else:
            enc_out_sum = enc_out_shared

        cls_token = self.cls_idx if self.cls_idx is not None else 1
        dec_in    = torch.full((B, 1), cls_token, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            T          = dec_in.size(1)
            look_ahead = (
                torch.triu(torch.ones(T, T, device=device), diagonal=1)
                .bool().unsqueeze(0).unsqueeze(1)
            )
            tar_emb    = self.embedding(dec_in) * math.sqrt(self.d_model)
            tar_emb_pe = self.pos_encoding(tar_emb)

            dec_out, cross_attn = self.decoder_sum(
                tar_emb_pe, enc_out_sum,
                training=False,
                look_ahead_mask=look_ahead,
                padding_mask=enc_padding_mask,
            )

            last_hidden = dec_out[:, -1:, :]
            raw_logits  = self.final_sum_proj(last_hidden)

            if self.use_copy and cross_attn is not None:
                last_cross  = cross_attn[:, -1:, :]
                context     = torch.bmm(last_cross, enc_out_sum)
                last_emb    = tar_emb_pe[:, -1:, :]
                p_gen       = self.copy_gate(last_hidden, context, last_emb)

                vocab_probs = torch.softmax(raw_logits, dim=-1)
                copy_dist   = torch.zeros(B, 1, self.vocab_size, device=device)
                src_safe    = inp.clamp(0, self.vocab_size - 1).unsqueeze(1)
                copy_dist.scatter_add_(2, src_safe, last_cross)

                final_probs = p_gen * vocab_probs + (1.0 - p_gen) * copy_dist
                next_token  = final_probs.squeeze(1).argmax(dim=-1)
            else:
                next_token = raw_logits.squeeze(1).argmax(dim=-1)

            if self.sep_idx is not None:
                finished = finished | (next_token == self.sep_idx)

            dec_in = torch.cat([dec_in, next_token.unsqueeze(1)], dim=1)

            if self.sep_idx is not None and finished.all():
                break

        results = []
        for b in range(B):
            seq = dec_in[b, 1:].cpu().tolist()
            if self.sep_idx is not None and self.sep_idx in seq:
                seq = seq[: seq.index(self.sep_idx)]
            results.append(seq)
        return results

    # ------------------------------------------------------------------ #
    #  Beam Search với copy mechanism                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def beam_search_generate_batch(self, inp, max_len, beam_size=4, len_penalty=0.6,
                                   n_gram_block=3, kg_batch=None):
        self.eval()
        device     = inp.device
        batch_size = inp.size(0)

        enc_padding_mask = (inp == self.pad_idx).to(device).unsqueeze(1).unsqueeze(2)
        x = self.embedding(inp) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        enc_out_shared = self.encoder(x, training=False, mask=enc_padding_mask)

        # OntoKG fusion
        enc_out_shared = self._apply_ontokg_fusion(enc_out_shared, kg_batch, device)

        if self.use_mmoe:
            mask_tokens = (inp != self.pad_idx)
            mmoe_outs   = self.mmoe(enc_out_shared, mask=mask_tokens, return_gates=False)
            enc_out_sum = mmoe_outs[0]
        else:
            enc_out_sum = enc_out_shared

        enc_out_exp      = enc_out_sum.repeat_interleave(beam_size, dim=0)
        enc_pad_mask_exp = enc_padding_mask.repeat_interleave(beam_size, dim=0)
        src_exp          = inp.repeat_interleave(beam_size, dim=0)

        cls_token     = self.cls_idx if self.cls_idx is not None else 1
        decoder_input = torch.full(
            (batch_size * beam_size, 1), cls_token, dtype=torch.long, device=device
        )
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        done = [False] * (batch_size * beam_size)

        for step in range(max_len):
            T = decoder_input.size(1)
            look_ahead = (
                torch.triu(torch.ones(T, T, device=device), diagonal=1)
                .bool().unsqueeze(0).unsqueeze(1)
            )
            tar_emb    = self.embedding(decoder_input) * math.sqrt(self.d_model)
            tar_emb_pe = self.pos_encoding(tar_emb)

            dec_out, cross_attn = self.decoder_sum(
                tar_emb_pe, enc_out_exp,
                training=False,
                look_ahead_mask=look_ahead,
                padding_mask=enc_pad_mask_exp,
            )

            last_hidden = dec_out[:, -1:, :]
            raw_logits  = self.final_sum_proj(last_hidden)

            if self.use_copy and cross_attn is not None:
                last_cross = cross_attn[:, -1:, :]
                context    = torch.bmm(last_cross, enc_out_exp)
                last_emb   = tar_emb_pe[:, -1:, :]
                p_gen      = self.copy_gate(last_hidden, context, last_emb)

                vocab_probs = torch.softmax(raw_logits, dim=-1)
                Bb = batch_size * beam_size
                copy_dist = torch.zeros(Bb, 1, self.vocab_size, device=device)
                src_safe  = src_exp.clamp(0, self.vocab_size - 1).unsqueeze(1)
                copy_dist.scatter_add_(2, src_safe, last_cross)

                final_probs = p_gen * vocab_probs + (1.0 - p_gen) * copy_dist
                log_probs   = torch.log(final_probs.clamp(min=1e-9)).squeeze(1)
            else:
                log_probs = F.log_softmax(raw_logits.squeeze(1), dim=-1)

            if n_gram_block > 0 and T >= n_gram_block:
                cur_seqs = decoder_input.cpu().numpy()
                for idx in range(batch_size * beam_size):
                    if done[idx]: continue
                    seq        = cur_seqs[idx].tolist()
                    cur_prefix = tuple(seq[-(n_gram_block - 1):])
                    for i in range(len(seq) - n_gram_block + 1):
                        if tuple(seq[i: i + n_gram_block - 1]) == cur_prefix:
                            log_probs[idx, seq[i + n_gram_block - 1]] = -float("inf")

            next_scores = (beam_scores.unsqueeze(-1) + log_probs).view(
                batch_size, beam_size * self.vocab_size
            )
            top_scores, top_indices = torch.topk(next_scores, beam_size, dim=1)

            beam_indices  = top_indices // self.vocab_size
            token_indices = top_indices % self.vocab_size
            batch_offset  = (torch.arange(batch_size, device=device) * beam_size).unsqueeze(1)
            global_beam   = (batch_offset + beam_indices).view(-1)

            decoder_input = torch.cat([decoder_input[global_beam], token_indices.view(-1, 1)], dim=1)
            beam_scores   = top_scores.view(-1)

            prior_done = [done[i] for i in global_beam.cpu().numpy()]
            just_done  = (token_indices.view(-1) == self.sep_idx).cpu().numpy()
            done       = [p or j for p, j in zip(prior_done, just_done)]
            if all(done):
                break

        final_sequences = decoder_input.view(batch_size, beam_size, -1)
        final_scores    = beam_scores.view(batch_size, beam_size)
        lengths         = (final_sequences != self.pad_idx).sum(dim=2).float()
        scores_pen      = final_scores / (lengths ** len_penalty + 1e-8)
        best_idx        = scores_pen.argmax(dim=1)

        return [final_sequences[b, best_idx[b]].cpu().numpy() for b in range(batch_size)]