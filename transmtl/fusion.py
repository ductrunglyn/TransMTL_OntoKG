# ontokg_fusion.py
"""
Cac thanh phan tich hop OntoKG vao TransMTL.
Them vao model: GraphEncoder (R-GCN + GAT) + GatedFusion.

Xu ly mismatch dimension: entity embedding 768 chieu -> d_model (vd 300).
Thiet ke ADDITIVE: neu kg_batch=None, model chay nhu cu (phuc vu ablation).

Dependencies: torch, torch_geometric (cho R-GCN/GAT)
  pip install torch_geometric
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import RGCNConv, GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# So loai quan he ngu nghia (khop SEMANTIC_RELATIONS trong module9)
NUM_RELATIONS = 9


class GraphEncoder(nn.Module):
    """
    Ma hoa subgraph OntoKG: 768-d entity embedding -> d_model.
    Kien truc: Linear(768->d) -> R-GCN x2 (theo loai quan he) -> GAT-v2.
    """
    def __init__(self, in_dim=768, d_model=300, num_relations=NUM_RELATIONS,
                 num_bases=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        if HAS_PYG:
            self.rgcn1 = RGCNConv(d_model, d_model, num_relations, num_bases=num_bases)
            self.rgcn2 = RGCNConv(d_model, d_model, num_relations, num_bases=num_bases)
            self.gat   = GATv2Conv(d_model, d_model, heads=1, concat=False)
        else:
            # Fallback: chi dung projection + MLP neu khong co torch_geometric
            self.fallback = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        """
        x:          (N, 768) node features
        edge_index: (2, E)
        edge_type:  (E,)
        -> tra ve (N, d_model)
        """
        h = self.proj(torch.nan_to_num(x))    # 768 -> d_model (chống NaN từ embedding)
        if not HAS_PYG or edge_index.size(1) == 0:
            # Khong co canh hoac thieu PyG -> chi dung node feature
            return self.dropout(self.fallback(h) if not HAS_PYG else h)
        h = F.relu(self.rgcn1(h, edge_index, edge_type))
        h = self.dropout(h)
        h = F.relu(self.rgcn2(h, edge_index, edge_type))
        h = self.gat(h, edge_index)
        return self.dropout(h)


class GatedFusion(nn.Module):
    """
    Hoi tu bieu dien token (H_tok) voi entity embedding (E_kg).
    Cross-attention + sigmoid gate -> H'_tok.
    Cong sigmoid tu dieu phoi ty trong text vs KG (chong hallucination).
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.gate  = nn.Linear(d_model * 2, 1)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, H_tok, E_kg, kg_padding_mask=None):
        """
        H_tok:           (B, T, d) bieu dien token tu text encoder
        E_kg:            (B, N, d) entity embedding tu GraphEncoder
        kg_padding_mask: (B, N) True o vi tri padding (entity gia)
        -> tra ve (B, T, d) H'_tok da tich hop tri thuc do thi
        """
        attn_out, _ = self.cross_attn(
            query=H_tok, key=E_kg, value=E_kg,
            key_padding_mask=kg_padding_mask,
        )
        # Chong NaN: neu mot sample bi mask toan bo (khong co entity) thi
        # MultiheadAttention tra ve NaN. nan_to_num dua ve 0 -> sample do
        # khong nhan tri thuc KG (tuong duong baseline), khong lam hong batch.
        attn_out = torch.nan_to_num(attn_out)
        gate   = torch.sigmoid(self.gate(torch.cat([H_tok, attn_out], dim=-1)))
        fused  = H_tok + gate * attn_out
        return self.norm(fused)


def encode_kg_batch(graph_encoder, kg_batch, d_model, device):
    """
    Ma hoa list subgraph (moi sample 1 subgraph) -> tensor padded.
    kg_batch: list cac dict {x, edge_index, edge_type} hoac None, do dai = B.

    Tra ve:
      E_kg:        (B, N_max, d_model)
      padding_mask: (B, N_max) True o vi tri padding
    """
    encoded = []
    max_n = 1
    for sg in kg_batch:
        if sg is None or sg["x"].size(0) == 0:
            encoded.append(None)
        else:
            e = graph_encoder(
                sg["x"].to(device),
                sg["edge_index"].to(device),
                sg["edge_type"].to(device),
            )  # (N, d)
            encoded.append(e)
            max_n = max(max_n, e.size(0))

    B = len(kg_batch)
    E_kg = torch.zeros(B, max_n, d_model, device=device)
    mask = torch.ones(B, max_n, dtype=torch.bool, device=device)  # True = padding

    for i, e in enumerate(encoded):
        if e is not None:
            n = e.size(0)
            E_kg[i, :n] = e
            mask[i, :n] = False  # vi tri that
        else:
            # Bai khong co subgraph: giu 1 slot zero KHONG bi mask, tranh
            # hang key_padding_mask toan True -> softmax tren toan -inf -> NaN.
            # Attention vao vector 0 -> khong anh huong (ve baseline cho bai do).
            mask[i, 0] = False

    return E_kg, mask