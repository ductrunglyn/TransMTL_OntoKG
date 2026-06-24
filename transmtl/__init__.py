# transmtl/ — gói model + huấn luyện + dữ liệu + tích hợp OntoKG
"""
Package TransMTL: kiến trúc model đa nhiệm (tóm tắt + từ khóa) và toàn bộ
thành phần huấn luyện/đánh giá/tích hợp đồ thị tri thức.

Module:
  model        — TransformerMTL (Encoder/Decoder, MMoE, Copy, CRF, fusion OntoKG)
  train        — train_model(), train_one_epoch(), validate()
  tester       — run_test() đánh giá ROUGE + keyphrase
  data         — MultiTaskDataset, get_loaders()
  preprocessing— FastText embedding, BPE, BIOES, ids<->text
  losses       — hàm mất mát + tiện ích huấn luyện
  evaluation   — chỉ số keyphrase / ROUGE
  fusion       — GraphEncoder (R-GCN+GATv2) + GatedFusion
  bridge       — OntoKGBridge: truy vấn Neo4j theo article_id
"""
