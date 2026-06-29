# OntoKG‑TransMTL — Tài liệu kỹ thuật

> Hệ thống **học đa nhiệm tiếng Việt** (tóm tắt văn bản + trích xuất từ khóa) được **tích hợp đồ thị tri thức có bản thể học (OntoKG)**.
> Tài liệu mô tả kiến trúc, vai trò và **sản phẩm (output)** của từng file/module, luồng dữ liệu, cách hệ thống được vận hành, định dạng dữ liệu và các vấn đề thường gặp.

---

## 1. Tổng quan kiến trúc

Hệ thống được tổ chức thành **hai phần lớn**:

```
            ┌─────────────────────── PHẦN 1: OntoKG (module 1→9) ───────────────────────┐
 CSV tin tức → preprocess → NER+concept → entity linking → relation → KG → ontology → KGE
            └───────────────────────────────────┬───────────────────────────────────────┘
                                                 │  (subgraph theo article_id được truy vấn ở module 9)
            ┌────────────────────────────────────▼──────────────────────────────────────┐
 CSV tin tức → Text Encoder ──► H_tok ──► Gated Fusion ◄── E_kg ◄── Graph Encoder(R‑GCN+GATv2)
            →                     │                                                       
            →                 H'_tok ──► MMoE ──► ┬─► Decoder + Copy ──► Tóm tắt          
            →                                     └─► CRF/BIOES        ──► Từ khóa         
            └────────────────────── PHẦN 2: TransMTL (model) ──────────────────────────────┘
```

- **Thiết kế "additive":** khi `kg_batch=None` (hoặc `USE_ONTOKG=False`), nhánh đồ thị được tắt và model vận hành như **baseline TransMTL thuần** — phục vụ ablation công bằng.
- **Giao tiếp giữa hai phần:** lúc train/test, subgraph của từng bài được lấy theo `article_id` qua `OntoKGBridge` rồi đưa vào model. Subgraph này có thể được lấy theo **một trong hai backend** (xem Mục 8): đọc thẳng file (`local`) hoặc truy vấn Neo4j (`neo4j`).

---

## 2. Cây thư mục & vai trò từng file

```
TransMTLOntoKG/
├── pipeline_config.py     ★ CẤU HÌNH TRUNG TÂM: đường dẫn dữ liệu, USE_ONTOKG, ONTOKG_BACKEND, Neo4j, thư mục output
├── conf.py                ★ HYPERPARAMETER model: d_model, layers, lr, batch, MMoE...
├── requirements.txt       Danh sách thư viện phụ thuộc
│
├── run_ontokg.py          ★ BƯỚC 1: xây OntoKG (split + module 1‑7, và 8 nếu dùng Neo4j). Chạy 1 lần.
├── run_transmtl.py        ★ BƯỚC 2: train + test TransMTL. Chạy lại nhiều lần.
├── main.py                ★ Orchestrator 1 lệnh: --stage {all,split,ontokg,train,test}, --use-ontokg/--no-ontokg, --skip-existing
├── split_dataset.py       Chia CSV gốc -> data_split/{train,val,test,trainval}.csv (+ cột article_id)
│
├── transmtl/              ★ PACKAGE MODEL + HUẤN LUYỆN + TÍCH HỢP KG
│   ├── __init__.py
│   ├── model.py           ★ KIẾN TRÚC MODEL: Encoder/Decoder, MMoE, Copy, CRF, fusion OntoKG, beam/greedy
│   ├── train.py           Vòng lặp huấn luyện: train_model(), train_one_epoch(), validate()
│   ├── tester.py          Đánh giá test: run_test() -> ROUGE + keyphrase P/R/F1 (+ OntoKG bridge)
│   ├── data.py            Dataset + DataLoader: MultiTaskDataset, CollateCPU, get_loaders() (trả article_id)
│   ├── preprocessing.py   Tiền xử lý: FastText embedding, BPE tokenizer, BIOES, ids<->text, keyphrase
│   ├── losses.py          Hàm loss + tiện ích: summary loss, key loss, gộp trọng số đa nhiệm, load checkpoint
│   ├── evaluation.py      Chỉ số đánh giá: keyphrase P/R/F1, ROUGE summary
│   ├── fusion.py          ★ CẦU NỐI MODEL↔KG: GraphEncoder (R‑GCN+GATv2), GatedFusion, encode_kg_batch (có chống NaN)
│   └── bridge.py          ★ OntoKGBridge: lấy subgraph theo article_id -> kg_batch (backend 'local' hoặc 'neo4j')
│
├── OntoKG/                ★ PIPELINE XÂY KG (xem Mục 7)
│   ├── module1_preprocess.py            Chuẩn hóa văn bản, tách câu/từ, POS
│   ├── module2_ner_concept.py           NER (PhoBERT/ELECTRA) + trích khái niệm
│   ├── module3_entity_linking.py        Liên kết thực thể 4 tầng + giải đồng nghĩa (alias)
│   ├── module4_relation_extraction.py   Trích bộ ba (h,r,t) ràng buộc ontology
│   ├── module5_kg_construction.py       Dựng KG (RDF/Turtle, NetworkX, entity_index, pykeen TSV)
│   ├── module6_ontology_learning.py     UMAP + HDBSCAN -> sinh lớp khái niệm
│   ├── module7_kge_training.py          KGE TransE (PyKEEN) + trộn PhoBERT -> entity_embeddings.pt
│   ├── module8_neo4j_loader.py          Nạp KG vào Neo4j (CHỈ cần khi ONTOKG_BACKEND='neo4j')
│   ├── module9_neo4j_retrieval.py       Truy vấn subgraph qua Neo4j (backend 'neo4j')
│   ├── module9_local_retrieval.py       ★ Truy vấn subgraph từ FILE — KHÔNG cần Neo4j (backend 'local', mặc định)
│   └── aliases.json                     Từ điển alias thủ công (HN->Hà Nội, TP.HCM->...)
│
├── data_split/   (gitignored)  train/val/test/trainval.csv
├── data/         (gitignored)  toàn bộ artifact OntoKG (jsonl, kg/, kge/, ontology/)
└── Results_Score/(gitignored)  checkpoint model + *_test_results.txt
```

---

## 3. Cấu hình

### 3.1 `pipeline_config.py` — bảng điều khiển trung tâm
| Biến | Ý nghĩa |
|---|---|
| `RAW_DATA_CSV` | CSV gốc (cột: `title, summary, content, publish_time, topic, cleaned_keywords`) |
| `PRETRAINED_VEC` | FastText `.bin` tiếng Việt (cc.vi.300.bin) |
| `USE_ONTOKG` | **Công tắc thường** True/False — bật/tắt nhánh OntoKG |
| `ONTOKG_BACKEND` | `"local"` (mặc định, đọc file, KHÔNG cần Neo4j) hoặc `"neo4j"` (truy vấn server) |
| `NEO4J_URI/USER/PASSWORD/DATABASE` | Kết nối Neo4j (chỉ dùng khi backend = `"neo4j"`) |
| `SPLIT_DIR / DATA_DIR / SAVE_PATH` | Thư mục output |
| `ENTITY_EMB / ENTITY_IDX_JSON / M4_TRIPLES` | Artifact OntoKG mà backend local đọc trực tiếp |
| `TRAIN_DATA_CSV / TEST_DATA_CSV` | Mặc định = `trainval.csv` (get_loaders tự chia held‑out theo seed=42) |
| `OKG_DEVICE` | cuda/cpu cho các module nặng |
| `ensure_dirs()`, `ontokg_env()` | tạo thư mục; xuất biến môi trường cho subprocess module 1‑8 |

> Mọi giá trị (trừ `USE_ONTOKG`) được phép override bằng biến môi trường cùng tên.

### 3.2 `conf.py` — hyperparameter model
`D_MODEL=300, NUM_LAYER=4, NUM_HEADS=6, DFF=1024, DROPOUT=0.3, BATCH_SIZE=24, NUM_EPOCHS=100, LR_BASE=5e-5, LABEL_SMOOTHING=0.15, SIZE_VOCAB=40000`, nhóm MMoE (`USE_MMOE, MMOE_NUM_EXPERTS=4,...`), `LABELS` (BIOES: O/B/I/E/S), `IGNORE_INDEX=-100`, `PAD_IDX=0`.

### 3.3 Biến môi trường đặc biệt (tăng tốc / điều khiển module 3)
| Env | Tác dụng |
|---|---|
| `ONTOKG_BACKEND` | Override backend OntoKG (`local`/`neo4j`) |
| `OKG_EMB_MATCH=0` | Tắt so khớp embedding (Tầng 3) ở entity linking — nhanh nhất |
| `OKG_WD_ALIASES=0` | Tắt nạp alias Wikidata |
| `OKG_ALIAS_JSON=path` | Đổi file từ điển alias (mặc định `OntoKG/aliases.json`) |
| `OKG_DATA_DIR / OKG_INPUT_CSV / OKG_DEVICE` | Được `main.py` tự set cho module 1‑8 |

---

## 4. Luồng dữ liệu end‑to‑end (data flow)

```
RAW_DATA_CSV
  │ split_dataset.py
  ▼
data_split/{train,val,test,trainval}.csv         (thêm cột article_id = "<split>_000123")
  │ module1 (input = trainval.csv)
  ▼ data/preprocessed_articles.jsonl
  │ module2
  ▼ data/module2_ner_concept.jsonl
  │ module3  (+ data/entity_registry.pkl, data/wikidata_cache.json)
  ▼ data/module3_entity_linked.jsonl
  │ module4
  ▼ data/module4_triples.jsonl  (+ module4_enriched.jsonl)
  │ module5
  ▼ data/kg/{kg_global.ttl, kg_networkx.pkl, entity_index.pkl, pykeen_triples.tsv}
  │ module6                         module7
  ▼ data/ontology/*                 ▼ data/kge/{entity_embeddings.pt, entity_to_idx.json, idx_to_uri.json}
  │
  ├── (backend 'local')  module9_local_retrieval đọc thẳng entity_embeddings.pt + entity_to_idx.json + module4_triples.jsonl
  │
  └── (backend 'neo4j')  module8 nạp Neo4j ──► module9_neo4j_retrieval truy vấn DB
        │
        ▼ subgraph theo article_id (OntoKGBridge)
        ▼ kg_batch ──► TransformerMTL
```

---

## 5. Cách hệ thống được vận hành

### 5.1 Hai file runner
| Lệnh | Tác dụng |
|---|---|
| `python run_ontokg.py` | **Bước 1** — split (nếu chưa) + module 1‑7 (xây KG). Module 8 chỉ được chạy khi backend = `neo4j`. Tự `--skip-existing`. |
| `python run_transmtl.py` | **Bước 2** — split (nếu chưa) + train + test. `USE_ONTOKG` quyết định bật/tắt KG; `ONTOKG_BACKEND` quyết định nguồn subgraph. |

> Cả hai được gọi qua `make_args(cfg_module=P)`, nên khi `import pipeline_config` được thay bằng config khác (vd `pipeline_config_baseline`) thì config đó được áp dụng xuyên suốt mọi stage.

### 5.2 `main.py` — orchestrator linh hoạt
```bash
python main.py                       # theo USE_ONTOKG trong config
python main.py --use-ontokg          # ép bật KG cho lần chạy này
python main.py --no-ontokg           # ép baseline
python main.py --stage split         # chỉ chia dữ liệu
python main.py --stage ontokg --use-ontokg   # chỉ xây KG
python main.py --stage train         # chỉ train
python main.py --stage test          # chỉ test
python main.py --skip-existing       # resume (bỏ qua module/đầu ra đã có)
```

### 5.3 Các kịch bản phổ biến
- **Baseline TransMTL:** `USE_ONTOKG=False` → `python run_transmtl.py` (không cần OntoKG, không cần Neo4j).
- **TransMTL + OntoKG (không Neo4j):** `USE_ONTOKG=True`, `ONTOKG_BACKEND="local"` (mặc định) → `python run_ontokg.py` (1 lần) rồi `python run_transmtl.py`.
- **TransMTL + OntoKG (qua Neo4j):** `USE_ONTOKG=True`, `ONTOKG_BACKEND="neo4j"` → cần Neo4j chạy + module 8 nạp KG.

---

## 6. Chi tiết package `transmtl/` (Phần 2 — TransMTL)

### 6.1 `model.py` — `class TransformerMTL`
Kiến trúc Transformer encoder–decoder đa nhiệm được tích hợp OntoKG.
- **Khối con:** `PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `EncoderLayer/Encoder`, `DecoderLayer/Decoder_Sum`, `CopyGate`, `MMoE`.
- **Tham số khởi tạo quan trọng:** `emb_matrix` (FastText), `word2idx/idx2word`, `num_key_labels=5` (BIOES), `use_mmoe`, `use_copy=True`, `use_ontokg`, `kg_in_dim=768`, `kg_num_relations=9`.
- **Hai đầu ra (heads):**
  - Tóm tắt: `Decoder_Sum` → `summary_logits`; khi copy bật → `summary_log_probs` (pointer‑generator qua `CopyGate`).
  - Từ khóa: `final_key_proj` (d_model→5) → emissions → **CRF** (`crf_decoder`).
- **`_apply_ontokg_fusion(enc_out_shared, kg_batch, device)`** — chỉ được chạy khi `use_ontokg` và có `kg_batch`: `encode_kg_batch` → `GatedFusion`. Ngược lại trả nguyên `enc_out_shared` (baseline).
- **`forward(...)`** → dict gồm `summary_logits`/`summary_log_probs`, `key_nll`/`key_decoded`.
- **Sinh chuỗi:** `greedy_decode_batch(...)` và `beam_search_generate_batch(...)` (auto‑regressive, có copy, n‑gram blocking).

### 6.2 `train.py`
- **`train_model(...)`**: loader được tạo (`get_loaders`), FastText được nạp, `TransformerMTL` được dựng, **`OntoKGBridge`** được khởi tạo theo `ontokg_backend`. Optimizer được tổ chức **theo nhóm tham số** (summary/shared/crf, lr nhân hệ số); `weight_logits` (trọng số đa nhiệm học được) có optimizer riêng; scheduler `ReduceLROnPlateau`; PCGrad chống xung đột gradient. Tham số OntoKG: `use_ontokg, entity_emb_path, entity_idx_path, neo4j_uri, neo4j_pass, ontokg_backend`.
- **`train_one_epoch(...)` / `validate(...)`**: batch **8‑tuple** (có `article_ids`) được unpack; `kg_batch = bridge.build_kg_batch(article_ids)`; model được gọi kèm `kg_batch`. ROUGE được tính qua greedy decode lúc validate.

### 6.3 `tester.py` — `run_test(...)`
Loader được tạo, model được dựng (khớp `use_ontokg`), checkpoint được nạp (`load_checkpoint_state`, `strict=False`), **bridge OntoKG** được khởi tạo theo `ontokg_backend`. Tóm tắt được sinh bằng beam/greedy → **ROUGE‑1/2/L** + map subword→word → **keyphrase P/R/F1**. Kết quả được ghi `<ckpt>_test_results.txt`.

### 6.4 `data.py`
- **`MultiTaskDataset`**: CSV được đọc, BPE tokenize, nhãn **BIOES** cho từ khóa được sinh, `article_ids` được lưu (cột `article_id`, fallback `row_xxxxx`).
- **`CollateCPU`**: batch được pad → trả **8 phần tử**: `src, summary_ids, attn, labels, raw_texts, token_maps, word_texts, article_ids`.
- **`get_loaders(...)`** → train/val/test loader + vocab + word2idx/idx2word + `ds`. Dữ liệu được **tự chia 60/20/20** (seed cố định ⇒ tập test held‑out tái lập được).

### 6.5 `preprocessing.py`
`seed_everything`, `load_fasttext_bin_embeddings(...)`, BPE tokenizer (`ensure_tokenizer_for_csv`), ánh xạ keyword→span, `subword_labels_to_word_labels`, `ids_to_text`, `convert_tags_to_keyphrases`.

### 6.6 `losses.py`
`compute_summary_loss_from_logits` (CE + label smoothing), `compute_summary_loss_from_logprobs` (NLL cho copy), `compute_key_loss_from_raw`, `compute_entropy_regularizer`, **`combine_task_losses(...)`** (trọng số đa nhiệm học được), `ensure_decoder_sos`, `load_checkpoint_state`.

### 6.7 `evaluation.py`
`evaluate_keyphrase_lists` (P/R/F1 theo tập cụm từ), `evaluate_summaries` (ROUGE).

### 6.8 `fusion.py` (cầu nối model↔KG)
- **`GraphEncoder(in_dim=768, d_model, num_relations=9, ...)`**: `Linear(768→d) → RGCNConv×2 → GATv2Conv`. Đầu vào được đưa qua `nan_to_num` để chống NaN từ embedding hỏng. Cần `torch_geometric`; nếu thiếu → fallback MLP.
- **`GatedFusion(...)`**: cross‑attention (query=H_tok, key/value=E_kg) + **cổng sigmoid** + residual + LayerNorm → `H'_tok`. Đầu ra attention được đưa qua `nan_to_num` để phòng hờ.
- **`encode_kg_batch(...)`**: list subgraph được mã hóa → tensor padded `E_kg (B,N_max,d)` + `padding_mask`. **Chống NaN:** bài không có entity (subgraph rỗng) được giữ một slot vector‑0 **không bị mask**, nhờ đó không tồn tại hàng `key_padding_mask` toàn `True` (vốn khiến `MultiheadAttention` sinh NaN); sample đó vận hành như baseline.

### 6.9 `bridge.py` — `class OntoKGBridge`
`__init__(uri, user, password, d_model, enabled, backend='local'|'neo4j', entity_emb_path, entity_idx_path)`. Khi `enabled`:
- backend `local` → `LocalKGRetriever` được khởi tạo (đọc file, đường dẫn `module4_triples.jsonl` được suy ra từ thư mục `data/`).
- backend `neo4j` → `Neo4jRetriever` được khởi tạo (truy vấn server).

**`build_kg_batch(article_ids)`** → list subgraph torch (hoặc None nếu tắt/không có entity). `close()`.

---

## 7. Chi tiết PIPELINE OntoKG — INPUT → SẢN PHẨM từng module

> Đường dẫn dưới đây tương đối với `DATA_DIR` (mặc định `./data`).

### Module 1 — `module1_preprocess.py` (Tiền xử lý)
- **Input:** `trainval.csv` (qua env `OKG_INPUT_CSV`).
- **Sản phẩm:** `preprocessed_articles.jsonl` (tokens, sentences, `full_text_pos`, topic_list, cleaned_keywords, `article_id`), `preprocess_errors.jsonl`.

### Module 2 — `module2_ner_concept.py` (NER + khái niệm)
- **Input:** `preprocessed_articles.jsonl`.
- **Xử lý:** NER bằng PhoBERT/NlpHUST‑ELECTRA (GPU+fp16) + underthesea, nhãn chuẩn hóa về `{PER,ORG,LOC,TIME,EVENT,MISC}`; khái niệm được trích từ noun‑phrase.
- **Sản phẩm:** `module2_ner_concept.jsonl` (`ner_entities[]`, `concept_mentions[]`).

### Module 3 — `module3_entity_linking.py` (Liên kết + giải đồng nghĩa) ★
- **Input:** `module2_ner_concept.jsonl`.
- **Xử lý — 4 tầng (`_link_one`):** (0) chuẩn hóa alias thủ công → (1) khớp bề mặt registry → (2) Wikidata (LOC/PER/ORG, có cache + nạp alias) → (3) so khớp embedding cùng nhãn → (4) tạo URI nội bộ mới. URI duy nhất được gán; biến thể đồng nghĩa được gộp.
- **Sản phẩm:** `module3_entity_linked.jsonl`, `entity_registry.pkl`, `wikidata_cache.json`.

### Module 4 — `module4_relation_extraction.py` (Quan hệ)
- **Input:** `module3_entity_linked.jsonl`.
- **Xử lý:** metadata triple (`belongsTo`, `hasEntity`, `rdf:type`) + 9 quan hệ ngữ nghĩa ràng buộc domain/range: `occursAt, occursOn, organizedBy, participatesIn, locatedIn, manages, hasPart, causedBy, relatedTo`.
- **Sản phẩm:** `module4_triples.jsonl`, `module4_enriched.jsonl`.

### Module 5 — `module5_kg_construction.py` (Dựng KG)
- **Input:** `module3_entity_linked.jsonl` + `module4_triples.jsonl`.
- **Sản phẩm (`kg/`):** `kg_global.ttl` (RDF, cần `rdflib`), `kg_networkx.pkl`, **`entity_index.pkl`**, **`pykeen_triples.tsv`** (đầu vào module 7; metadata triple `rdf:type`/`hasEntity` được loại khỏi KGE).

### Module 6 — `module6_ontology_learning.py` (Học ontology)
- **Input:** `kg/entity_index.pkl`.
- **Xử lý:** giảm chiều **UMAP** → phân cụm **HDBSCAN** → gợi ý lớp khái niệm mới. (Khi thiếu UMAP/HDBSCAN, PCA/KMeans được dùng thay thế và chất lượng giảm.)
- **Sản phẩm (`ontology/`):** `ontology_v1.1.json`, `cluster_report.txt`, `cluster_matrix.npy`.

### Module 7 — `module7_kge_training.py` (KGE) ★
- **Input:** `kg/pykeen_triples.tsv` + `kg/entity_index.pkl`.
- **Xử lý:** **TransE (PyKEEN)** dim 256 được huấn luyện → chiếu lên 768 → **trộn với biểu diễn PhoBERT**.
- **Sản phẩm (`kge/`):** **`entity_embeddings.pt` (N×768) — ĐẦU VÀO CHO TRANSMTL**, `entity_to_idx.json`, `idx_to_uri.json`.

### Module 8 — `module8_neo4j_loader.py` (Nạp Neo4j — chỉ backend `neo4j`)
- **Input:** `kg/entity_index.pkl`, `kge/entity_embeddings.pt`, `kge/entity_to_idx.json`, `module4_triples.jsonl`; creds qua env `NEO4J_*`.
- **Sản phẩm:** Neo4j DB gồm `:Entity`, `:Article`, `:Topic` + quan hệ ngữ nghĩa + `HAS_ENTITY`.
- **Lưu ý:** ở backend `local`, module này được **bỏ qua hoàn toàn**.

### Module 9 (Neo4j) — `module9_neo4j_retrieval.py`
- `Neo4jRetriever` với `SEMANTIC_RELATIONS` (9 loại). `get_article_subgraph(article_id)` → {uris, node_feat, edge_index, edge_type}; `subgraph_to_torch(sg)` → tensor cho GraphEncoder.

### Module 9 (Local) — `module9_local_retrieval.py` ★ (mặc định)
- `LocalKGRetriever` cung cấp **đúng giao diện** như `Neo4jRetriever` nhưng đọc thẳng từ file, **không cần Neo4j/Docker**.
- **Input:** `entity_embeddings.pt` + `entity_to_idx.json` + `module4_triples.jsonl`.
- **Cơ chế:** ở lần khởi tạo, triples được quét một lần để dựng chỉ mục `article → entities` (qua `hasEntity`) và `entity → entity` (9 quan hệ ngữ nghĩa). `get_article_subgraph(article_id)` trả về subgraph gồm các entity của bài + các cạnh ngữ nghĩa giữa chúng, node feature lấy từ ma trận embedding — đồng nhất định dạng/ngữ nghĩa với backend Neo4j.

---

## 8. Hai backend OntoKG

| | `local` (mặc định) | `neo4j` |
|---|---|---|
| Phụ thuộc | chỉ cần file `data/` | Neo4j server + module 8 |
| Module 8 | bỏ qua | bắt buộc |
| Nguồn subgraph | `module9_local_retrieval.py` | `module9_neo4j_retrieval.py` |
| Bộ nhớ | nạp ma trận embedding (~298k×768 ≈ 0.9 GB) vào RAM một lần | embedding nằm trong DB |
| Ưu điểm | không cần cài đặt ngoài, tái lập dễ | hỗ trợ truy vấn vector index, k‑hop |

Backend được chọn qua `ONTOKG_BACKEND` trong `pipeline_config.py` (hoặc biến môi trường cùng tên). Khi không khai báo, `local` được dùng mặc định.

---

## 9. Định dạng dữ liệu chuẩn

- **CSV gốc:** `title, summary, content, publish_time, topic, cleaned_keywords` (+ `article_id` do `split_dataset` thêm).
- **Batch (CollateCPU, 8‑tuple):** `src, summary_ids, attn, labels(BIOES id), raw_texts, token_maps, word_texts, article_ids`.
- **Nhãn từ khóa (BIOES):** `O=0,B=1,I=2,E=3,S=4` (conf.LABELS), `IGNORE_INDEX=-100`, `PAD_IDX=0`.
- **subgraph torch:** `{x:(N,768) float, edge_index:(2,E) long, edge_type:(E,) long∈[0,8]}`.

---

## 10. Hàm loss & vòng huấn luyện
- **Tóm tắt:** CrossEntropy (label smoothing) hoặc **NLL** khi copy bật.
- **Từ khóa:** **CRF negative log‑likelihood** (`key_nll`).
- **Gộp đa nhiệm:** `combine_task_losses(...)` — `weight_logits` là tham số **học được** (softmax), optimizer riêng.
- Entropy regularizer (tùy chọn) cho gate MMoE; **PCGrad** chống xung đột gradient.
- Optimizer AdamW theo nhóm, scheduler ReduceLROnPlateau, checkpoint được chọn theo val score.

---

## 11. Vấn đề thường gặp (troubleshooting)
| Triệu chứng | Nguyên nhân & hướng xử lý |
|---|---|
| `FileNotFoundError: trainval.csv` | Chưa split — runner sẽ tự split khi `RAW_DATA_CSV` đúng. |
| `Train Loss nan` ngay epoch 1 (khi bật OntoKG) | Bài không có entity → subgraph rỗng → attention bị mask toàn bộ → NaN. Đã được vá trong `fusion.py` (giữ slot zero + `nan_to_num`). |
| Nghi `entity_embeddings.pt` chứa NaN | Kiểm tra `torch.isnan(torch.load(...)).any()`; `GraphEncoder` đã `nan_to_num` đầu vào để chống. |
| Kết nối Neo4j thất bại (backend `neo4j`) | Kiểm tra server chạy ở `NEO4J_URI`, đúng cổng (Bolt mặc định 7687) và `NEO4J_PASSWORD`. Hoặc chuyển sang backend `local`. |
| Module 3 chậm / số entity bùng nổ | Đã tối ưu O(N²) (vector hoá + bucket theo nhãn). `OKG_EMB_MATCH=0` để nhanh tối đa. |
| `ImportError: torch_geometric` | GraphEncoder cần PyG; thiếu thì fallback MLP (mất R‑GCN/GAT). |
| KG nạp rỗng ở backend local | Log "Local KG: 0 canh hasEntity" — kiểm tra `data/module4_triples.jsonl` tồn tại đúng thư mục `data/`. |
| GitHub từ chối push (file >100MB) | `data_split/*.csv`, `*.pt` lỡ commit — `.gitignore` đã chặn; cần dọn khỏi lịch sử. |

---

## 12. Hạn chế / chưa implement
Một số nội dung trong bản thảo bài báo **chưa có trong code**:
- **Streaming "Frozen Model – Evolving Graph" / `module10`:** chưa tồn tại. Inductive Router / Knowledge Buffer chưa có.
- **EWC** và **tái huấn luyện KGE/online ontology theo chu kỳ:** chưa có; module 6/7 chạy một lần (batch).
- **Topic Classifier** (đầu tác vụ thứ 3): không có; model chỉ có **2 đầu** (summary + keyphrase).
- **Copy trỏ thẳng vào E_kg:** copy hiện trỏ vào **token nguồn** (đã được làm giàu tri thức qua fusion), không copy trực tiếp nhãn KG.
- **Đánh giá test có OntoKG:** subgraph chỉ tồn tại cho bài đã được xây KG (train+val); bài test chưa được liên kết ⇒ KG đóng góp hạn chế ở test.

---

## 13. Phụ thuộc
`torch, transformers, underthesea, rouge-score, pandas, numpy, torchcrf, torch_geometric` (lõi train/test); `neo4j, pykeen, umap-learn, hdbscan, rdflib, networkx, scikit-learn` (xây OntoKG / backend Neo4j). Baseline TransMTL chỉ cần nhóm lõi; backend `local` không cần `neo4j`.
