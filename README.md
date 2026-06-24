# TransMTL_OntoKG

Tóm tắt văn bản tiếng Việt + trích xuất keyphrase đa nhiệm (**TransMTL**),
tích hợp tuỳ chọn đồ thị tri thức bản thể học (**OntoKG / Neo4j**).

---

## 1. Bạn chỉ cần nhớ 3 file

| File | Vai trò | Bạn có sửa không? |
|------|---------|-------------------|
| **`pipeline_config.py`** | Bảng điều khiển: đường dẫn dữ liệu, Neo4j, bật/tắt OntoKG | ✅ Sửa **trước khi chạy** |
| **`conf.py`** | Hyperparameter của model (d_model, lr, batch_size…) | ✅ Sửa nếu muốn tinh chỉnh |
| **`run_ontokg.py` / `run_transmtl.py`** | 2 file để **chạy** (xem mục 4) | ❌ Chỉ chạy, không cần sửa |

> Trước đây bị "loạn" giữa `main.py` và `run_code.py`. Nay đã gọn lại:
> dùng **`run_ontokg.py`** (bước 1) và **`run_transmtl.py`** (bước 2).
> `main.py` chỉ là tuỳ chọn "chạy tất cả trong 1 lệnh" (mục 5) — **không bắt buộc**.

---

## 2. Cài đặt

```bash
pip install torch transformers underthesea rouge-score pandas numpy neo4j \
            pykeen umap-learn hdbscan sentencepiece
```
<<<<<<< ours
(Phần OntoKG cần thêm `transformers`, `pykeen`, `umap-learn`, `hdbscan`; nếu chỉ
chạy baseline thì không cần các gói này.)
=======
TransMTLOntoKG/
├── pipeline_config.py     ★ CẤU HÌNH TRUNG TÂM: đường dẫn dữ liệu, Neo4j, USE_ONTOKG, thư mục output
├── conf.py                ★ HYPERPARAMETER model: d_model, layers, lr, batch, MMoE...
├── requirements.txt       Danh sách thư viện phụ thuộc
│
├── run_ontokg.py          ★ BƯỚC 1: xây OntoKG (split + module 1‑8). Chạy 1 lần.
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
│   ├── fusion.py          ★ CẦU NỐI MODEL↔KG: GraphEncoder (R‑GCN+GATv2), GatedFusion, encode_kg_batch
│   └── bridge.py          ★ OntoKGBridge: truy vấn Neo4j theo article_id -> kg_batch cho model
│
├── OntoKG/                ★ PIPELINE XÂY KG (9 module, xem Mục 7)
│   ├── module1_preprocess.py        Chuẩn hóa văn bản, tách câu/từ, POS
│   ├── module2_ner_concept.py       NER (PhoBERT/ELECTRA) + trích khái niệm
│   ├── module3_entity_linking.py    Liên kết thực thể 4 tầng + giải đồng nghĩa (alias)
│   ├── module4_relation_extraction.py  Trích bộ ba (h,r,t) ràng buộc ontology
│   ├── module5_kg_construction.py   Dựng KG (RDF/Turtle, NetworkX, entity_index, pykeen TSV)
│   ├── module6_ontology_learning.py UMAP + HDBSCAN -> sinh lớp khái niệm
│   ├── module7_kge_training.py      KGE TransE (PyKEEN) + trộn PhoBERT -> entity_embeddings.pt
│   ├── module8_neo4j_loader.py      Nạp KG vào Neo4j
│   ├── module9_neo4j_retrieval.py   Truy vấn subgraph (dùng khi train/test)
│   └── aliases.json                 Từ điển alias thủ công (HN->Hà Nội, TP.HCM->...)
│
├── data_split/   (gitignored)  train/val/test/trainval.csv
├── data/         (gitignored)  toàn bộ artifact OntoKG (jsonl, kg/, kge/, ontology/)
└── Results_Score/(gitignored)  checkpoint model + *_test_results.txt
```
>>>>>>> theirs

---

## 3. Sửa `pipeline_config.py` (chỉ vài dòng)

Mở `pipeline_config.py` và chỉnh đúng 4 chỗ:

```python
RAW_DATA_CSV   = ".../tintuc_gen_final.csv"   # CSV gốc của bạn
PRETRAINED_VEC = ".../cc.vi.300.bin"          # FastText tiếng Việt

USE_ONTOKG     = False     # False = baseline | True = dùng OntoKG
NEO4J_PASSWORD = "password" # chỉ cần khi USE_ONTOKG = True
```

> CSV gốc cần có các cột: `title, summary, content, publish_time, topic, cleaned_keywords`.
>
> Không thích sửa file? Có thể truyền qua biến môi trường:
> `RAW_DATA_CSV=/duong/dan.csv python run_transmtl.py`

---

## 4. CÁCH CHẠY (chọn 1 trong 2 trường hợp)

### 🟢 Trường hợp A — Baseline (KHÔNG dùng OntoKG, không cần Neo4j)

1. Trong `pipeline_config.py`: đặt `USE_ONTOKG = False`.
2. Chạy đúng **một lệnh**:

```bash
python run_transmtl.py
```

File này tự: chia dữ liệu → train TransMTL → test. Xong.

---

<<<<<<< ours
### 🔵 Trường hợp B — Dùng OntoKG (TransMTL + tri thức)

Cần Neo4j đang chạy. Làm **2 bước**, theo đúng thứ tự:

1. Trong `pipeline_config.py`: đặt `USE_ONTOKG = True`.

2. **BƯỚC 1 — xây OntoKG (chạy MỘT LẦN, đây là phần tốn thời gian):**

```bash
python run_ontokg.py
```
> Tạo `data/kge/entity_embeddings.pt` và nạp KG vào Neo4j.
> Tự **bỏ qua** module đã chạy xong (resume) → lần sau chạy lại rất nhanh.
> Muốn dựng lại từ đầu: xoá thư mục `data/` rồi chạy lại.

3. **BƯỚC 2 — train + test TransMTL (chạy lại bao nhiêu lần tuỳ ý):**

```bash
python run_transmtl.py
```
> Train TransMTL có gắn embedding tri thức (truy vấn Neo4j đã dựng ở Bước 1),
> rồi test. **Không phải dựng lại OntoKG** mỗi lần train.

**Ý tưởng tách bước:** Bước 1 (OntoKG) nặng nhưng chỉ cần làm 1 lần. Sau đó bạn
có thể thử nhiều cấu hình TransMTL khác nhau (sửa `conf.py`) và chạy lại Bước 2
nhiều lần mà tái dùng KG đã có.
=======
## 6. Chi tiết các file MODEL / DATA / UTIL (Phần 2 — TransMTL)

### 6.1 `transmtl/model.py` — `class TransformerMTL`
Kiến trúc Transformer encoder–decoder đa nhiệm + tích hợp OntoKG.
- **Khối con:** `PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `EncoderLayer/Encoder`, `DecoderLayer/Decoder_Sum`, `CopyGate`, `MMoE`.
- **Khởi tạo quan trọng:** `emb_matrix` (FastText), `word2idx/idx2word`, `num_key_labels=5` (BIOES), `use_mmoe`, `use_copy=True`, `use_ontokg`, `kg_in_dim=768`, `kg_num_relations=9`. `cls_idx=<sos>`, `sep_idx=<eos>`.
- **Đầu ra (heads):**
  - Tóm tắt: `Decoder_Sum` → `summary_logits`; nếu copy bật → `summary_log_probs` (pointer‑generator qua `CopyGate` + `_apply_copy`).
  - Từ khóa: `final_key_proj` (d_model→5) → emissions → **CRF** (`crf_decoder`, thư viện torchcrf): train trả `key_nll`, infer trả `key_decoded`.
- **`_apply_ontokg_fusion(enc_out_shared, kg_batch, device)`** — chỉ chạy khi `use_ontokg` và có `kg_batch`: `encode_kg_batch` → `GatedFusion`. Nếu không → trả nguyên `enc_out_shared` (baseline).
- **`forward(inp, tar, labels, task="both", training, kg_batch=None)`** → dict gồm `summary_logits`/`summary_log_probs`, `key_nll`/`key_decoded`, (gates nếu MMoE).
- **Sinh chuỗi:** `greedy_decode_batch(inp, max_len, kg_batch)` và `beam_search_generate_batch(inp, max_len, beam_size, len_penalty, n_gram_block, kg_batch)` (auto‑regressive, có copy, n‑gram blocking).

### 6.2 `transmtl/train.py`
- **`train_model(...)`** (chữ ký dài, gọi từ main/run): tạo loader (`get_loaders`), nạp FastText, dựng `TransformerMTL`, khởi tạo **`OntoKGBridge`**, optimizer **theo nhóm tham số** (summary/shared/crf, lr nhân hệ số), `weight_logits` (trọng số đa nhiệm học được, optimizer riêng), scheduler `ReduceLROnPlateau`, PCGrad. Tham số OntoKG: `use_ontokg, entity_emb_path, entity_idx_path, neo4j_uri, neo4j_pass`.
- **`train_one_epoch(...)`** / **`validate(...)`**: unpack batch **8‑tuple** (có `article_ids`); `kg_batch = bridge.build_kg_batch(article_ids)`; gọi model với `kg_batch`. Validate tính ROUGE qua greedy decode.

### 6.3 `transmtl/tester.py` — `run_test(...)`
Tạo loader, dựng model (khớp `use_ontokg`), nạp checkpoint (`load_checkpoint_state`, `strict=False`), **bridge OntoKG**, lặp test → beam/greedy sinh tóm tắt → **ROUGE‑1/2/L** (rouge_score) + map subword→word → **keyphrase P/R/F1** (`evaluate_keyphrase_lists`). Lưu `<ckpt>_test_results.txt`, trả dict kết quả.

### 6.4 `transmtl/data.py`
- **`MultiTaskDataset`**: đọc CSV, BPE tokenize (vocab_subword), sinh nhãn **BIOES** cho từ khóa (ánh xạ keyword→span token), lưu **`article_ids`** (cột `article_id`, fallback `row_xxxxx`).
- **`CollateCPU`**: pad batch → trả **8 phần tử**: `src, summary_ids, attn, labels, raw_texts, token_maps, word_texts, article_ids`.
- **`get_loaders(data_path, len_in, len_out, num_workers, batch_size, val_ratio=0.2, test_ratio=0.2, seed=42, min_freq=3, vocab_size)`** → train/val/test loader + vocab + word2idx/idx2word + `ds` (chứa `.tokenizer`). **Tự chia 60/20/20** từ CSV truyền vào (seed cố định ⇒ test held‑out tái lập được).

### 6.5 `transmtl/preprocessing.py`
`seed_everything`, `load_fasttext_bin_embeddings(word2idx, bin_path, d_model, pad_idx)` (**KHÔNG còn synonym**), BPE tokenizer (`ensure_tokenizer_for_csv`), ánh xạ keyword→span (`find_token_span_for_keyword`), `subword_labels_to_word_labels`, `ids_to_text`, `convert_tags_to_keyphrases`.

### 6.6 `transmtl/losses.py`
`compute_summary_loss_from_logits` (CE + label smoothing), `compute_summary_loss_from_logprobs` (NLL cho copy), `compute_key_loss_from_raw`, `compute_entropy_regularizer` (MMoE gate), **`combine_task_losses(loss_sum, key_nll, weight_logits, temp)`** (trọng số đa nhiệm học được), `ensure_decoder_sos`, `load_checkpoint_state`.

### 6.7 `transmtl/evaluation.py`
`evaluate_keyphrase_lists` (P/R/F1 theo tập cụm từ, chuẩn hóa + unique), `evaluate_summaries` (ROUGE).

### 6.8 `transmtl/fusion.py` (cầu nối model↔KG)
- **`GraphEncoder(in_dim=768, d_model, num_relations=9, num_bases=4, dropout)`**: `Linear(768→d) → RGCNConv×2 → GATv2Conv`. Cần `torch_geometric`; nếu thiếu → fallback MLP. `forward(x, edge_index, edge_type) → (N, d_model)`.
- **`GatedFusion(d_model, num_heads, dropout)`**: cross‑attention (query=H_tok, key/value=E_kg) + **cổng sigmoid** `gate*attn_out` + residual + LayerNorm → `H'_tok`.
- **`encode_kg_batch(graph_encoder, kg_batch, d_model, device)`**: mã hóa list subgraph (mỗi sample 1 subgraph hoặc None) → tensor padded `E_kg (B,N_max,d)` + `padding_mask`.

### 6.9 `transmtl/bridge.py` — `class OntoKGBridge`
`__init__(uri, user, password, d_model, enabled)`; nếu `enabled` → tạo `Neo4jRetriever` (module9, dim=768). **`build_kg_batch(article_ids)`** → list subgraph torch (hoặc None nếu tắt/không có). `close()`.
>>>>>>> theirs

---

## 5. (Tuỳ chọn) Chạy tất cả trong 1 lệnh: `main.py`

Nếu muốn chạy nguyên pipeline một phát (split → ontokg → train → test):

```bash
python main.py --use-ontokg     # chạy hết, CÓ OntoKG
python main.py --no-ontokg      # chạy hết, baseline
python main.py                  # theo USE_ONTOKG trong pipeline_config.py
```

### Giải thích các flag (đều là tuỳ chọn — không truyền thì lấy từ config)

| Flag | Ý nghĩa | Mặc định |
|------|---------|----------|
| `--stage {all,split,ontokg,train,test}` | Chỉ chạy 1 giai đoạn | `all` |
| `--use-ontokg` / `--no-ontokg` | Bật / tắt OntoKG cho lần chạy này | theo `USE_ONTOKG` |
| `--skip-existing` | Bỏ qua bước đã có output (resume) | tắt |
| `--save-path ĐƯỜNG_DẪN` | Nơi lưu checkpoint | `Results_Score/BestModel.pt` |
| `--neo4j-pass MẬT_KHẨU` | Mật khẩu Neo4j | theo config |

Ví dụ chỉ chạy lại phần test:
```bash
python main.py --stage test --use-ontokg
```

> `run_ontokg.py` ≈ `main.py --stage ontokg --use-ontokg --skip-existing`
> `run_transmtl.py` ≈ `main.py --stage train` + `main.py --stage test`
> Hai file ngắn này có sẵn để bạn **khỏi phải nhớ flag**.

---

## 6. Kết quả nằm ở đâu?

| Bước | Output |
|------|--------|
| split | `data_split/{train,val,test,trainval}.csv` |
| ontokg | `data/kge/entity_embeddings.pt`, KG trong Neo4j |
| train | `Results_Score/BestModel.pt` (checkpoint tốt nhất) |
| test | log ROUGE-1/2/L + Keyphrase P/R/F1, file `Results_Score/BestModel_test_results.txt` |

> OntoKG được xây trên **trainval**; `get_loaders` tự tách held-out test bằng
> cùng seed=42 nên `test` đánh giá đúng phần dữ liệu model chưa từng train.
> Muốn test trên file `test.csv` riêng: đổi `TEST_DATA_CSV = TEST_CSV` trong
> `pipeline_config.py`.

---

## 7. OntoKG chạy chậm? Mẹo tăng tốc

Đã tối ưu sẵn: NER + embedding model chạy **GPU + fp16** (trước đây chạy CPU),
và chỉ tính POS cho `full_text`. Muốn nhanh hơn nữa:

- Chạy `run_ontokg.py` **một lần** rồi tái dùng (đã resume sẵn nhờ skip-existing).
- `OKG_DEVICE=cuda` (mặc định) — đảm bảo có GPU.
- Trong `OntoKG/module2_ner_concept.py`: đặt `use_phonlp=False` nếu không cần.
- Trong `OntoKG/module3_entity_linking.py`: đặt `use_wikidata=False` để bỏ truy
  vấn mạng Wikidata (nhanh hẳn, nhưng mất liên kết Wikidata).
- Trong `OntoKG/module7_kge_training.py`: giảm `n_epochs` (vd 200 → 100).

---

## 8. Lỗi thường gặp

- **`FileNotFoundError: trainval.csv`** → chưa chia dữ liệu. Chạy `run_transmtl.py`
  hoặc `run_ontokg.py` (cả hai tự chia), hoặc kiểm tra `RAW_DATA_CSV`.
- **Kết nối Neo4j thất bại** (khi `USE_ONTOKG=True`) → kiểm tra Neo4j đang chạy ở
  `NEO4J_URI` và `NEO4J_PASSWORD` đúng.
- **Out of memory khi train** → giảm `BATCH_SIZE` trong `conf.py`.
