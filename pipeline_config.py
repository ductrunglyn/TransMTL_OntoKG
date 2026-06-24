# pipeline_config.py
"""
==========================================================================
 CẤU HÌNH TRUNG TÂM cho toàn bộ pipeline TransMTL + OntoKG.
==========================================================================
Triết lý: CHỈ cần sửa file NÀY (đường dẫn dữ liệu, Neo4j, bật/tắt OntoKG)
và conf.py (hyperparameter của model) là chạy được TOÀN BỘ pipeline bằng
ĐÚNG MỘT LỆNH:

    python main.py                 # chạy hết: split -> ontokg -> train -> test
    python main.py --stage train   # chỉ chạy 1 giai đoạn
    python main.py --skip-existing # bỏ qua bước đã có output (resume)

Mọi giá trị ở đây đều có thể override bằng biến môi trường cùng tên,
ví dụ:  RAW_DATA_CSV=/path/khac.csv python main.py
==========================================================================
"""
import os

import conf as _cfg


def _env(key, default):
    return os.environ.get(key, default)


# ───────────────────────── 1. DỮ LIỆU GỐC (SỬA Ở ĐÂY) ─────────────────────
# CSV gốc: cần các cột title, summary, content, publish_time, topic, cleaned_keywords
RAW_DATA_CSV = _env(
    "RAW_DATA_CSV",
    "/home/hoangtrung/hdtrungoi/CoKhanh/Data/tintuc_gen_final.csv",
)
# FastText .bin tiếng Việt (cc.vi.300.bin)
PRETRAINED_VEC = _env(
    "PRETRAINED_VEC",
    "/home/hoangtrung/hdtrungoi/CoKhanh/TransMTL_K-Fold/word_embedding_pretrain/cc.vi.300.bin",
)

# ───────────────────────── 2. THƯ MỤC LÀM VIỆC ────────────────────────────
SPLIT_DIR = _env("SPLIT_DIR", "./data_split")   # train/val/test/trainval.csv
DATA_DIR  = _env("OKG_DATA_DIR", "./data")      # toàn bộ artifact của OntoKG
SAVE_DIR  = _env("SAVE_DIR", "./Results_Score")
SAVE_PATH = _env("SAVE_PATH", os.path.join(SAVE_DIR, "BestModel.pt"))

# Tỷ lệ chia dữ liệu (đồng bộ với get_loaders: val=0.2, test=0.2, seed=42)
VAL_RATIO  = 0.2
TEST_RATIO = 0.2
SEED       = 42

# ───────────────────────── 3. ONTOKG ──────────────────────────────────────
# True  -> bật OntoKG: cần Neo4j đang chạy, sẽ chạy module1-8 để dựng KG.
# False -> baseline TransMTL thuần (KHÔNG cần Neo4j). Mặc định baseline.
# ════════════════ BẬT / TẮT ONTOKG — SỬA NGAY TẠI ĐÂY ════════════════
#   False = baseline TransMTL thuần  (KHÔNG cần Neo4j, KHÔNG cần run_ontokg.py)
#   True  = TransMTL + OntoKG         (cần Neo4j + đã chạy run_ontokg.py trước)
# Đây là công tắc thường: chỉ cần đổi True/False. Không phụ thuộc biến môi trường.
USE_ONTOKG = True

# Kịch bản đánh giá OntoKG khi test (chỉ áp dụng khi USE_ONTOKG=True):
#   "offline" = trích entity từ văn bản test -> link vào KG train+val (CHỈ ĐỌC),
#               KG KHÔNG được cập nhật. (Kịch bản 1)
#   "online"  = như offline + thêm entity/quan hệ MỚI từ văn bản test để bổ trợ
#               TransMTL (KG cập nhật động). (Kịch bản 2)
#   "static"  = truy vấn KG theo article_id (chỉ hợp lệ khi bài test nằm trong KG,
#               vd đánh giá transductive trên train/val).
TEST_MODE      = _env("TEST_MODE", "offline")

NEO4J_URI      = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = _env("NEO4J_DATABASE", "neo4j")
ONTOKG_BACKEND = _env("ONTOKG_BACKEND", "local")

# Thiết bị cho các module nặng (module3 linking, module7 KGE)
OKG_DEVICE = _env("OKG_DEVICE", _cfg.DEVICE)


# ───────────────────────── 4. ĐƯỜNG DẪN PHÁI SINH (không cần sửa) ──────────
def _p(*parts):
    return os.path.join(*parts)


TRAIN_CSV    = _p(SPLIT_DIR, "train.csv")
VAL_CSV      = _p(SPLIT_DIR, "val.csv")
TEST_CSV     = _p(SPLIT_DIR, "test.csv")
TRAINVAL_CSV = _p(SPLIT_DIR, "trainval.csv")

# CSV dùng cho train / val / test — CHIA MỘT LẦN DUY NHẤT (split_dataset.py).
#   - TRAIN/VAL: train.csv & val.csv (TransMTL train trên train, early-stopping trên val).
#   - TEST: test.csv (held-out THẬT, đánh giá trên TOÀN BỘ, KHÔNG re-split).
#   - TOKENIZER_CSV: corpus CỐ ĐỊNH để dựng BPE/vocab (trainval) -> vocab nhất quán
#     giữa train/val/test, tránh lệch vocab gây hỏng checkpoint.
#   OntoKG xây trên trainval (=train+val); test.csv có article_id riêng, KHÔNG nằm
#   trong KG -> đánh giá inductive sạch (không rò rỉ phía tri thức).
TRAIN_DATA_CSV = TRAIN_CSV
VAL_DATA_CSV   = VAL_CSV
TEST_DATA_CSV  = TEST_CSV
TOKENIZER_CSV  = TRAINVAL_CSV

# OntoKG artifacts (chuỗi module 1 -> 8)
M1_OUT          = _p(DATA_DIR, "preprocessed_articles.jsonl")
M2_OUT          = _p(DATA_DIR, "module2_ner_concept.jsonl")
M3_OUT          = _p(DATA_DIR, "module3_entity_linked.jsonl")
M4_TRIPLES      = _p(DATA_DIR, "module4_triples.jsonl")
KG_DIR          = _p(DATA_DIR, "kg")
KGE_DIR         = _p(DATA_DIR, "kge")
ONTOLOGY_DIR    = _p(DATA_DIR, "ontology")
ENTITY_INDEX    = _p(KG_DIR, "entity_index.pkl")
PYKEEN_TSV      = _p(KG_DIR, "pykeen_triples.tsv")
ENTITY_EMB      = _p(KGE_DIR, "entity_embeddings.pt")
ENTITY_IDX_JSON = _p(KGE_DIR, "entity_to_idx.json")

# Thứ tự chạy các module OntoKG (đường dẫn tương đối so với repo root)
ONTOKG_MODULES = [
    "OntoKG/module1_preprocess.py",
    "OntoKG/module2_ner_concept.py",
    "OntoKG/module3_entity_linking.py",
    "OntoKG/module4_relation_extraction.py",
    "OntoKG/module5_kg_construction.py",
    "OntoKG/module6_ontology_learning.py",
    "OntoKG/module7_kge_training.py",
]
# Output đại diện để --skip-existing biết module đã chạy xong hay chưa.
ONTOKG_MODULE_OUTPUTS = {
    "OntoKG/module1_preprocess.py":         M1_OUT,
    "OntoKG/module2_ner_concept.py":        M2_OUT,
    "OntoKG/module3_entity_linking.py":     M3_OUT,
    "OntoKG/module4_relation_extraction.py": M4_TRIPLES,
    "OntoKG/module5_kg_construction.py":    PYKEEN_TSV,
    "OntoKG/module6_ontology_learning.py":  _p(ONTOLOGY_DIR, "ontology_v1.1.json"),
    "OntoKG/module7_kge_training.py":       ENTITY_EMB,
}


def ensure_dirs():
    """Tạo sẵn mọi thư mục output cần thiết."""
    for d in (SPLIT_DIR, DATA_DIR, KG_DIR, KGE_DIR, ONTOLOGY_DIR,
              os.path.dirname(SAVE_PATH) or "."):
        os.makedirs(d, exist_ok=True)


def ontokg_env():
    """Biến môi trường truyền cho các module subprocess (1-8) để chúng đọc
    cùng một cấu hình đường dẫn / Neo4j thay vì path hardcode."""
    env = os.environ.copy()
    env["OKG_DATA_DIR"]   = DATA_DIR
    env["OKG_INPUT_CSV"]  = TRAINVAL_CSV
    env["OKG_DEVICE"]     = OKG_DEVICE
    env["NEO4J_URI"]      = NEO4J_URI
    env["NEO4J_USER"]     = NEO4J_USER
    env["NEO4J_PASSWORD"] = NEO4J_PASSWORD
    env["NEO4J_DATABASE"] = NEO4J_DATABASE
    return env
