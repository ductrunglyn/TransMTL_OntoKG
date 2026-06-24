# run_transmtl.py
"""
==========================================================================
 BƯỚC 2 — TRAIN + TEST TransMTL (có thể chạy lại NHIỀU LẦN)
==========================================================================
Chạy:
    python run_transmtl.py

File này làm:
  1. Chia dữ liệu -> data_split/ (bỏ qua nếu đã có).
  2. Train TransMTL trên trainval. Nếu USE_ONTOKG = True trong
     pipeline_config.py, model sẽ truy vấn Neo4j (KG đã dựng ở BƯỚC 1) để
     gắn embedding tri thức vào quá trình học.
  3. Test và in ROUGE-1/2/L + Keyphrase P/R/F1, lưu *_test_results.txt.

HAI CHẾ ĐỘ (chọn bằng USE_ONTOKG trong pipeline_config.py):
  • USE_ONTOKG = True  -> TransMTL + OntoKG. PHẢI chạy `python run_ontokg.py`
                          trước (một lần) để có entity_embeddings.pt + Neo4j.
  • USE_ONTOKG = False -> baseline TransMTL thuần, KHÔNG cần Neo4j, KHÔNG cần
                          chạy run_ontokg.py.

Hyperparameter của model nằm ở conf.py; đường dẫn dữ liệu ở pipeline_config.py.
==========================================================================
"""
import time

import pipeline_config as P
from main import make_args, stage_split, stage_train, stage_test


if __name__ == "__main__":
    P.ensure_dirs()
    # cfg_module=P để TOÀN BỘ cấu hình (kể cả USE_ONTOKG, đường dẫn) lấy từ
    # đúng file bạn import ở trên — đổi import sang config khác là ăn ngay.
    args = make_args(skip_existing=True, cfg_module=P)

    mode = "TransMTL + OntoKG" if args.use_ontokg else "baseline TransMTL"
    print(f"Chế độ: {mode}  (đổi bằng USE_ONTOKG trong pipeline_config.py)")
    if args.use_ontokg:
        print("Lưu ý: cần đã chạy `python run_ontokg.py` trước để có KG + Neo4j.")

    t0 = time.time()
    stage_split(args)     # tạo data_split/ nếu chưa có
    stage_train(args)     # train TransMTL (+ OntoKG nếu bật)
    stage_test(args)      # đánh giá

    print("\n" + "=" * 64)
    print(f"BƯỚC 2 (TransMTL) HOÀN TẤT sau {time.time() - t0:.1f}s")
    print(f"Checkpoint: {args.save_path}")
    print("=" * 64)
