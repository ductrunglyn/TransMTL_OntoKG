# run_ontokg.py
"""
==========================================================================
 BƯỚC 1 — XÂY ONTOKG (chạy MỘT LẦN, đây là phần tốn thời gian nhất)
==========================================================================
Chạy:
    python run_ontokg.py

File này làm:
  1. Chia dữ liệu gốc -> data_split/ (bỏ qua nếu đã có).
  2. Chạy Module 1-7: tiền xử lý -> NER -> entity linking -> trích quan hệ
     -> dựng KG -> học ontology -> huấn luyện KGE (entity_embeddings.pt).
  3. Module 8: nạp toàn bộ KG vào Neo4j để TransMTL truy vấn.

Sau khi chạy xong file này, các embedding/đồ thị tri thức đã sẵn sàng. Bạn
chỉ cần chạy BƯỚC 2 (`python run_transmtl.py`) — có thể chạy lại nhiều lần
mà KHÔNG phải dựng lại OntoKG.

Mẹo: file này tự BỎ QUA các module đã có output (resume). Muốn dựng lại từ
đầu, hãy xoá thư mục `data/` rồi chạy lại.

Mọi cấu hình (đường dẫn dữ liệu, Neo4j...) nằm ở pipeline_config.py.
==========================================================================
"""
import time

import pipeline_config as P
from main import make_args, stage_split, stage_ontokg


if __name__ == "__main__":
    P.ensure_dirs()
    # Bước 1 luôn xây OntoKG đầy đủ (gồm Module 8 nạp Neo4j) => ép use_ontokg=True.
    # skip_existing=True để chạy lại sẽ tiếp tục từ module còn dở, không làm lại từ đầu.
    args = make_args(use_ontokg=True, skip_existing=True, cfg_module=P)

    t0 = time.time()
    stage_split(args)     # tạo data_split/ nếu chưa có
    stage_ontokg(args)    # Module 1-8

    print("\n" + "=" * 64)
    print(f"BƯỚC 1 (OntoKG) HOÀN TẤT sau {time.time() - t0:.1f}s")
    print("Tiếp theo, đặt USE_ONTOKG = True trong pipeline_config.py rồi chạy:")
    print("    python run_transmtl.py")
    print("=" * 64)
