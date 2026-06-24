# main.py
"""
==========================================================================
 PIPELINE END-TO-END: TransMTL + OntoKG  —  CHẠY TẤT CẢ BẰNG 1 LỆNH
==========================================================================
Toàn bộ cấu hình (đường dẫn dữ liệu, Neo4j, bật/tắt OntoKG) nằm ở
`pipeline_config.py`; hyperparameter model nằm ở `conf.py`. Sửa 2 file đó
là đủ, KHÔNG cần chỉnh path trong từng module nữa.

4 giai đoạn:
  [A] split  : CSV gốc          -> data_split/{train,val,test,trainval}.csv
  [B] ontokg : module 1-7 (+8)  -> data/kge/entity_embeddings.pt + nạp Neo4j
  [C] train  : TransMTL trên trainval (truy vấn Neo4j qua module 9 nếu bật KG)
  [D] test   : đánh giá ROUGE / keyphrase trên tập held-out

Cách dùng:
  python main.py                       # chạy hết theo pipeline_config.py
  python main.py --use-ontokg          # bật OntoKG cho lần chạy này
  python main.py --no-ontokg           # ép baseline cho lần chạy này
  python main.py --stage train         # chỉ 1 giai đoạn: split|ontokg|train|test
  python main.py --skip-existing       # bỏ qua bước đã có output (resume)
==========================================================================
"""
import os
import sys
import time
import argparse
import subprocess
from types import SimpleNamespace

import conf as cfg
import pipeline_config as P
from split_dataset import split_dataset


# ──────────────────────────────────────────────────────────
# Tạo "args" mặc định từ pipeline_config (dùng cho run_ontokg.py / run_transmtl.py)
# ──────────────────────────────────────────────────────────
def make_args(stage="all", use_ontokg=None, skip_existing=False,
              data_csv=None, pretrained_vec=None, save_path=None, neo4j_pass=None,
              cfg_module=None):
    """Tạo 'args' từ một module config. Mặc định dùng pipeline_config, nhưng
    run_transmtl.py / run_ontokg.py có thể truyền config khác (vd baseline)
    qua cfg_module — config đó sẽ đi xuyên suốt mọi giai đoạn qua args.P."""
    Pc = cfg_module or P
    if use_ontokg is None:
        use_ontokg = Pc.USE_ONTOKG
    return SimpleNamespace(
        stage=stage,
        use_ontokg=use_ontokg,
        skip_existing=skip_existing,
        data_csv=data_csv or Pc.RAW_DATA_CSV,
        pretrained_vec=pretrained_vec or Pc.PRETRAINED_VEC,
        save_path=save_path or Pc.SAVE_PATH,
        neo4j_pass=neo4j_pass or Pc.NEO4J_PASSWORD,
        P=Pc,
    )


# ──────────────────────────────────────────────────────────
# Tiện ích chạy 1 module python con (Module 1-8) qua subprocess
# ──────────────────────────────────────────────────────────
def run_module(script: str, env: dict, output_path: str = None, skip_existing: bool = False):
    if skip_existing and output_path and os.path.exists(output_path):
        print(f"--- Bỏ qua {script} (đã có {output_path})")
        return
    print(f"\n>>> Chạy {script} ...")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        print(f"!!! {script} LỖI (returncode={result.returncode}). Dừng pipeline.")
        sys.exit(result.returncode)
    print(f"<<< {script} xong sau {time.time() - t0:.1f}s")


# ──────────────────────────────────────────────────────────
# [A] Chia dữ liệu
# ──────────────────────────────────────────────────────────
def stage_split(args):
    print("\n" + "=" * 64 + "\n[A] CHIA DỮ LIỆU TRAIN/VAL/TEST\n" + "=" * 64)
    if args.skip_existing and os.path.exists(args.P.TRAINVAL_CSV):
        print(f"--- Bỏ qua split (đã có {args.P.TRAINVAL_CSV})")
        return
    split_dataset(args.data_csv, args.P.SPLIT_DIR,
                  val_ratio=args.P.VAL_RATIO, test_ratio=args.P.TEST_RATIO, seed=args.P.SEED)


# ──────────────────────────────────────────────────────────
# [B] Xây OntoKG (Module 1-7, +8 nếu bật KG) trên trainval
# ──────────────────────────────────────────────────────────
def stage_ontokg(args):
    print("\n" + "=" * 64 + "\n[B] XÂY ONTOKG (Module 1-8) trên trainval\n" + "=" * 64)
    if not os.path.exists(args.P.TRAINVAL_CSV):
        print(f"!!! Chưa có {args.P.TRAINVAL_CSV}. Hãy chạy stage split trước.")
        sys.exit(1)

    env = args.P.ontokg_env()
    env["NEO4J_PASSWORD"] = args.neo4j_pass   # cho phép CLI override

    for script in args.P.ONTOKG_MODULES:
        run_module(script, env,
                   output_path=args.P.ONTOKG_MODULE_OUTPUTS.get(script),
                   skip_existing=args.skip_existing)

    backend = getattr(args.P, "ONTOKG_BACKEND", "local")
    if args.use_ontokg and backend == "neo4j":
        run_module("OntoKG/module8_neo4j_loader.py", env)   # nạp Neo4j
    elif args.use_ontokg:
        print("(Bỏ qua Module 8 — backend='local', train/test đọc thẳng file, "
              "KHÔNG cần Neo4j)")
    else:
        print("(Bỏ qua Module 8 — baseline không OntoKG, không nạp Neo4j)")


# ──────────────────────────────────────────────────────────
# Gom các tham số OntoKG truyền cho train/test
# ──────────────────────────────────────────────────────────
def _ontokg_kwargs(args):
    backend = getattr(args.P, "ONTOKG_BACKEND", "local")
    if not args.use_ontokg:
        return dict(use_ontokg=False, entity_emb_path=None, entity_idx_path=None,
                    neo4j_uri=None, neo4j_pass=None, ontokg_backend=backend)
    return dict(
        use_ontokg=True,
        entity_emb_path=args.P.ENTITY_EMB,
        entity_idx_path=args.P.ENTITY_IDX_JSON,
        neo4j_uri=args.P.NEO4J_URI,
        neo4j_pass=args.neo4j_pass,
        ontokg_backend=backend,
    )


# ──────────────────────────────────────────────────────────
# [C] Train TransMTL
# ──────────────────────────────────────────────────────────
def stage_train(args):
    tag = "(+OntoKG)" if args.use_ontokg else "(baseline)"
    print("\n" + "=" * 64 + f"\n[C] TRAIN TransMTL {tag}\n" + "=" * 64)
    from transmtl.train import train_model

    train_model(
        args.P.TRAIN_DATA_CSV, args.save_path, cfg.PAD_IDX, cfg.LABEL_SMOOTHING,
        args.pretrained_vec, cfg.NUM_LAYER, cfg.D_MODEL, cfg.NUM_HEADS, cfg.DFF,
        cfg.LEN_IN, cfg.LEN_OUT, cfg.DROPOUT, cfg.FREEZE_EMBEDDINGS,
        cfg.MMOE_NUM_EXPERTS, cfg.MMOE_EXPERT_HIDDEN, cfg.MMOE_GATE_HIDDEN,
        cfg.MMOE_DROPOUT, cfg.MMOE_USE_RESIDUAL, cfg.MMOE_GATE_TEMPERATURE,
        cfg.MMOE_RESIDUAL_SCALE, cfg.LR_BASE, cfg.WEIGHT_DECAY, cfg.NUM_EPOCHS,
        cfg.WARMUP_MMOE_EPOCHS, cfg.MMOE_ENTROPY_LAMBDA, cfg.CLIP_NORM,
        cfg.IGNORE_INDEX, cfg.NUM_WORKERS, cfg.BATCH_SIZE, cfg.USE_MMOE,
        cfg.DEVICE, cfg.SIZE_VOCAB,
        **_ontokg_kwargs(args),
    )


# ──────────────────────────────────────────────────────────
# [D] Test offline (batch)
# ──────────────────────────────────────────────────────────
def stage_test(args):
    print("\n" + "=" * 64 + "\n[D] TEST OFFLINE (batch)\n" + "=" * 64)
    from transmtl.tester import run_test

    if not os.path.exists(args.save_path):
        print(f"!!! Chưa có checkpoint {args.save_path}. Hãy train trước.")
        sys.exit(1)

    run_test(
        args.save_path, args.P.TEST_DATA_CSV, cfg.LEN_IN, cfg.LEN_OUT, cfg.NUM_WORKERS,
        cfg.BATCH_SIZE, cfg.D_MODEL, cfg.PAD_IDX, args.pretrained_vec,
        cfg.NUM_LAYER, cfg.NUM_HEADS, cfg.DFF, cfg.DROPOUT, cfg.FREEZE_EMBEDDINGS,
        cfg.MMOE_NUM_EXPERTS, cfg.MMOE_EXPERT_HIDDEN, cfg.MMOE_GATE_HIDDEN,
        cfg.MMOE_DROPOUT, cfg.MMOE_USE_RESIDUAL, cfg.MMOE_GATE_TEMPERATURE,
        cfg.MMOE_RESIDUAL_SCALE, cfg.IGNORE_INDEX, cfg.DEVICE, cfg.USE_MMOE,
        cfg.SIZE_VOCAB,
        **_ontokg_kwargs(args),
    )


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="TransMTL + OntoKG pipeline (1 lệnh)")
    ap.add_argument("--stage", default="all",
                    choices=["all", "split", "ontokg", "train", "test"])
    ap.add_argument("--data-csv", dest="data_csv", default=P.RAW_DATA_CSV)
    ap.add_argument("--pretrained-vec", dest="pretrained_vec", default=P.PRETRAINED_VEC)
    ap.add_argument("--save-path", dest="save_path", default=P.SAVE_PATH)
    ap.add_argument("--neo4j-pass", dest="neo4j_pass", default=P.NEO4J_PASSWORD)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    help="Bỏ qua bước đã có output (resume).")
    # OntoKG: mặc định lấy từ pipeline_config.USE_ONTOKG, cho phép override.
    ap.add_argument("--use-ontokg", dest="use_ontokg", action="store_true", default=None)
    ap.add_argument("--no-ontokg", dest="use_ontokg", action="store_false")
    ap.set_defaults(use_ontokg=None)   # None => lấy từ pipeline_config.USE_ONTOKG
    args = ap.parse_args()
    args.P = P   # main.py luôn dùng pipeline_config mặc định

    if args.use_ontokg is None:
        args.use_ontokg = P.USE_ONTOKG

    P.ensure_dirs()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print(f"OntoKG: {'BẬT' if args.use_ontokg else 'TẮT (baseline)'} | "
          f"stage={args.stage} | save={args.save_path}")

    t_start = time.time()
    if args.stage in ("all", "split"):
        stage_split(args)
    if args.stage in ("all", "ontokg") and args.use_ontokg:
        stage_ontokg(args)
    elif args.stage == "ontokg" and not args.use_ontokg:
        print("(stage=ontokg nhưng OntoKG đang TẮT — bỏ qua. Dùng --use-ontokg để bật.)")
    if args.stage in ("all", "train"):
        stage_train(args)
    if args.stage in ("all", "test"):
        stage_test(args)

    print("\n" + "=" * 64 + f"\nPIPELINE HOÀN TẤT sau {time.time() - t_start:.1f}s\n" + "=" * 64)


if __name__ == "__main__":
    main()
