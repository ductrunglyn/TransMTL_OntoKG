# split_dataset.py
"""
Buoc tien xu ly: chia CSV goc thanh 3 tap train/val/test.
Dam bao OntoKG xay tren train+val, test hold-out (theo methodology 3.3).

Dung CUNG seed va ty le voi get_loaders cu (random_split seed=42, val=0.2, test=0.2)
de ket qua nhat quan.

Input : tintuc_gen_final.csv
Output: data_split/train.csv, data_split/val.csv, data_split/test.csv
        data_split/trainval.csv  (gop train+val, dung de xay OntoKG)

Cach dung:
    python split_dataset.py
"""
import os
import argparse
import numpy as np
import pandas as pd


def split_dataset(
    csv_path: str,
    out_dir: str = "./data_split",
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"Tong so bai: {n}")

    # GAN article_id TOAN CUC ON DINH *truoc khi* chia, dua tren chi so hang goc.
    # Nho vay CUNG mot bai bao giu CUNG article_id trong train/val/test/trainval
    # va trong OntoKG (Neo4j/local). Day la dieu kien de:
    #   - OntoKG xay tren trainval (=train+val) co id khop voi luc train tren
    #     train.csv/val.csv,
    #   - test.csv co id RIENG, KHONG nam trong KG (danh gia inductive sach).
    if "article_id" not in df.columns:
        df.insert(0, "article_id", [f"art_{i:06d}" for i in range(n)])

    # Shuffle voi cung seed (tuong duong torch random_split manual_seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)

    # Ghi ra dia — article_id da co san (on dinh toan cuc), KHONG gan lai theo file.
    for name, d in [("train", df_train), ("val", df_val),
                    ("test", df_test), ("trainval", df_trainval)]:
        path = os.path.join(out_dir, f"{name}.csv")
        d.to_csv(path, index=False)
        print(f"  {name:9s}: {len(d):6d} bai -> {path}")

    print("\nLuu y: OntoKG (Module 1-7) xay tren trainval.csv (=train+val).")
    print("       test.csv giu article_id rieng, KHONG nam trong KG -> danh gia inductive.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/hoangtrung/hdtrungoi/CoKhanh/Data/tintuc_gen_final.csv")
    ap.add_argument("--out", default="./data_split")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split_dataset(args.csv, args.out, args.val_ratio, args.test_ratio, args.seed)
