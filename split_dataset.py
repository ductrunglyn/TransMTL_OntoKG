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

    # Them article_id de Module 1-9 + Neo4j tra cuu duoc
    for name, d in [("train", df_train), ("val", df_val),
                    ("test", df_test), ("trainval", df_trainval)]:
        d = d.copy()
        d.insert(0, "article_id",
                 [f"{name}_{i:06d}" for i in range(len(d))])
        path = os.path.join(out_dir, f"{name}.csv")
        d.to_csv(path, index=False)
        print(f"  {name:9s}: {len(d):6d} bai -> {path}")

    print("\nLuu y: OntoKG (Module 1-7) xay tren trainval.csv")
    print("       Test streaming (Module 10) dung test.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/hoangtrung/hdtrungoi/CoKhanh/Data/tintuc_gen_final.csv")
    ap.add_argument("--out", default="./data_split")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split_dataset(args.csv, args.out, args.val_ratio, args.test_ratio, args.seed)
