"""Preprocess labelled scans into a 32x256 uint8 array file for train.py.

  python prep_real.py --images train/train --csv written_name_train_v2.csv --out work/real --val 2500
"""
import argparse, os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from model import ALPHABET, MAX_LEN, fit_to_canvas


def _load(path):
    return (fit_to_canvas(Image.open(path)) * 255).astype(np.uint8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True, help="prefix: writes <out>_train.npz and <out>_val.npz")
    p.add_argument("--val", type=int, default=2500)
    a = p.parse_args()
    df = pd.read_csv(a.csv).dropna()
    df = df[df.FILENAME.isin(set(os.listdir(a.images)))]
    df["IDENTITY"] = df.IDENTITY.astype(str).str.upper().str.strip()
    df = df[(df.IDENTITY != "UNREADABLE") & (df.IDENTITY != "EMPTY")]
    df = df[df.IDENTITY.map(lambda s: 0 < len(s) <= MAX_LEN and all(c in ALPHABET for c in s))]
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    with Pool(4) as pool:
        X = np.stack(pool.map(_load, [os.path.join(a.images, f) for f in df.FILENAME], chunksize=256))
    y, f = df.IDENTITY.values.astype(str), df.FILENAME.values.astype(str)
    np.savez(a.out + "_val.npz", x=X[:a.val], y=y[:a.val], files=f[:a.val])
    np.savez(a.out + "_train.npz", x=X[a.val:], y=y[a.val:], files=f[a.val:])
    print("usable", len(df), "train", len(df) - a.val, "val", a.val)
