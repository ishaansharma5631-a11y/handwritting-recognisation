"""Batch-recognise a folder of scans.

  python predict.py --images ../test --out predictions.csv
  # also save confident predictions as a pseudo-labelled training set (self-training):
  python predict.py --images ../test --out preds.csv --pseudo work/pseudo.npz --min-conf 0.9
"""
import argparse, os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from model import CRNN, fit_to_canvas, greedy_decode

torch.set_num_threads(4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "model.pt"))
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--pseudo", default="")
    p.add_argument("--min-conf", type=float, default=0.9)
    p.add_argument("--exclude", default="", help="CSV of hand-labelled files to keep out of the pseudo-labels")
    a = p.parse_args()
    model = CRNN(); model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True)); model.eval()
    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    X = np.zeros((len(files), 32, 256), np.uint8)
    texts, confs = [], []
    with torch.no_grad():
        for i in range(0, len(files), 256):
            xb = np.stack([fit_to_canvas(Image.open(os.path.join(a.images, f))) for f in files[i:i + 256]])
            X[i:i + len(xb)] = (xb * 255).astype(np.uint8)
            t, c = greedy_decode(model(torch.from_numpy(xb)[:, None]))
            texts += t; confs += c
            print(f"{i + len(xb)}/{len(files)}", end="\r", flush=True)
    df = pd.DataFrame({"FILENAME": files, "IDENTITY": texts, "CONFIDENCE": np.round(confs, 4)})
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(df)} rows, {np.mean(np.array(confs) >= a.min_conf):.1%} with confidence >= {a.min_conf}")
    if a.pseudo:
        keep = (np.array(confs) >= a.min_conf) & (df.IDENTITY.str.len() > 0).values
        if a.exclude:
            keep &= ~df.FILENAME.isin(set(pd.read_csv(a.exclude).FILENAME)).values
        np.savez(a.pseudo, x=X[keep], y=df.IDENTITY.values.astype(str)[keep], files=np.array(files)[keep])
        print("pseudo-labels:", keep.sum())
