"""Shared model + preprocessing (CRNN adapted from hwr.py for CPU: CNN + 2-layer BiLSTM, CTC loss)."""
import string

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

H, W = 32, 256          # test scans are ~300x35 px, so 32 px of height keeps all the detail
ALPHABET = string.ascii_uppercase + "-' "
C2I = {c: i + 1 for i, c in enumerate(ALPHABET)}   # 0 = CTC blank
MAX_LEN = 24


def fit_to_canvas(img):
    """PIL image -> float32 HxW in [0,1], ink = 1. Grayscale, stretch contrast, keep aspect, pad right."""
    img = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    w, h = img.size
    scale = min(H / h, W / w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (W, H), 255)
    canvas.paste(img, (0, (H - nh) // 2))
    return 1.0 - np.asarray(canvas, dtype=np.float32) / 255.0


def block(i, o, pool):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.MaxPool2d(pool) if pool else nn.Identity())


class CRNN(nn.Module):
    def __init__(self, n_classes=len(ALPHABET) + 1, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            block(1, 32, (2, 2)), block(32, 64, (2, 2)), block(64, 96, None),
            block(96, 96, (2, 1)), block(96, 128, None), block(128, 128, (2, 1)),
            nn.Dropout2d(0.2))
        self.rnn = nn.LSTM(128 * 2, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):                       # B,1,32,256 -> B,64,C log-probs
        f = self.cnn(x)
        b, c, h, w = f.shape
        out, _ = self.rnn(f.permute(0, 3, 1, 2).reshape(b, w, c * h))
        return self.fc(out).log_softmax(-1)


def greedy_decode(logp):
    """Returns (texts, confidences). Confidence = product of per-frame max probabilities."""
    probs, best = logp.exp().max(-1)
    texts, confs = [], []
    for seq, p in zip(best.cpu().numpy(), probs.cpu().numpy()):
        s, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                s.append(ALPHABET[k - 1])
            prev = k
        texts.append("".join(s).strip())
        confs.append(float(np.exp(np.log(np.clip(p, 1e-9, 1)).sum())))
    return texts, confs


def char_errors(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]
