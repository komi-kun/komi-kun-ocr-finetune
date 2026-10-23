"""Shared helpers for the PP-OCRv6 manga fine-tune workspace.

rec_infer(): mirrors the runtime recogniser (ONNX, 3x48xW, BGR, [-1,1], CTC greedy)
so that offline evaluation uses the same code path the app ships.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

MODEL_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "models"


def load_dict(path=None):
    path = path or MODEL_DIR / "ppocrv6_small_dict.txt"
    with open(path, "r", encoding="utf-8") as f:
        chars = [line.rstrip("\n").rstrip("\r") for line in f]
    # runtime convention: ['#'] + dict + [' ']
    return ["#"] + chars + [" "]


class RecModel:
    def __init__(self, onnx_path=None, dict_path=None, providers=None):
        onnx_path = onnx_path or MODEL_DIR / "ppocrv6_small_rec.onnx"
        self.chars = load_dict(dict_path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        providers = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, bgr, max_w=320):
        import cv2

        h, w = bgr.shape[:2]
        ratio = w / h
        resized_w = max(1, min(max_w, int(np.ceil(48 * ratio))))
        img = cv2.resize(bgr, (resized_w, 48), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[None]
        return img

    def decode(self, logits):
        idx = logits.argmax(-1)[0]
        out, prev = [], -1
        for i in idx:
            if i != prev and i != 0:
                out.append(int(i))
            prev = i
        return "".join(self.chars[i] if i < len(self.chars) else "" for i in out)

    def __call__(self, bgr):
        logits = self.sess.run(None, {self.input_name: self.preprocess(bgr)})[0]
        return self.decode(logits), float(np.exp(logits.max(-1)).mean())


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
