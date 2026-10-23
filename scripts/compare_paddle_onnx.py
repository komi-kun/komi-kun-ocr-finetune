#!/usr/bin/env python3
"""Compare the exported Paddle inference model against the ONNX export.

Paddle and ONNX Runtime should agree to ~1e-4 on the same input; a larger gap
means the export dropped/changed part of the network (e.g. the CTC head).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "PaddleOCR"))

from common import RecModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-dir", type=Path, default=ROOT / "output/inference_manga")
    ap.add_argument("--onnx", type=Path, default=ROOT / "models/ppocrv6_small_rec_manga.onnx")
    ap.add_argument("--dict", type=Path, default=ROOT / "models/ppocrv6_small_dict.txt")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/manifests/val.txt")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    import paddle
    from paddle import inference

    cfg = inference.Config(
        str(args.inference_dir / "inference.json"),
        str(args.inference_dir / "inference.pdiparams"),
    )
    cfg.disable_gpu()
    cfg.disable_mkldnn()
    predictor = inference.create_predictor(cfg)
    in_names = predictor.get_input_names()
    out_names = predictor.get_output_names()

    onnx = RecModel(onnx_path=args.onnx, dict_path=args.dict)

    rows = [l.split("\t") for l in args.labels.read_text(encoding="utf-8").splitlines() if l.strip()]
    diffs, agree = [], 0
    for p, text in rows[: args.n]:
        img = cv2.imread(p)
        if img is None:
            continue
        x = onnx.preprocess(img)  # same 48px/BGR/[-1,1] normalisation
        ph, pw = x.shape[2], x.shape[3]
        xp = x.astype(np.float32)
        handle = predictor.get_input_handle(in_names[0])
        handle.copy_from_cpu(xp)
        predictor.run()
        yp = predictor.get_output_handle(out_names[0]).copy_to_cpu()
        yo = onnx.sess.run(None, {onnx.input_name: xp})[0]
        d = float(np.abs(yp - yo).max())
        diffs.append(d)
        a, b = onnx.decode(yp), onnx.decode(yo)
        agree += int(a == b)
        print(f"  {Path(p).name:32s} maxdiff={d:.2e} paddle={a!r} onnx={b!r}")

    print(f"\nmax |diff| = {max(diffs):.2e}, identical decode on {agree}/{len(diffs)}")
    assert max(diffs) < 5e-3, "paddle and onnx disagree -> broken export"


if __name__ == "__main__":
    main()
