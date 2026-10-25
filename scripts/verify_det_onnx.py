#!/usr/bin/env python3
"""Verify an exported detector ONNX against the runtime contract and Paddle.

Checks
  * one input  [N, 3, H, W] with dynamic batch and dynamic H/W,
  * one output [N, 1, H, W] (the single probability map the JNI shim expects,
    i.e. the training-only auxiliary heads are gone),
  * opset 14 (what the shipped stock detector uses),
  * numerical parity with Paddle inference on real Manga109-s pages through the
    *runtime* preprocessing (960 max side, ImageNet mean/std, /32 rounding,
    exactly what ``scripts/db_det.py`` / the app do): max |diff| of the raw map.

Run with the Paddle-enabled export venv (`spec.paddle` is imported here):
    .venv-export/bin/python scripts/verify_det_onnx.py \
        --model models/ppocrv6_small_det_manga.onnx \
        --paddle-dir output/ppocrv6_small_det_manga_paddle
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import paddle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from db_det import DBDetector  # noqa: E402
from build_det_dataset import book_split  # noqa: E402


def describe(path: Path) -> dict:
    m = onnx.load(str(path))
    inp, out = m.graph.input[0], m.graph.output[0]
    return {
        "input_name": inp.name,
        "input_dims": [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim],
        "output_name": out.name,
        "output_dims": [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim],
        "opset": [o.version for o in m.opset_import],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
        "inputs": len(m.graph.input),
        "outputs": len(m.graph.output),
    }


def test_pages(n: int) -> list[Path]:
    root = next(p.parent for p in (ROOT / "data/raw/Manga109-s").rglob("annotations")
                if p.is_dir() and (p.parent / "images").is_dir())
    out = []
    for book in sorted((root / "images").iterdir()):
        if not book.is_dir() or book_split(book.name) != "test":
            continue
        pages = sorted(book.glob("*.jpg"))
        if pages:
            out.append(pages[len(pages) // 2])
            if len(out) >= n:
                break
    return out


def paddle_predictor(paddle_dir: Path):
    model = paddle_dir / "inference.json"
    params = paddle_dir / "inference.pdiparams"
    cfg = paddle.inference.Config(str(model), str(params))
    # oneDNN cannot run this PIR graph's double-typed attributes; plain CPU kernels can.
    cfg.disable_mkldnn()
    try:
        cfg.disable_glog_info()
    except Exception:
        pass
    return paddle.inference.create_predictor(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--paddle-dir", type=Path, required=True,
                    help="export_model.py output dir (inference.json/pdiparams), unpatched")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--max-diff", type=float, default=2e-3,
                    help="max |logit| difference vs Paddle (CPU kernels differ; the masks are compared too)")
    ap.add_argument("--max-mask-disagreement", type=float, default=1e-4,
                    help="max fraction of pixels whose binarised map differs at the runtime threshold")
    args = ap.parse_args()

    info = describe(args.model)
    print("onnx:", {k: v for k, v in info.items() if k != "sha256"})
    assert info["inputs"] == 1 and info["outputs"] == 1, "expected a single in / single out graph"
    assert info["input_dims"][1] == 3, f"input must be NCHW, got {info['input_dims']}"
    for dim in info["input_dims"][2:]:
        assert isinstance(dim, str), f"static H/W {info['input_dims']} - runtime resizes freely"
    assert info["output_dims"][1] == 1, f"output must be [N,1,H,W], got {info['output_dims']}"
    assert 14 in info["opset"], f"expected opset 14, got {info['opset']}"

    ort_sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    predictor = paddle_predictor(args.paddle_dir)
    in_name = predictor.get_input_names()[0]
    out_name = predictor.get_output_names()[0]

    # DBDetector gives us the exact runtime preprocessing; it also supplies the
    # post-processing comparison below.
    det = DBDetector(str(args.model))
    worst = 0.0
    worst_mask = 0.0
    for page in test_pages(args.pages):
        img = cv2.imread(str(page))
        x, _ = det._preprocess(img)
        onnx_map = ort_sess.run(None, {ort_sess.get_inputs()[0].name: x})[0]
        handle = predictor.get_input_handle(in_name)
        handle.reshape(x.shape)
        handle.copy_from_cpu(x)
        predictor.run()
        paddle_map = predictor.get_output_handle(out_name).copy_to_cpu()
        diff = float(np.abs(onnx_map - paddle_map).max())
        worst = max(worst, diff)
        a = 1.0 / (1.0 + np.exp(-onnx_map[0, 0]))
        b = 1.0 / (1.0 + np.exp(-paddle_map[0, 0]))
        disagree = float(np.mean((a > det.thresh) != (b > det.thresh)))
        worst_mask = max(worst_mask, disagree)
        boxes = det(img)
        print(f"  {page.parent.name}/{page.name}: max|onnx-paddle| {diff:.2e}, "
              f"mask disagreement @{det.thresh} {100*disagree:.4f} %, "
              f"map {onnx_map.shape}, det boxes {len(boxes)}")

    print(f"worst map diff {worst:.2e} (limit {args.max_diff:.0e})")
    print(f"worst mask disagreement {100*worst_mask:.4f} % (limit {100*args.max_mask_disagreement:.4f} %)")
    assert worst <= args.max_diff, "ONNX does not reproduce Paddle inference"
    assert worst_mask <= args.max_mask_disagreement, "binarised maps differ from Paddle"
    print("inference ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
