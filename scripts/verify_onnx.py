#!/usr/bin/env python3
"""Check an exported recogniser ONNX against the shipped runtime contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def describe(path: Path) -> dict:
    m = onnx.load(str(path))
    inp = m.graph.input[0]
    out = m.graph.output[0]
    return {
        "input_name": inp.name,
        "input_dims": [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim],
        "output_dims": [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim],
        "opset": [o.version for o in m.opset_import],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    args = ap.parse_args()

    a, b = describe(args.model), describe(args.reference)
    print("new:", a)
    print("ref:", b)

    assert a["input_dims"][1:3] == [3, 48], f"bad input shape {a['input_dims']}"
    assert a["input_dims"][3] == b["input_dims"][3], "width must stay dynamic"
    assert a["output_dims"][-1] == b["output_dims"][-1], "class count changed -> dict mismatch"
    assert 11 in a["opset"], "expected opset 11"

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 48, 96).astype(np.float32) * 2 - 1
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    assert y.shape[0] == 1 and y.shape[2] == b["output_dims"][-1], y.shape
    print(f"inference ok: {y.shape}, argmax={int(y.argmax(-1).max())}")


if __name__ == "__main__":
    main()
