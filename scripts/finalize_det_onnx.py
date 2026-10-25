#!/usr/bin/env python3
"""Give the exported detector ONNX the shipped output shape metadata.

paddle2onnx cannot infer the channel dimension of the DB head's output, so it
writes [batch, ?, H, W] where the stock model has [batch, 1, H, W].  The runtime
tensor is [N, 1, H, W] either way; this only makes the graph's declared output
shape match the shipped contract (`models/ppocrv6_small_det.onnx`), so
`scripts/verify_det_onnx.py` can assert it.  No operator is touched.

    python scripts/finalize_det_onnx.py models/ppocrv6_small_det_manga.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    model = onnx.load(str(path))
    out = model.graph.output[0]
    dims = out.type.tensor_type.shape.dim
    assert len(dims) == 4, f"expected a 4-D detector output, got {len(dims)} dims"
    changed = False
    if dims[1].dim_value != 1:
        dims[1].dim_param = ""
        dims[1].dim_value = 1
        changed = True
    if changed:
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    print(f"finalize_det_onnx: output dim 1 = 1 ({'updated' if changed else 'already set'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
