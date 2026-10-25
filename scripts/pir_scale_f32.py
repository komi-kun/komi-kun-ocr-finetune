#!/usr/bin/env python3
"""Rewrite `scale` array attributes in an exported Paddle PIR inference graph.

paddle2onnx 2.1.0 (the newest release) reads the `interpolate.scale` attribute as
`std::vector<float>` but Paddle 3.3/3.4 store it as an array of *doubles*
(`0.a_f64`), so the converter aborts with

    (Unimplemented) the 0th elementwise MUST be ir::FloatAttribute

This rewrites exactly those attribute values to float32 in a copy of
`inference.json` (same numeric values), which the PIR parser accepts.  It touches
nothing else and is only used to feed the ONNX converter; the Paddle weights and
the unmodified export stay byte-identical for the parity check.

    python scripts/pir_scale_f32.py output/det_inference/inference.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def patch(obj, attr: str = "scale") -> int:
    n = 0
    if isinstance(obj, dict):
        if obj.get("N") == attr:
            at = obj.get("AT")
            if isinstance(at, dict) and at.get("#") == "0.a_array":
                for item in at.get("D", []):
                    if isinstance(item, dict) and item.get("#") in ("0.a_f64", "0.a_f32"):
                        if item["#"] != "0.a_f32":
                            item["#"] = "0.a_f32"
                            n += 1
        for value in obj.values():
            n += patch(value, attr)
    elif isinstance(obj, list):
        for value in obj:
            n += patch(value, attr)
    return n


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    doc = json.loads(path.read_text())
    n = patch(doc)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"pir_scale_f32: rewrote {n} f64 values under {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
