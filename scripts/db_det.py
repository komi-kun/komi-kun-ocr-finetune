"""Minimal DB (Differentiable Binarization) post-processing for the det ONNX.

Ported from PaddleOCR's `DBPostProcess` (Apache-2.0), with the same knobs the app
uses by default: thresh 0.15, box_thresh 0.25, unclip 1.40, limit_side_len 960.

Produces line quads in original-image coordinates, ready for `crop_line()`.
"""

from __future__ import annotations

import cv2
import numpy as np
import pyclipper


class DBDetector:
    def __init__(
        self,
        onnx_path,
        thresh: float = 0.15,
        box_thresh: float = 0.25,
        unclip_ratio: float = 1.40,
        max_candidates: int = 1000,
        min_size: int = 3,
        limit_side_len: int = 960,
    ):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.min_size = min_size
        self.limit_side_len = limit_side_len
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    # ---------------------------------------------------------------- pre/post
    def _preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        ratio = min(1.0, self.limit_side_len / max(h, w))
        rh, rw = int(round(h * ratio)), int(round(w * ratio))
        rh = max(32, int(round(rh / 32)) * 32)
        rw = max(32, int(round(rw / 32)) * 32)
        resized = cv2.resize(img, (rw, rh))
        x = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        x = (x - self.mean) / self.std
        x = x.transpose(2, 0, 1)[None]
        return x, (h / rh, w / rw)

    @staticmethod
    def _get_mini_boxes(contour):
        rect = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
        i1, i2, i3, i4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            i1, i4 = 0, 1
        else:
            i1, i4 = 1, 0
        if points[3][1] > points[2][1]:
            i2, i3 = 2, 3
        else:
            i2, i3 = 3, 2
        box = np.array([points[i1], points[i2], points[i3], points[i4]], np.float32)
        return box, min(rect[1])

    @staticmethod
    def _box_score(pred, box):
        h, w = pred.shape[:2]
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
        shifted = box.copy()
        shifted[:, 0] -= xmin
        shifted[:, 1] -= ymin
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
        return cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def _unclip(self, box):
        poly = pyclipper.PyclipperOffset()
        poly.AddPath([tuple(map(float, p)) for p in box], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        area = abs(cv2.contourArea(box))
        length = cv2.arcLength(box, True)
        distance = area * self.unclip_ratio / max(length, 1e-6)
        return poly.Execute(distance)

    def __call__(self, img: np.ndarray):
        x, (ratio_h, ratio_w) = self._preprocess(img)
        out = self.sess.run(None, {self.input_name: x})[0][0, 0]
        pred = 1.0 / (1.0 + np.exp(-out)) if out.min() < 0 or out.max() > 1 else out

        segmentation = (pred > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[: self.max_candidates]

        results = []
        for contour in contours:
            box, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            score = self._box_score(pred, box)
            if score < self.box_thresh:
                continue
            unclipped = self._unclip(box)
            if len(unclipped) == 0:
                continue
            box2, sside2 = self._get_mini_boxes(np.asarray(unclipped, np.float32).reshape(-1, 1, 2))
            if sside2 < self.min_size + 2:
                continue
            box2[:, 0] *= ratio_w
            box2[:, 1] *= ratio_h
            results.append((box2, float(score)))

        # simple top-to-bottom, right-to-left reading order (vertical manga)
        results.sort(key=lambda r: (round(r[0][:, 1].mean() / 24), -r[0][:, 0].mean()))
        return results


def crop_line(img: np.ndarray, quad: np.ndarray, pad: float = 0.06):
    """Perspective-crop a detector quad; rotate tall (vertical) lines 90 deg CCW."""
    quad = np.asarray(quad, np.float32)
    # order tl, tr, br, bl
    s = quad.sum(1)
    d = np.diff(quad, axis=1).ravel()
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(d)]
    bl = quad[np.argmax(d)]
    pts = np.array([tl, tr, br, bl], np.float32)

    w = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    h = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
    if w < 2 or h < 2:
        return None

    dw, dh = int(round(w)), int(round(h))
    dst = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (dw, dh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if dh > dw * 1.2:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if pad:
        ph, pw = int(crop.shape[0] * pad), int(crop.shape[1] * pad)
        crop = cv2.copyMakeBorder(crop, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
    return crop
