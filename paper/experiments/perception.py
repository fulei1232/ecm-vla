"""Fruit perception for the paper experiments: HSV colour prior, SAM segmentation prompted by it (SAM-FDM),
automatic SAM masks filtered by colour, ripeness from mask colour, and RGB-D back-projection to 3D fruit centres.

HSV follows OpenCV (H in 0..179, S and V in 0..255). Thresholds are calibrated on training frames
(exp_sam_segmentation.py calibrate) and loaded from paper/results/hsv_thresholds.json.
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from common import ECM, OUT

SAM_DIR = ECM / "cache/sam"
sys.path.insert(0, str(SAM_DIR / "pkg"))
CHECKPOINTS = {"vit_b": "sam_vit_b_01ec64.pth", "vit_l": "sam_vit_l_0b3195.pth", "vit_h": "sam_vit_h_4b8939.pth"}
RIPE, UNRIPE = 1, 0


def hsv(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)


def class_masks(rgb, thr, relax=0):
    """{RIPE: bool mask, UNRIPE: bool mask} from the calibrated boxes; `relax` lowers the S/V minima for prompt
    recall, but S never below min(s_min, 40): grey and white pixels (robot, gripper) have hue 0, i.e. "red"."""
    h = hsv(rgb)
    H, S, V = h[..., 0].astype(int), h[..., 1].astype(int), h[..., 2].astype(int)
    r, u = thr["ripe"], thr["unripe"]
    s_floor = lambda s: s if s - relax < 40 else s - relax  # noqa: E731
    ripe = ((H <= r["h_hi"]) | (H >= r["h_wrap"])) & (S >= s_floor(r["s_min"])) & (V >= r["v_min"] - relax)
    unripe = (H >= u["h_lo"]) & (H <= u["h_hi"]) & (S >= s_floor(u["s_min"])) & (V >= u["v_min"] - relax)
    return {RIPE: ripe, UNRIPE: unripe}


def components(mask, min_area):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            m = lab == i
            ys, xs = np.nonzero(m)
            j = np.argmin((xs - cent[i][0]) ** 2 + (ys - cent[i][1]) ** 2)
            out.append({"mask": m, "box": np.array([x, y, x + w - 1, y + h - 1], float),
                        "point": np.array([xs[j], ys[j]], float), "area": int(area)})
    return out


def hsv_instances(rgb, thr, min_area):
    """Colour-threshold baseline: connected components of each class mask."""
    out = []
    for cls, m in class_masks(rgb, thr).items():
        for c in components(m, min_area):
            out.append({"mask": c["mask"], "cls": cls, "score": 1.0})
    return out


def mask_class(rgb, mask, thr):
    """(class by majority of fruit-coloured pixels, fraction of the mask that is fruit-coloured)."""
    cm = class_masks(rgb, thr)
    r = int((cm[RIPE] & mask).sum())
    u = int((cm[UNRIPE] & ~cm[RIPE] & mask).sum())
    return RIPE if r >= u else UNRIPE, (r + u) / max(int(mask.sum()), 1)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


def nms(instances, thresh=0.5, contained=0.6):
    """Mask NMS by score, then drop masks lying mostly inside a larger kept mask (SAM proposes parts of a fruit
    alongside the whole fruit)."""
    keep = []
    for inst in sorted(instances, key=lambda x: -x["score"]):
        if all(iou(inst["mask"], k["mask"]) < thresh for k in keep):
            keep.append(inst)
    out = []
    for inst in sorted(keep, key=lambda x: -int(x["mask"].sum())):
        area = max(int(inst["mask"].sum()), 1)
        if all(np.logical_and(inst["mask"], k["mask"]).sum() / area < contained for k in out):
            out.append(inst)
    return out


class SamFruit:
    def __init__(self, model="vit_b", device="cuda"):
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        self.torch = torch
        sam = sam_model_registry[model](checkpoint=str(SAM_DIR / CHECKPOINTS[model])).to(device).eval()
        self.sam, self.model, self.device = sam, model, device
        self.predictor = SamPredictor(sam)
        self._auto = None

    def set_image(self, rgb):
        self.predictor.set_image(np.ascontiguousarray(rgb))
        self.shape = rgb.shape[:2]

    def boxes(self, boxes, points=None):
        """(masks bool [N, H, W], predicted IoU [N]) for box prompts (+ one positive point each)."""
        torch = self.torch
        if not len(boxes):
            return np.zeros((0, *self.shape), bool), np.zeros(0)
        tb = self.predictor.transform.apply_boxes_torch(torch.as_tensor(np.asarray(boxes), dtype=torch.float, device=self.device), self.shape)
        tp, tl = None, None
        if points is not None:
            tp = self.predictor.transform.apply_coords_torch(torch.as_tensor(np.asarray(points)[:, None, :], dtype=torch.float, device=self.device), self.shape)
            tl = torch.ones(tp.shape[:2], device=self.device)
        with torch.no_grad():
            masks, scores, _ = self.predictor.predict_torch(tp, tl, boxes=tb, multimask_output=True)
        best = scores.argmax(1)
        idx = torch.arange(len(best), device=self.device)
        return masks[idx, best].cpu().numpy(), scores[idx, best].cpu().numpy()

    def fdm(self, rgb, thr, min_area, pad=4, point=True):
        """SAM-FDM: colour components prompt SAM with a padded box (and a centre point); ripeness from the colour
        of the SAM mask; duplicates removed by mask NMS."""
        cands = []
        for cm in class_masks(rgb, thr).values():
            cands += components(cm, min_area)
        if not cands:
            return []
        h, w = self.shape
        boxes = np.array([[max(c["box"][0] - pad, 0), max(c["box"][1] - pad, 0),
                           min(c["box"][2] + pad, w - 1), min(c["box"][3] + pad, h - 1)] for c in cands])
        masks, scores = self.boxes(boxes, np.array([c["point"] for c in cands]) if point else None)
        out = []
        for m, s, c in zip(masks, scores, cands):
            # SAM sometimes returns the whole branch or background for a weak prompt: keep compact masks
            if m.sum() < min_area or m.sum() > 6 * max(c["area"], min_area) + 200:
                continue
            cls, purity = mask_class(rgb, m, thr)
            if purity < 0.3:
                continue
            out.append({"mask": m, "cls": cls, "score": float(s)})
        return nms(out)

    def auto(self, rgb, thr, min_area, max_frac=0.08):
        """Automatic SAM masks kept when their pixels are mostly fruit-coloured and the region is compact;
        max_frac: largest plausible fruit area as a fraction of the image (camera dependent)."""
        if self._auto is None:
            from segment_anything import SamAutomaticMaskGenerator
            self._auto = SamAutomaticMaskGenerator(self.sam, points_per_side=16, pred_iou_thresh=0.8,
                                                   stability_score_thresh=0.85, min_mask_region_area=min_area)
        out = []
        for r in self._auto.generate(np.ascontiguousarray(rgb)):
            m = r["segmentation"]
            area = int(m.sum())
            if area < min_area or area > max_frac * m.size:
                continue
            x, y, bw, bh = r["bbox"]
            if area / max(bw * bh, 1) < 0.45:  # a disc fills pi/4 of its box; branches and leaves do not
                continue
            cls, purity = mask_class(rgb, m, thr)
            if purity < 0.5:
                continue
            out.append({"mask": m, "cls": cls, "score": float(r["predicted_iou"])})
        return nms(out)


def load_thresholds():
    return json.loads((OUT / "hsv_thresholds.json").read_text())


def load_prompt():
    """SAM-FDM prompt settings chosen on training frames (exp_sam_segmentation.py tune)."""
    path = OUT / "fdm_prompt.json"
    return json.loads(path.read_text())["best"] if path.exists() else {"pad": 4, "point": True}


def photometric(rgb, kind):
    """Illumination changes of an orchard scene: dark (overcast / dusk), bright (direct sun), warm (low sun)."""
    x = rgb.astype(np.float32) / 255.0
    if kind == "dark":
        x = 0.55 * x ** 1.2
    elif kind == "bright":
        x = 1.35 * x ** 0.85
    elif kind == "warm":
        x = x * np.array([1.12, 1.0, 0.80], np.float32)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def backproject(mask, depth_mm, K, extrinsic):
    """World points (m) of the mask pixels with valid depth; extrinsic is world -> camera (OpenCV, 3x4)."""
    ys, xs = np.nonzero(mask & (depth_mm > 0))
    if not len(xs):
        return np.zeros((0, 3))
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    cam = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], 1)
    R, t = extrinsic[:, :3].astype(np.float64), extrinsic[:, 3].astype(np.float64)
    return (cam - t) @ R


def camera_center(extrinsic):
    R, t = extrinsic[:, :3].astype(np.float64), extrinsic[:, 3].astype(np.float64)
    return -R.T @ t


def fruit_center(mask, depth_mm, K, extrinsic):
    """Sphere-centre estimate: median visible surface point pushed back along the viewing ray by the radius
    estimated from the mask area and depth. Returns (centre, radius) or None."""
    pts = backproject(mask, depth_mm, K, extrinsic)
    if len(pts) < 5:
        return None
    surf = np.median(pts, 0)
    cc = camera_center(extrinsic)
    z = np.median(depth_mm[mask & (depth_mm > 0)]) / 1000.0
    radius = float(np.clip(np.sqrt(mask.sum() / np.pi) * z / K[0, 0], 0.01, 0.06))
    ray = (surf - cc) / np.linalg.norm(surf - cc)
    return surf + radius * ray, radius


def project(p, K, extrinsic):
    c = extrinsic[:, :3].astype(np.float64) @ np.asarray(p, np.float64) + extrinsic[:, 3]
    if c[2] <= 1e-6:
        return None, c[2]
    uv = K @ c
    return uv[:2] / uv[2], c[2]
