#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lanegen_evaluate.py  --  自動生成結果と正解 CVAT XML の突き合わせ

自動生成の良し悪しを目視ではなく数字で判断するための最小限の物差し。

出力する指標:

  recall     正解線上の点のうち、許容横距離内に予測がある割合
  precision  予測線上の点のうち、許容横距離内に正解がある割合
  RMSE       対応した点の横方向誤差
  分断数      1 本の正解線に対応した予測ポリラインの本数

最後の「分断数」が実務上は効く。精度指標には現れないのに、
1 本の線が 5 オブジェクトに割れていると作業者はマージ作業を強いられる。
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def load_polylines(path: Path) -> tuple[list[dict], tuple[int, int]]:
    root = ET.parse(path).getroot()
    image = root.find(".//image")
    size = (int(image.attrib.get("width", 0)),
            int(image.attrib.get("height", 0))) if image is not None else (0, 0)
    out = []
    for el in root.findall(".//image/polyline"):
        pts = np.array([[float(v) for v in pair.split(",")]
                        for pair in el.attrib["points"].split(";")])
        attrs = {a.attrib["name"]: (a.text or "")
                 for a in el.findall("attribute")}
        out.append({"label": el.attrib.get("label", ""), "px": pts,
                    "attributes": attrs})
    return out, size


def densify(px: np.ndarray, step_px: float) -> np.ndarray:
    if len(px) < 2:
        return px
    seg = np.linalg.norm(np.diff(px, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = float(cum[-1])
    if total <= 0:
        return px
    t = np.arange(0.0, total + step_px, step_px)
    return np.column_stack([np.interp(t, cum, px[:, 0]),
                            np.interp(t, cum, px[:, 1])])


def point_to_polyline_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """各点から折れ線までの最短距離。"""
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom[denom < 1e-12] = 1e-12
    best = np.full(len(pts), np.inf)
    for i in range(0, len(pts), 4096):
        chunk = pts[i:i + 4096]
        ap = chunk[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / denom[None, :], 0, 1)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        dist = np.linalg.norm(chunk[:, None, :] - proj, axis=2)
        best[i:i + 4096] = dist.min(axis=1)
    return best


def evaluate(pred_path: Path, gt_path: Path, resolution: float,
             tolerance: float, sample_step: float,
             labels: list[str]) -> dict:
    pred, size_p = load_polylines(pred_path)
    gt, size_g = load_polylines(gt_path)
    if size_p != size_g:
        print(f"[warn] 画像サイズが不一致: pred={size_p}, gt={size_g}")
    step_px = sample_step / resolution
    tol_px = tolerance / resolution

    report: dict = {"tolerance_m": tolerance, "sample_step_m": sample_step,
                    "labels": {}}
    for label in labels:
        P = [p for p in pred if p["label"] == label]
        G = [g for g in gt if g["label"] == label]
        entry = {"pred_polylines": len(P), "gt_polylines": len(G)}
        if not G and not P:
            report["labels"][label] = entry
            continue

        # --- recall と横方向誤差
        matched_err = []
        recalls = []
        fragments = []
        for g in G:
            gs = densify(g["px"], step_px)
            if not P:
                recalls.append(0.0)
                fragments.append(0)
                continue
            dists = np.full(len(gs), np.inf)
            owner = np.full(len(gs), -1)
            for k, p in enumerate(P):
                d = point_to_polyline_distance(gs, p["px"])
                better = d < dists
                dists[better] = d[better]
                owner[better] = k
            hit = dists <= tol_px
            recalls.append(float(hit.mean()))
            matched_err.extend((dists[hit] * resolution).tolist())
            fragments.append(int(len(np.unique(owner[hit]))))

        # --- precision
        prec = []
        for p in P:
            ps = densify(p["px"], step_px)
            if not G:
                prec.append(0.0)
                continue
            dists = np.full(len(ps), np.inf)
            for g in G:
                d = point_to_polyline_distance(ps, g["px"])
                dists = np.minimum(dists, d)
            prec.append(float((dists <= tol_px).mean()))

        entry.update({
            "recall": round(float(np.mean(recalls)), 4) if recalls else None,
            "precision": round(float(np.mean(prec)), 4) if prec else None,
            "lateral_rmse_m": round(float(np.sqrt(np.mean(
                np.square(matched_err)))), 4) if matched_err else None,
            "lateral_p95_m": round(float(np.percentile(
                matched_err, 95)), 4) if matched_err else None,
            "fragments_per_gt_line": fragments,
            "mean_fragments": round(float(np.mean(fragments)), 3)
                if fragments else None,
            "per_gt_recall": [round(r, 3) for r in recalls],
        })
        report["labels"][label] = entry
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CVAT XML 同士の突き合わせ評価")
    p.add_argument("predicted_xml")
    p.add_argument("ground_truth_xml")
    p.add_argument("--resolution", type=float, default=0.05,
                   help="XML のピクセル分解能 [m/px]")
    p.add_argument("--tolerance", type=float, default=0.20,
                   help="対応とみなす横距離 [m]")
    p.add_argument("--sample-step", type=float, default=1.0,
                   help="評価サンプリング間隔 [m]")
    p.add_argument("--labels", nargs="*",
                   default=["lane_line", "curb_boundary"])
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args(argv)

    report = evaluate(Path(a.predicted_xml), Path(a.ground_truth_xml),
                      a.resolution, a.tolerance, a.sample_step, a.labels)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if a.output:
        Path(a.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
