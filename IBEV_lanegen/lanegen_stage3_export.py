#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lanegen_stage3_export.py  --  Stage 3: 分類・出力 (軽い)

Stage1/2 の成果物から、信頼度・lane_number を付けて CVAT XML を書く。

方針:

* RGB は **主判定に使わない**。線種と色の整合チェック (confidence) だけに使う。
  夜間は色が壊れ、雨天は反射強度が壊れるので、どちらか一方に依存しない。
* RGB の健全性を自己診断し、壊れている条件では自動的に重みを 0 にする。
  静かに劣化させず、降りたことを meta に残す。
* 低信頼のラインは捨てない。**本体 XML と review XML の 2 本**に分ける。
  プリラベルでは「描き直す」より「消す」方が速いので、recall を優先する。
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

import lanegen_core as core

KIND_LANE, KIND_CURB, KIND_HATCH = 0, 1, 2


# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IBEV 1パス自動ライン生成 Stage3: 信頼度付与とCVAT出力",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("stage1_npz")
    p.add_argument("stage2_npz")
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--stem", default=None, help="出力ファイル名の共通部分")

    g = p.add_argument_group("XML 座標系")
    g.add_argument("--image-meta-json", default=None,
                   help="アノテーターで表示する BEV 画像のメタJSON")
    g.add_argument("--xml-image-name", default=None,
                   help="XML に書く image name")
    g.add_argument("--output-point-spacing", type=float, default=5.0,
                   help="平滑化後ポリラインの最大頂点間隔 [m]")
    g.add_argument("--simplify-tolerance", type=float, default=0.08,
                   help="平滑化なし・fallback時のDouglas-Peucker許容誤差 [m]")

    g = p.add_argument_group("ポリライン平滑化・頂点削減")
    g.add_argument("--geometry-smoothing",
                   choices=["auto", "line", "spline", "none"],
                   default="auto",
                   help="autoは直線判定後、必要な線だけロバスト曲線近似")
    g.add_argument("--straight-line-tolerance", type=float, default=0.05,
                   help="直線近似とみなすロバスト横誤差 [m]")
    g.add_argument("--spline-smoothing-tolerance", type=float, default=0.04,
                   help="ロバストスプライン平滑化の基準誤差 [m]")
    g.add_argument("--curve-approximation-tolerance", type=float, default=0.05,
                   help="スプラインをポリライン化する最大横偏差の目安 [m]")
    g.add_argument("--smoothing-robust-iterations", type=int, default=3,
                   help="外れ値を抑えるロバスト再重み付け回数")
    g.add_argument("--curb-straight-line-tolerance", type=float, default=0.08,
                   help="縁石を直線近似とみなすロバスト横誤差 [m]")
    g.add_argument("--curb-spline-smoothing-tolerance", type=float, default=0.06,
                   help="縁石スプライン平滑化の基準誤差 [m]")
    g.add_argument("--curb-curve-approximation-tolerance", type=float, default=0.08,
                   help="縁石曲線をポリライン化する横偏差 [m]")

    g = p.add_argument_group("信頼度")
    g.add_argument("--confidence-threshold", type=float, default=0.55,
                   help="本体XMLに入れる下限。下回ると review XML へ")
    g.add_argument("--review-gap-threshold", type=float, default=30.0,
                   help="最大補間ギャップがこれ以上ならreview [m]")
    g.add_argument("--review-interpolated-ratio", type=float, default=0.70,
                   help="補間率がこれ以上ならreview")
    g.add_argument("--review-unknown-gap-threshold", type=float, default=30.0,
                   help="センサー無効の連続区間がこれ以上ならreview [m]")
    g.add_argument("--review-frenet-ambiguity-ratio", type=float, default=0.03,
                   help="Frenet曖昧セル比率がこれ以上ならreview")
    g.add_argument("--review-frenet-ambiguity-run", type=float, default=1.0,
                   help="Frenet曖昧区間がこの長さ以上ならreview [m]")
    g.add_argument("--full-strength-sigma", type=float, default=6.0,
                   help="この強度で強度スコアを満点とする")
    g.add_argument("--full-length", type=float, default=30.0,
                   help="この長さで長さスコアを満点とする")
    g.add_argument("--lane-spacing-min", type=float, default=2.75,
                   help="構造整合とみなす車線間隔の下限 [m]")
    g.add_argument("--lane-spacing-max", type=float, default=3.90,
                   help="構造整合とみなす車線間隔の上限 [m]")
    g.add_argument("--curb-step-nominal", type=float, default=0.15)
    g.add_argument("--curb-step-sigma", type=float, default=0.07)

    g = p.add_argument_group("RGB 自己診断")
    g.add_argument("--rgb-white-ratio-min", type=float, default=0.004,
                   help="この比率を下回ると色情報は信用しない")
    g.add_argument("--rgb-white-ratio-max", type=float, default=0.25,
                   help="この比率を上回ると色情報は信用しない")
    g.add_argument("--rgb-white-luminance", type=float, default=120.0)
    g.add_argument("--rgb-white-saturation", type=float, default=0.35)
    g.add_argument("--rgb-yellow-hue-min", type=float, default=30.0)
    g.add_argument("--rgb-yellow-hue-max", type=float, default=75.0)
    g.add_argument("--rgb-yellow-saturation", type=float, default=0.30)
    g.add_argument("--rgb-weight", type=float, default=0.25,
                   help="信頼度全体に占める色の重み (健全時)")

    g = p.add_argument_group("出力内容")
    g.add_argument("--lane-number-style", choices=["ego", "left", "none"],
                   default="ego", help="lane_number の採番方式")
    g.add_argument("--curb-edge", choices=["ridge", "gutter", "top"],
                   default="gutter",
                   help="縁石ポリラインをどのエッジに置くか")
    g.add_argument("--curb-edge-offset", type=float, default=0.10,
                   help="ridge から gutter/top へのオフセット [m]")
    g.add_argument("--emit-hatch", action="store_true",
                   help="導流帯候補も review XML に出す")
    g.add_argument("--no-preview", action="store_true")
    return p.parse_args(argv)


# =============================================================================
# 読み込み
# =============================================================================

def load_tracks(stage2_npz: Path) -> tuple[list[dict], dict]:
    z = np.load(stage2_npz)
    meta = json.loads(Path(stage2_npz).with_suffix(".meta.json")
                      .read_text(encoding="utf-8"))
    offsets = z["t_offset"]
    tracks = []
    for i in range(len(z["t_kind"])):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        tracks.append({
            "kind": int(z["t_kind"][i]),
            "start": int(z["t_start"][i]),
            "stop": int(z["t_stop"][i]),
            "d": z["flat_d"][lo:hi],
            "observed": z["flat_observed"][lo:hi],
            "strength": z["flat_strength"][lo:hi],
            "valid_support": (z["flat_valid_support"][lo:hi]
                              if "flat_valid_support" in z.files
                              else np.ones(hi - lo, dtype=bool)),
            "meta": meta["tracks"][i],
        })
    return tracks, meta


# =============================================================================
# RGB
# =============================================================================

def rgb_health(npz, a: argparse.Namespace) -> dict:
    """色情報が使える条件かを自己診断する。"""
    if "rgb_occupied" not in npz:
        return {"available": False, "healthy": False, "weight": 0.0,
                "reason": "RGB証拠なし"}
    occ = npz["rgb_occupied"]
    lum = npz["rgb_luminance"]
    sat = npz["rgb_saturation"]
    if not occ.any():
        return {"available": True, "healthy": False, "weight": 0.0,
                "reason": "有効セルなし"}
    white = occ & (lum >= a.rgb_white_luminance) & (sat <= a.rgb_white_saturation)
    ratio = float(white.sum()) / float(occ.sum())
    healthy = a.rgb_white_ratio_min <= ratio <= a.rgb_white_ratio_max
    reason = "健全" if healthy else (
        "白画素が少なすぎる (夜間・低露光の可能性)"
        if ratio < a.rgb_white_ratio_min
        else "白画素が多すぎる (白飛び・積雪・濡れ路面の可能性)")
    return {
        "available": True, "healthy": healthy,
        "weight": a.rgb_weight if healthy else 0.0,
        "white_ratio": ratio,
        "valid_ratio": float(occ.mean()),
        "luminance_p50": float(np.percentile(lum[occ], 50)),
        "reason": reason,
    }


def sample_rgb_along(track: dict, npz, d_axis: np.ndarray, d_res: float,
                     a: argparse.Namespace) -> dict:
    """トラックに沿って色特徴を集計する。"""
    if "rgb_occupied" not in npz:
        return {}
    occ = npz["rgb_occupied"]
    lum = npz["rgb_luminance"]
    sat = npz["rgb_saturation"]
    rgb = npz["rgb_mean"]
    nd = occ.shape[0]
    rows = np.clip(np.round((track["d"] - d_axis[0]) / d_res).astype(int),
                   0, nd - 1)
    cols = np.arange(track["start"], track["stop"] + 1)
    valid = occ[rows, cols]
    if valid.sum() < 4:
        return {"rgb_valid_coverage": float(valid.mean())}
    L = lum[rows, cols]
    S = sat[rows, cols]
    is_white = valid & (L >= a.rgb_white_luminance) & (S <= a.rgb_white_saturation)
    px = rgb[rows, cols].astype(np.float32)
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    denom = np.maximum(mx - mn, 1e-6)
    hue = np.zeros(len(px))
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    m_r = (mx == r)
    m_g = (mx == g) & ~m_r
    m_b = ~m_r & ~m_g
    hue[m_r] = 60.0 * (((g - b)[m_r] / denom[m_r]) % 6.0)
    hue[m_g] = 60.0 * ((b - r)[m_g] / denom[m_g] + 2.0)
    hue[m_b] = 60.0 * ((r - g)[m_b] / denom[m_b] + 4.0)
    is_yellow = (valid & (S >= a.rgb_yellow_saturation)
                 & (hue >= a.rgb_yellow_hue_min)
                 & (hue <= a.rgb_yellow_hue_max))
    paint = is_white | is_yellow
    return {
        "rgb_valid_coverage": float(valid.mean()),
        "paint_coverage": float(paint.mean()),
        "white_coverage": float(is_white.mean()),
        "yellow_coverage": float(is_yellow.mean()),
        "median_luminance": float(np.median(L[valid])),
        "median_saturation": float(np.median(S[valid])),
    }


# =============================================================================
# 信頼度
# =============================================================================

def _gauss(x: float, mu: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - mu) / max(sigma, 1e-6)) ** 2))


def neighbour_spacing(tracks: list[dict], index: int) -> float:
    """同じ s 区間を持つ最寄り車線線との横距離。"""
    me = tracks[index]
    best = float("inf")
    for j, other in enumerate(tracks):
        if j == index or other["kind"] != KIND_LANE:
            continue
        lo = max(me["start"], other["start"])
        hi = min(me["stop"], other["stop"])
        if hi - lo < 40:
            continue
        da = me["d"][lo - me["start"]:hi - me["start"] + 1]
        db = other["d"][lo - other["start"]:hi - other["start"] + 1]
        gap = abs(float(np.median(db - da)))
        best = min(best, gap)
    return best


def lane_confidence(track: dict, spacing: float, rgb: dict, health: dict,
                    s_res: float, a: argparse.Namespace) -> tuple[float, dict]:
    """線の長さではなく、自動生成根拠の強さを評価する。"""
    m = track["meta"]
    obs = track["observed"]
    strength = float(track["strength"][obs].mean()) if obs.any() else 0.0
    observed_support = float(m.get("observed_on_valid_ratio",
                                   m.get("coverage", 0.0)))
    interpolated_ratio = float(m.get("interpolated_ratio", 1.0 - obs.mean()))

    if m["line_type"] == "Dashed_line":
        support_score = float(np.clip(m.get("on_run_count", 0) / 4.0, 0, 1))
    else:
        support_score = float(np.clip(observed_support / 0.75, 0, 1))

    parts = {
        "strength": float(np.clip(strength / a.full_strength_sigma, 0, 1)),
        "observation_support": support_score,
        "line_type": 1.0 if m["line_type"] in ("Solid_line", "Dashed_line")
                     else 0.35,
        "structure": 1.0 if a.lane_spacing_min <= spacing <= a.lane_spacing_max
                     else (0.75 if np.isfinite(spacing) else 0.6),
        "continuity": float(np.clip(1.0 - 0.5 * interpolated_ratio, 0, 1)),
        "frenet_unambiguity": float(np.clip(
            1.0 - float(m.get("frenet_ambiguity_ratio", 0.0)) / 0.10, 0, 1)),
    }
    weights = {"strength": 0.30, "observation_support": 0.25,
               "line_type": 0.15, "structure": 0.15, "continuity": 0.10,
               "frenet_unambiguity": 0.05}

    if health.get("weight", 0.0) > 0 and "paint_coverage" in rgb:
        cov = rgb["paint_coverage"]
        if m["line_type"] == "Solid_line":
            c_rgb = float(np.clip(cov / 0.60, 0, 1))
        elif m["line_type"] == "Dashed_line":
            c_rgb = float(1.0 - np.clip(abs(cov - 0.45) / 0.45, 0, 1))
        else:
            c_rgb = float(np.clip(cov / 0.40, 0, 1))
        parts["rgb_consistency"] = c_rgb
        weights["rgb_consistency"] = health["weight"]

    total_w = sum(weights.values())
    score = sum(parts[k] * weights[k] for k in parts) / total_w
    return float(score), parts


def curb_confidence(track: dict, a: argparse.Namespace) -> tuple[float, dict]:
    m = track["meta"]
    observed_support = float(m.get("observed_on_valid_ratio",
                                   m.get("coverage", 0.0)))
    parts = {
        "step_height": _gauss(m.get("median_step_m", 0.0),
                              a.curb_step_nominal, a.curb_step_sigma),
        "sign_consistency": float(m.get("sign_consistency", 0.0)),
        "observation_support": float(np.clip(observed_support / 0.6, 0, 1)),
        "sensor_validity": float(np.clip(m.get("sensor_valid_ratio", 1.0), 0, 1)),
        "frenet_unambiguity": float(np.clip(
            1.0 - float(m.get("frenet_ambiguity_ratio", 0.0)) / 0.10, 0, 1)),
    }
    weights = {"step_height": 0.33, "sign_consistency": 0.28,
               "observation_support": 0.24, "sensor_validity": 0.10,
               "frenet_unambiguity": 0.05}
    score = sum(parts[k] * weights[k] for k in parts) / sum(weights.values())
    return float(score), parts


# =============================================================================
# lane_number
# =============================================================================

def assign_lane_numbers(tracks: list[dict], style: str) -> None:
    lanes = [t for t in tracks if t["kind"] == KIND_LANE]
    if style == "none" or not lanes:
        for t in lanes:
            t["lane_number"] = ""
        return
    for t in lanes:
        t["_votes"] = []
    ns_lo = min(t["start"] for t in lanes)
    ns_hi = max(t["stop"] for t in lanes)
    for s in range(ns_lo, ns_hi + 1, 4):
        present = [(float(t["d"][s - t["start"]]), t)
                   for t in lanes if t["start"] <= s <= t["stop"]]
        if not present:
            continue
        present.sort(key=lambda kv: kv[0])
        if style == "left":
            for n, (_, t) in enumerate(present, start=1):
                t["_votes"].append(str(n))
        else:  # ego 基準
            left = [kv for kv in present if kv[0] < 0][::-1]
            right = [kv for kv in present if kv[0] >= 0]
            for n, (_, t) in enumerate(left, start=1):
                t["_votes"].append(f"L{n}")
            for n, (_, t) in enumerate(right, start=1):
                t["_votes"].append(f"R{n}")
    for t in lanes:
        votes = t.pop("_votes")
        if votes:
            vals, counts = np.unique(np.asarray(votes), return_counts=True)
            t["lane_number"] = str(vals[int(np.argmax(counts))])
        else:
            t["lane_number"] = ""


# =============================================================================
# World 変換
# =============================================================================

def track_to_world(track: dict, npz, d_axis: np.ndarray, d_res: float,
                   a: argparse.Namespace, d_offset: float = 0.0) -> np.ndarray:
    """FrenetトラックをGlobal XYZへ変換する。

    Zの復元元は走行軌跡中心ではなく、簡略化前のLane自身の密なXYを使う。
    """
    traj_xy = npz["traj_xy"]
    traj_n = npz["traj_normal"]
    traj_z = npz["traj_z"]

    dense_sel = np.arange(track["start"], track["stop"] + 1)
    d_dense = track["d"] + d_offset
    lane_xy_dense = traj_xy[dense_sel] + d_dense[:, None] * traj_n[dense_sel]

    if "z_surface" in npz:
        zs = npz["z_surface"]
        occ = npz["z_occupied"]
        rows = np.clip(np.round((d_dense - d_axis[0]) / d_res).astype(int),
                       0, zs.shape[0] - 1)
        z_dense = zs[rows, dense_sel]
        bad = ~occ[rows, dense_sel]
        z_dense = np.where(bad, traj_z[dense_sel], z_dense)
    else:
        z_dense = traj_z[dense_sel]

    if track["kind"] == KIND_CURB:
        straight_tolerance = a.curb_straight_line_tolerance
        spline_tolerance = a.curb_spline_smoothing_tolerance
        approximation_tolerance = a.curb_curve_approximation_tolerance
    else:
        straight_tolerance = a.straight_line_tolerance
        spline_tolerance = a.spline_smoothing_tolerance
        approximation_tolerance = a.curve_approximation_tolerance
    xy_out, fit_meta = core.smooth_polyline_xy(
        lane_xy_dense,
        mode=a.geometry_smoothing,
        straight_line_tolerance=straight_tolerance,
        spline_smoothing_tolerance=spline_tolerance,
        approximation_tolerance=approximation_tolerance,
        max_vertex_spacing=a.output_point_spacing,
        simplify_tolerance=a.simplify_tolerance,
        robust_iterations=a.smoothing_robust_iterations,
    )
    track["geometry_fit"] = fit_meta
    try:
        from scipy.spatial import cKDTree
        _, nearest = cKDTree(lane_xy_dense.astype(np.float64)).query(
            xy_out.astype(np.float64), k=1)
    except Exception:
        nearest = np.array([
            int(np.argmin(np.sum((lane_xy_dense - p) ** 2, axis=1)))
            for p in xy_out], dtype=np.int64)
    zz = z_dense[np.asarray(nearest, dtype=np.int64)]
    return np.column_stack([xy_out, zz])


# =============================================================================
# プレビュー
# =============================================================================

def save_preview(path: Path, npz, tracks: list[dict], d_axis: np.ndarray,
                 d_res: float) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[warn] Pillow が無いためプレビューをスキップ")
        return
    sigma = npz["i_sigma"]
    nd, ns = sigma.shape
    bg = np.clip(sigma / 3.5, 0, 1) ** 0.7
    img = (bg.T * 255).astype(np.uint8)          # (ns, nd) = 縦 s
    rgb = np.stack([img] * 3, axis=2)
    if "z_level_diff" in npz:
        step = np.clip(np.abs(npz["z_level_diff"]).T / 0.3, 0, 1)
        rgb[..., 2] = np.maximum(rgb[..., 2], (step * 160).astype(np.uint8))
    pil = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil)
    colors = {"lane_solid": (0, 220, 255), "lane_dashed": (255, 170, 0),
              "lane_virtual": (140, 140, 255), "curb": (220, 60, 255),
              "hatch": (255, 60, 60), "review": (120, 255, 120)}
    for t in tracks:
        pts = []
        for i, s in enumerate(range(t["start"], t["stop"] + 1, 2)):
            col = (t["d"][s - t["start"]] - d_axis[0]) / d_res
            pts.append((float(col), float(s)))
        if len(pts) < 2:
            continue
        if t["kind"] == KIND_CURB:
            c = colors["curb"]
        elif t["kind"] == KIND_HATCH:
            c = colors["hatch"]
        elif t.get("review_required"):
            c = colors["review"]
        elif t["meta"]["line_type"] == "Dashed_line":
            c = colors["lane_dashed"]
        elif t["meta"]["line_type"] == "Virtual_lane":
            c = colors["lane_virtual"]
        else:
            c = colors["lane_solid"]
        draw.line(pts, fill=c, width=2)
    pil.save(path)
    print(f"[out] preview -> {path}")


# =============================================================================
# Production V1 出力ヘルパー
# =============================================================================

def _mask_ranges(mask: np.ndarray, start: int, s_res: float) -> list[list[float]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.r_[0, change, len(mask)]
    out = []
    for i in range(len(bounds) - 1):
        b, e = int(bounds[i]), int(bounds[i + 1])
        if mask[b]:
            out.append([round((start + b) * s_res, 3),
                        round((start + e - 1) * s_res, 3)])
    return out


def _stable_auto_id(label: str, source: str, xyz: np.ndarray) -> str:
    rounded = np.round(np.asarray(xyz, dtype=np.float64), 3)
    h = hashlib.sha1()
    h.update(label.encode("utf-8"))
    h.update(source.encode("utf-8"))
    h.update(rounded.tobytes())
    return f"auto_{h.hexdigest()[:16]}"


def _review_reasons(t: dict, score: float, a: argparse.Namespace) -> list[str]:
    m = t["meta"]
    reasons: list[str] = []
    if score < a.confidence_threshold:
        reasons.append("low_detection_support")
    if t["kind"] == KIND_HATCH:
        reasons.append("hatch_candidate")
    if m.get("source") in ("structure_infill", "structure_virtual"):
        reasons.append("structure_inference")
    if m.get("line_type") in ("Uncertain", "Virtual_lane"):
        reasons.append("uncertain_line_type")
    if float(m.get("max_gap_m", 0.0)) >= a.review_gap_threshold:
        reasons.append("long_interpolated_gap")
    if float(m.get("interpolated_ratio", 0.0)) >= a.review_interpolated_ratio:
        reasons.append("high_interpolated_ratio")
    if float(m.get("max_unknown_gap_m", 0.0)) >= a.review_unknown_gap_threshold:
        reasons.append("long_sensor_invalid_gap")
    if (float(m.get("frenet_ambiguity_ratio", 0.0))
            >= a.review_frenet_ambiguity_ratio
            or float(m.get("max_frenet_ambiguity_run_m", 0.0))
            >= a.review_frenet_ambiguity_run):
        reasons.append("frenet_branch_ambiguity")
    return reasons


def _review_priority(reasons: list[str], score: float, meta: dict) -> float:
    """0=後回し、1=最優先。正誤判定ではなくレビュー順序にだけ使う。"""
    value = max(0.0, 1.0 - float(score))
    weights = {
        "long_interpolated_gap": 0.30,
        "long_sensor_invalid_gap": 0.25,
        "structure_inference": 0.25,
        "uncertain_line_type": 0.20,
        "high_interpolated_ratio": 0.15,
        "hatch_candidate": 0.10,
        "low_detection_support": 0.10,
        "frenet_branch_ambiguity": 0.35,
    }
    value += sum(weights.get(r, 0.0) for r in reasons)
    # 非常に長い補間はさらに先に確認する。
    value += min(float(meta.get("max_gap_m", 0.0)) / 200.0, 0.20)
    return float(np.clip(value, 0.0, 1.0))


def _line_type_for_annotator(value: str) -> str:
    return {"Solid_line": "solid", "Dashed_line": "dash",
            "Virtual_lane": "virtual"}.get(value, "solid")


def _annotator_record(index: int, t: dict, xyz: np.ndarray, label: str,
                      source: str, s_res: float) -> dict:
    m = t["meta"]
    review_reasons = t.get("review_reasons", [])
    observed = np.asarray(t["observed"], dtype=bool)
    valid = np.asarray(t.get("valid_support", np.ones(len(observed), bool)),
                       dtype=bool)
    rec = {
        "id": f"lane_{index:04d}",
        "auto_track_id": _stable_auto_id(label, source, xyz),
        "class": label,
        "geometry_type": "polyline",
        "points": [[round(float(p[0]), 6), round(float(p[1]), 6),
                    round(float(p[2]), 6)] for p in xyz],
        "buffer_m": 0.10,
        "coordinate_system": "map_world",
        "visibility": "visible",
        "quality": "review_required" if review_reasons else "ok",
        "source": "auto",
        "is_interpolated": "true" if m.get("interpolated_ratio", 0.0) > 0 else "false",
        "interpolation_reason": "unknown" if m.get("interpolated_ratio", 0.0) > 0 else "",
        "note": (f"support={t['confidence']:.3f}; "
                 f"observed={m.get('observed_ratio', 0.0):.3f}; "
                 f"max_gap_m={m.get('max_gap_m', 0.0):.2f}"),
        "generation": {
            "production_version": core.PRODUCTION_VERSION,
            "detection_support": round(float(t["confidence"]), 6),
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "review_priority": round(float(t.get("review_priority", 0.0)), 6),
            "observed_ratio": float(m.get("observed_ratio", 0.0)),
            "observed_on_valid_ratio": float(m.get("observed_on_valid_ratio",
                                                    m.get("coverage", 0.0))),
            "sensor_valid_ratio": float(m.get("sensor_valid_ratio", 1.0)),
            "interpolated_ratio": float(m.get("interpolated_ratio", 0.0)),
            "max_gap_m": float(m.get("max_gap_m", 0.0)),
            "max_valid_gap_m": float(m.get("max_valid_gap_m", 0.0)),
            "max_unknown_gap_m": float(m.get("max_unknown_gap_m", 0.0)),
            "frenet_ambiguity_ratio": float(
                m.get("frenet_ambiguity_ratio", 0.0)),
            "max_frenet_ambiguity_run_m": float(
                m.get("max_frenet_ambiguity_run_m", 0.0)),
            "observed_s_ranges_m": _mask_ranges(observed, t["start"], s_res),
            "sensor_invalid_s_ranges_m": _mask_ranges(~valid, t["start"], s_res),
            "source": source,
            "geometry_fit": dict(t.get("geometry_fit", {})),
        },
    }
    if label == "lane_line":
        rec.update({
            "line_type": _line_type_for_annotator(m.get("line_type", "")),
            "line_color": t.get("output_color", "white"),
            "line_count": "single",
            "lane_number": t.get("lane_number", ""),
            "lane_id_n": "", "lane_id_x": "", "lane_id_y": "",
            "lane_id_manual": False,
        })
    else:
        rec["boundary_id"] = t.get("boundary_id", "")
    return rec


# =============================================================================

def run(a: argparse.Namespace) -> dict:
    t0 = time.time()
    npz1 = np.load(Path(a.stage1_npz))
    meta1 = json.loads(Path(a.stage1_npz).with_suffix(".meta.json")
                       .read_text(encoding="utf-8"))
    tracks, meta2 = load_tracks(Path(a.stage2_npz))
    grid = meta2["grid"]
    d_res, s_res, d_min = (grid["d_resolution"], grid["s_resolution"],
                           grid["d_min"])
    d_axis = d_min + (np.arange(grid["nd"]) + 0.5) * d_res

    out_dir = Path(a.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = a.stem or Path(a.stage1_npz).stem.replace("_stage1", "")

    # ---------------- RGB 自己診断
    health = rgb_health(npz1, a)
    print(f"[info] RGB: {health['reason']} "
          f"(weight={health.get('weight', 0.0):.2f})")

    # ---------------- 特徴と信頼度
    assign_lane_numbers(tracks, a.lane_number_style)
    for i, t in enumerate(tracks):
        rgb_feat = sample_rgb_along(t, npz1, d_axis, d_res, a)
        t["rgb"] = rgb_feat
        if t["kind"] == KIND_CURB:
            score, parts = curb_confidence(t, a)
        else:
            spacing = neighbour_spacing(tracks, i)
            t["neighbour_spacing_m"] = spacing
            score, parts = lane_confidence(t, spacing, rgb_feat, health,
                                           s_res, a)
        t["confidence"] = score
        t["confidence_parts"] = parts
        t["review_reasons"] = _review_reasons(t, score, a)
        t["review_priority"] = _review_priority(
            t["review_reasons"], score, t["meta"])
        t["review_required"] = bool(t["review_reasons"])

    # ---------------- 画像座標系
    if a.image_meta_json:
        data = json.loads(Path(a.image_meta_json).read_text(encoding="utf-8"))
        image_meta = core.extract_image_meta(data)
        image_meta["source"] = str(a.image_meta_json)
    else:
        bbox = meta1["world_bbox"]
        res = float(meta1["parameters"].get("d_resolution", 0.05))
        image_meta = {
            "origin_x": bbox["x_min"], "origin_y": bbox["y_min"],
            "resolution": res,
            "width_px": int(np.ceil((bbox["x_max"] - bbox["x_min"]) / res)) + 1,
            "height_px": int(np.ceil((bbox["y_max"] - bbox["y_min"]) / res)) + 1,
            "source": "stage1 world bbox",
        }
    image_name = a.xml_image_name or f"{stem}.png"

    # ---------------- 形状生成
    main_shapes: list[core.CvatShape] = []
    review_shapes: list[core.CvatShape] = []
    curb_id = 0
    rows = []
    annotator_all = []
    annotator_main = []
    annotator_review = []
    output_index = 0
    for t in tracks:
        m = t["meta"]
        offset = 0.0
        if t["kind"] == KIND_CURB and a.curb_edge != "ridge":
            sign = float(m.get("step_sign", 1.0))
            direction = -1.0 if a.curb_edge == "gutter" else 1.0
            offset = direction * sign * a.curb_edge_offset
        xyz = track_to_world(t, npz1, d_axis, d_res, a, offset)

        if t["kind"] == KIND_CURB:
            curb_id += 1
            label = "curb_boundary"
            t["boundary_id"] = f"C{curb_id:03d}"
            attrs = {"boundary_id": t["boundary_id"]}
            source = "auto_z_step"
        elif t["kind"] == KIND_HATCH:
            label = "lane_line"
            attrs = {"line_type": "Solid_line", "line_color": "white",
                     "lane_number": ""}
            source = "auto_hatch_candidate"
        else:
            label = "lane_line"
            color = "white"
            r = t["rgb"]
            if health.get("weight", 0.0) > 0 and r.get("yellow_coverage", 0) > \
                    max(0.25, r.get("white_coverage", 0.0)):
                color = "yellow"
            attrs = {"line_type": m["line_type"] if m["line_type"] else "Solid_line",
                     "line_color": color,
                     "lane_number": t.get("lane_number", "")}
            source = m["source"]
            t["output_color"] = color

        attrs["_source"] = source
        attrs["_confidence"] = f"{t['confidence']:.3f}"
        attrs["_review_required"] = "true" if t["review_required"] else "false"
        attrs["_review_reasons"] = ",".join(t["review_reasons"])
        attrs["_review_priority"] = f"{t.get('review_priority', 0.0):.3f}"
        attrs["_observed_ratio"] = f"{m.get('observed_ratio', 0.0):.3f}"
        attrs["_interpolated_ratio"] = f"{m.get('interpolated_ratio', 0.0):.3f}"
        attrs["_max_gap_m"] = f"{m.get('max_gap_m', 0.0):.2f}"
        attrs["_frenet_ambiguity_ratio"] = (
            f"{m.get('frenet_ambiguity_ratio', 0.0):.4f}")
        attrs["_max_frenet_ambiguity_run_m"] = (
            f"{m.get('max_frenet_ambiguity_run_m', 0.0):.2f}")
        attrs["_lane_id_manual"] = "false"
        fit = t.get("geometry_fit", {})
        attrs["_geometry_fit"] = str(fit.get("mode", "none"))
        attrs["_vertices_before"] = str(fit.get("vertices_before", len(xyz)))
        attrs["_vertices_after"] = str(fit.get("vertices_after", len(xyz)))
        attrs["_fit_rmse_m"] = f"{float(fit.get('fit_rmse_m', 0.0)):.4f}"

        shape = core.CvatShape(label=label, xyz=xyz, attributes=attrs)
        if t["kind"] == KIND_HATCH and not a.emit_hatch:
            pass
        elif t["review_required"]:
            review_shapes.append(shape)
        else:
            main_shapes.append(shape)

        if not (t["kind"] == KIND_HATCH and not a.emit_hatch):
            output_index += 1
            rec = _annotator_record(output_index, t, xyz, label, source, s_res)
            annotator_all.append(rec)
            if t["review_required"]:
                annotator_review.append(rec)
            else:
                annotator_main.append(rec)

        row = {
            "index": m["index"], "kind": m["kind"], "output_label": label,
            "source": source, "line_type": m["line_type"],
            "lane_number": t.get("lane_number", ""),
            "confidence": round(t["confidence"], 4),
            "review_required": t["review_required"],
            "start_s_m": m["start_s_m"], "end_s_m": m["end_s_m"],
            "length_m": m["length_m"], "median_d_m": round(m["median_d_m"], 3),
            "coverage": round(m.get("coverage", 0.0), 4),
            "mean_strength_sigma": round(m.get("mean_strength", 0.0), 3),
            "median_on_m": m.get("median_on_m", 0.0),
            "median_off_m": m.get("median_off_m", 0.0),
            "max_gap_m": m.get("max_gap_m", 0.0),
            "max_valid_gap_m": m.get("max_valid_gap_m", 0.0),
            "max_unknown_gap_m": m.get("max_unknown_gap_m", 0.0),
            "observed_ratio": m.get("observed_ratio", 0.0),
            "observed_on_valid_ratio": m.get("observed_on_valid_ratio",
                                                m.get("coverage", 0.0)),
            "sensor_valid_ratio": m.get("sensor_valid_ratio", 1.0),
            "interpolated_ratio": m.get("interpolated_ratio", 0.0),
            "frenet_ambiguity_ratio": m.get("frenet_ambiguity_ratio", 0.0),
            "max_frenet_ambiguity_run_m": m.get(
                "max_frenet_ambiguity_run_m", 0.0),
            "review_reasons": ",".join(t["review_reasons"]),
            "review_priority": round(t.get("review_priority", 0.0), 4),
            "neighbour_spacing_m": round(t.get("neighbour_spacing_m",
                                               float("nan")), 3),
            "median_step_m": m.get("median_step_m", ""),
            "sign_consistency": m.get("sign_consistency", ""),
            "geometry_fit": t.get("geometry_fit", {}).get("mode", "none"),
            "vertices_before": t.get("geometry_fit", {}).get(
                "vertices_before", len(xyz)),
            "vertices_after": t.get("geometry_fit", {}).get(
                "vertices_after", len(xyz)),
            "vertex_reduction_ratio": round(float(
                t.get("geometry_fit", {}).get("reduction_ratio", 0.0)), 4),
            "fit_rmse_m": round(float(
                t.get("geometry_fit", {}).get("fit_rmse_m", 0.0)), 4),
            "fit_p95_m": round(float(
                t.get("geometry_fit", {}).get("fit_p95_m", 0.0)), 4),
            "fit_max_error_m": round(float(
                t.get("geometry_fit", {}).get("fit_max_error_m", 0.0)), 4),
            "vertices": len(xyz),
        }
        row.update({f"rgb_{k}": round(v, 4) if isinstance(v, float) else v
                    for k, v in t["rgb"].items()})
        row.update({f"conf_{k}": round(v, 3)
                    for k, v in t["confidence_parts"].items()})
        rows.append(row)

    fit_modes: dict[str, int] = {}
    total_vertices_before = 0
    total_vertices_after = 0
    for t in tracks:
        fit = t.get("geometry_fit", {})
        mode = str(fit.get("mode", "none"))
        fit_modes[mode] = fit_modes.get(mode, 0) + 1
        total_vertices_before += int(fit.get("vertices_before", 0))
        total_vertices_after += int(fit.get("vertices_after", 0))
    reduction_pct = (
        100.0 * (1.0 - total_vertices_after / total_vertices_before)
        if total_vertices_before > 0 else 0.0
    )
    modes_text = ", ".join(f"{key}={value}"
                             for key, value in sorted(fit_modes.items()))
    print(f"[info] geometry smoothing: {modes_text}; "
          f"vertices={total_vertices_before}->{total_vertices_after} "
          f"(-{reduction_pct:.1f}%)")

    # ---------------- 出力
    xml_main = out_dir / f"{stem}.xml"
    xml_review = out_dir / f"{stem}_review.xml"
    r1 = core.write_cvat_xml(xml_main, main_shapes, image_name, image_meta)
    r2 = core.write_cvat_xml(xml_review, review_shapes, image_name, image_meta)
    print(f"[out] main   -> {xml_main}  ({r1['shape_count']} shapes)")
    print(f"[out] review -> {xml_review}  ({r2['shape_count']} shapes)")

    csv_path = out_dir / f"{stem}_tracks.csv"
    if rows:
        import csv as _csv
        keys = sorted({k for r in rows for k in r},
                      key=lambda k: (k not in rows[0], k))
        import io as _io
        _buf = _io.StringIO(newline="")
        w = _csv.DictWriter(_buf, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
        core.atomic_write_text(csv_path, _buf.getvalue(), encoding="utf-8-sig")
        print(f"[out] csv    -> {csv_path}")

    annotator_all_path = out_dir / f"{stem}_annotator.json"
    annotator_main_path = out_dir / f"{stem}_annotator_main.json"
    annotator_review_path = out_dir / f"{stem}_annotator_review.json"
    core.dump_json(annotator_all_path, annotator_all)
    core.dump_json(annotator_main_path, annotator_main)
    core.dump_json(annotator_review_path, annotator_review)
    print(f"[out] annotator -> {annotator_all_path} ({len(annotator_all)} records)")

    if not a.no_preview:
        save_preview(out_dir / f"{stem}_preview.png", npz1, tracks,
                     d_axis, d_res)

    meta = {
        "stage": 3,
        "production_version": core.PRODUCTION_VERSION,
        "code": {
            "stage3_sha256": core.sha256_file(Path(__file__)),
            "core_sha256": core.sha256_file(Path(core.__file__)),
        },
        "inputs": {"stage1": str(Path(a.stage1_npz).resolve()),
                   "stage2": str(Path(a.stage2_npz).resolve()),
                   "signatures": {
                       "stage1_npz": core.file_signature(Path(a.stage1_npz)),
                       "stage2_npz": core.file_signature(Path(a.stage2_npz)),
                   }},
        "rgb_health": health,
        "counts": {
            "total_tracks": len(tracks),
            "main_shapes": r1["shape_count"],
            "review_shapes": r2["shape_count"],
            "lane": sum(t["kind"] == KIND_LANE for t in tracks),
            "curb": sum(t["kind"] == KIND_CURB for t in tracks),
            "hatch_excluded": sum(t["kind"] == KIND_HATCH for t in tracks),
            "solid": sum(t["kind"] == KIND_LANE
                         and t["meta"]["line_type"] == "Solid_line"
                         for t in tracks),
            "dashed": sum(t["kind"] == KIND_LANE
                          and t["meta"]["line_type"] == "Dashed_line"
                          for t in tracks),
        },
        "geometry_smoothing": {
            "modes": fit_modes,
            "vertices_before": total_vertices_before,
            "vertices_after": total_vertices_after,
            "reduction_ratio": (
                1.0 - total_vertices_after / total_vertices_before
                if total_vertices_before > 0 else 0.0),
        },
        "xml": {"main": r1, "review": r2},
        "outputs": {"xml_main": str(xml_main), "xml_review": str(xml_review),
                    "csv": str(csv_path),
                    "annotator_all": str(annotator_all_path),
                    "annotator_main": str(annotator_main_path),
                    "annotator_review": str(annotator_review_path)},
        "parameters": vars(a),
        "parameter_fingerprint": core.parameter_fingerprint(
            vars(a), exclude=(
                "stage1_npz", "stage2_npz", "output_dir", "stem",
                "no_preview",
            )
        ),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    core.dump_json(out_dir / f"{stem}_stage3.meta.json", meta)
    c = meta["counts"]
    print(f"[done] Stage3: lane={c['lane']} (solid={c['solid']}, "
          f"dashed={c['dashed']}), curb={c['curb']}, "
          f"hatch除外={c['hatch_excluded']} | "
          f"main={c['main_shapes']}, review={c['review_shapes']}  "
          f"({meta['elapsed_sec']}s)")
    return meta


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
