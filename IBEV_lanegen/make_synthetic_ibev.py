#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synthetic_ibev.py  --  合成テストデータ生成

実データの厄介な性質を意図的に再現した IBEV 相当の点群を作る。

再現している条件:
  * 反射強度が 8bit packed RGB グレースケールでしか手に入らない
  * 路面 55 前後 / 塗装 68 前後 = 分離幅わずか 13 カウント
  * 横方向の距離減衰と、s 方向のゆっくりしたベースライン変動
  * 破線 (8m 塗装 / 12m 空白)
  * 25m の欠測区間 (オクルージョン相当)
  * 1 本まるごと欠落した車線境界線 (構造穴埋めのテスト用)
  * 導流帯相当の 0.9m 間隔ストライプ (間隔フィルタのテスト用)
  * 高さ 0.15m の縁石と、高さ 0.95m の壁 (段差上限のテスト用)

正解値は *_truth.json に出力するので、Stage3 の結果と突き合わせられる。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LANE_LINES = [
    # (d, kind, s_start, s_end)
    (-8.75, "solid", 0.0, 100.0),
    (-5.25, "faint", 0.0, 100.0),       # 通常閾値では落ちる -> 構造穴埋めの標的
    (-1.75, "dashed", 0.0, 100.0),
    (1.75, "dashed", 0.0, 100.0),
    (5.25, "solid", 0.0, 100.0),
]
OCCLUSION = (1.75, 55.0, 80.0)          # この線のこの区間だけ証拠を消す
HATCH_D = [2.45, 3.35, 4.25]            # 0.9m 間隔ストライプ
HATCH_S = (30.0, 45.0)
CURB_D = 10.0
CURB_HEIGHT = 0.15
WALL_D = 11.0
WALL_HEIGHT = 0.95

PAINT_HALF_WIDTH = 0.075                # 白線幅 15cm
DASH_ON, DASH_OFF = 8.0, 12.0


def build_centerline(length: float, step: float = 0.05):
    s = np.arange(0.0, length + step, step)
    curvature = 0.004 * np.sin(2.0 * np.pi * s / 100.0)
    heading = np.cumsum(curvature) * step
    x = np.cumsum(np.cos(heading)) * step
    y = np.cumsum(np.sin(heading)) * step
    tangent = np.column_stack([np.cos(heading), np.sin(heading)])
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return s, np.column_stack([x, y]), tangent, normal


FAINT_RATIO = 0.42     # かすれた線の反射コントラスト比


def is_paint(d: np.ndarray, s: np.ndarray) -> np.ndarray:
    """各点の塗装コントラスト係数 (0 = 路面, 1 = 通常塗装)。"""
    paint = np.zeros(len(d), dtype=np.float32)
    for line_d, kind, s0, s1 in LANE_LINES:
        near = (np.abs(d - line_d) <= PAINT_HALF_WIDTH) & (s >= s0) & (s <= s1)
        if kind == "dashed":
            phase = np.mod(s, DASH_ON + DASH_OFF)
            near &= phase < DASH_ON
        level = FAINT_RATIO if kind == "faint" else 1.0
        paint = np.maximum(paint, near.astype(np.float32) * level)
    # 破線側のオクルージョン
    occ_d, occ_s0, occ_s1 = OCCLUSION
    occ = (np.abs(d - occ_d) <= PAINT_HALF_WIDTH) & (s >= occ_s0) & (s <= occ_s1)
    paint[occ] = 0.0
    # 導流帯ストライプ
    for hd in HATCH_D:
        paint = np.maximum(paint, (
            (np.abs(d - hd) <= PAINT_HALF_WIDTH)
            & (s >= HATCH_S[0]) & (s <= HATCH_S[1])).astype(np.float32))
    return paint


def surface_z(d: np.ndarray, s: np.ndarray) -> np.ndarray:
    z = 0.010 * s - 0.020 * np.abs(d)          # 縦断勾配 + 横断勾配
    curb = np.abs(d) >= CURB_D
    z = np.where(curb, z + CURB_HEIGHT, z)
    wall = np.abs(d) >= WALL_D
    z = np.where(wall, z + WALL_HEIGHT, z)
    return z


def pack_rgb(r, g, b) -> np.ndarray:
    raw = ((r.astype(np.uint32) << 16)
           | (g.astype(np.uint32) << 8)
           | b.astype(np.uint32))
    return raw.view(np.float32) if raw.dtype == np.uint32 else raw


def write_pcd(path: Path, xyz: np.ndarray, rgb_packed: np.ndarray) -> None:
    n = len(xyz)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA ascii\n"
    )
    body = np.column_stack([xyz, rgb_packed])
    with path.open("w", encoding="ascii") as f:
        f.write(header)
        np.savetxt(f, body, fmt="%.4f %.4f %.4f %.10g")
    print(f"[out] {path}  ({n:,} points)")


def main() -> int:
    p = argparse.ArgumentParser(description="合成 IBEV テストデータ生成")
    p.add_argument("-o", "--output-dir", default="synth")
    p.add_argument("--length", type=float, default=100.0)
    p.add_argument("--half-width", type=float, default=11.5)
    p.add_argument("--density", type=float, default=400.0,
                   help="点密度 [points/m^2]")
    p.add_argument("--seed", type=int, default=20260725)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s_dense, center, tangent, normal = build_centerline(a.length)

    area = a.length * 2.0 * a.half_width
    n = int(area * a.density)
    s = rng.uniform(0.0, a.length, n)
    d = rng.uniform(-a.half_width, a.half_width, n)

    # --- World 座標
    idx = np.clip((s / 0.05).astype(int), 0, len(s_dense) - 1)
    xy = center[idx] + d[:, None] * normal[idx]
    z = surface_z(d, s) + rng.normal(0.0, 0.015, n)
    xyz = np.column_stack([xy, z]).astype(np.float32)

    # --- 反射強度 (8bit 相当まで落とす)
    paint = is_paint(d, s)
    base = 55.0 + 13.0 * paint
    base += rng.normal(0.0, np.where(paint > 0, 4.0, 3.5), n)
    gain = (1.0 - 0.012 * np.abs(d)) * (1.0 + 0.08 * np.sin(2 * np.pi * s / 60.0))
    intensity = np.clip(np.rint(base * gain), 0, 255).astype(np.uint8)
    iu32 = intensity.astype(np.uint32)
    packed_gray = ((iu32 << 16) | (iu32 << 8) | iu32).view(np.float32) \
        if False else np.frombuffer(
            ((iu32 << 16) | (iu32 << 8) | iu32).astype(np.uint32).tobytes(),
            dtype=np.float32)
    write_pcd(out / "IBEV_synth.pcd", xyz, packed_gray)

    # --- 色つき点群 (1/3 に間引き)
    sub = rng.random(n) < 0.34
    col = np.empty((int(sub.sum()), 3), dtype=np.uint32)
    ps, ds = paint[sub] > 0.5, d[sub]
    sidewalk = np.abs(ds) >= CURB_D
    col[:, 0] = np.where(ps, 205, np.where(sidewalk, 150, 92))
    col[:, 1] = np.where(ps, 205, np.where(sidewalk, 148, 92))
    col[:, 2] = np.where(ps, 200, np.where(sidewalk, 142, 94))
    col = np.clip(col.astype(np.int32)
                  + rng.integers(-8, 9, col.shape), 0, 255).astype(np.uint32)
    packed_col = np.frombuffer(
        ((col[:, 0] << 16) | (col[:, 1] << 8) | col[:, 2]
         ).astype(np.uint32).tobytes(), dtype=np.float32)
    write_pcd(out / "RGBBEV_synth.pcd", xyz[sub], packed_col)

    # --- mapping_pose.txt (2m 間隔)
    pose_idx = np.arange(0, len(s_dense), int(2.0 / 0.05))
    heading = np.arctan2(tangent[pose_idx, 1], tangent[pose_idx, 0])
    pose = np.column_stack([
        np.arange(len(pose_idx)),
        center[pose_idx, 0], center[pose_idx, 1],
        0.010 * s_dense[pose_idx],
        np.zeros(len(pose_idx)), np.zeros(len(pose_idx)), heading,
    ])
    pose_path = out / "mapping_pose_synth.txt"
    with pose_path.open("w", encoding="ascii") as f:
        f.write("lidar_frame x y z roll pitch yaw\n")
        np.savetxt(f, pose, fmt="%d %.5f %.5f %.5f %.6f %.6f %.6f")
    print(f"[out] {pose_path}  ({len(pose)} poses)")

    # --- 正解 CVAT XML (評価用)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lanegen_core as core
    res = 0.05
    image_meta = {
        "origin_x": float(xyz[:, 0].min()), "origin_y": float(xyz[:, 1].min()),
        "resolution": res,
        "width_px": int(np.ceil((xyz[:, 0].max() - xyz[:, 0].min()) / res)) + 1,
        "height_px": int(np.ceil((xyz[:, 1].max() - xyz[:, 1].min()) / res)) + 1,
    }
    shapes = []
    s_gt = np.arange(0.0, a.length + 0.5, 1.0)
    gi = np.clip((s_gt / 0.05).astype(int), 0, len(s_dense) - 1)
    for line_d, kind, s0, s1 in LANE_LINES:
        sel = (s_gt >= s0) & (s_gt <= s1)
        pts = center[gi[sel]] + line_d * normal[gi[sel]]
        z_gt = surface_z(np.full(sel.sum(), line_d), s_gt[sel])
        shapes.append(core.CvatShape(
            "lane_line", np.column_stack([pts, z_gt]),
            {"line_type": "Dashed_line" if kind == "dashed" else "Solid_line",
             "line_color": "white", "lane_number": ""}))
    for cd in (-CURB_D, CURB_D):
        pts = center[gi] + cd * normal[gi]
        z_gt = surface_z(np.full(len(s_gt), cd), s_gt)
        shapes.append(core.CvatShape(
            "curb_boundary", np.column_stack([pts, z_gt]),
            {"boundary_id": f"GT{int(cd):+d}"}))
    core.write_cvat_xml(out / "truth_gt.xml", shapes, "IBEV_synth.png",
                        image_meta)
    print(f"[out] {out / 'truth_gt.xml'}  ({len(shapes)} shapes)")

    truth = {
        "length_m": a.length,
        "lane_lines": [
            {"d": ld, "kind": k, "s_start": s0, "s_end": s1}
            for ld, k, s0, s1 in LANE_LINES
        ],
        "occlusion": {"d": OCCLUSION[0], "s_start": OCCLUSION[1],
                      "s_end": OCCLUSION[2]},
        "hatch": {"d": HATCH_D, "s_start": HATCH_S[0], "s_end": HATCH_S[1]},
        "curb": {"d": [-CURB_D, CURB_D], "height_m": CURB_HEIGHT},
        "wall": {"d": [-WALL_D, WALL_D], "height_m": WALL_HEIGHT},
        "expected": {
            "detected_lane_lines": [-8.75, -1.75, 1.75, 5.25],
            "infill_candidate": -5.25,
            "faint_contrast_ratio": FAINT_RATIO,
            "rejected_as_hatch": HATCH_D,
            "curb_boundaries": [-CURB_D, CURB_D],
        },
    }
    (out / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[out] {out / 'truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
