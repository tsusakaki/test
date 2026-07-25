#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lanegen_core.py

IBEV 1パス自動ライン生成パイプラインの共通基盤。

このモジュールは以下を提供する。

* ASCII PCD 読み込み (x y z [rgb|intensity])
* packed RGB からの反射強度復号
* mapping_pose.txt からの等間隔軌跡生成
* World -> Frenet(s, d) 投影
* セル単位の上位k平均集約 (量子化ノイズ低減の要)
* CVAT XML 1.1 出力 (BevCvatConverter 互換)

既存の ibev_auto_lane_to_cvat_xml.py / ibev_parallel_lane_to_cvat_xml.py と
同一の座標式・ラベル定義を用いるため、生成物はそのまま
BevCvatConverter.cvat_xml_to_lanes() で読み戻せる。
"""

from __future__ import annotations

import io
import json
import math
import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from lanegen_pcd_io import (
    PcdHeader,
    PcdReadCancelled,
    PcdChunkReader,
    iter_pcd_chunks,
    load_pcd,
    load_ascii_pcd,
    read_pcd_header,
)

PRODUCTION_VERSION = "7.0.0-production-v7"

__all__ = [
    "PRODUCTION_VERSION",
    "sha256_file",
    "file_signature",
    "parameter_fingerprint",
    "PcdHeader",
    "PcdReadCancelled",
    "PcdChunkReader",
    "read_pcd_header",
    "iter_pcd_chunks",
    "load_pcd",
    "load_ascii_pcd",
    "decode_intensity",
    "Trajectory",
    "load_mapping_pose",
    "build_trajectory",
    "FrenetProjection",
    "project_to_frenet",
    "groupwise_topk_mean",
    "annulus_background",
    "row_robust_scale",
    "rdp",
    "resample_max_spacing",
    "meter_to_cvat_pixel",
    "CVAT_LABEL_DEFINITIONS",
    "write_cvat_xml",
    "atomic_write_text",
    "atomic_dump_json",
    "atomic_savez_compressed",
]


# =============================================================================
# 再現性・入力追跡
# =============================================================================

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """ファイルを全読込せず SHA-256 を計算する。"""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_signature(path: Path) -> dict:
    """量産ログ用の再現可能なファイル署名。"""
    path = Path(path).expanduser().resolve()
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "sha256": sha256_file(path),
    }


def parameter_fingerprint(parameters: dict, exclude: Sequence[str] = ()) -> str:
    """Return deterministic SHA-256 for semantic generation parameters."""
    excluded = set(exclude)
    payload = {
        str(k): v for k, v in parameters.items() if str(k) not in excluded
    }
    encoded = json.dumps(
        json_safe(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# =============================================================================
# PCD 入出力
# =============================================================================

def _load_ascii_pcd_legacy(path: Path) -> tuple[np.ndarray, Optional[np.ndarray],
                                         Optional[np.ndarray], dict]:
    """
    ASCII PCD を読む。

    Returns
    -------
    xyz        : (N,3) float32
    rgb_packed : (N,) float32 または None   -- FIELDS に rgb がある場合
    intensity  : (N,) float32 または None   -- FIELDS に intensity/i/reflectivity
    info       : dict
    """
    path = Path(path)
    with path.open("rb") as f:
        header: dict[str, list[str]] = {}
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PCDヘッダの途中でEOFになりました: {path}")
            line = raw.decode("ascii", "ignore").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            header[parts[0].lower()] = parts[1:]
            if parts[0].lower() == "data":
                break

        data_type = header.get("data", [""])[0].lower()
        if data_type != "ascii":
            raise NotImplementedError(
                f"ASCII PCD のみ対応です: DATA {data_type} ({path})"
            )
        fields = [s.lower() for s in header.get("fields", [])]
        for key in ("x", "y", "z"):
            if key not in fields:
                raise ValueError(f"FIELDS に {key} がありません: {fields}")

        text = f.read().decode("ascii", "ignore")
        arr = np.loadtxt(io.StringIO(text), dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(fields):
        raise ValueError(
            f"データ列数とFIELDSが不一致: {arr.shape[1]} vs {len(fields)} ({path})"
        )

    fi = {name: i for i, name in enumerate(fields)}
    xyz = arr[:, [fi["x"], fi["y"], fi["z"]]].astype(np.float32, copy=True)
    finite = np.isfinite(xyz).all(axis=1)

    rgb_packed = None
    if "rgb" in fi:
        rgb_packed = arr[:, fi["rgb"]].astype(np.float32, copy=True)
        finite &= np.isfinite(rgb_packed)

    intensity = None
    for key in ("intensity", "reflectivity", "i", "intensities"):
        if key in fi:
            intensity = arr[:, fi[key]].astype(np.float32, copy=True)
            finite &= np.isfinite(intensity)
            break

    xyz = xyz[finite]
    if rgb_packed is not None:
        rgb_packed = rgb_packed[finite]
    if intensity is not None:
        intensity = intensity[finite]

    info = {
        "path": str(path),
        "fields": fields,
        "header_points": int(header.get("points", [str(len(arr))])[0]),
        "valid_points": int(len(xyz)),
        "has_native_intensity": intensity is not None,
    }
    print(f"[info] PCD {path.name}: {len(xyz):,} points, fields={fields}")
    if intensity is None and rgb_packed is not None:
        print("[warn] 真の intensity フィールドがありません "
              "(packed RGB から復号します。実効ダイナミックレンジが狭くなります)")
    return xyz, rgb_packed, intensity, info


def decode_intensity(rgb_packed: np.ndarray) -> tuple[np.ndarray, dict]:
    """packed float RGB から 0..255 のグレースケール強度を復号する。"""
    raw = np.ascontiguousarray(rgb_packed, dtype=np.float32).view(np.uint32)
    r = ((raw >> 16) & 0xFF).astype(np.float32)
    g = ((raw >> 8) & 0xFF).astype(np.float32)
    b = (raw & 0xFF).astype(np.float32)
    gray_ratio = float(np.mean((r == g) & (g == b)))
    if gray_ratio >= 0.999:
        intensity = r
        mode = "packed_rgb_grayscale_red"
    else:
        intensity = (r + g + b) / 3.0
        mode = "packed_rgb_channel_mean"
    stats = {
        "decode_mode": mode,
        "grayscale_ratio": gray_ratio,
        "min": float(intensity.min()),
        "p50": float(np.percentile(intensity, 50)),
        "p90": float(np.percentile(intensity, 90)),
        "p99": float(np.percentile(intensity, 99)),
        "max": float(intensity.max()),
    }
    return intensity.astype(np.float32), stats


def decode_rgb_channels(rgb_packed: np.ndarray) -> np.ndarray:
    """packed float RGB を (N,3) uint8 に展開する。"""
    raw = np.ascontiguousarray(rgb_packed, dtype=np.float32).view(np.uint32)
    r = ((raw >> 16) & 0xFF).astype(np.uint8)
    g = ((raw >> 8) & 0xFF).astype(np.uint8)
    b = (raw & 0xFF).astype(np.uint8)
    return np.column_stack([r, g, b])


# =============================================================================
# 軌跡
# =============================================================================

@dataclass
class Trajectory:
    xy: np.ndarray        # (N,2) 等間隔
    z: np.ndarray         # (N,)
    tangent: np.ndarray   # (N,2)
    normal: np.ndarray    # (N,2) tangent の左法線
    s: np.ndarray         # (N,)
    resolution: float
    yaw: np.ndarray       # (N,) mapping_pose由来。無い場合は幾何接線yaw
    geometric_yaw: np.ndarray  # (N,) 軌跡接線由来
    yaw_valid: bool = False
    heading_error: np.ndarray = field(default_factory=lambda: np.zeros(0))
    raw_count: int = 0
    filtered_count: int = 0

    @property
    def length_m(self) -> float:
        return float(self.s[-1])


def load_mapping_pose(path: Path) -> np.ndarray:
    """header 1 行付き mapping_pose.txt を読む。"""
    path = Path(path)
    for skip in (1, 0):
        try:
            data = np.loadtxt(path, skiprows=skip, dtype=np.float64)
            break
        except Exception:
            data = None
    if data is None:
        raise ValueError(f"mapping_pose を読み込めません: {path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError("mapping_pose は [frame] x y z ... が必要です")
    valid = np.isfinite(data[:, 1:4]).all(axis=1)
    data = data[valid]
    if len(data) < 2:
        raise ValueError("有効な mapping_pose が 2 点未満です")
    return data


def _odd_window(value: int, maximum: int) -> int:
    value = max(3, int(value))
    if value % 2 == 0:
        value += 1
    max_odd = maximum if maximum % 2 == 1 else maximum - 1
    return max(3, min(value, max_odd))


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def build_trajectory(pose: np.ndarray, min_step: float = 0.1,
                     smooth_window: int = 31,
                     resolution: float = 0.25) -> Trajectory:
    """停止区間除去 -> 平滑化 -> 等間隔リサンプル。

    mapping_pose が roll/pitch/yaw を含む一般的な形式
    ``frame x y z roll pitch yaw ...`` の場合は yaw も保持する。
    yawは交差点・Uターン近傍のFrenet曖昧性診断に使用する。
    """
    raw_xy = pose[:, 1:3]
    raw_z = pose[:, 3]
    has_yaw = pose.shape[1] >= 7 and np.isfinite(pose[:, 6]).sum() >= 2
    raw_yaw = pose[:, 6].astype(np.float64) if has_yaw else None

    keep = [0]
    for i in range(1, len(raw_xy)):
        if float(np.linalg.norm(raw_xy[i] - raw_xy[keep[-1]])) >= min_step:
            keep.append(i)
    if keep[-1] != len(raw_xy) - 1:
        keep.append(len(raw_xy) - 1)

    xy = raw_xy[keep].astype(np.float64)
    z = raw_z[keep].astype(np.float64)
    yaw = np.unwrap(raw_yaw[keep]) if has_yaw else None
    if len(xy) < 4:
        raise ValueError("停止Pose除去後の軌跡点が不足しています")

    try:
        from scipy.signal import savgol_filter
        win = _odd_window(smooth_window, len(xy))
        if win >= 5:
            poly = min(3, win - 2)
            xy[:, 0] = savgol_filter(xy[:, 0], win, poly, mode="interp")
            xy[:, 1] = savgol_filter(xy[:, 1], win, poly, mode="interp")
            z = savgol_filter(z, win, poly, mode="interp")
            if yaw is not None:
                yaw = savgol_filter(yaw, win, poly, mode="interp")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] 軌跡平滑化をスキップ: {exc}")

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if not np.any(seg > 1e-6):
        raise ValueError("軌跡長が 0 です")
    cumulative = np.r_[0.0, np.cumsum(seg)]
    total = float(cumulative[-1])
    s = np.arange(0.0, total + resolution * 0.5, resolution)
    if s[-1] > total:
        s[-1] = total
    x_dense = np.interp(s, cumulative, xy[:, 0])
    y_dense = np.interp(s, cumulative, xy[:, 1])
    z_dense = np.interp(s, cumulative, z)

    dx = np.gradient(x_dense, s, edge_order=1)
    dy = np.gradient(y_dense, s, edge_order=1)
    nrm = np.hypot(dx, dy)
    nrm[nrm < 1e-9] = 1.0
    tangent = np.column_stack([dx / nrm, dy / nrm])
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    geometric_yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))

    if yaw is not None:
        yaw_dense = np.interp(s, cumulative, yaw)
        # 幾何yawと2π枝を合わせる。
        yaw_dense = geometric_yaw + _wrap_angle(yaw_dense - geometric_yaw)
        heading_error = np.abs(_wrap_angle(yaw_dense - geometric_yaw))
        yaw_valid = True
    else:
        yaw_dense = geometric_yaw.copy()
        heading_error = np.zeros_like(geometric_yaw)
        yaw_valid = False

    print(f"[info] trajectory: raw={len(raw_xy)}, filtered={len(xy)}, "
          f"dense={len(s)}, length={total:.1f}m, yaw={'yes' if yaw_valid else 'no'}")
    return Trajectory(
        xy=np.column_stack([x_dense, y_dense]), z=z_dense,
        tangent=tangent, normal=normal, s=s, resolution=resolution,
        yaw=yaw_dense, geometric_yaw=geometric_yaw,
        yaw_valid=yaw_valid, heading_error=heading_error,
        raw_count=len(raw_xy), filtered_count=len(xy),
    )


# =============================================================================
# Frenet 投影
# =============================================================================

@dataclass
class FrenetProjection:
    s_index: np.ndarray   # (M,) 採用点のtrajectory index
    d_value: np.ndarray   # (M,) 採用点の符号付き横距離
    keep: np.ndarray      # (N,) 元配列に対する採用マスク
    nearest_distance: np.ndarray  # (M,) XY最近傍距離
    ambiguous_input: np.ndarray   # (N,) 離れたs候補が競合した点
    candidate_margin: np.ndarray  # (M,) 代替分岐とのcost差。infなら代替なし
    best_s_index_all: np.ndarray  # (N,) drop前の最良候補
    best_d_value_all: np.ndarray  # (N,) drop前のd
    within_corridor_all: np.ndarray  # (N,)


def project_to_frenet(points: np.ndarray, trajectory: Trajectory,
                      half_width: float,
                      chunk: int = 1_000_000,
                      candidates: int = 8,
                      z_weight: float = 1.5,
                      heading_weight: float = 0.15,
                      ambiguity_margin: float = 0.20,
                      ambiguity_s_separation: float = 8.0,
                      drop_ambiguous: bool = True) -> FrenetProjection:
    """World点を軌跡基準のFrenetへ投影する。

    Production V2では最近傍1点だけでなく複数候補を比較する。
    XY距離に加え、入力にZがあれば軌跡Zとの差、mapping_pose yawと
    幾何接線の不整合をcostへ加える。さらに、最良候補とほぼ同点で
    ``ambiguity_s_separation`` 以上離れたs候補がある点を曖昧と判定する。
    交差点・Uターン・往復軌跡近接で別区間へ点を混入させないため、
    デフォルトでは曖昧点を証拠集約から除外する。
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy が必要です: pip install scipy") from exc

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points は (N,2) または (N,3) が必要です")
    xy = pts[:, :2].astype(np.float64, copy=False)
    has_z = pts.shape[1] >= 3
    point_z = pts[:, 2].astype(np.float64, copy=False) if has_z else None

    tree = cKDTree(trajectory.xy)
    n = len(xy)
    k = max(1, min(int(candidates), len(trajectory.xy)))
    best_idx = np.empty(n, dtype=np.int64)
    best_xy_dist = np.empty(n, dtype=np.float32)
    best_margin = np.full(n, np.inf, dtype=np.float32)
    ambiguous = np.zeros(n, dtype=bool)
    sep_cells = max(1, int(round(ambiguity_s_separation / trajectory.resolution)))

    for begin in range(0, n, chunk):
        end = min(begin + chunk, n)
        dd, ii = tree.query(xy[begin:end], k=k, workers=-1)
        if k == 1:
            dd = dd[:, None]
            ii = ii[:, None]
        dd = np.asarray(dd, dtype=np.float64)
        ii = np.asarray(ii, dtype=np.int64)

        cost = dd.copy()
        if has_z:
            dz = np.abs(point_z[begin:end, None] - trajectory.z[ii])
            cost += float(z_weight) * dz
        if trajectory.yaw_valid and len(trajectory.heading_error) == len(trajectory.s):
            cost += float(heading_weight) * trajectory.heading_error[ii]

        row = np.arange(end - begin)

        # XY最近傍を現在の局所分岐の基準とする。Zやyawは、sが大きく
        # 離れた別分岐を選び分けるためだけに使う。これにより通常の
        # カーブ上でZ差が隣接s indexを不必要にずらすことを防ぐ。
        base_idx = ii[:, 0]
        base_cost = cost[:, 0]
        far = np.abs(ii - base_idx[:, None]) >= sep_cells
        alt_cost_matrix = np.where(far, cost, np.inf)
        alt_col = np.argmin(alt_cost_matrix, axis=1)
        alt_cost = alt_cost_matrix[row, alt_col]
        finite = np.isfinite(alt_cost)
        diff = alt_cost - base_cost

        # 別分岐がambiguity_marginを超えて明確に良い場合だけ切替える。
        choose_alt = finite & (diff < -float(ambiguity_margin))
        bidx = np.where(choose_alt, ii[row, alt_col], base_idx)
        bxy = np.where(choose_alt, dd[row, alt_col], dd[:, 0])
        best_idx[begin:end] = bidx
        best_xy_dist[begin:end] = bxy.astype(np.float32)

        margin = np.abs(diff)
        best_margin[begin:end] = np.where(finite, margin, np.inf).astype(np.float32)
        ambiguous[begin:end] = finite & (margin <= float(ambiguity_margin))

    delta = xy - trajectory.xy[best_idx]
    d_all = np.einsum("ij,ij->i", delta, trajectory.normal[best_idx])
    within = np.abs(d_all) <= half_width
    keep = within & (~ambiguous if drop_ambiguous else True)

    return FrenetProjection(
        s_index=best_idx[keep],
        d_value=d_all[keep].astype(np.float32),
        keep=keep,
        nearest_distance=best_xy_dist[keep],
        ambiguous_input=ambiguous,
        candidate_margin=best_margin[keep],
        best_s_index_all=best_idx,
        best_d_value_all=d_all.astype(np.float32),
        within_corridor_all=within,
    )


# =============================================================================
# セル集約
# =============================================================================

def groupwise_topk_mean(flat: np.ndarray, values: np.ndarray, ncell: int,
                        k: int = 3, largest: bool = True
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    セルごとに上位(または下位) k 個の平均を求める。

    8bit 量子化された反射強度に対して max ではなく top-k mean を使うと、
    量子化ノイズが約 sqrt(k) 分の 1 になり実効分解能が上がる。
    1 セルあたり 10 点程度あるデータでは 8bit -> 約 9.5bit 相当。

    Returns
    -------
    mean  : (ncell,) float32   点が無いセルは 0
    count : (ncell,) int32     採用した点数 (0..k)
    """
    if len(flat) == 0:
        return (np.zeros(ncell, np.float32), np.zeros(ncell, np.int32))
    sign = -1.0 if largest else 1.0
    order = np.lexsort((sign * values.astype(np.float64), flat))
    f = flat[order]
    v = values[order].astype(np.float64)

    new_group = np.empty(len(f), dtype=bool)
    new_group[0] = True
    np.not_equal(f[1:], f[:-1], out=new_group[1:])
    group_start = np.flatnonzero(new_group)
    group_id = np.cumsum(new_group) - 1
    rank = np.arange(len(f), dtype=np.int64) - group_start[group_id]

    sel = rank < k
    sums = np.bincount(f[sel], weights=v[sel], minlength=ncell)
    cnts = np.bincount(f[sel], minlength=ncell)
    mean = np.zeros(ncell, dtype=np.float32)
    nz = cnts > 0
    mean[nz] = (sums[nz] / cnts[nz]).astype(np.float32)
    return mean, cnts.astype(np.int32)


def groupwise_mean(flat: np.ndarray, values: np.ndarray,
                   ncell: int) -> tuple[np.ndarray, np.ndarray]:
    """セル平均。"""
    cnts = np.bincount(flat, minlength=ncell)
    sums = np.bincount(flat, weights=values.astype(np.float64), minlength=ncell)
    mean = np.zeros(ncell, dtype=np.float32)
    nz = cnts > 0
    mean[nz] = (sums[nz] / cnts[nz]).astype(np.float32)
    return mean, cnts.astype(np.int32)


# =============================================================================
# 局所背景・ロバストスケール
# =============================================================================

def _masked_running_sum(values: np.ndarray, mask: np.ndarray,
                        lo: int, hi: int, axis: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    axis 方向の [i+lo, i+hi] 窓 (両端含む) について、mask=True のみの
    合計と個数を返す。累積和で O(N)。
    """
    v = np.where(mask, values, 0.0).astype(np.float64)
    m = mask.astype(np.float64)
    if axis != 0:
        v = np.moveaxis(v, axis, 0)
        m = np.moveaxis(m, axis, 0)
    n = v.shape[0]
    cv = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, axis=0)])
    cm = np.concatenate([np.zeros((1,) + m.shape[1:]), np.cumsum(m, axis=0)])
    idx = np.arange(n)
    a = np.clip(idx + lo, 0, n)
    b = np.clip(idx + hi + 1, 0, n)
    sv = cv[b] - cv[a]
    sm = cm[b] - cm[a]
    if axis != 0:
        sv = np.moveaxis(sv, 0, axis)
        sm = np.moveaxis(sm, 0, axis)
    return sv, sm


def annulus_background(values: np.ndarray, mask: np.ndarray,
                       inner_cells: int, outer_cells: int,
                       axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    自セル近傍 (±inner_cells) を除外した ±outer_cells の平均を背景とする。

    中央を除外するので、細い高反射帯が自分自身の背景を押し上げない。
    白線幅 15cm に対し inner=0.25m / outer=0.80m 程度が目安。

    Returns
    -------
    background : 同形状 float32 (有効近傍が無い場所は nan)
    n_used     : 同形状 int32
    """
    sv_out, sm_out = _masked_running_sum(values, mask, -outer_cells,
                                         outer_cells, axis=axis)
    sv_in, sm_in = _masked_running_sum(values, mask, -inner_cells,
                                       inner_cells, axis=axis)
    sv = sv_out - sv_in
    sm = sm_out - sm_in
    bg = np.full(values.shape, np.nan, dtype=np.float32)
    ok = sm > 0
    bg[ok] = (sv[ok] / sm[ok]).astype(np.float32)
    return bg, sm.astype(np.int32)


def row_robust_scale(excess: np.ndarray, mask: np.ndarray,
                     smooth_cells: int = 40,
                     floor_value: float = 0.5) -> np.ndarray:
    """
    s 列ごとに excess のロバストな散布度 (1.4826*MAD) を求め、
    s 方向に平滑化して返す。形状 (ns,)。

    走行距離・入射角によるベースライン変動を、しきい値側ではなく
    スケール側で吸収するための量。
    """
    nd, ns = excess.shape
    scale = np.full(ns, np.nan, dtype=np.float64)
    for j in range(ns):
        col = excess[:, j][mask[:, j]]
        if col.size >= 16:
            med = np.median(col)
            scale[j] = 1.4826 * np.median(np.abs(col - med))
    valid = np.isfinite(scale)
    if not np.any(valid):
        return np.full(ns, floor_value, dtype=np.float32)
    idx = np.arange(ns)
    scale = np.interp(idx, idx[valid], scale[valid])
    if smooth_cells > 1:
        kernel = np.ones(int(smooth_cells)) / float(smooth_cells)
        pad = int(smooth_cells)
        padded = np.pad(scale, pad, mode="edge")
        scale = np.convolve(padded, kernel, mode="same")[pad:pad + ns]
    return np.maximum(scale, floor_value).astype(np.float32)


# =============================================================================
# ポリライン整形
# =============================================================================

def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker 簡略化 (反復実装)。"""
    if len(points) < 3 or epsilon <= 0:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p, q = points[i], points[j]
        seg = q - p
        norm = float(np.hypot(*seg))
        sub = points[i + 1:j]
        if norm < 1e-12:
            dist = np.linalg.norm(sub - p, axis=1)
        else:
            dist = np.abs(np.cross(seg, sub - p)) / norm
        m = int(np.argmax(dist))
        if dist[m] > epsilon:
            k = i + 1 + m
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return points[keep]


def resample_max_spacing(points: np.ndarray, max_spacing: float) -> np.ndarray:
    """隣接頂点間隔が max_spacing を超えないよう中間点を挿入する。"""
    if len(points) < 2 or max_spacing <= 0:
        return points
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dist = float(np.linalg.norm(b - a))
        n = int(math.ceil(dist / max_spacing))
        for t in range(1, n + 1):
            out.append(a + (b - a) * (t / n))
    return np.asarray(out, dtype=points.dtype)



def _deduplicate_consecutive(points: np.ndarray,
                             minimum_distance: float = 1.0e-6) -> np.ndarray:
    """連続する重複点を除去する。端点順序は維持する。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= 1:
        return pts.copy()
    delta = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, delta > float(minimum_distance)]
    out = pts[keep]
    if len(out) == 1 and len(pts) >= 2:
        out = np.vstack([pts[0], pts[-1]])
    return out


def _point_to_segment_distance(point: np.ndarray,
                               start: np.ndarray,
                               end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1.0e-18:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + ratio * segment
    return float(np.linalg.norm(point - projection))


def _robust_line_model(points: np.ndarray,
                       iterations: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IRLS付き直交回帰直線を返す。

    Returns
    -------
    center, direction, residuals
    """
    pts = np.asarray(points, dtype=np.float64)
    weights = np.ones(len(pts), dtype=np.float64)
    direction = pts[-1] - pts[0]
    if float(np.linalg.norm(direction)) < 1.0e-12:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    center = np.mean(pts, axis=0)

    for _ in range(max(1, int(iterations))):
        weight_sum = max(float(weights.sum()), 1.0e-12)
        center = np.sum(pts * weights[:, None], axis=0) / weight_sum
        centered = pts - center
        covariance = (centered * weights[:, None]).T @ centered / weight_sum
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        if float(np.dot(direction, pts[-1] - pts[0])) < 0:
            direction = -direction
        projections = centered @ direction
        fitted = center + projections[:, None] * direction
        residuals = np.linalg.norm(pts - fitted, axis=1)
        median = float(np.median(residuals))
        mad = 1.4826 * float(np.median(np.abs(residuals - median)))
        scale = max(mad, median * 0.25, 1.0e-4)
        cutoff = 2.5 * scale
        weights = np.ones_like(residuals)
        mask = residuals > cutoff
        weights[mask] = cutoff / np.maximum(residuals[mask], 1.0e-12)

    centered = pts - center
    projections = centered @ direction
    fitted = center + projections[:, None] * direction
    residuals = np.linalg.norm(pts - fitted, axis=1)
    return center, direction, residuals


def _robust_spline(points: np.ndarray,
                   smoothing_tolerance: float,
                   robust_iterations: int = 3):
    """累積距離を媒介変数にしたロバストな2次元平滑化スプライン。"""
    pts = _deduplicate_consecutive(points)
    if len(pts) < 4:
        return None, None, None
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    total = float(distance[-1])
    if total <= 1.0e-9:
        return None, None, None
    try:
        from scipy.interpolate import UnivariateSpline
    except Exception:
        return None, None, None

    degree = min(3, len(pts) - 1)
    weights = np.ones(len(pts), dtype=np.float64)
    tolerance = max(float(smoothing_tolerance), 1.0e-5)
    smoothing_factor = float(len(pts)) * tolerance * tolerance
    spline_x = spline_y = None

    for _ in range(max(1, int(robust_iterations))):
        # scipyのwは残差二乗へ直接掛かるため、IRLS重みの平方根を渡す。
        scipy_weights = np.sqrt(np.maximum(weights, 1.0e-4))
        spline_x = UnivariateSpline(
            distance, pts[:, 0], w=scipy_weights,
            s=smoothing_factor, k=degree, ext=3)
        spline_y = UnivariateSpline(
            distance, pts[:, 1], w=scipy_weights,
            s=smoothing_factor, k=degree, ext=3)
        fitted = np.column_stack([spline_x(distance), spline_y(distance)])
        residuals = np.linalg.norm(pts - fitted, axis=1)
        median = float(np.median(residuals))
        mad = 1.4826 * float(np.median(np.abs(residuals - median)))
        scale = max(mad, tolerance * 0.25, 1.0e-4)
        cutoff = 2.5 * scale
        weights = np.ones_like(residuals)
        mask = residuals > cutoff
        weights[mask] = cutoff / np.maximum(residuals[mask], 1.0e-12)
        # 端点が不必要に縮まないよう重みを上げる。
        weights[0] = max(weights[0], 8.0)
        weights[-1] = max(weights[-1], 8.0)

    fitted = np.column_stack([spline_x(distance), spline_y(distance)])
    residuals = np.linalg.norm(pts - fitted, axis=1)
    return distance, (spline_x, spline_y), residuals


def _adaptive_spline_polyline(spline_x, spline_y,
                              start: float, stop: float,
                              approximation_tolerance: float,
                              max_spacing: float,
                              maximum_depth: int = 20) -> np.ndarray:
    """スプラインを、曲率に応じた疎密ポリラインへ変換する。"""
    tolerance = max(float(approximation_tolerance), 1.0e-5)
    spacing = max(float(max_spacing), tolerance * 2.0)

    def evaluate(value: float) -> np.ndarray:
        return np.array([float(spline_x(value)), float(spline_y(value))],
                        dtype=np.float64)

    p_start = evaluate(start)
    p_stop = evaluate(stop)
    output: list[tuple[float, np.ndarray]] = [(float(start), p_start)]

    def recurse(u0: float, p0: np.ndarray,
                u1: float, p1: np.ndarray,
                depth: int) -> None:
        u_a = u0 + (u1 - u0) / 3.0
        u_b = u0 + 2.0 * (u1 - u0) / 3.0
        u_mid = (u0 + u1) * 0.5
        p_a = evaluate(u_a)
        p_b = evaluate(u_b)
        p_mid = evaluate(u_mid)
        deviation = max(
            _point_to_segment_distance(p_a, p0, p1),
            _point_to_segment_distance(p_mid, p0, p1),
            _point_to_segment_distance(p_b, p0, p1),
        )
        chord = float(np.linalg.norm(p1 - p0))
        if (depth < maximum_depth and
                (deviation > tolerance or chord > spacing)):
            recurse(u0, p0, u_mid, p_mid, depth + 1)
            recurse(u_mid, p_mid, u1, p1, depth + 1)
        else:
            output.append((float(u1), p1))

    recurse(float(start), p_start, float(stop), p_stop, 0)
    output.sort(key=lambda item: item[0])
    return np.asarray([point for _, point in output], dtype=np.float64)


def smooth_polyline_xy(points: np.ndarray, *,
                       mode: str = "auto",
                       straight_line_tolerance: float = 0.08,
                       spline_smoothing_tolerance: float = 0.06,
                       approximation_tolerance: float = 0.08,
                       max_vertex_spacing: float = 5.0,
                       simplify_tolerance: float = 0.03,
                       robust_iterations: int = 3) -> tuple[np.ndarray, dict]:
    """密な検出点列を、直線または平滑曲線の疎なポリラインへ変換する。

    `auto`ではロバスト直交回帰の残差が小さい線を直線扱いし、
    それ以外をロバスト3次スプラインで平滑化する。スプラインは曲率に
    応じて適応サンプリングするため、直線部は疎、曲線部だけ密になる。
    """
    requested_mode = str(mode).lower()
    if requested_mode not in {"auto", "line", "spline", "none"}:
        raise ValueError(f"unsupported smoothing mode: {mode}")

    original = np.asarray(points, dtype=np.float64)
    pts = _deduplicate_consecutive(original)
    before = int(len(original))
    if len(pts) < 2:
        return pts, {
            "mode": "none", "vertices_before": before,
            "vertices_after": int(len(pts)), "fit_rmse_m": 0.0,
            "fit_p95_m": 0.0, "fit_max_error_m": 0.0,
            "reduction_ratio": 0.0,
        }

    center, direction, line_residuals = _robust_line_model(
        pts, iterations=robust_iterations)
    line_median = float(np.median(line_residuals))
    line_mad = 1.4826 * float(np.median(np.abs(line_residuals - line_median)))
    robust_line_error = line_median + 2.5 * line_mad
    use_line = requested_mode == "line" or (
        requested_mode == "auto" and
        robust_line_error <= float(straight_line_tolerance)
    )

    if requested_mode == "none":
        output = rdp(pts, float(simplify_tolerance))
        output = resample_max_spacing(output, float(max_vertex_spacing))
        residuals = np.zeros(len(pts), dtype=np.float64)
        fit_mode = "none"
    elif use_line:
        projection = (pts - center) @ direction
        start_value = float(projection[0])
        stop_value = float(projection[-1])
        line_endpoints = np.vstack([
            center + start_value * direction,
            center + stop_value * direction,
        ])
        output = resample_max_spacing(line_endpoints, float(max_vertex_spacing))
        residuals = line_residuals
        fit_mode = "line"
    else:
        distance, splines, residuals = _robust_spline(
            pts,
            smoothing_tolerance=float(spline_smoothing_tolerance),
            robust_iterations=robust_iterations,
        )
        if distance is None or splines is None:
            output = rdp(pts, max(float(simplify_tolerance),
                                  float(approximation_tolerance)))
            output = resample_max_spacing(output, float(max_vertex_spacing))
            residuals = np.zeros(len(pts), dtype=np.float64)
            fit_mode = "rdp_fallback"
        else:
            output = _adaptive_spline_polyline(
                splines[0], splines[1],
                float(distance[0]), float(distance[-1]),
                approximation_tolerance=float(approximation_tolerance),
                max_spacing=float(max_vertex_spacing),
            )
            # 適応サンプリング後にもDPを掛け、同一直線上の中間点を削る。
            output = rdp(output, float(approximation_tolerance))
            output = resample_max_spacing(output, float(max_vertex_spacing))
            fit_mode = "spline"

    if len(output) < 2:
        output = np.vstack([pts[0], pts[-1]])
    residuals = np.asarray(residuals, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(residuals * residuals))) if len(residuals) else 0.0
    p95 = float(np.percentile(residuals, 95)) if len(residuals) else 0.0
    maximum = float(np.max(residuals)) if len(residuals) else 0.0
    after = int(len(output))
    metadata = {
        "mode": fit_mode,
        "vertices_before": before,
        "vertices_after": after,
        "fit_rmse_m": rmse,
        "fit_p95_m": p95,
        "fit_max_error_m": maximum,
        "straight_line_robust_error_m": float(robust_line_error),
        "reduction_ratio": (1.0 - after / before) if before > 0 else 0.0,
    }
    return output.astype(np.float32), metadata


# =============================================================================
# CVAT XML
# =============================================================================

def meter_to_cvat_pixel(x: float, y: float, image_meta: dict) -> tuple[float, float]:
    """BevCvatConverter.meter_to_pixel() と同一式。"""
    px = (x - image_meta["origin_x"]) / image_meta["resolution"]
    py = (image_meta["origin_y"]
          + image_meta["height_px"] * image_meta["resolution"]
          - y) / image_meta["resolution"]
    return float(px), float(py)


CVAT_LABEL_DEFINITIONS = [
    {
        "name": "lane_line", "type": "polyline",
        "attributes": [
            {"name": "line_type", "mutable": False, "input_type": "select",
             "default_value": "Solid_line",
             "values": ["Solid_line", "Dashed_line", "WideDash_line",
                        "Virtual_lane"]},
            {"name": "line_color", "mutable": False, "input_type": "select",
             "default_value": "white",
             "values": ["white", "yellow", "orange", "blue", "other"]},
            {"name": "lane_number", "mutable": False, "input_type": "text",
             "default_value": "", "values": ""},
        ],
    },
    {"name": "stop_line", "type": "polyline", "attributes": []},
    {"name": "zebra_line", "type": "polygon", "attributes": []},
    {"name": "curb_boundary", "type": "polyline",
     "attributes": [{"name": "boundary_id", "mutable": False,
                     "input_type": "text", "default_value": "", "values": ""}]},
    {"name": "fence_boundary", "type": "polyline",
     "attributes": [{"name": "boundary_id", "mutable": False,
                     "input_type": "text", "default_value": "", "values": ""}]},
    {"name": "TrafficCone_boundary", "type": "polyline",
     "attributes": [{"name": "boundary_id", "mutable": False,
                     "input_type": "text", "default_value": "", "values": ""}]},
    {"name": "WaterSafety_boundary", "type": "polyline",
     "attributes": [{"name": "boundary_id", "mutable": False,
                     "input_type": "text", "default_value": "", "values": ""}]},
    {"name": "boundary", "type": "polyline",
     "attributes": [{"name": "boundary_id", "mutable": False,
                     "input_type": "text", "default_value": "", "values": ""}]},
    {"name": "split_point", "type": "points",
     "attributes": [
         {"name": "start_line_ids", "mutable": False, "input_type": "text",
          "default_value": "", "values": ""},
         {"name": "end_line_ids", "mutable": False, "input_type": "text",
          "default_value": "", "values": ""}]},
    {"name": "merge_point", "type": "points",
     "attributes": [
         {"name": "start_line_ids", "mutable": False, "input_type": "text",
          "default_value": "", "values": ""},
         {"name": "end_line_ids", "mutable": False, "input_type": "text",
          "default_value": "", "values": ""}]},
]


@dataclass
class CvatShape:
    label: str
    xyz: np.ndarray
    attributes: dict = field(default_factory=dict)


def write_cvat_xml(path: Path, shapes: Sequence[CvatShape], image_name: str,
                   image_meta: dict) -> dict:
    """CVAT XML 1.1 を書き出す (BevCvatConverter 互換)。"""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta_el = ET.SubElement(root, "meta")
    task_el = ET.SubElement(meta_el, "task")
    ET.SubElement(task_el, "size").text = "1"
    ET.SubElement(task_el, "mode").text = "annotation"
    labels_el = ET.SubElement(task_el, "labels")
    for label_def in CVAT_LABEL_DEFINITIONS:
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = label_def["name"]
        ET.SubElement(label_el, "type").text = label_def["type"]
        for attr_def in label_def.get("attributes", []):
            attr_el = ET.SubElement(label_el, "attribute")
            ET.SubElement(attr_el, "name").text = attr_def["name"]
            ET.SubElement(attr_el, "mutable").text = str(
                attr_def.get("mutable", False)).lower()
            ET.SubElement(attr_el, "input_type").text = attr_def.get(
                "input_type", "select")
            ET.SubElement(attr_el, "default_value").text = attr_def.get(
                "default_value", "")
            values = attr_def.get("values", "")
            if isinstance(values, list):
                values = "\n".join(values)
            ET.SubElement(attr_el, "values").text = values

    image_el = ET.SubElement(root, "image", id="0", name=image_name,
                             width=str(image_meta["width_px"]),
                             height=str(image_meta["height_px"]))
    outside = 0
    total_vertices = 0
    written = 0
    for shape in shapes:
        if shape.xyz is None or len(shape.xyz) < 2:
            continue
        pixels = [meter_to_cvat_pixel(float(p[0]), float(p[1]), image_meta)
                  for p in shape.xyz]
        total_vertices += len(pixels)
        outside += sum(1 for px, py in pixels
                       if px < 0 or py < 0
                       or px > image_meta["width_px"]
                       or py > image_meta["height_px"])
        points = ";".join(f"{px:.2f},{py:.2f}" for px, py in pixels)
        el = ET.SubElement(image_el, "polyline", label=shape.label,
                           occluded="0", source="manual", points=points,
                           z_order="0")
        for name, value in shape.attributes.items():
            ET.SubElement(el, "attribute", name=name).text = str(value)
        written += 1

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode", xml_declaration=False)
    atomic_write_text(
        Path(path), '<?xml version="1.0" encoding="utf-8"?>\n' + body,
        encoding="utf-8")
    return {
        "shape_count": written,
        "outside_vertices": outside,
        "total_vertices": total_vertices,
        "image_name": image_name,
        "image_meta": image_meta,
    }


def extract_image_meta(data: dict) -> dict:
    source = data.get("grid", data)
    required = ("origin_x", "origin_y", "resolution", "width_px", "height_px")
    missing = [k for k in required if k not in source]
    if missing:
        raise ValueError(f"画像メタJSONに必要なキーがありません: {missing}")
    return {
        "origin_x": float(source["origin_x"]),
        "origin_y": float(source["origin_y"]),
        "resolution": float(source["resolution"]),
        "width_px": int(source["width_px"]),
        "height_px": int(source["height_px"]),
    }


def json_safe(obj):
    """numpy 型を含む構造を JSON 化可能な形へ変換する。"""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text atomically in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_dump_json(path: Path, data: dict) -> None:
    """Write JSON atomically in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(json_safe(data), f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def dump_json(path: Path, data: dict) -> None:
    atomic_dump_json(path, data)


def atomic_savez_compressed(path: Path, **arrays) -> None:
    """Write an NPZ atomically so interrupted jobs never leave a partial cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".tmp.npz", dir=str(path.parent)
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
