#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lanegen_stage1_extract.py  --  Stage 1: 抽出 (重い・決定的)

点群と mapping_pose から、Frenet(s,d) 上の証拠グリッドを作って NPZ に保存する。
ここから先 (Stage 2/3) はこの NPZ だけを読むので、閾値調整は秒で回せる。

Stage 1 が行う「土台」の処理:

1. セル集約を max ではなく **上位k平均 (float32)** にする
   -> 8bit packed intensity の量子化ノイズを sqrt(k) 分の 1 に下げる
2. **中央を除外した annulus 背景** を引く
   -> 幅 15cm 程度の細い高反射帯だけが残り、広い明部は消える
3. **s 列ごとのロバスト散布度で正規化** して sigma 値にする
   -> 距離・入射角・走行条件によるベースライン変動を吸収し、
      しきい値がカウント値ではなく「路面比 n シグマ」になる
4. Z は s 列ごとに横断勾配を除去してから、左右の**符号付き**高さ差を出す

出力 NPZ の主なキー (すべて (nd, ns)):
    occupied, count, i_topk, i_bg, i_excess, i_sigma,
    z_surface, z_detrend, z_level_diff, dz_dd, z_occupied,
    rgb_mean(nd,ns,3), rgb_occupied
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

import lanegen_core as core
import lanegen_streaming as streaming


# -----------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IBEV 1パス自動ライン生成 Stage1: Frenet証拠抽出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("intensity_pcd", help="反射強度PCD (IBEV.pcd)")
    p.add_argument("mapping_pose", help="mapping_pose.txt")
    p.add_argument("-o", "--output", required=True,
                   help="出力NPZパス (例 out/IBEV_stage1.npz)")
    p.add_argument("--z-pcd", default=None,
                   help="標高用PCD。省略時は intensity_pcd を使用")
    p.add_argument("--rgb-pcd", default=None,
                   help="色判定用PCD。省略時は z-pcd がカラーなら自動使用")

    g = p.add_argument_group("Frenet グリッド")
    g.add_argument("--s-resolution", type=float, default=0.25,
                   help="進行方向の分解能 [m]")
    g.add_argument("--d-resolution", type=float, default=0.05,
                   help="横方向の分解能 [m]")
    g.add_argument("--corridor-half-width", type=float, default=15.0,
                   help="軌跡からの片側最大距離 [m]")
    g.add_argument("--pose-min-step", type=float, default=0.1)
    g.add_argument("--pose-smooth-window", type=int, default=31)

    g = p.add_argument_group("Frenet 複数候補・曖昧性")
    g.add_argument("--frenet-candidates", type=int, default=12,
                   help="各点で比較する軌跡最近傍候補数")
    g.add_argument("--frenet-z-weight", type=float, default=1.5,
                   help="候補costに加える軌跡Z差の重み")
    g.add_argument("--frenet-heading-weight", type=float, default=0.15,
                   help="mapping_pose yawと軌跡接線の不整合重み")
    g.add_argument("--frenet-ambiguity-margin", type=float, default=0.20,
                   help="遠いs候補とのcost差がこれ以下なら曖昧 [m相当]")
    g.add_argument("--frenet-ambiguity-s-separation", type=float, default=8.0,
                   help="別分岐候補とみなすs離隔 [m]")
    g.add_argument("--keep-ambiguous-frenet", action="store_true",
                   help="交差点等の曖昧点も証拠へ残す（既定は除外）")

    g = p.add_argument_group("反射強度")
    g.add_argument("--surface-band", type=float, default=0.12,
                   help="各セル最小Zからこの範囲だけを路面とみなす [m]")
    g.add_argument("--surface-low-k", type=int, default=3,
                   help="路面基準に使うセル内の下位Z点数")
    g.add_argument("--topk", type=int, default=3,
                   help="セル集約に使う上位点数 (量子化ノイズ低減)")
    g.add_argument("--bg-inner", type=float, default=0.25,
                   help="背景推定で除外する自セル近傍の半幅 [m]")
    g.add_argument("--bg-outer", type=float, default=0.80,
                   help="背景推定に使う近傍の半幅 [m]")
    g.add_argument("--scale-smooth", type=float, default=10.0,
                   help="ロバスト散布度をs方向に平滑化する長さ [m]")
    g.add_argument("--scale-floor", type=float, default=0.5,
                   help="散布度の下限 (0除算と過剰増幅の防止)")
    g.add_argument("--no-intensity-count-debias", dest="intensity_count_debias",
                   action="store_false",
                   help="セル内点数によるtop-k meanの偏りを補正しない (v7互換)")
    g.set_defaults(intensity_count_debias=True)
    g.add_argument("--no-intensity-weighted-background",
                   dest="intensity_weighted_background", action="store_false",
                   help="annulus背景を点数重み付きではなくセル平均で取る (v7互換)")
    g.set_defaults(intensity_weighted_background=True)
    g.add_argument("--intensity-pool-s", type=float, default=1.25,
                   help="超過量算出前にs方向へかける重み付き移動平均長 [m]")
    g.add_argument("--intensity-scale-mode", choices=["local", "row"],
                   default="local",
                   help="散布度を局所窓で取るか、s列全幅で取るか (row=v7互換)")
    g.add_argument("--intensity-scale-d", type=float, default=3.0,
                   help="局所散布度窓の d 方向半幅 [m]")
    g.add_argument("--intensity-scale-s", type=float, default=25.0,
                   help="局所散布度窓の s 方向半幅 [m]")
    g.add_argument("--no-intensity-dark-return",
                   dest="intensity_dark_return", action="store_false",
                   help="路面が観測できているのに反射の返りが無いセルを"
                        "『暗い』観測として扱わず、欠測扱いする (v7互換)")
    g.set_defaults(intensity_dark_return=True)
    g.add_argument("--intensity-empty-percentile", type=float, default=5.0,
                   help="無返りセルに与える反射強度の分位点 [%]")

    g = p.add_argument_group("標高・段差")
    g.add_argument("--z-low-k", type=int, default=3,
                   help="セル標高に使う下位点数")
    g.add_argument("--road-fit-half-width", type=float, default=5.0,
                   help="横断勾配フィットに使う中央部の半幅 [m]")
    g.add_argument("--level-inner", type=float, default=0.20,
                   help="左右レベル比較の内側オフセット [m]")
    g.add_argument("--level-outer", type=float, default=0.80,
                   help="左右レベル比較の外側オフセット [m]")
    g.add_argument("--grad-half-window", type=float, default=0.15,
                   help="dz/dd の中心差分半幅 [m]")

    g = p.add_argument_group("量産I/O")
    g.add_argument("--pcd-chunk-points", type=int, default=500000,
                   help="ASCII/binary PCDのデコード単位 [points]")
    g.add_argument("--aggregation-mode", choices=["streaming", "batch"],
                   default="streaming",
                   help="Stage1集約方式。量産はstreaming、比較検証用にbatchを残す")
    g.add_argument("--pcd-temp-dir", default=None,
                   help="binary_compressed展開用一時memmapディレクトリ")
    g.add_argument("--progress-json", default=None,
                   help="PCD読込進捗を書き出すJSON。GUI監視用")
    g.add_argument("--progress-base", type=float, default=0.0,
                   help="一括ジョブ内でのStage1進捗開始位置")
    g.add_argument("--progress-span", type=float, default=1.0,
                   help="一括ジョブ内でStage1が占める進捗幅")
    g.add_argument("--cancel-file", default=None,
                   help="このファイルが作成されたら安全にキャンセル")

    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------

def _cells(meters: float, resolution: float, minimum: int = 1) -> int:
    return max(minimum, int(round(meters / resolution)))


def _bin_frenet(proj: core.FrenetProjection, nd: int, d_min: float,
                d_resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """(s_index, d) -> flat セル index。範囲外は除外マスクで返す。"""
    d_index = np.floor((proj.d_value - d_min) / d_resolution).astype(np.int64)
    ok = (d_index >= 0) & (d_index < nd)
    return d_index, ok


def _project_points(xyz: np.ndarray, trajectory: core.Trajectory,
                    a: argparse.Namespace) -> core.FrenetProjection:
    """Production V2共通Frenet投影。"""
    return core.project_to_frenet(
        xyz, trajectory, a.corridor_half_width,
        candidates=a.frenet_candidates,
        z_weight=a.frenet_z_weight,
        heading_weight=a.frenet_heading_weight,
        ambiguity_margin=a.frenet_ambiguity_margin,
        ambiguity_s_separation=a.frenet_ambiguity_s_separation,
        drop_ambiguous=not a.keep_ambiguous_frenet,
    )


def _projection_stats(proj: core.FrenetProjection) -> dict:
    within = int(np.asarray(proj.within_corridor_all, dtype=bool).sum())
    ambiguous = int((np.asarray(proj.ambiguous_input, dtype=bool)
                     & np.asarray(proj.within_corridor_all, dtype=bool)).sum())
    return {
        "within_corridor_points_before_ambiguity_filter": within,
        "ambiguous_points": ambiguous,
        "ambiguous_ratio_in_corridor": float(ambiguous / max(within, 1)),
        "accepted_points": int(np.asarray(proj.keep, dtype=bool).sum()),
    }


def intensity_sigma_kwargs(a: argparse.Namespace,
                           surface_observed=None) -> dict:
    """build_intensity_sigma に渡す引数。両経路で同じ値を使うため一箇所に置く。"""
    return dict(
        inner_cells=_cells(a.bg_inner, a.d_resolution),
        outer_cells=_cells(a.bg_outer, a.d_resolution),
        scale_smooth_cells=_cells(a.scale_smooth, a.s_resolution),
        scale_floor=a.scale_floor,
        pool_s_cells=_cells(a.intensity_pool_s, a.s_resolution),
        count_debias=a.intensity_count_debias,
        weighted_background=a.intensity_weighted_background,
        surface_observed=surface_observed,
        empty_return_percentile=a.intensity_empty_percentile,
        scale_mode=a.intensity_scale_mode,
        scale_d_cells=_cells(a.intensity_scale_d, a.d_resolution),
        scale_s_cells=_cells(a.intensity_scale_s, a.s_resolution))


def apply_surface_observation(ev: dict, a: argparse.Namespace) -> None:
    """Z証拠が揃ってから、無返りセルを『暗い』観測として sigma を作り直す。

    反射強度パスの時点では路面が観測できたかどうかが分からないので、Z を
    読んだ後にもう一度グリッド演算だけやり直す。点群は読み直さない。
    """
    if not a.intensity_dark_return or "z_occupied" not in ev:
        return
    evidence = core.build_intensity_sigma(
        ev["i_topk"], ev["count"], ev["_k_used"],
        **intensity_sigma_kwargs(a, surface_observed=ev["z_occupied"]))
    for key in ("i_bg", "i_excess", "i_sigma", "i_scale", "i_scale_s",
                "bg_valid"):
        ev[key] = evidence[key]
    ev["i_count_debias"] = evidence["count_debias_table"]
    stats = ev.setdefault("_intensity_stats", {})
    stats["normalization"] = evidence["diagnostics"]
    stats["bg_valid_cells"] = int(evidence["bg_valid"].sum())


def build_intensity_evidence(xyz: np.ndarray, intensity: np.ndarray,
                             trajectory: core.Trajectory,
                             a: argparse.Namespace) -> dict:
    """反射強度の Frenet 証拠を作る。"""
    nd = int(round(2.0 * a.corridor_half_width / a.d_resolution))
    ns = len(trajectory.s)
    d_min = -a.corridor_half_width
    ncell = nd * ns

    proj = _project_points(xyz, trajectory, a)
    d_index, ok = _bin_frenet(proj, nd, d_min, a.d_resolution)
    d_index = d_index[ok]
    s_index = proj.s_index[ok]
    z = xyz[proj.keep, 2][ok]
    val = intensity[proj.keep][ok]
    flat = d_index * ns + s_index
    print(f"[info] intensity: corridor points {len(flat):,} / {len(xyz):,}")

    # 交差点・Uターン等で別s候補と競合した点の分布。
    amb = proj.ambiguous_input & proj.within_corridor_all
    amb_d = np.floor((proj.best_d_value_all[amb] - d_min)
                     / a.d_resolution).astype(np.int64)
    amb_s = proj.best_s_index_all[amb]
    amb_ok = (amb_d >= 0) & (amb_d < nd)
    amb_flat = amb_d[amb_ok] * ns + amb_s[amb_ok]
    ambiguous_count = np.bincount(amb_flat, minlength=ncell).reshape(nd, ns)
    ambiguous_grid = ambiguous_count > 0

    # --- 路面バンド抽出
    # 単一の最小Zは低い孤立ノイズに弱いので、セル内の下位k点平均を使う。
    z_base, z_base_count = core.groupwise_topk_mean(
        flat, z, ncell, k=max(1, int(a.surface_low_k)), largest=False)
    surface = z <= z_base[flat] + a.surface_band
    flat_s = flat[surface]
    val_s = val[surface]
    print(f"[info] surface band: {len(flat_s):,} points "
          f"({100.0 * len(flat_s) / max(len(flat), 1):.1f}%)")

    # --- 上位k平均 (量子化ノイズ低減の要)
    i_topk, k_count = core.groupwise_topk_mean(flat_s, val_s, ncell,
                                               k=a.topk, largest=True)
    i_topk = i_topk.reshape(nd, ns)
    count = np.bincount(flat_s, minlength=ncell).reshape(nd, ns).astype(np.int32)
    occupied = count > 0

    # --- 点数バイアス補正 -> s方向プール -> annulus 背景 -> 局所スケール
    evidence = core.build_intensity_sigma(
        i_topk, count, k_count.reshape(nd, ns),
        **intensity_sigma_kwargs(a))
    i_bg = evidence["i_bg"]
    i_excess = evidence["i_excess"]
    i_sigma = evidence["i_sigma"]
    bg_valid = evidence["bg_valid"]
    scale = evidence["i_scale_s"]

    valid_vals = i_topk[occupied]
    stats = {
        "corridor_points": int(len(flat)),
        "surface_points": int(len(flat_s)),
        "surface_low_k": int(a.surface_low_k),
        "surface_base_used_mean": float(z_base_count[z_base_count > 0].mean())
            if np.any(z_base_count > 0) else 0.0,
        "occupied_cells": int(occupied.sum()),
        "bg_valid_cells": int(bg_valid.sum()),
        "mean_points_per_cell": float(count[occupied].mean())
            if occupied.any() else 0.0,
        "topk_used_mean": float(k_count[k_count > 0].mean())
            if np.any(k_count > 0) else 0.0,
        "intensity_p50": float(np.percentile(valid_vals, 50))
            if valid_vals.size else 0.0,
        "intensity_p90": float(np.percentile(valid_vals, 90))
            if valid_vals.size else 0.0,
        "intensity_p99": float(np.percentile(valid_vals, 99))
            if valid_vals.size else 0.0,
        "excess_p50": float(np.percentile(i_excess[bg_valid], 50))
            if bg_valid.any() else 0.0,
        "excess_p99": float(np.percentile(i_excess[bg_valid], 99))
            if bg_valid.any() else 0.0,
        "scale_median": float(np.median(scale)),
        "sigma_p99": float(np.percentile(i_sigma[bg_valid], 99))
            if bg_valid.any() else 0.0,
        "sigma_p999": float(np.percentile(i_sigma[bg_valid], 99.9))
            if bg_valid.any() else 0.0,
        "normalization": evidence["diagnostics"],
        "frenet_projection": _projection_stats(proj),
    }
    print(f"[info] intensity p50={stats['intensity_p50']:.1f} "
          f"p99={stats['intensity_p99']:.1f}, "
          f"robust scale={stats['scale_median']:.2f} counts, "
          f"sigma p99={stats['sigma_p99']:.2f}")

    return {
        "occupied": occupied, "count": count, "i_topk": i_topk,
        "i_bg": i_bg,
        "i_excess": i_excess, "i_sigma": i_sigma,
        "i_scale": evidence["i_scale"],
        "i_scale_s": scale, "bg_valid": bg_valid,
        "i_count_debias": evidence["count_debias_table"],
        "_k_used": k_count.reshape(nd, ns),
        "frenet_ambiguous": ambiguous_grid,
        "frenet_ambiguous_count": ambiguous_count.astype(np.int32),
        "_stats": stats,
    }


def build_z_evidence(xyz: np.ndarray, trajectory: core.Trajectory,
                     a: argparse.Namespace) -> dict:
    """標高の Frenet 証拠 (横断勾配除去 + 符号付き段差) を作る。"""
    nd = int(round(2.0 * a.corridor_half_width / a.d_resolution))
    ns = len(trajectory.s)
    d_min = -a.corridor_half_width
    ncell = nd * ns

    proj = _project_points(xyz, trajectory, a)
    d_index, ok = _bin_frenet(proj, nd, d_min, a.d_resolution)
    flat = d_index[ok] * ns + proj.s_index[ok]
    z = xyz[proj.keep, 2][ok]
    print(f"[info] z: corridor points {len(flat):,}")

    z_low, _ = core.groupwise_topk_mean(flat, z, ncell, k=a.z_low_k,
                                        largest=False)
    z_surface = z_low.reshape(nd, ns)
    z_occupied = (np.bincount(flat, minlength=ncell).reshape(nd, ns) > 0)

    # --- s 列ごとに横断勾配 z = alpha*d + beta を最小二乗で除去
    d_axis = (d_min + (np.arange(nd) + 0.5) * a.d_resolution).astype(np.float64)
    fit_mask = z_occupied & (np.abs(d_axis)[:, None] <= a.road_fit_half_width)
    w = fit_mask.astype(np.float64)
    n = w.sum(axis=0)
    sd = (w * d_axis[:, None]).sum(axis=0)
    sdd = (w * (d_axis ** 2)[:, None]).sum(axis=0)
    sz = (w * z_surface).sum(axis=0)
    sdz = (w * d_axis[:, None] * z_surface).sum(axis=0)
    det = n * sdd - sd * sd
    alpha = np.zeros(ns)
    beta = np.zeros(ns)
    good = (n >= 20) & (np.abs(det) > 1e-9)
    alpha[good] = (n[good] * sdz[good] - sd[good] * sz[good]) / det[good]
    beta[good] = (sdd[good] * sz[good] - sd[good] * sdz[good]) / det[good]
    # 未フィット列は近傍から補間
    idx = np.arange(ns)
    if np.any(good):
        alpha = np.interp(idx, idx[good], alpha[good])
        beta = np.interp(idx, idx[good], beta[good])
    z_detrend = (z_surface - (alpha[None, :] * d_axis[:, None] + beta[None, :])
                 ).astype(np.float32)
    z_detrend[~z_occupied] = 0.0

    # --- 左右レベル差 (符号付き)。外側 - 内側。
    inner = _cells(a.level_inner, a.d_resolution)
    outer = _cells(a.level_outer, a.d_resolution)
    sv_hi, sm_hi = core._masked_running_sum(z_detrend, z_occupied,
                                            inner, outer, axis=0)
    sv_lo, sm_lo = core._masked_running_sum(z_detrend, z_occupied,
                                            -outer, -inner, axis=0)
    min_n = max(3, (outer - inner) // 3)
    pair_ok = (sm_hi >= min_n) & (sm_lo >= min_n)
    level_hi = np.divide(sv_hi, np.maximum(sm_hi, 1e-9))
    level_lo = np.divide(sv_lo, np.maximum(sm_lo, 1e-9))
    z_level_diff = np.where(pair_ok, level_hi - level_lo, 0.0).astype(np.float32)

    # --- dz/dd (中心差分)
    h = _cells(a.grad_half_window, a.d_resolution)
    dz_dd = np.zeros((nd, ns), dtype=np.float32)
    up = np.roll(z_detrend, -h, axis=0)
    dn = np.roll(z_detrend, h, axis=0)
    up_ok = np.roll(z_occupied, -h, axis=0)
    dn_ok = np.roll(z_occupied, h, axis=0)
    both = up_ok & dn_ok
    both[:h, :] = False
    both[-h:, :] = False
    dz_dd[both] = ((up - dn)[both] / (2.0 * h * a.d_resolution)).astype(np.float32)

    stats = {
        "corridor_points": int(len(flat)),
        "occupied_cells": int(z_occupied.sum()),
        "cross_slope_median": float(np.median(alpha)),
        "level_diff_p50": float(np.percentile(np.abs(z_level_diff[pair_ok]), 50))
            if pair_ok.any() else 0.0,
        "level_diff_p99": float(np.percentile(np.abs(z_level_diff[pair_ok]), 99))
            if pair_ok.any() else 0.0,
        "grad_p99": float(np.percentile(np.abs(dz_dd[both]), 99))
            if both.any() else 0.0,
        "frenet_projection": _projection_stats(proj),
    }
    print(f"[info] cross slope median={stats['cross_slope_median'] * 100:.2f}%, "
          f"|level diff| p99={stats['level_diff_p99']:.3f} m")

    return {
        "z_surface": z_surface, "z_detrend": z_detrend,
        "z_level_diff": z_level_diff, "dz_dd": dz_dd,
        "z_occupied": z_occupied, "z_pair_ok": pair_ok,
        "_stats": stats,
    }


def build_rgb_evidence(xyz: np.ndarray, rgb: np.ndarray,
                       trajectory: core.Trajectory,
                       a: argparse.Namespace) -> dict:
    """色の Frenet 証拠。あくまで confidence 用で、主判定には使わない。"""
    nd = int(round(2.0 * a.corridor_half_width / a.d_resolution))
    ns = len(trajectory.s)
    d_min = -a.corridor_half_width
    ncell = nd * ns

    proj = _project_points(xyz, trajectory, a)
    d_index, ok = _bin_frenet(proj, nd, d_min, a.d_resolution)
    flat = d_index[ok] * ns + proj.s_index[ok]
    cols = rgb[proj.keep][ok]
    z = xyz[proj.keep, 2][ok]

    zmin = np.full(ncell, np.inf, dtype=np.float32)
    np.minimum.at(zmin, flat, z)
    surface = z <= zmin[flat] + a.surface_band
    flat = flat[surface]
    cols = cols[surface]

    rgb_mean = np.zeros((nd, ns, 3), dtype=np.uint8)
    for c in range(3):
        m, _ = core.groupwise_mean(flat, cols[:, c].astype(np.float32), ncell)
        rgb_mean[:, :, c] = np.clip(m.reshape(nd, ns), 0, 255).astype(np.uint8)
    rgb_occupied = (np.bincount(flat, minlength=ncell).reshape(nd, ns) > 0)

    rf = rgb_mean.astype(np.float32)
    luminance = 0.299 * rf[..., 0] + 0.587 * rf[..., 1] + 0.114 * rf[..., 2]
    mx = rf.max(axis=2)
    mn = rf.min(axis=2)
    saturation = np.divide(mx - mn, np.maximum(mx, 1e-6))

    stats = {
        "occupied_cells": int(rgb_occupied.sum()),
        "valid_ratio": float(rgb_occupied.mean()),
        "luminance_p50": float(np.percentile(luminance[rgb_occupied], 50))
            if rgb_occupied.any() else 0.0,
        "luminance_p99": float(np.percentile(luminance[rgb_occupied], 99))
            if rgb_occupied.any() else 0.0,
        "frenet_projection": _projection_stats(proj),
    }
    print(f"[info] rgb: valid ratio={stats['valid_ratio']:.3f}, "
          f"luminance p50={stats['luminance_p50']:.1f}")
    return {
        "rgb_mean": rgb_mean, "rgb_occupied": rgb_occupied,
        "rgb_luminance": luminance.astype(np.float32),
        "rgb_saturation": saturation.astype(np.float32),
        "_stats": stats,
    }


# -----------------------------------------------------------------------------

def _cancel_requested(a: argparse.Namespace) -> bool:
    path = getattr(a, "cancel_file", None)
    return bool(path and Path(path).exists())


def _progress_callback(a: argparse.Namespace, source: str):
    progress_path = getattr(a, "progress_json", None)
    if not progress_path:
        return None
    progress_path = Path(progress_path)

    last_write = [0.0]

    def cb(done: int, total: int, phase: str) -> None:
        # Progress is advisory and high-frequency.  Do not fsync on network/FUSE
        # workspaces because it can stall the PCD read loop for minutes.
        now = time.time()
        if done < total and (now - last_write[0]) < 0.20:
            return
        payload = {
            "stage": 1,
            "source": source,
            "phase": phase,
            "points_done": int(done),
            "points_total": int(total),
            "fraction": float(
                max(0.0, min(1.0,
                    float(a.progress_base)
                    + float(a.progress_span) * (done / max(total, 1))
                ))
            ),
            "updated_unix": now,
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_path.with_name(progress_path.name + ".progress.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, progress_path)
        last_write[0] = now
    return cb


def _load_pcd(path: Path, a: argparse.Namespace, source: str):
    return core.load_pcd(
        path,
        chunk_points=max(1, int(a.pcd_chunk_points)),
        progress_callback=_progress_callback(a, source),
        cancel_callback=lambda: _cancel_requested(a),
    )


def run_batch(a: argparse.Namespace) -> dict:
    t0 = time.time()
    out_path = Path(a.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xyz_i, rgb_i, native_i, info_i = _load_pcd(Path(a.intensity_pcd), a, "intensity_pcd")
    if native_i is not None:
        intensity = native_i.astype(np.float32)
        intensity_stats = {
            "decode_mode": "native_intensity_field",
            "p50": float(np.percentile(intensity, 50)),
            "p99": float(np.percentile(intensity, 99)),
            "max": float(intensity.max()),
        }
    elif rgb_i is not None:
        intensity, intensity_stats = core.decode_intensity(rgb_i)
    else:
        raise ValueError("反射強度に使えるフィールドがありません")

    pose = core.load_mapping_pose(Path(a.mapping_pose))
    trajectory = core.build_trajectory(pose, a.pose_min_step,
                                       a.pose_smooth_window, a.s_resolution)

    ev = build_intensity_evidence(xyz_i, intensity, trajectory, a)
    meta = {
        "stage": 1,
        "production_version": core.PRODUCTION_VERSION,
        "code": {
            "stage1_sha256": core.sha256_file(Path(__file__)),
            "core_sha256": core.sha256_file(Path(core.__file__)),
        },
        "inputs": {
            "intensity_pcd": str(Path(a.intensity_pcd).resolve()),
            "mapping_pose": str(Path(a.mapping_pose).resolve()),
            "signatures": {
                "intensity_pcd": core.file_signature(Path(a.intensity_pcd)),
                "mapping_pose": core.file_signature(Path(a.mapping_pose)),
            },
            "intensity_pcd_info": info_i,
            "intensity_decode": intensity_stats,
            "pcd_io": {
                "reader": "lanegen_pcd_io",
                "chunk_points": int(a.pcd_chunk_points),
                "supported_data_modes": ["ascii", "binary", "binary_compressed"],
            },
        },
        "trajectory": {
            "raw_count": trajectory.raw_count,
            "filtered_count": trajectory.filtered_count,
            "dense_count": int(len(trajectory.s)),
            "length_m": trajectory.length_m,
            "yaw_valid": bool(trajectory.yaw_valid),
            "heading_error_deg_p95": float(np.degrees(
                np.percentile(trajectory.heading_error, 95)))
                if len(trajectory.heading_error) else 0.0,
        },
        "intensity": ev.pop("_stats"),
        "grid": {
            "nd": int(ev["occupied"].shape[0]),
            "ns": int(ev["occupied"].shape[1]),
            "d_min": -a.corridor_half_width,
            "d_resolution": a.d_resolution,
            "s_resolution": a.s_resolution,
        },
        "world_bbox": {
            "x_min": float(xyz_i[:, 0].min()), "x_max": float(xyz_i[:, 0].max()),
            "y_min": float(xyz_i[:, 1].min()), "y_max": float(xyz_i[:, 1].max()),
        },
        "parameters": vars(a),
        "parameter_fingerprint": core.parameter_fingerprint(
            vars(a), exclude=(
                "intensity_pcd", "mapping_pose", "output", "z_pcd",
                "rgb_pcd", "pcd_chunk_points", "progress_json",
                "cancel_file", "pcd_temp_dir", "quiet",
                "aggregation_mode",
            )
        ),
    }

    # --- Z
    z_path = Path(a.z_pcd) if a.z_pcd else Path(a.intensity_pcd)
    if a.z_pcd:
        xyz_z, rgb_z, _, info_z = _load_pcd(z_path, a, "z_pcd")
        meta["inputs"]["z_pcd"] = str(z_path.resolve())
        meta["inputs"]["z_pcd_info"] = info_z
        meta["inputs"]["signatures"]["z_pcd"] = core.file_signature(z_path)
    else:
        xyz_z, rgb_z = xyz_i, rgb_i
        meta["inputs"]["z_pcd"] = str(z_path.resolve()) + " (= intensity_pcd)"
        meta["inputs"]["signatures"]["z_pcd"] = meta["inputs"]["signatures"]["intensity_pcd"]
    zev = build_z_evidence(xyz_z, trajectory, a)
    meta["z"] = zev.pop("_stats")
    ev.update(zev)
    apply_surface_observation(ev, a)
    meta["intensity"].update(ev.pop("_intensity_stats", {}))

    # --- RGB (任意)
    rgb_source = None
    if a.rgb_pcd:
        xyz_c, rgb_c, _, info_c = _load_pcd(Path(a.rgb_pcd), a, "rgb_pcd")
        meta["inputs"]["signatures"]["rgb_pcd"] = core.file_signature(Path(a.rgb_pcd))
        if rgb_c is None:
            print("[warn] --rgb-pcd に rgb フィールドがありません。色はスキップ")
        else:
            rgb_source = (xyz_c, core.decode_rgb_channels(rgb_c),
                          str(Path(a.rgb_pcd).resolve()))
    elif rgb_z is not None:
        ch = core.decode_rgb_channels(rgb_z)
        gray = float(np.mean((ch[:, 0] == ch[:, 1]) & (ch[:, 1] == ch[:, 2])))
        if gray < 0.999:
            rgb_source = (xyz_z, ch, str(z_path.resolve()))
        else:
            print("[info] z-pcd の rgb はグレースケールのため色判定はスキップ")

    if rgb_source is not None:
        xyz_c, ch, src = rgb_source
        cev = build_rgb_evidence(xyz_c, ch, trajectory, a)
        meta["rgb"] = cev.pop("_stats")
        meta["rgb"]["source"] = src
        ev.update(cev)
    else:
        meta["rgb"] = None

    arrays = {k: v for k, v in ev.items()
              if isinstance(v, np.ndarray) and not k.startswith("_")}
    arrays["traj_xy"] = trajectory.xy.astype(np.float32)
    arrays["traj_z"] = trajectory.z.astype(np.float32)
    arrays["traj_normal"] = trajectory.normal.astype(np.float32)
    arrays["traj_tangent"] = trajectory.tangent.astype(np.float32)
    arrays["traj_s"] = trajectory.s.astype(np.float32)
    arrays["traj_yaw"] = trajectory.yaw.astype(np.float32)
    arrays["traj_geometric_yaw"] = trajectory.geometric_yaw.astype(np.float32)
    arrays["traj_heading_error"] = trajectory.heading_error.astype(np.float32)
    core.atomic_savez_compressed(out_path, **arrays)

    meta["elapsed_sec"] = round(time.time() - t0, 2)
    meta["output_npz"] = str(out_path)
    meta_path = out_path.with_suffix(".meta.json")
    core.dump_json(meta_path, meta)
    if getattr(a, "progress_json", None):
        core.atomic_dump_json(Path(a.progress_json), {
            "stage": 1, "phase": "complete",
            "fraction": float(max(0.0, min(1.0,
                float(a.progress_base) + float(a.progress_span)))),
            "output": str(out_path), "updated_unix": time.time(),
        })
    print(f"[done] Stage1 -> {out_path}  ({meta['elapsed_sec']}s)")
    print(f"[done] meta   -> {meta_path}")
    return meta


def _streaming_reader(
    stack: ExitStack,
    path: Path,
    a: argparse.Namespace,
    source: str,
    cache: dict[Path, core.PcdChunkReader],
) -> core.PcdChunkReader:
    resolved = Path(path).expanduser().resolve()
    if resolved in cache:
        return cache[resolved]
    temp_dir = Path(a.pcd_temp_dir).expanduser().resolve() if a.pcd_temp_dir else None
    reader = core.PcdChunkReader(
        resolved,
        chunk_points=max(1, int(a.pcd_chunk_points)),
        progress_callback=_progress_callback(a, source),
        cancel_callback=lambda: _cancel_requested(a),
        temp_dir=temp_dir,
    )
    stack.enter_context(reader)
    cache[resolved] = reader
    return reader


def run_streaming(a: argparse.Namespace) -> dict:
    """Production V6 fully streaming Stage1.

    Only the current PCD chunk and O(Frenet-grid-size) accumulators are held in
    memory.  Complete XYZ/RGB/intensity arrays are never concatenated.
    """
    t0 = time.time()
    out_path = Path(a.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pose_path = Path(a.mapping_pose).expanduser().resolve()
    pose = core.load_mapping_pose(pose_path)
    trajectory = core.build_trajectory(
        pose, a.pose_min_step, a.pose_smooth_window, a.s_resolution
    )

    intensity_path = Path(a.intensity_pcd).expanduser().resolve()
    z_path = Path(a.z_pcd).expanduser().resolve() if a.z_pcd else intensity_path
    rgb_path = Path(a.rgb_pcd).expanduser().resolve() if a.rgb_pcd else z_path

    with ExitStack() as stack:
        readers: dict[Path, core.PcdChunkReader] = {}
        intensity_reader = _streaming_reader(
            stack, intensity_path, a, "intensity_pcd", readers
        )
        intensity_mode, intensity_decode = streaming.detect_intensity_mode(
            intensity_reader
        )
        ev, bbox = streaming.build_intensity_streaming(
            intensity_reader, trajectory, a, intensity_mode
        )

        info_i = dict(intensity_reader.info)
        info_i["valid_points"] = int(bbox.count)
        meta = {
            "stage": 1,
            "production_version": core.PRODUCTION_VERSION,
            "aggregation": {
                "mode": "fully_streaming_grid",
                "full_point_arrays_materialized": False,
                "point_memory_complexity": "O(chunk_points)",
                "grid_memory_complexity": "O(nd*ns*k)",
            },
            "code": {
                "stage1_sha256": core.sha256_file(Path(__file__)),
                "core_sha256": core.sha256_file(Path(core.__file__)),
                "streaming_sha256": core.sha256_file(Path(streaming.__file__)),
            },
            "inputs": {
                "intensity_pcd": str(intensity_path),
                "mapping_pose": str(pose_path),
                "signatures": {
                    "intensity_pcd": core.file_signature(intensity_path),
                    "mapping_pose": core.file_signature(pose_path),
                },
                "intensity_pcd_info": info_i,
                "intensity_decode": intensity_decode,
                "pcd_io": {
                    "reader": "PcdChunkReader",
                    "chunk_points": int(a.pcd_chunk_points),
                    "supported_data_modes": ["ascii", "binary", "binary_compressed"],
                    "binary_compressed_strategy": "temporary_memmap_reused_across_passes",
                    "temp_dir": str(Path(a.pcd_temp_dir).resolve()) if a.pcd_temp_dir else None,
                },
            },
            "trajectory": {
                "raw_count": trajectory.raw_count,
                "filtered_count": trajectory.filtered_count,
                "dense_count": int(len(trajectory.s)),
                "length_m": trajectory.length_m,
                "yaw_valid": bool(trajectory.yaw_valid),
                "heading_error_deg_p95": float(
                    np.degrees(np.percentile(trajectory.heading_error, 95))
                ) if len(trajectory.heading_error) else 0.0,
            },
            "intensity": ev.pop("_stats"),
            "grid": {
                "nd": int(ev["occupied"].shape[0]),
                "ns": int(ev["occupied"].shape[1]),
                "d_min": -a.corridor_half_width,
                "d_resolution": a.d_resolution,
                "s_resolution": a.s_resolution,
            },
            "world_bbox": bbox.world_xy_dict(),
            "parameters": vars(a),
        "parameter_fingerprint": core.parameter_fingerprint(
            vars(a), exclude=(
                "intensity_pcd", "mapping_pose", "output", "z_pcd",
                "rgb_pcd", "pcd_chunk_points", "progress_json",
                "cancel_file", "pcd_temp_dir", "quiet",
                "aggregation_mode",
            )
        ),
        }

        z_reader = _streaming_reader(stack, z_path, a, "z_pcd", readers)
        meta["inputs"]["z_pcd"] = (
            str(z_path) if a.z_pcd else str(z_path) + " (= intensity_pcd)"
        )
        meta["inputs"]["z_pcd_info"] = dict(z_reader.info)
        meta["inputs"]["signatures"]["z_pcd"] = (
            meta["inputs"]["signatures"]["intensity_pcd"]
            if z_path == intensity_path
            else core.file_signature(z_path)
        )
        zev = streaming.build_z_streaming(z_reader, trajectory, a)
        meta["z"] = zev.pop("_stats")
        ev.update(zev)
        apply_surface_observation(ev, a)
        meta["intensity"].update(ev.pop("_intensity_stats", {}))

        # RGB is optional. Explicit --rgb-pcd wins; otherwise use z-pcd only
        # when it contains non-grayscale colour.
        rgb_reader = _streaming_reader(stack, rgb_path, a, "rgb_pcd", readers)
        has_rgb = bool(rgb_reader.info.get("has_rgb"))
        gray_ratio = 1.0
        if has_rgb:
            _, gray_ratio = streaming.inspect_rgb(rgb_reader)
        use_rgb = has_rgb and (bool(a.rgb_pcd) or gray_ratio < 0.999)
        if use_rgb:
            cev = streaming.build_rgb_streaming(rgb_reader, trajectory, a)
            meta["rgb"] = cev.pop("_stats")
            meta["rgb"]["source"] = str(rgb_path)
            meta["rgb"]["grayscale_ratio"] = float(gray_ratio)
            meta["inputs"]["rgb_pcd"] = str(rgb_path)
            meta["inputs"]["rgb_pcd_info"] = dict(rgb_reader.info)
            meta["inputs"]["signatures"]["rgb_pcd"] = (
                meta["inputs"]["signatures"]["z_pcd"]
                if rgb_path == z_path
                else core.file_signature(rgb_path)
            )
            ev.update(cev)
        else:
            if a.rgb_pcd and not has_rgb:
                print("[warn] --rgb-pcd に rgb/rgba フィールドがありません。色はスキップ")
            elif has_rgb:
                print("[info] RGB PCDはグレースケールのため色判定はスキップ")
            meta["rgb"] = None

    arrays = {k: v for k, v in ev.items()
              if isinstance(v, np.ndarray) and not k.startswith("_")}
    arrays["traj_xy"] = trajectory.xy.astype(np.float32)
    arrays["traj_z"] = trajectory.z.astype(np.float32)
    arrays["traj_normal"] = trajectory.normal.astype(np.float32)
    arrays["traj_tangent"] = trajectory.tangent.astype(np.float32)
    arrays["traj_s"] = trajectory.s.astype(np.float32)
    arrays["traj_yaw"] = trajectory.yaw.astype(np.float32)
    arrays["traj_geometric_yaw"] = trajectory.geometric_yaw.astype(np.float32)
    arrays["traj_heading_error"] = trajectory.heading_error.astype(np.float32)
    core.atomic_savez_compressed(out_path, **arrays)

    meta["elapsed_sec"] = round(time.time() - t0, 2)
    meta["output_npz"] = str(out_path)
    meta_path = out_path.with_suffix(".meta.json")
    core.atomic_dump_json(meta_path, meta)
    if getattr(a, "progress_json", None):
        core.atomic_dump_json(Path(a.progress_json), {
            "stage": 1,
            "phase": "complete",
            "fraction": float(max(0.0, min(1.0,
                float(a.progress_base) + float(a.progress_span)))),
            "output": str(out_path),
            "updated_unix": time.time(),
        })
    print(f"[done] Stage1 streaming -> {out_path}  ({meta['elapsed_sec']}s)")
    print(f"[done] meta             -> {meta_path}")
    return meta


def run(a: argparse.Namespace) -> dict:
    if getattr(a, "aggregation_mode", "streaming") == "batch":
        return run_batch(a)
    return run_streaming(a)


def main(argv=None) -> int:
    a = parse_args(argv)
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
