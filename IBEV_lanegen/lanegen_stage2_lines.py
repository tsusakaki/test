#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lanegen_stage2_lines.py  --  Stage 2: 対応付け (軽い)

Stage1 の NPZ だけを読み、線を組み立てる。点群には触れないので数秒で回る。

方式の要点:

* 画像の大きな Closing と細線化トレースは使わない。
  代わりに **塗装片をオブジェクトとして検出し、グラフで対応付ける**。
* 対応付けは短ギャップ (破線の空白) と長ギャップ (オクルージョン) を
  同じ枠組みで扱う。実データで 25〜70m の分断が観測されているため、
  破線周期だけを想定した接続では足りない。
* 縁石は **符号付き左右レベル差** で検出する。大きさだけの勾配では
  壁・法面・側溝と区別できない。
* 反射リッジと段差リッジは独立に追跡し、どちらかを他方へ吸収しない。
  近接している場合だけ重複を落とす。
* レーン構造 (間隔 3.0〜3.5m) を事前知識として使い、
  - 間隔 2.5m 未満の群 -> 導流帯・横断歩道として除外
  - 間隔 5.5〜7.5m の空き -> その帯だけ閾値を下げて再探索
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter

import lanegen_core as core

KIND_LANE, KIND_CURB, KIND_HATCH = 0, 1, 2
KIND_NAME = {KIND_LANE: "lane", KIND_CURB: "curb", KIND_HATCH: "hatch"}


# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IBEV 1パス自動ライン生成 Stage2: 塗装片グラフと段差追跡",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("stage1_npz")
    p.add_argument("-o", "--output", required=True,
                   help="出力NPZ (例 out/IBEV_stage2.npz)")

    g = p.add_argument_group("反射リッジ")
    g.add_argument("--sigma-threshold", type=float, default=2.5,
                   help="路面比 何シグマ以上を塗装候補とするか")
    g.add_argument("--max-paint-width", type=float, default=0.50,
                   help="塗装帯として許容する最大横幅 [m]")
    g.add_argument("--min-paint-width", type=float, default=0.05,
                   help="塗装帯として許容する最小横幅 [m]")

    g = p.add_argument_group("RGB局所コントラストによるLane補完")
    g.add_argument("--rgb-lane-fusion", choices=["auto", "on", "off"],
                   default="auto",
                   help="反射強度で不足するLaneをRGBBEVの局所コントラストで補完")
    g.add_argument("--rgb-auto-min-span-ratio", type=float, default=0.80,
                   help="自動モードで、検出Lane総延長/軌跡長がこの値未満ならRGB補完")
    g.add_argument("--rgb-auto-min-tracks-per-100m", type=float, default=1.0,
                   help="自動モードで、100m当たりLane本数がこの値未満ならRGB補完")
    g.add_argument("--rgb-auto-min-coverage", type=float, default=0.75,
                   help="自動モードで、Lane被覆率の中央値がこの値未満ならRGB補完")
    g.add_argument("--rgb-auto-min-on-length", type=float, default=1.5,
                   help="自動モードで、ON区間長の中央値がこの値[m]未満ならRGB補完")
    g.add_argument("--rgb-scale-mode", choices=["local", "row"],
                   default="local",
                   help="RGBコントラストの散布度を局所窓で取るか s列全幅か")
    g.add_argument("--rgb-scale-d", type=float, default=3.0,
                   help="RGB局所散布度窓の d 方向半幅 [m]")
    g.add_argument("--rgb-scale-s", type=float, default=25.0,
                   help="RGB局所散布度窓の s 方向半幅 [m]")
    g.add_argument("--rgb-sigma-threshold", type=float, default=3.0,
                   help="局所背景に対するRGB輝度コントラスト閾値 [sigma]")
    g.add_argument("--rgb-strong-sigma-threshold", type=float, default=4.5,
                   help="強いRGB証拠として単独採用する閾値 [sigma]")
    g.add_argument("--rgb-min-luminance", type=float, default=110.0,
                   help="白・黄ペイント候補の最小輝度")
    g.add_argument("--rgb-max-white-saturation", type=float, default=0.22,
                   help="白線候補の最大彩度")
    g.add_argument("--rgb-yellow-saturation-min", type=float, default=0.10,
                   help="黄線候補の最小彩度")
    g.add_argument("--rgb-yellow-chroma-min", type=float, default=12.0,
                   help="黄線候補の min(R,G)-B 最小値")
    g.add_argument("--rgb-min-local-contrast", type=float, default=10.0,
                   help="周辺路面より明るい必要量 [0-255]")
    g.add_argument("--rgb-bg-inner", type=float, default=0.25,
                   help="局所背景算出で除外する内側半幅 [m]")
    g.add_argument("--rgb-bg-outer", type=float, default=0.80,
                   help="局所背景算出の外側半幅 [m]")
    g.add_argument("--rgb-scale-smooth", type=float, default=10.0,
                   help="RGBコントラストスケールのs方向平滑化長 [m]")
    g.add_argument("--rgb-scale-floor", type=float, default=1.0,
                   help="RGBロバストスケール下限")
    g.add_argument("--rgb-intensity-support-sigma", type=float, default=0.20,
                   help="通常RGB候補に要求する近傍反射強度証拠")
    g.add_argument("--rgb-intensity-support-d", type=float, default=0.10,
                   help="反射強度近傍支持を探す横半幅 [m]")
    g.add_argument("--rgb-intensity-support-s", type=float, default=0.50,
                   help="反射強度近傍支持を探す進行方向半幅 [m]")
    g.add_argument("--rgb-z-edge-dilation", type=float, default=0.05,
                   help="縁石段差からこの横距離以内のRGB候補をLaneから除外 [m]")
    g.add_argument("--rgb-max-paint-width", type=float, default=0.35,
                   help="RGB候補として許容する最大横幅 [m]")
    g.add_argument("--rgb-min-track-length", type=float, default=8.0,
                   help="RGB補完Laneとして採用する最小全長 [m]")

    g = p.add_argument_group("塗装片")
    g.add_argument("--link-d", type=float, default=0.15,
                   help="同一塗装片とみなす s 隣接間の横ずれ上限 [m]")
    g.add_argument("--segment-tiny-gap", type=float, default=0.75,
                   help="塗装片内部で許す欠測長 [m]")
    g.add_argument("--min-segment-length", type=float, default=0.5,
                   help="塗装片として採用する最小長 [m]")
    g.add_argument("--assignment-method", choices=["hungarian", "greedy"],
                   default="hungarian",
                   help="同一s列の既存片と新規リッジの対応方法")
    g.add_argument("--curb-assignment-method",
                   choices=["hungarian", "greedy"], default="greedy",
                   help="縁石リッジの対応方法。ノイズ片が多いため既定はGreedy")

    g = p.add_argument_group("グラフ対応付け")
    g.add_argument("--max-gap", type=float, default=90.0,
                   help="接続を許す最大空白長 [m]")
    g.add_argument("--d-tolerance", type=float, default=0.30,
                   help="接続時の横位置許容差の基本値 [m]")
    g.add_argument("--d-tolerance-rate", type=float, default=0.004,
                   help="空白長 1m あたりに緩める横位置許容量 [m/m]")
    g.add_argument("--max-slope-diff", type=float, default=0.27,
                   help="接続を許す傾き差 (dd/ds), 0.27 = 約15度")
    g.add_argument("--gap-penalty", type=float, default=0.08,
                   help="空白 1m あたりのスコア減点")
    g.add_argument("--min-track-length", type=float, default=8.0,
                   help="トラックとして採用する最小全長 [m]")
    g.add_argument("--long-gap-threshold", type=float, default=15.0,
                   help="これ以上を長距離接続として厳格判定 [m]")
    g.add_argument("--long-gap-d-tolerance-scale", type=float, default=0.65,
                   help="長距離接続時の横位置許容差倍率")
    g.add_argument("--long-gap-max-slope-diff", type=float, default=0.12,
                   help="長距離接続時の傾き差上限 dd/ds")
    g.add_argument("--bridge-crossing-tolerance", type=float, default=0.10,
                   help="接続補間線と他の検出片の交差判定余裕 [m]")
    g.add_argument("--allow-bridge-crossing", action="store_true",
                   help="他の検出片を横切る長距離接続も許可する")
    g.add_argument("--curb-strict-long-gap", action="store_true",
                   help="縁石にもLaneと同じ厳格な長距離接続制約を適用")

    g = p.add_argument_group("線種判定")
    g.add_argument("--solid-coverage", type=float, default=0.75)
    g.add_argument("--dash-close-gap", type=float, default=1.25,
                   help="ラン長を測る前に埋める検出途切れの最大長 [m]")
    g.add_argument("--solid-max-run-ratio", type=float, default=0.6,
                   help="最長ON区間が全長のこの割合以上なら実線と判定")
    g.add_argument("--dash-regularity", type=float, default=0.6,
                   help="破線と認める ON長・周期 の変動係数の上限")
    g.add_argument("--dash-min-runs", type=int, default=3)
    g.add_argument("--dash-on-min", type=float, default=2.0,
                   help="破線ダッシュとして認める最小ON長 [m]")
    g.add_argument("--dash-on-max", type=float, default=12.0)
    g.add_argument("--dash-off-min", type=float, default=2.0,
                   help="破線の空白として認める最小OFF長 [m]")
    g.add_argument("--dash-off-max", type=float, default=15.0)

    g = p.add_argument_group("レーン構造")
    g.add_argument("--hatch-max-spacing", type=float, default=2.5,
                   help="この間隔未満で3本以上並ぶ群は車線境界線ではない [m]")
    g.add_argument("--hatch-min-members", type=int, default=3)
    g.add_argument("--hatch-min-overlap", type=float, default=4.0,
                   help="群と判定する共通区間長 [m]")
    g.add_argument("--hatch-self-overlap", type=float, default=0.70,
                   help="近接線との重なりが自分の何割以上なら群とみなすか")
    g.add_argument("--hatch-max-length", type=float, default=60.0,
                   help="導流帯・横断歩道として扱う最大全長 [m]")
    g.add_argument("--infill-min-spacing", type=float, default=5.5,
                   help="穴埋め再探索を起動する下限間隔 [m]")
    g.add_argument("--infill-max-spacing", type=float, default=7.6,
                   help="穴埋め再探索を起動する上限間隔 [m]")
    g.add_argument("--infill-min-overlap", type=float, default=20.0,
                   help="穴埋めを検討する共通区間長 [m]")
    g.add_argument("--infill-band", type=float, default=0.40,
                   help="穴埋め再探索の横方向探索半幅 [m]")
    g.add_argument("--infill-sigma-ratio", type=float, default=0.55,
                   help="穴埋め時にしきい値へ掛ける係数")
    g.add_argument("--infill-min-coverage", type=float, default=0.25,
                   help="穴埋め採用に必要な証拠被覆率")
    g.add_argument("--infill-max-d-std", type=float, default=0.12,
                   help="穴埋め証拠の横位置ばらつき上限 [m] (ノイズ棄却)")
    g.add_argument("--infill-emit-virtual", action="store_true",
                   help="証拠が無くても幾何的に Virtual_lane を出す")
    g.add_argument("--duplicate-d", type=float, default=0.25,
                   help="重複とみなす横距離 [m]")

    g = p.add_argument_group("縁石")
    g.add_argument("--curb-step-min", type=float, default=0.08,
                   help="縁石とみなす段差の下限 [m]")
    g.add_argument("--curb-step-max", type=float, default=0.30,
                   help="縁石とみなす段差の上限 [m] (超過は擁壁・法面)")
    g.add_argument("--curb-grad-min", type=float, default=0.10,
                   help="段差エッジとみなす |dz/dd| の下限")
    g.add_argument("--curb-max-width", type=float, default=0.60,
                   help="段差エッジの最大横幅 [m]")
    g.add_argument("--curb-min-length", type=float, default=8.0,
                   help="縁石として採用する最小全長 [m]")
    g.add_argument("--curb-sign-consistency", type=float, default=0.80,
                   help="段差符号が一致すべき割合")
    g.add_argument("--curb-duplicate-d", type=float, default=0.50,
                   help="縁石同士を重複とみなす横距離 [m]")
    g.add_argument("--curb-min-coverage", type=float, default=0.30,
                   help="縁石として採用する最小の証拠被覆率")
    g.add_argument("--curb-lane-duplicate-d", type=float, default=0.18,
                   help="車線線と同一とみなす横距離 [m]")

    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# =============================================================================
# RGB局所コントラスト証拠
# =============================================================================

def build_rgb_lane_evidence(npz, d_res: float, s_res: float,
                            a: argparse.Namespace
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    RGBBEVの絶対的な「白画素率」ではなく、横方向の局所路面背景との差から
    白線・黄線候補を作る。

    灰色舗装全体が明るいシーンでも、細いペイントだけを周辺路面との
    コントラストで抽出する。通常候補には近傍の弱い反射強度支持を要求し、
    非常に強いRGB候補だけはRGB単独で採用する。Z段差エッジ近傍は縁石との
    二重検出を避けるため除外する。
    """
    required = ("rgb_occupied", "rgb_luminance", "rgb_saturation", "rgb_mean")
    if not all(k in npz.files for k in required):
        shape = npz["i_sigma"].shape
        return (np.zeros(shape, dtype=bool),
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=bool),
                {"available": False, "reason": "RGB evidence arrays not found"})

    occ = npz["rgb_occupied"].astype(bool)
    lum = npz["rgb_luminance"].astype(np.float32)
    sat = npz["rgb_saturation"].astype(np.float32)
    rgb = npz["rgb_mean"].astype(np.float32)

    inner = max(1, int(round(a.rgb_bg_inner / d_res)))
    outer = max(inner + 1, int(round(a.rgb_bg_outer / d_res)))
    bg, bg_count = core.annulus_background(
        lum, occ, inner_cells=inner, outer_cells=outer, axis=0)
    valid = occ & np.isfinite(bg) & (bg_count >= 4)
    excess = np.where(valid, lum - bg, 0.0).astype(np.float32)
    if a.rgb_scale_mode == "local":
        scale = core.local_robust_scale(
            excess, valid,
            d_cells=max(1, int(round(a.rgb_scale_d / d_res))),
            s_cells=max(1, int(round(a.rgb_scale_s / s_res))),
            floor_value=a.rgb_scale_floor)
    else:
        smooth_cells = max(1, int(round(a.rgb_scale_smooth / s_res)))
        scale = core.row_robust_scale(
            excess, valid, smooth_cells=smooth_cells,
            floor_value=a.rgb_scale_floor)[None, :]
    sigma = np.where(valid, excess / scale, 0.0).astype(np.float32)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    white = ((sat <= a.rgb_max_white_saturation)
             & (lum >= a.rgb_min_luminance))
    yellow = ((sat >= a.rgb_yellow_saturation_min)
              & (r >= a.rgb_min_luminance * 0.75)
              & (g >= a.rgb_min_luminance * 0.65)
              & ((np.minimum(r, g) - b) >= a.rgb_yellow_chroma_min))
    color_ok = white | yellow

    d_half = max(0, int(round(a.rgb_intensity_support_d / d_res)))
    s_half = max(0, int(round(a.rgb_intensity_support_s / s_res)))
    support_size = (2 * d_half + 1, 2 * s_half + 1)
    intensity_near = maximum_filter(
        npz["i_sigma"].astype(np.float32),
        size=support_size, mode="nearest")
    weak_intensity_support = intensity_near >= a.rgb_intensity_support_sigma

    # 反射強度の有効セルが近傍に1つも無い場所で「反射強度の支持」を要求すると
    # 循環論法になる。反射強度点群が疎なクリップ (IBEV.pcd が RGBBEV.pcd の
    # 1/10 の点数しか無い等) では、線の位置の s列被覆が 34〜37% しかなく、
    # 残り 2/3 は支持を返しようがない。そこは支持要求を外す。
    intensity_observed = maximum_filter(
        npz["bg_valid"].astype(np.uint8),
        size=support_size, mode="nearest") > 0
    weak_intensity_support = weak_intensity_support | ~intensity_observed

    z_edge = np.zeros_like(valid)
    if "z_level_diff" in npz.files and "dz_dd" in npz.files:
        z_pair_ok = (npz["z_pair_ok"].astype(bool)
                     if "z_pair_ok" in npz.files else np.ones_like(valid))
        z_edge = (z_pair_ok
                  & (np.abs(npz["z_level_diff"]) >= a.curb_step_min)
                  & (np.abs(npz["dz_dd"]) >= a.curb_grad_min))
        dilate_cells = max(0, int(round(a.rgb_z_edge_dilation / d_res)))
        if dilate_cells > 0:
            z_edge = binary_dilation(
                z_edge,
                structure=np.ones((2 * dilate_cells + 1, 1), dtype=bool))

    base = (valid & color_ok
            & (excess >= a.rgb_min_local_contrast)
            & (sigma >= a.rgb_sigma_threshold))
    strong = sigma >= a.rgb_strong_sigma_threshold
    mask = base & (weak_intensity_support | strong) & ~z_edge

    values = sigma[valid]
    stats = {
        "available": True,
        "valid_cells": int(valid.sum()),
        "candidate_cells": int(mask.sum()),
        "candidate_ratio_of_valid": float(mask.sum() / max(valid.sum(), 1)),
        "white_candidate_cells": int((mask & white).sum()),
        "yellow_candidate_cells": int((mask & yellow).sum()),
        "local_contrast_p99": float(np.percentile(excess[valid], 99))
            if valid.any() else 0.0,
        "sigma_p99": float(np.percentile(values, 99))
            if values.size else 0.0,
        "parameters": {
            "sigma_threshold": float(a.rgb_sigma_threshold),
            "strong_sigma_threshold": float(a.rgb_strong_sigma_threshold),
            "min_luminance": float(a.rgb_min_luminance),
            "min_local_contrast": float(a.rgb_min_local_contrast),
        },
    }
    return mask, sigma, valid, stats


def append_nonduplicate_tracks(base: list["Track"], candidates: list["Track"],
                               max_d: float) -> tuple[list["Track"], int]:
    """
    既存の反射強度Laneを優先し、RGB補完候補の重複だけを落とす。

    remove_duplicates()のように全体を長さ順へ並べ替えると、長いRGB誤候補が
    反射強度Laneを置換し得るため、基準Laneを先に固定する。
    """
    result = list(base)
    dropped = 0
    for t in sorted(candidates, key=lambda item: -item.span):
        duplicate = False
        for r in result:
            lo, hi = _overlap(t, r)
            if hi < lo:
                continue
            overlap = (hi - lo + 1) / max(min(t.span, r.span), 1)
            if overlap < 0.50:
                continue
            spacing, _ = _median_spacing(t, r)
            if np.isfinite(spacing) and abs(spacing) <= max_d:
                duplicate = True
                break
        if duplicate:
            dropped += 1
        else:
            result.append(t)
    return result, dropped


# =============================================================================
# リッジ点抽出
# =============================================================================

def extract_ridge_points(mask: np.ndarray, weight: np.ndarray,
                         d_axis: np.ndarray, d_resolution: float,
                         min_width: float, max_width: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]:
    """
    各 s 列について、mask の連続ランを 1 本のリッジ点へ縮約する。

    細線化と違い分岐で壊れない。幅の上限を課すので、広い明部
    (コンクリート舗装・路肩の白っぽい面) はここで落ちる。

    Returns
    -------
    s_idx, d_center, strength, width  (すべて 1 次元, 長さ = リッジ点数)
    """
    nd, ns = mask.shape
    min_cells = max(1, int(round(min_width / d_resolution)))
    max_cells = max(min_cells, int(round(max_width / d_resolution)))

    pad = np.zeros((nd + 2, ns), dtype=bool)
    pad[1:-1] = mask
    diff = np.diff(pad.astype(np.int8), axis=0)
    starts_d, starts_s = np.nonzero(diff == 1)
    ends_d, ends_s = np.nonzero(diff == -1)
    # 同じ列内で start と end は同数・同順
    if len(starts_d) != len(ends_d):  # pragma: no cover
        raise RuntimeError("ラン検出の不整合")
    width_cells = ends_d - starts_d
    ok = (width_cells >= min_cells) & (width_cells <= max_cells)
    starts_d, ends_d, s_idx = starts_d[ok], ends_d[ok], starts_s[ok]
    width_cells = width_cells[ok]

    s_out = np.empty(len(s_idx), dtype=np.int64)
    d_out = np.empty(len(s_idx), dtype=np.float64)
    w_out = np.empty(len(s_idx), dtype=np.float64)
    for n, (a, b, sj) in enumerate(zip(starts_d, ends_d, s_idx)):
        w = weight[a:b, sj].astype(np.float64)
        total = w.sum()
        if total <= 0:
            d_out[n] = d_axis[a:b].mean()
            w_out[n] = 0.0
        else:
            d_out[n] = float((d_axis[a:b] * w).sum() / total)
            w_out[n] = float(w.max())
        s_out[n] = sj
    return s_out, d_out, w_out, (width_cells * d_resolution)


# =============================================================================
# 塗装片
# =============================================================================

@dataclass
class Segment:
    s_idx: np.ndarray
    d_val: np.ndarray
    strength: np.ndarray

    @property
    def start(self) -> int:
        return int(self.s_idx[0])

    @property
    def stop(self) -> int:
        return int(self.s_idx[-1])

    @property
    def n(self) -> int:
        return len(self.s_idx)

    def slope(self) -> float:
        if self.n < 3:
            return 0.0
        x = self.s_idx.astype(np.float64)
        y = self.d_val
        x = x - x.mean()
        den = float((x * x).sum())
        return float((x * (y - y.mean())).sum() / den) if den > 1e-9 else 0.0


def _update_active_segment(seg: dict, sj: int, d: float, strength: float) -> None:
    seg["s"].append(int(sj))
    seg["d"].append(float(d))
    seg["w"].append(float(strength))
    if len(seg["s"]) >= 5:
        x = np.asarray(seg["s"][-12:], dtype=np.float64)
        y = np.asarray(seg["d"][-12:], dtype=np.float64)
        xc = x - x.mean()
        den = float((xc * xc).sum())
        seg["slope"] = float((xc * (y - y.mean())).sum() / den) \
            if den > 1e-9 else 0.0
    seg["last_s"] = int(sj)
    seg["last_d"] = float(d)


def build_segments(s_idx: np.ndarray, d_val: np.ndarray, strength: np.ndarray,
                   link_d: float, tiny_gap_cells: int,
                   min_cells: int,
                   assignment_method: str = "hungarian") -> list[Segment]:
    """リッジ点を短い塗装片へまとめる。

    Production V2の既定はHungarian法による列単位の一括1対1対応である。
    旧Greedyのようにactive片の処理順で近接する平行線の対応が入れ替わらない。
    """
    if len(s_idx) == 0:
        return []
    order = np.argsort(s_idx, kind="stable")
    s_idx, d_val, strength = s_idx[order], d_val[order], strength[order]
    ns_total = int(s_idx[-1]) + 1
    bounds = np.searchsorted(s_idx, np.arange(ns_total + 1))

    use_hungarian = assignment_method == "hungarian"
    linear_sum_assignment = None
    if use_hungarian:
        try:
            from scipy.optimize import linear_sum_assignment as _lsa
            linear_sum_assignment = _lsa
        except Exception as exc:  # pragma: no cover
            print(f"[warn] Hungarian対応付けを利用できないためGreedyへ戻します: {exc}")
            use_hungarian = False

    active: list[dict] = []
    done: list[Segment] = []
    for sj in range(ns_total):
        lo, hi = bounds[sj], bounds[sj + 1]
        cand_d = d_val[lo:hi]
        cand_w = strength[lo:hi]
        used = np.zeros(len(cand_d), dtype=bool)

        if active and len(cand_d):
            pred = np.array([
                seg["last_d"] + seg["slope"] * (sj - seg["last_s"])
                for seg in active], dtype=np.float64)
            gap = np.array([sj - seg["last_s"] for seg in active], dtype=np.float64)
            cost = np.abs(pred[:, None] - cand_d[None, :])
            # 欠測が長いactive片をわずかに不利にし、直近片を優先する。
            cost += 0.002 * gap[:, None]
            valid = np.abs(pred[:, None] - cand_d[None, :]) <= link_d

            if use_hungarian and linear_sum_assignment is not None:
                work = np.where(valid, cost, 1e9)
                rows, cols = linear_sum_assignment(work)
                for r, c in zip(rows, cols):
                    if work[r, c] >= 1e8:
                        continue
                    used[c] = True
                    _update_active_segment(active[int(r)], sj,
                                           float(cand_d[c]), float(cand_w[c]))
            else:
                # 互換用Greedy。ただし候補は全ペアcost昇順で選ぶため、
                # activeリスト順だけに依存する旧実装より安定する。
                pairs = np.argwhere(valid)
                if len(pairs):
                    pairs = sorted((int(r), int(c), float(cost[r, c]))
                                   for r, c in pairs)
                    matched_rows = set()
                    for r, c, _ in sorted(pairs, key=lambda x: x[2]):
                        if r in matched_rows or used[c]:
                            continue
                        matched_rows.add(r)
                        used[c] = True
                        _update_active_segment(active[r], sj,
                                               float(cand_d[c]), float(cand_w[c]))

        still: list[dict] = []
        for seg in active:
            if sj - seg["last_s"] > tiny_gap_cells:
                if len(seg["s"]) >= min_cells:
                    done.append(Segment(np.asarray(seg["s"], dtype=np.int64),
                                        np.asarray(seg["d"]),
                                        np.asarray(seg["w"])))
            else:
                still.append(seg)
        active = still

        for k in np.flatnonzero(~used):
            active.append({"s": [sj], "d": [float(cand_d[k])],
                           "w": [float(cand_w[k])], "slope": 0.0,
                           "last_s": sj, "last_d": float(cand_d[k])})

    for seg in active:
        if len(seg["s"]) >= min_cells:
            done.append(Segment(np.asarray(seg["s"], dtype=np.int64),
                                np.asarray(seg["d"]),
                                np.asarray(seg["w"])))
    done.sort(key=lambda s: s.start)
    return done


# =============================================================================
# グラフ対応付け
# =============================================================================

def _bridge_crosses_other_segment(segments: list[Segment], i: int, j: int,
                                  tolerance: float) -> bool:
    """i→jの線形補間が別の検出片を横切るか判定する。

    合流そのものを否定する判定ではなく、長距離欠測を別線へ誤接続する
    リスクを下げる安全側判定。検出片が途中にある場合は、その片を介した
    短い接続が優先される。
    """
    a, b = segments[i], segments[j]
    s0, s1 = a.stop, b.start
    if s1 <= s0:
        return False
    d0, d1 = float(a.d_val[-1]), float(b.d_val[0])
    for k, other in enumerate(segments):
        if k in (i, j) or other.stop < s0 or other.start > s1:
            continue
        mask = (other.s_idx >= s0) & (other.s_idx <= s1)
        if mask.sum() < 2:
            continue
        ss = other.s_idx[mask].astype(np.float64)
        bridge = d0 + (d1 - d0) * ((ss - s0) / max(s1 - s0, 1))
        diff = other.d_val[mask] - bridge
        # 明確な符号反転、または補間線とほぼ重なる片が途中にある。
        if (float(diff.min()) < -tolerance and float(diff.max()) > tolerance):
            return True
        if float(np.min(np.abs(diff))) <= tolerance:
            return True
    return False


def build_edges(segments: list[Segment], s_res: float,
                a: argparse.Namespace) -> tuple[list[tuple[int, int, float]], dict]:
    """接続可能な塗装片ペアを列挙する。

    15m未満の通常接続と長距離接続を分離し、長距離では前後両側からの
    外挿一致、厳しい傾き差、他線横断禁止を適用する。
    """
    n = len(segments)
    starts = np.array([s.start for s in segments])
    stops = np.array([s.stop for s in segments])
    d_start = np.array([s.d_val[0] for s in segments])
    d_stop = np.array([s.d_val[-1] for s in segments])
    slopes = np.array([s.slope() for s in segments])
    max_gap_cells = a.max_gap / s_res

    stats = {
        "candidates": 0, "accepted": 0, "long_gap_candidates": 0,
        "long_gap_accepted": 0, "rejected_position": 0,
        "rejected_slope": 0, "rejected_bridge_crossing": 0,
    }
    edges: list[tuple[int, int, float]] = []
    order = np.argsort(starts, kind="stable")
    for ii in range(n):
        i = int(order[ii])
        for jj in range(ii + 1, n):
            j = int(order[jj])
            gap_cells = starts[j] - stops[i]
            if gap_cells <= 0:
                continue
            if gap_cells > max_gap_cells:
                break
            stats["candidates"] += 1
            gap_m = gap_cells * s_res
            is_long = gap_m >= a.long_gap_threshold
            if is_long:
                stats["long_gap_candidates"] += 1

            tol = a.d_tolerance + a.d_tolerance_rate * gap_m
            max_slope = a.max_slope_diff
            if is_long:
                tol *= a.long_gap_d_tolerance_scale
                max_slope = min(max_slope, a.long_gap_max_slope_diff)

            pred_forward = d_stop[i] + slopes[i] * gap_cells
            pred_backward = d_start[j] - slopes[j] * gap_cells
            if (abs(pred_forward - d_start[j]) > tol
                    or abs(pred_backward - d_stop[i]) > tol):
                stats["rejected_position"] += 1
                continue
            slope_diff = abs(slopes[i] - slopes[j]) / max(s_res, 1e-9)
            if slope_diff > max_slope:
                stats["rejected_slope"] += 1
                continue
            if (is_long and not a.allow_bridge_crossing
                    and _bridge_crosses_other_segment(
                        segments, i, j, a.bridge_crossing_tolerance)):
                stats["rejected_bridge_crossing"] += 1
                continue

            # 長距離は通常より高いcostにし、短い根拠片を介す経路を優先。
            cost = a.gap_penalty * gap_m * (1.35 if is_long else 1.0)
            edges.append((i, j, cost))
            stats["accepted"] += 1
            if is_long:
                stats["long_gap_accepted"] += 1
    return edges, stats


def extract_chains(segments: list[Segment], edges, s_res: float,
                   min_length: float) -> list[list[int]]:
    """最長経路 DP を繰り返し、スコアの高い連鎖から順に取り出す。"""
    n = len(segments)
    lengths = np.array([s.n * s_res for s in segments], dtype=np.float64)
    order = sorted(range(n), key=lambda i: segments[i].start)
    incoming: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for i, j, cost in edges:
        incoming[j].append((i, cost))

    alive = np.ones(n, dtype=bool)
    chains: list[list[int]] = []
    while True:
        best = np.full(n, -np.inf)
        prev = np.full(n, -1, dtype=np.int64)
        for i in order:
            if not alive[i]:
                continue
            value = lengths[i]
            bp = -1
            for k, cost in incoming[i]:
                if not alive[k] or not np.isfinite(best[k]):
                    continue
                cand = best[k] - cost + lengths[i]
                if cand > value:
                    value, bp = cand, k
            best[i], prev[i] = value, bp
        if not np.any(np.isfinite(best)):
            break
        top = int(np.argmax(np.where(alive, best, -np.inf)))
        if not np.isfinite(best[top]):
            break
        chain = []
        cur = top
        while cur != -1:
            chain.append(cur)
            cur = int(prev[cur])
        chain.reverse()
        total = float(sum(lengths[c] for c in chain))
        span = (segments[chain[-1]].stop - segments[chain[0]].start) * s_res
        alive[chain] = False
        if span >= min_length or total >= min_length:
            chains.append(chain)
        elif best[top] < min_length:
            break        # 以降の鎖はこれより短い
        if not np.any(alive):
            break
    return chains


# =============================================================================
# トラック
# =============================================================================

@dataclass
class Track:
    kind: int
    start: int
    stop: int
    d: np.ndarray            # (span,) 密な横位置
    observed: np.ndarray     # (span,) 実測があった位置
    strength: np.ndarray     # (span,) 証拠強度 (欠測は 0)
    valid_support: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    source: str = "auto"
    line_type: str = "Uncertain"
    features: dict = field(default_factory=dict)

    @property
    def span(self) -> int:
        return self.stop - self.start + 1

    def s_slice(self) -> slice:
        return slice(self.start, self.stop + 1)


def chain_to_track(segments: list[Segment], chain: list[int], kind: int,
                   source: str) -> Track:
    s_all = np.concatenate([segments[c].s_idx for c in chain])
    d_all = np.concatenate([segments[c].d_val for c in chain])
    w_all = np.concatenate([segments[c].strength for c in chain])
    order = np.argsort(s_all, kind="stable")
    s_all, d_all, w_all = s_all[order], d_all[order], w_all[order]
    uniq, first = np.unique(s_all, return_index=True)
    s_all, d_all, w_all = uniq, d_all[first], w_all[first]

    start, stop = int(s_all[0]), int(s_all[-1])
    grid = np.arange(start, stop + 1)
    d_dense = np.interp(grid, s_all, d_all)
    observed = np.zeros(len(grid), dtype=bool)
    observed[s_all - start] = True
    strength = np.zeros(len(grid), dtype=np.float32)
    strength[s_all - start] = w_all
    return Track(kind=kind, start=start, stop=stop, d=d_dense,
                 observed=observed, strength=strength, source=source)


def _runs(flags: np.ndarray) -> list[tuple[bool, int]]:
    if len(flags) == 0:
        return []
    change = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    bounds = np.r_[0, change, len(flags)]
    return [(bool(flags[bounds[i]]), int(bounds[i + 1] - bounds[i]))
            for i in range(len(bounds) - 1)]


def _max_true_run(flags: np.ndarray) -> int:
    return max((n for flag, n in _runs(np.asarray(flags, dtype=bool)) if flag),
               default=0)


def _valid_observation_runs(observed: np.ndarray, valid: np.ndarray
                            ) -> tuple[list[int], list[int]]:
    """センサー有効区間だけでON/OFFランを抽出する。

    valid=False は破線のOFFではなく観測不能としてランを分断する。
    """
    on_cells: list[int] = []
    off_cells: list[int] = []
    i = 0
    n = len(observed)
    while i < n:
        while i < n and not valid[i]:
            i += 1
        begin = i
        while i < n and valid[i]:
            i += 1
        if i <= begin:
            continue
        runs = _runs(observed[begin:i])
        for k, (flag, cells) in enumerate(runs):
            if flag:
                on_cells.append(cells)
            elif 0 < k < len(runs) - 1:
                off_cells.append(cells)
    return on_cells, off_cells


def _close_short_gaps(observed: np.ndarray, max_cells: int) -> np.ndarray:
    """内部の短い OFF を埋める。両端の OFF は線の外側なので埋めない。

    反射強度の検出は 1〜2 セル単位で途切れる。IBEV 実測では 16 本すべての
    レーンで生の median_on が 0.25〜0.75m (1〜3 セル) しかなく、実線と破線
    の区別が「OFF が 1 セルか 2 セルか」で決まってしまっていた。ラン長を
    測る前に検出の途切れを埋めることで、塗装の周期そのものを見る。
    """
    flags = np.asarray(observed, dtype=bool)
    if max_cells <= 0 or not flags.any():
        return flags.copy()
    out = flags.copy()
    position = 0
    runs = _runs(flags)
    for k, (flag, cells) in enumerate(runs):
        if (not flag) and 0 < k < len(runs) - 1 and cells <= max_cells:
            out[position:position + cells] = True
        position += cells
    return out


def _baseline_fragmentation(tracks: list["Track"], s_res: float
                            ) -> tuple[float, float]:
    """RGB補完の自動判定用に、反射強度だけで得た線の断片化度合いを測る。"""
    coverages: list[float] = []
    on_lengths: list[float] = []
    for t in tracks:
        valid = t.valid_support
        if len(valid) != t.span:
            valid = np.ones(t.span, dtype=bool)
        observed_valid = np.asarray(t.observed, dtype=bool) & valid
        if valid.any():
            coverages.append(float(observed_valid.sum()) / float(valid.sum()))
        on_cells, _ = _valid_observation_runs(observed_valid, valid)
        on_lengths.extend(n * s_res for n in on_cells)
    return (float(np.median(coverages)) if coverages else 0.0,
            float(np.median(on_lengths)) if on_lengths else 0.0)


def _coefficient_of_variation(values: list[float]) -> float:
    """変動係数。実在の破線は周期が揃うが、検出の途切れは揃わない。"""
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 0:
        return float("inf")
    return float(arr.std() / mean)


def attach_valid_support(track: Track, valid_grid: np.ndarray,
                         d_axis: np.ndarray, d_res: float) -> None:
    rows = np.clip(np.round((track.d - d_axis[0]) / d_res).astype(int),
                   0, valid_grid.shape[0] - 1)
    cols = np.arange(track.start, track.stop + 1)
    track.valid_support = valid_grid[rows, cols].astype(bool, copy=True)


def attach_frenet_ambiguity(track: Track, ambiguity_grid: np.ndarray,
                             d_axis: np.ndarray, d_res: float,
                             s_res: float) -> None:
    """トラック周辺のFrenet分岐曖昧性を特徴量として保持する。"""
    if ambiguity_grid is None or ambiguity_grid.size == 0:
        track.features["frenet_ambiguity_ratio"] = 0.0
        track.features["max_frenet_ambiguity_run_m"] = 0.0
        return
    rows = np.clip(np.round((track.d - d_axis[0]) / d_res).astype(int),
                   0, ambiguity_grid.shape[0] - 1)
    cols = np.arange(track.start, track.stop + 1)
    flags = ambiguity_grid[rows, cols].astype(bool)
    track.features["frenet_ambiguity_ratio"] = float(flags.mean())
    track.features["max_frenet_ambiguity_run_m"] = float(
        _max_true_run(flags) * s_res)


def classify_line_type(track: Track, s_res: float,
                       a: argparse.Namespace) -> None:
    valid = track.valid_support
    if len(valid) != track.span:
        valid = np.ones(track.span, dtype=bool)
    observed = np.asarray(track.observed, dtype=bool)
    observed_valid = observed & valid

    raw_on_cells, raw_off_cells = _valid_observation_runs(observed_valid, valid)
    raw_on = [n * s_res for n in raw_on_cells]
    raw_off = [n * s_res for n in raw_off_cells]

    # 塗装の周期は、検出の途切れを埋めてから測る。
    close_cells = max(0, int(round(a.dash_close_gap / s_res)))
    closed = _close_short_gaps(observed_valid, close_cells) & valid
    on_cells, off_cells = _valid_observation_runs(closed, valid)
    on = [n * s_res for n in on_cells]
    off = [n * s_res for n in off_cells]

    observed_ratio = float(observed.mean())
    valid_ratio = float(valid.mean())
    observed_on_valid = (float(observed_valid.sum()) / float(valid.sum())
                         if valid.any() else 0.0)
    interpolated = ~observed
    valid_gap = valid & ~observed
    unknown_gap = ~valid

    closed_on_valid = (float(closed.sum()) / float(valid.sum())
                       if valid.any() else 0.0)
    max_on_m = max(on) if on else 0.0
    span_m = float(track.span * s_res)
    period = [on[i] + off[i] for i in range(min(len(on), len(off)))]
    on_cv = _coefficient_of_variation(on)
    period_cv = _coefficient_of_variation(period)

    track.features.update({
        "coverage": observed_on_valid,
        "observed_ratio": observed_ratio,
        "observed_on_valid_ratio": observed_on_valid,
        "sensor_valid_ratio": valid_ratio,
        "interpolated_ratio": float(interpolated.mean()),
        "on_run_count": len(on),
        "median_on_m": float(np.median(on)) if on else 0.0,
        "median_off_m": float(np.median(off)) if off else 0.0,
        "max_gap_m": float(_max_true_run(interpolated) * s_res),
        "max_valid_gap_m": float(_max_true_run(valid_gap) * s_res),
        "max_unknown_gap_m": float(_max_true_run(unknown_gap) * s_res),
        # 途切れを埋める前の生の値。断片化の度合いを診断するために残す。
        "raw_on_run_count": len(raw_on),
        "raw_median_on_m": float(np.median(raw_on)) if raw_on else 0.0,
        "raw_median_off_m": float(np.median(raw_off)) if raw_off else 0.0,
        "closed_coverage": closed_on_valid,
        "max_on_m": max_on_m,
        "on_cv": on_cv,
        "period_cv": period_cv,
    })

    # 1本の長い連続塗装があるなら、途中の欠測に関係なく実線。
    if span_m > 0 and max_on_m >= a.solid_max_run_ratio * span_m:
        track.line_type = "Solid_line"
        return
    if closed_on_valid >= a.solid_coverage:
        track.line_type = "Solid_line"
        return
    # 実在の破線は周期が揃う。検出の途切れは揃わないので変動係数で弾く。
    if (len(on) >= a.dash_min_runs and off
            and a.dash_on_min <= np.median(on) <= a.dash_on_max
            and a.dash_off_min <= np.median(off) <= a.dash_off_max
            and on_cv <= a.dash_regularity
            and period_cv <= a.dash_regularity):
        track.line_type = "Dashed_line"
        return
    track.line_type = "Solid_line" if closed_on_valid >= 0.5 else "Uncertain"


# =============================================================================
# レーン構造
# =============================================================================

def _overlap(a: Track, b: Track) -> tuple[int, int]:
    return max(a.start, b.start), min(a.stop, b.stop)


def _median_spacing(a: Track, b: Track) -> tuple[float, float]:
    lo, hi = _overlap(a, b)
    if hi < lo:
        return float("nan"), 0.0
    da = a.d[lo - a.start:hi - a.start + 1]
    db = b.d[lo - b.start:hi - b.start + 1]
    return float(np.median(db - da)), float(hi - lo + 1)


def flag_hatch_groups(tracks: list[Track], s_res: float,
                      a: argparse.Namespace) -> int:
    """
    間隔が狭すぎる並走群を導流帯・横断歩道として除外する。

    単純な近接クラスタリングだと、導流帯の隣を走る本物の車線境界線まで
    同じ群に取り込んでしまう。そこで「近接線との重なりが自分自身の
    大半を占めるか」を条件に加える。導流帯のストライプ同士は s 区間が
    ほぼ一致する一方、隣接する車線境界線はずっと長く伸びるため分離できる。
    """
    lanes = [t for t in tracks if t.kind == KIND_LANE]
    n = len(lanes)
    if n < a.hatch_min_members:
        return 0
    min_cells = a.hatch_min_overlap / s_res

    close: dict[int, list[int]] = {i: [] for i in range(n)}
    overlap_cells = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            spacing, cells = _median_spacing(lanes[i], lanes[j])
            if cells < min_cells or not np.isfinite(spacing):
                continue
            if abs(spacing) < a.hatch_max_spacing:
                close[i].append(j)
                close[j].append(i)
                overlap_cells[i, j] = overlap_cells[j, i] = cells

    flagged = 0
    for i in range(n):
        if len(close[i]) < a.hatch_min_members - 1:
            continue
        if lanes[i].span * s_res > a.hatch_max_length:
            continue
        ratios = [overlap_cells[i, j] / max(lanes[i].span, 1)
                  for j in close[i]]
        if float(np.mean(ratios)) < a.hatch_self_overlap:
            continue
        lanes[i].kind = KIND_HATCH
        lanes[i].source = "auto_hatch_group"
        lanes[i].features["hatch_neighbors"] = len(close[i])
        lanes[i].features["hatch_self_overlap"] = float(np.mean(ratios))
        flagged += 1
    return flagged


def structure_infill(tracks: list[Track], i_sigma: np.ndarray,
                     bg_valid: np.ndarray, d_axis: np.ndarray,
                     s_res: float, d_res: float,
                     a: argparse.Namespace) -> list[Track]:
    """
    隣接線間隔が 1 車線分空いている区間について、その帯だけ
    しきい値を下げて再探索する。全体を下げないので誤検出は増えない。
    """
    lanes = sorted([t for t in tracks if t.kind == KIND_LANE],
                   key=lambda t: float(np.median(t.d)))
    added: list[Track] = []
    min_cells = a.infill_min_overlap / s_res
    band_cells = max(1, int(round(a.infill_band / d_res)))
    threshold = a.sigma_threshold * a.infill_sigma_ratio

    for left, right in zip(lanes[:-1], lanes[1:]):
        spacing, cells = _median_spacing(left, right)
        if cells < min_cells or not np.isfinite(spacing):
            continue
        if not (a.infill_min_spacing <= spacing <= a.infill_max_spacing):
            continue
        lo, hi = _overlap(left, right)
        dl = left.d[lo - left.start:hi - left.start + 1]
        dr = right.d[lo - right.start:hi - right.start + 1]
        target = 0.5 * (dl + dr)
        rows = np.clip(np.round((target - d_axis[0]) / d_res).astype(int),
                       band_cells, len(d_axis) - band_cells - 1)
        cols = np.arange(lo, hi + 1)
        offsets = np.arange(-band_cells, band_cells + 1)
        patch = i_sigma[rows[None, :] + offsets[:, None], cols[None, :]]
        valid = bg_valid[rows[None, :] + offsets[:, None], cols[None, :]]
        patch = np.where(valid, patch, 0.0)

        best_row = np.argmax(patch, axis=0)
        best_val = patch[best_row, np.arange(patch.shape[1])]
        hit = best_val >= threshold
        coverage = float(hit.mean())
        d_found = d_axis[rows + offsets[best_row]]
        residual = d_found[hit] - target[hit] if hit.any() else np.zeros(0)
        d_std = float(np.std(residual)) if residual.size >= 4 else np.inf
        accept = (coverage >= a.infill_min_coverage
                  and d_std <= a.infill_max_d_std)
        if not accept and not a.infill_emit_virtual:
            continue
        d_dense = np.where(hit, d_found, target)
        # 観測のあった位置だけで滑らかに引き直す
        if hit.sum() >= 2:
            d_dense = np.interp(np.arange(len(cols)),
                                np.flatnonzero(hit), d_found[hit])
        track = Track(kind=KIND_LANE, start=int(lo), stop=int(hi),
                      d=d_dense if accept else target,
                      observed=hit if accept else np.zeros(len(cols), bool),
                      strength=np.where(hit, best_val, 0.0).astype(np.float32),
                      valid_support=valid.any(axis=0).astype(bool),
                      source="structure_infill" if accept
                             else "structure_virtual")
        track.features["infill_spacing_m"] = spacing
        track.features["infill_coverage"] = coverage
        track.features["infill_d_std_m"] = d_std
        if not accept:
            track.line_type = "Virtual_lane"
        added.append(track)
    return added


def remove_duplicates(tracks: list[Track], s_res: float,
                      max_d: float) -> list[Track]:
    """横位置がほぼ同じトラックの重複を落とす (長い方を残す)。"""
    keep = sorted(tracks, key=lambda t: -t.span)
    result: list[Track] = []
    for t in keep:
        dup = False
        for r in result:
            if r.kind != t.kind:
                continue
            lo, hi = _overlap(t, r)
            if hi < lo:
                continue
            overlap = (hi - lo + 1) / max(t.span, 1)
            if overlap < 0.5:
                continue
            spacing, _ = _median_spacing(t, r)
            if np.isfinite(spacing) and abs(spacing) <= max_d:
                dup = True
                break
        if not dup:
            result.append(t)
    return sorted(result, key=lambda t: (-t.span, t.start))


# =============================================================================

def run(a: argparse.Namespace) -> dict:
    t0 = time.time()
    npz = np.load(Path(a.stage1_npz))
    meta1 = {}
    meta1_path = Path(a.stage1_npz).with_suffix(".meta.json")
    if meta1_path.is_file():
        import json
        meta1 = json.loads(meta1_path.read_text(encoding="utf-8"))
    grid = meta1.get("grid", {})
    d_res = float(grid.get("d_resolution", 0.05))
    s_res = float(grid.get("s_resolution", 0.25))
    d_min = float(grid.get("d_min", -15.0))

    i_sigma = npz["i_sigma"]
    bg_valid = npz["bg_valid"]
    nd, ns = i_sigma.shape
    d_axis = d_min + (np.arange(nd) + 0.5) * d_res
    ambiguity_grid = (npz["frenet_ambiguous"].astype(bool)
                      if "frenet_ambiguous" in npz
                      else np.zeros_like(bg_valid, dtype=bool))
    stats: dict = {}

    # ---------------- 反射: リッジ -> 塗装片 -> グラフ -> トラック
    lane_mask = bg_valid & (i_sigma >= a.sigma_threshold)
    rs, rd, rw, rwidth = extract_ridge_points(
        lane_mask, i_sigma, d_axis, d_res,
        a.min_paint_width, a.max_paint_width)
    print(f"[info] paint ridge points: {len(rs):,}")

    segments = build_segments(
        rs, rd, rw, a.link_d,
        max(1, int(round(a.segment_tiny_gap / s_res))),
        max(1, int(round(a.min_segment_length / s_res))),
        assignment_method=a.assignment_method)
    print(f"[info] paint segments: {len(segments)}")

    edges, edge_stats = build_edges(segments, s_res, a)
    chains = extract_chains(segments, edges, s_res, a.min_track_length)
    lane_tracks = [chain_to_track(segments, c, KIND_LANE, "auto_intensity")
                   for c in chains]
    lane_tracks = [t for t in lane_tracks
                   if t.span * s_res >= a.min_track_length]
    for t in lane_tracks:
        attach_valid_support(t, bg_valid, d_axis, d_res)
        attach_frenet_ambiguity(t, ambiguity_grid, d_axis, d_res, s_res)
    print(f"[info] paint chains: {len(chains)} -> lane tracks: "
          f"{len(lane_tracks)}")
    stats["intensity"] = {
        "ridge_points": int(len(rs)),
        "segments": int(len(segments)),
        "edges": int(len(edges)),
        "edge_diagnostics": edge_stats,
        "assignment_method": a.assignment_method,
        "chains": int(len(chains)),
        "tracks": int(len(lane_tracks)),
        "median_ridge_width_m": float(np.median(rwidth)) if len(rwidth) else 0.0,
    }

    # ---------------- RGB局所コントラスト補完
    trajectory_length_m = max((ns - 1) * s_res, s_res)
    baseline_span_m = float(sum(t.span * s_res for t in lane_tracks))
    baseline_span_ratio = baseline_span_m / trajectory_length_m
    baseline_tracks_per_100m = (
        len(lane_tracks) * 100.0 / trajectory_length_m)
    baseline_coverage, baseline_on_m = _baseline_fragmentation(
        lane_tracks, s_res)
    # 総延長と本数だけでは、線が引けていても中身がスカスカな状態を
    # 見逃す。IBEV 実測では span_ratio 1.64・2.78本/100m と健全に見える
    # 一方で、median coverage 0.66・median ON長 0.5m まで断片化していた。
    use_rgb = (
        a.rgb_lane_fusion == "on"
        or (
            a.rgb_lane_fusion == "auto"
            and (
                baseline_span_ratio < a.rgb_auto_min_span_ratio
                or baseline_tracks_per_100m
                < a.rgb_auto_min_tracks_per_100m
                or baseline_coverage < a.rgb_auto_min_coverage
                or baseline_on_m < a.rgb_auto_min_on_length
            )
        )
    )
    rgb_stats = {
        "mode": a.rgb_lane_fusion,
        "used": False,
        "baseline_span_m": baseline_span_m,
        "baseline_span_ratio": baseline_span_ratio,
        "baseline_tracks_per_100m": baseline_tracks_per_100m,
        "baseline_median_coverage": baseline_coverage,
        "baseline_median_on_m": baseline_on_m,
    }
    if use_rgb:
        rgb_mask, rgb_sigma, rgb_valid, evidence_stats = (
            build_rgb_lane_evidence(npz, d_res, s_res, a))
        rgb_stats.update(evidence_stats)
        if evidence_stats.get("available", False) and rgb_mask.any():
            rrs, rrd, rrw, rrwidth = extract_ridge_points(
                rgb_mask, rgb_sigma, d_axis, d_res,
                a.min_paint_width, a.rgb_max_paint_width)
            rgb_segments = build_segments(
                rrs, rrd, rrw, a.link_d,
                max(1, int(round(a.segment_tiny_gap / s_res))),
                max(1, int(round(a.min_segment_length / s_res))),
                assignment_method=a.assignment_method)
            rgb_edges, rgb_edge_stats = build_edges(rgb_segments, s_res, a)
            rgb_chains = extract_chains(
                rgb_segments, rgb_edges, s_res, a.rgb_min_track_length)
            rgb_tracks = [
                chain_to_track(
                    rgb_segments, c, KIND_LANE,
                    "auto_rgb_local_contrast")
                for c in rgb_chains
            ]
            rgb_tracks = [
                t for t in rgb_tracks
                if t.span * s_res >= a.rgb_min_track_length
            ]
            for t in rgb_tracks:
                attach_valid_support(t, rgb_valid, d_axis, d_res)
                attach_frenet_ambiguity(
                    t, ambiguity_grid, d_axis, d_res, s_res)
                t.features["rgb_local_contrast"] = True

            before_merge = len(lane_tracks)
            lane_tracks, rgb_duplicate_dropped = append_nonduplicate_tracks(
                lane_tracks, rgb_tracks, a.duplicate_d)
            rgb_stats.update({
                "used": True,
                "ridge_points": int(len(rrs)),
                "segments": int(len(rgb_segments)),
                "edges": int(len(rgb_edges)),
                "edge_diagnostics": rgb_edge_stats,
                "chains": int(len(rgb_chains)),
                "tracks_before_merge": int(len(rgb_tracks)),
                "duplicate_dropped": int(rgb_duplicate_dropped),
                "tracks_added": int(len(lane_tracks) - before_merge),
                "median_ridge_width_m":
                    float(np.median(rrwidth)) if len(rrwidth) else 0.0,
            })
            print(
                "[info] RGB lane fusion: "
                f"mode={a.rgb_lane_fusion}, "
                f"baseline_span_ratio={baseline_span_ratio:.3f}, "
                f"ridge={len(rrs):,}, added={len(lane_tracks)-before_merge}, "
                f"duplicate_drop={rgb_duplicate_dropped}")
        else:
            print(
                "[info] RGB lane fusion: requested but no usable local "
                "contrast candidates")
    elif a.rgb_lane_fusion == "off":
        # off でも「auto なら発火したか」を出す。反射強度が疎なクリップで
        # off のまま回して線が出ない、という取り違えを防ぐ。
        would_fire = (
            baseline_span_ratio < a.rgb_auto_min_span_ratio
            or baseline_tracks_per_100m < a.rgb_auto_min_tracks_per_100m
            or baseline_coverage < a.rgb_auto_min_coverage
            or baseline_on_m < a.rgb_auto_min_on_length)
        note = (" ** auto なら発火する条件です。--rgb-lane-fusion auto を検討 **"
                if would_fire else "")
        print(
            "[info] RGB lane fusion: disabled by --rgb-lane-fusion off "
            f"(span_ratio={baseline_span_ratio:.3f}, "
            f"tracks_per_100m={baseline_tracks_per_100m:.2f}, "
            f"median_coverage={baseline_coverage:.2f}, "
            f"median_on={baseline_on_m:.2f}m){note}")
    else:
        print(
            "[info] RGB lane fusion: not required "
            f"(baseline_span_ratio={baseline_span_ratio:.3f}, "
            f"tracks_per_100m={baseline_tracks_per_100m:.2f}, "
            f"median_coverage={baseline_coverage:.2f}, "
            f"median_on={baseline_on_m:.2f}m)")
    stats["rgb_lane_fusion"] = rgb_stats

    # ---------------- 段差: 符号付きレベル差から縁石トラック
    curb_tracks: list[Track] = []
    if "z_level_diff" in npz:
        level = npz["z_level_diff"]
        grad = npz["dz_dd"]
        step_ok = ((np.abs(level) >= a.curb_step_min)
                   & (np.abs(level) <= a.curb_step_max))
        curb_mask = step_ok & (np.abs(grad) >= a.curb_grad_min)
        cs, cd, cw, cwidth = extract_ridge_points(
            curb_mask, np.abs(grad), d_axis, d_res, d_res, a.curb_max_width)
        csegments = build_segments(
            cs, cd, cw, a.link_d,
            max(1, int(round(a.segment_tiny_gap / s_res))),
            max(1, int(round(a.min_segment_length / s_res))),
            assignment_method=a.curb_assignment_method)
        curb_edge_args = a
        if not a.curb_strict_long_gap:
            curb_edge_args = copy.copy(a)
            curb_edge_args.allow_bridge_crossing = True
            curb_edge_args.long_gap_d_tolerance_scale = 1.0
            curb_edge_args.long_gap_max_slope_diff = a.max_slope_diff
        cedges, curb_edge_stats = build_edges(csegments, s_res, curb_edge_args)
        cchains = extract_chains(csegments, cedges, s_res, a.curb_min_length)
        raw_curbs = [chain_to_track(csegments, c, KIND_CURB, "auto_z_step")
                     for c in cchains]

        curb_valid_grid = npz["z_occupied"] if "z_occupied" in npz else bg_valid
        for t in raw_curbs:
            attach_valid_support(t, curb_valid_grid, d_axis, d_res)
            attach_frenet_ambiguity(t, ambiguity_grid, d_axis, d_res, s_res)

        for t in raw_curbs:
            if t.span * s_res < a.curb_min_length:
                continue
            rows = np.clip(np.round((t.d - d_axis[0]) / d_res).astype(int),
                           0, nd - 1)
            cols = np.arange(t.start, t.stop + 1)
            vals = level[rows, cols]
            obs = t.observed & (vals != 0)
            if obs.sum() < 4:
                continue
            signs = np.sign(vals[obs])
            major = 1.0 if (signs > 0).mean() >= 0.5 else -1.0
            consistency = float((signs == major).mean())
            if consistency < a.curb_sign_consistency:
                continue
            t.features.update({
                "median_step_m": float(np.median(np.abs(vals[obs]))),
                "step_sign": float(major),
                "sign_consistency": consistency,
                "median_grad": float(np.median(np.abs(grad[rows, cols][obs]))),
            })
            curb_tracks.append(t)
        print(f"[info] curb: ridge={len(cs):,}, segments={len(csegments)}, "
              f"chains={len(cchains)} -> tracks={len(curb_tracks)}")
        stats["curb"] = {
            "ridge_points": int(len(cs)),
            "segments": int(len(csegments)),
            "edges": int(len(cedges)),
            "edge_diagnostics": curb_edge_stats,
            "assignment_method": a.curb_assignment_method,
            "strict_long_gap": bool(a.curb_strict_long_gap),
            "chains": int(len(cchains)),
            "tracks": int(len(curb_tracks)),
        }

    # ---------------- レーン構造
    lane_tracks = remove_duplicates(lane_tracks, s_res, a.duplicate_d)
    hatch_n = flag_hatch_groups(lane_tracks, s_res, a)
    infill = structure_infill(lane_tracks, i_sigma, bg_valid, d_axis,
                              s_res, d_res, a)
    for t in infill:
        attach_frenet_ambiguity(t, ambiguity_grid, d_axis, d_res, s_res)
    lane_tracks.extend(infill)
    for t in lane_tracks:
        if len(t.valid_support) != t.span:
            attach_valid_support(t, bg_valid, d_axis, d_res)
    lane_tracks = remove_duplicates(lane_tracks, s_res, a.duplicate_d)
    print(f"[info] structure: hatch flagged={hatch_n}, infill added={len(infill)}")
    stats["structure"] = {"hatch_flagged": int(hatch_n),
                          "infill_added": int(len(infill))}

    curb_tracks = [t for t in curb_tracks
                   if float(t.observed.mean()) >= a.curb_min_coverage]
    kept_curbs_pre = remove_duplicates(curb_tracks, s_res, a.curb_duplicate_d)
    print(f"[info] curb duplicates removed: "
          f"{len(curb_tracks) - len(kept_curbs_pre)}")
    curb_tracks = kept_curbs_pre

    # ---------------- 縁石と車線線の重複解消 (吸収ではなく削除)
    lanes_only = [t for t in lane_tracks if t.kind == KIND_LANE]
    kept_curbs = []
    for c in curb_tracks:
        dup = False
        for l in lanes_only:
            lo, hi = _overlap(c, l)
            if hi < lo or (hi - lo + 1) / max(c.span, 1) < 0.5:
                continue
            spacing, _ = _median_spacing(c, l)
            if np.isfinite(spacing) and abs(spacing) <= a.curb_lane_duplicate_d:
                dup = True
                break
        if not dup:
            kept_curbs.append(c)
    dropped = len(curb_tracks) - len(kept_curbs)
    print(f"[info] curb-lane duplicates dropped: {dropped}")
    stats.setdefault("curb", {})["lane_duplicate_dropped"] = int(dropped)

    # ---------------- 線種判定
    for t in lane_tracks:
        if t.line_type != "Virtual_lane":
            classify_line_type(t, s_res, a)
        else:
            classify_line_type(t, s_res, a)
            t.line_type = "Virtual_lane"
    for t in kept_curbs:
        classify_line_type(t, s_res, a)
        t.line_type = ""   # 縁石に線種は無い

    all_tracks = lane_tracks + kept_curbs
    all_tracks.sort(key=lambda t: (t.kind, float(np.median(t.d))))

    # ---------------- 保存
    out = Path(a.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(all_tracks) + 1, dtype=np.int64)
    for i, t in enumerate(all_tracks):
        offsets[i + 1] = offsets[i] + t.span
    arrays = {
        "t_kind": np.array([t.kind for t in all_tracks], dtype=np.int32),
        "t_start": np.array([t.start for t in all_tracks], dtype=np.int64),
        "t_stop": np.array([t.stop for t in all_tracks], dtype=np.int64),
        "t_offset": offsets,
        "flat_d": np.concatenate([t.d for t in all_tracks])
            if all_tracks else np.zeros(0),
        "flat_observed": np.concatenate([t.observed for t in all_tracks])
            if all_tracks else np.zeros(0, bool),
        "flat_strength": np.concatenate([t.strength for t in all_tracks])
            if all_tracks else np.zeros(0, np.float32),
        "flat_valid_support": np.concatenate([
            t.valid_support if len(t.valid_support) == t.span
            else np.ones(t.span, dtype=bool) for t in all_tracks])
            if all_tracks else np.zeros(0, bool),
    }
    core.atomic_savez_compressed(out, **arrays)

    track_meta = []
    for i, t in enumerate(all_tracks):
        track_meta.append({
            "index": i, "kind": KIND_NAME[t.kind], "source": t.source,
            "line_type": t.line_type,
            "start_s_m": t.start * s_res, "end_s_m": t.stop * s_res,
            "length_m": t.span * s_res,
            "median_d_m": float(np.median(t.d)),
            "mean_strength": float(t.strength[t.observed].mean())
                if t.observed.any() else 0.0,
            **t.features,
        })
    meta = {
        "stage": 2,
        "production_version": core.PRODUCTION_VERSION,
        "code": {
            "stage2_sha256": core.sha256_file(Path(__file__)),
            "core_sha256": core.sha256_file(Path(core.__file__)),
        },
        "stage1_npz": str(Path(a.stage1_npz).resolve()),
        "input_signatures": {
            "stage1_npz": core.file_signature(Path(a.stage1_npz)),
            "stage1_meta": core.file_signature(meta1_path)
                if meta1_path.is_file() else None,
        },
        "grid": {"nd": nd, "ns": ns, "d_min": d_min,
                 "d_resolution": d_res, "s_resolution": s_res},
        "stats": stats,
        "counts": {
            "lane": sum(t.kind == KIND_LANE for t in all_tracks),
            "curb": sum(t.kind == KIND_CURB for t in all_tracks),
            "hatch": sum(t.kind == KIND_HATCH for t in all_tracks),
            "solid": sum(t.kind == KIND_LANE and t.line_type == "Solid_line"
                         for t in all_tracks),
            "dashed": sum(t.kind == KIND_LANE and t.line_type == "Dashed_line"
                          for t in all_tracks),
        },
        "tracks": track_meta,
        "parameters": vars(a),
        "parameter_fingerprint": core.parameter_fingerprint(
            vars(a), exclude=("stage1_npz", "output")
        ),
        "elapsed_sec": round(time.time() - t0, 2),
        "output_npz": str(out),
    }
    core.dump_json(out.with_suffix(".meta.json"), meta)
    print(f"[done] Stage2 -> {out}  ({meta['elapsed_sec']}s)")
    print(f"[done] lane={meta['counts']['lane']} "
          f"(solid={meta['counts']['solid']}, dashed={meta['counts']['dashed']}), "
          f"curb={meta['counts']['curb']}, hatch={meta['counts']['hatch']}")
    return meta


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
