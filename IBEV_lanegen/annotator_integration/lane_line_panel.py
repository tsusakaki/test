# lane_line_panel.py
"""
LaneLinePanel – ポリライン描画 + ポリゴン描画 + 属性編集を1画面で完結させるパネル。

lane_line_annotation_spec.md の仕様に基づき、BEV Lane Line のアノテーションを行う。
既存の EditLanePanel / DrawLanePanel の機能を統合し、spec.md の全属性に対応する。
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QItemSelection, QItemSelectionModel
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QSizePolicy, QSlider, QDoubleSpinBox, QSpinBox,
    QLineEdit, QTextEdit, QCheckBox, QFileDialog, QMessageBox,
    QScrollArea, QStackedWidget, QApplication, QFrame,
)
from PyQt5.QtGui import QColor

# ---------------------------------------------------------------------------
# 国際化 (i18n)
try:
    from core.i18n import t
except ImportError:
    def t(key: str) -> str:  # type: ignore[misc]
        return key

# ---------------------------------------------------------------------------
# spec.md に基づくクラス定義
# 4_BEV LD Annotation Guidelines_JP.md 準拠
# ---------------------------------------------------------------------------
POLYLINE_CLASSES = [
    "lane_line", "stop_line",
    "curb_boundary", "fence_boundary",
    "TrafficCone_boundary", "WaterSafety_boundary", "boundary",
]
POLYGON_CLASSES_LIST = [
    "zebra_line",
]
LANE_CLASSES = POLYLINE_CLASSES + POLYGON_CLASSES_LIST

# ポリゴンがデフォルトのクラス (それ以外はポリライン)
POLYGON_CLASSES = set(POLYGON_CLASSES_LIST)

# 点クラス (分岐/合流点) — ガイドライン準拠: 2クラスに分離
POINT_CLASSES_LIST = [
    "split_point",
    "merge_point",
]
POINT_CLASSES = set(POINT_CLASSES_LIST)

# ---------------------------------------------------------------------------
# geometry_type 一貫性ルール
# ---------------------------------------------------------------------------
# class が geometry を一意に決める。保存・読込・描画・class変更の全経路で共通利用する。
_GEOMETRY_MIN_VERTICES = {
    "point": 1,
    "polyline": 2,
    "polygon": 4,   # 本プロジェクトではポリゴンは4点以上を必須とする
}


def _geometry_type_from_class(cls_name: str) -> str:
    """class から正規 geometry_type を返す。"""
    if cls_name in POLYGON_CLASSES:
        return "polygon"
    if cls_name in POINT_CLASSES:
        return "point"
    return "polyline"


def _default_class_for_geometry(geom: str) -> str:
    """class情報がない旧データを読み込む際の既定class。"""
    geom = str(geom).strip().lower()
    if geom == "polygon":
        return POLYGON_CLASSES_LIST[0] if POLYGON_CLASSES_LIST else "zebra_line"
    if geom == "point":
        return POINT_CLASSES_LIST[0] if POINT_CLASSES_LIST else "split_point"
    return "lane_line"


def _normalize_record_geometry(
    rec: dict,
    *,
    infer_class_from_geometry: bool = False,
) -> str:
    """class / geometry_type / shape を常に同じgeometryへ正規化する。

    原則は class を正とする。
    ただし旧形式など class が存在しない場合のみ、
    infer_class_from_geometry=True で geometry からclassを補完する。
    """
    raw_geom = str(
        rec.get("geometry_type", rec.get("shape", "polyline"))
    ).strip().lower()
    if raw_geom == "merged":
        raw_geom = "polyline"
    if raw_geom not in ("point", "polyline", "polygon"):
        raw_geom = "polyline"

    cls_name = str(rec.get("class", "")).strip()
    if infer_class_from_geometry and cls_name not in ALL_CLASSES:
        cls_name = _default_class_for_geometry(raw_geom)
        rec["class"] = cls_name
    elif cls_name not in ALL_CLASSES:
        # 不明classは安全側でlane_lineへ寄せる。
        cls_name = "lane_line"
        rec["class"] = cls_name

    geom = _geometry_type_from_class(cls_name)
    rec["geometry_type"] = geom
    rec["shape"] = geom
    return geom


def _geometry_validation_error(rec: dict) -> str:
    """geometry頂点数が規約を満たさない場合にエラーメッセージを返す。"""
    geom = _normalize_record_geometry(rec)
    n = len(rec.get("vertices", []) or [])

    if geom == "point":
        if n != 1:
            return f"point は1点ちょうど必要です（現在 {n} 点）"
        return ""

    min_n = _GEOMETRY_MIN_VERTICES[geom]
    if n < min_n:
        label = "polygon" if geom == "polygon" else "polyline"
        return f"{label} は{min_n}点以上必要です（現在 {n} 点）"
    return ""


# 全クラス (点クラスを含む)
ALL_CLASSES = LANE_CLASSES + POINT_CLASSES_LIST

# ---------------------------------------------------------------------------
# 属性の選択肢
# 4_BEV LD Annotation Guidelines_JP.md 準拠
# ---------------------------------------------------------------------------
ATTR_CHOICES = {
    # 車線線属性
    "line_type":         ["solid", "dash", "fish_bone", "dash_solid", "unknown",
                          "wide_dash", "virtual"],
    "line_color":        ["white", "yellow"],
    "line_count":        ["single", "double"],
    "lane_number":       [],  # テキスト入力 (N_0-0 形式)
    # 道路端属性
    "boundary_id":       [],  # テキスト入力 (連番)
    # 分岐/合流点属性
    "start_line_ids":    [],  # テキスト入力
    "end_line_ids":      [],  # テキスト入力
    # 共通 (内部管理用)
    "visibility":        ["visible", "partially_occluded", "occluded", "unclear"],
    "quality":           ["ok", "uncertain", "review_required", "ignore"],
    "source":            ["manual", "auto", "auto_corrected", "imported"],
    "is_interpolated":   ["false", "true"],
    "interpolation_reason": ["", "occlusion", "worn", "shadow", "night",
                             "rain", "unknown"],
    "coordinate_system": ["ego_vehicle", "bev_grid", "map_world", "image_pixel"],
}

# ---------------------------------------------------------------------------
# クラスごとに表示する属性キーのリスト
# 4_BEV LD Annotation Guidelines_JP.md 準拠
# ---------------------------------------------------------------------------
_LANE_LINE_ATTRS = ["line_type", "line_color", "line_count"]
_BOUNDARY_ATTRS = ["boundary_id"]
_POINT_ATTRS = ["start_line_ids", "end_line_ids"]

CLASS_ATTR_MAP = {
    "lane_line":              _LANE_LINE_ATTRS,
    "stop_line":              [],
    "curb_boundary":          _BOUNDARY_ATTRS,
    "fence_boundary":         _BOUNDARY_ATTRS,
    "TrafficCone_boundary":   _BOUNDARY_ATTRS,
    "WaterSafety_boundary":   _BOUNDARY_ATTRS,
    "boundary":               _BOUNDARY_ATTRS,
    "zebra_line":             [],
    "split_point":            _POINT_ATTRS,
    "merge_point":            _POINT_ATTRS,
}

# ボタン式で表示する属性 (ガイドライン準拠)
BUTTON_ATTRS = {
    "line_type": {
        "title": "線種 [1-7]",
        "choices": ["solid", "dash", "fish_bone", "dash_solid", "unknown",
                    "wide_dash", "virtual"],
        "keys": {"1": "solid", "2": "dash", "3": "fish_bone",
                 "4": "dash_solid", "5": "unknown",
                 "6": "wide_dash", "7": "virtual"},
        "horizontal": True,
        "labels": {"solid": "実線", "dash": "破線", "fish_bone": "魚骨",
                   "dash_solid": "破実混合", "unknown": "不明",
                   "wide_dash": "幅広破線", "virtual": "仮想"},
    },
    "line_color": {
        "title": "色 [Q/W]",
        "choices": ["white", "yellow"],
        "keys": {"Q": "white", "W": "yellow"},
        "horizontal": True,
        "labels": {"white": "白", "yellow": "黄"},
    },
    "line_count": {
        "title": "本数 [A/S]",
        "choices": ["single", "double"],
        "keys": {"A": "single", "S": "double"},
        "horizontal": True,
        "labels": {"single": "単線", "double": "二重線"},
    },
}

# コンボボックスで表示する属性
COMBO_ATTRS = set()  # プルダウン属性なし

# ---------------------------------------------------------------------------
# クラスごとの表示色 (R, G, B)  0.0-1.0
# 4_BEV LD Annotation Guidelines_JP.md 準拠
# ---------------------------------------------------------------------------
CLASS_DISPLAY_COLORS = {
    "lane_line":              (0.9, 0.9, 0.9),
    "stop_line":              (1.0, 0.3, 0.3),
    "curb_boundary":          (0.5, 0.7, 1.0),
    "fence_boundary":         (0.6, 0.4, 0.2),
    "TrafficCone_boundary":   (1.0, 0.8, 0.0),
    "WaterSafety_boundary":   (0.3, 0.6, 1.0),
    "boundary":               (0.3, 0.8, 0.3),
    "zebra_line":             (1.0, 0.6, 0.2),
    "split_point":            (1.0, 0.0, 1.0),
    "merge_point":            (0.8, 0.2, 0.8),
}

# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _sort_int_or_str(val) -> tuple:
    """ソートキー: 数値文字列は整数比較、それ以外は文字列比較。"""
    s = str(val)
    try:
        return (0, int(s), "")
    except (ValueError, TypeError):
        return (1, 0, s)


def _default_lane_record(next_id: int = 1) -> dict:
    """新規 Lane 用のデフォルト属性辞書を返す (ガイドライン準拠)"""
    rec = {
        "id": f"lane_{next_id:04d}",
        "class": "lane_line",
        "geometry_type": "polyline",
        "vertices": [],
        "buffer": 0.10,
        "poly3d": None,
        "_color": _generate_random_color(),
        "_is_spline": False,
        # 共通
        "coordinate_system": "ego_vehicle",
        "visibility": "visible",
        "quality": "ok",
        "source": "manual",
        "is_interpolated": "false",
        "interpolation_reason": "",
        "note": "",
        # lane_line 属性 (pkl車線タイプ対応)
        "line_type": "solid",
        "line_color": "white",
        "line_count": "single",
        "lane_number": "",
        # lane_line ID 分離フィールド (N_X-Y)
        "lane_id_n": "",
        "lane_id_x": "",
        "lane_id_y": "",
        "lane_id_manual": False,
        # 道路端属性
        "boundary_id": "",
        # 分岐/合流点属性
        "start_line_ids": "",
        "end_line_ids": "",
    }
    return rec


def _record_from_json_entry(entry: dict, next_id: int = 1) -> dict:
    """JSON entryをLaneLinePanel内部recordへ変換する。

    既知属性だけでなく、LaneGenの ``auto_track_id`` / ``generation``
    などの非表示metadataも保持する。描画専用のprivate keyは読み込まない。
    """
    if not isinstance(entry, dict):
        raise ValueError("annotation record must be an object")

    rec = _default_lane_record(next_id)
    entry_has_valid_class = (
        str(entry.get("class", "")).strip() in ALL_CLASSES
    )

    for k, v in entry.items():
        if k == "points":
            if not isinstance(v, (list, tuple)):
                raise ValueError("points must be a list")
            rec["vertices"] = [
                (
                    float(p[0]),
                    float(p[1]),
                    float(p[2]) if len(p) > 2 else 0.0,
                )
                for p in v
            ]
        elif k == "vertices":
            rec["vertices"] = [
                (
                    float(p[0]),
                    float(p[1]),
                    float(p[2]) if len(p) > 2 else 0.0,
                )
                for p in v
            ]
        elif k == "buffer_m":
            rec["buffer"] = float(v)
        elif k == "buffer":
            rec["buffer"] = float(v)
        elif k == "poly3d" or str(k).startswith("_"):
            # 描画キャッシュ/private keyはファイルから復元しない。
            continue
        else:
            # LaneGen metadataを含め、公開keyはそのまま保持する。
            rec[k] = v

    if not entry_has_valid_class:
        raw_geom = entry.get(
            "geometry_type",
            entry.get("shape", "polyline"),
        )
        rec["class"] = _default_class_for_geometry(raw_geom)

    _normalize_record_geometry(rec)
    err = _geometry_validation_error(rec)
    if err:
        raise ValueError(
            f"不正geometry: {rec.get('id','?')} "
            f"class={rec.get('class')} "
            f"geometry={rec.get('geometry_type')}: {err}"
        )

    _rebuild_poly3d(rec)
    return rec


def _is_lanegen_auto_record(rec: dict) -> bool:
    """LaneGenが生成した自動recordかを判定する。"""
    if str(rec.get("source", "")) != "auto":
        return False
    if rec.get("auto_track_id"):
        return True
    generation = rec.get("generation")
    return bool(
        isinstance(generation, dict)
        and generation.get("production_version")
    )


def _spline_interpolate_3d(pts: np.ndarray, n_samples: int = 60) -> np.ndarray:
    """制御点列をスプライン補間して滑らかな3D折れ線を返す。

    pts: shape (N, 3)  制御点
    戻り値: shape (M, 3)  補間後の点列
    """
    n = len(pts)
    if n < 2:
        return pts.copy()
    if n == 2:
        # 2点なら線形補間
        t = np.linspace(0, 1, max(n_samples, 2))
        out = np.zeros((len(t), 3), dtype=np.float64)
        for ax in range(3):
            out[:, ax] = pts[0, ax] + t * (pts[1, ax] - pts[0, ax])
        return out.astype(np.float32)

    # 累積弦長をパラメータとして使用
    dists = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dists[i] = dists[i-1] + np.linalg.norm(pts[i] - pts[i-1])

    # 重複点を除去（距離が0の連続点）
    mask = np.ones(n, dtype=bool)
    for i in range(1, n):
        if dists[i] - dists[i-1] < 1e-9:
            mask[i] = False
    if not np.all(mask):
        pts = pts[mask]
        dists = dists[mask]
        n = len(pts)
        if n < 2:
            return pts.copy()

    total_len = dists[-1]
    if total_len < 1e-9:
        return pts.copy()

    t_sample = np.linspace(0, total_len, max(n_samples, n))

    if n >= 3:
        try:
            from scipy.interpolate import CubicSpline
            out = np.zeros((len(t_sample), 3), dtype=np.float64)
            for ax in range(3):
                cs = CubicSpline(dists, pts[:, ax].astype(np.float64),
                                 bc_type='natural')
                out[:, ax] = cs(t_sample)
            return out.astype(np.float32)
        except ImportError:
            pass

    # scipy がない場合は線形補間
    out = np.zeros((len(t_sample), 3), dtype=np.float64)
    for ax in range(3):
        out[:, ax] = np.interp(t_sample, dists, pts[:, ax].astype(np.float64))
    return out.astype(np.float32)


def _polyline_buffer_3d(pts: np.ndarray, buf_m: float) -> np.ndarray:
    """ポリライン頂点列からバッファ付き GL_QUADS 頂点列を生成する。

    pointcloud_viewer.py 内の同名関数と同一ロジック。
    DrawLanePanel 互換のために残す。
    """
    if len(pts) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    verts = []
    for i in range(len(pts) - 1):
        p0 = pts[i]; p1 = pts[i + 1]
        dx = p1[0] - p0[0]; dy = p1[1] - p0[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            continue
        nx = -dy / length * buf_m
        ny = dx / length * buf_m
        z0 = float(p0[2]); z1 = float(p1[2])
        verts += [
            (p0[0] - nx, p0[1] - ny, z0),
            (p1[0] - nx, p1[1] - ny, z1),
            (p1[0] + nx, p1[1] + ny, z1),
            (p0[0] + nx, p0[1] + ny, z0),
        ]
    return np.array(verts, dtype=np.float32)


def _rebuild_poly3d(rec: dict, use_spline: bool = False):
    """rec["vertices"] から rec["poly3d"] を再計算する。

    ここでも必ず class -> geometry_type/shape を正規化する。
    point    : 1点ちょうど
    polyline : 2点以上
    polygon  : 4点以上

    頂点数が不正な場合は poly3d=None とし、不正geometryを描画・保存へ流さない。
    """
    geom = _normalize_record_geometry(rec)
    pts = np.asarray(rec.get("vertices", []), dtype=np.float32)

    if pts.ndim == 1 and pts.size > 0:
        pts = pts.reshape(1, -1)

    if geom == "point":
        if len(pts) == 1:
            rec["poly3d"] = pts[:, :3].copy()
        else:
            rec["poly3d"] = None
        rec["_is_spline"] = False
        return

    if geom == "polyline":
        if len(pts) < 2:
            rec["poly3d"] = None
            rec["_is_spline"] = False
            return
        if use_spline:
            rec["poly3d"] = _spline_interpolate_3d(
                pts[:, :3],
                n_samples=max(60, len(pts) * 10),
            )
            rec["_is_spline"] = True
        else:
            rec["poly3d"] = pts[:, :3].copy()
            rec["_is_spline"] = False
        return

    # polygon は4点以上
    if geom == "polygon" and len(pts) >= 4:
        rec["poly3d"] = pts[:, :3].copy()
        rec["_is_spline"] = False
    else:
        rec["poly3d"] = None
        rec["_is_spline"] = False


def _remove_parallel_short_lanes(lanes: list,
                                 buf_m: float = 1.0) -> list:
    """長いポリラインのバッファ内に全頂点が含まれる短いポリラインを除去する。

    アルゴリズム:
    1. 全Laneを長さ降順にソート
    2. 長い方から順に、スプライン補間した密な点列でバッファ領域を定義
    3. 他の（まだ生存している）短いLaneの全頂点がバッファ内に収まるか判定
    4. 全頂点がバッファ内 → そのLaneを削除

    高速化: 各Laneのスプライン点列を結合した KD-Tree で最近傍距離を一括計算
    """
    if len(lanes) < 2:
        return lanes

    def _seg_length(rec):
        verts = rec.get("vertices", [])
        total = 0.0
        for i in range(len(verts) - 1):
            dx = verts[i+1][0] - verts[i][0]
            dy = verts[i+1][1] - verts[i][1]
            total += math.sqrt(dx*dx + dy*dy)
        return total

    # 長さ降順にソート（インデックス付き）
    n = len(lanes)
    lengths = [_seg_length(rec) for rec in lanes]
    order = sorted(range(n), key=lambda i: -lengths[i])

    alive = [True] * n  # 生存フラグ

    # 各Laneのスプライン密点列を事前計算 (XY のみ)
    dense_pts = []
    for rec in lanes:
        poly = rec.get("poly3d")
        if poly is not None and len(poly) >= 2:
            dense_pts.append(poly[:, :2].astype(np.float64))
        else:
            verts = rec.get("vertices", [])
            if len(verts) >= 2:
                dense_pts.append(
                    np.array([(v[0], v[1]) for v in verts], dtype=np.float64))
            else:
                dense_pts.append(np.zeros((0, 2), dtype=np.float64))

    # 各Laneの頂点 (XY) を事前取得
    vert_pts = []
    for rec in lanes:
        verts = rec.get("vertices", [])
        if verts:
            vert_pts.append(
                np.array([(v[0], v[1]) for v in verts], dtype=np.float64))
        else:
            vert_pts.append(np.zeros((0, 2), dtype=np.float64))

    try:
        from scipy.spatial import cKDTree
        use_kdtree = True
    except ImportError:
        use_kdtree = False

    for ai in order:
        if not alive[ai]:
            continue
        pts_a = dense_pts[ai]
        if len(pts_a) < 2:
            continue

        if use_kdtree:
            tree_a = cKDTree(pts_a)

        # ai より短い生存Laneをチェック
        for bi in order:
            if bi == ai or not alive[bi]:
                continue
            # ai より長いものはスキップ（長い方を消さない）
            if lengths[bi] >= lengths[ai]:
                continue
            vb = vert_pts[bi]
            if len(vb) == 0:
                continue

            # bi の全頂点が ai のバッファ内にあるか判定
            if use_kdtree:
                dists, _ = tree_a.query(vb)
                if np.all(dists <= buf_m):
                    alive[bi] = False
            else:
                # KD-Tree なしのフォールバック（遅い）
                all_inside = True
                for pt in vb:
                    min_d = float('inf')
                    for pa in pts_a:
                        d = math.sqrt((pt[0]-pa[0])**2 + (pt[1]-pa[1])**2)
                        if d < min_d:
                            min_d = d
                        if min_d <= buf_m:
                            break
                    if min_d > buf_m:
                        all_inside = False
                        break
                if all_inside:
                    alive[bi] = False

    return [rec for i, rec in enumerate(lanes) if alive[i]]


def _remove_sharp_turns(verts: list, angle_thresh_deg: float = 30.0) -> list:
    """頂点列から急な折り返し（Uターン）の頂点を除去する。

    3点 A-B-C について、進行方向 A→B と B→C の角度差を計算。
    角度差が (180 - angle_thresh_deg) 度を超える場合（ほぼUターン）に B を除去。
    angle_thresh_deg=30 なら、150度以上の方向転換（ほぼ折り返し）のみ除去。
    """
    if len(verts) <= 2:
        return list(verts)

    changed = True
    result = list(verts)
    while changed:
        changed = False
        new_result = [result[0]]
        for i in range(1, len(result) - 1):
            a = result[i - 1]
            b = result[i]
            c = result[i + 1]
            # 進行方向ベクトル
            ab = (b[0] - a[0], b[1] - a[1])
            bc = (c[0] - b[0], c[1] - b[1])
            len_ab = math.sqrt(ab[0]**2 + ab[1]**2)
            len_bc = math.sqrt(bc[0]**2 + bc[1]**2)
            if len_ab < 1e-9 or len_bc < 1e-9:
                new_result.append(result[i])
                continue
            # 進行方向の cos（1.0=直進、-1.0=Uターン）
            cos_fwd = (ab[0]*bc[0] + ab[1]*bc[1]) / (len_ab * len_bc)
            cos_fwd = max(-1.0, min(1.0, cos_fwd))
            angle_deg = math.degrees(math.acos(cos_fwd))
            # angle_deg が大きい = 方向転換が大きい = 折り返し
            if angle_deg > (180.0 - angle_thresh_deg):
                changed = True  # この点を除去（Uターン）
            else:
                new_result.append(result[i])
        new_result.append(result[-1])
        result = new_result
        if len(result) <= 2:
            break

    return result


def _resample_by_curvature(verts: list, min_pts: int = 4, max_pts: int = 10) -> list:
    """頂点列をスプライン補間し、曲率に応じて min_pts〜max_pts の制御点に再サンプリングする。

    直線に近い → min_pts、曲率が大きい → max_pts。
    """
    if len(verts) < 2:
        return list(verts)

    pts = np.array(verts, dtype=np.float64)
    n = len(pts)

    # スプライン補間で密な点列を生成
    dists = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dists[i] = dists[i-1] + np.linalg.norm(pts[i] - pts[i-1])
    total_len = dists[-1]
    if total_len < 1e-6:
        return list(verts)

    n_dense = max(100, n * 20)
    t_dense = np.linspace(0, total_len, n_dense)

    if n >= 3:
        try:
            from scipy.interpolate import CubicSpline
            dense = np.zeros((n_dense, 3), dtype=np.float64)
            # 重複除去
            mask = np.ones(n, dtype=bool)
            for i in range(1, n):
                if dists[i] - dists[i-1] < 1e-9:
                    mask[i] = False
            pts_c = pts[mask]; dists_c = dists[mask]
            if len(pts_c) < 3:
                return list(verts)
            for ax in range(3):
                cs = CubicSpline(dists_c, pts_c[:, ax], bc_type='natural')
                dense[:, ax] = cs(t_dense)
        except (ImportError, ValueError):
            return list(verts)
    else:
        return list(verts)

    # 曲率を計算（各点での方向変化の平均）
    angles = []
    for i in range(1, n_dense - 1):
        d1 = dense[i] - dense[i-1]
        d2 = dense[i+1] - dense[i]
        l1 = np.linalg.norm(d1[:2])
        l2 = np.linalg.norm(d2[:2])
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(d1[:2], d2[:2]) / (l1 * l2), -1, 1)
        angles.append(math.acos(cos_a))

    avg_curvature = np.mean(angles) if angles else 0.0
    # 曲率 0 → min_pts、曲率 0.05rad(≈3°) 以上 → max_pts
    t = min(1.0, avg_curvature / 0.05)
    target_pts = int(min_pts + t * (max_pts - min_pts) + 0.5)
    target_pts = max(min_pts, min(max_pts, target_pts))

    # 等間隔で再サンプリング
    indices = np.linspace(0, n_dense - 1, target_pts).astype(int)
    result = [(float(dense[i][0]), float(dense[i][1]), float(dense[i][2]))
              for i in indices]
    return result


import random as _random

def _generate_random_color():
    """視認性の高いランダムカラーを生成する"""
    h = _random.random()
    s = 0.6 + _random.random() * 0.4   # 0.6-1.0
    v = 0.7 + _random.random() * 0.3   # 0.7-1.0
    # HSV → RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)


def _lane_color(rec: dict):
    """rec のランダムカラー or class 色を返す"""
    c = rec.get("_color")
    if c:
        return c
    return CLASS_DISPLAY_COLORS.get(rec.get("class", ""), (0.7, 0.7, 0.7))


def _screen_to_world_ray(gl, screen_pt, ground_z: float = 0.0):
    """GL ウィジェットのスクリーン座標を地面 Z 平面上の 3D 座標に変換する。

    2Dモード:
        XY は実際の表示面 Z=0 との交点から取得する。
        データ保持用の Z 値は ground_z を維持する。
    3Dモード:
        ground_z 平面との交点をそのまま返す。
    """
    _stw = getattr(gl, '_screen_to_world', None)
    if _stw is not None:
        # ★ plane_z 引数に対応した新しい呼び出し方
        # 2Dモードでは _screen_to_world 内で自動的に Z=0 平面が使われる
        # 3Dモードでは ground_z 平面を使う
        import inspect
        sig = inspect.signature(_stw)
        if 'plane_z' in sig.parameters:
            result = _stw(screen_pt, plane_z=ground_z)
        else:
            # 旧 API（plane_z 未対応）へのフォールバック
            bev = getattr(gl, '_bev_image', None)
            orig_gz = None
            if bev is not None and hasattr(bev, 'ground_z'):
                orig_gz = bev.ground_z
                bev.ground_z = ground_z
            try:
                result = _stw(screen_pt)
            finally:
                if bev is not None and orig_gz is not None:
                    bev.ground_z = orig_gz

        if result is None:
            return None

        # 2Dモードの場合: XY は Z=0 表示面の正確な位置、Z はデータ保持用
        if getattr(gl, '_view_mode', '2d') == '2d':
            return (float(result[0]), float(result[1]), float(ground_z))

        return result

    # フォールバック: BevGLWidget._screen_to_world が使えない場合
    w = getattr(gl, '_viewport_w', None) or gl.width() or 1
    h = getattr(gl, '_viewport_h', None) or gl.height() or 1

    # 2Dモードでは表示面 Z=0 でXYを決定
    target_z = (
        0.0
        if getattr(gl, '_view_mode', '2d') == '2d'
        else ground_z
    )

    sx, sy = float(screen_pt.x()), float(screen_pt.y())
    dist   = 3.0 / (gl.zoom * gl._scale + 1e-9)
    fovy   = 45.0
    aspect = w / max(h, 1)
    f      = 1.0 / math.tan(math.radians(fovy / 2))
    xndc   = (2.0 * sx / w) - 1.0
    yndc   = 1.0 - (2.0 * sy / h)
    rx     = xndc * aspect / f
    ry     = yndc / f
    rz     = -1.0

    cx = math.radians(gl.rot_x)
    cz = math.radians(gl.rot_z)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(cx), -math.sin(cx)],
                   [0, math.sin(cx),  math.cos(cx)]])
    Rz = np.array([[math.cos(cz), -math.sin(cz), 0],
                   [math.sin(cz),  math.cos(cz), 0],
                   [0,             0,             1]])
    R = Rx @ Rz

    cam_pos_world = (
        R.T @ np.array([0.0, 0.0, dist])
        + gl._center
        - np.array([gl.pan_x, gl.pan_y, 0.0])
    )
    ray_dir_world = R.T @ np.array([rx, ry, rz])
    ray_dir_world /= (np.linalg.norm(ray_dir_world) + 1e-12)

    dz = ray_dir_world[2]
    if abs(dz) < 1e-9:
        return None
    t = (target_z - cam_pos_world[2]) / dz
    if t < 0:
        return None
    hit = cam_pos_world + t * ray_dir_world

    if getattr(gl, '_view_mode', '2d') == '2d':
        return (float(hit[0]), float(hit[1]), float(ground_z))
    return (float(hit[0]), float(hit[1]), float(hit[2]))


# =========================================================================
# LaneLinePanel 本体
# =========================================================================
class LaneLinePanel(QWidget):
    """ポリライン/ポリゴン描画 + 属性編集を1画面で完結させるパネル。"""

    # シグナル
    request_draw_mode = pyqtSignal(str)   # "polyline" | "polygon" | ""

    # スタイル定数
    _GS = ("QGroupBox{color:#bbc;font-size:11px;font-weight:bold;"
           "border:1px solid #3a3a5a;border-radius:4px;margin-top:8px;}"
           "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}")
    _LS = "color:#c0c8e8;font-size:11px;"
    _BTN_OFF_PL = (
        "QPushButton{background:rgba(26,42,26,80);border:1px solid #3a7a3a;"
        "color:#4a8a4a;font-size:11px;padding:4px 8px;border-radius:3px;}"
        "QPushButton:hover{background:rgba(34,50,34,120);color:#88dd88;}")
    _BTN_ON_PL = (
        "QPushButton{background:#2a6a2a;border:2px solid #50ee50;"
        "color:#ccffcc;font-size:11px;padding:4px 8px;border-radius:3px;"
        "font-weight:bold;}"
        "QPushButton:hover{background:#3a7a3a;}")
    _BTN_OFF_PG = (
        "QPushButton{background:rgba(26,26,42,80);border:1px solid #3a3a7a;"
        "color:#4a4a8a;font-size:11px;padding:4px 8px;border-radius:3px;}"
        "QPushButton:hover{background:rgba(34,34,50,120);color:#8888dd;}")
    _BTN_ON_PG = (
        "QPushButton{background:#2a2a6a;border:2px solid #5050ee;"
        "color:#ccccff;font-size:11px;padding:4px 8px;border-radius:3px;"
        "font-weight:bold;}"
        "QPushButton:hover{background:#3a3a7a;}")

    def __init__(self, gl, main_win=None, parent=None):
        super().__init__(parent)
        self.gl       = gl
        self.main_win = main_win

        # Lane データ
        self._lanes: list = []
        self._next_id     = 1

        # 描画状態
        self._draw_mode    = ""        # "polyline" | "polygon" | ""
        self._cur_verts_2d = []
        self._cur_verts_3d = []
        self._use_spline   = False     # スプライン補間使用フラグ（デフォルト: オフ=折れ線）

        # 頂点編集状態
        self._edit_lane_idx  = None
        self._edit_vert_idx  = None
        self._edit_dragging  = False
        self._selection_order = []   # 選択順を記録するリスト (行インデックス)

        # 選択元フラグ:
        # True の間は 2D/3Dビュー上のアノテーションクリックによる選択。
        # この場合、テーブル選択状態は同期するが2Dビュー中心は移動しない。
        self._selection_from_gl_click: bool = False

        # Ctrl+Z アンドゥ用（1操作分のみ保持）
        self._undo_state = None  # (lane_idx, old_vertices) のタプル

        # ベースレイヤー (Z値参照用)
        self._base_layer     = None
        self._base_combo_data = []

        # setting.ini 互換
        self._line_types = []
        self._lane_ids   = []

        # 直前に確定した属性 (次回のデフォルト)
        self._last_class     = "lane_line"
        self._last_attrs     = {}

        # 属性ウィジェット参照 (動的生成)
        self._attr_widgets: dict = {}   # attr_name -> QComboBox | QLineEdit

        # BEV 背景画像 (3画像対応: composite / rgb / z_gradient)
        self._bev_images: dict = {"composite": None, "rgb": None, "z_gradient": None}
        self._bev_mode: str    = "composite"   # 現在表示中のモード
        self._bev_alpha: float = 0.85          # 不透明度 (0.0-1.0)
        self._bev_meta: dict   = {}            # 画像メタ情報 (origin_x/y, resolution)
        self._pcd_dir: str     = ""            # PCD ファイルのあるフォルダ

        # Laneポリゴンファイルパス (自動生成ソース)
        self._polygon_file_path: str = ""

        self._build()
        self._load_ini()

    # ==================================================================
    # UI 構築
    # ==================================================================
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # タイトル
        title = QLabel("🛣  Lane Line Editor")
        title.setStyleSheet(
            "color:#a0c8ff;font-weight:bold;font-size:13px;"
            "border-bottom:1px solid #3a3a6a;padding-bottom:4px;")
        root.addWidget(title)

        # ── ① Lane ポリライン自動生成（sampling_lane_global.txt から） ───
        grp_io = QGroupBox(t("llp_grp_autogen"))
        grp_io.setStyleSheet(self._GS)
        io_lay = QVBoxLayout(grp_io)
        io_lay.setContentsMargins(6, 12, 6, 6); io_lay.setSpacing(6)

        # クラスターファイル（非表示で互換保持）
        self._cmb_cluster = QComboBox()
        self._cmb_cluster.setVisible(False)
        self._cmb_cluster_paths = []

        # 全自動生成 + 半自動生成 ボタンを横並び
        autogen_row = QHBoxLayout(); autogen_row.setSpacing(6)

        # 左側: 全自動生成ボタン（現在無効化中）
        self._btn_full_autogen = QPushButton(t("llp_btn_full_autogen"))
        self._btn_full_autogen.setStyleSheet(
            "QPushButton{background:#2a2a2a;border:1px solid #444;color:#555;"
            "font-size:11px;padding:4px 10px;border-radius:3px;}"
        )
        self._btn_full_autogen.setToolTip("全自動生成は現在無効化されています")
        self._btn_full_autogen.setEnabled(False)
        # self._btn_full_autogen.clicked.connect(self._auto_generate_lanes)
        autogen_row.addWidget(self._btn_full_autogen, stretch=1)

        # 右側: 半自動生成ボタン
        self._btn_semi_autogen = QPushButton(t("llp_btn_semi_autogen"))
        self._btn_semi_autogen.setStyleSheet(
            "QPushButton{background:#1a2a3a;border:1px solid #3a6a8a;color:#7abcde;"
            "font-size:11px;padding:4px 10px;border-radius:3px;}"
            "QPushButton:hover{background:#2a3a4a;color:#aaddff;}")
        self._btn_semi_autogen.setToolTip(t("tooltip_semi_autogen"))
        self._btn_semi_autogen.clicked.connect(self._semi_auto_generate_lanes)
        autogen_row.addWidget(self._btn_semi_autogen, stretch=1)

        io_lay.addLayout(autogen_row)
        root.addWidget(grp_io)

        # ── ② 描画ツール ─────────────────────────────────────────
        grp_draw = QGroupBox(t("llp_grp_draw"))
        grp_draw.setStyleSheet(self._GS)
        draw_lay = QVBoxLayout(grp_draw)
        draw_lay.setContentsMargins(6, 12, 6, 6); draw_lay.setSpacing(6)

        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self._btn_polyline = QPushButton(t("llp_btn_polyline"))
        self._btn_polyline.setCheckable(True)
        self._btn_polyline.setFixedHeight(28)
        self._btn_polyline.setStyleSheet(self._BTN_OFF_PL)
        self._btn_polyline.setToolTip(t("llp_tooltip_polyline"))
        self._btn_polyline.clicked.connect(lambda: self._toggle_draw("polyline"))
        btn_row.addWidget(self._btn_polyline)

        self._btn_polygon = QPushButton(t("llp_btn_polygon"))
        self._btn_polygon.setCheckable(True)
        self._btn_polygon.setFixedHeight(28)
        self._btn_polygon.setStyleSheet(self._BTN_OFF_PG)
        self._btn_polygon.setToolTip(t("llp_tooltip_polygon"))
        self._btn_polygon.clicked.connect(lambda: self._toggle_draw("polygon"))
        btn_row.addWidget(self._btn_polygon)

        _BTN_OFF_BB = ("QPushButton{background:rgba(42,26,42,80);border:1px solid #7a3a7a;"
                       "color:#8a4a8a;font-size:11px;padding:4px 8px;border-radius:3px;}"
                       "QPushButton:hover{background:rgba(50,34,50,120);color:#dd88dd;}")
        _BTN_ON_BB  = ("QPushButton{background:#6a2a6a;border:2px solid #ee50ee;"
                       "color:#ffccff;font-size:11px;padding:4px 8px;border-radius:3px;"
                       "font-weight:bold;}"
                       "QPushButton:hover{background:#7a3a7a;}")
        self._BTN_OFF_BB = _BTN_OFF_BB
        self._BTN_ON_BB  = _BTN_ON_BB
        self._btn_rbbox = QPushButton(t("llp_btn_point"))
        self._btn_rbbox.setCheckable(True)
        self._btn_rbbox.setFixedHeight(28)
        self._btn_rbbox.setStyleSheet(_BTN_OFF_BB)
        self._btn_rbbox.setToolTip(t("llp_tooltip_point"))
        self._btn_rbbox.clicked.connect(lambda: self._toggle_draw("point"))
        btn_row.addWidget(self._btn_rbbox)

        draw_lay.addLayout(btn_row)

        self._lbl_hint = QLabel(t("llp_hint_default"))
        self._lbl_hint.setStyleSheet("color:#669966;font-size:10px;")
        self._lbl_hint.setWordWrap(True)
        draw_lay.addWidget(self._lbl_hint)

        # 操作ボタン（削除・分割・統合）— 高さ統一
        _OP_BTN = ("QPushButton{font-size:11px;padding:4px 10px;"
                   "border-radius:4px;min-height:22px;max-height:22px;}")
        op_row = QHBoxLayout(); op_row.setSpacing(6)
        btn_del = QPushButton(t("llp_btn_delete"))
        btn_del.setFixedHeight(28)
        btn_del.setStyleSheet(_OP_BTN + "QPushButton{background:#3a1a1a;"
                              "border:1px solid #7a3a3a;color:#dd8888;}"
                              "QPushButton:hover{background:#4a2a2a;}")
        btn_del.clicked.connect(self._delete_selected)
        op_row.addWidget(btn_del)
        btn_split = QPushButton(t("llp_btn_split"))
        btn_split.setFixedHeight(28)
        btn_split.setStyleSheet(_OP_BTN + "QPushButton{background:#1a2a3a;"
                                "border:1px solid #3a6a8a;color:#88bbdd;}"
                                "QPushButton:hover{background:#2a3a4a;}")
        btn_split.clicked.connect(self._split_selected)
        op_row.addWidget(btn_split)
        btn_merge = QPushButton(t("llp_btn_merge"))
        btn_merge.setFixedHeight(28)
        btn_merge.setStyleSheet(_OP_BTN + "QPushButton{background:#1a3a1a;"
                                "border:1px solid #3a7a3a;color:#88dd88;}"
                                "QPushButton:hover{background:#2a4a2a;}")
        btn_merge.clicked.connect(self._merge_selected)
        op_row.addWidget(btn_merge)
        draw_lay.addLayout(op_row)

        # スプライン補間チェックボックス（デフォルト: オフ = 折れ線）
        self._chk_spline = QCheckBox(t("llp_chk_spline"))
        self._chk_spline.setChecked(False)   # デフォルト: 折れ線
        self._chk_spline.setStyleSheet(
            "QCheckBox{color:#aaa;font-size:10px;}"
            "QCheckBox::indicator{width:13px;height:13px;}"
        )
        self._chk_spline.setToolTip(t("tooltip_spline"))
        self._chk_spline.toggled.connect(self._on_spline_toggled)
        draw_lay.addWidget(self._chk_spline)

        root.addWidget(grp_draw)

        # ── ③ 属性パネル (選択中Lane用) ──────────────────────────
        grp_attr = QGroupBox(t("llp_grp_attr"))
        grp_attr.setStyleSheet(self._GS)
        self._attr_layout = QVBoxLayout(grp_attr)
        self._attr_layout.setContentsMargins(6, 12, 6, 6)
        self._attr_layout.setSpacing(2)

        # class 選択 (ボタン式) — ガイドライン準拠
        _CLS_LABELS = {
            "lane_line":              t("llp_cls_lane_line"),
            "stop_line":              t("llp_cls_stop_line"),
            "curb_boundary":          t("llp_cls_curb_boundary"),
            "fence_boundary":         t("llp_cls_fence_boundary"),
            "TrafficCone_boundary":   t("llp_cls_cone_boundary"),
            "WaterSafety_boundary":   t("llp_cls_water_boundary"),
            "boundary":               t("llp_cls_boundary"),
            "zebra_line":             t("llp_cls_zebra_line"),
            "split_point":            t("llp_cls_split_point"),
            "merge_point":            t("llp_cls_merge_point"),
        }
        _BTN_CLS_OFF = ("QPushButton{background:#3c3c3c;color:#ccc;"
                        "border:1px solid #555;border-radius:3px;"
                        "padding:3px 6px;font-size:11px;}"
                        "QPushButton:hover{background:#4a4a4a;color:white;}")
        _BTN_CLS_ON  = ("QPushButton{background:#2a6ebb;color:white;"
                        "border:2px solid #4a8edb;border-radius:3px;"
                        "padding:3px 6px;font-size:11px;font-weight:bold;}")
        _BTN_CLS_DIS = ("QPushButton{background:#2a2a2a;color:#555;"
                        "border:1px solid #333;border-radius:3px;"
                        "padding:3px 6px;font-size:11px;}")
        self._BTN_CLS_OFF = _BTN_CLS_OFF
        self._BTN_CLS_ON  = _BTN_CLS_ON
        self._BTN_CLS_DIS = _BTN_CLS_DIS
        self._CLS_LABELS  = _CLS_LABELS

        grp_cls = QGroupBox(t("llp_grp_class"))
        grp_cls.setStyleSheet(self._GS)
        cls_outer = QVBoxLayout(grp_cls)
        cls_outer.setContentsMargins(4, 10, 4, 4)
        cls_outer.setSpacing(2)
        # ポリライン用クラス（上段）
        cls_row_pl = QHBoxLayout(); cls_row_pl.setSpacing(2)
        # ポリゴン・点クラス（下段）
        cls_row_pg = QHBoxLayout(); cls_row_pg.setSpacing(2)
        self._cls_buttons = {}
        for cls_name in POLYLINE_CLASSES:
            label = _CLS_LABELS.get(cls_name, cls_name)
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setStyleSheet(_BTN_CLS_OFF)
            btn.clicked.connect(
                lambda checked, c=cls_name: self._on_class_btn(c))
            self._cls_buttons[cls_name] = btn
            cls_row_pl.addWidget(btn)
        for cls_name in POLYGON_CLASSES_LIST + POINT_CLASSES_LIST:
            label = _CLS_LABELS.get(cls_name, cls_name)
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setStyleSheet(_BTN_CLS_OFF)
            btn.clicked.connect(
                lambda checked, c=cls_name: self._on_class_btn(c))
            self._cls_buttons[cls_name] = btn
            cls_row_pg.addWidget(btn)
        # デフォルト選択
        self._cls_buttons["lane_line"].setChecked(True)
        self._cls_buttons["lane_line"].setStyleSheet(_BTN_CLS_ON)
        cls_outer.addLayout(cls_row_pl)
        cls_outer.addLayout(cls_row_pg)
        self._attr_layout.addWidget(grp_cls)

        # ID (N_X-Y 分離: 3フィールド)
        _ROW_H = 28  # 全ボタン・入力欄の統一高さ
        _FLD = (f"QLineEdit{{background:#2a2a2a;color:#ccc;border:1px solid #555;"
                f"border-radius:3px;padding:2px 6px;font-size:11px;"
                f"min-height:{_ROW_H - 6}px;max-height:{_ROW_H - 6}px;}}"
                f"QLineEdit:focus{{border:1px solid #4a8edb;}}")
        _LBL = "color:#999;font-size:10px;"

        r_id = QHBoxLayout(); r_id.setSpacing(4)
        lbl_n = QLabel("N:"); lbl_n.setStyleSheet(_LBL); lbl_n.setFixedWidth(16)
        r_id.addWidget(lbl_n)
        self._edt_id_n = QLineEdit()
        self._edt_id_n.setStyleSheet(_FLD)
        self._edt_id_n.setFixedHeight(_ROW_H)
        self._edt_id_n.setPlaceholderText(t("llp_placeholder_id_n"))
        self._edt_id_n.setToolTip(t("llp_tooltip_id_n"))
        self._edt_id_n.editingFinished.connect(self._on_id_field_changed)
        r_id.addWidget(self._edt_id_n)

        lbl_x = QLabel("X:"); lbl_x.setStyleSheet(_LBL); lbl_x.setFixedWidth(14)
        r_id.addWidget(lbl_x)
        self._edt_id_x = QLineEdit()
        self._edt_id_x.setStyleSheet(_FLD)
        self._edt_id_x.setFixedHeight(_ROW_H)
        self._edt_id_x.setPlaceholderText(t("llp_placeholder_id_x"))
        self._edt_id_x.setToolTip(t("llp_tooltip_id_x"))
        self._edt_id_x.editingFinished.connect(self._on_id_field_changed)
        r_id.addWidget(self._edt_id_x)

        lbl_y = QLabel("Y:"); lbl_y.setStyleSheet(_LBL); lbl_y.setFixedWidth(14)
        r_id.addWidget(lbl_y)
        self._edt_id_y = QLineEdit()
        self._edt_id_y.setStyleSheet(_FLD)
        self._edt_id_y.setFixedHeight(_ROW_H)
        self._edt_id_y.setPlaceholderText(t("llp_placeholder_id_y"))
        self._edt_id_y.setToolTip(t("llp_tooltip_id_y"))
        self._edt_id_y.editingFinished.connect(self._on_id_field_changed)
        r_id.addWidget(self._edt_id_y)

        self._attr_layout.addLayout(r_id)
        self._ROW_H = _ROW_H  # 他のメソッドから参照用

        # 動的属性フィールド用コンテナ
        self._attr_container = QWidget()
        self._attr_container_layout = QVBoxLayout(self._attr_container)
        self._attr_container_layout.setContentsMargins(0, 0, 0, 0)
        self._attr_container_layout.setSpacing(2)
        self._attr_layout.addWidget(self._attr_container)

        # note (自由記述)
        note_row = QHBoxLayout(); note_row.setSpacing(4)
        lbl_note = QLabel("note:"); lbl_note.setStyleSheet(self._LS)
        lbl_note.setFixedWidth(38)
        note_row.addWidget(lbl_note)
        self._edt_note = QLineEdit()
        self._edt_note.setStyleSheet(
            f"font-size:11px;background:#1c1c2e;"
            f"border:1px solid #3a3a50;color:#c8c8e8;"
            f"min-height:{_ROW_H - 6}px;max-height:{_ROW_H - 6}px;")
        self._edt_note.setFixedHeight(_ROW_H)
        self._edt_note.setPlaceholderText(t("llp_placeholder_note"))
        self._edt_note.editingFinished.connect(self._on_note_changed)
        note_row.addWidget(self._edt_note, stretch=1)
        self._attr_layout.addLayout(note_row)

        root.addWidget(grp_attr)

        # ── ④ Lane 一覧テーブル ──────────────────────────────────
        grp_tbl = QGroupBox(t("llp_grp_table"))
        grp_tbl.setStyleSheet(self._GS)
        tbl_lay = QVBoxLayout(grp_tbl)
        tbl_lay.setContentsMargins(4, 10, 4, 4); tbl_lay.setSpacing(4)

        # 色塗り方法ボタン
        color_row = QHBoxLayout(); color_row.setSpacing(4)
        lbl_cm = QLabel(t("llp_lbl_color_mode"))
        lbl_cm.setStyleSheet("color:#999;font-size:10px;")
        color_row.addWidget(lbl_cm)
        _CM_OFF = ("QPushButton{background:#3c3c3c;color:#ccc;"
                   "border:1px solid #555;border-radius:3px;"
                   "padding:2px 10px;font-size:11px;}"
                   "QPushButton:hover{background:#4a4a4a;}")
        _CM_ON  = ("QPushButton{background:#2a6ebb;color:white;"
                   "border:2px solid #4a8edb;border-radius:3px;"
                   "padding:2px 10px;font-size:11px;font-weight:bold;}")
        self._CM_OFF = _CM_OFF
        self._CM_ON  = _CM_ON
        self._btn_color_class = QPushButton(t("llp_btn_color_class"))
        self._btn_color_class.setCheckable(True)
        self._btn_color_class.setFixedHeight(28)
        self._btn_color_class.setStyleSheet(_CM_OFF)
        self._btn_color_class.clicked.connect(lambda: self._set_color_mode("class"))
        color_row.addWidget(self._btn_color_class)
        self._btn_color_id = QPushButton(t("llp_btn_color_id"))
        self._btn_color_id.setCheckable(True)
        self._btn_color_id.setChecked(True)
        self._btn_color_id.setFixedHeight(28)
        self._btn_color_id.setStyleSheet(_CM_ON)
        self._btn_color_id.clicked.connect(lambda: self._set_color_mode("id"))
        color_row.addWidget(self._btn_color_id)
        color_row.addStretch()

        # 右端: チェック済み行の削除ボタン
        self._btn_delete_checked = QPushButton(t("btn_delete_checked"))
        self._btn_delete_checked.setFixedHeight(28)
        self._btn_delete_checked.setStyleSheet(
            "QPushButton{background:#3a1a1a;color:#dd8888;"
            "border:1px solid #7a3a3a;border-radius:3px;"
            "font-size:11px;padding:2px 10px;}"
            "QPushButton:hover{background:#4a2a2a;color:#ffaaaa;}")
        self._btn_delete_checked.setToolTip(t("tooltip_delete_checked"))
        self._btn_delete_checked.clicked.connect(self._delete_checked_rows)
        color_row.addWidget(self._btn_delete_checked)

        tbl_lay.addLayout(color_row)
        self._color_mode = "id"  # デフォルト: ID別（ランダムカラー）

        # N 自動付与ボタン（テーブル先頭）
        self._btn_auto_n = QPushButton(t("llp_btn_auto_assign_n"))
        self._btn_auto_n.setFixedHeight(24)
        self._btn_auto_n.setStyleSheet(
            "QPushButton{background:#1a3a1a;color:#88ff88;"
            "border:1px solid #2a6a2a;border-radius:3px;"
            "font-size:11px;font-weight:bold;}"
            "QPushButton:hover{background:#2a4a2a;}")
        self._btn_auto_n.clicked.connect(self._on_btn_auto_n)
        tbl_lay.addWidget(self._btn_auto_n)

        self._tbl = QTableWidget(0, 9)
        # 列: ✓(0), 型(1), N(2), X(3), Y(4), class(5), 線種(6), 色(7), 頂点数(8)
        self._tbl.setHorizontalHeaderLabels(
            ["✓", t("tbl_hdr_type"), "N", "X", "Y", "class",
             "線種", "色", t("llp_tbl_hdr_verts")])
        hdr = self._tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Fixed)
        self._tbl.setColumnWidth(0, 24)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.Stretch)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        # ヘッダクリックで昇順/降順ソート（_lanes を直接並べ替え）
        hdr.setSectionsClickable(True)
        self._tbl_sort_col: int = -1
        self._tbl_sort_asc: bool = True
        hdr.sectionClicked.connect(self._on_tbl_header_clicked)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tbl.setEditTriggers(QAbstractItemView.DoubleClicked)
        self._tbl.setStyleSheet(
            "QTableWidget{background:#0e0e18;color:#c8c8e8;font-size:10px;"
            "gridline-color:#2a2a3a;}"
            "QHeaderView::section{background:#1a1a2e;color:#88aacc;"
            "font-size:10px;border:1px solid #2a2a3a;}"
            "QHeaderView::section:hover{background:#2a2a4e;}")
        self._tbl.setMinimumHeight(100)
        self._tbl.itemSelectionChanged.connect(self._on_tbl_select)
        self._tbl.cellChanged.connect(self._on_tbl_cell_changed)
        tbl_lay.addWidget(self._tbl)
        root.addWidget(grp_tbl, stretch=1)

        # 初期属性フィールド構築
        self._rebuild_attr_fields("lane_line")

    # ==================================================================
    # BEV 背景画像 (3画像: composite / rgb / z_gradient)
    # ==================================================================

    def set_pcd_path(self, pcd_path: str):
        """外部（MainWindow等）からPCDパスを通知してBEV画像を自動検出する。"""
        self._pcd_dir = str(Path(pcd_path).parent)
        self._auto_detect_bev(silent=True)

    def _auto_detect_bev(self, silent: bool = False):
        """PCDフォルダ内の *_rgb.png / *_composite.png / *_z_gradient.png を自動検出。"""
        search_dirs = []
        if self._pcd_dir:
            search_dirs.append(Path(self._pcd_dir))
        if self.main_win is not None:
            try:
                for _, (layer, _) in self.main_win.solo_layers.items():
                    if hasattr(layer, 'filepath') and layer.filepath:
                        search_dirs.append(Path(layer.filepath).parent)
                for _, (group, _) in self.main_win.groups.items():
                    if group.layers:
                        search_dirs.append(Path(group.layers[0].filepath).parent)
            except Exception:
                pass

        composite_path = rgb_path = z_gradient_path = None
        for d in search_dirs:
            if not d.exists():
                continue
            for p in sorted(d.glob("*_composite.png")):
                composite_path = p; break
            for p in sorted(d.glob("*_rgb.png")):
                rgb_path = p; break
            for p in sorted(d.glob("*_z_gradient.png")):
                z_gradient_path = p; break
            if composite_path or rgb_path or z_gradient_path:
                break

        if not composite_path and not rgb_path and not z_gradient_path:
            if not silent:
                QMessageBox.information(
                    self, "BEV 自動検出",
                    "同フォルダに *_rgb.png / *_composite.png / *_z_gradient.png が見つかりません。\n"
                    "「開く」で手動指定してください。")
            return
        self._load_bev_files(composite_path, rgb_path, z_gradient_path)

    def _load_bev_image(self):
        """BEV 画像を開く。1つ選べば同フォルダから残りを自動補完、不足分は個別ダイアログ。"""
        init_dir = self._pcd_dir or ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "BEV 画像を選択（1つ以上選択。残りは自動補完します）",
            init_dir, "PNG Files (*.png);;All (*)")
        if not paths:
            return

        composite_path = rgb_path = z_gradient_path = None
        unmatched = []
        for p in paths:
            pp = Path(p)
            name = pp.name.lower()
            if "composite" in name:
                composite_path = pp
            elif "z_gradient" in name or "zgrad" in name:
                z_gradient_path = pp
            elif "rgb" in name:
                rgb_path = pp
            else:
                unmatched.append(pp)
        for pp in unmatched:
            if   composite_path  is None: composite_path  = pp
            elif rgb_path        is None: rgb_path        = pp
            elif z_gradient_path is None: z_gradient_path = pp

        # 同フォルダから不足分を自動補完
        search_dir = next((p.parent for p in [composite_path, rgb_path, z_gradient_path]
                           if p is not None), None)
        if search_dir is not None:
            if composite_path  is None:
                found = sorted(search_dir.glob("*_composite.png"))
                if found: composite_path  = found[0]
            if rgb_path        is None:
                found = sorted(search_dir.glob("*_rgb.png"))
                if found: rgb_path        = found[0]
            if z_gradient_path is None:
                found = sorted(search_dir.glob("*_z_gradient.png"))
                if found: z_gradient_path = found[0]

        # それでも不足している種別は個別ダイアログ
        missing = {
            "composite":  composite_path  is None,
            "rgb":        rgb_path        is None,
            "z_gradient": z_gradient_path is None,
        }
        _LABELS = {
            "composite":  ("Composite",  "構造把握用（RGB + Z-Gradient合成）"),
            "rgb":        ("RGB",        "線色判別用（白/黄が最も鮮明）"),
            "z_gradient": ("Z-Gradient", "縁石・段差判別用"),
        }
        if any(missing.values()):
            missing_names = [_LABELS[k][0] for k, v in missing.items() if v]
            ret = QMessageBox.question(
                self, "BEV 画像の補完",
                "以下の画像が見つかりませんでした:\n  " +
                "\n  ".join(f"・{n}" for n in missing_names) +
                "\n\nそれぞれのファイルを個別に指定しますか？\n"
                "（「いいえ」を選ぶと見つかった画像のみで続行します）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                base_dir = str(search_dir) if search_dir else init_dir
                if missing["composite"]:
                    p, _ = QFileDialog.getOpenFileName(
                        self, f"Composite 画像を選択（{_LABELS['composite'][1]}）",
                        base_dir, "PNG Files (*.png);;All (*)")
                    if p: composite_path  = Path(p)
                if missing["rgb"]:
                    p, _ = QFileDialog.getOpenFileName(
                        self, f"RGB 画像を選択（{_LABELS['rgb'][1]}）",
                        base_dir, "PNG Files (*.png);;All (*)")
                    if p: rgb_path        = Path(p)
                if missing["z_gradient"]:
                    p, _ = QFileDialog.getOpenFileName(
                        self, f"Z-Gradient 画像を選択（{_LABELS['z_gradient'][1]}）",
                        base_dir, "PNG Files (*.png);;All (*)")
                    if p: z_gradient_path = Path(p)

        self._load_bev_files(composite_path, rgb_path, z_gradient_path)

    def _load_bev_files(self, composite_path, rgb_path, z_gradient_path):
        """composite / rgb / z_gradient の画像ファイルを numpy 配列として読み込む。"""
        try:
            import cv2
        except ImportError:
            QMessageBox.critical(self, "BEV 読み込み",
                "OpenCV が必要です: pip install opencv-python")
            return

        def _read(p):
            if p is None: return None
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None: return None
            if img.ndim == 3 and img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            elif img.ndim == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        comp_img = _read(composite_path)
        rgb_img  = _read(rgb_path)
        zg_img   = _read(z_gradient_path)

        self._bev_images["composite"]  = comp_img
        self._bev_images["rgb"]        = rgb_img
        self._bev_images["z_gradient"] = zg_img

        # メタ情報 JSON を検索
        self._bev_meta = {}
        for p in [composite_path, rgb_path, z_gradient_path]:
            if p is None: continue
            meta_p = p.with_suffix(".json")
            if meta_p.exists():
                try:
                    import json as _json
                    with open(meta_p, "r", encoding="utf-8") as f:
                        self._bev_meta = _json.load(f)
                    break
                except Exception:
                    pass

        # 表示モード: composite → rgb → z_gradient の優先順
        if   comp_img is not None: self._bev_mode = "composite"
        elif rgb_img  is not None: self._bev_mode = "rgb"
        elif zg_img   is not None: self._bev_mode = "z_gradient"

        self._apply_bev_to_gl()

        # ステータス表示
        parts = []
        if comp_img is not None: parts.append(f"composite ({comp_img.shape[1]}×{comp_img.shape[0]})")
        if rgb_img  is not None: parts.append(f"rgb ({rgb_img.shape[1]}×{rgb_img.shape[0]})")
        if zg_img   is not None: parts.append(f"z_gradient ({zg_img.shape[1]}×{zg_img.shape[0]})")
        if parts:
            self._lbl_bev_status.setText(t("llp_bev_loaded") + " " + " / ".join(parts))
            self._lbl_bev_status.setStyleSheet("color:#88cc88;font-size:10px;")
        else:
            self._lbl_bev_status.setText(t("llp_bev_fail"))
            self._lbl_bev_status.setStyleSheet("color:#cc8888;font-size:10px;")

        # ボタン有効/無効
        self._btn_bev_composite.setEnabled(comp_img is not None)
        self._btn_bev_rgb.setEnabled(rgb_img        is not None)
        self._btn_bev_zgrad.setEnabled(zg_img       is not None)
        if comp_img is None and rgb_img is not None:
            self._set_bev_mode("rgb")
        elif comp_img is None and rgb_img is None and zg_img is not None:
            self._set_bev_mode("z_gradient")

    def _apply_bev_to_gl(self):
        """現在のモード・アルファ値で GL ウィジェットへ背景画像をセットする。"""
        img = self._bev_images.get(self._bev_mode)
        if img is None:
            # フォールバック: 他のモードを試す
            for key in ("composite", "rgb", "z_gradient"):
                img = self._bev_images.get(key)
                if img is not None:
                    break
        if img is None:
            setattr(self.gl, 'bev_image', None)
        else:
            if img.ndim == 2:
                rgba = np.stack([img, img, img, np.full_like(img, 255)], axis=-1)
            elif img.shape[2] == 3:
                rgba = np.concatenate(
                    [img, np.full((*img.shape[:2], 1), 255, dtype=img.dtype)], axis=-1)
            else:
                rgba = img.copy()
            setattr(self.gl, 'bev_image', rgba)
        setattr(self.gl, 'bev_alpha', self._bev_alpha)
        setattr(self.gl, 'bev_meta',  self._bev_meta)
        if hasattr(self.gl, 'update'):
            self.gl.update()

    def _set_bev_mode(self, mode: str):
        """composite / rgb / z_gradient 表示モードを切り替える。"""
        self._bev_mode = mode
        for btn, key in [(self._btn_bev_composite, "composite"),
                         (self._btn_bev_rgb,        "rgb"),
                         (self._btn_bev_zgrad,      "z_gradient")]:
            btn.setChecked(mode == key)
            btn.setStyleSheet(self._BEV_ON if mode == key else self._BEV_OFF)
        self._apply_bev_to_gl()

    def _on_bev_alpha_changed(self, value: int):
        self._bev_alpha = value / 100.0
        self._lbl_bev_alpha_val.setText(f"{value}%")
        self._apply_bev_to_gl()

    def _clear_bev(self):
        self._bev_images = {"composite": None, "rgb": None, "z_gradient": None}
        self._bev_meta   = {}
        setattr(self.gl, 'bev_image', None)
        if hasattr(self.gl, 'update'):
            self.gl.update()
        self._lbl_bev_status.setText(t("llp_bev_not_loaded"))
        self._lbl_bev_status.setStyleSheet("color:#666;font-size:10px;")
        for btn in [self._btn_bev_composite, self._btn_bev_rgb, self._btn_bev_zgrad]:
            btn.setEnabled(True)

    # ==================================================================
    # Phase1・2 自動属性付与
    # ==================================================================

    def _auto_assign_attributes(self, lanes: list) -> int:
        """
        Phase 1 (確実): source / is_interpolated / coordinate_system / boundary_id
        Phase 2 (画像+点群): class縁石判別 / line_color白黄 / visibility
        戻り値: 更新したレコード数
        """
        if not lanes:
            return 0

        rgbbev = self._bev_images.get("rgb")        # 線色判別専用
        zg_img = self._bev_images.get("z_gradient") # 縁石判別専用
        meta   = self._bev_meta

        has_rgbbev = rgbbev is not None and rgbbev.ndim == 3 and rgbbev.shape[2] >= 3
        has_zgrad  = zg_img is not None

        layer   = self._base_layer
        has_pcd = (layer is not None
                   and getattr(layer, 'loaded', False)
                   and getattr(layer, 'xyz', None) is not None)

        # z_gradient Canny エッジ（縁石判別用）
        zg_edges = None
        if has_zgrad:
            try:
                import cv2
                gray = zg_img if zg_img.ndim == 2 else zg_img[:, :, 0]
                zg_edges = cv2.Canny(gray.astype(np.uint8), 30, 80)
            except Exception:
                pass

        # KDTree（点群密度用）
        kd_tree = None
        if has_pcd:
            try:
                from scipy.spatial import cKDTree
                kd_tree = cKDTree(layer.xyz[:, :2])
            except Exception:
                pass

        # boundary_id 連番（既存の最大値から続番）
        existing_ids = []
        for rec in self._lanes:
            try: existing_ids.append(int(rec.get("boundary_id", "")))
            except (ValueError, TypeError): pass
        next_boundary_id = (max(existing_ids) + 1) if existing_ids else 1

        n_updated = 0
        for rec in lanes:
            changed = False

            # ── Phase 1 ──────────────────────────────────────────
            if rec.get("source") != "auto_corrected":
                rec["source"] = "auto"; changed = True
            if not rec.get("is_interpolated"):
                rec["is_interpolated"] = "false"; changed = True
            if not rec.get("coordinate_system"):
                rec["coordinate_system"] = "ego_vehicle"; changed = True
            cls = rec.get("class", "lane_line")
            if cls in ("curb_boundary", "fence_boundary",
                       "TrafficCone_boundary", "WaterSafety_boundary", "boundary"):
                if not rec.get("boundary_id"):
                    rec["boundary_id"] = str(next_boundary_id)
                    next_boundary_id += 1; changed = True

            # ── Phase 2-A: 縁石判別 ──────────────────────────────
            if (zg_edges is not None and meta
                    and cls == "lane_line"
                    and rec.get("geometry_type") == "polyline"):
                if self._classify_curb_by_zgrad(rec, zg_edges, meta):
                    rec["class"] = "curb_boundary"
                    if not rec.get("boundary_id"):
                        rec["boundary_id"] = str(next_boundary_id)
                        next_boundary_id += 1
                    changed = True

            # ── Phase 2-B: 線色判別 ──────────────────────────────
            cur_cls = rec.get("class", "lane_line")
            if (has_rgbbev and meta and cur_cls == "lane_line"
                    and not rec.get("line_color")):
                color = self._classify_line_color(rec, rgbbev, meta)
                if color:
                    rec["line_color"] = color; changed = True

            # ── Phase 2-C: visibility ─────────────────────────────
            if kd_tree is not None and has_pcd:
                vis = self._estimate_visibility(rec, kd_tree, layer.xyz)
                if vis:
                    rec["visibility"] = vis; changed = True

            if changed:
                n_updated += 1
        return n_updated

    def _classify_curb_by_zgrad(self, rec: dict, zg_edges: np.ndarray,
                                 meta: dict) -> bool:
        """z_gradient Canny エッジ密度で縁石か車線線かを判別する。"""
        CURB_EDGE_RATIO  = 0.45
        SAMPLE_RADIUS_PX = 3
        pts_px = self._verts_to_pixel(rec, meta, zg_edges.shape)
        if pts_px is None or len(pts_px) < 2:
            return False
        try:
            import cv2
            mask = np.zeros(zg_edges.shape, dtype=np.uint8)
            cv2.polylines(mask, [pts_px.astype(np.int32)], False, 255,
                          thickness=SAMPLE_RADIUS_PX * 2)
            total = int(mask.sum() // 255)
            if total == 0: return False
            ratio = int(np.logical_and(zg_edges > 0, mask > 0).sum()) / total
            return ratio >= CURB_EDGE_RATIO
        except Exception:
            return False

    def _classify_line_color(self, rec: dict, rgbbev: np.ndarray,
                              meta: dict) -> str:
        """RGBBEV画像のHSV解析でwhite/yellow/""を返す。"""
        MIN_VALID_RATIO = 0.20
        SAMPLE_WIDTH_PX = 2
        pts_px = self._verts_to_pixel(rec, meta, rgbbev.shape[:2])
        if pts_px is None or len(pts_px) < 2:
            return ""
        try:
            import cv2
            mask = np.zeros(rgbbev.shape[:2], dtype=np.uint8)
            cv2.polylines(mask, [pts_px.astype(np.int32)], False, 255,
                          thickness=SAMPLE_WIDTH_PX * 2 + 1)
            hsv = cv2.cvtColor(rgbbev[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
            roi = hsv[mask > 0]
            if len(roi) == 0: return ""
            valid = roi[roi[:, 2] >= 30]
            if len(valid) == 0 or len(valid) / len(roi) < MIN_VALID_RATIO:
                return ""
            white  = ((valid[:, 1] < 60) & (valid[:, 2] > 160)).sum()
            yellow = ((valid[:, 0] >= 15) & (valid[:, 0] <= 35)
                      & (valid[:, 1] > 80)).sum()
            if white == 0 and yellow == 0: return ""
            return "white" if white >= yellow else "yellow"
        except Exception:
            return ""

    def _estimate_visibility(self, rec: dict, kd_tree,
                              xyz: np.ndarray) -> str:
        """点群密度からvisible/partially_occluded/occludedを推定する。"""
        RADIUS = 0.5; MIN_PTS = 3; OCC_RATIO = 0.6
        verts = rec.get("vertices", [])
        if not verts: return ""
        low_count = sum(
            1 for vx, vy, _ in verts
            if len(kd_tree.query_ball_point([vx, vy], RADIUS)) < MIN_PTS)
        ratio = low_count / len(verts)
        if ratio == 0.0:      return "visible"
        if ratio < OCC_RATIO: return "partially_occluded"
        return "occluded"

    def _verts_to_pixel(self, rec: dict, meta: dict,
                        img_shape: tuple) -> "np.ndarray | None":
        """ワールド座標→BEV画像ピクセル座標変換。メタJSONがあれば精確、なければ点群範囲から推定。"""
        verts = rec.get("vertices", [])
        if not verts: return None
        H, W = img_shape[:2]
        ox  = meta.get("origin_x")
        oy  = meta.get("origin_y")
        res = meta.get("resolution")
        if ox is None or oy is None or res is None:
            layer = self._base_layer
            if (layer is not None and getattr(layer, 'loaded', False)
                    and getattr(layer, 'xyz', None) is not None):
                xmin = float(layer.xyz[:, 0].min()); ymin = float(layer.xyz[:, 1].min())
                xmax = float(layer.xyz[:, 0].max()); ymax = float(layer.xyz[:, 1].max())
                res = max((xmax - xmin) / W, (ymax - ymin) / H, 1e-6)
                ox, oy = xmin, ymin
            else:
                xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
                margin = 5.0
                ox = min(xs) - margin; oy = min(ys) - margin
                span = max(max(xs) - min(xs), max(ys) - min(ys)) + margin * 2
                res = span / min(W, H) if min(W, H) > 0 else 1.0
        pts_px = []
        for vx, vy, _ in verts:
            px = max(0, min(W - 1, int((vx - ox) / res)))
            py = max(0, min(H - 1, int(H - (vy - oy) / res)))
            pts_px.append([[px, py]])
        return np.array(pts_px, dtype=np.int32) if pts_px else None

    # ==================================================================
    # setting.ini 読み込み (既存互換)
    # ==================================================================
    def _load_ini(self):
        """parse_setting_ini を呼んで line_types / lane_ids を取得する"""
        try:
            # pointcloud_viewer.py 内の parse_setting_ini を使う
            from pointcloud_viewer import parse_setting_ini
        except ImportError:
            # 同一ファイル内にいない場合のフォールバック
            parse_setting_ini = None

        if parse_setting_ini is None:
            return

        ini_path = Path(sys.argv[0]).parent / "setting.ini"
        if not ini_path.exists():
            ini_path = Path(__file__).parent / "setting.ini"
        if ini_path.exists():
            self._line_types, self._lane_ids = parse_setting_ini(str(ini_path))

    # ==================================================================
    # ベースレイヤーコンボ
    # ==================================================================
    def refresh_base_combo(self):
        """MainWindow から呼ぶ — クラスターファイルコンボも更新"""
        self.refresh_cluster_combo()

    def refresh_cluster_combo(self):
        """読み込み済みレイヤーのフォルダから *cluster.txt をスキャン"""
        self._cmb_cluster.blockSignals(True)
        self._cmb_cluster.clear()
        self._cmb_cluster_paths = []
        search_dirs = set()
        if self.main_win is not None:
            for _, (layer, _) in self.main_win.solo_layers.items():
                if hasattr(layer, 'filepath') and layer.filepath:
                    search_dirs.add(str(Path(layer.filepath).parent))
            for _, (group, _) in self.main_win.groups.items():
                if group.layers:
                    search_dirs.add(str(Path(group.layers[0].filepath).parent))
        found = []
        for d in search_dirs:
            try:
                for p in sorted(Path(d).glob("*cluster.txt")):
                    found.append(str(p))
            except Exception:
                pass
        if not found:
            self._cmb_cluster.addItem(t("llp_cluster_not_found"))
            self._cmb_cluster_paths = [None]
        else:
            for fp in found:
                self._cmb_cluster.addItem(Path(fp).name)
                self._cmb_cluster_paths.append(fp)
        self._cmb_cluster.blockSignals(False)

    def _select_polygon_file(self):
        """Laneポリゴンファイルを選択する。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Laneポリゴンファイルを選択",
            self._pcd_dir if self._pcd_dir else "",
            "Text Files (*.txt);;All (*)")
        if not path:
            return
        self._polygon_file_path = path
        self._lbl_polygon_file.setText(Path(path).name)
        self._lbl_polygon_file.setToolTip(path)
        self._lbl_polygon_file.setStyleSheet("color:#aac8ff;font-size:10px;")

    def _clear_polygon_file(self):
        """Laneポリゴンファイルの選択をクリアする。"""
        self._polygon_file_path = ""
        self._lbl_polygon_file.setText(t("llp_lbl_not_selected"))
        self._lbl_polygon_file.setToolTip("")
        self._lbl_polygon_file.setStyleSheet("color:#666;font-size:10px;")

    def _auto_generate_lanes(self):
        """Lane ポリライン自動生成。

        mapping_pose.txt が読み込まれている場合は gt_lane_generator を使用し、
        sampling_lane_global.json の各ラインを軌跡に最もフィットするように
        平行移動して精度の高いポリラインを生成する（交差点・車線変更に対応）。

        mapping_pose が未読み込みの場合は従来の sampling_lane_importer を使用する。

        再生成時は source=="sampling_auto" のレコード（前回の自動生成結果）を
        先に削除してから新たに生成する。手動で追加したレコードは変更しない。
        """
        # ── sampling_lane_hint を取得 ──────────────────────────────
        hint = None
        if self.main_win is not None:
            hint = getattr(self.main_win, '_sampling_lane_hint', None)

        if not hint or not Path(hint).exists():
            QMessageBox.warning(
                self, t("llp_grp_autogen"),
                t("warn_sampling_lane_not_found"))
            return

        # ── mapping_pose.txt のパスを取得 ─────────────────────────
        mapping_pose_path = None
        if self.main_win is not None:
            traj_path = getattr(self.main_win, '_trajectory_path', None)
            if traj_path and Path(traj_path).exists():
                mapping_pose_path = traj_path

        # ── εはデフォルト値を使用（εメニュー廃止のため固定値）──────
        eps_val = 1.0

        try:
            import numpy as np

            # ── 前回の自動生成レコードを削除（手動レコードは保持）──────
            before_count = len(self._lanes)
            self._lanes = [
                r for r in self._lanes
                if r.get("source") != "sampling_auto"
            ]
            removed = before_count - len(self._lanes)
            if removed:
                print(f"[LaneLinePanel] 前回の自動生成レコード {removed} 本を削除")

            # ── 自動生成ロジック選択 ────────────────────────────────
            if mapping_pose_path is not None:
                # 【新ロジック】mapping_pose × sampling_lane フィット方式
                print(f"[LaneLinePanel] gt_lane_generator を使用 "
                      f"(sampling={Path(hint).name}, pose={Path(mapping_pose_path).name})")
                from core.gt_lane_generator import generate_lanes_from_sampling
                new_records = generate_lanes_from_sampling(
                    sampling_lane_path=hint,
                    mapping_pose_path=mapping_pose_path,
                )
            else:
                # 【旧ロジック】mapping_pose 未読み込み時は従来方式にフォールバック
                print(f"[LaneLinePanel] mapping_pose 未読み込みのため "
                      f"sampling_lane_importer を使用 (eps={eps_val}m)")
                from core.sampling_lane_importer import load_sampling_lane_json
                new_records = load_sampling_lane_json(hint, epsilon=eps_val)

            if not new_records:
                QMessageBox.warning(self, t("llp_grp_autogen"),
                    f"{t('warn_sampling_lane_empty')}\n{hint}")
                return

            # poly3d を計算して ID を振り直す
            for rec in new_records:
                verts = rec.get("vertices", [])
                if len(verts) >= 2:
                    pts = np.array(
                        [(float(v[0]), float(v[1]),
                          float(v[2]) if len(v) > 2 else 0.0)
                         for v in verts], dtype=np.float32)
                    rec["poly3d"] = pts
                else:
                    rec["poly3d"] = None
                rec["_is_spline"] = False
                rec["id"] = f"lane_{self._next_id:04d}"
                self._next_id += 1

            self._lanes.extend(new_records)
            self.gl.draw_lane_data = self._lanes
            self.gl.update()
            self._refresh_table()

            # レイヤーパネルのアノテーション表示を有効化
            if self.main_win is not None and hasattr(self.main_win, '_layer_panel'):
                lp = self.main_win._layer_panel
                if lp is not None:
                    lp.set_loaded("annotations", True)
                    # チェックが外れていれば有効にする
                    row = getattr(lp, '_rows', {}).get("annotations")
                    if row is not None and not row._chk.isChecked():
                        row._chk.setChecked(True)

            n = len(new_records)
            method = "フィット生成" if mapping_pose_path else "従来生成"
            self._lbl_hint.setText(
                t("hint_autogen_done").format(n) + f" [{method}]"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, t("err_auto_generate").rstrip(":"), str(e))

    def _semi_auto_generate_lanes(self):
        """半自動生成: アノテーション一覧の手動ポリラインを参照し、
        mapping_pose.txt に平行するオフセット位置にポリラインを自動生成する。

        【アルゴリズム】
          1. _lanes の中から手動付与ポリライン（source が "manual", "auto_corrected"
             またはその他の手動系、かつ geometry_type が "polyline"）を抽出する
          2. 各手動ポリラインの各頂点と mapping_pose の最近接フレームを求め、
             軌跡法線方向への平均横断オフセット d_i を計算する
          3. mapping_pose の全フレームを d_i だけ法線方向にコピーしてポリラインを生成する
          4. 生成ライン（source="semi_auto"）は再生成時に先に削除し、手動ラインは保持する
        """
        # ── mapping_pose.txt のパスを取得 ─────────────────────────────
        mapping_pose_path = None
        if self.main_win is not None:
            traj_path = getattr(self.main_win, '_trajectory_path', None)
            if traj_path and Path(traj_path).exists():
                mapping_pose_path = traj_path

        if not mapping_pose_path:
            QMessageBox.warning(
                self, t("llp_grp_autogen"),
                t("warn_semi_autogen_no_pose"))
            return

        # ── 手動ポリラインを抽出 ──────────────────────────────────────
        MANUAL_SOURCES = {"manual", "auto_corrected", "imported"}
        manual_lanes = [
            r for r in self._lanes
            if r.get("geometry_type") == "polyline"
            and r.get("source", "manual") in MANUAL_SOURCES
        ]

        if not manual_lanes:
            QMessageBox.warning(
                self, t("llp_grp_autogen"),
                t("warn_semi_autogen_no_manual"))
            return

        try:
            import numpy as np
            from core.gt_lane_generator import (
                parse_mapping_pose, _compute_mean_offset, POSE_STRIDE,
            )
            import colorsys, random

            poses = parse_mapping_pose(mapping_pose_path)
            pose_xy_full  = poses[:, :2]
            pose_z_full   = poses[:, 2]
            pose_yaw_full = poses[:, 3]

            pose_xy_ds  = pose_xy_full[::POSE_STRIDE]
            pose_yaw_ds = pose_yaw_full[::POSE_STRIDE]

            cos_yaw = np.cos(pose_yaw_full)
            sin_yaw = np.sin(pose_yaw_full)
            normal_x = -sin_yaw
            normal_y =  cos_yaw

            # ── 前回の半自動生成レコードを削除（手動レコードは保持） ──
            before_count = len(self._lanes)
            self._lanes = [
                r for r in self._lanes
                if r.get("source") != "semi_auto"
            ]
            removed = before_count - len(self._lanes)
            if removed:
                print(f"[LaneLinePanel] 前回の半自動生成レコード {removed} 本を削除")

            # ── 各手動ラインのオフセットを計算してラインを生成 ─────────
            new_records = []
            for manual_rec in manual_lanes:
                verts = manual_rec.get("vertices", [])
                if len(verts) < 2:
                    continue

                lane_pts_xy = np.array(
                    [[float(v[0]), float(v[1])] for v in verts],
                    dtype=np.float64)

                # 軌跡に対する平均横断オフセット d を計算
                d = _compute_mean_offset(lane_pts_xy, pose_xy_ds, pose_yaw_ds)

                # mapping_pose 全フレームを d だけ法線方向にコピー
                # （範囲は全軌跡、オフセット値のみ手動ラインの位置から決定）
                new_x = pose_xy_full[:, 0] + normal_x * d
                new_y = pose_xy_full[:, 1] + normal_y * d
                new_z = pose_z_full

                vertices = [(float(new_x[i]), float(new_y[i]), float(new_z[i]))
                            for i in range(len(poses))]

                # カラー
                h = 0.12 + random.random() * 0.60
                r_c, g_c, b_c = colorsys.hsv_to_rgb(
                    h,
                    0.7 + random.random() * 0.3,
                    0.7 + random.random() * 0.3,
                )

                # 参照元の手動ラインの属性を引き継ぐ
                ref_id = manual_rec.get("id", "?")
                rec = {
                    "id": f"semi_{self._next_id:04d}",
                    "class": manual_rec.get("class", "lane_line"),
                    "geometry_type": "polyline",
                    "shape": "polyline",
                    "vertices": vertices,
                    "buffer": manual_rec.get("buffer", 0.10),
                    "poly3d": None,
                    "_color": (r_c, g_c, b_c),
                    "_is_spline": False,
                    "coordinate_system": "local_map",
                    "visibility": manual_rec.get("visibility", "visible"),
                    "quality": manual_rec.get("quality", "ok"),
                    "source": "semi_auto",
                    "is_interpolated": "false",
                    "interpolation_reason": "",
                    "note": (f"半自動生成 (参照: {ref_id}, "
                             f"軌跡平行コピー d={d:+.3f}m)"),
                    "line_type": manual_rec.get("line_type", "solid"),
                    "line_color": manual_rec.get("line_color", "white"),
                    "line_count": manual_rec.get("line_count", "single"),
                    "lane_number": manual_rec.get("lane_number", ""),
                    "lane_id_n": manual_rec.get("lane_id_n", ""),
                    "lane_id_x": manual_rec.get("lane_id_x", ""),
                    "lane_id_y": manual_rec.get("lane_id_y", ""),
                    "lane_id_manual": False,
                    "boundary_id": manual_rec.get("boundary_id", ""),
                    "start_line_ids": "",
                    "end_line_ids": "",
                }

                # poly3d を計算
                pts = np.array(
                    [(float(v[0]), float(v[1]), float(v[2]))
                     for v in vertices], dtype=np.float32)
                rec["poly3d"] = pts if len(pts) >= 2 else None

                new_records.append(rec)
                self._next_id += 1

                print(f"[LaneLinePanel] 半自動生成: 参照={ref_id} "
                      f"d={d:+.3f}m → {len(vertices)}頂点")

            if not new_records:
                QMessageBox.warning(
                    self, t("llp_grp_autogen"),
                    t("warn_semi_autogen_no_result"))
                return

            self._lanes.extend(new_records)
            self.gl.draw_lane_data = self._lanes
            self.gl.update()
            self._refresh_table()

            # ── 参照元の手動ラインを削除 ──────────────────────────────
            # 半自動生成に使用した manual_lanes を _lanes から除去する
            manual_ids = {id(r) for r in manual_lanes}
            before_del = len(self._lanes)
            self._lanes = [r for r in self._lanes if id(r) not in manual_ids]
            n_removed = before_del - len(self._lanes)
            if n_removed:
                print(f"[LaneLinePanel] 半自動生成: 参照元の手動ライン {n_removed} 本を削除")
            self.gl.draw_lane_data = self._lanes
            self.gl.update()
            self._refresh_table()

            # レイヤーパネルのアノテーション表示を有効化
            if self.main_win is not None and hasattr(self.main_win, '_layer_panel'):
                lp = self.main_win._layer_panel
                if lp is not None:
                    lp.set_loaded("annotations", True)
                    row = getattr(lp, '_rows', {}).get("annotations")
                    if row is not None and not row._chk.isChecked():
                        row._chk.setChecked(True)

            n = len(new_records)
            self._lbl_hint.setText(
                t("hint_semi_autogen_done").format(n, len(manual_lanes))
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, t("err_auto_generate").rstrip(":"), str(e))

    def _auto_generate_lanes_from_cluster(self):
        """
        後方互換: クラスターファイルから自動生成する (従来ロジック)。
        選択されたクラスターファイルを読み込み、各クラスの点群を
        スムージングスプラインで近似し、ポリラインLaneとして追加する。
        """
        idx = self._cmb_cluster.currentIndex()
        if not hasattr(self, '_cmb_cluster_paths') or idx < 0 or idx >= len(self._cmb_cluster_paths):
            QMessageBox.warning(self, t("llp_grp_autogen"), t("err_auto_gen_cluster"))
            return
        fpath = self._cmb_cluster_paths[idx]
        if fpath is None:
            QMessageBox.warning(self, t("llp_grp_autogen"), t("err_auto_gen_cluster_invalid"))
            return
        try:
            rows = []
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    sep = ',' if ',' in line else '\t' if '\t' in line else ' '
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    if len(parts) < 4:
                        continue
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        if len(parts) >= 5:
                            cls = int(float(parts[4]))
                        else:
                            cls = int(float(parts[3]))
                        rows.append((x, y, z, cls))
                    except (ValueError, IndexError):
                        continue
            if not rows:
                QMessageBox.warning(self, t("llp_grp_autogen"), t("err_auto_gen_no_points"))
                return
        except Exception as e:
            QMessageBox.warning(self, t("llp_grp_autogen"), f"{t('dialog_load_fail_msg')} {e}")
            return

        arr = np.array(rows, dtype=np.float32)
        xyz_all = arr[:, :3]
        cls_all = arr[:, 3].astype(int)

        def _z_at(cx, cy, pts3d, radius=0.5):
            diffs = pts3d[:, :2] - np.array([cx, cy], dtype=np.float32)
            dists = np.linalg.norm(diffs, axis=1)
            near = pts3d[dists < radius]
            if len(near) == 0:
                near = pts3d
            return float(near[:, 2].mean())

        def _fit_curve_entry(pts, pts_per_10m=0.5, min_verts=2, max_verts=5):
            """点群にスムージングスプラインを当てて近似曲線エントリを返す。"""
            if len(pts) < 4:
                return None
            center = pts[:, :2].mean(axis=0)
            centered = pts[:, :2] - center
            cov = np.cov(centered.T)
            if cov.ndim < 2:
                main_dir = np.array([1.0, 0.0])
            else:
                _, vecs = np.linalg.eigh(cov)
                main_dir = vecs[:, -1]
            perp_dir = np.array([-main_dir[1], main_dir[0]])
            t_vals = (centered @ main_dir).astype(np.float64)
            u_vals = (centered @ perp_dir).astype(np.float64)
            z_vals = pts[:, 2].astype(np.float64)
            span = float(t_vals.max() - t_vals.min())
            if span < 0.05:
                return None
            try:
                from scipy.interpolate import UnivariateSpline as _US
                order = np.argsort(t_vals)
                t_s, u_s, z_s = t_vals[order], u_vals[order], z_vals[order]
                _, uniq = np.unique(t_s, return_index=True)
                t_u, u_u, z_u = t_s[uniq], u_s[uniq], z_s[uniq]
                if len(t_u) < 4:
                    return None
                lateral_std = float(np.std(u_u))
                s_factor = max(len(t_u) * (lateral_std ** 2), 1e-6)
                spl_u = _US(t_u, u_u, k=3, s=s_factor)
                spl_z = _US(t_u, z_u, k=1, s=len(t_u) * 1e-4)
                n_out = int(round(span / 10.0 * pts_per_10m))
                n_out = max(min_verts, min(max_verts, n_out))
                t_out = np.linspace(t_u[0], t_u[-1], n_out)
                u_out = spl_u(t_out)
                z_out = spl_z(t_out)
                xy_out = center + np.outer(t_out, main_dir) + np.outer(u_out, perp_dir)
                verts = [
                    (float(xy_out[i, 0]), float(xy_out[i, 1]), float(z_out[i]))
                    for i in range(n_out)
                ]
            except Exception:
                p0_xy = center + float(t_vals.min()) * main_dir
                p1_xy = center + float(t_vals.max()) * main_dir
                verts = [
                    (float(p0_xy[0]), float(p0_xy[1]), _z_at(p0_xy[0], p0_xy[1], pts)),
                    (float(p1_xy[0]), float(p1_xy[1]), _z_at(p1_xy[0], p1_xy[1], pts)),
                ]
            if len(verts) < 2:
                return None
            rec = _default_lane_record(self._next_id)
            rec["id"] = self._next_id
            rec["geometry_type"] = "polyline"
            rec["shape"] = "polyline"
            rec["vertices"] = verts
            rec["buffer"] = 0.05
            _rebuild_poly3d(rec)
            return rec

        added = 0
        for cid in sorted(set(cls_all)):
            mask = cls_all == cid
            pts = xyz_all[mask]
            if len(pts) < 4:
                continue
            entry = _fit_curve_entry(pts)
            if entry is None:
                continue
            self._lanes.append(entry)
            self._next_id += 1
            added += 1

        if added == 0:
            QMessageBox.information(self, t("llp_grp_autogen"), t("err_auto_gen_no_lanes"))
            return

        # 重複除去
        target_count = len(self._lanes)
        before_prune = len(self._lanes)
        try:
            from pointcloud_viewer import _prune_overlapping_lanes
            self._lanes = _prune_overlapping_lanes(self._lanes, target_count)
        except ImportError:
            pass
        pruned = before_prune - len(self._lanes)

        # ID を振り直す
        for i, rec in enumerate(self._lanes):
            rec["id"] = i + 1
        self._next_id = len(self._lanes) + 1

        self.gl.draw_lane_data = self._lanes
        self.gl.update()
        self._refresh_table()
        final = len(self._lanes)
        self._lbl_hint.setText(
            f"✨ {final}本のLaneを生成しました（{before_prune}本、重複除去{pruned}本）。"
            f"属性を編集してください。")

        # Phase1・2 自動属性付与
        n_updated = self._auto_assign_attributes(self._lanes)
        if n_updated > 0:
            self._refresh_table()
            self.gl.update()
            self._lbl_hint.setText(
                f"✨ {final}本生成・{n_updated}本に属性を自動付与しました。")

        # クラスターファイルのレイヤーを非表示にする
        if self.main_win is not None and fpath:
            cluster_dir = str(Path(fpath).parent)
            for _, (layer, row) in self.main_win.solo_layers.items():
                if hasattr(layer, 'filepath') and layer.filepath:
                    lname = Path(layer.filepath).stem
                    ldir = str(Path(layer.filepath).parent)
                    if ldir == cluster_dir and "cluster" in lname.lower():
                        layer.visible = False
                        if hasattr(row, 'set_vis_silent'):
                            row.set_vis_silent(False)
            self.gl.update()

    def _load_lane_file(self):
        """DrawLanePanel._load_lane_file と同一ロジック。TXTファイルを読み込む。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Lane Polygon File", "",
            "Text Files (*.txt);;JSON (*.json);;All (*)")
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext == ".json":
            try:
                self._load_json(path)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", str(e))
            return
        try:
            self._lanes.clear()
            self._next_id = 1
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 5:
                        continue
                    lid     = int(parts[0])
                    shape   = parts[1]
                    lt      = parts[2]
                    lani    = parts[3]
                    buf_m   = float(parts[4]) if len(parts) > 4 else 1.0
                    vraw    = parts[5] if len(parts) > 5 else ""
                    verts   = []
                    for tok in vraw.split("|"):
                        tok = tok.strip()
                        if not tok:
                            continue
                        xyz = [float(v) for v in tok.split(",")]
                        if len(xyz) >= 3:
                            verts.append((xyz[0], xyz[1], xyz[2]))
                    rec = _default_lane_record(lid)
                    rec["id"] = lid
                    raw_geom = "polyline" if shape == "merged" else shape
                    rec["class"] = _default_class_for_geometry(raw_geom)
                    rec["shape"] = raw_geom
                    rec["geometry_type"] = raw_geom
                    rec["vertices"] = verts
                    rec["buffer"] = buf_m
                    rec["lane_type"] = lt
                    rec["lane_id"] = lani
                    _normalize_record_geometry(rec)
                    err = _geometry_validation_error(rec)
                    if err:
                        raise ValueError(
                            f"不正geometry: {rec.get('id','?')} "
                            f"geometry={rec.get('geometry_type')}: {err}"
                        )
                    _rebuild_poly3d(rec)
                    self._lanes.append(rec)
                    self._next_id = max(self._next_id, lid + 1)
            self._refresh_table()
            self.gl.draw_lane_data = self._lanes
            self.gl.update()
            self._lbl_hint.setText(
                f"📂 {len(self._lanes)} lanes を読み込みました: {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _ground_z(self) -> float:
        """ベースレイヤーまたは読み込み済みレイヤーの中央 Z 値を返す"""
        if (self._base_layer is not None
                and hasattr(self._base_layer, 'xyz')
                and self._base_layer.xyz is not None):
            return float(np.median(self._base_layer.xyz[:, 2]))
        # MainWindow のレイヤーから Z 値を取得
        if self.main_win is not None:
            for _, (layer, _) in self.main_win.solo_layers.items():
                if hasattr(layer, 'xyz') and layer.xyz is not None and layer.loaded:
                    return float(np.median(layer.xyz[:, 2]))
            for _, (group, _) in self.main_win.groups.items():
                for layer in group.layers:
                    if hasattr(layer, 'xyz') and layer.xyz is not None and layer.loaded:
                        return float(np.median(layer.xyz[:, 2]))
        # 既存のLaneの頂点から推定
        if self._lanes:
            zs = []
            for rec in self._lanes:
                for v in rec.get("vertices", []):
                    zs.append(v[2])
            if zs:
                return float(np.median(zs))
        return 0.0

    # ==================================================================
    # プレースホルダ (後続の項目で実装)
    # ==================================================================
    # ==================================================================
    # ② 描画ツール
    # ==================================================================
    def _toggle_draw(self, mode: str):
        """ポリライン / ポリゴン / ポイント 描画モードの切替"""
        if self._draw_mode == mode:
            self._cancel_draw()
            return
        self._draw_mode    = mode
        self._cur_verts_2d = []
        self._cur_verts_3d = []

        # ── ポリラインモード開始時はクラス・属性を lane_line デフォルトにリセット ──
        if mode == "polyline":
            self._last_class = "lane_line"
            self._last_attrs = {
                "line_type":  "solid",
                "line_color": "white",
                "line_count": "single",
            }
            # クラスボタン UI を lane_line に切り替え
            for c, btn in self._cls_buttons.items():
                is_sel = (c == "lane_line")
                btn.setChecked(is_sel)
                btn.setStyleSheet(self._BTN_CLS_ON if is_sel else self._BTN_CLS_OFF)
            # 属性フィールドを lane_line 用に再構築
            self._rebuild_attr_fields("lane_line")

        # ボタンスタイル — 全ボタンをリセットしてから選択中のみON
        self._btn_polyline.blockSignals(True)
        self._btn_polygon.blockSignals(True)
        self._btn_rbbox.blockSignals(True)
        self._btn_polyline.setChecked(mode == "polyline")
        self._btn_polyline.setStyleSheet(
            self._BTN_ON_PL if mode == "polyline" else self._BTN_OFF_PL)
        self._btn_polygon.setChecked(mode == "polygon")
        self._btn_polygon.setStyleSheet(
            self._BTN_ON_PG if mode == "polygon" else self._BTN_OFF_PG)
        self._btn_rbbox.setChecked(mode == "point")
        self._btn_rbbox.setStyleSheet(
            self._BTN_ON_BB if mode == "point" else self._BTN_OFF_BB)
        self._btn_polyline.blockSignals(False)
        self._btn_polygon.blockSignals(False)
        self._btn_rbbox.blockSignals(False)

        # GL 側の状態を設定
        self.gl._draw_lane_mode    = mode
        self.gl._draw_lane_verts2d = []
        self.gl._draw_lane_panel   = self
        self.gl.draw_lane_preview  = None
        self.gl.setCursor(Qt.CrossCursor)
        self.gl.update()

        if mode == "polyline":
            self._lbl_hint.setText(t("llp_tooltip_polyline"))
        elif mode == "polygon":
            self._lbl_hint.setText(t("llp_tooltip_polygon"))
        elif mode == "point":
            self._lbl_hint.setText(t("llp_hint_point"))

        # クラスボタンの有効/無効を描画モードに応じて制御
        effective_mode = mode
        self._update_class_buttons_for_mode(effective_mode)

    def _cancel_draw(self):
        """描画モードをキャンセルする（軽量版）"""
        self._draw_mode = ""

    def _exit_draw_mode(self):
        """描画モードを完全に終了する（ボタン・GL状態・カーソルをすべてリセット）。

        ESC2回、または頂点なし状態でのESCで呼ばれる。
        """
        self._draw_mode    = ""
        self._cur_verts_2d = []
        self._cur_verts_3d = []
        self._esc_once     = False

        # ボタンのOFF状態を設定
        for btn, off_style in [
            (getattr(self, '_btn_polyline', None), self._BTN_OFF_PL),
            (getattr(self, '_btn_polygon',  None), self._BTN_OFF_PG),
            (getattr(self, '_btn_rbbox',    None), getattr(self, '_BTN_OFF_BB', '')),
        ]:
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.setStyleSheet(off_style)
                btn.blockSignals(False)

        # GL 側をリセット
        self.gl._draw_lane_mode    = ""
        self.gl._draw_lane_verts2d = []
        self.gl.setCursor(Qt.ArrowCursor)
        self.gl.draw_lane_preview  = None
        self.gl.update()

        self._lbl_hint.setText(t("llp_hint_default"))
        if hasattr(self, '_update_class_buttons_for_mode'):
            self._update_class_buttons_for_mode("")

    def _on_spline_toggled(self, checked: bool) -> None:
        """スプライン補間チェックボックスのトグル処理。
        既存レコードすべての poly3d を再計算して即座に表示に反映する。"""
        self._use_spline = checked
        for rec in self._lanes:
            if rec.get("geometry_type", rec.get("shape", "")) == "polyline":
                _rebuild_poly3d(rec, use_spline=checked)
        self.gl.draw_lane_data = self._lanes
        self.gl.update()

    def _rebuild_poly3d_cur(self, rec: dict) -> None:
        """現在のスプライン設定を適用して rec の poly3d を再計算するショートカット。"""
        _rebuild_poly3d(rec, use_spline=getattr(self, '_use_spline', False))
        self._cur_verts_2d = []
        self._cur_verts_3d = []
        self._btn_polyline.blockSignals(True)
        self._btn_polygon.blockSignals(True)
        self._btn_rbbox.blockSignals(True)
        self._btn_polyline.setChecked(False)
        self._btn_polyline.setStyleSheet(self._BTN_OFF_PL)
        self._btn_polygon.setChecked(False)
        self._btn_polygon.setStyleSheet(self._BTN_OFF_PG)
        self._btn_rbbox.setChecked(False)
        self._btn_rbbox.setStyleSheet(self._BTN_OFF_BB)
        self._btn_polyline.blockSignals(False)
        self._btn_polygon.blockSignals(False)
        self._btn_rbbox.blockSignals(False)
        self.gl._draw_lane_mode    = ""
        self.gl._draw_lane_verts2d = []
        self.gl.setCursor(Qt.ArrowCursor)
        self.gl.draw_lane_preview  = None
        self.gl.update()
        self._lbl_hint.setText(t("llp_hint_default"))
        self._update_class_buttons_for_mode("")

    def _notify_draw_position_follow(
        self,
        world_pt,
    ) -> None:
        """現在の描画位置をMainWindowへ通知する。

        MainWindow側の「描画位置に追従」がONなら、
        最寄りmapping_pose/FrameSyncフレームへ移動する。
        """
        if self.main_win is None or world_pt is None:
            return

        callback = getattr(
            self.main_win,
            "_on_annotation_draw_position",
            None,
        )
        if not callable(callback):
            return

        try:
            callback(
                float(world_pt[0]),
                float(world_pt[1]),
                str(self._draw_mode),
            )
        except Exception as exc:
            print(
                "[Draw Follow] callback warning: "
                f"{exc}"
            )

    # ── GL から呼ばれる: 左クリック → 頂点追加 ────────────────────
    def on_draw_click(self, screen_pt):
        gz = self._ground_z()
        world = _screen_to_world_ray(self.gl, screen_pt, gz)
        if world is None:
            return
        self.gl._draw_lane_verts2d.append(screen_pt)
        self._cur_verts_2d.append(screen_pt)
        self._cur_verts_3d.append(world)
        n = len(self._cur_verts_2d)

        # 新規ポリライン/ポリゴン/ポイントの「1点目」だけ、
        # 下段カメラ画像と黄色い車両矩形を描画位置へ追従させる。
        # 2点目以降はフレームを動かさず、連続描画を邪魔しない。
        if n == 1:
            self._notify_draw_position_follow(world)

        # 頂点追加でESC連続検出フラグをリセット（クリック後のESCは1回目扱い）
        self._esc_once = False
        self._update_draw_preview()
        self.gl.update()

        # ポイントモード: 1点で即確定
        if self._draw_mode == "point":
            self._finalize_point(world)
            return

        if self._draw_mode == "polyline":
            self._lbl_hint.setText(
                t("llp_hint_polyline").format(n=n))
        else:
            self._lbl_hint.setText(
                t("llp_hint_polygon").format(n=n))

    # ── GL から呼ばれる: 右クリック → ポリライン確定 ──────────────
    def on_draw_finish(self):
        if self._draw_mode == "polyline":
            if len(self._cur_verts_3d) < 2:
                self._cancel_draw(); return
            self._finalize_lane("polyline", list(self._cur_verts_3d))
        elif self._draw_mode == "polygon":
            # 本プロジェクトではポリゴンは4頂点以上で確定する。
            if len(self._cur_verts_3d) < 4:
                self._lbl_hint.setText("⚠ ポリゴンは4点以上必要です。")
                return
            self._finalize_lane("polygon", list(self._cur_verts_3d))

    # ── GL から呼ばれる: ダブルクリック → ポリゴン確定 ────────────
    def on_draw_double_click(self, screen_pt):
        if self._draw_mode != "polygon":
            return
        # ダブルクリック時も4頂点以上を必須とする。
        if len(self._cur_verts_3d) < 4:
            self._lbl_hint.setText("⚠ ポリゴンは4点以上必要です。")
            return
        self._finalize_lane("polygon", list(self._cur_verts_3d))

    # ── ESC → 直前の頂点を削除 / 2回連続で描画モード終了 ──────────
    def undo_last_vertex(self):
        """ESC キーの処理。

        1回目: 直前の頂点を削除（頂点がなければ描画モード終了）
        2回目（連続）: 描画モードを完全に終了する
        """
        # 2回目の ESC（前回の ESC から頂点追加がなかった場合）→ 描画モード終了
        if getattr(self, '_esc_once', False):
            self._esc_once = False
            self._exit_draw_mode()  # 描画モード完全終了
            return

        # 頂点がなければそのまま描画モード終了
        if not self._cur_verts_2d:
            self._exit_draw_mode()
            return

        # 1回目の ESC: 直前の頂点を削除
        self._esc_once = True
        self._cur_verts_2d.pop()
        self._cur_verts_3d.pop()
        self.gl._draw_lane_verts2d = list(self._cur_verts_2d)
        self._update_draw_preview()
        self.gl.update()
        n = len(self._cur_verts_2d)
        if n == 0:
            # 頂点がなくなったら次の ESC で終了できる旨を表示
            self._lbl_hint.setText("頂点なし  もう一度ESCで描画モード終了")
        elif self._draw_mode == "polyline":
            self._lbl_hint.setText(
                f"頂点 {n}個  R-click で確定 / ESC で1点削除 / 2回ESCで終了")
        else:
            self._lbl_hint.setText(
                f"頂点 {n}個  Dbl-click で確定 (3個以上) / ESC で1点削除 / 2回ESCで終了")

    # ── 描画プレビュー更新 ───────────────────────────────────────
    def _update_draw_preview(self):
        pts = self._cur_verts_3d
        if self._draw_mode == "polyline" and len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.float32)
            if getattr(self, '_use_spline', False) and len(pts) >= 3:
                self.gl.draw_lane_preview = _spline_interpolate_3d(
                    pts_arr, n_samples=max(60, len(pts) * 10))
                self.gl._draw_preview_is_spline = True
            else:
                # 折れ線モード: クリック点をそのまま表示（描画とカーソル位置が一致）
                self.gl.draw_lane_preview = pts_arr
                self.gl._draw_preview_is_spline = False
        elif self._draw_mode == "polygon" and len(pts) >= 3:
            self.gl.draw_lane_preview = np.array(pts, dtype=np.float32)
            self.gl._draw_preview_is_spline = False
        else:
            self.gl.draw_lane_preview = None
            self.gl._draw_preview_is_spline = False

    # ── ポリライン中間点補間 ─────────────────────────────────────────
    @staticmethod
    def _interpolate_long_segments(vertices: list,
                                   max_dist: float = 20.0,
                                   target_interval: float = 17.5) -> list:
        """隣接頂点間が max_dist を超える場合に中間点を等間隔に補間する。

        target_interval: 補間後の目標間隔 (15～20m の中間値)
        """
        if len(vertices) < 2:
            return vertices
        result = [vertices[0]]
        for i in range(1, len(vertices)):
            ax, ay, az = vertices[i - 1][:3]
            bx, by, bz = vertices[i][:3]
            dx = bx - ax
            dy = by - ay
            dz = bz - az
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > max_dist:
                # 必要な分割数を計算（各セグメントが target_interval になるように）
                n_segments = max(2, round(dist / target_interval))
                for j in range(1, n_segments):
                    t = j / n_segments
                    mx = ax + dx * t
                    my = ay + dy * t
                    mz = az + dz * t
                    result.append((mx, my, mz))
            result.append(vertices[i])
        return result

    # ── ポイント確定 → 分岐/合流レコード追加 ────────────────────────
    def _finalize_point(self, world_pt):
        """1点で分岐/合流ポイントレコードを追加する"""
        # 描画状態をリセット（ポイントモードは1点配置後も継続）
        self.gl._draw_lane_verts2d = []
        self._cur_verts_2d = []
        self._cur_verts_3d = []
        self.gl.draw_lane_preview = None
        self.gl.update()

        # レコード作成（split_point として追加）
        rec = _default_lane_record(self._next_id)
        self._next_id += 1
        rec["class"] = "split_point"
        rec["vertices"] = [(float(world_pt[0]), float(world_pt[1]),
                            float(world_pt[2]))]
        _normalize_record_geometry(rec)
        self._last_class = "split_point"
        rec["start_line_ids"] = ""
        rec["end_line_ids"] = ""
        _rebuild_poly3d(rec)

        self._lanes.append(rec)
        self.gl.draw_lane_data = self._lanes
        self.gl.update()
        self._refresh_table()
        self._lbl_hint.setText(
            t("hint_point_added").format(rec['id']))

        # 作成したポイントを選択
        self._tbl.selectRow(len(self._lanes) - 1)

    # ── 描画確定 → Lane レコード追加 ─────────────────────────────
    def _finalize_lane(self, geom_type: str, pts3d: list):
        """描画を確定し、新しい Lane レコードを追加する"""
        # 描画モードをリセット
        mode = self._draw_mode
        self.gl._draw_lane_mode    = ""
        self.gl._draw_lane_verts2d = []
        self.gl.setCursor(Qt.ArrowCursor)
        self._btn_polyline.setChecked(False)
        self._btn_polygon.setChecked(False)
        self._btn_polyline.setStyleSheet(self._BTN_OFF_PL)
        self._btn_polygon.setStyleSheet(self._BTN_OFF_PG)
        self._draw_mode    = ""
        self._cur_verts_2d = []
        self._cur_verts_3d = []

        # 頂点数の最終防御
        min_vertices = _GEOMETRY_MIN_VERTICES.get(geom_type, 2)
        if (
            (geom_type == "point" and len(pts3d) != 1)
            or (geom_type != "point" and len(pts3d) < min_vertices)
        ):
            self._lbl_hint.setText(
                f"⚠ {geom_type} の頂点数が不正です。"
            )
            return

        # レコード作成
        rec = _default_lane_record(self._next_id)
        self._next_id += 1
        rec["vertices"] = [
            (float(p[0]), float(p[1]), float(p[2]))
            for p in pts3d
        ]

        # 現在の描画geometryとclassが食い違う場合は、
        # 描画geometryに対応する既定classへ正規化する。
        final_class = self._last_class
        if _geometry_type_from_class(final_class) != geom_type:
            final_class = _default_class_for_geometry(geom_type)
        rec["class"] = final_class
        _normalize_record_geometry(rec)
        self._last_class = final_class

        # ポリライン: 隣接頂点間が20mを超える場合に中間点を補間
        if geom_type == "polyline":
            rec["vertices"] = self._interpolate_long_segments(rec["vertices"])
            rec["buffer"] = 0.10  # デフォルト10cm

        # 直前の属性をコピー（class/geometryは上で確定済みなので上書きしない）
        for k, v in self._last_attrs.items():
            if k in rec and k not in ("class", "geometry_type", "shape"):
                rec[k] = v

        # poly3d 生成（スプライン設定に従う）
        self._rebuild_poly3d_cur(rec)

        # デバッグ: poly3d の設定を確認
        poly3d = rec.get("poly3d")
        print(f"[_finalize_lane] poly3d={'None' if poly3d is None else f'shape={poly3d.shape}'}, "
              f"geom={geom_type}, verts={len(rec['vertices'])}")

        self._lanes.append(rec)
        self.gl.draw_lane_data    = self._lanes
        self.gl.draw_lane_preview = None
        self.gl.update()
        self._refresh_table()

        # デバッグ: draw_lane_dataに正しく追加されているか確認
        print(f"[_finalize_lane] draw_lane_data件数={len(self.gl.draw_lane_data)}, "
              f"_lanes件数={len(self._lanes)}, "
              f"同一オブジェクト={self.gl.draw_lane_data is self._lanes}")
        if len(self.gl.draw_lane_data) > 0:
            last = self.gl.draw_lane_data[-1]
            poly = last.get("poly3d")
            print(f"  最後のrec: id={last.get('id')}, "
                  f"poly3d={'None' if poly is None else poly.shape}, "
                  f"_is_spline={last.get('_is_spline')}, "
                  f"source={last.get('source')}")
            if poly is not None and len(poly) > 0:
                print(f"  poly3d 先頭: {poly[0]}, 末尾: {poly[-1]}")
            verts = last.get("vertices", [])
            if verts:
                print(f"  vertices 先頭: {verts[0]}, 末尾: {verts[-1]}")
        print(f"  GL center={self.gl._center}, scale={self.gl._scale:.6f}, zoom={self.gl.zoom}")

        # レイヤーパネルのアノテーション表示を有効化
        if self.main_win is not None and hasattr(self.main_win, '_layer_panel'):
            lp = self.main_win._layer_panel
            if lp is not None:
                lp.set_loaded("annotations", True)
                row = getattr(lp, '_rows', {}).get("annotations")
                if row is not None and not row._chk.isChecked():
                    row._chk.setChecked(True)

        self._lbl_hint.setText(
            t("hint_lane_confirmed").format(rec['id']))

        # BEV画像が未読み込みでスケールがデフォルト値の場合は
        # 新しいラインが見えるようにカメラ中心を調整する
        if self.gl._bev_image is None and rec.get("poly3d") is not None:
            poly = rec["poly3d"]
            if len(poly) > 0:
                center = poly.mean(axis=0)
                self.gl._center = center.astype(np.float32)
                # スケールがデフォルト（1.0）のまま、またはすでに設定済みなら維持
                if self.gl._scale < 1e-6 or self.gl._scale >= 0.99:
                    xy_span = np.max(poly[:, :2].max(axis=0) - poly[:, :2].min(axis=0))
                    if xy_span > 1e-6:
                        self.gl._scale = 1.0 / (xy_span + 1e-9)
                self.gl.zoom = 1.0
                self.gl.pan_x = 0.0
                self.gl.pan_y = 0.0
                self.gl.update()

        # 描画モードをOFFにして表示を更新する
        self.gl.draw_lane_edit_verts = None
        self.gl.draw_lane_highlight  = None
        self.gl.draw_lane_highlights = []
        self._edit_lane_idx = None
        self._cancel_draw()
        # 選択は解除して通常描画で表示させる（ハイライト非表示による不具合を回避）
        self._tbl.clearSelection()
        self.gl.update()

    def _finalize_rbbox(self, pts3d: list):
        """4点から最適な回転長方形を計算してポリゴンとして追加する。

        OpenCV の minAreaRect と同等のロジック:
        1. 4点の凸包を求める
        2. 回転キャリパー法で最小面積の外接長方形を求める
        3. 長方形の4頂点をポリゴンとして追加
        """
        if len(pts3d) < 4:
            self._cancel_draw()
            return

        pts_xy = np.array([(p[0], p[1]) for p in pts3d], dtype=np.float64)
        z_avg = float(np.mean([p[2] for p in pts3d]))

        # 凸包
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts_xy)
            hull_pts = pts_xy[hull.vertices]
        except Exception:
            hull_pts = pts_xy

        # 回転キャリパー法で最小面積外接長方形
        best_area = float('inf')
        best_rect = None
        n_h = len(hull_pts)
        for i in range(n_h):
            # 辺の方向ベクトル
            edge = hull_pts[(i + 1) % n_h] - hull_pts[i]
            angle = math.atan2(edge[1], edge[0])
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)
            # 回転して軸に揃える
            rotated = np.zeros_like(hull_pts)
            for j in range(n_h):
                rotated[j, 0] = hull_pts[j, 0] * cos_a - hull_pts[j, 1] * sin_a
                rotated[j, 1] = hull_pts[j, 0] * sin_a + hull_pts[j, 1] * cos_a
            min_x, max_x = rotated[:, 0].min(), rotated[:, 0].max()
            min_y, max_y = rotated[:, 1].min(), rotated[:, 1].max()
            area = (max_x - min_x) * (max_y - min_y)
            if area < best_area:
                best_area = area
                # 長方形の4頂点（回転座標系）
                rect_rot = np.array([
                    [min_x, min_y], [max_x, min_y],
                    [max_x, max_y], [min_x, max_y],
                ])
                # 元の座標系に戻す
                cos_b = math.cos(angle)
                sin_b = math.sin(angle)
                rect_world = np.zeros_like(rect_rot)
                for j in range(4):
                    rect_world[j, 0] = rect_rot[j, 0] * cos_b - rect_rot[j, 1] * sin_b
                    rect_world[j, 1] = rect_rot[j, 0] * sin_b + rect_rot[j, 1] * cos_b
                best_rect = rect_world
                best_angle = math.degrees(angle)
                best_w = max_x - min_x
                best_h = max_y - min_y

        if best_rect is None:
            self._cancel_draw()
            return

        # 描画モードをリセット
        self.gl._draw_lane_mode    = ""
        self.gl._draw_lane_verts2d = []
        self.gl.setCursor(Qt.ArrowCursor)
        self._draw_mode    = ""
        self._cur_verts_2d = []
        self._cur_verts_3d = []

        # レコード作成
        rec = _default_lane_record(self._next_id)
        self._next_id += 1
        rec["geometry_type"] = "polygon"
        rec["shape"] = "polygon"
        rec["class"] = self._last_class
        if rec["class"] not in POLYGON_CLASSES:
            rec["class"] = "zebra_line"
        rec["vertices"] = [(float(best_rect[j, 0]), float(best_rect[j, 1]), z_avg)
                           for j in range(4)]
        # 回転角と寸法を note に記録
        short_side = min(best_w, best_h)
        long_side = max(best_w, best_h)
        rec["note"] = (f"RBBOX {long_side:.2f}x{short_side:.2f}m "
                       f"angle={best_angle:.1f}°")
        for k, v in self._last_attrs.items():
            if k in rec:
                rec[k] = v
        _rebuild_poly3d(rec)

        self._lanes.append(rec)
        self.gl.draw_lane_data    = self._lanes
        self.gl.draw_lane_preview = None
        self.gl.update()
        self._refresh_table()
        self._cancel_draw()

        # 作成したLaneを選択
        idx = len(self._lanes) - 1
        self._tbl.selectRow(idx)
        self._edit_lane_idx = idx
        self.gl.draw_lane_edit_verts = np.array(
            rec["vertices"], dtype=np.float32)
        self.gl.draw_lane_highlight = rec.get("poly3d")
        self.gl.draw_lane_edit_panel = self
        self._populate_attr_from_rec(rec)
        self.gl.update()
        self._lbl_hint.setText(
            f"▣ 回転BBOX: {long_side:.2f}×{short_side:.2f}m, "
            f"角度={best_angle:.1f}°")

    # ==================================================================
    # プレースホルダ (後続の項目で実装)
    # ==================================================================
    def _update_class_buttons_for_mode(self, mode: str):
        """描画モードに応じてクラスボタンの有効/無効を切り替える"""
        is_polygon = (mode == "polygon")
        is_polyline = (mode == "polyline")
        for cls_name, btn in self._cls_buttons.items():
            is_pg_cls = cls_name in POLYGON_CLASSES
            if is_polygon:
                # ポリゴンモード: ポリゴンクラスのみ有効
                btn.setEnabled(is_pg_cls)
                if not is_pg_cls:
                    btn.setStyleSheet(self._BTN_CLS_DIS)
            elif is_polyline:
                # ポリラインモード: ポリラインクラスのみ有効
                btn.setEnabled(not is_pg_cls)
                if is_pg_cls:
                    btn.setStyleSheet(self._BTN_CLS_DIS)
            else:
                # 描画モードでない: 全て有効
                btn.setEnabled(True)
        # ポリゴンモードでデフォルトを横断歩道に
        if is_polygon:
            current = self._last_class
            if current not in POLYGON_CLASSES:
                self._on_class_btn("zebra_line")
        elif is_polyline:
            current = self._last_class
            if current in POLYGON_CLASSES:
                self._on_class_btn("lane_line")

    # ==================================================================
    # ③ 属性パネル
    # ==================================================================
    def _set_class_button_state(self, cls_name: str) -> None:
        """classボタン表示だけを指定classへ同期する。"""
        for c, btn in self._cls_buttons.items():
            btn.setChecked(c == cls_name)
            btn.setStyleSheet(
                self._BTN_CLS_ON if c == cls_name else self._BTN_CLS_OFF
            )

    def _on_class_btn(self, cls_name: str):
        """クラスボタンがクリックされたとき。

        geometry遷移に必要な頂点数を満たさない場合は変更を拒否し、
        元のclass/geometryを維持する。
        """
        sel_rows = [
            r.row()
            for r in self._tbl.selectionModel().selectedRows()
        ]
        if not sel_rows and self._edit_lane_idx is not None:
            sel_rows = [self._edit_lane_idx]

        previous_class = self._last_class
        if sel_rows and sel_rows[0] < len(self._lanes):
            previous_class = self._lanes[sel_rows[0]].get(
                "class", self._last_class
            )

        if self._on_class_changed(cls_name):
            self._set_class_button_state(cls_name)
        else:
            self._set_class_button_state(previous_class)

    def _on_class_changed(self, cls_name: str) -> bool:
        """class変更とgeometry遷移を原子的に行う。

        規約:
          point    = 1点ちょうど
          polyline = 2点以上
          polygon  = 4点以上

        遷移先の頂点数条件を満たさないレコードが1件でもあれば、
        選択中レコードは1件も変更しない。
        """
        sel_rows = [
            r.row()
            for r in self._tbl.selectionModel().selectedRows()
        ]
        if not sel_rows and self._edit_lane_idx is not None:
            sel_rows = [self._edit_lane_idx]

        new_geom = _geometry_type_from_class(cls_name)

        # 選択レコードがある場合は先に全件検証し、部分更新を防ぐ。
        invalid = []
        for row in sel_rows:
            if row >= len(self._lanes):
                continue
            rec = self._lanes[row]
            n = len(rec.get("vertices", []) or [])
            if new_geom == "point":
                ok = (n == 1)
                requirement = "1点ちょうど"
            else:
                min_n = _GEOMETRY_MIN_VERTICES[new_geom]
                ok = (n >= min_n)
                requirement = f"{min_n}点以上"
            if not ok:
                invalid.append(
                    f"{rec.get('id','?')}: {n}点 "
                    f"(必要: {requirement})"
                )

        if invalid:
            QMessageBox.warning(
                self,
                "Geometry変更不可",
                f"{cls_name} ({new_geom}) へ変更できません。\n"
                "頂点数条件を満たしていないデータがあります。\n\n"
                + "\n".join(invalid[:10]),
            )
            return False

        # 検証後に一括更新。
        for row in sel_rows:
            if row >= len(self._lanes):
                continue
            rec = self._lanes[row]
            old_geom = rec.get(
                "geometry_type",
                rec.get("shape", "polyline"),
            )
            rec["class"] = cls_name
            _normalize_record_geometry(rec)

            use_spline = bool(
                rec["geometry_type"] == "polyline"
                and rec.get("_is_spline", False)
            )
            _rebuild_poly3d(rec, use_spline=use_spline)

            if old_geom != rec["geometry_type"]:
                print(
                    "[LaneLinePanel] geometry transition: "
                    f"id={rec.get('id','?')} "
                    f"{old_geom} -> {rec['geometry_type']} "
                    f"class={cls_name}"
                )

        if sel_rows:
            self.gl.draw_lane_data = self._lanes
            if (
                self._edit_lane_idx is not None
                and 0 <= self._edit_lane_idx < len(self._lanes)
            ):
                current = self._lanes[self._edit_lane_idx]
                self.gl.draw_lane_highlight = current.get("poly3d")
            self._refresh_table()
            self.gl.update()

        self._last_class = cls_name
        self._rebuild_attr_fields(cls_name)
        return True

    def _rebuild_attr_fields(self, cls_name: str):
        """cls_name に応じた属性フィールドを動的に構築する。"""
        self._attr_widgets.clear()
        self._btn_groups = {}
        layout = self._attr_container_layout
        while layout.count():
            child = layout.takeAt(0)
            w = child.widget()
            if w:
                w.deleteLater()
            sub = child.layout()
            if sub:
                while sub.count():
                    sc = sub.takeAt(0)
                    sw = sc.widget()
                    if sw:
                        sw.deleteLater()

        attr_keys = CLASS_ATTR_MAP.get(cls_name, [])

        _BTN_OFF = ("QPushButton{background:#3c3c3c;color:#ccc;"
                    "border:1px solid #555;border-radius:3px;"
                    "padding:2px 4px;font-size:10px;}"
                    "QPushButton:hover{background:#4a4a4a;color:white;}")
        _BTN_ON  = ("QPushButton{background:#2a6ebb;color:white;"
                    "border:2px solid #4a8edb;border-radius:3px;"
                    "padding:2px 4px;font-size:10px;font-weight:bold;}")

        # BUTTON_ATTRSのtitleとラベルのi18nキーマッピング
        _ATTR_TITLE_KEYS = {
            "line_type":  "attr_line_type_title",
            "line_color": "attr_line_color_title",
            "line_count": "attr_line_count_title",
        }
        _ATTR_LABEL_KEYS = {
            "solid": "attr_lbl_solid", "dash": "attr_lbl_dash",
            "fish_bone": "attr_lbl_fish_bone", "dash_solid": "attr_lbl_dash_solid",
            "unknown": "attr_lbl_unknown", "wide_dash": "attr_lbl_wide_dash",
            "virtual": "attr_lbl_virtual", "white": "attr_lbl_white",
            "yellow": "attr_lbl_yellow", "single": "attr_lbl_single",
            "double": "attr_lbl_double",
        }

        def _make_btn_group(attr_name, btn_def):
            """ボタングループを作成して返す (QGroupBox)"""
            title_key = _ATTR_TITLE_KEYS.get(attr_name)
            title = t(title_key) if title_key else btn_def["title"]
            grp = QGroupBox(title)
            grp.setStyleSheet(
                "QGroupBox{color:#aaa;border:1px solid #444;"
                "border-radius:4px;margin-top:6px;font-size:10px;"
                "font-weight:bold;}"
                "QGroupBox::title{subcontrol-origin:margin;"
                "left:6px;padding:0 3px;}")
            row = QHBoxLayout(grp)
            row.setContentsMargins(3, 10, 3, 3)
            row.setSpacing(2)
            btns = {}
            labels = btn_def.get("labels", {})
            for val in btn_def["choices"]:
                lbl_key = _ATTR_LABEL_KEYS.get(val)
                short = t(lbl_key) if lbl_key else labels.get(val, val)
                btn = QPushButton(short)
                btn.setCheckable(True)
                btn.setFixedHeight(28)
                btn.setStyleSheet(_BTN_OFF)
                btn.clicked.connect(
                    lambda checked, a=attr_name, v=val:
                        self._on_btn_attr(a, v))
                row.addWidget(btn)
                btns[val] = btn
            self._btn_groups[attr_name] = btns
            self._attr_widgets[attr_name] = btns
            return grp

        skip_set = set()  # inline で処理済みの属性

        for attr_name in attr_keys:
            if attr_name == "note" or attr_name in skip_set:
                continue

            btn_def = BUTTON_ATTRS.get(attr_name)
            if btn_def is not None:
                # inline 指定: 2つのボタングループを1行に並べる
                inline_name = btn_def.get("inline")
                if inline_name and inline_name in attr_keys:
                    inline_def = BUTTON_ATTRS.get(inline_name)
                    if inline_def:
                        row = QHBoxLayout()
                        row.setSpacing(4)
                        row.addWidget(_make_btn_group(attr_name, btn_def))
                        row.addWidget(_make_btn_group(inline_name, inline_def))
                        layout.addLayout(row)
                        skip_set.add(inline_name)
                        continue
                layout.addWidget(_make_btn_group(attr_name, btn_def))

            elif attr_name in COMBO_ATTRS:
                r = QHBoxLayout(); r.setSpacing(4)
                lbl = QLabel(f"{attr_name}:")
                lbl.setStyleSheet("color:#aac8ff;font-size:10px;")
                lbl.setFixedWidth(80)
                r.addWidget(lbl)
                cmb = QComboBox()
                cmb.setStyleSheet("font-size:11px;background:#1c1c2e;"
                                  "border:1px solid #3a3a50;color:#c8c8e8;")
                choices = ATTR_CHOICES.get(attr_name, [""])
                cmb.addItems(choices)
                cmb.currentTextChanged.connect(
                    lambda val, a=attr_name: self._on_attr_changed(a, val))
                r.addWidget(cmb, stretch=1)
                self._attr_widgets[attr_name] = cmb
                layout.addLayout(r)

            else:
                # テキスト入力フィールド (lane_number, boundary_id, start_line_ids 等)
                r = QHBoxLayout(); r.setSpacing(4)
                _ATTR_LABELS = {
                    "lane_number": "番号ID:",
                    "boundary_id": "境界ID:",
                    "start_line_ids": "開始線:",
                    "end_line_ids": "終了線:",
                }
                lbl_text = _ATTR_LABELS.get(attr_name, f"{attr_name}:")
                lbl = QLabel(lbl_text)
                lbl.setStyleSheet("color:#aac8ff;font-size:10px;")
                lbl.setFixedWidth(56)
                r.addWidget(lbl)
                edt = QLineEdit()
                edt.setStyleSheet(
                    f"font-size:11px;background:#1c1c2e;"
                    f"border:1px solid #3a3a50;color:#c8c8e8;"
                    f"min-height:{self._ROW_H - 6}px;"
                    f"max-height:{self._ROW_H - 6}px;")
                edt.setFixedHeight(self._ROW_H)
                _PLACEHOLDERS = {
                    "lane_number": "N_X-Y (例: 1_0-0)",
                    "boundary_id": "連番 (例: 1)",
                    "start_line_ids": "開始線ID (例: 2_0-1, 3_0-0)",
                    "end_line_ids": "終了線ID (例: 2_0-2)",
                }
                edt.setPlaceholderText(
                    _PLACEHOLDERS.get(attr_name, ""))
                edt.editingFinished.connect(
                    lambda a=attr_name, e=edt:
                        self._on_attr_changed(a, e.text()))
                r.addWidget(edt, stretch=1)
                self._attr_widgets[attr_name] = edt
                layout.addLayout(r)

    def _on_btn_attr(self, attr_name: str, value: str):
        """ボタン式属性がクリックされたとき"""
        _BTN_OFF = ("QPushButton{background:#3c3c3c;color:#ccc;"
                    "border:1px solid #555;border-radius:3px;"
                    "padding:2px 4px;font-size:10px;}"
                    "QPushButton:hover{background:#4a4a4a;color:white;}")
        _BTN_ON  = ("QPushButton{background:#2a6ebb;color:white;"
                    "border:2px solid #4a8edb;border-radius:3px;"
                    "padding:2px 4px;font-size:10px;font-weight:bold;}")
        # ボタンの排他選択
        btns = self._btn_groups.get(attr_name, {})
        for v, btn in btns.items():
            btn.setChecked(v == value)
            btn.setStyleSheet(_BTN_ON if v == value else _BTN_OFF)
        # レコードに反映
        self._on_attr_changed(attr_name, value)

    def _on_attr_changed(self, attr_name: str, value: str):
        """属性値が変更されたとき、全選択中の Lane レコードに反映する"""
        # 全選択行を取得
        sel_rows = [r.row() for r in self._tbl.selectionModel().selectedRows()]
        if not sel_rows and self._edit_lane_idx is not None:
            sel_rows = [self._edit_lane_idx]
        for row in sel_rows:
            if row < len(self._lanes):
                rec = self._lanes[row]
                rec[attr_name] = value
                if attr_name == "buffer":
                    try:
                        rec["buffer"] = float(value) / 100.0
                    except ValueError:
                        pass
                    _rebuild_poly3d(rec)
        if sel_rows:
            self.gl.draw_lane_data = self._lanes
            self.gl.update()
            self._refresh_table()
        # 次回のデフォルトとして記憶
        self._last_attrs[attr_name] = value

    def _on_note_changed(self):
        """note フィールド変更時"""
        if self._edit_lane_idx is not None and self._edit_lane_idx < len(self._lanes):
            self._lanes[self._edit_lane_idx]["note"] = self._edt_note.text()

    def _on_id_field_changed(self):
        """N / X / Y フィールド変更時: レコードに反映し lane_number を再合成する"""
        sel_rows = [r.row() for r in self._tbl.selectionModel().selectedRows()]
        if not sel_rows and self._edit_lane_idx is not None:
            sel_rows = [self._edit_lane_idx]
        if not sel_rows:
            return

        n_val = self._edt_id_n.text().strip()
        x_val = self._edt_id_x.text().strip()
        y_val = self._edt_id_y.text().strip()

        for row in sel_rows:
            if row < len(self._lanes):
                rec = self._lanes[row]
                rec["lane_id_n"] = n_val
                rec["lane_id_x"] = x_val
                rec["lane_id_y"] = y_val
                # いずれかが手動入力されたらフラグを立てる
                if n_val or x_val or y_val:
                    rec["lane_id_manual"] = True
                # lane_number を再合成 (N_X-Y 形式)
                if n_val:
                    x_part = x_val if x_val else "0"
                    y_part = y_val if y_val else "0"
                    rec["lane_number"] = f"{n_val}_{x_part}-{y_part}"
                else:
                    rec["lane_number"] = ""

        self._refresh_table()
        self.gl.update()

    def _populate_attr_from_rec(self, rec: dict):
        """rec の属性値を属性パネルのウィジェットに反映する"""
        _BTN_OFF = ("QPushButton{background:#3c3c3c;color:#ccc;"
                    "border:1px solid #555;border-radius:3px;"
                    "padding:2px 4px;font-size:10px;}"
                    "QPushButton:hover{background:#4a4a4a;color:white;}")
        _BTN_ON  = ("QPushButton{background:#2a6ebb;color:white;"
                    "border:2px solid #4a8edb;border-radius:3px;"
                    "padding:2px 4px;font-size:10px;font-weight:bold;}")

        # class (ボタン式)
        cls_val = rec.get("class", "lane_line")
        for c, btn in self._cls_buttons.items():
            btn.setChecked(c == cls_val)
            btn.setStyleSheet(
                self._BTN_CLS_ON if c == cls_val else self._BTN_CLS_OFF)

        # 属性フィールドを再構築
        self._rebuild_attr_fields(cls_val)

        # ID (N_X-Y 分離フィールド)
        self._edt_id_n.blockSignals(True)
        self._edt_id_x.blockSignals(True)
        self._edt_id_y.blockSignals(True)
        self._edt_id_n.setText(str(rec.get("lane_id_n", "")))
        self._edt_id_x.setText(str(rec.get("lane_id_x", "")))
        self._edt_id_y.setText(str(rec.get("lane_id_y", "")))
        self._edt_id_n.blockSignals(False)
        self._edt_id_x.blockSignals(False)
        self._edt_id_y.blockSignals(False)

        # 各属性値をセット
        for attr_name, widget in self._attr_widgets.items():
            val = rec.get(attr_name, "")
            if isinstance(widget, dict):
                # ボタン式
                for v, btn in widget.items():
                    btn.setChecked(v == val)
                    btn.setStyleSheet(_BTN_ON if v == val else _BTN_OFF)
            elif isinstance(widget, QComboBox):
                widget.blockSignals(True)
                ci = widget.findText(str(val))
                if ci >= 0:
                    widget.setCurrentIndex(ci)
                else:
                    widget.setCurrentIndex(0)
                widget.blockSignals(False)
            elif isinstance(widget, QLineEdit):
                widget.blockSignals(True)
                widget.setText(str(val))
                widget.blockSignals(False)

        # note
        self._edt_note.blockSignals(True)
        self._edt_note.setText(rec.get("note", ""))
        self._edt_note.blockSignals(False)

    def _clear_attr_panel(self):
        """属性パネルをクリアする"""
        _BTN_OFF = ("QPushButton{background:#3c3c3c;color:#ccc;"
                    "border:1px solid #555;border-radius:3px;"
                    "padding:2px 4px;font-size:10px;}"
                    "QPushButton:hover{background:#4a4a4a;color:white;}")
        self._edt_id_n.setText("")
        self._edt_id_x.setText("")
        self._edt_id_y.setText("")
        self._edt_note.setText("")
        for widget in self._attr_widgets.values():
            if isinstance(widget, dict):
                for btn in widget.values():
                    btn.setChecked(False)
                    btn.setStyleSheet(_BTN_OFF)
            elif isinstance(widget, QComboBox):
                widget.blockSignals(True)
                widget.setCurrentIndex(0)
                widget.blockSignals(False)
            elif isinstance(widget, QLineEdit):
                widget.blockSignals(True)
                widget.clear()
                widget.blockSignals(False)

    # ==================================================================
    # プレースホルダ (後続の項目で実装)
    # ==================================================================
    def _set_color_mode(self, mode: str):
        """色塗り方法を切り替える。mode: "class" or "id" """
        self._color_mode = mode
        self._btn_color_class.setChecked(mode == "class")
        self._btn_color_class.setStyleSheet(
            self._CM_ON if mode == "class" else self._CM_OFF)
        self._btn_color_id.setChecked(mode == "id")
        self._btn_color_id.setStyleSheet(
            self._CM_ON if mode == "id" else self._CM_OFF)
        # テーブルの色も更新する
        self._refresh_table()
        self.gl.update()

    def _on_tbl_header_clicked(self, logical_index: int) -> None:
        """テーブルヘッダクリック時。

        列0（✓）: 全行のチェックボックスを全選択/全解除トグル。
        その他の列: _lanes を昇順/降順に並べ替え。
        同じ列を2回クリックすると昇順↔降順を切り替える。
        """
        if logical_index == 0:
            # ✓ 列: 全選択 / 全解除のトグル
            # 現在の状態を確認（1行でもチェック済みがあれば全解除、なければ全選択）
            any_checked = False
            for row in range(self._tbl.rowCount()):
                item = self._tbl.item(row, 0)
                if item is not None and item.checkState() == Qt.Checked:
                    any_checked = True
                    break

            new_state = Qt.Unchecked if any_checked else Qt.Checked
            self._tbl.blockSignals(True)
            for row in range(self._tbl.rowCount()):
                item = self._tbl.item(row, 0)
                if item is not None:
                    item.setCheckState(new_state)
            self._tbl.blockSignals(False)
            return

        # 同列を再クリックで昇降順トグル
        if self._tbl_sort_col == logical_index:
            self._tbl_sort_asc = not self._tbl_sort_asc
        else:
            self._tbl_sort_col = logical_index
            self._tbl_sort_asc = True

        # 列インデックスに対応するソートキーを定義
        _COL_KEYS = {
            1: lambda r: r.get("geometry_type", r.get("shape", "")),  # 型
            2: lambda r: _sort_int_or_str(r.get("lane_id_n", "")),    # N
            3: lambda r: _sort_int_or_str(r.get("lane_id_x", "")),    # X
            4: lambda r: _sort_int_or_str(r.get("lane_id_y", "")),    # Y
            5: lambda r: r.get("class", ""),                           # class
            6: lambda r: r.get("line_type", ""),                       # 線種
            7: lambda r: r.get("line_color", ""),                      # 色
            8: lambda r: len(r.get("vertices", [])),                   # 頂点数
        }
        key_fn = _COL_KEYS.get(logical_index)
        if key_fn is None:
            return

        self._lanes.sort(key=key_fn, reverse=not self._tbl_sort_asc)
        self.gl.draw_lane_data = self._lanes
        self._refresh_table()
        self.gl.update()

        # ヘッダにソートインジケータを表示
        hdr = self._tbl.horizontalHeader()
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(
            logical_index,
            Qt.AscendingOrder if self._tbl_sort_asc else Qt.DescendingOrder)

    # ==================================================================
    # ④ Lane 一覧テーブル
    # ==================================================================
    def _refresh_table(self):
        """_lanes の内容をテーブルに反映する"""
        self._tbl.blockSignals(True)
        self._tbl.setRowCount(0)
        rh = getattr(self, '_ROW_H', 28)
        # クラス別モード判定
        is_class_mode = (self._color_mode == "class")

        # source 別の行背景色 (手動=青み、自動系=緑み、半自動=シアン、インポート=紫)
        _SOURCE_BG = {
            "manual":        QColor(30,  50,  80,  60),   # 青み
            "auto":          QColor(20,  60,  20,  60),   # 緑み
            "sampling_auto": QColor(20,  60,  20,  60),   # 緑み
            "semi_auto":     QColor(20,  60,  55,  60),   # シアン
            "auto_corrected":QColor(60,  50,  20,  60),   # 琥珀
            "imported":      QColor(50,  20,  70,  60),   # 紫
        }

        for rec in self._lanes:
            row = self._tbl.rowCount()
            self._tbl.insertRow(row)
            self._tbl.setRowHeight(row, rh)

            cls_name = rec.get("class", "")
            geom = rec.get("geometry_type", rec.get("shape", ""))
            source = rec.get("source", "manual")
            row_bg = _SOURCE_BG.get(source, _SOURCE_BG["manual"])

            # 列0: チェックボックス
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)
            chk_item.setTextAlignment(Qt.AlignCenter)
            chk_item.setBackground(row_bg)
            self._tbl.setItem(row, 0, chk_item)

            # 列1: 型 (ply / pgn / pnt) + source アイコン
            if geom in ("polygon",):
                type_label = "pgn"
            elif geom in ("points", "point"):
                type_label = "pnt"
            else:
                type_label = "ply"

            # source を型欄に文字で表示（手動/自動/半自動 等）
            _SOURCE_LABEL = {
                "manual":        "手動",
                "auto":          "自動",
                "sampling_auto": "自動",
                "semi_auto":     "半自動",
                "auto_corrected":"修正済",
                "imported":      "取込",
            }
            src_label = _SOURCE_LABEL.get(source, "?")
            type_label = f"{src_label}/{type_label}"

            # 色: クラス別モードはクラス色、ID別モードは _color
            if is_class_mode:
                r_c, g_c, b_c = CLASS_DISPLAY_COLORS.get(cls_name, (0.7, 0.7, 0.7))
            else:
                r_c, g_c, b_c = _lane_color(rec)

            type_item = QTableWidgetItem(type_label)
            type_item.setBackground(QColor(int(r_c*255), int(g_c*255), int(b_c*255), 120))
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            type_item.setTextAlignment(Qt.AlignCenter)
            self._tbl.setItem(row, 1, type_item)

            # 列2,3,4: N, X, Y
            # stop_line / zebra_line / 点クラス は ID 付与対象外のため編集不可・グレー表示
            _ID_NO_CLASS = {"stop_line", "zebra_line",
                            "split_point", "merge_point"}
            id_editable = cls_name not in _ID_NO_CLASS

            n_item = QTableWidgetItem(str(rec.get("lane_id_n", "")))
            x_item = QTableWidgetItem(str(rec.get("lane_id_x", "")))
            y_item = QTableWidgetItem(str(rec.get("lane_id_y", "")))

            if not id_editable:
                _gray = QColor(50, 50, 50, 120)
                for item in (n_item, x_item, y_item):
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(_gray)
                    item.setForeground(QColor(100, 100, 100))
            else:
                for item in (n_item, x_item, y_item):
                    item.setBackground(row_bg)

            self._tbl.setItem(row, 2, n_item)
            self._tbl.setItem(row, 3, x_item)
            self._tbl.setItem(row, 4, y_item)

            # 列5: class
            cls_item = QTableWidgetItem(cls_name)
            cls_item.setFlags(cls_item.flags() & ~Qt.ItemIsEditable)
            if is_class_mode:
                cls_item.setBackground(QColor(int(r_c*255), int(g_c*255), int(b_c*255), 120))
            else:
                cls_item.setBackground(row_bg)
            self._tbl.setItem(row, 5, cls_item)

            # 列6: 線種 (lane_lineのみ表示)
            if cls_name == "lane_line" and geom not in ("polygon", "point", "points"):
                line_type_val = str(rec.get("line_type", ""))
            else:
                line_type_val = ""
            lt_item = QTableWidgetItem(line_type_val)
            lt_item.setFlags(lt_item.flags() & ~Qt.ItemIsEditable)
            lt_item.setTextAlignment(Qt.AlignCenter)
            lt_item.setBackground(row_bg)
            self._tbl.setItem(row, 6, lt_item)

            # 列7: 色 (lane_lineのみ表示)
            if cls_name == "lane_line" and geom not in ("polygon", "point", "points"):
                line_color_val = str(rec.get("line_color", ""))
            else:
                line_color_val = ""
            lc_item = QTableWidgetItem(line_color_val)
            lc_item.setFlags(lc_item.flags() & ~Qt.ItemIsEditable)
            lc_item.setTextAlignment(Qt.AlignCenter)
            lc_item.setBackground(row_bg)
            self._tbl.setItem(row, 7, lc_item)

            # 列8: 頂点数（編集不可）
            vert_item = QTableWidgetItem(str(len(rec.get("vertices", []))))
            vert_item.setFlags(vert_item.flags() & ~Qt.ItemIsEditable)
            vert_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            vert_item.setBackground(row_bg)
            self._tbl.setItem(row, 8, vert_item)

        self._tbl.blockSignals(False)

    def _on_tbl_cell_changed(self, row: int, col: int):
        """テーブルセル編集時: N/X/Y列の変更をレコードに反映する"""
        if col < 2 or col > 4:
            return  # N=2, X=3, Y=4 列のみ処理
        if row < 0 or row >= len(self._lanes):
            return
        rec = self._lanes[row]
        item = self._tbl.item(row, col)
        val = item.text().strip() if item else ""
        if col == 2:
            rec["lane_id_n"] = val
        elif col == 3:
            rec["lane_id_x"] = val
        elif col == 4:
            rec["lane_id_y"] = val
        # テーブル直接編集でも手動フラグを立てる
        if val:
            rec["lane_id_manual"] = True

    def _delete_checked_rows(self):
        """チェックボックスがオンの行のレーンを削除する。"""
        # チェック済み行のインデックスを収集（逆順で削除するため降順）
        checked_rows = []
        for row in range(self._tbl.rowCount()):
            chk_item = self._tbl.item(row, 0)
            if chk_item and chk_item.checkState() == Qt.Checked:
                checked_rows.append(row)

        if not checked_rows:
            return

        # 対象レーンを削除（逆順でインデックスがずれないように）
        for row in sorted(checked_rows, reverse=True):
            if row < len(self._lanes):
                del self._lanes[row]

        # 選択・ハイライト状態をクリア
        self._edit_lane_idx = None
        self._selection_order = []
        self.gl.draw_lane_highlight  = None
        self.gl.draw_lane_highlights = []
        self.gl.draw_lane_edit_verts = None

        # テーブルを更新してGL描画を再描画
        self.gl.draw_lane_data = self._lanes
        self._refresh_table()
        self.gl.update()

        if self.main_win is not None:
            self.main_win._has_unsaved_changes = True

    def _on_btn_auto_n(self) -> None:
        """テーブル先頭のN自動付与ボタンが押されたときの処理。
        MainWindowの_auto_assign_nに委譲する。
        """
        if self.main_win is not None and hasattr(self.main_win, '_auto_assign_n'):
            self.main_win._auto_assign_n()

    def _on_lane_points_visibility(self, checked: bool):
        """レーン点群の表示/非表示を切り替える。"""
        self.gl.set_layer_visible("lane_points", checked)

    def _retranslate_ui(self) -> None:
        """言語切り替え時に固定ウィジェットのテキストを更新し、動的フィールドを再構築する。"""
        # チェックボックス
        self._chk_lane_points.setText(t("llp_chk_lane_points"))
        # 描画ツールボタン
        self._btn_polyline.setText(t("llp_btn_polyline"))
        self._btn_polyline.setToolTip(t("llp_tooltip_polyline"))
        self._btn_polygon.setText(t("llp_btn_polygon"))
        self._btn_polygon.setToolTip(t("llp_tooltip_polygon"))
        self._btn_rbbox.setText(t("llp_btn_point"))
        self._btn_rbbox.setToolTip(t("llp_tooltip_point"))
        # ヒントラベル
        self._lbl_hint.setText(t("llp_hint_default"))
        # クラスボタン
        cls_label_keys = {
            "lane_line": "llp_cls_lane_line", "stop_line": "llp_cls_stop_line",
            "curb_boundary": "llp_cls_curb_boundary",
            "fence_boundary": "llp_cls_fence_boundary",
            "TrafficCone_boundary": "llp_cls_cone_boundary",
            "WaterSafety_boundary": "llp_cls_water_boundary",
            "boundary": "llp_cls_boundary", "zebra_line": "llp_cls_zebra_line",
            "split_point": "llp_cls_split_point", "merge_point": "llp_cls_merge_point",
        }
        for cls_name, btn in self._cls_buttons.items():
            key = cls_label_keys.get(cls_name)
            if key:
                btn.setText(t(key))
        # 動的属性フィールドを再構築（ボタングループのタイトル・ラベルを更新）
        current_cls = self._last_class or "lane_line"
        self._rebuild_attr_fields(current_cls)
        # 現在の値を再反映
        if self._edit_lane_idx is not None and self._edit_lane_idx < len(self._lanes):
            self._populate_attr_from_rec(self._lanes[self._edit_lane_idx])

    def _notify_annotation_selected_first_point(self, rec: dict) -> None:
        """一覧選択したアノテーションの先頭点をMainWindowへ通知する。"""
        if self.main_win is None or rec is None:
            return

        verts = rec.get("vertices")
        if verts is None or len(verts) == 0:
            verts = rec.get("poly3d")
        if verts is None or len(verts) == 0:
            return

        first = verts[0]
        if len(first) < 2:
            return

        callback = getattr(
            self.main_win,
            "_on_annotation_selected_first_point",
            None,
        )
        if not callable(callback):
            return

        try:
            callback(
                float(first[0]),
                float(first[1]),
            )
        except Exception as exc:
            print(
                "[Selection Follow] callback warning: "
                f"{exc}"
            )

    def _on_tbl_select(self):
        """テーブル行選択時: 属性パネルに反映 + GL ハイライト"""
        rows = self._tbl.selectionModel().selectedRows()
        if not rows:
            self._edit_lane_idx = None
            self.gl.draw_lane_highlight  = None
            self.gl.draw_lane_highlights = []
            self.gl.draw_lane_edit_verts = None
            self._clear_attr_panel()
            self.gl.update()
            return

        idx = rows[0].row()
        self._edit_lane_idx = idx
        if idx < len(self._lanes):
            rec = self._lanes[idx]
            # 属性パネルに反映
            self._populate_attr_from_rec(rec)
            # GL ハイライト
            self.gl.draw_lane_highlight  = rec.get("poly3d")
            self.gl.draw_lane_highlights = []
            self.gl.draw_lane_edit_verts = np.array(
                rec["vertices"], dtype=np.float32)
            self.gl.draw_lane_edit_panel = self

            # 「アノテーション一覧を直接クリックして選択した場合」だけ、
            # 先頭点が2Dビュー中心になるようMainWindowへ通知する。
            #
            # 2D/3Dビュー上でポリライン・ポリゴン・ポイントをクリックして
            # 選択した場合は、テーブル選択だけ同期し、表示範囲は一切動かさない。
            if not getattr(self, "_selection_from_gl_click", False):
                self._notify_annotation_selected_first_point(rec)

            # 複数選択時
            if len(rows) > 1:
                self.gl.draw_lane_highlights = [
                    self._lanes[r.row()].get("poly3d")
                    for r in rows
                    if r.row() < len(self._lanes)
                    and self._lanes[r.row()].get("poly3d") is not None
                ]
                self.gl.draw_lane_highlight = None
        self.gl.update()

    def _delete_selected(self):
        """選択行を削除する"""
        rows = sorted(
            [r.row() for r in self._tbl.selectionModel().selectedRows()],
            reverse=True)
        for r in rows:
            if r < len(self._lanes):
                self._lanes.pop(r)
        self._edit_lane_idx          = None
        self.gl.draw_lane_data       = self._lanes
        self.gl.draw_lane_highlight  = None
        self.gl.draw_lane_highlights = []
        self.gl.draw_lane_edit_verts = None
        self.gl.update()
        self._refresh_table()
        self._clear_attr_panel()

    def _split_selected(self):
        """分割モードを開始する。GL上で2点をクリックして分割線を指定する。
        分割線と交差するすべてのポリラインを交差地点で分割する。
        選択不要。"""
        self._split_pts = []
        self._split_mode = True
        self.gl.setCursor(Qt.CrossCursor)
        self._lbl_hint.setText(t("hint_split_start"))

    def on_split_click(self, screen_pt):
        """分割モード中のクリック処理"""
        if not getattr(self, '_split_mode', False):
            return False
        gz = self._ground_z()
        world = _screen_to_world_ray(self.gl, screen_pt, gz)
        if world is None:
            return True
        self._split_pts.append(world)
        if len(self._split_pts) == 1:
            self._lbl_hint.setText(t("hint_split_end"))
            return True
        elif len(self._split_pts) >= 2:
            self._execute_split()
            return True
        return True

    def _execute_split(self):
        """横切る線と交差するすべてのポリラインを交差地点で分割する。

        対象: geometry_type == "polyline" のレコードのみ（ポリゴン・点は除外）
        分割線と複数のポリラインが交差する場合、すべてを分割する。
        """
        if len(self._split_pts) < 2:
            self._cancel_split()
            return

        p1 = self._split_pts[0]
        p2 = self._split_pts[1]

        # 分割対象: ポリラインのみ
        target_indices = [
            i for i, rec in enumerate(self._lanes)
            if rec.get("geometry_type", rec.get("shape", "")) == "polyline"
            and len(rec.get("vertices", [])) >= 2
        ]

        if not target_indices:
            self._lbl_hint.setText(t("hint_split_no_cross"))
            self._cancel_split()
            return

        # 交差するポリラインを収集（元インデックス昇順で処理）
        splits_to_do = []  # [(original_idx, cross_pt, split_insert_idx), ...]
        for idx in target_indices:
            rec = self._lanes[idx]
            verts = rec.get("vertices", [])
            poly = rec.get("poly3d")
            pts_src = poly if (poly is not None and len(poly) >= 2) else verts
            if len(pts_src) < 2:
                continue

            # 横切る線分と各セグメントの交差を検出
            best_t = None
            best_seg = None
            for i in range(len(pts_src) - 1):
                a = (float(pts_src[i][0]),   float(pts_src[i][1]))
                b = (float(pts_src[i+1][0]), float(pts_src[i+1][1]))
                c = (float(p1[0]), float(p1[1]))
                d = (float(p2[0]), float(p2[1]))
                t_param = self._segment_intersection(a, b, c, d)
                if t_param is not None:
                    best_t = t_param
                    best_seg = i
                    break  # 最初の交差のみ

            if best_seg is None:
                continue  # このポリラインは交差しない

            # 交差点の3D座標を計算
            cross_x = float(pts_src[best_seg][0]) + best_t * (float(pts_src[best_seg+1][0]) - float(pts_src[best_seg][0]))
            cross_y = float(pts_src[best_seg][1]) + best_t * (float(pts_src[best_seg+1][1]) - float(pts_src[best_seg][1]))
            cross_z = float(pts_src[best_seg][2]) + best_t * (float(pts_src[best_seg+1][2]) - float(pts_src[best_seg][2]))
            cross_pt = (cross_x, cross_y, cross_z)

            # 制御点列（vertices）上で交差点に最も近いエッジを見つける
            best_edge = 0
            best_dist = float('inf')
            for i in range(len(verts) - 1):
                mx = (verts[i][0] + verts[i+1][0]) / 2
                my = (verts[i][1] + verts[i+1][1]) / 2
                dist = math.sqrt((cross_x - mx)**2 + (cross_y - my)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_edge = i

            split_insert_idx = best_edge + 1  # この位置で分割

            verts_a = list(verts[:split_insert_idx]) + [cross_pt]
            verts_b = [cross_pt] + list(verts[split_insert_idx:])

            if len(verts_a) < 2 or len(verts_b) < 2:
                continue  # 端すぎて分割不可

            splits_to_do.append((idx, verts_a, verts_b, rec))

        if not splits_to_do:
            self._lbl_hint.setText(t("hint_split_no_cross"))
            self._cancel_split()
            return

        # 後ろから順に分割（インデックスのずれを防ぐ）
        splits_to_do.sort(key=lambda x: x[0], reverse=True)

        def _make(vlist, src_rec):
            new_rec = _default_lane_record(self._next_id)
            self._next_id += 1
            new_rec["geometry_type"] = "polyline"
            new_rec["shape"]         = "polyline"
            new_rec["class"]         = src_rec.get("class", "lane_line")
            new_rec["buffer"]        = src_rec.get("buffer", 0.10)
            new_rec["line_type"]     = src_rec.get("line_type", "solid")
            new_rec["line_color"]    = src_rec.get("line_color", "white")
            new_rec["line_count"]    = src_rec.get("line_count", "single")
            new_rec["source"]        = src_rec.get("source", "manual")
            new_rec["vertices"]      = [(float(v[0]), float(v[1]), float(v[2]))
                                        for v in vlist]
            _rebuild_poly3d(new_rec)
            return new_rec

        n_split = len(splits_to_do)
        for orig_idx, verts_a, verts_b, src_rec in splits_to_do:
            entry_a = _make(verts_a, src_rec)
            entry_b = _make(verts_b, src_rec)
            self._lanes.pop(orig_idx)
            self._lanes.insert(orig_idx, entry_b)
            self._lanes.insert(orig_idx, entry_a)

        self._cancel_split()
        self.gl.draw_lane_data = self._lanes
        self.gl.update()
        self._refresh_table()
        self._lbl_hint.setText(t("hint_split_done").format(n_split) if "{" in t("hint_split_done") else t("hint_split_done") + f" ({n_split}本を分割)")

    @staticmethod
    def _segment_intersection(a, b, c, d):
        """2D線分 AB と CD の交差パラメータ t (AB上) を返す。交差なしなら None。"""
        dx1 = b[0] - a[0]; dy1 = b[1] - a[1]
        dx2 = d[0] - c[0]; dy2 = d[1] - c[1]
        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-12:
            return None
        t = ((c[0] - a[0]) * dy2 - (c[1] - a[1]) * dx2) / denom
        u = ((c[0] - a[0]) * dy1 - (c[1] - a[1]) * dx1) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def _cancel_split(self):
        """分割モードをキャンセルする"""
        self._split_mode = False
        self._split_pts = []
        self.gl.setCursor(Qt.ArrowCursor)

    def _merge_selected(self):
        """選択中の全Laneの頂点を主軸方向に並べ直し、1本のスプラインに統合する。

        1. 全頂点を収集
        2. PCA で主軸方向を求め、主軸に射影してソート
        3. 主軸からの距離が大きい外れ値を除去
        4. 近すぎる点を間引き
        5. スプライン再構築
        """
        rows = sorted([r.row() for r in self._tbl.selectionModel().selectedRows()])
        if len(rows) < 2:
            QMessageBox.information(self, t("dlg_merge_title"),
                t("dlg_merge_too_few"))
            return

        recs = [self._lanes[r] for r in rows if r < len(self._lanes)]
        print(f"[MERGE] rows={rows}, recs={len(recs)}")
        if len(recs) < 2:
            return

        # ── 1. 全頂点を収集 ──────────────────────────────────────
        all_pts = []
        for r in recs:
            for v in r.get("vertices", []):
                all_pts.append((float(v[0]), float(v[1]), float(v[2])))
        if len(all_pts) < 2:
            return

        pts = np.array(all_pts, dtype=np.float64)
        n = len(pts)

        # ── 2. PCA で主軸方向を求める ────────────────────────────
        center = pts[:, :2].mean(axis=0)
        centered = pts[:, :2] - center
        cov = np.cov(centered.T)
        if cov.ndim < 2:
            main_dir = np.array([1.0, 0.0])
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_dir = eigvecs[:, -1]  # 最大固有値の固有ベクトル

        # 主軸に射影
        proj = centered @ main_dir  # 各点の主軸上の位置
        # 主軸からの垂直距離
        perp_dir = np.array([-main_dir[1], main_dir[0]])
        perp_dist = np.abs(centered @ perp_dir)

        # ── 3. 外れ値除去（主軸からの距離が中央値+2σ以上） ──────
        if n > 4:
            med_perp = np.median(perp_dist)
            std_perp = np.std(perp_dist)
            thresh = med_perp + 2.0 * std_perp + 0.5  # 最低0.5mのマージン
            inlier = perp_dist <= thresh
            pts = pts[inlier]
            proj = proj[inlier]
            perp_dist = perp_dist[inlier]
            print(f"[MERGE] outlier removal: {n} -> {len(pts)} "
                  f"(thresh={thresh:.2f}m)")
            n = len(pts)

        if n < 2:
            QMessageBox.warning(self, t("dlg_merge_title"),
                t("dlg_merge_outlier_removed"))
            return

        # ── 4. 主軸射影値でソート ────────────────────────────────
        order = np.argsort(proj)
        pts = pts[order]
        proj = proj[order]

        # ── 5. 近すぎる点を間引き ────────────────────────────────
        # 全体の長さに対して適切な間隔を計算
        total_len = proj[-1] - proj[0]
        min_spacing = max(total_len / 200.0, 0.05)  # 最低5cm

        thinned = [pts[0]]
        for i in range(1, n):
            d = math.sqrt((pts[i][0] - thinned[-1][0])**2 +
                          (pts[i][1] - thinned[-1][1])**2)
            if d >= min_spacing:
                thinned.append(pts[i])
        # 最後の点は必ず含める
        d_last = math.sqrt((pts[-1][0] - thinned[-1][0])**2 +
                           (pts[-1][1] - thinned[-1][1])**2)
        if d_last > 0.01 and len(thinned) > 1:
            thinned.append(pts[-1])

        print(f"[MERGE] thinned: {n} -> {len(thinned)} "
              f"(spacing={min_spacing:.3f}m)")

        if len(thinned) < 2:
            QMessageBox.warning(self, t("dlg_merge_title"),
                t("dlg_merge_too_few_after_thin"))
            return

        # ── 6. 曲率に応じた頂点数調整 + 統合レコード作成 ────────
        thinned = _resample_by_curvature(thinned, min_pts=4, max_pts=10)

        merged = _default_lane_record(self._next_id)
        self._next_id += 1
        merged["geometry_type"] = "polyline"
        merged["shape"] = "polyline"
        merged["class"] = recs[0].get("class", "lane_line")
        merged["vertices"] = [(float(v[0]), float(v[1]), float(v[2]))
                              for v in thinned]
        _rebuild_poly3d(merged)

        # 元の全レコードを削除して統合レコードを挿入
        insert_pos = min(rows)
        for r in sorted(rows, reverse=True):
            if r < len(self._lanes):
                self._lanes.pop(r)
        self._lanes.insert(insert_pos, merged)

        # 状態リセット
        self._edit_lane_idx          = None
        self._selection_order        = []
        self.gl.draw_lane_highlight  = None
        self.gl.draw_lane_highlights = []
        self.gl.draw_lane_edit_verts = None
        self.gl.draw_lane_data       = self._lanes
        self.gl.update()
        self._refresh_table()
        self._tbl.selectRow(insert_pos)
        self._lbl_hint.setText(
            t("hint_merge_done").format(len(recs), len(thinned)))

    # ==================================================================
    # プレースホルダ (後続の項目で実装)
    # ==================================================================
    def normalize_all_geometries(
        self,
        *,
        show_error: bool = True,
    ) -> bool:
        """全レコードを保存可能な最終geometryへ正規化・検証する。

        これは保存前の最終防衛線として外部(MainWindow/Ctrl+S/CVAT)からも呼べる。
        classを正として geometry_type / shape を同期し、
        poly3dも最終geometryから再構築する。

        Returns:
            True: 全件valid
            False: 1件以上invalid（保存は中止すべき）
        """
        errors = []

        for i, rec in enumerate(self._lanes):
            before = rec.get(
                "geometry_type",
                rec.get("shape", "polyline"),
            )
            geom = _normalize_record_geometry(rec)
            err = _geometry_validation_error(rec)

            if err:
                errors.append(
                    f"{rec.get('id', i)} "
                    f"[{rec.get('class','?')}/{geom}]: {err}"
                )
                continue

            use_spline = bool(
                geom == "polyline"
                and rec.get("_is_spline", False)
            )
            _rebuild_poly3d(
                rec,
                use_spline=use_spline,
            )

            if before != geom:
                print(
                    "[Geometry Normalize] "
                    f"id={rec.get('id',i)} "
                    f"class={rec.get('class','?')} "
                    f"{before} -> {geom}"
                )

        if errors:
            if show_error:
                QMessageBox.warning(
                    self,
                    "Geometry整合性エラー",
                    "保存できません。geometryの頂点数を修正してください。\n\n"
                    + "\n".join(errors[:20]),
                )
            return False

        self.gl.draw_lane_data = self._lanes
        self.gl.update()
        self._refresh_table()
        return True

    # ==================================================================
    # ⑤ 保存 / 読み込み
    # ==================================================================
    def _save_json(self):
        """spec.md 準拠の JSON 形式で保存する"""
        if not self._lanes:
            QMessageBox.information(self, "Save", "保存する Lane がありません。")
            return

        if not self.normalize_all_geometries(show_error=True):
            return

        init_dir = ""
        if self._base_layer and hasattr(self._base_layer, 'filepath'):
            init_dir = str(Path(self._base_layer.filepath).parent)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Lane JSON",
            str(Path(init_dir) / "lane_lines.json") if init_dir else "lane_lines.json",
            "JSON (*.json);;All (*)")
        if not path:
            return

        # Z 値を点群から補間
        self._interpolate_z_all()

        # JSON 出力用にレコードを整形
        out_list = []
        for rec in self._lanes:
            entry = {}
            for k, v in rec.items():
                if k == "poly3d":
                    continue  # 描画用データは除外
                if k == "vertices":
                    entry["points"] = [[round(p[0], 6), round(p[1], 6),
                                        round(p[2], 6)] for p in v]
                elif k == "buffer":
                    if rec.get("geometry_type") == "polyline":
                        entry["buffer_m"] = round(v, 4)
                elif k == "geometry_type":
                    entry["geometry_type"] = v
                else:
                    if v != "" and v is not None:
                        entry[k] = v
            out_list.append(entry)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out_list, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Saved",
                f"保存しました:\n{path}\n({len(out_list)} lanes)")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _save_txt(self):
        """既存互換の TXT 形式で保存する (DrawLanePanel と同一フォーマット)"""
        if not self._lanes:
            QMessageBox.information(self, "Save", "保存する Lane がありません。")
            return

        if not self.normalize_all_geometries(show_error=True):
            return

        # Z 値を点群から補間
        self._interpolate_z_all()

        init_dir = ""
        if self._base_layer and hasattr(self._base_layer, 'filepath'):
            init_dir = str(Path(self._base_layer.filepath).parent)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Lane TXT",
            str(Path(init_dir) / "lanes.txt") if init_dir else "lanes.txt",
            "Text Files (*.txt);;All (*)")
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# id\tclass\tgeometry_type\tlane_type\tlane_id"
                        "\tbuffer_m\tvertices(x,y,z|...)\n")
                for rec in self._lanes:
                    vstr = "|".join(
                        f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}"
                        for v in rec["vertices"])
                    buf = rec.get("buffer", 0.0)
                    f.write(
                        f"{rec['id']}\t{rec.get('class','')}\t"
                        f"{rec.get('geometry_type','')}\t"
                        f"{rec.get('lane_type','')}\t"
                        f"{rec.get('lane_id','')}\t"
                        f"{buf:.4f}\t{vstr}\n")
            QMessageBox.information(self, "Saved",
                f"保存しました:\n{path}\n({len(self._lanes)} lanes)")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_file(self):
        """JSON または TXT ファイルを読み込む"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Lane File", "",
            "JSON/TXT (*.json *.txt);;All (*)")
        if not path:
            return

        ext = Path(path).suffix.lower()
        try:
            if ext == ".json":
                self._load_json(path)
            else:
                self._load_txt(path)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def import_lanegen_json(
        self,
        path: str,
        *,
        replace_previous: bool = True,
    ) -> dict:
        """LaneGen Annotator JSONを作業データへ安全に取り込む。

        - 手動・半自動・通常import済みrecordは保持する。
        - ``replace_previous=True`` では、以前のLaneGen自動recordだけ置換する。
        - LaneGenの ``auto_track_id`` と ``generation`` metadataを保持する。
        - 表示IDが既存IDと衝突する場合だけ新しい ``lane_XXXX`` を付与する。
        - 全件の変換・geometry検証に成功してから一括反映する。
        """
        source_path = Path(path).expanduser().resolve()
        with source_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("LaneGen JSONのトップレベルはリストである必要があります")

        kept = [
            rec for rec in self._lanes
            if not (replace_previous and _is_lanegen_auto_record(rec))
        ]
        removed = len(self._lanes) - len(kept)
        used_ids = {str(rec.get("id", "")) for rec in kept if rec.get("id")}

        def allocate_id(preferred: str = "") -> str:
            nonlocal_next = self._next_id
            preferred = str(preferred or "").strip()
            if preferred and preferred not in used_ids:
                used_ids.add(preferred)
                return preferred
            while True:
                candidate = f"lane_{nonlocal_next:04d}"
                nonlocal_next += 1
                if candidate not in used_ids:
                    used_ids.add(candidate)
                    # Pythonのnonlocalを避け、panel値を単調増加させる。
                    self._next_id = max(self._next_id, nonlocal_next)
                    return candidate

        imported = []
        # 失敗しても現在のself._lanesを変更しないよう、一時list上で構築する。
        saved_next_id = self._next_id
        try:
            for index, entry in enumerate(data, start=1):
                rec = _record_from_json_entry(entry, saved_next_id + index - 1)
                generated_id = str(rec.get("id", ""))
                rec["id"] = allocate_id(generated_id)
                rec["source"] = "auto"
                if generated_id and generated_id != rec["id"]:
                    rec.setdefault("generation", {})
                    if isinstance(rec["generation"], dict):
                        rec["generation"].setdefault(
                            "generated_display_id", generated_id
                        )
                imported.append(rec)
        except Exception:
            self._next_id = saved_next_id
            raise

        candidate = kept + imported
        # IDの最終一意性確認。
        ids = [str(rec.get("id", "")) for rec in candidate]
        if len(ids) != len(set(ids)):
            self._next_id = saved_next_id
            raise ValueError("LaneGen取込後のannotation IDが重複しています")

        # ここから初めて作業データへ反映する。
        self._lanes = candidate
        max_lane_num = 0
        for rec in self._lanes:
            rid = str(rec.get("id", ""))
            if rid.startswith("lane_"):
                try:
                    max_lane_num = max(max_lane_num, int(rid.split("_")[1]))
                except (ValueError, IndexError):
                    pass
        self._next_id = max(self._next_id, max_lane_num + 1, len(self._lanes) + 1)

        self.gl.draw_lane_data = self._lanes
        self._refresh_table()
        self.gl.update()
        if self.main_win is not None:
            self.main_win._has_unsaved_changes = True

        self._lbl_hint.setText(
            f"LaneGen: {len(imported)}件取込 / {removed}件置換 "
            f"({source_path.name})"
        )
        return {
            "imported": len(imported),
            "replaced": removed,
            "total": len(self._lanes),
            "path": str(source_path),
        }

    def _load_json(self, path: str):
        """JSONファイルをtransactionalに読み込む。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON のトップレベルはリストである必要があります")

        new_lanes = []
        next_id = 1
        for entry in data:
            rec = _record_from_json_entry(entry, next_id)
            new_lanes.append(rec)
            rid = str(rec.get("id", ""))
            if rid.startswith("lane_"):
                try:
                    next_id = max(next_id, int(rid.split("_")[1]) + 1)
                except (ValueError, IndexError):
                    pass
            next_id = max(next_id, len(new_lanes) + 1)

        ids = [str(rec.get("id", "")) for rec in new_lanes]
        if len(ids) != len(set(ids)):
            raise ValueError("JSON内のannotation IDが重複しています")

        self._lanes = new_lanes
        self._next_id = next_id
        self.gl.draw_lane_data = self._lanes
        self.gl.update()
        self._refresh_table()
        self._lbl_hint.setText(
            f"📂 {len(self._lanes)} lanes を読み込みました: {Path(path).name}"
        )

    def _load_txt(self, path: str):
        """TXT ファイルを読み込む (DrawLanePanel 互換)"""
        self._lanes.clear()
        self._next_id = 1
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                rec = _default_lane_record(self._next_id)
                rec["id"] = parts[0]
                # 2列目: class or shape (既存互換)
                col1 = parts[1]
                if col1 in ("polyline", "polygon", "point", "merged"):
                    # 旧フォーマット: id shape lane_type lane_id buffer verts
                    raw_geom = "polyline" if col1 == "merged" else col1
                    rec["class"] = _default_class_for_geometry(raw_geom)
                    rec["geometry_type"] = raw_geom
                    rec["shape"] = raw_geom
                    rec["lane_type"] = parts[2] if len(parts) > 2 else ""
                    rec["lane_id"]   = parts[3] if len(parts) > 3 else ""
                    rec["buffer"]    = float(parts[4]) if len(parts) > 4 else 0.10
                    vraw = parts[5] if len(parts) > 5 else ""
                else:
                    # 新フォーマット: id class geom lane_type lane_id buffer verts
                    raw_geom = parts[2] if len(parts) > 2 else "polyline"
                    rec["class"] = (
                        col1
                        if col1 in ALL_CLASSES
                        else _default_class_for_geometry(raw_geom)
                    )
                    rec["geometry_type"] = raw_geom
                    rec["shape"]         = raw_geom
                    rec["lane_type"]     = parts[3] if len(parts) > 3 else ""
                    rec["lane_id"]       = parts[4] if len(parts) > 4 else ""
                    rec["buffer"]        = float(parts[5]) if len(parts) > 5 else 0.10
                    vraw = parts[6] if len(parts) > 6 else ""

                verts = []
                for tok in vraw.split("|"):
                    tok = tok.strip()
                    if not tok:
                        continue
                    xyz = [float(v) for v in tok.split(",")]
                    if len(xyz) >= 3:
                        verts.append((xyz[0], xyz[1], xyz[2]))
                rec["vertices"] = verts
                _normalize_record_geometry(rec)
                err = _geometry_validation_error(rec)
                if err:
                    raise ValueError(
                        f"不正geometry: {rec.get('id','?')} "
                        f"class={rec.get('class')} "
                        f"geometry={rec.get('geometry_type')}: {err}"
                    )
                _rebuild_poly3d(rec)
                self._lanes.append(rec)
                try:
                    num = int(rec["id"].replace("lane_", ""))
                    self._next_id = max(self._next_id, num + 1)
                except ValueError:
                    pass
                self._next_id = max(self._next_id,
                                    len(self._lanes) + 1)

        self.gl.draw_lane_data = self._lanes
        self.gl.update()
        self._refresh_table()
        self._lbl_hint.setText(
            f"📂 {len(self._lanes)} lanes を読み込みました: {Path(path).name}")

    # ── Z 値補間ヘルパー ─────────────────────────────────────────
    def _interpolate_z_all(self):
        """全 Lane の頂点 Z 値をベースレイヤーの点群から補間する"""
        layer = self._base_layer
        if (layer is None or not getattr(layer, 'loaded', False)
                or layer.xyz is None):
            return
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return

        tree = cKDTree(layer.xyz[:, :2])

        def _z_from_xy(x, y, radius=0.5, k=8):
            dists, idxs = tree.query([x, y], k=k, workers=1,
                                     distance_upper_bound=radius)
            valid = [i for d, i in zip(dists, idxs)
                     if np.isfinite(d) and i < len(layer.xyz)]
            if not valid:
                dists2, idxs2 = tree.query([x, y], k=1)
                if np.isfinite(dists2):
                    return float(layer.xyz[idxs2, 2])
                return 0.0
            return float(np.mean(layer.xyz[valid, 2]))

        for rec in self._lanes:
            new_verts = []
            for vx, vy, vz in rec["vertices"]:
                z = _z_from_xy(vx, vy)
                new_verts.append((vx, vy, z))
            rec["vertices"] = new_verts
            _rebuild_poly3d(rec)

    # ==================================================================
    # ⑥ 頂点編集
    # ==================================================================
    def on_vertex_drag_start(self, vert_idx: int):
        """GL から呼ばれる: 頂点ドラッグ開始"""
        # アンドゥ用スナップショット保存
        if self._edit_lane_idx is not None and self._edit_lane_idx < len(self._lanes):
            rec = self._lanes[self._edit_lane_idx]
            self._undo_state = (
                self._edit_lane_idx,
                [list(v) for v in rec["vertices"]]
            )
        self._edit_vert_idx = vert_idx
        self._edit_dragging = True

    def on_vertex_drag_move(self, screen_pt):
        """GL から呼ばれる: ドラッグ中 — 頂点をワールド座標に移動

        回転BBOX (note に 'RBBOX' を含む4頂点ポリゴン) の場合:
        - 通常ドラッグ: 直角を維持したまま長方形を変形
        - Shift+ドラッグ: 中心を固定して回転
        - 中心付近をドラッグ: 全体を平行移動
        """
        if not self._edit_dragging or self._edit_lane_idx is None:
            return
        if self._edit_vert_idx is None:
            return
        rec = self._lanes[self._edit_lane_idx]
        verts = list(rec["vertices"])
        vi = self._edit_vert_idx
        if vi >= len(verts):
            return
        current_z = float(verts[vi][2]) if len(verts[vi]) > 2 else self._ground_z()
        world = _screen_to_world_ray(self.gl, screen_pt, current_z)
        if world is None:
            world = _screen_to_world_ray(self.gl, screen_pt, self._ground_z())
        if world is None:
            return

        # 回転BBOX の特殊操作
        is_rbbox = (len(verts) == 4
                    and "RBBOX" in rec.get("note", "")
                    and rec.get("geometry_type") == "polygon")
        if is_rbbox:
            # Shift キー判定
            from PyQt5.QtWidgets import QApplication
            mods = QApplication.keyboardModifiers()
            shift = bool(mods & Qt.ShiftModifier)

            if shift:
                # Shift+ドラッグ → 中心を固定して回転
                verts = self._rbbox_rotate(verts, vi, world)
            else:
                # 通常ドラッグ → 直角維持変形
                verts = self._rbbox_constrained_move(verts, vi, world)

            # note の角度情報を更新
            self._update_rbbox_note(rec, verts)
        else:
            verts[vi] = world

        rec["vertices"] = verts
        self._rebuild_poly3d_cur(rec)
        self.gl.draw_lane_edit_verts = np.array(verts, dtype=np.float32)
        self.gl.draw_lane_highlight  = rec.get("poly3d")
        self.gl.draw_lane_data       = self._lanes
        self.gl.update()

    @staticmethod
    def _rbbox_constrained_move(verts: list, vi: int, new_pos: tuple) -> list:
        """回転BBOX の1頂点を移動し、直角を維持したまま長方形を変形する。

        4頂点は反時計回り (BL, BR, TR, TL) を想定。
        移動した頂点の対角頂点は固定し、隣接2頂点を直角が維持されるよう再計算する。

        Args:
            verts: 4頂点のリスト [(x,y,z), ...]
            vi: 移動する頂点のインデックス (0-3)
            new_pos: 移動先のワールド座標 (x, y, z)

        Returns:
            更新された4頂点のリスト
        """
        verts = [list(v) for v in verts]
        n = len(verts)
        if n != 4:
            verts[vi] = list(new_pos)
            return verts

        # 対角頂点のインデックス
        opp = (vi + 2) % 4
        # 隣接頂点のインデックス
        prev_i = (vi - 1) % 4
        next_i = (vi + 1) % 4

        # 対角頂点は固定
        ox, oy = float(verts[opp][0]), float(verts[opp][1])
        oz = float(verts[opp][2]) if len(verts[opp]) > 2 else 0.0

        # 移動先
        nx, ny = float(new_pos[0]), float(new_pos[1])
        nz = float(new_pos[2]) if len(new_pos) > 2 else oz

        # 元の長方形の辺方向を取得 (prev → vi の方向)
        px, py = float(verts[prev_i][0]), float(verts[prev_i][1])
        vx, vy = float(verts[vi][0]), float(verts[vi][1])

        # 辺1: prev → vi の方向
        e1x = vx - px
        e1y = vy - py
        e1_len = math.sqrt(e1x * e1x + e1y * e1y)
        if e1_len < 1e-9:
            verts[vi] = list(new_pos)
            return verts

        # 単位ベクトル
        u1x = e1x / e1_len
        u1y = e1y / e1_len
        # 直交方向
        u2x = -u1y
        u2y = u1x

        # 対角頂点から移動先への差分
        dx = nx - ox
        dy = ny - oy

        # 辺方向への射影
        proj1 = dx * u1x + dy * u1y
        proj2 = dx * u2x + dy * u2y

        # 新しい4頂点を計算
        # opp は固定、vi は移動先、prev と next は直角を維持
        avg_z = (oz + nz) / 2.0
        verts[vi] = (nx, ny, nz)
        verts[opp] = (ox, oy, oz)
        # prev: opp から u2 方向に proj2 だけ移動
        verts[prev_i] = (ox + proj2 * u2x, oy + proj2 * u2y, avg_z)
        # next: opp から u1 方向に proj1 だけ移動
        verts[next_i] = (ox + proj1 * u1x, oy + proj1 * u1y, avg_z)

        return [tuple(v) for v in verts]

    @staticmethod
    def _rbbox_rotate(verts: list, vi: int, new_pos: tuple) -> list:
        """回転BBOX を中心を固定して回転する。

        ドラッグした頂点の移動方向から回転角を計算し、
        全4頂点を中心周りに回転する。長方形の形状（幅・高さ）は維持される。

        Args:
            verts: 4頂点のリスト [(x,y,z), ...]
            vi: ドラッグ中の頂点インデックス (0-3)
            new_pos: ドラッグ先のワールド座標 (x, y, z)

        Returns:
            回転後の4頂点のリスト
        """
        verts = [list(v) for v in verts]
        if len(verts) != 4:
            verts[vi] = list(new_pos)
            return [tuple(v) for v in verts]

        # 中心を計算
        cx = sum(float(v[0]) for v in verts) / 4.0
        cy = sum(float(v[1]) for v in verts) / 4.0
        cz = sum(float(v[2]) if len(v) > 2 else 0.0 for v in verts) / 4.0

        # 元の頂点 vi から中心への角度
        old_x = float(verts[vi][0]) - cx
        old_y = float(verts[vi][1]) - cy
        old_angle = math.atan2(old_y, old_x)

        # 新しい位置から中心への角度
        new_x = float(new_pos[0]) - cx
        new_y = float(new_pos[1]) - cy
        new_angle = math.atan2(new_y, new_x)

        # 回転角
        d_angle = new_angle - old_angle

        cos_a = math.cos(d_angle)
        sin_a = math.sin(d_angle)

        # 全頂点を中心周りに回転
        result = []
        for v in verts:
            vx = float(v[0]) - cx
            vy = float(v[1]) - cy
            vz = float(v[2]) if len(v) > 2 else cz
            rx = vx * cos_a - vy * sin_a + cx
            ry = vx * sin_a + vy * cos_a + cy
            result.append((rx, ry, vz))

        return result

    @staticmethod
    def _update_rbbox_note(rec: dict, verts: list) -> None:
        """回転BBOX の note フィールドを現在の寸法と角度で更新する。"""
        if len(verts) != 4:
            return
        # 辺の長さを計算
        def _dist(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        w = _dist(verts[0], verts[1])
        h = _dist(verts[1], verts[2])
        long_side = max(w, h)
        short_side = min(w, h)
        # 角度: 最初の辺の方向
        dx = float(verts[1][0]) - float(verts[0][0])
        dy = float(verts[1][1]) - float(verts[0][1])
        angle = math.degrees(math.atan2(dy, dx))
        rec["note"] = (f"RBBOX {long_side:.2f}x{short_side:.2f}m "
                       f"angle={angle:.1f}°")

    def on_gl_double_click(self, screen_pt):
        """GL からダブルクリック時に呼ばれる: 最寄りのレーンを選択して編集モードに入る。

        ポリライン、ポリゴン、回転BBOX いずれも頂点編集モードに入る。
        """
        picked = self._pick_nearest_lane(screen_pt)
        if not picked:
            return False
        # _pick_nearest_lane がテーブル行を選択するので、
        # _on_tbl_select が自動的に呼ばれて編集モードに入る
        return True

    def on_vertex_drag_end(self):
        """GL から呼ばれる: ドラッグ終了"""
        self._edit_dragging = False
        self._edit_vert_idx = None
        self._refresh_table()

    def undo_vertex_edit(self):
        """Ctrl+Z: 直前の頂点編集操作（移動/追加/削除）を1回だけ元に戻す。"""
        if self._undo_state is None:
            return
        lane_idx, old_verts = self._undo_state
        self._undo_state = None
        if lane_idx >= len(self._lanes):
            return
        rec = self._lanes[lane_idx]
        rec["vertices"] = [tuple(v) for v in old_verts]
        self._rebuild_poly3d_cur(rec)
        # 現在の選択を更新
        if self._edit_lane_idx == lane_idx:
            self.gl.draw_lane_edit_verts = np.array(
                rec["vertices"], dtype=np.float32)
            self.gl.draw_lane_highlight = rec.get("poly3d")
        self.gl.draw_lane_data = self._lanes
        self.gl.update()
        self._refresh_table()

    def on_vertex_right_click(self, screen_pt, vert_idx_near):
        """GL から呼ばれる: 頂点付近で右クリック → 追加 or 削除。

        - 頂点の上で右クリック → その頂点を削除
        - ライン上の頂点以外の点で右クリック → 新たな頂点を追加
        - 回転BBOX (note に 'RBBOX') の場合は頂点の追加・削除を無効化
        """
        if self._edit_lane_idx is None:
            return

        rec   = self._lanes[self._edit_lane_idx]
        verts = list(rec["vertices"])

        # アンドゥ用スナップショット保存
        self._undo_state = (
            self._edit_lane_idx,
            [list(v) for v in verts]
        )

        # 回転BBOX は頂点の追加・削除を禁止（4頂点固定）
        is_rbbox = (len(verts) == 4
                    and "RBBOX" in rec.get("note", "")
                    and rec.get("geometry_type") == "polygon")
        if is_rbbox:
            return

        if vert_idx_near is not None:
            # ── 頂点の上で右クリック → 削除 ──────────────────────
            min_verts = 3 if rec.get("geometry_type") == "polygon" else 2
            if len(verts) > min_verts:
                verts.pop(vert_idx_near)
            else:
                return  # 最低頂点数を下回る場合は何もしない
        else:
            # ── ライン上の頂点以外の点で右クリック → 追加 ────────
            gz = self._ground_z()
            world = _screen_to_world_ray(self.gl, screen_pt, gz)
            if world is None:
                return
            # 最も近い辺を見つけて、その辺の後に挿入
            best_edge = len(verts) - 1  # デフォルト: 末尾
            best_dist = float('inf')
            n = len(verts)
            edges = n if rec.get("geometry_type") == "polygon" else n - 1
            for i in range(edges):
                j = (i + 1) % n
                ax, ay = verts[i][0], verts[i][1]
                bx, by = verts[j][0], verts[j][1]
                # 点から線分への距離
                dx, dy = bx - ax, by - ay
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq < 1e-12:
                    continue
                t = max(0.0, min(1.0,
                    ((world[0] - ax) * dx + (world[1] - ay) * dy) / seg_len_sq))
                px = ax + t * dx
                py = ay + t * dy
                d = math.sqrt((world[0] - px)**2 + (world[1] - py)**2)
                if d < best_dist:
                    best_dist = d
                    best_edge = i
            verts.insert(best_edge + 1, world)

        rec["vertices"] = verts
        self._rebuild_poly3d_cur(rec)
        self.gl.draw_lane_edit_verts = np.array(verts, dtype=np.float32)
        self.gl.draw_lane_highlight  = rec.get("poly3d")
        self.gl.draw_lane_data       = self._lanes
        self.gl.update()
        self._refresh_table()

    # ── Ctrl 矩形選択 ───────────────────────────────────────────
    def on_rect_select(self, p1, p2):
        """GL から呼ばれる: Ctrl+ドラッグ解放時。矩形内の Lane を複数選択。

        矩形の中心点から各ラインへのスクリーン座標上の最短距離が最小の
        ライン1本を選択する（密集したラインでも正確に意図したラインを選べる）。
        """
        if not self._lanes:
            return
        x1, x2 = sorted([p1.x(), p2.x()])
        y1, y2 = sorted([p1.y(), p2.y()])
        if x2 - x1 < 3 and y2 - y1 < 3:
            return
        w = getattr(self.gl, '_viewport_w', self.gl.width())
        h = getattr(self.gl, '_viewport_h', self.gl.height())

        # 矩形中心のスクリーン座標
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # 各ラインの「矩形中心からの最小スクリーン距離」を計算
        # 矩形内外問わず全ラインを対象にして最近傍を選ぶ
        best_idx = None
        best_dist = float('inf')

        for i, rec in enumerate(self._lanes):
            poly = rec.get("poly3d")
            if poly is not None and len(poly) > 0:
                pts_to_check = poly
            else:
                verts = rec.get("vertices", [])
                pts_to_check = verts if verts else []

            if not len(pts_to_check):
                continue

            # 線分ベースの距離: 各頂点をスクリーン座標に変換して
            # 矩形中心への最短距離を計算
            prev_s = None
            for v in pts_to_check:
                sx, sy = self.gl._world_to_screen(
                    float(v[0]), float(v[1]),
                    float(v[2]) if len(v) > 2 else 0.0,
                    w, h)
                if sx is None:
                    prev_s = None
                    continue
                if prev_s is not None:
                    # 線分への最短距離
                    d = self._point_to_segment_dist(
                        cx, cy, prev_s[0], prev_s[1], sx, sy)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                prev_s = (sx, sy)

        if best_idx is None:
            return

        hit_rows = [best_idx]
        rec = self._lanes[best_idx]
        print(f"[RECT_SELECT] 選択: index={best_idx}, id={rec.get('id')}, "
              f"矩形中心からの距離={best_dist:.1f}px, "
              f"note={rec.get('note','')[:60]}")
        if not hit_rows:
            return
        # テーブルで選択を追加（既存選択を維持して逐次追加）
        self._tbl.blockSignals(True)   # _on_tbl_select の発火を抑制
        sel_model = self._tbl.selectionModel()
        # 既存の選択行を取得
        prev_rows = set(r.row() for r in sel_model.selectedRows())
        all_rows = prev_rows | set(hit_rows)
        col_count = self._tbl.columnCount()
        for r in hit_rows:
            top_left     = self._tbl.model().index(r, 0)
            bottom_right = self._tbl.model().index(r, col_count - 1)
            sel = QItemSelection(top_left, bottom_right)
            sel_model.select(sel, QItemSelectionModel.Select)
            # チェックボックス（列0）もチェック状態にする
            chk_item = self._tbl.item(r, 0)
            if chk_item is not None:
                chk_item.setCheckState(Qt.Checked)
        self._tbl.blockSignals(False)
        all_rows_sorted = sorted(all_rows)
        # 選択順を記録（既存順序を維持し、新規分を末尾に追加）
        for r in hit_rows:
            if r not in self._selection_order:
                self._selection_order.append(r)
        # 削除された行を除去
        self._selection_order = [r for r in self._selection_order if r in all_rows]
        print(f"[RECT_SELECT] prev={sorted(prev_rows)} new={sorted(hit_rows)} all={all_rows_sorted} order={self._selection_order}")
        self._edit_lane_idx = all_rows_sorted[0]
        # 全選択Laneのスプラインをハイライト
        self.gl.draw_lane_highlights = [
            self._lanes[r].get("poly3d") for r in all_rows_sorted
            if r < len(self._lanes) and self._lanes[r].get("poly3d") is not None
        ]
        self.gl.draw_lane_highlight = None
        # 最初の選択Laneの頂点を編集可能に
        rec0 = self._lanes[self._edit_lane_idx]
        self.gl.draw_lane_edit_verts = np.array(
            rec0["vertices"], dtype=np.float32)
        self.gl.draw_lane_edit_panel = self
        self._populate_attr_from_rec(rec0)
        self.gl.update()
        total = len(all_rows_sorted)
        self._lbl_hint.setText(
            t("hint_rect_selected").format(total))

    def delete_rect_selected(self):
        """Del キー: チェック済みの行をすべて削除する。
        チェックがなければテーブル選択行を削除する（従来動作）。
        """
        try:
            # チェック済み行があればそちらを優先して削除
            checked_rows = []
            for row in range(self._tbl.rowCount()):
                chk_item = self._tbl.item(row, 0)
                if chk_item and chk_item.checkState() == Qt.Checked:
                    checked_rows.append(row)

            if checked_rows:
                # _delete_checked_rows は _refresh_table でテーブルを再構築するため
                # 後続の clearSelection は不要（新しいテーブルには選択がない）
                self._delete_checked_rows()
            else:
                # チェックがなければ従来通りテーブル選択行を削除
                self._delete_selected()
        except Exception as e:
            print(f"[delete_rect_selected] error: {e}")
        finally:
            self._edit_lane_idx = None
            self._selection_order = []
            self.gl.draw_lane_highlight  = None
            self.gl.draw_lane_highlights = []
            self.gl.draw_lane_edit_verts = None

    def _deselect_all(self):
        """すべてのポリラインの選択を解除する"""
        self._tbl.clearSelection()
        self._edit_lane_idx = None
        self._selection_order = []
        self.gl.draw_lane_highlight  = None
        self.gl.draw_lane_highlights = []
        self.gl.draw_lane_edit_verts = None
        # draw_lane_edit_panel は維持（次の選択時に必要）
        self._clear_attr_panel()
        self.gl.update()
        self._lbl_hint.setText(t("hint_select_tool"))

    def _pick_nearest_lane(self, screen_pt, threshold_px=20):
        """GL上のクリック位置に最も近いポリラインを選択する。

        すべてQt論理pixel座標で判定するため、zoom/pan/rot_z/HiDPIに依存しない。
        - Point:   中心から threshold_px 以内
        - Polyline: 表示線分から threshold_px 以内
        - Polygon:  内部クリック、または境界から threshold_px 以内
        """
        if not self._lanes:
            return False
        import numpy as _np

        # ★ Qt論理pixelで統一（_viewport_wは物理pixelなので使わない）
        w = self.gl.width() or 1
        h = self.gl.height() or 1
        sx = float(screen_pt.x())
        sy = float(screen_pt.y())

        best_dist = float(threshold_px)
        best_idx = None

        MAX_SEG_PX = 100.0
        MAX_PTS = 600

        def _interp_screen_pts(s0, s1):
            d = math.sqrt((s1[0]-s0[0])**2 + (s1[1]-s0[1])**2)
            if d <= MAX_SEG_PX:
                return [s0, s1]
            n = max(2, int(d / MAX_SEG_PX) + 1)
            return [(s0[0] + k/n*(s1[0]-s0[0]),
                     s0[1] + k/n*(s1[1]-s0[1])) for k in range(n+1)]

        def point_in_polygon(px, py, poly):
            """スクリーン多角形の内外判定（Ray Casting）。"""
            n = len(poly)
            if n < 3:
                return False
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > py) != (yj > py)):
                    x_cross = (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi
                    if px < x_cross:
                        inside = not inside
                j = i
            return inside

        for i, rec in enumerate(self._lanes):
            geom = rec.get("geometry_type", rec.get("shape", "polyline"))
            verts_raw = rec.get("vertices", [])
            poly3d_raw = rec.get("poly3d")

            # ★ スプライン表示中は poly3d を優先（表示と選択を一致させる）
            if (rec.get("_is_spline", False)
                    and poly3d_raw is not None
                    and len(poly3d_raw) >= 2):
                pts = poly3d_raw
            elif verts_raw is not None and len(verts_raw) >= 1:
                # ★ >= 1 に変更（Point = 1頂点を拾えるようにする）
                pts = verts_raw
            elif poly3d_raw is not None and len(poly3d_raw) >= 1:
                pts = poly3d_raw
            else:
                continue

            # ── world → screen（_world_to_screen が2D時にZ=0を自動適用）──
            if len(pts) > MAX_PTS:
                step = max(1, len(pts) // MAX_PTS)
                sampled = pts[::step]
                if len(pts) % step != 0:
                    sampled = _np.vstack([sampled, pts[-1:]])
            else:
                sampled = pts

            screen_pts = []
            for v in sampled:
                wz = float(v[2]) if len(v) > 2 else 0.0
                psx, psy = self.gl._world_to_screen(
                    float(v[0]), float(v[1]), wz, w, h)
                if psx is not None:
                    screen_pts.append((float(psx), float(psy)))

            if not screen_pts:
                continue

            # ── Point ──────────────────────────────────────────────
            if geom in ("point", "points") or len(screen_pts) == 1:
                px, py = screen_pts[0]
                d = math.hypot(sx - px, sy - py)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                continue

            # ── Polygon ────────────────────────────────────────────
            if geom == "polygon":
                # 内部クリックは即候補（距離0扱い）
                if point_in_polygon(sx, sy, screen_pts):
                    if 0.0 < best_dist:
                        best_dist = 0.0
                        best_idx = i
                    continue
                # ★ 最終辺（最終点→先頭点）も含めて全辺を判定
                n = len(screen_pts)
                for j in range(n):
                    a = screen_pts[j]
                    b = screen_pts[(j + 1) % n]
                    d = self._point_to_segment_dist(
                        sx, sy, a[0], a[1], b[0], b[1])
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                continue

            # ── Polyline ───────────────────────────────────────────
            prev_s = None
            for sp in screen_pts:
                if prev_s is not None:
                    seg_pts = _interp_screen_pts(prev_s, sp)
                    prev_sp = seg_pts[0]
                    for nsp in seg_pts[1:]:
                        d = self._point_to_segment_dist(
                            sx, sy,
                            prev_sp[0], prev_sp[1],
                            nsp[0], nsp[1])
                        if d < best_dist:
                            best_dist = d
                            best_idx = i
                        prev_sp = nsp
                prev_s = sp

        if best_idx is None:
            return False

        # GL上のクリック選択であることを _on_tbl_select() へ伝える。
        # selectRow() は selectionChanged を同期的に発火するため、
        # この間だけフラグを立てれば、テーブル同期は行いつつ
        # 2Dビューの表示中心移動だけを抑止できる。
        self._selection_from_gl_click = True
        try:
            self._tbl.selectRow(best_idx)
        finally:
            self._selection_from_gl_click = False

        return True

    @staticmethod
    def _point_to_segment_dist(px, py, ax, ay, bx, by):
        """点(px,py)から線分(ax,ay)-(bx,by)へのスクリーン距離を返す。"""
        dx = bx - ax
        dy = by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return math.sqrt((px - ax)**2 + (py - ay)**2)
        t = max(0.0, min(1.0,
            ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
