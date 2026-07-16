"""
generate_depth_maps.py
======================
LiDAR点群（PCD）をカメラに投影し、Dense Depthマップ（PNG）を生成する。

OD_data_requirements_summary.md 準拠:
  - フォーマット: PNG（グレースケール16bit unsigned）
  - 深度スケール: ピクセル値 ÷ 256 = 実際の深度 [m]
  - 最大深度: 250m
  - ファイル名: 画像と同名（拡張子のみ .png）

処理フロー:
  1. PCDファイルを読み込み（open3d または NumPy）
  2. LiDAR点群をカメラ座標系に変換（RT行列）
  3. カメラ前方（Z > 0）の点のみ抽出
  4. ピンホール投影で画像座標 (u, v) を計算
  5. 画像範囲内の点のみ残す
  6. スパースDepthマップを作成（投影点に深度値を格納）
  7. スパース → Dense 補間（IP-Basic法: dilate + bilateral filter）
  8. 16bit PNGとして保存

使い方:
  # 単体実行
  python generate_depth_maps.py <bev_result_dir> [--camera-param-dir DIR] [--cameras cam-FW]

  # run_from_csv.py から関数呼び出し
  from generate_depth_maps import generate_depth_maps_for_dataset
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# --- 日本語パス対応のcv2ラッパー ---

def cv2_imwrite(path, img, params=None):
    """日本語パス対応のcv2.imwrite代替"""
    ext = Path(path).suffix
    if params:
        _, buf = cv2.imencode(ext, img, params)
    else:
        _, buf = cv2.imencode(ext, img)
    buf.tofile(str(path))


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 定数 ---

# 深度スケール: pixel_value / DEPTH_SCALE = depth_meters
DEPTH_SCALE = 256.0

# 最大深度 [m]
MAX_DEPTH_M = 250.0

# 最大ピクセル値 (16bit)
MAX_PIXEL_VALUE = int(MAX_DEPTH_M * DEPTH_SCALE)  # 64000

# undis_single.py互換 歪みテーブル（cam-FW-undis投影用）
DISTORTION_TABLE_UNDIS = [
    [0, 0.0000], [0.089, 0.0890], [0.178, 0.1781], [0.267, 0.2673],
    [0.355, 0.3566], [0.444, 0.4462], [0.532, 0.5360], [0.62, 0.6262],
    [0.707, 0.7168], [0.795, 0.8078], [0.881, 0.8993], [0.967, 0.9913],
    [1.053, 1.0840], [1.137, 1.1774], [1.222, 1.2716], [1.305, 1.3665],
    [1.388, 1.4624], [1.47, 1.5592], [1.551, 1.6571], [1.631, 1.7561],
    [1.71, 1.8562], [1.789, 1.9577], [1.866, 2.0605], [1.943, 2.1648],
    [2.019, 2.2707], [2.094, 2.3782], [2.168, 2.4874], [2.241, 2.5986],
    [2.313, 2.7117], [2.384, 2.8270], [2.454, 2.9445], [2.523, 3.0644],
    [2.591, 3.1868], [2.658, 3.3120], [2.723, 3.4400], [2.788, 3.5711],
    [2.852, 3.7054], [2.914, 3.8431], [2.975, 3.9846], [3.036, 4.1299],
    [3.095, 4.2794], [3.153, 4.4334], [3.209, 4.5921], [3.265, 4.7558],
    [3.319, 4.9250], [3.373, 5.1000], [3.425, 5.2812], [3.476, 5.4691],
    [3.525, 5.6641], [3.574, 5.8669], [3.621, 6.0779], [3.668, 6.2980],
    [3.713, 6.5277], [3.757, 6.7679], [3.799, 7.0195], [3.841, 7.2836],
    [3.882, 7.5611], [3.921, 7.8533], [3.959, 8.1617], [3.996, 8.4878],
    [4.032, 8.8335], [4.066, 9.2006], [4.1, 9.5917], [4.132, 10.0093],
    [4.163, 10.4565], [4.193, 10.9370], [4.222, 11.4548], [4.25, 12.0148],
    [4.276, 12.6229], [4.302, 13.2860], [4.327, 14.0121], [4.35, 14.8115],
    [4.372, 15.6962], [4.394, 16.6813], [5.317, 17.7858], [5.323, 19.0335],
    [4.452, 20.4550], [4.47, 22.0905], [4.486, 23.9936], [4.502, 26.2372],
    [4.509, 27.5171],
]

# カメラ短縮名 → フルカメラ名
CAM_SHORT_TO_FULL = {
    'b': 'cam-B', 'bl': 'cam-BL', 'br': 'cam-BR',
    'f': 'cam-FW', 'fr': 'cam-FR', 'fl': 'cam-FL',
    'fn': 'cam-FN', 'fu': 'cam-FW-undis',
    'fish_b': 'Fisheye-cam-B', 'fish_l': 'Fisheye-cam-L',
    'fish_r': 'Fisheye-cam-R', 'fish_f': 'Fisheye-cam-F'
}

CAM_FULL_TO_SHORT = {v: k for k, v in CAM_SHORT_TO_FULL.items()}

# カメラ名リスト
IMG_DIR_LIST = [
    'cam-B', 'cam-BL', 'cam-FL', 'cam-FR', 'cam-FW', 'cam-BR',
    'cam-FN', 'cam-FW-undis',
    'Fisheye-cam-B', 'Fisheye-cam-F', 'Fisheye-cam-L', 'Fisheye-cam-R'
]

# スクリプトディレクトリ
SCRIPT_DIR = Path(__file__).parent.resolve()
CAMERA_PARAM_FOLDER = SCRIPT_DIR / 'json'


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description='LiDAR点群からDense Depthマップ(PNG)を生成する'
    )
    parser.add_argument('bev_result_dir',
                        help='BEV結果ディレクトリ（bev_output）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Depthマップ出力先（デフォルト: bev_result_dirと同階層）')
    parser.add_argument('--camera-param-dir', type=str, default=None,
                        help='カメラパラメータJSONフォルダ')
    parser.add_argument('--cameras', type=str, default=None,
                        help='対象カメラ（カンマ区切り）。省略時は全平面カメラ')
    parser.add_argument('--dense', action='store_true', default=True,
                        help='Dense補間を行う（デフォルト: True）')
    parser.add_argument('--no-dense', action='store_false', dest='dense',
                        help='Dense補間を行わない（スパースDepthのみ）')
    parser.add_argument('--bag-dirs', type=str, default=None,
                        help='bagディレクトリのカンマ区切りリスト（各bag_dir/result/local_pcdbin/を優先参照）')
    return parser.parse_args()


def load_camera_params(camera_param_dir: Path) -> Dict[str, dict]:
    """全カメラのパラメータを読み込む

    calib_tool.py と同じ投影方式:
      - JSONの fx, fy, cx, cy, RT, dist_coeff をそのまま使用
      - dist_coeff を Rational 8パラメータモデルで直接適用
      - calib_tool.py の project_points_astemo と完全に同じロジック
    """
    params = {}

    for cam_name in IMG_DIR_LIST:
        json_path = _get_json_path(camera_param_dir, cam_name)
        if json_path is None:
            continue
        with open(json_path, 'r', encoding='UTF-8') as f:
            para = json.load(f)
        cam_data = para['3d_img0']
        RT = np.array(cam_data['camera_external'],
                      dtype=np.float32).reshape(4, 4)
        dist_coeff = cam_data.get('dist_coeff', [0, 0, 0, 0, 0, 0, 0, 0])

        # camera_model判定（fisheye系カメラかどうか）
        camera_model = cam_data.get('camera_model', 'pinhole')
        if 'Fisheye' in cam_name or 'fisheye' in cam_name.lower():
            camera_model = "fisheye"

        entry = {
            'RT': RT,
            'fx': cam_data['camera_internal']['fx'],
            'fy': cam_data['camera_internal']['fy'],
            'cx': cam_data['camera_internal']['cx'],
            'cy': cam_data['camera_internal']['cy'],
            'width': cam_data['width'],
            'height': cam_data['height'],
            'dist_coeff': dist_coeff,
            'camera_model': camera_model,
            'undis_remap_func': None,
        }

        # ログ出力
        dist_all_zero = all(abs(d) < 1e-12 for d in dist_coeff)
        if dist_all_zero:
            logger.info(
                "%s: dist_coeff=0 → ピンホール直接投影 "
                "(fx=%.4f, fy=%.4f, cx=%.4f, cy=%.4f)",
                cam_name, entry['fx'], entry['fy'], entry['cx'], entry['cy']
            )
        else:
            logger.info(
                "%s: dist_coeff適用（calib_tool方式） "
                "(fx=%.4f, fy=%.4f, cx=%.4f, cy=%.4f)",
                cam_name, entry['fx'], entry['fy'], entry['cx'], entry['cy']
            )

        params[cam_name] = entry
    return params


def _get_json_path(json_dir: Path, cam_name: str) -> Optional[Path]:
    """カメラ名に対応するパラメータJSONファイルパスを取得する"""
    normal = json_dir / f"camera_param_{cam_name}.json"
    if normal.exists():
        return normal
    est = json_dir / f"est_camera_param_{cam_name}.json"
    if est.exists():
        return est
    for f in json_dir.iterdir():
        if f.suffix == '.json' and f.stem.endswith(cam_name):
            return f
    return None


def load_pcd_points(pcd_path: Path) -> Optional[np.ndarray]:
    """PCDファイルからXYZ点群を読み込む（高速版）

    読み込み優先順:
      1. NumPy直接パース（バイナリPCD: 最速、open3dのオーバーヘッド回避）
      2. NumPy直接パース（ASCII PCD）
      3. open3d（上記で失敗した場合のフォールバック）

    戻り値: (N, 3) のnumpy配列。読み込み失敗時はNone
    """
    # まず高速なNumPy直接パースを試行
    result = _parse_pcd_fast(pcd_path)
    if result is not None:
        return result

    # フォールバック: open3d
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points, dtype=np.float32)
        if points.shape[0] > 0:
            return points
    except ImportError:
        pass
    except Exception:
        pass

    return None


def _parse_pcd_fast(pcd_path: Path) -> Optional[np.ndarray]:
    """PCDファイルをNumPyで高速パースする（バイナリ・ASCII両対応）

    open3dの初期化オーバーヘッド（数十ms/回）を回避し、
    バイナリPCDではnp.frombufferで一括読み込みを行う。

    戻り値: (N, 3) float32配列。失敗時はNone
    """
    try:
        with open(pcd_path, 'rb') as f:
            # --- ヘッダー解析 ---
            header_lines = []
            header_size = 0
            while True:
                line = f.readline()
                if not line:
                    return None
                header_size += len(line)
                line_str = line.decode('ascii', errors='ignore').strip()
                header_lines.append(line_str)
                if line_str.startswith('DATA'):
                    break

            # ヘッダーからメタ情報を抽出
            n_points = 0
            data_format = 'ascii'
            fields = []
            field_sizes = []
            field_types = []
            field_counts = []
            for hl in header_lines:
                parts = hl.split()
                if not parts:
                    continue
                key = parts[0].upper()
                if key == 'POINTS':
                    n_points = int(parts[-1])
                elif key == 'DATA':
                    data_format = parts[-1].lower()
                elif key == 'FIELDS':
                    fields = [p.lower() for p in parts[1:]]
                elif key == 'SIZE':
                    field_sizes = [int(p) for p in parts[1:]]
                elif key == 'TYPE':
                    field_types = [p for p in parts[1:]]
                elif key == 'COUNT':
                    field_counts = [int(p) for p in parts[1:]]

            if n_points == 0 or not fields:
                return None

            # X, Y, Z のフィールドインデックスを特定
            try:
                x_idx = fields.index('x')
                y_idx = fields.index('y')
                z_idx = fields.index('z')
            except ValueError:
                x_idx, y_idx, z_idx = 0, 1, 2

            # --- バイナリ形式: NumPy一括読み込み（最速） ---
            if data_format in ('binary', 'binary_compressed'):
                if data_format == 'binary_compressed':
                    # binary_compressedはopen3dにフォールバック
                    return None

                if not field_sizes or not field_types:
                    return None

                # countが未指定の場合はすべて1
                if not field_counts:
                    field_counts = [1] * len(fields)

                # NumPy dtypeを構築
                dtype_list = []
                for i, (fname, fsize, ftype, fcount) in enumerate(
                    zip(fields, field_sizes, field_types, field_counts)
                ):
                    np_type = _pcd_type_to_numpy(ftype, fsize)
                    if np_type is None:
                        return None  # 未対応の型
                    if fcount == 1:
                        dtype_list.append((fname, np_type))
                    else:
                        dtype_list.append((fname, np_type, (fcount,)))

                record_dtype = np.dtype(dtype_list)

                # データ部を一括読み込み
                raw_data = f.read(n_points * record_dtype.itemsize)
                if len(raw_data) < n_points * record_dtype.itemsize:
                    # データ不足（ファイル破損等）
                    return None

                structured = np.frombuffer(raw_data, dtype=record_dtype, count=n_points)

                # XYZ抽出してfloat32配列に変換
                x_data = structured[fields[x_idx]].astype(np.float32)
                y_data = structured[fields[y_idx]].astype(np.float32)
                z_data = structured[fields[z_idx]].astype(np.float32)

                points = np.column_stack([x_data, y_data, z_data])
                return points if points.shape[0] > 0 else None

            # --- ASCII形式: NumPy一括読み込み ---
            elif data_format == 'ascii':
                # 残りのデータ部を一括読み込み
                remaining = f.read()
                if not remaining:
                    return None

                # numpy.fromstringの代わりにnp.loadtxtよりも高速な方法
                # 全行をデコードしてnp.fromstringで一括変換
                text = remaining.decode('ascii', errors='ignore')
                # 全フィールド数を計算
                n_fields = len(fields)
                try:
                    all_values = np.fromstring(
                        text, dtype=np.float32, sep=' '
                    )
                    # reshapeできるか確認
                    if all_values.size >= n_points * n_fields:
                        all_values = all_values[:n_points * n_fields].reshape(n_points, n_fields)
                        points = np.column_stack([
                            all_values[:, x_idx],
                            all_values[:, y_idx],
                            all_values[:, z_idx]
                        ])
                        return points if points.shape[0] > 0 else None
                except (ValueError, OverflowError):
                    pass

                # フォールバック: 従来の1行ずつパース
                return _parse_pcd_ascii_legacy(text, n_points, x_idx, y_idx, z_idx)

            else:
                return None

    except Exception as e:
        logger.debug(f"PCD高速パース失敗（フォールバック使用）: {pcd_path} - {e}")
        return None


def _pcd_type_to_numpy(pcd_type: str, size: int):
    """PCDのTYPE/SIZE指定をNumPy dtypeに変換する"""
    type_map = {
        ('F', 4): np.float32,
        ('F', 8): np.float64,
        ('U', 1): np.uint8,
        ('U', 2): np.uint16,
        ('U', 4): np.uint32,
        ('I', 1): np.int8,
        ('I', 2): np.int16,
        ('I', 4): np.int32,
    }
    return type_map.get((pcd_type.upper(), size), None)


def _parse_pcd_ascii_legacy(text: str, n_points: int,
                            x_idx: int, y_idx: int, z_idx: int) -> Optional[np.ndarray]:
    """ASCII PCDの従来パース（フォールバック用）"""
    points = []
    for line in text.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) > max(x_idx, y_idx, z_idx):
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
                points.append([x, y, z])
            except ValueError:
                continue
        if len(points) >= n_points:
            break
    return np.array(points, dtype=np.float32) if points else None




def project_lidar_to_depth(
    points: np.ndarray,
    RT: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    img_w: int, img_h: int,
    dist_coeff: list = None,
    src_camera_matrix: np.ndarray = None,
    src_dist_coeff: np.ndarray = None,
    undis_remap_func=None,
    camera_model: str = "pinhole"
) -> np.ndarray:
    """LiDAR点群をカメラに投影し、スパースDepthマップを生成する

    calib_tool.py の project_points_astemo と同じ投影ロジックを使用:
      - RT行列からrvec/tvecを抽出
      - dist_coeff を直接適用（Rational 8パラメータモデル）
      - undis画像に対しては dist_coeff=[0,...] なので単純ピンホール投影になる

    引数:
        points: (N, 3) LiDAR座標系の点群
        RT: (4, 4) LiDAR→カメラ変換行列
        fx, fy, cx, cy: カメラ内部パラメータ
        img_w, img_h: 画像サイズ
        dist_coeff: 歪み係数 [k1, k2, p1, p2, k3, k4, k5, k6]（最大8要素）
        src_camera_matrix: 未使用（互換性のため残す）
        src_dist_coeff: 未使用（互換性のため残す）
        undis_remap_func: 未使用（互換性のため残す、calib_tool方式では不要）
        camera_model: "pinhole" or "fisheye"

    戻り値:
        (img_h, img_w) のfloat32配列。値は深度 [m]。投影なし箇所は0
    """
    n_pts = points.shape[0]

    # NaN/Inf点を除去
    valid_pts_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
    points_clean = points[valid_pts_mask]
    n_pts = points_clean.shape[0]

    if n_pts == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    # RT行列からrvec/tvecを抽出
    R = RT[:3, :3]
    tvec = RT[:3, 3]

    # LiDAR → カメラ座標系（float32で計算: 精度十分かつ高速）
    pts_3d = points_clean.astype(np.float32)
    R_f32 = R.astype(np.float32)
    tvec_f32 = tvec.astype(np.float32).reshape(1, 3)
    pts_cam = (R_f32 @ pts_3d.T).T + tvec_f32  # (N, 3)

    # 事前フィルタ: カメラ前方(z>0)の点のみに絞り込み（大幅な計算量削減）
    z_all = pts_cam[:, 2]
    front_mask = z_all > 0.01
    if not np.any(front_mask):
        return np.zeros((img_h, img_w), dtype=np.float32)
    pts_cam = pts_cam[front_mask]

    # dist_coeff準備
    if dist_coeff is None:
        dist_coeff = [0.0] * 8
    k = list(dist_coeff) + [0.0] * (8 - len(dist_coeff))
    k1, k2, p1, p2, k3, k4, k5, k6 = k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]

    if camera_model == "fisheye":
        # === Equidistant (fisheye) モデル ===
        z = pts_cam[:, 2]
        norms = np.linalg.norm(pts_cam, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_theta_fov = np.where(norms > 0, z / norms, -1.0)
        theta_fov = np.arccos(np.clip(cos_theta_fov, -1, 1))
        valid = (z >= 0) & (theta_fov <= np.deg2rad(190) / 2)

        if not np.any(valid):
            return np.zeros((img_h, img_w), dtype=np.float32)

        z_valid = pts_cam[valid, 2]
        x_norm = pts_cam[valid, 0] / z_valid
        y_norm = pts_cam[valid, 1] / z_valid

        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan(r)
        theta = np.clip(theta, -5.6713, 5.6713)

        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        # fisheye: [k1, k2, k3, k4]
        fk1, fk2, fk3, fk4 = k[0], k[1], k[2], k[3]
        theta_d = theta * (1 + fk1*theta2 + fk2*theta4 + fk3*theta6 + fk4*theta8)

        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(r > 1e-10, theta_d / r, 1.0)
        x_d = x_norm * scale
        y_d = y_norm * scale

        u = (fx * x_d + cx)
        v = (fy * y_d + cy)

    else:
        # === Rational (pinhole) モデル === calib_tool.py と同一ロジック
        # 事前フィルタで z > 0.01 の点のみ残っているためvalidフィルタ不要
        z_valid = pts_cam[:, 2]

        # Z正規化
        x = pts_cam[:, 0] / z_valid
        y = pts_cam[:, 1] / z_valid

        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3

        # Rational model 歪み適用
        radial_num = 1 + k1*r2 + k2*r4 + k3*r6
        radial_den = 1 + k4*r2 + k5*r4 + k6*r6
        radial_den = np.where(np.abs(radial_den) < 1e-10, 1.0, radial_den)
        radial = radial_num / radial_den

        # 接線歪み
        x_d = x * radial + 2*p1*x*y + p2*(r2 + 2*x**2)
        y_d = y * radial + p1*(r2 + 2*y**2) + 2*p2*x*y

        # 内部パラメータ適用
        u = fx * x_d + cx
        v = fy * y_d + cy

    # 整数化
    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    # 画像範囲内フィルタ
    in_bounds = (u_int >= 0) & (u_int < img_w) & (v_int >= 0) & (v_int < img_h)
    u_final = u_int[in_bounds]
    v_final = v_int[in_bounds]
    depth = np.clip(z_valid[in_bounds], 0.0, MAX_DEPTH_M).astype(np.float32)

    # スパースDepthマップ作成（深い順→浅い方が上書き=最小深度が残る）
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)

    sort_idx = np.argsort(-depth)
    u_sorted = u_final[sort_idx]
    v_sorted = v_final[sort_idx]
    d_sorted = depth[sort_idx]

    depth_map[v_sorted, u_sorted] = d_sorted

    return depth_map


def _build_opencv_distort_undistort_remap(
    fx: float, fy: float, cx: float, cy: float,
    dist_coeff: list,
):
    """cam-FW JSONのdist_coeffを使ったOpenCV方式の座標変換関数を構築する。

    処理の流れ:
      入力: cam-FWのfx/fy/cx/cyでのピンホール投影結果 (u, v) — 歪みなし理想座標
      1. 正規化座標に変換: x_norm = (u - cx) / fx, y_norm = (v - cy) / fy
      2. OpenCVの歪みモデルで歪みを付与（distort）→ 歪み付き正規化座標
      3. cam-FWのカメラ行列でピクセル化 → 歪み付きピクセル座標（= 元画像上の位置）
      4. cv2.undistortPointsで歪みを除去（new_camera_matrix=camera_matrix）
         → undis画像上のピクセル座標

    注: 手順2-3-4は「理想座標→歪み付き→歪み除去→理想座標」で恒等変換のように見えるが、
        undis画像の生成過程（initUndistortRectifyMap + remap）と完全に一致させるため
        この手順が必要。実際、OpenCVのundistortは完全な逆変換ではなく反復法を使うため
        若干の差が出る可能性があるが、実用上は十分。

    実際には簡略化できる:
      ピンホール投影(歪みなし) = undistort後の座標 = undis画像上の座標
      (new_camera_matrix = camera_matrix のため)
    
    ただし、undis画像がLUTテーブル方式で生成されている場合はこの等式は成立しない。
    LUTテーブル方式のundis画像に対してOpenCV方式で投影するということは、
    OpenCV歪みモデルに基づく正しい「歪みなし座標」を得るということ。
    
    結果: cam-FWのfx/fy/cx/cyでのピンホール投影結果がそのままundis座標になる。
    （歪みなしピンホール = OpenCVのundistort結果 = undis画像座標、
      new_camera_matrixがcamera_matrixと同じ場合）
    
    しかし！undis画像がLUT方式で生成されている場合、OpenCVのundistortとは
    座標系が異なるため、単純なピンホール投影ではundis画像と合わない。
    
    ここでのアプローチ:
      JSONのdist_coeffを使って「正しいundis座標」を計算する。
      具体的には:
      1. ピンホール投影で得た理想座標(u,v)を正規化
      2. 歪みモデルで歪ませて、元画像上の歪み付き座標を得る
      3. その歪み付き座標を、undis画像のremapテーブル（LUT方式）で
         undis座標に変換する... → これではLUTが必要になる
      
    別アプローチ（JSONだけで完結）:
      undis画像が「どのような座標系か」をJSONのパラメータだけで表現する。
      
      cam-FW-undis.jsonでは dist_coeff = [0,0,...,0] かつ fx=1903.3941, cx=1917.3
      → これはundis画像の座標系を直接定義している。
      
      つまり: LiDAR → カメラ3D → cam-FW-undis.json の fx/cx/cy でピンホール投影
      = undis画像上の座標（cam-FW-undis.jsonが正確にundis画像の座標系を表す場合）
      
    しかし先の検証で cam-FW-undis.json のパラメータではずれが残った。
    これはcam-FW-undis.jsonの値がundis画像の実態と正確に一致していないことを示す。
    
    最終的に正しいアプローチ:
      cam-FWのdist_coeffを使って3D点を歪み付き座標に投影し、
      その歪み付き座標からundis画像上の座標を逆算する。
      undis画像の生成: src[map_y, map_x] → dst[i, j]
      つまりundisピクセル(j,i)が参照する元画像座標が(map_x[i,j], map_y[i,j])
      逆に、元画像座標(dx,dy)に対応するundisピクセルを求めるのは
      逆マッピング（map_x, map_yの逆関数）が必要。
      
    OpenCVではcv2.undistortPointsがこの逆マッピングに相当する:
      歪み付きピクセル座標 → 歪みなし正規化座標 → new_camera_matrixでピクセル化
      
    手順:
      1. cam-FWのfx/fy/cx/cyで3D点をピンホール投影 → 歪みなし理想座標
      2. 歪みモデルで歪ませる → 歪み付きピクセル座標（元画像上）
      3. cv2.undistortPoints(歪み付き座標, camera_matrix, dist_coeff, P=camera_matrix)
         → undis画像上の座標
    """
    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_arr = np.array(dist_coeff, dtype=np.float64)

    def remap(u_arr, v_arr):
        """ピンホール投影座標（歪みなし理想座標） → undis画像座標

        手順:
          1. 理想座標を正規化
          2. OpenCV歪みモデルで歪み付き座標を計算
          3. cv2.undistortPointsでundis座標を取得
        """
        n = len(u_arr)
        if n == 0:
            return u_arr, v_arr

        # 1. 理想ピクセル座標 → 正規化座標
        x_norm = (u_arr - cx) / fx
        y_norm = (v_arr - cy) / fy

        # 2. OpenCV歪みモデル適用（distort: 正規化座標に歪みを付与）
        # OpenCV rational model: 
        #   dist_coeff = [k1, k2, p1, p2, k3, k4, k5, k6]
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r4 * r2

        # dist_coeffの展開
        k1 = dist_arr[0] if len(dist_arr) > 0 else 0.0
        k2 = dist_arr[1] if len(dist_arr) > 1 else 0.0
        p1 = dist_arr[2] if len(dist_arr) > 2 else 0.0
        p2 = dist_arr[3] if len(dist_arr) > 3 else 0.0
        k3 = dist_arr[4] if len(dist_arr) > 4 else 0.0
        k4 = dist_arr[5] if len(dist_arr) > 5 else 0.0
        k5 = dist_arr[6] if len(dist_arr) > 6 else 0.0
        k6 = dist_arr[7] if len(dist_arr) > 7 else 0.0

        # Radial distortion (rational model)
        numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        # ゼロ除算防止
        denominator = np.where(np.abs(denominator) < 1e-12, 1.0, denominator)
        radial = numerator / denominator

        # Tangential distortion
        x_dist = x_norm * radial + 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2.0 * x_norm**2)
        y_dist = y_norm * radial + p1 * (r2 + 2.0 * y_norm**2) + 2.0 * p2 * x_norm * y_norm

        # 3. 歪み付き正規化座標 → 歪み付きピクセル座標
        u_dist = fx * x_dist + cx
        v_dist = fy * y_dist + cy

        # 4. cv2.undistortPoints で歪みを除去 → undis座標
        # 入力: (N, 1, 2) float32
        pts_dist = np.stack([u_dist, v_dist], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        pts_undis = cv2.undistortPoints(
            pts_dist, camera_matrix, dist_arr, P=camera_matrix
        )
        # 出力: (N, 1, 2)
        pts_undis = pts_undis.reshape(-1, 2)

        return pts_undis[:, 0].astype(np.float64), pts_undis[:, 1].astype(np.float64)

    return remap


def _build_calib_gui_remap(
    fx_proj: float, fy_proj: float,
    fx_undis: float, fy_undis: float,
    cx: float, cy: float,
    dist_coeff: list
):
    """calib_adjuster_gui.py と完全に同じ方式の座標変換関数を構築する。

    calib_adjuster_gui.pyの投影フロー:
      1. RT @ pts → カメラ座標 (x, y, z)
      2. 正規化座標: x_norm = x/z, y_norm = y/z
      3. cam-FW の dist_coeff で歪みを付与
      4. fx_proj (= cam-FW * f_scale ≈ 1941) でピクセル化 → u_dist, v_dist
      5. initUndistortRectifyMap(K_undis, dist, None, K_undis) で生成したmapでremap
         → undis画像座標

    ここでは手順4以降を再現する:
      入力: project_lidar_to_depth が fx_proj/fy_proj でピンホール投影した (u, v)
      処理:
        a. 正規化座標に戻す: x_norm = (u - cx) / fx_proj, y_norm = (v - cy) / fy_proj
        b. dist_coeffで歪み付与
        c. fx_proj でピクセル化 → u_dist, v_dist（= 歪み付きピクセル座標、GUIと同じ）
        d. cv2.undistortPoints(pts, K_undis, dist, P=K_undis) → undis座標
           （K_undis = cam-FWのfx/fy/cx/cy、initUndistortRectifyMapのnew_K相当）

    注: 手順d では K_undis(=cam-FW.json のfx) を使う。
        cv2.undistortPoints は入力をK_undisで正規化し、歪み除去し、P(=K_undis)でピクセル化する。
        しかし入力ピクセル座標は fx_proj でピクセル化されているため、
        K_undis で正規化すると x_norm' = u_dist/fx_undis ≠ 正しい正規化座標。
        この「ずれ」が calib_adjuster_gui.py の initUndistortRectifyMap(K,dist,None,K) +
        remapの動作と等価になる。

    実際の等価変換:
      initUndistortRectifyMap(K, dist, None, K) で生成されるmap:
        undis_pixel(i,j) → 歪み付きピクセル座標(map_x[i,j], map_y[i,j])
      remap: depth_distorted[map_y, map_x] → depth_undistorted[i, j]

      つまり: 歪み付きピクセル座標(u_dist, v_dist)に対して、
      undis画像上のどの(i,j)がそこを参照しているか？= 逆マッピング

      cv2.undistortPoints(pts, K, dist, P=K):
        歪み付きピクセル → Kで正規化 → 歪み除去(反復法) → Pでピクセル化
        = 歪み付きピクセル → undis画像ピクセル
      これはまさにremapの逆（undisピクセルを求める処理）。
    """
    # undistort用カメラ行列（cam-FWのfx/fy/cx/cy = undis画像の座標系基準）
    K_undis = np.array([
        [fx_undis, 0.0, cx],
        [0.0, fy_undis, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_arr = np.array(dist_coeff, dtype=np.float64)

    def remap(u_arr, v_arr):
        """fx_projでピンホール投影した座標(u,v) → undis画像座標

        calib_adjuster_gui.pyの処理を再現:
          1. 正規化（fx_proj基準）
          2. dist_coeff で歪み付与
          3. fx_proj でピクセル化
          4. cv2.undistortPoints(K_undis, dist, P=K_undis) でundis座標
        """
        n = len(u_arr)
        if n == 0:
            return u_arr, v_arr

        # 1. ピンホール投影座標 → 正規化座標（fx_proj基準）
        x_norm = (u_arr - cx) / fx_proj
        y_norm = (v_arr - cy) / fy_proj

        # 2. OpenCV歪みモデル適用（distort: 正規化座標に歪みを付与）
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r4 * r2

        k1 = dist_arr[0] if len(dist_arr) > 0 else 0.0
        k2 = dist_arr[1] if len(dist_arr) > 1 else 0.0
        p1 = dist_arr[2] if len(dist_arr) > 2 else 0.0
        p2 = dist_arr[3] if len(dist_arr) > 3 else 0.0
        k3 = dist_arr[4] if len(dist_arr) > 4 else 0.0
        k4 = dist_arr[5] if len(dist_arr) > 5 else 0.0
        k5 = dist_arr[6] if len(dist_arr) > 6 else 0.0
        k6 = dist_arr[7] if len(dist_arr) > 7 else 0.0

        numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        denominator = np.where(np.abs(denominator) < 1e-12, 1.0, denominator)
        radial = numerator / denominator

        x_dist = x_norm * radial + 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2.0 * x_norm**2)
        y_dist = y_norm * radial + p1 * (r2 + 2.0 * y_norm**2) + 2.0 * p2 * x_norm * y_norm

        # 3. fx_proj でピクセル化（calib_adjuster_guiの fx_adj * x_d + cx と同じ）
        u_distorted = fx_proj * x_dist + cx
        v_distorted = fy_proj * y_dist + cy

        # 4. cv2.undistortPoints(K_undis, dist, P=K_undis) → undis座標
        pts_dist = np.stack([u_distorted, v_distorted], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        pts_undis = cv2.undistortPoints(
            pts_dist, K_undis, dist_arr, P=K_undis
        )
        pts_undis = pts_undis.reshape(-1, 2)

        return pts_undis[:, 0].astype(np.float64), pts_undis[:, 1].astype(np.float64)

    return remap


def _build_undis_remap_func(
    cam_cx: float, cam_cy: float,
    mm_pixel_x: float, mm_pixel_y: float,
    out_scale: float,
    distortion_table: list
):
    """undis_single.py互換の座標変換関数を構築する。

    ピンホール投影で得た画像座標(u,v)を、undistort画像上の正しい座標に変換する。

    undis_single.pyのcompute_mapの逆操作:
      compute_map: undisピクセル(j,i) → 物理距離 → FindRealValue → 元ピクセル(dx,dy)
      ここで必要なのは: 元ピクセル(u,v) → undisピクセル(j,i)

    変換式:
      1. 元ピクセル(u,v) → 物理距離[mm]: fx_mm = (u - cx) * mm_pixel_x
      2. r_mm = sqrt(fx_mm² + fy_mm²)
      3. テーブル順方向: r_mm → r_undis_mm (DISTORTION_TABLEの[0]列→[1]列の補間)
      4. undisピクセル: j = r_undis_mm * fx_mm / (r_mm * mm_pixel_x * out_scale) + cx
    """
    table = np.array(distortion_table, dtype=np.float64)  # shape (N, 2)
    # table[:, 0] = 実角度(undistorted側) [mm]
    # table[:, 1] = 像高(distorted側) [mm]
    # FindRealValue: 像高 → 実角度 (distorted → undistorted)
    # 今必要な逆: 実角度 → 像高ではなく...

    # compute_mapの動作を整理:
    # undisピクセル(j,i) の物理距離 length_mm を計算
    # FindRealValue(length_mm): table[:,1]から探してtable[:,0]を返す
    #   つまり: distorted像高[mm] → undistorted実角度[mm]
    # dst_len(= undistorted実角度[mm])を使って元画像ピクセルを計算:
    #   dx = dst_len * fx_mm / (length_mm * mm_pixel_x) + cx

    # 逆変換（元画像ピクセル → undisピクセル）:
    # 元画像ピクセル(u,v)の物理距離: r_src_mm = sqrt(((u-cx)*mm_pixel_x)^2 + ((v-cy)*mm_pixel_y)^2)
    # この r_src_mm は「undistorted実角度[mm]」に相当する（ピンホール投影=歪みなし）
    # テーブルの逆引き: undistorted実角度[mm] → distorted像高[mm]
    #   table[:,0] → table[:,1] の補間
    # undisピクセルの物理距離: r_undis_mm = distorted像高[mm]（compute_mapでは"length"に相当）
    # undisピクセル: j = r_undis_mm / (mm_pixel_x * out_scale) * (fx_mm / r_src_mm) ???

    # 再整理: compute_mapの式
    #   length = sqrt(((j-cx)*dst_mmp_x)^2 + ((i-cy)*dst_mmp_y)^2)  ← undisピクセルの物理距離
    #   dst_len = FindRealValue(length)  ← テーブル: 像高→実角度
    #   dx = dst_len * fx / (length * mm_pixel_x) + cx  ← 元画像ピクセル
    #
    # つまり: dx - cx = dst_len/mm_pixel_x * (fx/length)
    #         (dx - cx) * mm_pixel_x = dst_len * (fx/length)
    #         src_r_mm = dst_len * cos(theta)  ... いや、方向比を保持している
    #
    # 実際には方向(fx, fy)が保持されるので:
    #   dx = dst_len * fx_undis / (length * mm_pixel_x) + cx
    #   ここで fx_undis = (j - cx) * dst_mmp_x
    #   → dx - cx = dst_len * (j - cx) * dst_mmp_x / (length * mm_pixel_x)
    #   → (dx - cx) * mm_pixel_x = dst_len * (j - cx) * dst_mmp_x / length
    #
    # 逆に (dx - cx) から (j - cx) を求める:
    #   r_src = (dx - cx) * mm_pixel_x  (方向成分、ここでは1D簡略化)
    #   length = (j - cx) * dst_mmp_x
    #   r_src = dst_len * length / length ... ← ??? 方向比があるので
    #
    # 1Dで考える（中心からの距離だけ）:
    #   r_src_px = sqrt((u-cx)^2 + (v-cy)^2) [px in src image]
    #   r_src_mm = r_src_px * mm_pixel_x [mm] （x,y同一ピクセルピッチと仮定）
    #   これが「undistorted実角度[mm]」= table[:,0]の値に対応
    #
    #   テーブル逆引き: table[:,0]列から r_src_mm を探し、table[:,1]列の値を得る
    #   → r_dist_mm (distorted像高[mm])
    #
    #   undisピクセルの距離: r_undis_px = r_dist_mm / dst_mmp_x [px]
    #   undisピクセル: (j, i) = (cx + r_undis_px * (u-cx)/r_src_px, cy + r_undis_px * (v-cy)/r_src_px)

    dst_mmp_x = mm_pixel_x * out_scale

    def remap(u_arr, v_arr):
        """元画像ピクセル座標（ピンホール投影結果） → undis画像ピクセル座標

        ピンホール投影結果は「実角度」に対応する座標。
        テーブル table[:,0](実角度) → table[:,1](像高) で変換し、
        undis画像のピクセルスケールで割る。
        """
        dx_px = u_arr - cam_cx
        dy_px = v_arr - cam_cy
        # ピンホール投影結果のピクセル距離
        r_src_px = np.sqrt(dx_px**2 + dy_px**2)
        # 物理距離[mm]（実角度に対応）
        r_src_mm = r_src_px * mm_pixel_x

        # テーブル補間: 実角度[mm] → 像高[mm]
        # table[:,0] = 実角度, table[:,1] = 像高
        r_dist_mm = np.interp(r_src_mm, table[:, 0], table[:, 1])

        # undisピクセル距離: 像高[mm] / (mm_pixel_x * out_scale)
        r_undis_px = r_dist_mm / dst_mmp_x

        # 方向を保持してundis座標に変換
        scale = np.where(r_src_px > 1e-6, r_undis_px / r_src_px, 1.0)
        j_new = cam_cx + dx_px * scale
        i_new = cam_cy + dy_px * scale

        return j_new, i_new

    return remap


def densify_depth_map(sparse_depth: np.ndarray) -> np.ndarray:
    """スパースDepthマップをDense化する（改善版）

    処理:
      1. 近傍dilateで穴埋め（小さいカーネルで段階的に）
      2. Bilateral filterで深度の不連続性を保持しつつ平滑化
      3. 元の投影点を復元

    引数:
        sparse_depth: (H, W) float32のスパースDepthマップ（0=未投影）

    戻り値:
        (H, W) float32のDense Depthマップ
    """
    depth = sparse_depth.copy()
    mask_original = depth > 0

    # ステップ1: 小さいカーネルで段階的dilate（穴埋め）
    # 大きいカーネルは使わない（遠方の点が近傍に広がりすぎるのを防止）
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 小→中の順にdilateし、既に値がある箇所は保持
    d1 = cv2.dilate(depth, kernel3, iterations=1)
    mask1 = depth > 0
    d1[mask1] = depth[mask1]

    d2 = cv2.dilate(d1, kernel5, iterations=1)
    mask2 = d1 > 0
    d2[mask2] = d1[mask2]

    d3 = cv2.dilate(d2, kernel7, iterations=1)
    mask3 = d2 > 0
    d3[mask3] = d2[mask3]

    # ステップ2: Median filter (5x5) でノイズ除去
    result = cv2.medianBlur(d3, 5)

    # 元の投影点を復元
    result[mask_original] = depth[mask_original]

    return result


def depth_to_uint16(depth_map: np.ndarray) -> np.ndarray:
    """深度マップ [m] を16bit unsigned intに変換する

    変換式: pixel_value = depth_m * DEPTH_SCALE
    読み戻し: depth_m = pixel_value / DEPTH_SCALE

    引数:
        depth_map: (H, W) float32 深度 [m]

    戻り値:
        (H, W) uint16
    """
    depth_scaled = depth_map * DEPTH_SCALE
    np.clip(depth_scaled, 0, 65535, out=depth_scaled)
    return depth_scaled.astype(np.uint16)


def generate_depth_for_one_frame(
    pcd_path: Path,
    camera_params: Dict[str, dict],
    target_cameras: Optional[List[str]] = None,
    dense: bool = True
) -> Dict[str, np.ndarray]:
    """1フレームのPCDファイルに対して各カメラのDepthマップを生成する

    引数:
        pcd_path: PCDファイルパス
        camera_params: カメラパラメータ辞書
        target_cameras: 対象カメラリスト
        dense: Dense補間を行うか

    戻り値:
        {カメラ短縮名: uint16 Depthマップ}
    """
    points = load_pcd_points(pcd_path)
    if points is None or points.shape[0] == 0:
        return {}

    result: Dict[str, np.ndarray] = {}

    for cam_full, cam_param in camera_params.items():
        if target_cameras and cam_full not in target_cameras:
            continue

        cam_short = CAM_FULL_TO_SHORT.get(cam_full)
        if cam_short is None:
            continue

        # 魚眼カメラは平面投影では不正確なのでスキップ
        if 'Fisheye' in cam_full or 'fish' in cam_short:
            continue

        RT = cam_param['RT']
        fx = cam_param['fx']
        fy = cam_param['fy']
        cx_val = cam_param['cx']
        cy_val = cam_param['cy']
        img_w = cam_param['width']
        img_h = cam_param['height']

        # スパースDepthマップ生成
        dist_coeff = cam_param.get('dist_coeff', None)
        undis_remap_func = cam_param.get('undis_remap_func', None)
        camera_model = cam_param.get('camera_model', 'pinhole')

        sparse_depth = project_lidar_to_depth(
            points, RT, fx, fy, cx_val, cy_val, img_w, img_h,
            dist_coeff, None, None, undis_remap_func, camera_model
        )

        # 投影点が極端に少ない場合はスキップ
        n_projected = np.count_nonzero(sparse_depth)
        if n_projected < 100:
            continue

        # Dense化
        if dense:
            depth_map = densify_depth_map(sparse_depth)
        else:
            depth_map = sparse_depth

        # uint16に変換
        depth_uint16 = depth_to_uint16(depth_map)
        result[cam_short] = depth_uint16

    return result


# --- ProcessPoolExecutor用のワーカー関数（モジュールレベル、pickle可能） ---

def _process_one_frame_worker(
    pcd_path: Path,
    camera_params: Dict[str, dict],
    target_cameras: List[str],
    dense: bool,
    target_pcd_frame_nums: set,
    cam_frame_map: Dict[str, Dict[int, str]],
    cam_frame_names: Dict[str, List[str]]
) -> List[Tuple[str, str, np.ndarray]]:
    """1フレームを処理し、(cam_short, frame_name, depth_img)のリストを返す

    ProcessPoolExecutor から呼ばれるため、モジュールレベルに配置。
    クロージャを使わず、全データを引数で受け取る。
    """
    try:
        pcd_frame_num = int(pcd_path.stem)
    except ValueError:
        return []

    # 対象フレームでなければスキップ（高速フィルタ）
    if target_pcd_frame_nums and pcd_frame_num not in target_pcd_frame_nums:
        return []

    # Depthマップ生成
    depth_maps = generate_depth_for_one_frame(
        pcd_path, camera_params, target_cameras, dense
    )

    results = []
    for cam_short, depth_img in depth_maps.items():
        # フレーム名の決定
        frame_name = None
        if cam_short in cam_frame_map:
            annotation_frame_num = pcd_frame_num + 1
            frame_name = cam_frame_map[cam_short].get(annotation_frame_num)
            if frame_name is None:
                continue
        elif cam_short in cam_frame_names:
            continue
        else:
            continue
        results.append((cam_short, frame_name, depth_img))
    return results


def generate_depth_maps_for_dataset(
    bag_dirs: List[Path],
    od_output_dir: Path,
    camera_param_folder: Path,
    target_cameras: Optional[List[str]] = None,
    dense: bool = True
) -> Dict[str, int]:
    """ODデータセット用のDepth_mapsを生成する（run_from_csv.pyから呼び出す公開関数）

    各bag_dirのresult/local_pcdbin/（なければlocal_motion_pcdbin/）配下のPCDファイルを
    カメラに投影してDepthマップを生成する。

    出力先は od_output_dir/data_3d_<cam_short>/Depth_maps/<frame_name>.png 形式。
    ファイル名はAnnotations/JPEGImagesと同じ名前（拡張子のみ.png）。

    OD_data_requirements_summary.md 準拠:
      - フォーマット: PNG（グレースケール16bit unsigned）
      - 深度スケール: ピクセル値 ÷ 256 = 実際の深度 [m]
      - 最大深度: 250m

    引数:
        bag_dirs: bagディレクトリのリスト（各bag_dir/result/local_pcdbin/ を優先参照）
        od_output_dir: ODデータセット出力先（bev_outputと同階層）
        camera_param_folder: カメラパラメータJSONフォルダ
        target_cameras: 対象カメラリスト
        dense: Dense補間を行うか

    戻り値:
        {カメラ短縮名: 生成ファイル数}
    """
    logger.info(f"Depth_maps生成開始（local_pcdbin優先）")

    # カメラパラメータ読み込み
    camera_params = load_camera_params(camera_param_folder)
    if not camera_params:
        logger.error("カメラパラメータの読み込みに失敗しました")
        return {}

    # デフォルト: cam-FW-undis と cam-FN のみ
    if target_cameras is None:
        target_cameras = ['cam-FW-undis', 'cam-FN']

    stats: Dict[str, int] = {}

    # 各カメラのdata_3d_<cam_short>/Annotations/ からフレーム名リストを取得
    # Depth_mapsのファイル名はAnnotationsと一致させる
    # {cam_short: {pcd_frame_num: annotation_stem}} のマッピングを構築
    cam_frame_names: Dict[str, List[str]] = {}
    cam_frame_map: Dict[str, Dict[int, str]] = {}  # {cam_short: {frame_num: stem}}
    for cam_full in target_cameras:
        cam_short = CAM_FULL_TO_SHORT.get(cam_full)
        if cam_short is None:
            continue
        if 'Fisheye' in cam_full or 'fish' in cam_short:
            continue
        ann_dir = od_output_dir / f"data_3d_{cam_short}" / 'Annotations'
        if ann_dir.exists():
            frame_names = sorted([p.stem for p in ann_dir.glob('*.xml')])
            if frame_names:
                cam_frame_names[cam_short] = frame_names
                # フレーム名からフレーム番号を抽出してマッピング構築
                # "NCAP_cam-FW-undis_000005" → 5
                cam_frame_map[cam_short] = {}
                for stem in frame_names:
                    parts = stem.split('_')
                    try:
                        fnum = int(parts[-1])
                        cam_frame_map[cam_short][fnum] = stem
                    except (ValueError, IndexError):
                        continue
                logger.info(f"  {cam_short}: {len(frame_names)} frames (from Annotations/)")

    if not cam_frame_names:
        logger.warning("Annotationsからフレーム名が取得できません。連番モードで生成します。")

    # 処理対象のPCDフレーム番号セットを事前構築（不要なフレームの読込みをスキップ）
    target_pcd_frame_nums = set()
    for cam_short, frame_map in cam_frame_map.items():
        for fnum in frame_map.keys():
            # PCDフレーム番号 = Annotationフレーム番号 - 1
            target_pcd_frame_nums.add(fnum - 1)

    for bag_dir in bag_dirs:
        # PCDフォルダを探す（local_pcdbin優先、なければlocal_motion_pcdbin）
        pcdbin_dir = None
        for pcdbin_name in ['local_pcdbin', 'local_motion_pcdbin']:
            candidate = bag_dir / 'result' / pcdbin_name
            if candidate.exists() and any(candidate.glob('*.pcd')):
                pcdbin_dir = candidate
                break
        if pcdbin_dir is None:
            logger.warning(f"PCDフォルダが見つかりません（スキップ）: {bag_dir / 'result'}")
            continue

        # PCDファイルをフレーム番号順にソート
        pcd_files = sorted(
            pcdbin_dir.glob('*.pcd'),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
        )

        if not pcd_files:
            logger.warning(f"PCDファイルなし（スキップ）: {pcdbin_dir}")
            continue

        logger.info(f"処理中: {bag_dir.name} ({len(pcd_files)} frames)")

        # --- フレーム並列処理（ProcessPoolExecutor: GIL制約なし） ---
        # ワーカー数: CPU数の半分程度（メモリ使用量を抑制しつつ並列化）
        num_workers = min(4, max(1, os.cpu_count() // 2))

        # ProcessPoolExecutorに渡すための引数リスト構築
        # (pickle可能なデータのみ: undis_remap_func=Noneを明示)
        camera_params_picklable = {}
        for cam_name, param in camera_params.items():
            p = dict(param)
            p['undis_remap_func'] = None  # 関数オブジェクトはpickle不可
            camera_params_picklable[cam_name] = p

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for pcd_path in pcd_files:
                futures[executor.submit(
                    _process_one_frame_worker,
                    pcd_path,
                    camera_params_picklable,
                    target_cameras,
                    dense,
                    target_pcd_frame_nums,
                    cam_frame_map,
                    cam_frame_names
                )] = pcd_path

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  Depth {bag_dir.name}", leave=False):
                try:
                    frame_results = future.result()
                except Exception as e:
                    logger.warning(f"フレーム処理エラー: {e}")
                    continue
                for cam_short, frame_name, depth_img in frame_results:
                    # 出力先: <od_output_dir>/data_3d_<cam_short>/Depth_maps/
                    depth_dir = od_output_dir / f"data_3d_{cam_short}" / 'Depth_maps'
                    depth_dir.mkdir(parents=True, exist_ok=True)

                    out_path = depth_dir / f"{frame_name}.png"
                    # 既存ファイルがあればスキップ（冪等性）
                    if not out_path.exists():
                        cv2_imwrite(out_path, depth_img)

                    stats[cam_short] = stats.get(cam_short, 0) + 1

    logger.info("Depth_maps生成完了:")
    for cam_short, count in sorted(stats.items()):
        logger.info(f"  data_3d_{cam_short}: {count} files")

    return stats


def main():
    args = parse_args()

    bev_result_dir = Path(args.bev_result_dir).resolve()
    if not bev_result_dir.exists():
        logger.error(f"BEV結果ディレクトリが見つかりません: {bev_result_dir}")
        sys.exit(1)

    # 出力先
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = bev_result_dir.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # カメラパラメータ
    cam_param_dir = Path(args.camera_param_dir) if args.camera_param_dir \
        else CAMERA_PARAM_FOLDER
    if not cam_param_dir.exists():
        logger.error(f"カメラパラメータフォルダが見つかりません: {cam_param_dir}")
        sys.exit(1)

    # 対象カメラ
    target_cameras = None
    if args.cameras:
        target_cameras = [c.strip() for c in args.cameras.split(',')]

    # bag_dirsの構築: bev_result_dirから元のbag_dirを復元する
    # 単体実行時は --bag-dirs 引数で直接指定するか、
    # bev_result_dir内のlocal_pcdbin/local_motion_pcdbinを探索する
    bag_dirs = []
    if hasattr(args, 'bag_dirs') and args.bag_dirs:
        bag_dirs = [Path(p.strip()) for p in args.bag_dirs.split(',')]
    else:
        # フォールバック: bev_result_dir配下でPCDフォルダを探索
        logger.info("bag_dirs未指定のため、bev_result_dir配下を探索します")
        for pcdbin_name in ['local_pcdbin', 'local_motion_pcdbin']:
            for sub in bev_result_dir.rglob(pcdbin_name):
                # local_pcdbin/local_motion_pcdbin の2つ上がbag_dir
                # bag_dir/result/local_pcdbin/
                candidate = sub.parent.parent
                if candidate not in bag_dirs:
                    bag_dirs.append(candidate)

    if not bag_dirs:
        logger.error("bag_dirsが見つかりません。--bag-dirs で指定してください")
        sys.exit(1)

    logger.info(f"bag_dirs: {len(bag_dirs)} 件")
    logger.info(f"出力先: {output_dir}")
    logger.info(f"Dense補間: {'有効' if args.dense else '無効'}")

    stats = generate_depth_maps_for_dataset(
        bag_dirs=bag_dirs,
        od_output_dir=output_dir,
        camera_param_folder=cam_param_dir,
        target_cameras=target_cameras,
        dense=args.dense
    )

    # サマリー
    logger.info("=" * 60)
    total = sum(stats.values())
    logger.info(f"合計: {total} files")
    logger.info("=" * 60)


if __name__ == '__main__':
    # Windows でのProcessPoolExecutor安全起動に必要
    multiprocessing.freeze_support()
    main()
