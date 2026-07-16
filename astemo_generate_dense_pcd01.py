"""astemo_generate_dense_pcd.py
============================
NRACapPairパイプラインのresult/フォルダ構造から高密度点群(dense PCD)を生成する。

元のgenerate_dense_pcd.py（BEV学習用・Linux/GPU環境想定）をAstemo NRACapPairパイプライン
向けに再実装したもの。

処理概要:
1. mapping_pose.txt からグローバルポーズ（回転・並進）を取得
2. local_pcdbin/*.pcd（なければlocal_motion_pcdbin）から点群を読み込み
3. GT.csv から動的物体の3D BBoxを取得し、動的物体の点を除去
4. チャンク内の全フレーム点群を基準座標系に集約（元版と同様）
5. 各フレームの座標系に戻し、ボクセルダウンサンプリングして .npz 保存

元版との主な統一点:
- チャンク内の全フレームを集約対象とする（valid_indicesに限定しない）
- マルチスレッド並列処理対応（num_threads指定）
- mmdet3dのpoints_in_boxes_cpu利用可能時は使用（フォールバックあり）
- GPU(PyTorch CUDA)利用可能時は自動的にGPU演算を使用

高速化版での主な変更点:
- フェーズ2: XY平面グリッド索引による空間プレフィルタを導入。
  各出力フレームで軌跡全体の点を変換するのではなく、PC_RANGEに入りうる
  半径内（約125m）のセルの点だけを変換・クリップする（O(全点×フレーム数)
  → O(近傍点×フレーム数)）。
- フェーズ2 GPU: ワールド座標点群をGPUに1回だけ常駐転送し、フレーム毎の
  CPU→GPU全点転送を排除。
- _voxel_down_sample_cpu: np.add.at → np.bincount（5〜10倍高速）。
- parse_gt_csv: 行毎dict構築を廃止し、タイムスタンプ毎の(M,7) BBox配列を
  事前計算して返す（z底面シフト・拡大係数適用済み）。キャッシュはv3。
- フェーズ1: ThreadPoolExecutorによる連続投入（バッチ毎の全join待ちを排除）。
- pyquaternion依存を排除（オイラー角→回転行列を直接計算、結果は同一）。

入力（bag_dir/result/ 配下）:
- mapping/mapping_pose.txt: ポーズ情報
- local_pcdbin/*.pcd（優先）またはlocal_motion_pcdbin/*.pcd: 点群
- GT.csv: 3D BoundingBox正解データ
- (外部) relevance.json: 処理対象フレームの一覧

出力:
- dense_pcd_npz/*.npz: 高密度化された点群 (キー "pcd", shape (M,3))
"""

import os
import gc
import sys
import shutil
import numpy as np
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --- open3d はPCD読み込み時のみ必要（テスト・解析用途でのimportを妨げない） ---
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False
    logger.warning("open3d未インストール: PCD読み込みは利用できません")

# --- GPU/CPU 自動選択 ---
try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        logger.info(f"PyTorch CUDA GPU 使用: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        logger.info("PyTorch CPU モード（CUDA不可）")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    logger.info("PyTorch未インストール: NumPy CPUモードで動作")

# --- mmdet3d のpoints_in_boxes_cpu を使用可能ならインポート ---
try:
    from mmdet3d.ops.roiaware_pool3d import points_in_boxes_cpu as _mmdet3d_points_in_boxes
    HAS_MMDET3D = True
    logger.info("mmdet3d.points_in_boxes_cpu 利用可能")
except ImportError:
    HAS_MMDET3D = False
    logger.info("mmdet3d未インストール: 自前実装のpoints_in_boxesを使用")

# --- 設定定数 ---
# 有効範囲 [xmin, ymin, zmin, xmax, ymax, zmax] (メートル)
PC_RANGE = [-120.0, -30.0, -4.0, 120.0, 30.0, 5.0]

# 自車除去範囲 (メートル)
SELF_RANGE = [3.1, 3.1, 3.1]

# ボクセルダウンサンプリング設定
VOXEL_SIZE_NEAR = 0.2   # 近距離 (|X| <= 50m)
VOXEL_SIZE_FAR = 0.4    # 遠距離 (|X| > 50m)
LONG_DISTANCE_THRESHOLD = 50.0  # 近距離/遠距離の閾値 (メートル)

# BBox拡大係数 [length, width, height]
OBJECT_SIZE_EXPAND_FACTOR = [1.1, 1.1, 1.1]

# マルチスレッド設定
DEFAULT_NUM_THREADS = 2

# フェーズ2空間プレフィルタ設定
GRID_CELL_SIZE = 40.0  # XY平面グリッドのセル辺長 (メートル)
# フレーム原点からPC_RANGE内に入りうる点の最大距離（3D距離の上界 + マージン）。
# ワールドXY距離 <= 3D距離 なので、この半径でのXY検索は保守的（取りこぼしなし）。
PREFILTER_RADIUS = float(np.sqrt(
    max(abs(PC_RANGE[0]), abs(PC_RANGE[3])) ** 2 +
    max(abs(PC_RANGE[1]), abs(PC_RANGE[4])) ** 2 +
    max(abs(PC_RANGE[2]), abs(PC_RANGE[5])) ** 2)) + 1.0

# グリッドセル座標→1Dキーのエンコード定数（|セル座標| < 2^20 を想定）
_GRID_KEY_OFFSET = 1 << 20
_GRID_KEY_STRIDE = 1 << 21

# GPU排他ロック（複数スレッドの同時GPU転送によるOOMを防止）
_GPU_LOCK = threading.Lock()


def parse_mapping_pose(pose_path: Path) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, int]]:
    """mapping_pose.txt を解析し、回転行列・並進ベクトル・タイムスタンプマッピングを返す。

    フォーマット:
    lidar_frame x y z roll pitch yaw match_score lidar_frame_time

    戻り値:
        rotations: {frame_idx: 3x3回転行列}
        translations: {frame_idx: [tx, ty, tz]}
        timestamp_idx_mapping: {timestamp_str(12桁): frame_idx}
    """
    rotations = OrderedDict()
    translations = OrderedDict()
    timestamp_idx_mapping = {}

    with open(pose_path, 'r') as f:
        lines = f.readlines()[1:]  # ヘッダースキップ

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        idx = int(parts[0])
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        roll, pitch, yaw = float(parts[4]), float(parts[5]), float(parts[6])
        # parts[7] = match_score (使わない)
        timestamp_raw = parts[8]

        # タイムスタンプを12桁文字列に統一（秒10桁 + 小数点以下2桁）
        ts_parts = timestamp_raw.split('.')
        if len(ts_parts) == 2:
            timestamp_str = ts_parts[0] + ts_parts[1][:2]
        else:
            timestamp_str = timestamp_raw[:12]

        # オイラー角 → 回転行列 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        # （元実装のオイラー角→クォータニオン→回転行列と数学的に同一）
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rot = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]], dtype=np.float32)

        rotations[idx] = rot
        translations[idx] = np.array([tx, ty, tz], dtype=np.float32)
        timestamp_idx_mapping[timestamp_str] = idx

    return rotations, translations, timestamp_idx_mapping


def parse_gt_csv(gt_csv_path: Path, timestamp_idx_mapping: Dict[str, int],
                 work_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """GT.csv を解析し、タイムスタンプをキーとしたGT BBox配列辞書を返す。

    キャッシュ戦略:
    - gt_parsed_v3.pkl が存在し、GT.csvより新しければそこから読込み（高速）
    - 存在しない or GT.csvの方が新しければCSVから構築し、gt_parsed_v3.pkl に保存

    GT.csvのカラム:
    stamp_sec, frame_num, track_id, type, center.x, center.y, center.z,
    obj_yaw, height, length, width, ...

    引数:
        gt_csv_path: GT.csv のパス
        timestamp_idx_mapping: タイムスタンプ→インデックスのマッピング
        work_dir: キャッシュpklの保存先ディレクトリ（指定時はwork_dir内に保存）

    戻り値:
        gts: {timestamp_str: (M, 7) float32 gt_boxes配列}
        各行は [cx, cy, cz_bottom, length, width, height, yaw]。
        z底面シフト(-height/2 - 0.2)とOBJECT_SIZE_EXPAND_FACTORは適用済みで、
        そのまま points_in_boxes に渡せる形式。
    """
    # キャッシュパスの決定（work_dir指定時はwork_dir内に保存）
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = work_dir / 'gt_parsed_v3.pkl'
    else:
        pkl_path = gt_csv_path.parent / 'gt_parsed_v3.pkl'

    # キャッシュの有効性チェック（v3: BBox配列を事前計算した形式）
    if pkl_path.exists():
        pkl_mtime = pkl_path.stat().st_mtime
        csv_mtime = gt_csv_path.stat().st_mtime
        if pkl_mtime > csv_mtime:
            logger.info(f"gt_parsed_v3.pkl キャッシュから読込み: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)

    # CSVから構築（全カラムをベクトル化処理、Python行ループなし）
    logger.info(f"GT.csv をパース中: {gt_csv_path}")
    df = pd.read_csv(gt_csv_path, low_memory=False)

    # NaN行を除去し、type範囲フィルタ（ベクトル化）
    df = df.dropna(subset=['stamp_sec'])
    df = df[df['type'].notna()]
    type_int = df['type'].astype(int)
    df = df[(type_int >= 0) & (type_int <= 14)]

    # タイムスタンプ文字列を一括生成
    stamp_strs = df['stamp_sec'].apply(
        lambda s: f"{float(s):.6f}".replace('.', '')[:12]).to_numpy()

    cx = df['center.x'].to_numpy(dtype=np.float64)
    cy = df['center.y'].to_numpy(dtype=np.float64)
    cz = df['center.z'].to_numpy(dtype=np.float64)
    obj_yaws = df['obj_yaw'].to_numpy(dtype=np.float64)
    heights = df['height'].to_numpy(dtype=np.float64)
    widths = df['width'].to_numpy(dtype=np.float64)
    lengths = df['length'].to_numpy(dtype=np.float64)

    # points_in_boxes 仕様のBBoxを事前計算:
    # z は底面座標（center.z - height/2 - 0.2）、寸法は拡大係数適用後
    boxes = np.column_stack([
        cx,
        cy,
        cz - heights / 2.0 - 0.2,
        lengths * OBJECT_SIZE_EXPAND_FACTOR[0],
        widths * OBJECT_SIZE_EXPAND_FACTOR[1],
        heights * OBJECT_SIZE_EXPAND_FACTOR[2],
        obj_yaws,
    ]).astype(np.float32)

    # タイムスタンプ毎にグルーピング（ソート + スライス分割で一括処理）
    order = np.argsort(stamp_strs, kind='stable')
    stamps_sorted = stamp_strs[order]
    boxes_sorted = boxes[order]

    gts = OrderedDict()
    if len(stamps_sorted) > 0:
        unique_stamps, starts = np.unique(stamps_sorted, return_index=True)
        ends = np.append(starts[1:], len(stamps_sorted))
        for ts, s, e in zip(unique_stamps, starts, ends):
            gts[str(ts)] = boxes_sorted[s:e]

    # キャッシュとして保存（読み取り専用ファイルシステムの場合はスキップ）
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(gts, f)
        logger.info(f"gt_parsed_v3.pkl キャッシュ保存: {pkl_path}")
    except OSError as e:
        logger.warning(f"gt_parsed_v3.pkl キャッシュ保存スキップ（書き込み不可）: {e}")

    return gts


def parse_relevance_json(relevance_path: Path) -> List[int]:
    """relevance.json から処理対象のフレームインデックスを取得する。

    relevance.jsonのエントリ: {"pcd": "...\\NCAP_000005.pcd", ...}
    → フレーム番号 5 を抽出

    戻り値:
        valid_indices: [5, 10, 15, ...] 処理対象フレームのリスト
    """
    with open(relevance_path, 'r') as f:
        data = json.load(f)

    valid_indices = []
    for entry in data:
        pcd_path = entry.get('pcd', '')
        basename = os.path.splitext(os.path.basename(pcd_path))[0]
        # "NCAP_000005" → 5
        parts = basename.split('_')
        if len(parts) >= 2:
            try:
                idx = int(parts[-1])
                valid_indices.append(idx)
            except ValueError:
                continue

    return sorted(valid_indices)


def parse_result_json_dir(result_dir: Path) -> List[int]:
    """result/ フォルダ内の *.json ファイル名からフレームインデックスを取得する。

    result/*.json のファイル名形式: NCAP_000005.json
    → フレーム番号 5 を抽出

    これにより、実際に生成されたBEV結果JSONと1対1対応するnpzを生成できる。

    戻り値:
        valid_indices: [5, 10, 15, ...] 処理対象フレームのリスト
    """
    if not result_dir.exists():
        logger.warning(f"result JSONフォルダが見つかりません: {result_dir}")
        return []

    valid_indices = []
    for json_file in sorted(result_dir.glob('*.json')):
        basename = json_file.stem  # "NCAP_000005"
        parts = basename.split('_')
        if len(parts) >= 2:
            try:
                idx = int(parts[-1])
                valid_indices.append(idx)
            except ValueError:
                continue

    return sorted(valid_indices)


def build_pcd_file_mapping(pcd_dir: Path,
                           mapping_pose_path: Path) -> Tuple[Dict[int, dict], Dict[str, int]]:
    """PCDフォルダ（local_pcdbin/local_motion_pcdbin）のPCDファイルとmapping_poseのタイムスタンプを紐付ける。

    PCDフォルダのファイル名: 0.pcd, 1.pcd, 2.pcd, ...
    mapping_pose.txt の lidar_frame: 0, 1, 2, ...（同じインデックス）

    戻り値:
        pcd_files: {frame_idx: {'pcd_file_path': str, 'timestamp': str}}
        timestamp_idx_mapping: {timestamp_str: pose_idx}
    """
    rotations, translations, timestamp_idx_mapping = parse_mapping_pose(mapping_pose_path)

    # idx→timestamp の逆引き
    idx_to_timestamp = {v: k for k, v in timestamp_idx_mapping.items()}

    pcd_files = OrderedDict()
    for pcd_file in sorted(pcd_dir.glob('*.pcd')):
        try:
            frame_idx = int(pcd_file.stem)
        except ValueError:
            continue
        if frame_idx in idx_to_timestamp:
            pcd_files[frame_idx] = {
                'pcd_file_path': str(pcd_file),
                'timestamp': idx_to_timestamp[frame_idx]
            }

    pcd_files = OrderedDict(sorted(pcd_files.items()))
    return pcd_files, timestamp_idx_mapping


def lidar_to_world_to_lidar(pc: np.ndarray,
                            cur_rot: np.ndarray, cur_trans: np.ndarray,
                            base_rot: np.ndarray, base_trans: np.ndarray) -> np.ndarray:
    """点群を現在のLiDAR座標系から基準LiDAR座標系へ変換する。

    GPU(PyTorch CUDA)が利用可能な場合は自動的にGPU演算を使用。
    数式: p' = R_base^T × (R_cur × p + t_cur - t_base)

    引数:
        pc: (N, 3) 点群
        cur_rot: (3, 3) 現在フレームの回転行列
        cur_trans: (3,) 現在フレームの並進ベクトル
        base_rot: (3, 3) 基準フレームの回転行列
        base_trans: (3,) 基準フレームの並進ベクトル

    戻り値:
        (N, 3) 変換後の点群 (numpy配列)
    """
    if HAS_TORCH and DEVICE.type == 'cuda':
        return _lidar_to_world_to_lidar_gpu(pc, cur_rot, cur_trans, base_rot, base_trans)
    else:
        return _lidar_to_world_to_lidar_cpu(pc, cur_rot, cur_trans, base_rot, base_trans)


def _lidar_to_world_to_lidar_gpu(pc: np.ndarray,
                                  cur_rot: np.ndarray, cur_trans: np.ndarray,
                                  base_rot: np.ndarray, base_trans: np.ndarray,
                                  gpu_id: int = 0) -> np.ndarray:
    """GPU(PyTorch)版の座標変換。OOM時はCPUフォールバック。"""
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else DEVICE
    try:
        with _GPU_LOCK:
            pc_t = torch.from_numpy(pc).to(torch.float32).to(device)
            cur_rot_t = torch.from_numpy(cur_rot).to(torch.float32).to(device)
            cur_trans_t = torch.from_numpy(cur_trans).to(torch.float32).to(device).reshape(3, 1)
            base_rot_t = torch.from_numpy(base_rot).to(torch.float32).to(device)
            base_trans_t = torch.from_numpy(base_trans).to(torch.float32).to(device).reshape(3, 1)

            pc_t = pc_t.permute(1, 0)  # (N, 3) → (3, N)
            pc_world = torch.matmul(cur_rot_t, pc_t) + cur_trans_t
            pc_base = torch.matmul(base_rot_t.permute(1, 0), pc_world - base_trans_t)

            result = pc_base.permute(1, 0).cpu().numpy()

            # 明示的にGPUテンソルを解放
            del pc_t, cur_rot_t, cur_trans_t, base_rot_t, base_trans_t, pc_world, pc_base
            torch.cuda.synchronize(device)

        return result
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # OOM発生: GPUメモリ解放してCPUフォールバック
        if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
            logger.warning(f"GPU OOM検出 ({pc.shape[0]:,}点) → CPUフォールバック")
            torch.cuda.empty_cache()
            gc.collect()
            return _lidar_to_world_to_lidar_cpu(pc, cur_rot, cur_trans, base_rot, base_trans)
        raise


def _lidar_to_world_to_lidar_cpu(pc: np.ndarray,
                                  cur_rot: np.ndarray, cur_trans: np.ndarray,
                                  base_rot: np.ndarray, base_trans: np.ndarray) -> np.ndarray:
    """CPU(NumPy)版の座標変換。"""
    cur_trans = cur_trans.reshape(3, 1)
    base_trans = base_trans.reshape(3, 1)
    pc_t = pc.T  # (N, 3) → (3, N)
    pc_world = cur_rot @ pc_t + cur_trans
    pc_base = base_rot.T @ (pc_world - base_trans)
    return pc_base.T  # (N, 3)


def points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """各点がどの3D BBox内に存在するか判定する。

    mmdet3dが利用可能ならそちらを使用（高速）、なければ自前実装。

    引数:
        points: (N, 3) 点群
        boxes: (M, 7) BBox [cx, cy, cz, length, width, height, yaw]

    戻り値:
        mask: (N,) bool配列。いずれかのBBox内にある点はTrue。
    """
    if HAS_MMDET3D:
        return _points_in_boxes_mmdet3d(points, boxes)
    else:
        return _points_in_boxes_cpu(points, boxes)


def _points_in_boxes_mmdet3d(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """mmdet3d版のpoints_in_boxes。"""
    import torch as _torch
    pts_tensor = _torch.from_numpy(points[:, :3]).float().unsqueeze(0)  # (1, N, 3)
    boxes_tensor = _torch.from_numpy(boxes).float().unsqueeze(0)  # (1, M, 7)
    box_idxs = _mmdet3d_points_in_boxes(pts_tensor, boxes_tensor)  # (1, N)
    mask = (box_idxs.squeeze(0) >= 0).numpy()
    return mask


def _points_in_boxes_cpu(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """自前実装のpoints_in_boxes（CPU）。mmdet3d互換。ベクトル化版。

    mmdet3d の points_in_boxes_cpu と同じ仕様:
    - boxes[:, 2] (cz) はBBoxの底面z座標
    - 高さ方向の判定: 0 <= (z - cz) <= height
    - XY方向はyaw回転後に length/2, width/2 で判定

    全ボックスを一括で判定するベクトル化実装。
    ボックス数が少ない場合（<=8）はループ版にフォールバック。
    """
    N = points.shape[0]
    M = boxes.shape[0]

    if M == 0:
        return np.zeros(N, dtype=bool)

    # ボックス数が少ない場合はメモリ効率的なループ版
    if M <= 8:
        mask = np.zeros(N, dtype=bool)
        for box in boxes:
            cx, cy, cz, length, width, height, yaw = box
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            dx = points[:, 0] - cx
            dy = points[:, 1] - cy
            dz = points[:, 2] - cz
            local_x = dx * cos_yaw - dy * sin_yaw
            local_y = dx * sin_yaw + dy * cos_yaw
            in_box = ((np.abs(local_x) <= length / 2.0) &
                      (np.abs(local_y) <= width / 2.0) &
                      (dz >= 0) & (dz <= height))
            mask |= in_box
        return mask

    # ベクトル化版: 全ボックスを一括判定（メモリ使用量 N*M に注意）
    # メモリが大きすぎる場合はバッチ分割
    BATCH_LIMIT = 50_000_000  # N*M の上限
    if N * M > BATCH_LIMIT:
        # バッチ分割版
        mask = np.zeros(N, dtype=bool)
        batch_size = max(1, BATCH_LIMIT // M)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pts_batch = points[start:end]  # (B, 3)

            # (B, M)
            dx = pts_batch[:, 0:1] - boxes[:, 0]  # (B, M)
            dy = pts_batch[:, 1:2] - boxes[:, 1]
            dz = pts_batch[:, 2:3] - boxes[:, 2]

            cos_yaw = np.cos(-boxes[:, 6])  # (M,)
            sin_yaw = np.sin(-boxes[:, 6])

            local_x = dx * cos_yaw - dy * sin_yaw
            local_y = dx * sin_yaw + dy * cos_yaw

            half_l = boxes[:, 3] / 2.0  # (M,)
            half_w = boxes[:, 4] / 2.0
            height = boxes[:, 5]

            in_box = ((np.abs(local_x) <= half_l) &
                      (np.abs(local_y) <= half_w) &
                      (dz >= 0) & (dz <= height))
            mask[start:end] = in_box.any(axis=1)
        return mask
    else:
        # 全展開版（高速だがメモリ使用量大）
        # points: (N, 3), boxes: (M, 7)
        dx = points[:, 0:1] - boxes[:, 0]  # (N, M)
        dy = points[:, 1:2] - boxes[:, 1]
        dz = points[:, 2:3] - boxes[:, 2]

        cos_yaw = np.cos(-boxes[:, 6])  # (M,)
        sin_yaw = np.sin(-boxes[:, 6])

        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        half_l = boxes[:, 3] / 2.0
        half_w = boxes[:, 4] / 2.0
        height = boxes[:, 5]

        in_box = ((np.abs(local_x) <= half_l) &
                  (np.abs(local_y) <= half_w) &
                  (dz >= 0) & (dz <= height))
        return in_box.any(axis=1)


def voxel_down_sample(points: np.ndarray, voxel_size: float,
                      gpu_id: int = 0) -> np.ndarray:
    """ボクセルダウンサンプリング。GPU利用可能時は自動でGPU版を使用。

    各ボクセル内の点の重心を代表点とする。

    引数:
        points: (N, 3) 点群
        voxel_size: ボクセルの辺長 (メートル)
        gpu_id: 使用するGPU番号

    戻り値:
        (M, 3) ダウンサンプリング後の点群
    """
    if points.shape[0] == 0:
        return points

    if HAS_TORCH and DEVICE.type == 'cuda':
        return _voxel_down_sample_gpu(points, voxel_size, gpu_id)
    else:
        return _voxel_down_sample_cpu(points, voxel_size)


def _voxel_down_sample_gpu(points, voxel_size: float,
                            gpu_id: int = 0) -> np.ndarray:
    """GPU(PyTorch)版のボクセルダウンサンプリング。scatter_add_使用で高速。

    引数:
        points: np.ndarray (N,3) または torch.Tensor (N,3)（GPU上のテンソルも直接受付可）
        voxel_size: ボクセルの辺長
        gpu_id: 使用するGPU番号

    戻り値:
        (M, 3) np.ndarray ダウンサンプリング後の点群
    """
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else DEVICE

    # torch.Tensor が直接渡された場合はGPU転送をスキップ（CPU往復排除）
    if isinstance(points, torch.Tensor):
        pts = points.to(torch.float32)
        if pts.device != device:
            pts = pts.to(device)
    else:
        pts = torch.from_numpy(points).to(torch.float32).to(device)

    min_coords = pts.min(dim=0)[0]
    voxel_indices = torch.floor((pts - min_coords) / voxel_size).long()

    dims = voxel_indices.max(dim=0)[0] + 1
    keys = (voxel_indices[:, 0] * (dims[1] * dims[2]) +
            voxel_indices[:, 1] * dims[2] +
            voxel_indices[:, 2])

    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    num_voxels = unique_keys.shape[0]

    sums = torch.zeros((num_voxels, 3), device=device, dtype=torch.float32)
    counts = torch.zeros(num_voxels, device=device, dtype=torch.float32)

    expand_indices = inverse.unsqueeze(1).expand(-1, 3)
    sums.scatter_add_(0, expand_indices, pts)
    ones = torch.ones(pts.shape[0], device=device, dtype=torch.float32)
    counts.scatter_add_(0, inverse, ones)

    result = (sums / (counts.unsqueeze(1) + 1e-8)).cpu().numpy()
    return result.astype(np.float32)


def _voxel_down_sample_cpu(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """CPU(NumPy)版のボクセルダウンサンプリング。

    np.add.at は非常に遅いため、np.bincount で合計・カウントを集計する
    （同一結果で2〜5倍高速）。
    """
    min_coords = points.min(axis=0)
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(np.int64)

    dims = voxel_indices.max(axis=0) + 1
    keys = (voxel_indices[:, 0] * (dims[1] * dims[2]) +
            voxel_indices[:, 1] * dims[2] +
            voxel_indices[:, 2])

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    num_voxels = len(unique_keys)

    counts = np.bincount(inverse, minlength=num_voxels)
    sums = np.empty((num_voxels, 3), dtype=np.float64)
    for k in range(3):
        sums[:, k] = np.bincount(inverse, weights=points[:, k], minlength=num_voxels)

    result = (sums / counts[:, np.newaxis]).astype(np.float32)
    return result


def _build_xy_grid_index(points: np.ndarray,
                         cell_size: float = GRID_CELL_SIZE
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ワールド座標点群をXYグリッドセル順にソートし、セル→スライスの索引を構築する。

    フェーズ2で「フレーム原点の近傍セルの点だけ」を取り出すための空間索引。
    ソートは1回だけ行い、以降の各フレームではスライス参照のみで近傍点を取得できる。

    引数:
        points: (N, 3) ワールド座標点群
        cell_size: セル辺長 (メートル)

    戻り値:
        sorted_points: (N, 3) セルキー順にソートされた点群
        cell_starts: (K,) 各ユニークセルの開始インデックス
        cell_ends: (K,) 各ユニークセルの終了インデックス（排他的）
        cell_x, cell_y: (K,) 各ユニークセルの2Dグリッド座標
    """
    if points.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return points, empty, empty, empty, empty

    cell_xy = np.floor(points[:, :2] / cell_size).astype(np.int64)
    keys = ((cell_xy[:, 0] + _GRID_KEY_OFFSET) * _GRID_KEY_STRIDE
            + (cell_xy[:, 1] + _GRID_KEY_OFFSET))

    order = np.argsort(keys)
    sorted_points = points[order]
    sorted_keys = keys[order]

    unique_keys, cell_starts = np.unique(sorted_keys, return_index=True)
    cell_ends = np.append(cell_starts[1:], sorted_keys.shape[0])
    cell_x = unique_keys // _GRID_KEY_STRIDE - _GRID_KEY_OFFSET
    cell_y = unique_keys % _GRID_KEY_STRIDE - _GRID_KEY_OFFSET

    return sorted_points, cell_starts, cell_ends, cell_x, cell_y


def _select_grid_slices(center_xy: np.ndarray, radius: float,
                        cell_starts: np.ndarray, cell_ends: np.ndarray,
                        cell_x: np.ndarray, cell_y: np.ndarray,
                        cell_size: float = GRID_CELL_SIZE) -> List[Tuple[int, int]]:
    """中心±radius のAABBと交差するセル群のスライス範囲リストを返す。

    保守的な選択（AABB判定）のため取りこぼしはない。範囲外の点は後段の
    PC_RANGEマスクで除去される。ソート済み配列上で連続するセルは
    1つのスライスに統合し、コピー回数を最小化する。

    戻り値:
        [(start, end), ...] sorted_points に対するスライス範囲
    """
    if cell_starts.shape[0] == 0:
        return []

    cx_min = int(np.floor((center_xy[0] - radius) / cell_size))
    cx_max = int(np.floor((center_xy[0] + radius) / cell_size))
    cy_min = int(np.floor((center_xy[1] - radius) / cell_size))
    cy_max = int(np.floor((center_xy[1] + radius) / cell_size))

    sel = np.where((cell_x >= cx_min) & (cell_x <= cx_max) &
                   (cell_y >= cy_min) & (cell_y <= cy_max))[0]
    if sel.size == 0:
        return []

    # 連続するユニークセルはソート済み配列上で連続 → スライス統合
    slices = []
    run_start = sel[0]
    prev = sel[0]
    for s in sel[1:]:
        if s == prev + 1:
            prev = s
            continue
        slices.append((int(cell_starts[run_start]), int(cell_ends[prev])))
        run_start = s
        prev = s
    slices.append((int(cell_starts[run_start]), int(cell_ends[prev])))
    return slices


def _clip_to_pc_range_np(pts: np.ndarray) -> np.ndarray:
    """PC_RANGE内の点だけを残す（NumPy版）。"""
    mask = ((pts[:, 0] > PC_RANGE[0]) & (pts[:, 0] < PC_RANGE[3]) &
            (pts[:, 1] > PC_RANGE[1]) & (pts[:, 1] < PC_RANGE[4]) &
            (pts[:, 2] > PC_RANGE[2]) & (pts[:, 2] < PC_RANGE[5]))
    return pts[mask]


def _phase2_frame_cpu(world_pts: np.ndarray, slices: List[Tuple[int, int]],
                      cur_rot: np.ndarray, cur_trans: np.ndarray,
                      obj_pts: Optional[np.ndarray]) -> List[np.ndarray]:
    """1フレーム分のフェーズ2処理（CPU）。

    近傍スライスの点をフレーム座標系に変換 → PC_RANGEクリップ →
    動的物体点再追加 → near/farボクセルダウンサンプリング。

    戻り値:
        ダウンサンプリング済み点群のリスト（空なら有効点なし）
    """
    # 合成オフセット形式: p_frame = R^T p_world - R^T t（行ベクトル形: p @ R - offset）
    combined_offset = (cur_rot.T @ cur_trans.reshape(3, 1)).flatten().astype(np.float32)

    pts_list = []
    for s, e in slices:
        pc_frame = world_pts[s:e] @ cur_rot - combined_offset
        clipped = _clip_to_pc_range_np(pc_frame)
        if clipped.shape[0] > 0:
            pts_list.append(clipped)

    if not pts_list:
        return []

    all_pts = np.concatenate(pts_list, axis=0)
    del pts_list

    if obj_pts is not None and obj_pts.shape[0] > 0:
        all_pts = np.concatenate([all_pts, obj_pts], axis=0)

    far_mask = np.abs(all_pts[:, 0]) > LONG_DISTANCE_THRESHOLD
    pts_near = all_pts[~far_mask]
    pts_far = all_pts[far_mask]
    del all_pts

    results = []
    if pts_near.shape[0] > 0:
        results.append(_voxel_down_sample_cpu(pts_near, VOXEL_SIZE_NEAR))
    if pts_far.shape[0] > 0:
        results.append(_voxel_down_sample_cpu(pts_far, VOXEL_SIZE_FAR))
    return results


def _phase2_frame_gpu(world_t, slices: List[Tuple[int, int]],
                      cur_rot: np.ndarray, cur_trans: np.ndarray,
                      obj_pts: Optional[np.ndarray]) -> List[np.ndarray]:
    """1フレーム分のフェーズ2処理（GPU）。

    world_t はGPU常駐のワールド座標テンソル。スライス参照のみで近傍点を
    取り出すため、フレーム毎のCPU→GPU転送は発生しない。
    """
    device = world_t.device
    combined_offset = (cur_rot.T @ cur_trans.reshape(3, 1)).flatten().astype(np.float32)
    cur_rot_t = torch.from_numpy(np.ascontiguousarray(cur_rot)).to(device)
    offset_t = torch.from_numpy(combined_offset).to(device)

    cand = torch.cat([world_t[s:e] for s, e in slices], dim=0)
    pc_frame = cand @ cur_rot_t - offset_t
    del cand

    mask = ((pc_frame[:, 0] > PC_RANGE[0]) & (pc_frame[:, 0] < PC_RANGE[3]) &
            (pc_frame[:, 1] > PC_RANGE[1]) & (pc_frame[:, 1] < PC_RANGE[4]) &
            (pc_frame[:, 2] > PC_RANGE[2]) & (pc_frame[:, 2] < PC_RANGE[5]))
    clipped = pc_frame[mask]
    del pc_frame, mask

    if clipped.shape[0] == 0:
        return []

    if obj_pts is not None and obj_pts.shape[0] > 0:
        obj_t = torch.from_numpy(obj_pts).to(device)
        clipped = torch.cat([clipped, obj_t], dim=0)
        del obj_t

    far_mask = torch.abs(clipped[:, 0]) > LONG_DISTANCE_THRESHOLD
    pts_near_t = clipped[~far_mask]
    pts_far_t = clipped[far_mask]
    del clipped, far_mask

    results = []
    if pts_near_t.shape[0] > 0:
        results.append(_voxel_down_sample_gpu(pts_near_t, VOXEL_SIZE_NEAR))
    if pts_far_t.shape[0] > 0:
        results.append(_voxel_down_sample_gpu(pts_far_t, VOXEL_SIZE_FAR))
    del pts_near_t, pts_far_t
    return results


def _phase1_collect_chunk(chunk_frame_indices: List[int],
                          pcd_files: Dict[int, dict],
                          rotations: Dict[int, np.ndarray],
                          translations: Dict[int, np.ndarray],
                          timestamp_idx_mapping: Dict[str, int],
                          gts: Dict[str, np.ndarray],
                          valid_indices_set: set,
                          base_rot: np.ndarray,
                          base_trans: np.ndarray,
                          temp_dir: Path,
                          chunk_idx: int,
                          gpu_id: int = 0,
                          chunk_label: str = "") -> Tuple[Optional[Path], Dict[int, np.ndarray]]:
    """フェーズ1: 小チャンクのPCDを読み込み → 基準座標系に変換 → 中間npyとしてディスク保存。

    ダウンサンプリングは行わない（フェーズ2で統合後に実施）。

    引数:
        chunk_frame_indices: このチャンクで処理するフレームインデックスリスト
        pcd_files: フレーム→PCDパスのマッピング
        rotations: フレーム→回転行列
        translations: フレーム→並進ベクトル
        timestamp_idx_mapping: タイムスタンプ→フレームインデックス
        gts: タイムスタンプ→GT BBox配列 (M,7)（parse_gt_csvの事前計算済み形式）
        valid_indices_set: 出力対象フレームのセット（動的物体点保持用）
        base_rot: 基準フレームの回転行列
        base_trans: 基準フレームの並進ベクトル
        temp_dir: 中間ファイル保存先
        chunk_idx: チャンク番号
        gpu_id: 使用するGPU番号
        chunk_label: ログ用ラベル

    戻り値:
        (中間npyパス or None, {frame_idx: object_points})
    """
    all_static_points_list = []
    frame_object_points = {}

    for idx in tqdm(chunk_frame_indices, desc=f"{chunk_label} Phase1", leave=False):
        if idx not in pcd_files:
            continue

        frame_info = pcd_files[idx]
        frame_timestamp = frame_info['timestamp']

        if frame_timestamp not in timestamp_idx_mapping:
            continue

        pose_idx = timestamp_idx_mapping[frame_timestamp]
        cur_rot = rotations[pose_idx]
        cur_trans = translations[pose_idx]

        # PCD読み込み
        try:
            cloud = o3d.io.read_point_cloud(frame_info['pcd_file_path'])
            points = np.asarray(cloud.points, dtype=np.float32)
        except Exception as e:
            logger.warning(f"PCD読み込み失敗: {frame_info['pcd_file_path']}: {e}")
            continue

        if points.shape[0] == 0:
            continue

        # --- 動的物体の除去 ---
        # gt_boxes はparse_gt_csvで事前計算済み（z底面シフト・拡大係数適用済み）
        gt_boxes = gts.get(frame_timestamp)
        if gt_boxes is not None and gt_boxes.shape[0] > 0:
            in_box_mask = points_in_boxes(points, gt_boxes)

            # 動的物体点を保存（validフレームのみ: フェーズ2で再追加用）
            if (idx + 1) in valid_indices_set:
                obj_pts = points[in_box_mask]
                if obj_pts.shape[0] > 0:
                    frame_object_points[idx] = obj_pts

            points = points[~in_box_mask]

        # --- 自車除去 ---
        ego_mask = ((np.abs(points[:, 0]) > SELF_RANGE[0]) |
                    (np.abs(points[:, 1]) > SELF_RANGE[1]) |
                    (np.abs(points[:, 2]) > SELF_RANGE[2]))
        points = points[ego_mask]

        if points.shape[0] == 0:
            continue

        # --- 基準座標系に変換 ---
        if HAS_TORCH and DEVICE.type == 'cuda':
            transformed = _lidar_to_world_to_lidar_gpu(
                points, cur_rot, cur_trans, base_rot, base_trans, gpu_id)
        else:
            transformed = _lidar_to_world_to_lidar_cpu(
                points, cur_rot, cur_trans, base_rot, base_trans)

        all_static_points_list.append(transformed)

    if not all_static_points_list:
        return None, frame_object_points

    # 結合して中間npyに保存（ダウンサンプリングなし・非圧縮で高速）
    chunk_points = np.concatenate(all_static_points_list, axis=0).astype(np.float32)
    del all_static_points_list

    temp_path = temp_dir / f"chunk_{chunk_idx:04d}.npy"
    np.save(str(temp_path), chunk_points)
    logger.info(f"{chunk_label} 中間保存: {chunk_points.shape[0]:,} 点 → {temp_path.name}")

    del chunk_points
    gc.collect()

    return temp_path, frame_object_points


def generate_dense_pcd_for_bag(bag_dir: Path,
                               relevance_json_path: Path,
                               output_dir: Path,
                               num_threads: int = DEFAULT_NUM_THREADS,
                               phase1_chunk_size: int = 50,
                               result_json_dir: Optional[Path] = None,
                               ) -> Optional[Dict[str, int]]:
    """1つのbagディレクトリに対して高密度点群を生成する（2フェーズ方式）。

    全フレームを集約対象とし、メモリ効率的に高密度化を実現する。

    フェーズ1: 全フレームを小チャンク（phase1_chunk_size）ずつ読み込み
              → 基準座標系に変換 → 中間npyとしてディスク保存（ダウンサンプリングなし）
    フェーズ2: 中間npyをワールド座標に変換して1本の配列に統合し、
              XYグリッド索引を構築。各validフレームでは
              → 近傍セルの点だけをスライス参照で取得 → フレーム座標系に変換
              → PC_RANGEクリップ → 動的物体点再追加
              → near/farボクセルダウンサンプリング → npz保存
              （範囲外の点は変換前に除外されるため、従来の全点変換より大幅に高速。
              　結果は従来方式と同一）

    引数:
        bag_dir: bagフォルダ（result/ を含む親フォルダ）
        relevance_json_path: relevance.json のパス
        output_dir: 出力先フォルダ（dense_pcd_npz/ が作られる）
        num_threads: 並列スレッド数（フェーズ1で使用）
        phase1_chunk_size: フェーズ1の1チャンクあたりのフレーム数
        result_json_dir: BEV結果JSONフォルダ（指定時、このフォルダ内のjsonと1対1でnpzを生成）

    戻り値:
        統計情報 {'total_frames': N, 'generated': M, 'skipped': K} または None（失敗時）
    """
    if not HAS_OPEN3D:
        logger.error("open3dがインストールされていないため、PCDを読み込めません")
        return None

    result_dir = bag_dir / 'result'
    mapping_pose_path = result_dir / 'mapping' / 'mapping_pose.txt'
    # PCDフォルダ: local_pcdbin優先、なければlocal_motion_pcdbin
    pcd_dir = None
    for pcdbin_name in ['local_pcdbin', 'local_motion_pcdbin']:
        candidate = result_dir / pcdbin_name
        if candidate.exists() and any(candidate.glob('*.pcd')):
            pcd_dir = candidate
            break
    gt_csv_path = result_dir / 'GT.csv'

    # 入力ファイルの存在チェック
    if not mapping_pose_path.exists():
        logger.error(f"mapping_pose.txt が見つかりません: {mapping_pose_path}")
        return None
    if pcd_dir is None:
        logger.error(f"PCDフォルダ（local_pcdbin/local_motion_pcdbin）が見つかりません: {result_dir}")
        return None
    if not gt_csv_path.exists():
        logger.error(f"GT.csv が見つかりません: {gt_csv_path}")
        return None
    if not relevance_json_path.exists():
        logger.error(f"relevance.json が見つかりません: {relevance_json_path}")
        return None

    # 出力先
    dense_pcd_dir = output_dir / 'dense_pcd_npz'
    dense_pcd_dir.mkdir(parents=True, exist_ok=True)

    # 完了チェック
    finish_marker = dense_pcd_dir / 'finish.txt'
    if finish_marker.exists():
        logger.info(f"高密度点群生成済み（スキップ）: {dense_pcd_dir}")
        return {'total_frames': 0, 'generated': 0, 'skipped': 0}

    # --- データ読み込み ---
    logger.info(f"mapping_pose.txt 読み込み: {mapping_pose_path}")
    rotations, translations, timestamp_idx_mapping = parse_mapping_pose(mapping_pose_path)

    logger.info(f"PCDファイルマッピング構築: {pcd_dir}")
    pcd_files, _ = build_pcd_file_mapping(pcd_dir, mapping_pose_path)

    logger.info(f"GT.csv 読み込み: {gt_csv_path}")
    gts = parse_gt_csv(gt_csv_path, timestamp_idx_mapping, work_dir=output_dir)

    # --- 処理対象フレームの決定 ---
    # result_json_dir が指定されている場合: result/*.json と1対1対応するnpzを生成
    # 未指定の場合: relevance.json ベース（従来動作）
    if result_json_dir is not None and result_json_dir.exists():
        logger.info(f"result JSONフォルダからフレーム一覧取得: {result_json_dir}")
        valid_indices = parse_result_json_dir(result_json_dir)
        if not valid_indices:
            # フォールバック: relevance.json
            logger.warning("result JSONフォルダにファイルがありません。relevance.jsonにフォールバック")
            logger.info(f"relevance.json 読み込み: {relevance_json_path}")
            valid_indices = parse_relevance_json(relevance_json_path)
        else:
            logger.info(f"result JSONベース: {len(valid_indices)} フレーム対象")
    else:
        logger.info(f"relevance.json 読み込み: {relevance_json_path}")
        valid_indices = parse_relevance_json(relevance_json_path)

    valid_indices_set = set(valid_indices)

    if not valid_indices:
        logger.warning("処理対象フレームが0件です")
        return {'total_frames': 0, 'generated': 0, 'skipped': 0}

    all_frame_indices = sorted(pcd_files.keys())
    total_len = len(all_frame_indices)

    logger.info(f"処理対象フレーム数: {len(valid_indices)}, 全フレーム数: {total_len}, "
                f"フェーズ1チャンクサイズ: {phase1_chunk_size}")

    stats = {'total_frames': len(valid_indices), 'generated': 0, 'skipped': 0}

    # 基準フレーム: 全体の最初のフレーム
    base_idx = all_frame_indices[0]
    base_timestamp = pcd_files[base_idx]['timestamp']
    base_pose_idx = timestamp_idx_mapping[base_timestamp]
    base_rot = rotations[base_pose_idx]
    base_trans = translations[base_pose_idx]

    # ====== フェーズ1: 全フレームをチャンク分割で読込み → 基準座標系変換 → 中間npy保存 ======
    temp_dir = dense_pcd_dir / '_temp_chunks'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # チャンク分割
    num_chunks = (total_len + phase1_chunk_size - 1) // phase1_chunk_size
    chunks = []
    for chunk_idx in range(num_chunks):
        start_pos = chunk_idx * phase1_chunk_size
        end_pos = min((chunk_idx + 1) * phase1_chunk_size, total_len)
        chunk_frames = all_frame_indices[start_pos:end_pos]
        if chunk_frames:
            chunks.append((chunk_idx, chunk_frames))

    logger.info(f"=== フェーズ1開始: {num_chunks}チャンク × {phase1_chunk_size}フレーム ===")

    # GPU数を確認
    num_gpus = 1
    if HAS_TORCH and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

    temp_npz_paths = []
    all_frame_object_points = {}  # {frame_idx: object_points} 全フレーム分

    # ThreadPoolExecutorで全チャンクを連続投入
    # （旧実装のバッチ毎join待ちを排除し、遅いチャンクによるアイドルを防ぐ）
    with ThreadPoolExecutor(max_workers=max(1, num_threads)) as executor:
        futures = {}
        for c_idx, c_frames in chunks:
            label = f"[Phase1 Chunk {c_idx+1}/{num_chunks}]"
            fut = executor.submit(
                _phase1_collect_chunk,
                chunk_frame_indices=c_frames,
                pcd_files=pcd_files,
                rotations=rotations,
                translations=translations,
                timestamp_idx_mapping=timestamp_idx_mapping,
                gts=gts,
                valid_indices_set=valid_indices_set,
                base_rot=base_rot,
                base_trans=base_trans,
                temp_dir=temp_dir,
                chunk_idx=c_idx,
                gpu_id=c_idx % num_gpus,
                chunk_label=label,
            )
            futures[fut] = label

        for fut in as_completed(futures):
            try:
                temp_path, obj_points = fut.result()
            except Exception as e:
                logger.error(f"{futures[fut]} Phase1チャンク処理失敗: {e}")
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if temp_path is not None:
                temp_npz_paths.append(temp_path)
            all_frame_object_points.update(obj_points)

    temp_npz_paths.sort()
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"フェーズ1完了: {len(temp_npz_paths)} 中間ファイル生成")

    if not temp_npz_paths:
        logger.warning("フェーズ1で集約された点群がありません")
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        return stats

    # ====== フェーズ2: グリッド索引による空間プレフィルタ + GPU常駐テンソル ======
    logger.info(f"=== フェーズ2開始: {len(valid_indices)}フレーム × {len(temp_npz_paths)}チャンク ===")

    # 中間npy/npzを読み込み、ワールド座標に変換して1本の配列に統合
    world_list = []
    for temp_path in temp_npz_paths:
        if str(temp_path).endswith('.npy'):
            chunk_np = np.load(str(temp_path)).astype(np.float32)
        else:
            chunk_np = np.load(str(temp_path))['pcd'].astype(np.float32)
        # 基準座標系 → ワールド座標（全フレームで共通、1回だけ実施）
        world_list.append((chunk_np @ base_rot.T + base_trans).astype(np.float32))
        del chunk_np

    world_pts = np.concatenate(world_list, axis=0)
    del world_list

    total_points = world_pts.shape[0]
    ram_mb = world_pts.nbytes / 1024 / 1024
    logger.info(f"中間チャンク全読込み完了（ワールド座標変換済み）: "
                f"合計 {total_points:,} 点 (RAM: ~{ram_mb:.1f} MB)")

    # XYグリッド索引を構築（各フレームで軌跡全体を変換するのを回避）
    world_pts, cell_starts, cell_ends, cell_x, cell_y = _build_xy_grid_index(world_pts)
    logger.info(f"空間グリッド索引構築完了: {cell_starts.shape[0]:,} セル "
                f"(セルサイズ {GRID_CELL_SIZE:.0f}m, 検索半径 {PREFILTER_RADIUS:.1f}m)")

    # GPU使用判定: VRAMに収まるならワールド点群を1回だけGPUへ常駐転送
    world_t = None
    if HAS_TORCH and DEVICE.type == 'cuda':
        try:
            vram_free = torch.cuda.mem_get_info(0)[0]
            # 常駐分 + フレーム毎の作業領域（候補点の変換・マスク）の余裕をみる
            if world_pts.nbytes * 2 < vram_free * 0.8:
                world_t = torch.from_numpy(world_pts).to(DEVICE)
                logger.info(f"ワールド点群をGPUへ常駐転送 (~{ram_mb:.1f} MB)")
            else:
                logger.warning(f"VRAM不足（空き{vram_free/1024/1024:.0f}MB）→ CPUで実行")
        except Exception as e:
            logger.warning(f"GPU常駐転送失敗 → CPUで実行: {e}")
            world_t = None

    for valid_frame_num in tqdm(valid_indices, desc="Phase2 output"):
        frame_idx = valid_frame_num - 1  # 1-indexed → 0-indexed

        if frame_idx not in pcd_files:
            continue

        save_name = f"NCAP_{valid_frame_num:06d}.npz"
        save_path = dense_pcd_dir / save_name

        # 既存チェック（冪等性）
        if save_path.exists():
            stats['skipped'] += 1
            continue

        # このフレームのポーズ
        pcd_info = pcd_files[frame_idx]
        frame_timestamp = pcd_info['timestamp']
        pose_idx = timestamp_idx_mapping[frame_timestamp]
        cur_rot = rotations[pose_idx]
        cur_trans = translations[pose_idx]

        # 空間プレフィルタ: PC_RANGEに入りうる近傍セルのスライスだけを対象にする
        slices = _select_grid_slices(cur_trans[:2], PREFILTER_RADIUS,
                                     cell_starts, cell_ends, cell_x, cell_y)

        # 動的物体点（フレーム座標系のまま保持していたもの）を範囲クリップ
        obj_pts = None
        if frame_idx in all_frame_object_points:
            op = _clip_to_pc_range_np(all_frame_object_points[frame_idx])
            if op.shape[0] > 0:
                obj_pts = op.astype(np.float32)

        results = []
        if slices:
            done = False
            if world_t is not None:
                try:
                    results = _phase2_frame_gpu(world_t, slices, cur_rot, cur_trans, obj_pts)
                    done = True
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                        logger.warning(f"Phase2 GPU OOM (frame {valid_frame_num}) → CPUフォールバック")
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        raise
            if not done:
                results = _phase2_frame_cpu(world_pts, slices, cur_rot, cur_trans, obj_pts)

        if results:
            final_points = np.concatenate(results, axis=0).astype(np.float32)
        else:
            final_points = np.zeros((0, 3), dtype=np.float32)

        np.savez(str(save_path), pcd=final_points[:, :3])
        stats['generated'] += 1

    # GPU解放
    if world_t is not None:
        del world_t
        torch.cuda.empty_cache()
    del world_pts

    # 中間ファイルクリーンアップ
    shutil.rmtree(str(temp_dir), ignore_errors=True)
    logger.info("中間ファイル削除完了")

    # 完了マーカー
    with open(finish_marker, 'w') as f:
        f.write(f"Generated {stats['generated']} dense PCD files\n")

    logger.info(f"高密度点群生成完了: 生成={stats['generated']}, スキップ={stats['skipped']}")
    return stats


def generate_dense_pcd_for_bev_output(bev_output_dir: Path,
                                       bag_dirs: List[Path],
                                       num_threads: int = DEFAULT_NUM_THREADS,
                                       force_rebuild: bool = False) -> Dict[str, int]:
    """BEV出力ディレクトリ配下の各サンプルに対して高密度点群を生成する。

    run_from_csv.py の後段処理として呼び出される想定。

    引数:
        bev_output_dir: BEV出力ディレクトリ (bev_output/)
        bag_dirs: 処理対象のbag_dirリスト
        num_threads: 並列スレッド数
        force_rebuild: True の場合、既存の finish.txt と .npz を削除して再生成

    戻り値:
        全体統計 {'total_generated': N, 'total_skipped': M, 'errors': E}
    """
    total_stats = {'total_generated': 0, 'total_skipped': 0, 'errors': 0}

    for bag_dir in bag_dirs:
        sample_name = bag_dir.name
        sub_folder_name = bag_dir.parent.name

        # relevance.json のパス
        relevance_path = bev_output_dir / sub_folder_name / sample_name / 'relevance.json'
        if not relevance_path.exists():
            logger.warning(f"relevance.json なし（スキップ）: {relevance_path}")
            total_stats['errors'] += 1
            continue

        # 出力先: bev_output/<sub>/<sample>/dense_pcd_npz/
        output_dir = bev_output_dir / sub_folder_name / sample_name
        dense_pcd_dir = output_dir / 'dense_pcd_npz'

        # --force-rebuild: 既存の finish.txt を削除して再生成を強制
        if force_rebuild and dense_pcd_dir.exists():
            finish_marker = dense_pcd_dir / 'finish.txt'
            if finish_marker.exists():
                finish_marker.unlink()
                logger.info(f"--force-rebuild: finish.txt 削除: {finish_marker}")
            # 既存の .npz も削除
            for npz_file in dense_pcd_dir.glob('*.npz'):
                npz_file.unlink()
            logger.info(f"--force-rebuild: 既存 .npz 削除: {dense_pcd_dir}")

        logger.info(f"高密度点群生成: {sub_folder_name}/{sample_name}")

        # result/*.json フォルダを渡し、JSONと1対1対応するnpzを生成する
        result_json_dir = output_dir / 'result'

        result = generate_dense_pcd_for_bag(
            bag_dir=bag_dir,
            relevance_json_path=relevance_path,
            output_dir=output_dir,
            num_threads=num_threads,
            result_json_dir=result_json_dir,
        )

        if result is None:
            total_stats['errors'] += 1
        else:
            total_stats['total_generated'] += result['generated']
            total_stats['total_skipped'] += result['skipped']

    return total_stats


# --- スタンドアロン実行用 ---
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='高密度点群(Dense PCD)生成')
    parser.add_argument('bag_dir', type=str, help='bagフォルダのパス（result/を含む親）')
    parser.add_argument('relevance_json', type=str, help='relevance.jsonのパス')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='出力先（デフォルト: bag_dir/result/）')
    parser.add_argument('--num-threads', type=int, default=DEFAULT_NUM_THREADS,
                        help=f'並列スレッド数（デフォルト: {DEFAULT_NUM_THREADS}）')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    bag_dir = Path(args.bag_dir)
    relevance_path = Path(args.relevance_json)
    output_dir = Path(args.output_dir) if args.output_dir else bag_dir / 'result'

    result = generate_dense_pcd_for_bag(
        bag_dir, relevance_path, output_dir,
        num_threads=args.num_threads
    )
    if result:
        print(f"完了: 生成={result['generated']}, スキップ={result['skipped']}")
    else:
        print("エラーが発生しました")
        sys.exit(1)
