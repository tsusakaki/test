#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IBEV LaneGen Production V7 orchestration.

Stage1 -> Stage2 -> Stage3 with:
- input/cache signature validation
- stage-aware resume
- atomic job-state/progress JSON
- cooperative cancellation
- single-job lock
- unified ASCII/binary/binary_compressed PCD reader options
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lanegen_core as core
import lanegen_stage1_extract as stage1
import lanegen_stage2_lines as stage2
import lanegen_stage3_export as stage3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



# v8 で i_sigma の正規化 (点数バイアス補正・s方向プール・局所スケール) が
# 変わったため、v7 以前の Stage1 NPZ は再利用しない。再利用すると改善が
# 効かないまま古い証拠で走ってしまう。
_STAGE1_CACHE_COMPATIBLE_VERSIONS = {
    "8.1.0-production-v8",
}
_STAGE2_CACHE_COMPATIBLE_VERSIONS = {
    "8.1.0-production-v8",
}

_EXPECTED_SKIP_MESSAGES = {
    "停止Pose除去後の軌跡点が不足しています": (
        "TRAJECTORY_TOO_SHORT",
        "停止Poseを除くと有効な走行軌跡が不足するため、このタスクは処理対象外です",
    ),
    "有効な mapping_pose が 2 点未満です": (
        "MAPPING_POSE_TOO_SHORT",
        "有効なmapping_poseが2点未満のため、このタスクは処理対象外です",
    ),
    "軌跡長が 0 です": (
        "TRAJECTORY_ZERO_LENGTH",
        "走行軌跡の長さが0のため、このタスクは処理対象外です",
    ),
}


def _expected_skip(exc: BaseException) -> tuple[str, str] | None:
    """Classify data that is valid to skip in a continuing production batch."""
    text = str(exc)
    for fragment, payload in _EXPECTED_SKIP_MESSAGES.items():
        if fragment in text:
            return payload
    return None


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _signature_matches(path: Path, expected: dict | None) -> bool:
    if not path.is_file() or not expected:
        return False
    try:
        return core.file_signature(path)["sha256"] == expected.get("sha256")
    except Exception:
        return False


def _parameter_subset_matches(cached: dict, current: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for key, value in current.items():
        if key not in cached:
            errors.append(f"parameter {key}: missing in cache")
            continue
        cached_value = cached.get(key)
        if isinstance(value, float):
            try:
                if abs(float(cached_value) - value) > 1e-12:
                    errors.append(f"parameter {key}: {cached_value} != {value}")
            except Exception:
                errors.append(f"parameter {key}: {cached_value} != {value}")
        elif cached_value != value:
            errors.append(f"parameter {key}: {cached_value} != {value}")
    return not errors, errors


def _stage1_cache_valid(npz1: Path, intensity: Path, mapping: Path,
                        z_pcd: Path | None, rgb_pcd: Path | None,
                        current_parameters: dict) -> tuple[bool, list[str]]:
    meta_path = npz1.with_suffix(".meta.json")
    if not npz1.is_file() or not meta_path.is_file():
        return False, ["Stage1 NPZ/meta missing"]
    meta = _load_json(meta_path)
    expected = meta.get("inputs", {}).get("signatures", {})
    errors = []
    if meta.get("production_version") not in _STAGE1_CACHE_COMPATIBLE_VERSIONS:
        errors.append(
            "production_version: "
            f"{meta.get('production_version')} is not Stage1-compatible"
        )
    params_ok, param_errors = _parameter_subset_matches(
        meta.get("parameters", {}), current_parameters
    )
    if not params_ok:
        errors.extend(param_errors)
    current = {"intensity_pcd": intensity, "mapping_pose": mapping}
    if z_pcd is not None:
        current["z_pcd"] = z_pcd
    if rgb_pcd is not None:
        current["rgb_pcd"] = rgb_pcd
    for key, path in current.items():
        if key not in expected:
            errors.append(f"{key}: cache signature missing")
        elif not _signature_matches(path, expected[key]):
            errors.append(f"{key}: SHA-256 mismatch")
    return not errors, errors


def _stage2_cache_valid(npz2: Path, npz1: Path,
                        current_parameters: dict) -> bool:
    meta_path = npz2.with_suffix(".meta.json")
    if not npz2.is_file() or not meta_path.is_file() or not npz1.is_file():
        return False
    meta = _load_json(meta_path)
    if meta.get("production_version") not in _STAGE2_CACHE_COMPATIBLE_VERSIONS:
        return False
    params_ok, _ = _parameter_subset_matches(
        meta.get("parameters", {}), current_parameters
    )
    expected = meta.get("input_signatures", {}).get("stage1_npz")
    return params_ok and _signature_matches(npz1, expected)


def _stage3_cache_valid(out: Path, stem: str, npz1: Path, npz2: Path,
                        current_parameters: dict) -> bool:
    meta_path = out / f"{stem}_stage3.meta.json"
    required = [
        out / f"{stem}.xml",
        out / f"{stem}_review.xml",
        out / f"{stem}_annotator.json",
        out / f"{stem}_annotator_main.json",
        out / f"{stem}_annotator_review.json",
    ]
    if not meta_path.is_file() or not all(p.is_file() for p in required):
        return False
    meta = _load_json(meta_path)
    if meta.get("production_version") != core.PRODUCTION_VERSION:
        return False
    params_ok, _ = _parameter_subset_matches(
        meta.get("parameters", {}), current_parameters
    )
    sig = meta.get("inputs", {}).get("signatures", {})
    return (params_ok and
            _signature_matches(npz1, sig.get("stage1_npz")) and
            _signature_matches(npz2, sig.get("stage2_npz")))


class JobState:
    def __init__(self, path: Path, progress_path: Path | None,
                 cancel_file: Path | None):
        self.path = path
        self.progress_path = progress_path
        self.cancel_file = cancel_file
        self.data: dict = {}

    def write(self, *, status: str, stage: int | None = None,
              message: str = "", fraction: float | None = None,
              extra: dict | None = None) -> None:
        self.data.update({
            "production_version": core.PRODUCTION_VERSION,
            "status": status,
            "stage": stage,
            "message": message,
            "updated_at_utc": _utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        })
        if fraction is not None:
            self.data["fraction"] = float(max(0.0, min(1.0, fraction)))
        if extra:
            self.data.update(extra)
        core.atomic_dump_json(self.path, self.data)
        if self.progress_path is not None:
            core.atomic_dump_json(self.progress_path, self.data)

    def cancelled(self) -> bool:
        return bool(self.cancel_file and self.cancel_file.exists())


class JobLock:
    def __init__(self, path: Path, stale_hours: float = 24.0):
        self.path = path
        self.stale_hours = max(0.0, float(stale_hours))
        self.fd: int | None = None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def _remove_if_stale(self) -> bool:
        if not self.path.exists():
            return True
        owner = _load_json(self.path)
        host = str(owner.get("host", ""))
        pid = int(owner.get("pid", -1)) if str(owner.get("pid", "")).lstrip("-").isdigit() else -1
        same_host = host == socket.gethostname()
        if same_host and not self._pid_alive(pid):
            self.path.unlink(missing_ok=True)
            return True
        try:
            age_hours = (time.time() - self.path.stat().st_mtime) / 3600.0
        except OSError:
            age_hours = 0.0
        if (not same_host) and self.stale_hours > 0 and age_hours >= self.stale_hours:
            self.path.unlink(missing_ok=True)
            return True
        return False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self._remove_if_stale():
            owner = _load_json(self.path)
            raise RuntimeError(
                f"別のLaneGenジョブが実行中です: {self.path} owner={owner}"
            )
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = _load_json(self.path)
            raise RuntimeError(
                f"別のLaneGenジョブが実行中です: {self.path} owner={owner}"
            ) from exc
        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                              "production_version": core.PRODUCTION_VERSION,
                              "started_at_utc": _utc_now()}, ensure_ascii=False)
        os.write(self.fd, payload.encode("utf-8"))
        os.fsync(self.fd)

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IBEV LaneGen Production V7 個別PCD実行",
        epilog=(
            "SQLiteバッチ: python lanegen_run.py --input tasks.sqlite "
            "--output D:\\sakaki\\BEV_results20260716"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("intensity_pcd")
    p.add_argument("mapping_pose")
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--stem", default=None)
    p.add_argument("--z-pcd", default=None)
    p.add_argument("--rgb-pcd", default=None)

    g = p.add_argument_group("量産ジョブ制御")
    g.add_argument("--resume", action="store_true",
                   help="署名が一致する完了Stageを再利用して途中再開")
    g.add_argument("--skip-stage1", action="store_true",
                   help="既存Stage1を必ず再利用（署名検証あり）")
    g.add_argument("--force-cache", action="store_true",
                   help="入力署名不一致でもStage1キャッシュを強制利用")
    g.add_argument("--state-file", default=None,
                   help="ジョブ状態JSON。省略時は出力dir/stem_job_state.json")
    g.add_argument("--progress-json", default=None,
                   help="GUI監視用進捗JSON。省略時はstate-fileと共用")
    g.add_argument("--cancel-file", default=None,
                   help="存在すると安全な位置でキャンセル")
    g.add_argument("--reset-cancel-file", action="store_true",
                   help="開始時に既存cancel-fileを削除")
    g.add_argument("--lock-file", default=None,
                   help="二重実行防止lock。省略時は出力dir/stem.lock")
    g.add_argument("--stale-lock-hours", type=float, default=24.0,
                   help="別hostのlockを期限切れとみなす時間。same hostの死んだPIDは即時回収")
    g.add_argument("--pcd-chunk-points", type=int, default=500000,
                   help="PCDデコード単位。Stage1の点メモリ上限を制御")
    g.add_argument("--stage1-aggregation-mode", choices=["streaming", "batch"],
                   default="streaming",
                   help="量産はstreaming。batchは回帰比較用")
    g.add_argument("--pcd-temp-dir", default=None,
                   help="binary_compressed展開用一時memmapディレクトリ")
    g.add_argument("--debug-traceback", action="store_true",
                   help="スキップ・失敗時にも詳細なPython Tracebackを表示")

    p.add_argument("--corridor-half-width", type=float, default=15.0)
    p.add_argument("--surface-band", type=float, default=0.12)
    p.add_argument("--surface-low-k", type=int, default=3)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--no-intensity-count-debias",
                   dest="intensity_count_debias", action="store_false",
                   help="セル内点数によるtop-k meanの偏りを補正しない (v7互換)")
    p.set_defaults(intensity_count_debias=True)
    p.add_argument("--no-intensity-weighted-background",
                   dest="intensity_weighted_background", action="store_false",
                   help="annulus背景をセル平均で取る (v7互換)")
    p.set_defaults(intensity_weighted_background=True)
    p.add_argument("--intensity-pool-s", type=float, default=1.25)
    p.add_argument("--intensity-scale-mode", choices=["local", "row"],
                   default="local")
    p.add_argument("--intensity-scale-d", type=float, default=3.0)
    p.add_argument("--intensity-scale-s", type=float, default=25.0)
    p.add_argument("--no-intensity-dark-return",
                   dest="intensity_dark_return", action="store_false",
                   help="路面が観測できているのに反射の返りが無いセルを"
                        "『暗い』観測として扱わない (v7互換)")
    p.set_defaults(intensity_dark_return=True)
    p.add_argument("--intensity-empty-percentile", type=float, default=5.0)
    p.add_argument("--frenet-candidates", type=int, default=12)
    p.add_argument("--frenet-z-weight", type=float, default=1.5)
    p.add_argument("--frenet-heading-weight", type=float, default=0.15)
    p.add_argument("--frenet-ambiguity-margin", type=float, default=0.20)
    p.add_argument("--frenet-ambiguity-s-separation", type=float, default=8.0)
    p.add_argument("--keep-ambiguous-frenet", action="store_true")

    p.add_argument("--sigma-threshold", type=float, default=2.5)
    p.add_argument("--dash-close-gap", type=float, default=1.25)
    p.add_argument("--solid-max-run-ratio", type=float, default=0.6)
    p.add_argument("--dash-regularity", type=float, default=0.6)
    p.add_argument("--dash-on-min", type=float, default=2.0)
    p.add_argument("--dash-off-min", type=float, default=2.0)
    p.add_argument("--rgb-auto-min-coverage", type=float, default=0.75)
    p.add_argument("--rgb-auto-min-on-length", type=float, default=1.5)
    p.add_argument("--rgb-scale-mode", choices=["local", "row"],
                   default="local")
    p.add_argument("--rgb-lane-fusion", choices=["auto", "on", "off"],
                   default="auto")
    p.add_argument("--rgb-auto-min-span-ratio", type=float, default=0.80)
    p.add_argument("--rgb-auto-min-tracks-per-100m", type=float, default=1.0)
    p.add_argument("--rgb-sigma-threshold", type=float, default=3.0)
    p.add_argument("--rgb-strong-sigma-threshold", type=float, default=4.5)
    p.add_argument("--rgb-min-luminance", type=float, default=110.0)
    p.add_argument("--rgb-max-white-saturation", type=float, default=0.22)
    p.add_argument("--rgb-yellow-saturation-min", type=float, default=0.10)
    p.add_argument("--rgb-yellow-chroma-min", type=float, default=12.0)
    p.add_argument("--rgb-min-local-contrast", type=float, default=10.0)
    p.add_argument("--rgb-intensity-support-sigma", type=float, default=0.20)
    p.add_argument("--rgb-intensity-support-d", type=float, default=0.10)
    p.add_argument("--rgb-intensity-support-s", type=float, default=0.50)
    p.add_argument("--rgb-z-edge-dilation", type=float, default=0.05)
    p.add_argument("--rgb-max-paint-width", type=float, default=0.35)
    p.add_argument("--rgb-min-track-length", type=float, default=8.0)
    p.add_argument("--max-gap", type=float, default=90.0)
    p.add_argument("--min-track-length", type=float, default=8.0)
    p.add_argument("--assignment-method", choices=["hungarian", "greedy"], default="hungarian")
    p.add_argument("--curb-assignment-method", choices=["hungarian", "greedy"], default="greedy")
    p.add_argument("--curb-strict-long-gap", action="store_true")
    p.add_argument("--long-gap-threshold", type=float, default=15.0)
    p.add_argument("--long-gap-d-tolerance-scale", type=float, default=0.65)
    p.add_argument("--long-gap-max-slope-diff", type=float, default=0.12)
    p.add_argument("--bridge-crossing-tolerance", type=float, default=0.10)
    p.add_argument("--allow-bridge-crossing", action="store_true")
    p.add_argument("--curb-step-min", type=float, default=0.08)
    p.add_argument("--curb-step-max", type=float, default=0.30)

    p.add_argument("--confidence-threshold", type=float, default=0.55)
    p.add_argument("--review-gap-threshold", type=float, default=30.0)
    p.add_argument("--review-interpolated-ratio", type=float, default=0.70)
    p.add_argument("--review-unknown-gap-threshold", type=float, default=30.0)
    p.add_argument("--review-frenet-ambiguity-ratio", type=float, default=0.03)
    p.add_argument("--review-frenet-ambiguity-run", type=float, default=1.0)
    p.add_argument("--lane-number-style", default="ego", choices=["ego", "left", "none"])
    p.add_argument("--curb-edge", default="gutter", choices=["ridge", "gutter", "top"])
    p.add_argument("--image-meta-json", default=None)
    p.add_argument("--xml-image-name", default=None)
    p.add_argument("--emit-hatch", action="store_true")

    g = p.add_argument_group("出力ポリライン平滑化")
    g.add_argument("--geometry-smoothing",
                   choices=["auto", "line", "spline", "none"],
                   default="auto",
                   help="autoは直線またはロバスト曲線を自動選択")
    g.add_argument("--output-point-spacing", type=float, default=5.0,
                   help="平滑化後ポリラインの最大頂点間隔 [m]")
    g.add_argument("--straight-line-tolerance", type=float, default=0.05,
                   help="直線近似とみなすロバスト横誤差 [m]")
    g.add_argument("--spline-smoothing-tolerance", type=float, default=0.04,
                   help="曲線平滑化の基準誤差 [m]")
    g.add_argument("--curve-approximation-tolerance", type=float, default=0.05,
                   help="曲線を疎なポリラインへ変換する横偏差 [m]")
    g.add_argument("--simplify-tolerance", type=float, default=0.08,
                   help="平滑化なし・fallback時の簡略化許容誤差 [m]")
    g.add_argument("--smoothing-robust-iterations", type=int, default=3,
                   help="外れ値を抑える再重み付け回数")
    g.add_argument("--curb-straight-line-tolerance", type=float, default=0.08,
                   help="縁石を直線近似とみなすロバスト横誤差 [m]")
    g.add_argument("--curb-spline-smoothing-tolerance", type=float, default=0.06,
                   help="縁石曲線平滑化の基準誤差 [m]")
    g.add_argument("--curb-curve-approximation-tolerance", type=float, default=0.08,
                   help="縁石曲線のポリライン化横偏差 [m]")
    return p


def main(argv=None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if any(token == "--input" or token.startswith("--input=") for token in argv_list):
        import lanegen_sqlite_batch
        return lanegen_sqlite_batch.main(argv_list)
    a = build_parser().parse_args(argv_list)
    out = Path(a.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = a.stem or Path(a.intensity_pcd).stem
    npz1 = out / f"{stem}_stage1.npz"
    npz2 = out / f"{stem}_stage2.npz"
    state_path = Path(a.state_file).resolve() if a.state_file else out / f"{stem}_job_state.json"
    progress_path = (Path(a.progress_json).resolve() if a.progress_json
                     else out / f"{stem}_progress.json")
    cancel_file = Path(a.cancel_file).resolve() if a.cancel_file else out / f"{stem}.cancel"
    lock_file = Path(a.lock_file).resolve() if a.lock_file else out / f"{stem}.lock"
    if a.reset_cancel_file:
        cancel_file.unlink(missing_ok=True)

    state = JobState(state_path, progress_path if progress_path != state_path else None, cancel_file)
    lock = JobLock(lock_file, stale_hours=a.stale_lock_hours)
    started = time.time()
    try:
        lock.acquire()
        state.write(status="RUNNING", stage=0, message="job started", fraction=0.0,
                    extra={"started_at_utc": _utc_now(), "output_dir": str(out),
                           "stem": stem, "parameters": vars(a)})
        if state.cancelled():
            state.write(status="CANCELLED", stage=0, message="cancel requested before start")
            return 2

        intensity_path = Path(a.intensity_pcd).resolve()
        mapping_path = Path(a.mapping_pose).resolve()
        z_path = Path(a.z_pcd).resolve() if a.z_pcd else intensity_path
        rgb_path = Path(a.rgb_pcd).resolve() if a.rgb_pcd else None
        stage1_params = {
            "corridor_half_width": a.corridor_half_width,
            "surface_band": a.surface_band,
            "surface_low_k": a.surface_low_k,
            "topk": a.topk,
            "intensity_count_debias": a.intensity_count_debias,
            "intensity_weighted_background": a.intensity_weighted_background,
            "intensity_pool_s": a.intensity_pool_s,
            "intensity_scale_mode": a.intensity_scale_mode,
            "intensity_scale_d": a.intensity_scale_d,
            "intensity_scale_s": a.intensity_scale_s,
            "intensity_dark_return": a.intensity_dark_return,
            "intensity_empty_percentile": a.intensity_empty_percentile,
            "frenet_candidates": a.frenet_candidates,
            "frenet_z_weight": a.frenet_z_weight,
            "frenet_heading_weight": a.frenet_heading_weight,
            "frenet_ambiguity_margin": a.frenet_ambiguity_margin,
            "frenet_ambiguity_s_separation": a.frenet_ambiguity_s_separation,
            "keep_ambiguous_frenet": a.keep_ambiguous_frenet,
        }
        stage2_params = {
            "sigma_threshold": a.sigma_threshold,
            "dash_close_gap": a.dash_close_gap,
            "solid_max_run_ratio": a.solid_max_run_ratio,
            "dash_regularity": a.dash_regularity,
            "dash_on_min": a.dash_on_min,
            "dash_off_min": a.dash_off_min,
            "rgb_auto_min_coverage": a.rgb_auto_min_coverage,
            "rgb_auto_min_on_length": a.rgb_auto_min_on_length,
            "rgb_scale_mode": a.rgb_scale_mode,
            "rgb_lane_fusion": a.rgb_lane_fusion,
            "rgb_auto_min_span_ratio": a.rgb_auto_min_span_ratio,
            "rgb_auto_min_tracks_per_100m": a.rgb_auto_min_tracks_per_100m,
            "rgb_sigma_threshold": a.rgb_sigma_threshold,
            "rgb_strong_sigma_threshold": a.rgb_strong_sigma_threshold,
            "rgb_min_luminance": a.rgb_min_luminance,
            "rgb_max_white_saturation": a.rgb_max_white_saturation,
            "rgb_yellow_saturation_min": a.rgb_yellow_saturation_min,
            "rgb_yellow_chroma_min": a.rgb_yellow_chroma_min,
            "rgb_min_local_contrast": a.rgb_min_local_contrast,
            "rgb_intensity_support_sigma": a.rgb_intensity_support_sigma,
            "rgb_intensity_support_d": a.rgb_intensity_support_d,
            "rgb_intensity_support_s": a.rgb_intensity_support_s,
            "rgb_z_edge_dilation": a.rgb_z_edge_dilation,
            "rgb_max_paint_width": a.rgb_max_paint_width,
            "rgb_min_track_length": a.rgb_min_track_length,
            "max_gap": a.max_gap,
            "min_track_length": a.min_track_length,
            "assignment_method": a.assignment_method,
            "curb_assignment_method": a.curb_assignment_method,
            "curb_strict_long_gap": a.curb_strict_long_gap,
            "long_gap_threshold": a.long_gap_threshold,
            "long_gap_d_tolerance_scale": a.long_gap_d_tolerance_scale,
            "long_gap_max_slope_diff": a.long_gap_max_slope_diff,
            "bridge_crossing_tolerance": a.bridge_crossing_tolerance,
            "allow_bridge_crossing": a.allow_bridge_crossing,
            "curb_step_min": a.curb_step_min,
            "curb_step_max": a.curb_step_max,
        }
        stage3_params = {
            "confidence_threshold": a.confidence_threshold,
            "review_gap_threshold": a.review_gap_threshold,
            "review_interpolated_ratio": a.review_interpolated_ratio,
            "review_unknown_gap_threshold": a.review_unknown_gap_threshold,
            "review_frenet_ambiguity_ratio": a.review_frenet_ambiguity_ratio,
            "review_frenet_ambiguity_run": a.review_frenet_ambiguity_run,
            "lane_number_style": a.lane_number_style,
            "curb_edge": a.curb_edge,
            "image_meta_json": a.image_meta_json,
            "xml_image_name": a.xml_image_name,
            "emit_hatch": a.emit_hatch,
            "geometry_smoothing": a.geometry_smoothing,
            "output_point_spacing": a.output_point_spacing,
            "straight_line_tolerance": a.straight_line_tolerance,
            "spline_smoothing_tolerance": a.spline_smoothing_tolerance,
            "curve_approximation_tolerance": a.curve_approximation_tolerance,
            "simplify_tolerance": a.simplify_tolerance,
            "smoothing_robust_iterations": a.smoothing_robust_iterations,
            "curb_straight_line_tolerance": a.curb_straight_line_tolerance,
            "curb_spline_smoothing_tolerance": a.curb_spline_smoothing_tolerance,
            "curb_curve_approximation_tolerance": a.curb_curve_approximation_tolerance,
        }
        stage1_valid, stage1_errors = _stage1_cache_valid(
            npz1, intensity_path, mapping_path, z_path, rgb_path,
            stage1_params)
        reuse_stage1 = a.skip_stage1 or (a.resume and stage1_valid)
        if a.skip_stage1 and not stage1_valid and not a.force_cache:
            raise RuntimeError("Stage1 cache invalid: " + "; ".join(stage1_errors))

        if not reuse_stage1:
            state.write(status="RUNNING", stage=1, message="Stage1 extracting PCD evidence", fraction=0.02)
            argv1 = [a.intensity_pcd, a.mapping_pose, "-o", str(npz1),
                     "--corridor-half-width", str(a.corridor_half_width),
                     "--surface-band", str(a.surface_band),
                     "--surface-low-k", str(a.surface_low_k),
                     "--topk", str(a.topk),
                     "--intensity-pool-s", str(a.intensity_pool_s),
                     "--intensity-scale-mode", a.intensity_scale_mode,
                     "--intensity-scale-d", str(a.intensity_scale_d),
                     "--intensity-scale-s", str(a.intensity_scale_s),
                     "--intensity-empty-percentile",
                     str(a.intensity_empty_percentile),
                     "--frenet-candidates", str(a.frenet_candidates),
                     "--frenet-z-weight", str(a.frenet_z_weight),
                     "--frenet-heading-weight", str(a.frenet_heading_weight),
                     "--frenet-ambiguity-margin", str(a.frenet_ambiguity_margin),
                     "--frenet-ambiguity-s-separation", str(a.frenet_ambiguity_s_separation),
                     "--pcd-chunk-points", str(a.pcd_chunk_points),
                     "--aggregation-mode", a.stage1_aggregation_mode,
                     "--progress-json", str(progress_path),
                     "--progress-base", "0.02",
                     "--progress-span", "0.53",
                     "--cancel-file", str(cancel_file)]
            if a.pcd_temp_dir:
                argv1 += ["--pcd-temp-dir", a.pcd_temp_dir]
            if a.keep_ambiguous_frenet:
                argv1.append("--keep-ambiguous-frenet")
            if not a.intensity_count_debias:
                argv1.append("--no-intensity-count-debias")
            if not a.intensity_weighted_background:
                argv1.append("--no-intensity-weighted-background")
            if not a.intensity_dark_return:
                argv1.append("--no-intensity-dark-return")
            if a.z_pcd:
                argv1 += ["--z-pcd", a.z_pcd]
            if a.rgb_pcd:
                argv1 += ["--rgb-pcd", a.rgb_pcd]
            stage1.run(stage1.parse_args(argv1))
        else:
            message = "Stage1 reused"
            if not stage1_valid and a.force_cache:
                message += " with --force-cache"
            state.write(status="RUNNING", stage=1, message=message, fraction=0.55)

        if state.cancelled():
            state.write(status="CANCELLED", stage=1, message="cancelled after Stage1")
            return 2

        reuse_stage2 = bool(
            a.resume and _stage2_cache_valid(npz2, npz1, stage2_params)
        )
        if not reuse_stage2:
            state.write(status="RUNNING", stage=2, message="Stage2 linking lines", fraction=0.60)
            argv2 = [str(npz1), "-o", str(npz2),
                     "--sigma-threshold", str(a.sigma_threshold),
                     "--dash-close-gap", str(a.dash_close_gap),
                     "--solid-max-run-ratio", str(a.solid_max_run_ratio),
                     "--dash-regularity", str(a.dash_regularity),
                     "--dash-on-min", str(a.dash_on_min),
                     "--dash-off-min", str(a.dash_off_min),
                     "--rgb-auto-min-coverage", str(a.rgb_auto_min_coverage),
                     "--rgb-auto-min-on-length",
                     str(a.rgb_auto_min_on_length),
                     "--rgb-scale-mode", a.rgb_scale_mode,
                     "--rgb-lane-fusion", a.rgb_lane_fusion,
                     "--rgb-auto-min-span-ratio", str(a.rgb_auto_min_span_ratio),
                     "--rgb-auto-min-tracks-per-100m",
                     str(a.rgb_auto_min_tracks_per_100m),
                     "--rgb-sigma-threshold", str(a.rgb_sigma_threshold),
                     "--rgb-strong-sigma-threshold",
                     str(a.rgb_strong_sigma_threshold),
                     "--rgb-min-luminance", str(a.rgb_min_luminance),
                     "--rgb-max-white-saturation",
                     str(a.rgb_max_white_saturation),
                     "--rgb-yellow-saturation-min",
                     str(a.rgb_yellow_saturation_min),
                     "--rgb-yellow-chroma-min",
                     str(a.rgb_yellow_chroma_min),
                     "--rgb-min-local-contrast",
                     str(a.rgb_min_local_contrast),
                     "--rgb-intensity-support-sigma",
                     str(a.rgb_intensity_support_sigma),
                     "--rgb-intensity-support-d",
                     str(a.rgb_intensity_support_d),
                     "--rgb-intensity-support-s",
                     str(a.rgb_intensity_support_s),
                     "--rgb-z-edge-dilation", str(a.rgb_z_edge_dilation),
                     "--rgb-max-paint-width", str(a.rgb_max_paint_width),
                     "--rgb-min-track-length", str(a.rgb_min_track_length),
                     "--max-gap", str(a.max_gap),
                     "--min-track-length", str(a.min_track_length),
                     "--assignment-method", a.assignment_method,
                     "--curb-assignment-method", a.curb_assignment_method,
                     "--long-gap-threshold", str(a.long_gap_threshold),
                     "--long-gap-d-tolerance-scale", str(a.long_gap_d_tolerance_scale),
                     "--long-gap-max-slope-diff", str(a.long_gap_max_slope_diff),
                     "--bridge-crossing-tolerance", str(a.bridge_crossing_tolerance),
                     "--curb-step-min", str(a.curb_step_min),
                     "--curb-step-max", str(a.curb_step_max)]
            if a.allow_bridge_crossing:
                argv2.append("--allow-bridge-crossing")
            if a.curb_strict_long_gap:
                argv2.append("--curb-strict-long-gap")
            stage2.run(stage2.parse_args(argv2))
        else:
            state.write(status="RUNNING", stage=2, message="Stage2 reused", fraction=0.78)

        if state.cancelled():
            state.write(status="CANCELLED", stage=2, message="cancelled after Stage2")
            return 2

        reuse_stage3 = bool(
            a.resume and _stage3_cache_valid(
                out, stem, npz1, npz2, stage3_params
            )
        )
        if not reuse_stage3:
            state.write(status="RUNNING", stage=3, message="Stage3 exporting annotations", fraction=0.82)
            argv3 = [str(npz1), str(npz2), "-o", str(out), "--stem", stem,
                     "--confidence-threshold", str(a.confidence_threshold),
                     "--review-gap-threshold", str(a.review_gap_threshold),
                     "--review-interpolated-ratio", str(a.review_interpolated_ratio),
                     "--review-unknown-gap-threshold", str(a.review_unknown_gap_threshold),
                     "--review-frenet-ambiguity-ratio", str(a.review_frenet_ambiguity_ratio),
                     "--review-frenet-ambiguity-run", str(a.review_frenet_ambiguity_run),
                     "--lane-number-style", a.lane_number_style,
                     "--curb-edge", a.curb_edge,
                     "--geometry-smoothing", a.geometry_smoothing,
                     "--output-point-spacing", str(a.output_point_spacing),
                     "--straight-line-tolerance", str(a.straight_line_tolerance),
                     "--spline-smoothing-tolerance", str(a.spline_smoothing_tolerance),
                     "--curve-approximation-tolerance", str(a.curve_approximation_tolerance),
                     "--simplify-tolerance", str(a.simplify_tolerance),
                     "--smoothing-robust-iterations",
                     str(a.smoothing_robust_iterations),
                     "--curb-straight-line-tolerance",
                     str(a.curb_straight_line_tolerance),
                     "--curb-spline-smoothing-tolerance",
                     str(a.curb_spline_smoothing_tolerance),
                     "--curb-curve-approximation-tolerance",
                     str(a.curb_curve_approximation_tolerance)]
            if a.image_meta_json:
                argv3 += ["--image-meta-json", a.image_meta_json]
            if a.xml_image_name:
                argv3 += ["--xml-image-name", a.xml_image_name]
            if a.emit_hatch:
                argv3.append("--emit-hatch")
            stage3.run(stage3.parse_args(argv3))
        else:
            state.write(status="RUNNING", stage=3, message="Stage3 reused", fraction=0.98)

        state.write(status="COMPLETED", stage=3, message="all stages completed", fraction=1.0,
                    extra={"elapsed_sec": round(time.time() - started, 2),
                           "outputs": {"stage1": str(npz1), "stage2": str(npz2),
                                       "annotator": str(out / f"{stem}_annotator.json"),
                                       "xml": str(out / f"{stem}.xml")}})
        return 0
    except core.PcdReadCancelled as exc:
        state.write(status="CANCELLED", stage=1, message=str(exc),
                    extra={"elapsed_sec": round(time.time() - started, 2)})
        return 2
    except Exception as exc:
        skip = _expected_skip(exc)
        if skip is not None:
            reason_code, friendly_message = skip
            state.write(
                status="SKIPPED",
                stage=1,
                message=friendly_message,
                extra={
                    "elapsed_sec": round(time.time() - started, 2),
                    "skip_reason": reason_code,
                    "technical_detail": str(exc),
                },
            )
            batch_mode = os.environ.get("LANEGEN_BATCH_MODE") == "1"
            if not batch_mode:
                print(f"[skip] {friendly_message}", file=sys.stderr)
            if a.debug_traceback:
                traceback.print_exc()
            return 3

        trace = traceback.format_exc()
        state.write(
            status="FAILED",
            message=str(exc),
            extra={
                "elapsed_sec": round(time.time() - started, 2),
                "traceback": trace,
            },
        )
        batch_mode = os.environ.get("LANEGEN_BATCH_MODE") == "1"
        if not batch_mode:
            print(f"[failed] 予期しない処理失敗: {exc}", file=sys.stderr)
        if a.debug_traceback:
            traceback.print_exc()
        elif not batch_mode:
            print(
                "[hint] 詳細なPython Tracebackを表示する場合は --debug-traceback を追加してください。",
                file=sys.stderr,
            )
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
