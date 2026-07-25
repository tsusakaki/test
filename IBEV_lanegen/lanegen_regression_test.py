#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production V6 回帰試験。

合成PCD生成 -> Stage1/2/3 -> XML評価 -> Annotator JSON検証 ->
Stage1キャッシュ再利用時の決定性確認までを一括実行する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import time
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[test]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _merge_cvat_xml(main_path: Path, review_path: Path,
                    dest: Path) -> Path:
    """本体XMLとreview XMLのshapeを1本にまとめた評価用XMLを書く。"""
    tree = ET.parse(main_path)
    root = tree.getroot()
    image = root.find("image")
    if image is not None and Path(review_path).exists():
        for review_image in ET.parse(review_path).getroot().findall("image"):
            for shape in list(review_image):
                image.append(shape)
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    return dest


def _validate_annotator_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise AssertionError("Annotator JSON が空です")

    allowed_line_type = {"solid", "dash", "fish_bone", "dash_solid",
                         "unknown", "wide_dash", "virtual"}
    allowed_quality = {"ok", "uncertain", "review_required", "ignore"}
    allowed_source = {"manual", "auto", "auto_corrected", "imported"}

    ids = set()
    auto_ids = set()
    for rec in data:
        if rec.get("geometry_type") != "polyline":
            raise AssertionError(f"geometry_type不正: {rec.get('id')}")
        points = rec.get("points", [])
        if len(points) < 2:
            raise AssertionError(f"頂点不足: {rec.get('id')}")
        if rec.get("id") in ids:
            raise AssertionError(f"ID重複: {rec.get('id')}")
        ids.add(rec.get("id"))
        aid = rec.get("auto_track_id")
        if not aid or aid in auto_ids:
            raise AssertionError(f"auto_track_id不正/重複: {aid}")
        auto_ids.add(aid)
        if rec.get("quality") not in allowed_quality:
            raise AssertionError(f"quality不正: {rec.get('quality')}")
        if rec.get("source") not in allowed_source:
            raise AssertionError(f"source不正: {rec.get('source')}")
        if rec.get("class") == "lane_line":
            if rec.get("line_type") not in allowed_line_type:
                raise AssertionError(f"line_type不正: {rec.get('line_type')}")
        generation = rec.get("generation", {})
        for key in ("detection_support", "review_priority", "observed_ratio",
                    "interpolated_ratio", "max_gap_m",
                    "frenet_ambiguity_ratio",
                    "max_frenet_ambiguity_run_m", "geometry_fit"):
            if key not in generation:
                raise AssertionError(f"generation.{key} がありません")
        fit = generation.get("geometry_fit", {})
        if fit.get("mode") not in {"line", "spline", "none", "rdp_fallback"}:
            raise AssertionError(f"geometry_fit.mode不正: {fit}")
        if int(fit.get("vertices_after", 0)) != len(points):
            raise AssertionError(f"geometry_fit頂点数不整合: {rec.get('id')}")
    return {"records": len(data), "ids_unique": len(ids),
            "auto_ids_unique": len(auto_ids)}



def _validate_polyline_smoothing(here: Path) -> dict:
    sys.path.insert(0, str(here))
    import lanegen_core as core

    rng = np.random.default_rng(20260725)
    x = np.linspace(0.0, 100.0, 401)
    straight = np.column_stack([
        x,
        1.5 + rng.normal(0.0, 0.02, len(x)),
    ])
    curved = np.column_stack([
        x,
        0.0025 * (x - 50.0) ** 2 + rng.normal(0.0, 0.02, len(x)),
    ])

    straight_out, straight_meta = core.smooth_polyline_xy(
        straight, mode="auto", straight_line_tolerance=0.05,
        spline_smoothing_tolerance=0.04,
        approximation_tolerance=0.05, max_vertex_spacing=5.0)
    curved_out, curved_meta = core.smooth_polyline_xy(
        curved, mode="auto", straight_line_tolerance=0.05,
        spline_smoothing_tolerance=0.04,
        approximation_tolerance=0.05, max_vertex_spacing=5.0)

    if straight_meta.get("mode") != "line":
        raise AssertionError(f"直線がline判定されません: {straight_meta}")
    if curved_meta.get("mode") != "spline":
        raise AssertionError(f"曲線がspline判定されません: {curved_meta}")
    if len(straight_out) > 25 or len(curved_out) > 50:
        raise AssertionError("頂点削減が不十分です")
    if straight_meta.get("fit_p95_m", 1.0) > 0.06:
        raise AssertionError(f"直線近似誤差が大きすぎます: {straight_meta}")
    if curved_meta.get("fit_p95_m", 1.0) > 0.08:
        raise AssertionError(f"曲線近似誤差が大きすぎます: {curved_meta}")
    return {
        "straight": straight_meta,
        "curve": curved_meta,
        "straight_vertices": len(straight_out),
        "curve_vertices": len(curved_out),
    }


def _validate_frenet_branch_ambiguity(here: Path) -> dict:
    sys.path.insert(0, str(here))
    import lanegen_core as core

    # sが遠く離れた2本の軌跡枝が原点で交差する人工ケース。
    x = np.linspace(-5.0, 5.0, 41)
    branch1 = np.column_stack([x, np.zeros_like(x)])
    filler = np.column_stack([np.linspace(50.0, 70.0, 20),
                              np.full(20, 50.0)])
    y = np.linspace(-5.0, 5.0, 41)
    branch2 = np.column_stack([np.zeros_like(y), y])
    xy = np.vstack([branch1, filler, branch2])
    n = len(xy)
    tangent = np.tile(np.array([[1.0, 0.0]]), (n, 1))
    normal = np.tile(np.array([[0.0, 1.0]]), (n, 1))
    s = np.arange(n, dtype=np.float64) * 0.25
    traj = core.Trajectory(
        xy=xy, z=np.zeros(n), tangent=tangent, normal=normal,
        s=s, resolution=0.25, yaw=np.zeros(n),
        geometric_yaw=np.zeros(n), yaw_valid=False,
        heading_error=np.zeros(n), raw_count=n, filtered_count=n)

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    proj = core.project_to_frenet(
        points, traj, half_width=10.0, candidates=16,
        ambiguity_margin=0.20, ambiguity_s_separation=5.0,
        drop_ambiguous=True)
    if not bool(proj.ambiguous_input[0]) or bool(proj.keep[0]):
        raise AssertionError("交差点のFrenet曖昧点が除外されません")
    if bool(proj.ambiguous_input[1]) or not bool(proj.keep[1]):
        raise AssertionError("非曖昧点を誤って除外しました")
    return {"crossing_ambiguous_dropped": True,
            "ordinary_point_kept": True}


def _validate_long_gap_bridge_guard(here: Path) -> dict:
    sys.path.insert(0, str(here))
    import lanegen_stage2_lines as stage2

    def seg(s0, s1, d0, d1):
        ss = np.arange(s0, s1 + 1, dtype=np.int64)
        dd = np.linspace(d0, d1, len(ss))
        return stage2.Segment(ss, dd, np.ones(len(ss)))

    # 0→1の長距離橋が、途中のsegment 2を横切る。
    segments = [seg(0, 10, -1.0, -1.0),
                seg(30, 40, 1.0, 1.0),
                seg(15, 25, 0.0, 0.0)]
    a = SimpleNamespace(
        max_gap=90.0, d_tolerance=2.5, d_tolerance_rate=0.0,
        max_slope_diff=1.0, gap_penalty=0.08,
        long_gap_threshold=15.0, long_gap_d_tolerance_scale=1.0,
        long_gap_max_slope_diff=1.0, allow_bridge_crossing=False,
        bridge_crossing_tolerance=0.10)
    edges, stats = stage2.build_edges(segments, 1.0, a)
    if any(i == 0 and j == 1 for i, j, _ in edges):
        raise AssertionError("他線を横切る長距離接続が許可されました")
    if stats["rejected_bridge_crossing"] < 1:
        raise AssertionError("bridge crossing拒否が記録されません")
    return {"crossing_bridge_rejected": True,
            "rejected_count": stats["rejected_bridge_crossing"]}


def _lzf_literal_encode(raw: bytes) -> bytes:
    """Valid LZF stream using literal blocks only (test fixture writer)."""
    out = bytearray()
    for start in range(0, len(raw), 32):
        block = raw[start:start + 32]
        out.append(len(block) - 1)
        out.extend(block)
    return bytes(out)


def _write_pcd_fixture(path: Path, mode: str, values: np.ndarray) -> None:
    n = len(values)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb intensity\n"
        "SIZE 4 4 4 4 4\n"
        "TYPE F F F F F\n"
        "COUNT 1 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA {mode}\n"
    ).encode("ascii")
    cols = [values[:, i].astype("<f4") for i in range(5)]
    with path.open("wb") as f:
        f.write(header)
        if mode == "ascii":
            np.savetxt(f, values, fmt="%.9g")
        elif mode == "binary":
            f.write(np.ascontiguousarray(values.astype("<f4")).tobytes())
        elif mode == "binary_compressed":
            raw = b"".join(np.ascontiguousarray(c).tobytes() for c in cols)
            compressed = _lzf_literal_encode(raw)
            f.write(struct.pack("<II", len(compressed), len(raw)))
            f.write(compressed)
        else:
            raise ValueError(mode)


def _validate_pcd_modes(here: Path, work: Path) -> dict:
    sys.path.insert(0, str(here))
    import lanegen_core as core

    rng = np.random.default_rng(20260725)
    n = 257
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    # packed RGB bits stored in float32, as in common PCD files
    rgb_u = rng.integers(0, 2**24, size=n, dtype=np.uint32)
    rgb_f = np.ascontiguousarray(rgb_u).view(np.float32)
    intensity = rng.uniform(0, 255, size=n).astype(np.float32)
    values = np.column_stack([xyz, rgb_f, intensity]).astype(np.float32)

    results = {}
    baseline = None
    for mode in ("ascii", "binary", "binary_compressed"):
        path = work / f"fixture_{mode}.pcd"
        _write_pcd_fixture(path, mode, values)
        got_xyz, got_rgb, got_i, info = core.load_pcd(path, chunk_points=37)
        if baseline is None:
            baseline = (got_xyz, got_rgb, got_i)
        else:
            if not np.allclose(got_xyz, baseline[0], atol=1e-6):
                raise AssertionError(f"PCD XYZ mismatch: {mode}")
            if not np.array_equal(np.ascontiguousarray(got_rgb).view(np.uint32),
                                  np.ascontiguousarray(baseline[1]).view(np.uint32)):
                raise AssertionError(f"PCD RGB bits mismatch: {mode}")
            if not np.allclose(got_i, baseline[2], atol=1e-5):
                raise AssertionError(f"PCD intensity mismatch: {mode}")
        results[mode] = {"points": len(got_xyz), "data_type": info["data_type"]}
    return results


def _validate_cancel_before_start(here: Path, synth: Path, work: Path) -> dict:
    cancel = work / "cancel.request"
    cancel.write_text("cancel", encoding="utf-8")
    out = work / "cancel_out"
    state = work / "cancel_state.json"
    cmd = [
        sys.executable, "lanegen_run.py",
        str(synth / "IBEV_synth.pcd"),
        str(synth / "mapping_pose_synth.txt"),
        "-o", str(out), "--z-pcd", str(synth / "RGBBEV_synth.pcd"),
        "--cancel-file", str(cancel), "--state-file", str(state),
    ]
    proc = subprocess.run(cmd, cwd=str(here), check=False)
    if proc.returncode != 2:
        raise AssertionError(f"cancel return code mismatch: {proc.returncode}")
    payload = json.loads(state.read_text(encoding="utf-8"))
    if payload.get("status") != "CANCELLED":
        raise AssertionError(f"cancel state mismatch: {payload}")
    return {"return_code": proc.returncode, "status": payload.get("status")}


def _validate_streaming_topk(here: Path) -> dict:
    sys.path.insert(0, str(here))
    import lanegen_core as core
    from lanegen_streaming import StreamingTopK

    rng = np.random.default_rng(42)
    ncell = 137
    flat = rng.integers(0, ncell, size=5000, dtype=np.int64)
    values = rng.normal(size=5000).astype(np.float32)
    result = {}
    for largest in (True, False):
        acc = StreamingTopK(ncell, 3, largest=largest)
        for start in range(0, len(flat), 113):
            acc.update(flat[start:start + 113], values[start:start + 113])
        got_mean, got_count = acc.mean_and_count()
        exp_mean, exp_count = core.groupwise_topk_mean(
            flat, values, ncell, k=3, largest=largest
        )
        if not np.allclose(got_mean, exp_mean, atol=1e-6):
            raise AssertionError(f"StreamingTopK mean mismatch largest={largest}")
        if not np.array_equal(got_count, exp_count):
            raise AssertionError(f"StreamingTopK count mismatch largest={largest}")
        result["largest" if largest else "smallest"] = True
    return result


def _validate_compressed_reader_reuse(here: Path, work: Path) -> dict:
    sys.path.insert(0, str(here))
    from lanegen_pcd_io import PcdChunkReader

    rng = np.random.default_rng(77)
    n = 509
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    rgb_u = rng.integers(0, 2**24, size=n, dtype=np.uint32)
    rgb_f = np.ascontiguousarray(rgb_u).view(np.float32)
    intensity = rng.uniform(0, 255, size=n).astype(np.float32)
    values = np.column_stack([xyz, rgb_f, intensity]).astype(np.float32)
    path = work / "reuse_binary_compressed.pcd"
    _write_pcd_fixture(path, "binary_compressed", values)
    temp_dir = work / "decompressed_temp"
    with PcdChunkReader(path, chunk_points=41, temp_dir=temp_dir) as reader:
        temp_path = reader._temp_path
        if temp_path is None or not temp_path.is_file():
            raise AssertionError("binary_compressed temp memmap was not created")
        passes = []
        for _ in range(2):
            parts = list(reader.iter_chunks(phase="reuse_test"))
            passes.append(np.concatenate([x[0] for x in parts]))
        if not np.allclose(passes[0], passes[1], atol=1e-7):
            raise AssertionError("reused decompressed memmap produced different data")
    if temp_path.exists():
        raise AssertionError("binary_compressed temp memmap was not cleaned up")
    return {"passes": 2, "temp_cleaned": True, "points": n}


def _validate_streaming_batch_equivalence(
    here: Path, work: Path
) -> dict:
    """Compare Stage1 modes on a compact fixture to keep regression fast."""
    mini = work / "mini_equivalence_synth"
    _run([
        sys.executable, "make_synthetic_ibev.py", "-o", str(mini),
        "--length", "30", "--half-width", "8", "--density", "45",
        "--seed", "20260726",
    ], here)
    streaming_npz = work / "mini_streaming_stage1.npz"
    batch_npz = work / "mini_batch_stage1.npz"
    common = [
        str(mini / "IBEV_synth.pcd"),
        str(mini / "mapping_pose_synth.txt"),
        "--z-pcd", str(mini / "RGBBEV_synth.pcd"),
    ]
    _run([
        sys.executable, "lanegen_stage1_extract.py", *common,
        "-o", str(streaming_npz), "--aggregation-mode", "streaming",
        "--pcd-chunk-points", "7000",
    ], here)
    _run([
        sys.executable, "lanegen_stage1_extract.py", *common,
        "-o", str(batch_npz), "--aggregation-mode", "batch",
    ], here)
    a = np.load(streaming_npz)
    b = np.load(batch_npz)
    keys = [
        "occupied", "count", "i_topk", "i_excess", "i_sigma",
        "z_surface", "z_level_diff", "dz_dd", "rgb_mean",
    ]
    result = {}
    for key in keys:
        if key not in a or key not in b:
            raise AssertionError(f"comparison key missing: {key}")
        if a[key].dtype == bool or np.issubdtype(a[key].dtype, np.integer):
            equal = np.array_equal(a[key], b[key])
            max_diff = 0.0 if equal else float(np.max(np.abs(a[key].astype(float) - b[key].astype(float))))
        else:
            max_diff = float(np.max(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))))
            equal = bool(np.allclose(a[key], b[key], atol=1e-5, rtol=1e-5))
        if not equal:
            raise AssertionError(f"streaming/batch mismatch {key}: max_diff={max_diff}")
        result[key] = {"equal": True, "max_abs_diff": max_diff}
    meta = json.loads(streaming_npz.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if meta.get("aggregation", {}).get("full_point_arrays_materialized") is not False:
        raise AssertionError("streaming meta does not confirm bounded point memory")
    return result


def _validate_parameter_aware_resume(here: Path, synth: Path, work: Path) -> dict:
    out = work / "parameter_resume_out"
    base = [
        sys.executable, "lanegen_run.py",
        str(synth / "IBEV_synth.pcd"),
        str(synth / "mapping_pose_synth.txt"),
        "-o", str(out),
        "--z-pcd", str(synth / "RGBBEV_synth.pcd"),
        "--stem", "IBEV",
    ]
    _run(base, here)
    s1_before = (out / "IBEV_stage1.npz").stat().st_mtime_ns
    s2_before = (out / "IBEV_stage2.npz").stat().st_mtime_ns
    time.sleep(0.02)
    _run(base + ["--resume", "--sigma-threshold", "2.6"], here)
    s1_after = (out / "IBEV_stage1.npz").stat().st_mtime_ns
    s2_after = (out / "IBEV_stage2.npz").stat().st_mtime_ns
    if s1_after != s1_before:
        raise AssertionError("Stage2 parameter change incorrectly invalidated Stage1")
    if s2_after == s2_before:
        raise AssertionError("Stage2 parameter change did not invalidate Stage2 cache")
    return {"stage1_reused": True, "stage2_invalidated": True}


def _validate_stale_lock_recovery(here: Path, work: Path) -> dict:
    sys.path.insert(0, str(here))
    import socket
    from lanegen_run import JobLock

    path = work / "stale.lock"
    path.write_text(json.dumps({
        "pid": 99999999,
        "host": socket.gethostname(),
        "started_at_utc": "2000-01-01T00:00:00+00:00",
    }), encoding="utf-8")
    lock = JobLock(path, stale_hours=24)
    lock.acquire()
    if not path.is_file():
        raise AssertionError("recovered lock was not acquired")
    lock.release()
    if path.exists():
        raise AssertionError("lock was not released")
    return {"dead_same_host_pid_recovered": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--work-dir", default="_regression_work")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    work = Path(args.work_dir).expanduser().resolve()
    if work.exists():
        shutil.rmtree(work)
    synth = work / "synth"
    out = work / "out"
    work.mkdir(parents=True)

    pcd_modes = _validate_pcd_modes(here, work)
    streaming_topk_test = _validate_streaming_topk(here)
    compressed_reuse_test = _validate_compressed_reader_reuse(here, work)
    stale_lock_test = _validate_stale_lock_recovery(here, work)

    py = sys.executable
    _run([py, "make_synthetic_ibev.py", "-o", str(synth)], here)
    run_cmd = [
        py, "lanegen_run.py",
        str(synth / "IBEV_synth.pcd"),
        str(synth / "mapping_pose_synth.txt"),
        "-o", str(out),
        "--z-pcd", str(synth / "RGBBEV_synth.pcd"),
        "--stem", "IBEV",
    ]
    _run(run_cmd, here)
    streaming_batch_test = _validate_streaming_batch_equivalence(
        here, work
    )
    _run([py, "lanegen_evaluate.py", str(out / "IBEV.xml"),
          str(synth / "truth_gt.xml"), "-o", str(out / "IBEV_eval.json")], here)
    # 低信頼のラインは捨てず review XML へ回す設計なので、検出できたか
    # (recall) は本体 + review の和で測る。本体の純度 (precision) だけを
    # 本体 XML で測る。
    detected = _merge_cvat_xml(out / "IBEV.xml", out / "IBEV_review.xml",
                               out / "IBEV_detected.xml")
    _run([py, "lanegen_evaluate.py", str(detected),
          str(synth / "truth_gt.xml"),
          "-o", str(out / "IBEV_eval_detected.json")], here)

    metrics = json.loads((out / "IBEV_eval.json").read_text(encoding="utf-8"))
    detected_metrics = json.loads(
        (out / "IBEV_eval_detected.json").read_text(encoding="utf-8"))
    lane = metrics["labels"]["lane_line"]
    curb = metrics["labels"]["curb_boundary"]
    lane_detected = detected_metrics["labels"]["lane_line"]
    curb_detected = detected_metrics["labels"]["curb_boundary"]
    if lane_detected["recall"] < 0.90 or lane["precision"] < 0.95:
        raise AssertionError(
            f"lane回帰失敗: main={lane} detected={lane_detected}")
    if curb_detected["recall"] < 0.80 or curb["precision"] < 0.80:
        raise AssertionError(
            f"curb回帰失敗: main={curb} detected={curb_detected}")
    # review が受け皿になりすぎていないかも見る。
    if lane["recall"] < 0.70:
        raise AssertionError(
            f"本体XMLのlane recallが低すぎる (reviewへの流出過多): {lane}")

    schema = _validate_annotator_json(out / "IBEV_annotator.json")
    hash_before = {
        "xml": _sha256(out / "IBEV.xml"),
        "annotator": _sha256(out / "IBEV_annotator.json"),
    }
    _run(run_cmd + ["--skip-stage1"], here)
    hash_after = {
        "xml": _sha256(out / "IBEV.xml"),
        "annotator": _sha256(out / "IBEV_annotator.json"),
    }
    if hash_before != hash_after:
        raise AssertionError(f"再実行結果が非決定的です: {hash_before} != {hash_after}")

    stage1_mtime = (out / "IBEV_stage1.npz").stat().st_mtime_ns
    stage2_mtime = (out / "IBEV_stage2.npz").stat().st_mtime_ns
    _run(run_cmd + ["--resume"], here)
    if (out / "IBEV_stage1.npz").stat().st_mtime_ns != stage1_mtime:
        raise AssertionError("--resumeがStage1を再生成しました")
    if (out / "IBEV_stage2.npz").stat().st_mtime_ns != stage2_mtime:
        raise AssertionError("--resumeがStage2を再生成しました")
    state = json.loads((out / "IBEV_job_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETED":
        raise AssertionError(f"resume state不正: {state}")

    cancel_test = _validate_cancel_before_start(here, synth, work)
    parameter_resume_test = _validate_parameter_aware_resume(here, synth, work)
    frenet_test = _validate_frenet_branch_ambiguity(here)
    bridge_test = _validate_long_gap_bridge_guard(here)
    smoothing_test = _validate_polyline_smoothing(here)

    import lanegen_core as core
    report = {
        "status": "PASS",
        "production_version": core.PRODUCTION_VERSION,
        "pcd_modes": pcd_modes,
        "streaming_topk_test": streaming_topk_test,
        "compressed_reader_reuse_test": compressed_reuse_test,
        "streaming_batch_equivalence": streaming_batch_test,
        "parameter_aware_resume_test": parameter_resume_test,
        "stale_lock_test": stale_lock_test,
        "resume_test": {"status": state.get("status"),
                        "stage1_reused": True, "stage2_reused": True},
        "cancel_test": cancel_test,
        "frenet_ambiguity_test": frenet_test,
        "long_gap_bridge_test": bridge_test,
        "polyline_smoothing_test": smoothing_test,
        "lane": lane,
        "curb": curb,
        "lane_detected": lane_detected,
        "curb_detected": curb_detected,
        "annotator_schema": schema,
        "deterministic_hashes": hash_after,
    }
    (work / "regression_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.keep:
        # 出力結果はレポート確認用に残し、合成PCDだけ削除して容量を抑える。
        shutil.rmtree(synth, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
