#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight validation for LaneGen production jobs."""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import lanegen_core as core


def _module_version(name: str):
    try:
        mod = importlib.import_module(name)
        return {"available": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="LaneGen Production V3 preflight check")
    p.add_argument("intensity_pcd")
    p.add_argument("mapping_pose")
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--z-pcd", default=None)
    p.add_argument("--rgb-pcd", default=None)
    p.add_argument("--hash-inputs", action="store_true")
    p.add_argument("--report", default=None)
    a = p.parse_args(argv)

    out = Path(a.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "intensity_pcd": Path(a.intensity_pcd).expanduser().resolve(),
        "mapping_pose": Path(a.mapping_pose).expanduser().resolve(),
    }
    if a.z_pcd:
        files["z_pcd"] = Path(a.z_pcd).expanduser().resolve()
    if a.rgb_pcd:
        files["rgb_pcd"] = Path(a.rgb_pcd).expanduser().resolve()

    errors = []
    file_info = {}
    for key, path in files.items():
        if not path.is_file():
            errors.append(f"missing: {key}={path}")
            continue
        item = {"path": str(path), "size_bytes": path.stat().st_size}
        if key.endswith("pcd"):
            try:
                h = core.read_pcd_header(path)
                item["pcd"] = {
                    "data": h.data, "fields": list(h.fields),
                    "sizes": list(h.sizes), "types": list(h.types),
                    "counts": list(h.counts), "points": h.points,
                }
            except Exception as exc:
                errors.append(f"invalid PCD {key}: {exc}")
        if a.hash_inputs:
            item["sha256"] = core.sha256_file(path)
        file_info[key] = item

    usage = shutil.disk_usage(out)
    total_input = sum(v.get("size_bytes", 0) for v in file_info.values())
    # Conservative warning only; output/cache size depends on corridor and trajectory.
    if usage.free < max(2 * total_input, 2 * 1024**3):
        errors.append("output disk free space may be insufficient")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "production_version": core.PRODUCTION_VERSION,
        "python": sys.version,
        "dependencies": {
            "numpy": _module_version("numpy"),
            "scipy": _module_version("scipy"),
            "matplotlib": _module_version("matplotlib"),
            "PyQt5_optional": _module_version("PyQt5"),
        },
        "files": file_info,
        "output": {
            "path": str(out), "free_bytes": usage.free,
            "total_bytes": usage.total,
        },
        "errors": errors,
    }
    report_path = Path(a.report).resolve() if a.report else out / "lanegen_preflight.json"
    core.atomic_dump_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
