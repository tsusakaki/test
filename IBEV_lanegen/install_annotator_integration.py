#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install LaneGen V5 integration files into BevLaneAnnotator safely."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Install LaneGen Production V6 GUI integration"
    )
    p.add_argument(
        "--annotator-root",
        required=True,
        help="BevLaneAnnotator project root containing ui/main_window.py",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    integration_dir = package_dir / "annotator_integration"
    root = Path(args.annotator_root).expanduser().resolve()

    mappings = [
        (integration_dir / "main_window.py", root / "ui" / "main_window.py"),
        (integration_dir / "lane_line_panel.py", root / "lane_line_panel.py"),
    ]

    missing = [str(src) for src, _ in mappings if not src.is_file()]
    if missing:
        raise FileNotFoundError("integration source missing: " + ", ".join(missing))
    if not (root / "ui").is_dir():
        raise FileNotFoundError(root / "ui")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for src, dst in mappings:
        print(f"{src} -> {dst}")
        if args.dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            backup = dst.with_name(dst.name + f".bak_lanegen_v5_{stamp}")
            shutil.copy2(dst, backup)
            print(f"  backup: {backup}")
        shutil.copy2(src, dst)

    print("LaneGen V5 integration installed." if not args.dry_run else "Dry run completed.")
    print(
        "Keep the IBEV_lanegen_production_v6 folder under the Annotator root "
        "or select it from the Generate button when prompted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
