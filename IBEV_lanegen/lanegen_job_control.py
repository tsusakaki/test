#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect or cancel a Production V3 job from command line."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("status")
    s.add_argument("state_json")
    c = sub.add_parser("cancel")
    c.add_argument("cancel_file")
    cc = sub.add_parser("clear-cancel")
    cc.add_argument("cancel_file")
    a = p.parse_args(argv)

    if a.command == "status":
        path = Path(a.state_json)
        if not path.is_file():
            print(json.dumps({"status": "NOT_FOUND", "path": str(path)}, ensure_ascii=False))
            return 1
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")),
                         ensure_ascii=False, indent=2))
        return 0

    path = Path(a.cancel_file)
    if a.command == "cancel":
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("cancel\n", encoding="utf-8")
        os.replace(tmp, path)
        print(f"cancel requested: {path}")
    else:
        path.unlink(missing_ok=True)
        print(f"cancel cleared: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
