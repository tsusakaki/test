#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite-driven batch launcher for IBEV LaneGen Production V6.

The database is treated as read-only.  For each unique clip, this module:

1. Reads ``lidar_path`` (or the common typo ``ladar_path``), falling back to
   ``lidar_source`` when the path column is empty.
2. Resolves the clip's ``result`` directory.
3. Uses these fixed inputs relative to that directory::

       result/mapqr_input/IBEV.pcd
       result/mapqr_input/RGBBEV.pcd
       result/mapping/mapping_pose.txt

4. Searches recursively below ``--output`` for::

       <clip folder name>_output

5. Writes LaneGen products below::

       <matched output folder>/work_dir/bevld

Duplicate database rows that resolve to the same result directory are merged
and processed only once.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Iterable, Sequence

import lanegen_core as core


_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SAFE_SQL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _looks_windows_path(value: str) -> bool:
    return bool(_WINDOWS_DRIVE_RE.match(value)) or "\\" in value


def _pure_path(value: str) -> PurePath:
    value = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if _looks_windows_path(value):
        return PureWindowsPath(value)
    return PurePosixPath(value)


def _native_path(value: str | PurePath) -> Path:
    """Convert to a path usable on the current OS.

    Windows database paths are expected to be executed on Windows.  On Linux
    they remain display-only paths and normal (non-dry-run) validation will
    report them as inaccessible rather than silently rewriting them.
    """
    text = str(value)
    if os.name != "nt" and _looks_windows_path(text):
        return Path(text)
    return Path(text).expanduser()


def _replace_part(path: PurePath, index: int, value: str) -> PurePath:
    cls = PureWindowsPath if isinstance(path, PureWindowsPath) else PurePosixPath
    parts = list(path.parts)
    parts[index] = value
    return cls(*parts[: index + 1])


def resolve_result_root(raw_path: str) -> tuple[PurePath, str]:
    """Resolve the canonical ``result`` directory from a DB path.

    Exact ``result`` wins.  A path under ``result_bagEx`` or another
    ``result*`` directory is normalized to its sibling ``result`` because the
    requested IBEV/RGBBEV/mapping layout is rooted there.
    """
    if not raw_path or not str(raw_path).strip():
        raise ValueError("LiDAR path is empty")
    path = _pure_path(raw_path)
    parts = list(path.parts)

    for index, part in enumerate(parts):
        if part.casefold() == "result":
            cls = PureWindowsPath if isinstance(path, PureWindowsPath) else PurePosixPath
            return cls(*parts[: index + 1]), "exact_result"

    for index, part in enumerate(parts):
        if part.casefold().startswith("result"):
            return _replace_part(path, index, "result"), f"normalized_from_{part}"

    # Support a DB field that already points at the clip directory.
    if path.name.casefold().startswith("clip_"):
        return path / "result", "appended_result"

    raise ValueError(
        "path does not contain a result directory: " + str(raw_path)
    )


def _canonical_key(path: PurePath) -> str:
    text = str(path)
    return text.casefold() if isinstance(path, PureWindowsPath) else text


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class DatabaseTask:
    result_root: str
    clip_name: str
    row_ids: list[int] = field(default_factory=list)
    selected_path_column: str = ""
    source_paths: list[str] = field(default_factory=list)
    result_resolution: list[str] = field(default_factory=list)
    row_metadata: list[dict] = field(default_factory=list)


@dataclass
class TaskPlan:
    result_root: str
    clip_name: str
    row_ids: list[int]
    selected_path_column: str
    source_paths: list[str]
    intensity_pcd: str
    rgb_pcd: str
    mapping_pose: str
    output_container: str | None
    bevld_dir: str | None
    status: str
    message: str = ""
    return_code: int | None = None
    elapsed_sec: float | None = None


def _quote_identifier(name: str) -> str:
    if not _SAFE_SQL_NAME_RE.fullmatch(name):
        raise ValueError(f"unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def load_database_tasks(
    database: Path,
    *,
    table: str,
    task_ids: set[int] | None,
) -> tuple[list[DatabaseTask], dict]:
    """Read and de-duplicate tasks without modifying the database."""
    uri = database.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table_q = _quote_identifier(table)
        columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_q})")
        }
        if not columns:
            raise RuntimeError(f"SQLite table not found: {table}")

        path_candidates = [
            name for name in ("lidar_path", "ladar_path", "lidar_source")
            if name in columns
        ]
        if not path_candidates:
            raise RuntimeError(
                f"{table} has no lidar_path, ladar_path, or lidar_source column"
            )

        id_column = "id" if "id" in columns else "rowid"
        sql = f"SELECT rowid AS __rowid__, * FROM {table_q}"
        params: list[int] = []
        if task_ids:
            placeholders = ",".join("?" for _ in sorted(task_ids))
            sql += f" WHERE {_quote_identifier(id_column)} IN ({placeholders})"
            params.extend(sorted(task_ids))
        sql += f" ORDER BY {_quote_identifier(id_column)}"

        grouped: dict[str, DatabaseTask] = {}
        invalid_rows: list[dict] = []
        path_usage = {name: 0 for name in path_candidates}
        total_rows = 0

        for row in connection.execute(sql, params):
            total_rows += 1
            row_dict = {key: _json_safe(row[key]) for key in row.keys()}
            row_id = int(row[id_column] if id_column in row.keys() else row["__rowid__"])

            selected_column = ""
            selected_value = ""
            for candidate in path_candidates:
                value = row[candidate]
                if value is not None and str(value).strip():
                    selected_column = candidate
                    selected_value = str(value).strip()
                    break
            if not selected_value:
                invalid_rows.append({
                    "row_id": row_id,
                    "reason": "all LiDAR path columns are empty",
                })
                continue

            try:
                result_root, resolution = resolve_result_root(selected_value)
            except Exception as exc:
                invalid_rows.append({
                    "row_id": row_id,
                    "selected_column": selected_column,
                    "path": selected_value,
                    "reason": str(exc),
                })
                continue

            path_usage[selected_column] += 1
            key = _canonical_key(result_root)
            clip_name = result_root.parent.name
            task = grouped.get(key)
            if task is None:
                task = DatabaseTask(
                    result_root=str(result_root),
                    clip_name=clip_name,
                    selected_path_column=selected_column,
                )
                grouped[key] = task
            # Prefer a real lidar_path/ladar_path label when any duplicate has it.
            if task.selected_path_column == "lidar_source" and selected_column != "lidar_source":
                task.selected_path_column = selected_column
            task.row_ids.append(row_id)
            if selected_value not in task.source_paths:
                task.source_paths.append(selected_value)
            if resolution not in task.result_resolution:
                task.result_resolution.append(resolution)
            # Keep only useful metadata to avoid copying a huge DB row blindly.
            task.row_metadata.append({
                key: row_dict.get(key)
                for key in (
                    "id", "bag_datetime_str", "data_type", "postfix",
                    "lidar_node", "priority", "prelabel", "lidar_path",
                    "ladar_path", "lidar_source",
                )
                if key in row_dict
            })

        tasks = sorted(
            grouped.values(),
            key=lambda item: (item.clip_name.casefold(), item.result_root.casefold()),
        )
        diagnostics = {
            "table": table,
            "columns": sorted(columns),
            "path_columns_available": path_candidates,
            "path_column_usage": path_usage,
            "rows_selected": total_rows,
            "unique_result_roots": len(tasks),
            "invalid_rows": invalid_rows,
        }
        return tasks, diagnostics
    finally:
        connection.close()


def index_output_containers(output_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not output_root.is_dir():
        return index
    for dirpath, dirnames, _filenames in os.walk(output_root):
        base = Path(dirpath)
        # Copy because pruning may alter the list while os.walk uses it.
        for dirname in list(dirnames):
            if dirname.casefold().endswith("_output"):
                path = base / dirname
                index.setdefault(dirname.casefold(), []).append(path)
    for matches in index.values():
        matches.sort(key=lambda p: (len(p.parts), str(p).casefold()))
    return index


def build_plan(
    task: DatabaseTask,
    *,
    output_root: Path,
    output_index: dict[str, list[Path]],
    create_output_container: bool,
    dry_run: bool,
) -> TaskPlan:
    result_pure = _pure_path(task.result_root)
    intensity_pure = result_pure / "mapqr_input" / "IBEV.pcd"
    rgb_pure = result_pure / "mapqr_input" / "RGBBEV.pcd"
    mapping_pure = result_pure / "mapping" / "mapping_pose.txt"

    expected_name = f"{task.clip_name}_output"
    matches = output_index.get(expected_name.casefold(), [])
    if len(matches) > 1:
        return TaskPlan(
            result_root=task.result_root,
            clip_name=task.clip_name,
            row_ids=task.row_ids,
            selected_path_column=task.selected_path_column,
            source_paths=task.source_paths,
            intensity_pcd=str(intensity_pure),
            rgb_pcd=str(rgb_pure),
            mapping_pose=str(mapping_pure),
            output_container=None,
            bevld_dir=None,
            status="AMBIGUOUS_OUTPUT",
            message=(
                f"multiple folders named {expected_name}: "
                + "; ".join(str(path) for path in matches)
            ),
        )

    if matches:
        container = matches[0]
    elif create_output_container:
        container = output_root / expected_name
        if not dry_run:
            container.mkdir(parents=True, exist_ok=True)
        output_index.setdefault(expected_name.casefold(), []).append(container)
    else:
        return TaskPlan(
            result_root=task.result_root,
            clip_name=task.clip_name,
            row_ids=task.row_ids,
            selected_path_column=task.selected_path_column,
            source_paths=task.source_paths,
            intensity_pcd=str(intensity_pure),
            rgb_pcd=str(rgb_pure),
            mapping_pose=str(mapping_pure),
            output_container=None,
            bevld_dir=None,
            status="OUTPUT_NOT_FOUND",
            message=(
                f"{expected_name} was not found recursively below {output_root}"
            ),
        )

    bevld = container / "work_dir" / "bevld"
    return TaskPlan(
        result_root=task.result_root,
        clip_name=task.clip_name,
        row_ids=task.row_ids,
        selected_path_column=task.selected_path_column,
        source_paths=task.source_paths,
        intensity_pcd=str(intensity_pure),
        rgb_pcd=str(rgb_pure),
        mapping_pose=str(mapping_pure),
        output_container=str(container),
        bevld_dir=str(bevld),
        status="PLANNED",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core.atomic_dump_json(path, payload)


def _source_metadata(task: DatabaseTask, plan: TaskPlan, database: Path) -> dict:
    return {
        "production_version": core.PRODUCTION_VERSION,
        "created_at_utc": _utc_now(),
        "source_database": str(database.resolve()),
        "source_database_signature": core.file_signature(database),
        "row_ids": task.row_ids,
        "selected_path_column": task.selected_path_column,
        "source_paths": task.source_paths,
        "result_resolution": task.result_resolution,
        "result_root": plan.result_root,
        "clip_name": plan.clip_name,
        "inputs": {
            "intensity_pcd": plan.intensity_pcd,
            "rgb_pcd": plan.rgb_pcd,
            "mapping_pose": plan.mapping_pose,
        },
        "output": {
            "container": plan.output_container,
            "bevld_dir": plan.bevld_dir,
        },
        "database_rows": task.row_metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SQLiteのLiDARパスからresult/mapqr_inputとresult/mappingを解決し、"
            "<clip>_output/work_dir/bevldへLaneGen出力を保存"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="入力SQLiteファイル")
    parser.add_argument("--output", required=True, help="出力検索ルート")
    parser.add_argument("--table", default="match_result_tasks")
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        default=None,
        help="処理するDB id。複数指定可。省略時は全行",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--stem", default="IBEV")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--create-output-container",
        action="store_true",
        help=(
            "<clip>_outputが見つからない場合、--output直下へ作成。"
            "未指定時は見つからないタスクをエラー扱い"
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="既存Stageキャッシュを再利用しない",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="バッチ結果JSON。省略時は--output/bevld_lanegen_batch_latest.json",
    )
    return parser


def _reject_conflicting_forwarded_args(extra: Sequence[str]) -> None:
    forbidden = {
        "--input", "--output", "--output-dir", "-o", "--z-pcd",
        "--rgb-pcd", "--stem", "--state-file", "--progress-json",
        "--cancel-file", "--lock-file",
    }
    for token in extra:
        key = token.split("=", 1)[0]
        if key in forbidden:
            raise ValueError(
                f"{key} is managed by SQLite batch mode and cannot be forwarded"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    try:
        _reject_conflicting_forwarded_args(extra)
    except ValueError as exc:
        parser.error(str(exc))

    database = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not database.is_file():
        parser.error(f"SQLite file not found: {database}")
    if not output_root.exists() and not args.create_output_container:
        parser.error(
            f"output search root not found: {output_root}; "
            "create it first or use --create-output-container"
        )
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    task_ids = set(args.task_id or []) or None
    tasks, diagnostics = load_database_tasks(
        database,
        table=args.table,
        task_ids=task_ids,
    )
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]

    output_index = index_output_containers(output_root)
    plans = [
        build_plan(
            task,
            output_root=output_root,
            output_index=output_index,
            create_output_container=args.create_output_container,
            dry_run=args.dry_run,
        )
        for task in tasks
    ]

    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_root / "bevld_lanegen_batch_latest.json"
    )
    summary = {
        "production_version": core.PRODUCTION_VERSION,
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
        "started_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "input_database": str(database),
        "output_root": str(output_root),
        "database_diagnostics": diagnostics,
        "forwarded_lanegen_arguments": list(extra),
        "tasks": [asdict(plan) for plan in plans],
    }
    if not args.dry_run:
        _write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0 if all(plan.status == "PLANNED" for plan in plans) else 1

    task_by_root = {_canonical_key(_pure_path(task.result_root)): task for task in tasks}
    script = Path(__file__).resolve().with_name("lanegen_run.py")
    any_failure = False
    any_cancelled = False

    for index, plan in enumerate(plans, start=1):
        task = task_by_root[_canonical_key(_pure_path(plan.result_root))]
        if plan.status != "PLANNED":
            any_failure = True
            if args.fail_fast:
                break
            continue

        input_paths = {
            "IBEV.pcd": _native_path(plan.intensity_pcd),
            "RGBBEV.pcd": _native_path(plan.rgb_pcd),
            "mapping_pose.txt": _native_path(plan.mapping_pose),
        }
        missing = [name for name, path in input_paths.items() if not path.is_file()]
        if missing:
            plan.status = "MISSING_INPUT"
            plan.message = ", ".join(
                f"{name}: {input_paths[name]}" for name in missing
            )
            any_failure = True
            summary["tasks"] = [asdict(item) for item in plans]
            _write_json(summary_path, summary)
            print(f"[skip] {plan.clip_name}: {plan.message}", file=sys.stderr)
            if args.fail_fast:
                break
            continue

        bevld = Path(plan.bevld_dir)
        bevld.mkdir(parents=True, exist_ok=True)
        source_meta_path = bevld / f"{args.stem}_lanegen_source.json"
        _write_json(source_meta_path, _source_metadata(task, plan, database))

        cmd = [
            sys.executable,
            str(script),
            str(input_paths["IBEV.pcd"]),
            str(input_paths["mapping_pose.txt"]),
            "-o", str(bevld),
            "--z-pcd", str(input_paths["RGBBEV.pcd"]),
            "--rgb-pcd", str(input_paths["RGBBEV.pcd"]),
            "--stem", args.stem,
        ]
        if not args.no_resume:
            cmd.append("--resume")
        cmd.extend(extra)

        print(
            f"[{index}/{len(plans)}] {plan.clip_name} -> {bevld}",
            flush=True,
        )
        started = time.time()
        child_env = os.environ.copy()
        child_env["LANEGEN_BATCH_MODE"] = "1"
        completed = subprocess.run(cmd, check=False, env=child_env)
        plan.return_code = int(completed.returncode)
        plan.elapsed_sec = round(time.time() - started, 2)

        state_path = bevld / f"{args.stem}_job_state.json"
        child_state = {}
        if state_path.is_file():
            try:
                child_state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                child_state = {}

        if completed.returncode == 0:
            plan.status = "COMPLETED"
            plan.message = "LaneGen completed"
            print(
                f"[ok] {plan.clip_name}: 完了 ({plan.elapsed_sec:.2f}s)",
                flush=True,
            )
        elif completed.returncode == 2:
            plan.status = "CANCELLED"
            plan.message = child_state.get("message") or "LaneGen cancelled"
            any_cancelled = True
            print(f"[cancelled] {plan.clip_name}: {plan.message}", file=sys.stderr)
        elif completed.returncode == 3:
            plan.status = "SKIPPED"
            plan.message = child_state.get("message") or "処理対象外のためスキップ"
            print(
                f"[skip] {plan.clip_name}: {plan.message}。次のタスクへ進みます。",
                file=sys.stderr,
            )
        else:
            plan.status = "FAILED"
            plan.message = child_state.get("message") or f"LaneGen exit code {completed.returncode}"
            any_failure = True
            print(
                f"[failed] {plan.clip_name}: {plan.message}。次のタスクへ進みます。",
                file=sys.stderr,
            )

        summary["tasks"] = [asdict(item) for item in plans]
        _write_json(summary_path, summary)
        if (any_failure or any_cancelled) and args.fail_fast:
            break

    skipped_count = sum(plan.status == "SKIPPED" for plan in plans)
    if any_cancelled:
        final_status = "CANCELLED"
        return_code = 2
    elif any_failure or any(
        plan.status not in {"COMPLETED", "SKIPPED"} for plan in plans
    ):
        final_status = "COMPLETED_WITH_ERRORS"
        return_code = 1
    elif skipped_count:
        final_status = "COMPLETED_WITH_SKIPS"
        return_code = 0
    else:
        final_status = "COMPLETED"
        return_code = 0

    summary.update({
        "status": final_status,
        "completed_at_utc": _utc_now(),
        "tasks": [asdict(plan) for plan in plans],
        "counts": {
            status: sum(plan.status == status for plan in plans)
            for status in sorted({plan.status for plan in plans})
        },
    })
    _write_json(summary_path, summary)
    counts = summary["counts"]
    print(
        "[summary] "
        f"completed={counts.get('COMPLETED', 0)}, "
        f"skipped={counts.get('SKIPPED', 0)}, "
        f"failed={counts.get('FAILED', 0)}, "
        f"cancelled={counts.get('CANCELLED', 0)}",
        flush=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
