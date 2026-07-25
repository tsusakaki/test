#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyQt5 bridge for embedding LaneGen Production V6 in BEV Lane Annotator.

This module intentionally contains no tool-specific UI. MainWindow can connect
signals to its own progress bar, log widget and Japanese/English messages.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from PyQt5.QtCore import QObject, QProcess, QTimer, pyqtSignal


class LaneGenQtRunner(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(dict)
    log = pyqtSignal(str)
    finished = pyqtSignal(int, dict)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_process_error)

        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._poll_progress)

        self._state_path: Optional[Path] = None
        self._progress_path: Optional[Path] = None
        self._cancel_path: Optional[Path] = None
        self._last_progress_text = ""

    @property
    def is_running(self) -> bool:
        return self._process.state() != QProcess.NotRunning

    @property
    def state_path(self) -> Optional[Path]:
        return self._state_path

    def start_job(
        self,
        package_dir: str | Path,
        intensity_pcd: str | Path,
        mapping_pose: str | Path,
        output_dir: str | Path,
        *,
        z_pcd: str | Path | None = None,
        rgb_pcd: str | Path | None = None,
        stem: str | None = None,
        resume: bool = True,
        extra_args: Iterable[str] = (),
        python_executable: str | Path | None = None,
    ) -> None:
        if self.is_running:
            raise RuntimeError("LaneGen job is already running")

        package_dir = Path(package_dir).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        script = package_dir / "lanegen_run.py"
        if not script.is_file():
            raise FileNotFoundError(script)

        stem = stem or Path(intensity_pcd).stem
        self._state_path = output_dir / f"{stem}_job_state.json"
        self._progress_path = output_dir / f"{stem}_progress.json"
        self._cancel_path = output_dir / f"{stem}.cancel"
        self._cancel_path.unlink(missing_ok=True)
        self._last_progress_text = ""

        args = [
            str(script), str(Path(intensity_pcd)), str(Path(mapping_pose)),
            "-o", str(output_dir), "--stem", stem,
            "--state-file", str(self._state_path),
            "--progress-json", str(self._progress_path),
            "--cancel-file", str(self._cancel_path),
            "--reset-cancel-file",
        ]
        if resume:
            args.append("--resume")
        if z_pcd is not None:
            args += ["--z-pcd", str(Path(z_pcd))]
        if rgb_pcd is not None:
            args += ["--rgb-pcd", str(Path(rgb_pcd))]
        args += [str(x) for x in extra_args]

        python_executable = str(python_executable or sys.executable)
        self._process.setWorkingDirectory(str(package_dir))
        self._process.start(python_executable, args)
        if not self._process.waitForStarted(5000):
            raise RuntimeError(self._process.errorString())
        self._timer.start()
        self.started.emit()

    def request_cancel(self) -> None:
        """Request cooperative cancellation. Does not kill during file writes."""
        if self._cancel_path is None:
            return
        self._cancel_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cancel_path.with_suffix(self._cancel_path.suffix + ".tmp")
        tmp.write_text("cancel\n", encoding="utf-8")
        os.replace(tmp, self._cancel_path)
        self.log.emit("[LaneGen] cancel requested")

    def force_kill(self) -> None:
        """Emergency-only process kill. Prefer request_cancel()."""
        if self.is_running:
            self._process.kill()

    def _read_output(self) -> None:
        raw = bytes(self._process.readAllStandardOutput())
        text = raw.decode("utf-8", "replace")
        for line in text.splitlines():
            self.log.emit(line)

    def _poll_progress(self) -> None:
        path = self._progress_path or self._state_path
        if path is None or not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8")
            if text == self._last_progress_text:
                return
            payload = json.loads(text)
            self._last_progress_text = text
            self.progress.emit(payload)
        except Exception:
            # Writer uses atomic replace; a transient parse failure is harmless.
            return

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        self._timer.stop()
        self._poll_progress()
        payload = {}
        if self._state_path and self._state_path.is_file():
            try:
                payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        self.finished.emit(int(exit_code), payload)

    def _on_process_error(self, _error) -> None:
        self.failed.emit(self._process.errorString())
